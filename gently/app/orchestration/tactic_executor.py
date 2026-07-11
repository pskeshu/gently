"""Tactic Executor — turns a declarative tactic into orchestrator actions.

A *tactic* (a dict in the session Operation Plan) describes HOW to image a scoped
set of embryos: ``{kind, scope, structure, ...}``. This module is the single
place that makes that language *executable* — it resolves the tactic's scope to
concrete embryo ids and dispatches by ``kind`` to the TimelapseOrchestrator, then
marks the tactic active. It is the first (and only) caller of
``resolve_scope_embryos``; both the Operate "Run" surface and the agent reach
imaging through this one path, so the kind→action mapping lives here, not
duplicated across call sites.

Deterministic and side-effecting only through the orchestrator + context store —
no LLM. Real acquisition/motion remain the orchestrator's concern (RIG-DEFERRED).
"""

from __future__ import annotations

import logging

from gently.app.orchestration.role_scope import resolve_scope_embryos

logger = logging.getLogger(__name__)


def _roster(agent) -> list[dict]:
    """Build the [{embryo_id, role}] roster resolve_scope_embryos expects."""
    exp = getattr(agent, "experiment", None)
    embryos = getattr(exp, "embryos", {}) if exp is not None else {}
    roster = []
    for eid, emb in embryos.items():
        roster.append({"embryo_id": eid, "role": getattr(emb, "role", "unassigned")})
    return roster


def _num(v, default=None):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


async def execute_tactic(agent, tactic: dict) -> dict:
    """Execute one tactic against the agent's orchestrator.

    Returns ``{ok, kind, embryo_ids, message}``. Never raises for an unknown
    kind or empty scope — it reports them in the result so callers can surface a
    clear message. Marks the tactic ``active`` in the Operation Plan on success.
    """
    kind = (tactic or {}).get("kind")
    scope = (tactic or {}).get("scope")
    structure = (tactic or {}).get("structure") or {}
    tactic_id = (tactic or {}).get("id")

    orchestrator = getattr(agent, "timelapse_orchestrator", None)
    if orchestrator is None:
        return {"ok": False, "kind": kind, "embryo_ids": [], "message": "no orchestrator"}

    embryo_ids = resolve_scope_embryos(scope, _roster(agent))
    if not embryo_ids and kind not in ("oneshot", "custom", "scripted_protocol"):
        return {
            "ok": False,
            "kind": kind,
            "embryo_ids": [],
            "message": "scope resolved to no embryos",
        }

    message = ""
    try:
        if kind == "standing_timelapse":
            interval = _num(structure.get("cadence_s"), _num(structure.get("interval"), 120.0))
            message = await orchestrator.start(
                embryo_ids=embryo_ids,
                stop_condition=str(structure.get("stop_condition", "manual")),
                base_interval_seconds=interval,
                condition_value=structure.get("condition_value"),
            )
            mode = structure.get("monitoring_mode")
            if mode and mode != "idle":
                try:
                    mres = orchestrator.enable_monitoring_mode(mode, embryo_ids=embryo_ids)
                    message += " | " + str(mres)
                except Exception as exc:  # monitoring is best-effort
                    message += f" | monitoring '{mode}' failed: {exc}"

        elif kind == "reactive_monitor":
            mode = structure.get("monitoring_mode") or "expression_monitoring"
            message = orchestrator.enable_monitoring_mode(mode, embryo_ids=embryo_ids)

        elif kind == "exclusive_burst":
            frames = int(_num(structure.get("frames"), 60))
            results = []
            for eid in embryo_ids:
                results.append(
                    orchestrator.queue_burst(
                        eid,
                        frames=frames,
                        mode=str(structure.get("mode", "1hz")),
                        num_slices=int(_num(structure.get("num_slices"), 1)),
                        tactic_id=tactic_id,
                    )
                )
            message = "; ".join(results)

        elif kind in ("oneshot", "scripted_protocol", "custom"):
            # No standing orchestrator mechanism backs these here — the tactic is
            # recorded (and, for oneshot, driven by the manual per-embryo loop).
            message = f"{kind} recorded (no orchestrator mechanism)"

        else:
            return {
                "ok": False,
                "kind": kind,
                "embryo_ids": embryo_ids,
                "message": f"unknown tactic kind '{kind}'",
            }
    except Exception as exc:
        logger.exception("tactic execution failed (kind=%s)", kind)
        return {"ok": False, "kind": kind, "embryo_ids": embryo_ids, "message": str(exc)}

    # Mark the tactic active in the Operation Plan (best-effort).
    cs = getattr(agent, "context_store", None)
    sid = getattr(agent, "session_id", None)
    if cs is not None and sid and tactic_id:
        try:
            cs.transition_tactic(sid, tactic_id, "active")
        except Exception:
            logger.debug("transition_tactic failed for %s", tactic_id, exc_info=True)

    return {"ok": True, "kind": kind, "embryo_ids": embryo_ids, "message": message}


def append_tactic_to_plan(agent, tactic: dict) -> dict | None:
    """Append a (validated) tactic to the session Operation Plan and return it.

    Creates a minimal plan if none exists. Returns the stored tactic dict (with a
    generated id if absent), or None if there is no session/context store.
    """
    import uuid

    from gently.app.tools.operation_plan_tools import _validate_tactics

    cs = getattr(agent, "context_store", None)
    sid = getattr(agent, "session_id", None)
    if cs is None or not sid:
        return None
    t = dict(tactic)
    t.setdefault("id", f"op_{uuid.uuid4().hex[:8]}")
    t.setdefault("kind", "custom")
    t.setdefault("state", "planned")
    t.setdefault("name", t.get("kind", "tactic"))
    (validated,) = _validate_tactics([t])
    plan = cs.get_operation_plan(sid) or {
        "session_id": sid,
        "title": "Operate session",
        "goal": "",
        "tactics": [],
    }
    plan.setdefault("tactics", []).append(validated)
    plan["updated_reason"] = "operate tactic appended"
    cs.set_operation_plan(sid, plan)
    return validated
