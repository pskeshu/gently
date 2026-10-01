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
from datetime import datetime

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


async def _apply_plan_settings(agent, structure: dict, embryo_ids: list[str]) -> None:
    """The SPIM channel's settings, applied to exactly the embryos the run will image.

    Slices and exposure are per-embryo settings the orchestrator reads off the
    EmbryoState; the laser preset is set on the controller once for the run.
    Mirrors what the Operate start route does, so a plan runs the same from
    the pane, from the library and from the agent.
    """
    experiment = getattr(agent, "experiment", None)
    embryos = getattr(experiment, "embryos", None) or {}
    changes: dict = {}
    if structure.get("num_slices") is not None:
        changes["num_slices"] = int(structure["num_slices"])
    if structure.get("exposure_ms") is not None:
        changes["exposure_ms"] = float(structure["exposure_ms"])
    # The plan's per-line powers, saved as {"488": 4.0}. Out-of-range values
    # are refused by the device layer at the first volume.
    powers = structure.get("laser_powers")
    if isinstance(powers, dict):
        for wl, pct in powers.items():
            if pct is not None:
                changes[f"laser_power_{int(wl)}_pct"] = float(pct)
    set_params = getattr(experiment, "set_params", None)
    for eid in embryo_ids:
        if eid not in embryos or not changes:
            continue
        if set_params is not None:
            # Through the one door: recorded and announced (PANELS.md rule 8).
            set_params(eid, changes, by="tactic", reason="run from a saved tactic")
        else:
            for name, value in changes.items():
                setattr(embryos[eid], name, value)
    preset = structure.get("laser_config")
    client = getattr(agent, "client", None)
    if preset and client is not None and hasattr(client, "set_laser_config"):
        # Refused rather than shrugged off: a run that starts on the wrong
        # lasers images every timepoint with them.
        await client.set_laser_config(str(preset))


def _keep_plan(
    agent, structure: dict, embryo_ids: list[str], message, tactic: dict | None = None
) -> None:
    """The plan a run was started with, kept in the session as acquisition.yaml
    so the pane reads it back on resume. Best-effort; the run is already going.
    With the tactic: which saved tactic it was, so the pane comes back on it."""
    if isinstance(message, str) and message.startswith("Timelapse already running"):
        return
    store = getattr(agent, "store", None)
    sid = getattr(agent, "session_id", None)
    if store is None or not sid or not hasattr(store, "save_acquisition_plan"):
        return
    try:
        plan = dict(structure)
        plan["embryo_ids"] = list(embryo_ids)
        t = tactic or {}
        plan["tactic_id"] = t.get("id")
        plan["name"] = t.get("name")
        plan["library_id"] = t.get("library_id")
        plan["mode"] = "library" if t.get("library_id") else "tactic"
        plan.setdefault("scope", "selected")
        store.save_acquisition_plan(sid, plan)
    except Exception:
        logger.warning("could not keep the acquisition plan", exc_info=True)


def _num(v, default=None):
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def timelapse_tactics_going(agent) -> list[str]:
    """The ids of the session's timelapse tactics the plan calls active or paused."""
    cs = getattr(agent, "context_store", None)
    sid = getattr(agent, "session_id", None)
    if cs is None or not sid:
        return []
    try:
        plan = cs.get_operation_plan(sid) or {}
    except Exception:
        return []
    return [
        t["id"]
        for t in (plan.get("tactics") or [])
        if t.get("kind") == "standing_timelapse"
        and t.get("state") in ("active", "paused")
        and t.get("id")
    ]


def when(dt) -> str:
    """A moment, as a card says it: '30 Sep 16:45'. Local time, like the logs."""
    try:
        return dt.strftime("%d %b %H:%M").lstrip("0")
    except Exception:
        return str(dt)


def span(seconds: float) -> str:
    """A duration, as a card says it: '20 s', '12 min', '8 h 15 min'."""
    s = max(0, int(round(float(seconds))))
    if s < 60:
        return f"{s} s"
    m = s // 60
    if m < 60:
        return f"{m} min"
    h, m = divmod(m, 60)
    return f"{h} h {m} min" if m else f"{h} h"


def run_facts(orchestrator, reason: str | None = None) -> dict[str, str]:
    """What a run was, for the card of the tactic it ran as.

    Two Starts in one session leave two tactics that say only "Adaptive
    timelapse · done", so a 20-second false start and the eight-hour run
    that followed it read as a duplicate. These are the facts that tell them
    apart: when it started, how it ended and after how long, what it took.
    Only what the orchestrator knows; a fake with none of it binds nothing.
    """
    facts: dict[str, str] = {}
    started = getattr(orchestrator, "_started_at", None)
    if started is not None:
        facts["started"] = when(started)
    ended = getattr(orchestrator, "_ended", None)
    if ended in ("stopped", "completed", "failed"):
        how = ended
        if ended == "stopped" and reason:
            how = f"stopped by {reason}"
        if ended == "failed" and getattr(orchestrator, "_error_message", None):
            how = f"failed: {orchestrator._error_message}"
        if started is not None:
            how += f" after {span((datetime.now() - started).total_seconds())}"
        facts["ended"] = how
    if getattr(orchestrator, "_volumes", True) is False:
        frames = getattr(orchestrator, "_dic_frames", None)
        if frames is not None:
            facts["acquired"] = f"{int(frames)} brightfield frame{'' if frames == 1 else 's'}"
    else:
        n = getattr(orchestrator, "_total_timepoints", None)
        if n is not None:
            facts["acquired"] = f"{int(n)} volume{'' if n == 1 else 's'}"
    return facts


def close_timelapse_tactics(agent, orchestrator=None, reason: str | None = None) -> list[str]:
    """The run has ended: its tactics are done. Returns the ids closed.

    Called for every way a run ends, whoever ended it: the Stop button, the
    assistant's tool, the last embryo reaching its ending, an error. The
    Stop button used to be the only one that said so, and only for a run
    started from the pane.

    The tactic is told how (``run_facts``), so its card can say "stopped by
    operator after 20 s · 4 volumes" and not only "done".
    """
    cs = getattr(agent, "context_store", None)
    sid = getattr(agent, "session_id", None)
    if cs is None or not sid:
        return []
    ids = timelapse_tactics_going(agent)
    for tid in list(getattr(orchestrator, "_operate_tactic_ids", None) or []):
        if tid and tid not in ids:
            ids.append(tid)
    facts = run_facts(orchestrator, reason) if orchestrator is not None else {}
    closed = []
    for tid in ids:
        try:
            if cs.transition_tactic(sid, tid, "done", **facts):
                closed.append(tid)
        except Exception:
            logger.debug("could not close tactic %s", tid, exc_info=True)
    if orchestrator is not None:
        try:
            orchestrator._operate_tactic_ids = []
        except Exception:
            pass
    return closed


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
            # The whole plan, not just its cadence. A saved plan that lost its
            # channels and endings on the way to the orchestrator would run as
            # something other than what its sentence says.
            # A saved brightfield plan is a brightfield run: no volume settings
            # are written onto embryos, no preset is set, and the run is told
            # it takes no volumes.
            volumes = structure.get("volumes") is not False
            if volumes:
                await _apply_plan_settings(agent, structure, embryo_ids)
            start_kwargs: dict = {
                "embryo_ids": embryo_ids,
                "stop_condition": str(structure.get("stop_condition", "manual")),
                "base_interval_seconds": interval,
                "condition_value": structure.get("condition_value"),
            }
            dic = structure.get("dic")
            if isinstance(dic, dict) and dic.get("enabled"):
                start_kwargs["dic"] = dic
            overrides = structure.get("stop_conditions")
            if volumes and isinstance(overrides, dict) and overrides:
                start_kwargs["stop_conditions"] = overrides
            if not volumes:
                start_kwargs["volumes"] = False
            # Carried by the run, not only set once: every volume routes its
            # own lines as it starts.
            preset = structure.get("laser_config")
            if volumes and preset:
                start_kwargs["laser_config"] = str(preset)
            message = await orchestrator.start(**start_kwargs)
            _keep_plan(agent, structure, embryo_ids, message, tactic)
            # The run knows which tactic it is, so that pausing and resuming
            # it can say so on the tactic. A run started from the pane has
            # always been linked this way; one started from a saved tactic
            # was not, and its tactic stayed "active" after the run stopped.
            if tactic_id:
                try:
                    orchestrator._operate_tactic_ids = [tactic_id]
                    # When it started, on the card, from the moment it did.
                    cs = getattr(agent, "context_store", None)
                    sid = getattr(agent, "session_id", None)
                    if cs is not None and sid:
                        cs.transition_tactic(sid, tactic_id, None, started=when(datetime.now()))
                except Exception:
                    logger.debug("could not link the run to tactic %s", tactic_id, exc_info=True)
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
