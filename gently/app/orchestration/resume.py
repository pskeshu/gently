"""What a resumed session gets back of its acquisition.

"when i resume a session, i do not have all the acquisition configuration
restoration."

Two things were lost. The orchestrator checkpoints its runtime state to
``timelapse.yaml`` every round — cadence, the DIC channel, every embryo's
stop condition and timepoint count — and nothing ever read it back: the
loader existed with no caller. A resumed run would have started its numbering
from whatever the conversation snapshot last said, over the volumes on disk.
And the plan the operator composed on the Acquisition pane was never kept
anywhere the pane could read on its next visit.

Now a run's plan is written beside the checkpoint as ``acquisition.yaml``
when it starts, the checkpoint is applied on resume, and a session that
predates the plan file gets one derived from the checkpoint and the last
volume's sidecar, so the sessions already on the rig read back too.
"""

from __future__ import annotations

import logging
from collections import Counter
from typing import Any

logger = logging.getLogger(__name__)


def restore_acquisition_state(agent: Any) -> dict[str, Any]:
    """Point the orchestrator at the agent's session and rebuild its state.

    Called after the session manager has loaded the session's embryos. A run
    that is live is left alone: it belongs to whichever session started it.
    Best-effort — a resume never fails because its checkpoint would not load.
    """
    orch = getattr(agent, "timelapse_orchestrator", None)
    sid = getattr(agent, "session_id", None)
    if orch is None or not sid:
        return {"restored": False, "reason": "no orchestrator or session"}
    status = getattr(orch, "_status", None)
    name = getattr(status, "value", status)
    if name in ("running", "paused"):
        return {"restored": False, "reason": f"a run is {name}"}
    # The orchestrator learns its session once, at construction. Switching
    # sessions in-app left it filing traces and checkpoints under the old one.
    try:
        orch._session_id = sid
        orch._trace_dir = None
    except Exception:
        logger.debug("could not repoint orchestrator session", exc_info=True)
    try:
        message = orch.load_state()
    except Exception as exc:
        logger.warning("timelapse state restore failed for %s: %s", sid, exc)
        return {"restored": False, "reason": str(exc)}
    restored = isinstance(message, str) and message.startswith("Restored")
    (logger.info if restored else logger.debug)("Session %s: %s", sid, message)
    return {"restored": restored, "message": message}


def plan_from_session(store: Any, session_id: str) -> tuple[dict[str, Any] | None, str | None]:
    """The session's acquisition plan, and where it came from.

    ``acquisition.yaml`` when the run was started after it existed; else a
    plan read off the checkpoint and the latest volume sidecar; else nothing.
    The shape is the plan `structure` the pane, the templates and the seeded
    tactic share (cadence_s, num_slices, exposure_ms, laser_config, dic,
    stop_condition, stop_conditions, monitoring_mode).
    """
    if store is None or not session_id:
        return None, None
    try:
        saved = store.get_acquisition_plan(session_id)
    except Exception:
        saved = None
    if isinstance(saved, dict) and saved:
        return saved, "saved"
    derived = _plan_from_checkpoint(store, session_id)
    return (derived, "run") if derived else (None, None)


def _plan_from_checkpoint(store: Any, session_id: str) -> dict[str, Any] | None:
    try:
        import yaml

        sd = store._session_dir(session_id)
        if sd is None:
            return None
        path = sd / "timelapse.yaml"
        if not path.exists():
            return None
        with open(path, encoding="utf-8") as f:
            doc = yaml.safe_load(f) or {}
    except Exception:
        logger.debug("checkpoint unreadable for %s", session_id, exc_info=True)
        return None
    if not isinstance(doc, dict):
        return None
    embryos = doc.get("embryos") or {}
    specs = {}
    for eid, ed in embryos.items():
        sc = (ed or {}).get("stop_condition") or {}
        spec = sc.get("spec") if isinstance(sc, dict) else None
        specs[str(eid)] = str(spec) if spec else "manual"
    # The run's default is the ending most embryos share; the rest are overrides.
    default = Counter(specs.values()).most_common(1)[0][0] if specs else "manual"
    overrides = {eid: spec for eid, spec in specs.items() if spec != default} or None
    try:
        interval = float(doc.get("base_interval_seconds") or 120.0)
    except (TypeError, ValueError):
        interval = 120.0
    params = None
    try:
        params = store.get_acquisition_params(session_id)
    except Exception:
        params = None
    params = params if isinstance(params, dict) else {}
    dic = doc.get("dic") if isinstance(doc.get("dic"), dict) else None
    modes = doc.get("active_monitoring_modes") or []
    return {
        "cadence_s": interval,
        "interval": interval,
        "stop_condition": default,
        "condition_value": None,
        "monitoring_mode": str(modes[0]).lower() if modes else "idle",
        "num_slices": params.get("num_slices"),
        "exposure_ms": params.get("exposure_ms"),
        "laser_config": None,
        "dic": dic if dic and dic.get("enabled") else None,
        # Absent in a checkpoint from before brightfield-only runs: a volume run.
        "volumes": doc.get("volumes") is not False,
        "stop_conditions": overrides,
        "embryo_ids": list(embryos.keys()),
    }
