"""
Operation Plan tool — lets the agent declare/update its typed Operation Plan.

The agent calls this at experiment planning time to register the tactics it
intends to run, and calls it again on each tactic transition (planned →
active → done).  The plan is stored in FileContextStore and fires
CONTEXT_UPDATED so the Operations UI refreshes live.
"""

import logging
from datetime import datetime, timezone

from gently.harness.tools.helpers import require_agent
from gently.harness.tools.registry import ToolCategory, ToolExample, tool

logger = logging.getLogger(__name__)

# Valid kind / state values per §1 of the spec
_VALID_KINDS = frozenset(
    {
        "standing_timelapse",
        "reactive_monitor",
        "scripted_protocol",
        "exclusive_burst",
        "oneshot",
        "custom",
    }
)
_VALID_STATES = frozenset({"planned", "active", "done"})


def _validate_tactics(tactics: list) -> list[dict]:
    """Validate and normalise a tactics list.

    Each tactic must have id, name, kind, state.
    Unknown kinds are clamped to 'custom'.
    Invalid states raise ValueError.
    """
    validated = []
    for i, t in enumerate(tactics):
        if not isinstance(t, dict):
            raise ValueError(f"Tactic at index {i} must be a dict, got {type(t).__name__}")
        for required_field in ("id", "name", "kind", "state"):
            if not t.get(required_field):
                raise ValueError(f"Tactic at index {i} missing required field '{required_field}'")
        tactic = dict(t)
        kind = tactic["kind"]
        if kind not in _VALID_KINDS:
            logger.warning(
                "Tactic '%s' has unknown kind '%s' — clamped to 'custom'",
                tactic.get("id"),
                kind,
            )
            tactic["kind"] = "custom"
        state = tactic["state"]
        if state not in _VALID_STATES:
            raise ValueError(
                f"Tactic '{tactic.get('id')}' has invalid state '{state}'. "
                f"Must be one of: {sorted(_VALID_STATES)}"
            )
        validated.append(tactic)
    return validated


@tool(
    name="declare_operation_plan",
    description=(
        "Declare or update the Operation Plan for the current experiment session. "
        "Call this at experiment planning time to register the tactics you intend to "
        "run, and call it again whenever a tactic's state changes (planned → active → "
        "done) or a new tactic is added. "
        "kind ∈ {standing_timelapse, reactive_monitor, scripted_protocol, "
        "exclusive_burst, oneshot, custom}. "
        "state ∈ {planned, active, done}. "
        "Unknown kinds are clamped to 'custom'."
    ),
    category=ToolCategory.EXPERIMENT,
    examples=[
        ToolExample(
            user_query="Declare my experiment plan: baseline timelapse + onset monitor",
            tool_input={
                "title": "Expression-onset survey",
                "goal": "Catch GFP onset under 25 C ramp",
                "tactics": [
                    {
                        "id": "t1",
                        "name": "Baseline timelapse",
                        "kind": "standing_timelapse",
                        "state": "active",
                        "scope": {"mode": "global"},
                        "rationale": "Continuous pre-ramp imaging",
                        "structure": {"cadence_s": 120, "per_embryo": []},
                        "live_bind": ["cadence"],
                        "relations": {},
                    }
                ],
                "updated_reason": "experiment started",
            },
        ),
        ToolExample(
            user_query="Update the plan: baseline is done, onset monitor is now active",
            tool_input={
                "title": "Expression-onset survey",
                "goal": "Catch GFP onset under 25 C ramp",
                "tactics": [
                    {
                        "id": "t1",
                        "name": "Baseline timelapse",
                        "kind": "standing_timelapse",
                        "state": "done",
                        "scope": {"mode": "global"},
                        "rationale": "Continuous pre-ramp imaging",
                        "structure": {"cadence_s": 120, "per_embryo": []},
                        "live_bind": ["cadence"],
                        "relations": {},
                    },
                    {
                        "id": "t2",
                        "name": "Onset monitor",
                        "kind": "reactive_monitor",
                        "state": "active",
                        "scope": {"mode": "global"},
                        "rationale": "Watch for GFP signal crossing threshold",
                        "structure": {
                            "watch": "gfp_signal > 0.3",
                            "reaction": "burst_capture",
                            "status": "armed",
                        },
                        "live_bind": ["signal", "current_burst"],
                        "relations": {"after": ["t1"]},
                    },
                ],
                "updated_reason": "t1 done, t2 now active",
            },
        ),
    ],
)
async def declare_operation_plan(
    title: str,
    goal: str,
    tactics: list,
    updated_reason: str = "",
    context: dict | None = None,
) -> str:
    """Declare or update the typed Operation Plan for this session.

    Parameters
    ----------
    title : str
        Short title for this experiment's plan.
    goal : str
        The agent's framing of what this run is trying to achieve.
    tactics : list
        List of tactic dicts.  Each must have id, name, kind, state.
    updated_reason : str
        Why the plan is being updated (e.g. 't2 transitioned to active').
    context : dict
        Injected by the tool runtime — do not pass manually.
    """
    agent, err = require_agent(context)
    if err:
        return err

    cs = getattr(agent, "context_store", None)
    if cs is None:
        return "Error: Context store not available — cannot persist Operation Plan"

    session_id = getattr(agent, "session_id", None)
    if not session_id:
        return "Error: No active session — cannot persist Operation Plan"

    try:
        validated_tactics = _validate_tactics(tactics)
    except ValueError as e:
        return f"Error: {e}"

    plan = {
        "session_id": session_id,
        "title": title,
        "goal": goal,
        "tactics": validated_tactics,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "updated_reason": updated_reason,
    }

    cs.set_operation_plan(session_id, plan)

    n = len(validated_tactics)
    counts = {s: sum(1 for t in validated_tactics if t["state"] == s) for s in _VALID_STATES}
    parts = [f"{s}:{c}" for s, c in counts.items() if c]
    states_summary = ", ".join(parts)
    return (
        f"Operation Plan stored for session {session_id}: "
        f"'{title}' — {n} tactic{'s' if n != 1 else ''} ({states_summary})"
    )
