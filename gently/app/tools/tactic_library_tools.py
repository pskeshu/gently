"""
Tactic Library tools — lets the agent save, list, and apply reusable tactics.

save_tactic  — persist a tactic dict from the current Operation Plan as a
               reusable template in FileContextStore's tactic_library.
list_tactics — return a readable summary of all saved templates.
apply_tactic — instantiate a template into a fresh planned tactic and append
               it to the current session's Operation Plan.

Store + session resolution mirrors declare_operation_plan in operation_plan_tools.py.
"""

import logging
from datetime import datetime, timezone

from gently.harness.tools.helpers import require_agent
from gently.harness.tools.registry import ToolCategory, ToolExample, tool

logger = logging.getLogger(__name__)


@tool(
    name="save_tactic",
    description=(
        "Save a tactic from the current Operation Plan into the Tactic Library "
        "as a reusable template. Pass the tactic dict (e.g. lifted directly from "
        "your Operation Plan) and a human-readable name. Runtime state (live, state, "
        "original id) is stripped; the template receives a new id. "
        "Returns a confirmation with the new template id."
    ),
    category=ToolCategory.EXPERIMENT,
    examples=[
        ToolExample(
            user_query="Save my baseline timelapse tactic to the library",
            tool_input={
                "name": "Baseline timelapse",
                "tactic": {
                    "id": "t1",
                    "name": "Baseline timelapse",
                    "kind": "standing_timelapse",
                    "state": "active",
                    "scope": {"mode": "global"},
                    "rationale": "Continuous pre-ramp imaging",
                    "structure": {"cadence_s": 120, "per_embryo": []},
                    "live_bind": ["cadence"],
                    "relations": {},
                },
                "description": "Standard 2-minute cadence baseline timelapse",
            },
        )
    ],
)
async def save_tactic(
    name: str,
    tactic: dict,
    description: str = "",
    context: dict | None = None,
) -> str:
    """Save a tactic as a reusable template in the Tactic Library.

    Parameters
    ----------
    name : str
        Human-readable name for the template.
    tactic : dict
        The tactic dict to save (e.g. from the current Operation Plan).
    description : str
        Optional description; if provided, overrides tactic's rationale field.
    context : dict
        Injected by the tool runtime — do not pass manually.
    """
    agent, err = require_agent(context)
    if err:
        return err

    cs = getattr(agent, "context_store", None)
    if cs is None:
        return "Error: Context store not available — cannot save tactic"

    # Merge description into tactic so save_tactic picks it up via tactic.get("description")
    tactic_to_save = dict(tactic)
    if description:
        tactic_to_save["description"] = description

    tid = cs.save_tactic(tactic_to_save, name=name)
    return f"Tactic '{name}' saved to library with id {tid}"


@tool(
    name="list_tactics",
    description=(
        "List all tactics saved in the Tactic Library. "
        "Returns a summary of each template: id, name, kind."
    ),
    category=ToolCategory.EXPERIMENT,
    examples=[
        ToolExample(
            user_query="Show me the tactics in the library",
            tool_input={},
        )
    ],
)
async def list_tactics(
    context: dict | None = None,
) -> str:
    """List all saved tactic templates.

    Parameters
    ----------
    context : dict
        Injected by the tool runtime — do not pass manually.
    """
    agent, err = require_agent(context)
    if err:
        return err

    cs = getattr(agent, "context_store", None)
    if cs is None:
        return "Error: Context store not available — cannot list tactics"

    tactics = cs.list_tactics()
    if not tactics:
        return "Tactic Library is empty — no saved templates yet."

    lines = [f"Tactic Library ({len(tactics)} template{'s' if len(tactics) != 1 else ''}):"]
    for t in tactics:
        lines.append(
            f"  [{t.get('id', '?')}] {t.get('name', '(unnamed)')} — kind: {t.get('kind', '?')}"
        )
    return "\n".join(lines)


@tool(
    name="apply_tactic",
    description=(
        "Instantiate a saved tactic template and append it to the current session's "
        "Operation Plan. The template is looked up by id or name, a fresh planned tactic "
        "is created (new id, state='planned', no runtime state), and it is queued in the "
        "plan. If no plan exists for this session, a minimal one is created. "
        "Returns an error if the tactic is not found."
    ),
    category=ToolCategory.EXPERIMENT,
    examples=[
        ToolExample(
            user_query="Apply the baseline timelapse tactic to the current session",
            tool_input={"id_or_name": "Baseline timelapse"},
        )
    ],
)
async def apply_tactic(
    id_or_name: str,
    context: dict | None = None,
) -> str:
    """Instantiate a tactic template and append it to the current Operation Plan.

    Parameters
    ----------
    id_or_name : str
        The template id or name to look up.
    context : dict
        Injected by the tool runtime — do not pass manually.
    """
    agent, err = require_agent(context)
    if err:
        return err

    cs = getattr(agent, "context_store", None)
    if cs is None:
        return "Error: Context store not available — cannot apply tactic"

    session_id = getattr(agent, "session_id", None)
    if not session_id:
        return "Error: No active session — cannot apply tactic to Operation Plan"

    fresh = cs.apply_tactic(id_or_name)
    if fresh is None:
        return f"Error: No tactic found with id or name '{id_or_name}'"

    # Fetch or create the plan for this session
    plan = cs.get_operation_plan(session_id)
    if plan is None:
        plan = {
            "session_id": session_id,
            "title": "",
            "goal": "",
            "tactics": [],
        }

    plan.setdefault("tactics", [])
    plan["tactics"].append(fresh)
    plan["updated_at"] = datetime.now(timezone.utc).isoformat()
    plan["updated_reason"] = f"tactic '{fresh.get('name', id_or_name)}' applied from library"

    cs.set_operation_plan(session_id, plan)

    tactic_name = fresh.get("name", id_or_name)
    return (
        f"Tactic '{tactic_name}' (id {fresh['id']}) instantiated from library and "
        f"queued as 'planned' in the Operation Plan for session {session_id}"
    )
