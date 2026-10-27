"""Reading tool: the agent writes the between-nights reading of a project.

The same function the project page's "Read again" calls, so what the agent
writes from chat and what the operator asks for are one thing.
"""

import json

from gently.harness.tools.helpers import require_agent
from gently.harness.tools.registry import ToolCategory, ToolExample, tool
from gently.tools.reading import read_project


@tool(
    name="read_project",
    description=(
        "Write the project's reading: what its nights say about its question, what "
        "deviates, what is open, what to run next. Prose over the project's nights, the "
        "pooled measurements and the stream of lines people wrote at the rig. Filed under "
        "the project and shown at the top of its page. Takes a project id or shorthand."
    ),
    category=ToolCategory.ANALYSIS,
    examples=[
        ToolExample(
            user_query="What do the t2h nights tell us so far?",
            tool_input={"project": "t2h"},
        )
    ],
)
async def read_project_tool(project: str, context: dict | None = None) -> str:
    agent, err = require_agent(context)
    if err:
        return err
    cs = getattr(agent, "context_store", None)
    store = getattr(agent, "store", None)
    if cs is None or store is None:
        return "Error: store or context store unavailable"
    c = cs.resolve_campaign(project)
    if c is None:
        return f"Error: project {project!r} not found"
    try:
        out = read_project(store, cs, cs.root_campaign_id(c.id) or c.id)
    except Exception as e:
        return f"Error: no reading: {e}"
    return json.dumps(
        {"project": c.id, "written_at": out["meta"]["written_at"], "reading": out["text"]}
    )
