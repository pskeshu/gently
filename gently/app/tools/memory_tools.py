"""
Memory recall tools — let the agent query its persistent memory on demand.

Thin wrappers around AgentMemory. Available in both run and plan modes.
"""

from typing import Dict, Optional

from gently.harness.tools.registry import tool, ToolCategory, ToolExample


def _get_memory(context: Dict):
    """Extract AgentMemory from tool context."""
    agent = context.get("agent") if context else None
    if not agent or not hasattr(agent, "memory") or not agent.memory:
        return None
    return agent.memory


@tool(
    name="recall_campaigns",
    description=(
        "List research campaigns with their description, target, progress, "
        "and plan status. Use this to see what experiments are planned or "
        "in progress."
    ),
    category=ToolCategory.UTILITY,
    examples=[
        ToolExample(
            user_query="What campaigns do we have?",
            tool_input={"status": "active"},
        ),
        ToolExample(
            user_query="Show me all past campaigns",
            tool_input={"status": "all"},
        ),
    ],
)
async def recall_campaigns(
    status: str = "active",
    context: Dict = None,
) -> str:
    """List campaigns filtered by status."""
    memory = _get_memory(context)
    if not memory:
        return "No memory available (context store not connected)"
    return memory.recall_campaigns(status=status)


@tool(
    name="recall_learnings",
    description=(
        "Search or list what the lab has learned from past experiments. "
        "Returns insights with confidence levels and supporting evidence."
    ),
    category=ToolCategory.UTILITY,
    examples=[
        ToolExample(
            user_query="What have we learned about hatching?",
            tool_input={"query": "hatching"},
        ),
        ToolExample(
            user_query="Show me recent learnings",
            tool_input={},
        ),
    ],
)
async def recall_learnings(
    query: str = None,
    limit: int = 20,
    context: Dict = None,
) -> str:
    """Search or list learnings."""
    memory = _get_memory(context)
    if not memory:
        return "No memory available (context store not connected)"
    return memory.recall_learnings(query=query, limit=limit)


@tool(
    name="recall_observations",
    description=(
        "Search or list observations from past sessions. Can filter by "
        "keyword or embryo ID."
    ),
    category=ToolCategory.UTILITY,
    examples=[
        ToolExample(
            user_query="What did we observe about embryo_3?",
            tool_input={"embryo_id": "embryo_3"},
        ),
        ToolExample(
            user_query="Any observations about stage transitions?",
            tool_input={"query": "stage transition"},
        ),
    ],
)
async def recall_observations(
    query: str = None,
    embryo_id: str = None,
    limit: int = 20,
    context: Dict = None,
) -> str:
    """Search or list observations."""
    memory = _get_memory(context)
    if not memory:
        return "No memory available (context store not connected)"
    return memory.recall_observations(query=query, embryo_id=embryo_id, limit=limit)


@tool(
    name="recall_context",
    description=(
        "Get a full snapshot of accumulated knowledge: campaigns, learnings, "
        "expectations, watchpoints, and open questions. Use this to get "
        "caught up on what has happened across sessions."
    ),
    category=ToolCategory.UTILITY,
    examples=[
        ToolExample(
            user_query="Catch me up on everything",
            tool_input={},
        ),
    ],
)
async def recall_context(
    campaign_id: str = None,
    context: Dict = None,
) -> str:
    """Full context snapshot."""
    memory = _get_memory(context)
    if not memory:
        return "No memory available (context store not connected)"
    return memory.recall_full_context(campaign_id=campaign_id)
