"""
Lab context tools — query the lab's own history and capabilities.

These tools let the agent search past sessions, existing campaigns,
learnings, and hardware specs to inform experimental design.
"""

from ...tools.registry import ToolCategory, ToolExample, tool


@tool(
    name="query_lab_history",
    description=(
        "Search the lab's past sessions, campaigns, and learnings for relevant "
        "context. Use this to find out if similar experiments have been done "
        "before, what parameters worked, and what was learned."
    ),
    category=ToolCategory.UTILITY,
    examples=[
        ToolExample(
            user_query="Have we imaged rab-3p::GFP before?",
            tool_input={"query": "rab-3p GFP imaging"},
        ),
    ],
)
async def query_lab_history(
    query: str,
    context: dict | None = None,
) -> str:
    """Search lab history for relevant context."""
    agent = context.get("agent") if context else None
    if not agent or not hasattr(agent, "context_store") or not agent.context_store:
        return "No lab history available (context store not connected)"

    store = agent.context_store
    query_lower = query.lower()

    results = []

    # Search campaigns
    all_campaigns = store.get_all_campaigns(limit=50)
    matching_campaigns = []
    for c in all_campaigns:
        text = f"{c.description or ''} {c.shorthand or ''} {c.target or ''}".lower()
        if any(term in text for term in query_lower.split()):
            matching_campaigns.append(c)
    if matching_campaigns:
        results.append("## Matching Campaigns")
        for c in matching_campaigns[:5]:
            status = c.status.value if c.status else "?"
            results.append(
                f"- [{status}] {c.description}"
                f"{' (' + c.shorthand + ')' if c.shorthand else ''}"
                f" (id: {c.id})"
            )

    # Search learnings
    learnings = store.get_learnings(limit=100)
    matching_learnings = [
        learning
        for learning in learnings
        if any(term in learning.content.lower() for term in query_lower.split())
    ]
    if matching_learnings:
        results.append("\n## Relevant Learnings")
        for learning in matching_learnings[:10]:
            results.append(f"- {learning.content} (confidence: {learning.confidence.value})")

    # Search observations
    observations = store.get_recent_observations(limit=100)
    matching_obs = [
        o for o in observations if any(term in o.content.lower() for term in query_lower.split())
    ]
    if matching_obs:
        results.append("\n## Relevant Observations")
        for o in matching_obs[:5]:
            time_str = o.timestamp.strftime("%Y-%m-%d %H:%M")
            results.append(f"- [{time_str}] {o.content}")

    # Search session intents
    all_intents = store.get_recent_session_intents(limit=50)
    matching_intents = []
    for si in all_intents:
        text = f"{si.planned_intent or ''} {si.actual_summary or ''}".lower()
        if any(term in text for term in query_lower.split()):
            matching_intents.append(si)
    if matching_intents:
        results.append("\n## Matching Sessions")
        for s in matching_intents[:5]:
            intent = s.planned_intent or "no intent recorded"
            results.append(f"- Session {s.session_id}: {intent}")
            if s.actual_summary:
                results.append(f"  Result: {s.actual_summary[:100]}")

    if not results:
        return f"No matches found for '{query}' in lab history."

    return "\n".join(results)


@tool(
    name="check_hardware_capability",
    description=(
        "Check if the microscope hardware can perform a specific type of "
        "imaging. Returns hardware specs, limits, and recommendations."
    ),
    category=ToolCategory.UTILITY,
    examples=[
        ToolExample(
            user_query="Can we do sub-minute intervals?",
            tool_input={"question": "minimum interval between volumes"},
        ),
    ],
)
async def check_hardware_capability(
    question: str,
    context: dict | None = None,
) -> str:
    """Check hardware capabilities against a question."""
    from gently.hardware import get_hardware

    hardware = get_hardware()
    hardware_desc = hardware.HARDWARE_DESCRIPTION
    hardware_name = hardware.HARDWARE_DISPLAY_NAME

    return (
        f"Hardware: {hardware_name}\n\n"
        f"Refer to the hardware description in your system prompt for detailed "
        f"specifications. Key points relevant to '{question}':\n\n"
        f"{hardware_desc}"
    )
