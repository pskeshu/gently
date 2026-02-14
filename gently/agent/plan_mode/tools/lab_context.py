"""
Lab context tools — query the lab's own history and capabilities.

These tools let the copilot search past sessions, existing campaigns,
learnings, and hardware specs to inform experimental design.
"""

from typing import Dict, Optional

from ...tool_registry import tool, ToolCategory, ToolExample


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
    context: Dict = None,
) -> str:
    """Search lab history for relevant context."""
    copilot = context.get("copilot") if context else None
    if not copilot or not hasattr(copilot, "context_store") or not copilot.context_store:
        return "No lab history available (context store not connected)"

    store = copilot.context_store
    query_lower = query.lower()

    results = []

    # Search campaigns
    campaigns = store.get_active_campaigns()
    all_campaigns_rows = store._conn.execute(
        "SELECT * FROM campaigns ORDER BY created_at DESC LIMIT 50"
    ).fetchall()
    matching_campaigns = []
    for row in all_campaigns_rows:
        d = dict(row)
        text = f"{d.get('description', '')} {d.get('shorthand', '')} {d.get('target', '')}".lower()
        if any(term in text for term in query_lower.split()):
            matching_campaigns.append(d)
    if matching_campaigns:
        results.append("## Matching Campaigns")
        for c in matching_campaigns[:5]:
            status = c.get("status", "?")
            results.append(
                f"- [{status}] {c['description']}"
                f"{' (' + c['shorthand'] + ')' if c.get('shorthand') else ''}"
                f" (id: {c['id']})"
            )

    # Search learnings
    learnings = store.get_learnings(limit=100)
    matching_learnings = [
        l for l in learnings
        if any(term in l.content.lower() for term in query_lower.split())
    ]
    if matching_learnings:
        results.append("\n## Relevant Learnings")
        for l in matching_learnings[:10]:
            results.append(f"- {l.content} (confidence: {l.confidence.value})")

    # Search observations
    observations = store.get_recent_observations(limit=100)
    matching_obs = [
        o for o in observations
        if any(term in o.content.lower() for term in query_lower.split())
    ]
    if matching_obs:
        results.append("\n## Relevant Observations")
        for o in matching_obs[:5]:
            time_str = o.timestamp.strftime("%Y-%m-%d %H:%M")
            results.append(f"- [{time_str}] {o.content}")

    # Search session intents
    intent_rows = store._conn.execute(
        "SELECT * FROM session_intents ORDER BY created_at DESC LIMIT 50"
    ).fetchall()
    matching_intents = []
    for row in intent_rows:
        d = dict(row)
        text = f"{d.get('planned_intent', '')} {d.get('actual_summary', '')}".lower()
        if any(term in text for term in query_lower.split()):
            matching_intents.append(d)
    if matching_intents:
        results.append("\n## Matching Sessions")
        for s in matching_intents[:5]:
            intent = s.get("planned_intent", "no intent recorded")
            results.append(f"- Session {s['session_id']}: {intent}")
            if s.get("actual_summary"):
                results.append(f"  Result: {s['actual_summary'][:100]}")

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
    context: Dict = None,
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
