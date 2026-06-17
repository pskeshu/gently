"""
Memory recall tools — let the agent query its persistent memory on demand.

Thin wrappers around AgentMemory. Available in both run and plan modes.
"""

from gently.harness.tools.registry import ToolCategory, ToolExample, tool


def _get_memory(context: dict | None):
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
    context: dict | None = None,
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
    query: str | None = None,
    limit: int = 20,
    context: dict | None = None,
) -> str:
    """Search or list learnings."""
    memory = _get_memory(context)
    if not memory:
        return "No memory available (context store not connected)"
    return memory.recall_learnings(query=query, limit=limit)


@tool(
    name="recall_observations",
    description=(
        "Search or list observations from past sessions. Can filter by keyword or embryo ID."
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
    query: str | None = None,
    embryo_id: str | None = None,
    limit: int = 20,
    context: dict | None = None,
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
    campaign_id: str | None = None,
    context: dict | None = None,
) -> str:
    """Full context snapshot."""
    memory = _get_memory(context)
    if not memory:
        return "No memory available (context store not connected)"
    return memory.recall_full_context(campaign_id=campaign_id)


@tool(
    name="record_note",
    description=(
        "Record a note from the researcher into the shared lab notebook. Use this when the "
        "user says 'note that…', 'add a note…', 'remember that…'. First tidy the phrasing for "
        "clarity — keep their meaning and any specifics (numbers, strains, stages) — then save. "
        "The note is attributed to the human and tagged to the current session."
    ),
    category=ToolCategory.UTILITY,
    examples=[
        ToolExample(
            user_query="Note that the 4-embryo test ran clean — 329 timepoints, system nominal.",
            tool_input={
                "text": "Test run on 4 calibration embryos was clean: 329 timepoints, "
                "system behaved as expected."
            },
        ),
    ],
)
async def record_note(
    text: str,
    embryos: list[str] | None = None,
    strains: list[str] | None = None,
    context: dict | None = None,
) -> str:
    """Write a human-authored note into the notebook, tagged to the current session."""
    agent = context.get("agent") if context else None
    cs = getattr(agent, "context_store", None) if agent else None
    if cs is None:
        return "No notebook available (context store not connected)"
    from gently.harness.memory.notebook import Author, Note, NoteKind

    session_id = getattr(agent, "session_id", None)
    note = Note(
        id="",
        kind=NoteKind.OBSERVATION,
        body=text,
        author=Author.HUMAN,
        sessions=[session_id] if session_id else [],
        embryos=embryos or [],
        strains=strains or [],
    )
    note_id = cs.notebook.write_note(note)
    # Refresh the Notebook tab + Agent's-view live edge (both ride CONTEXT_UPDATED).
    try:
        from gently.core.event_bus import EventType, emit

        emit(EventType.CONTEXT_UPDATED, {"kind": "note"}, source="record_note")
    except Exception:
        pass
    scope = "this session" if session_id else "the notebook (no active session)"
    return f"Noted (id {note_id}) — saved to {scope}, attributed to you."
