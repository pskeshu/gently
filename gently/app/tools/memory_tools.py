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


def _project_of(agent, cs, project: str | None) -> str | None:
    """The project a note belongs to: the one named, else the live session's home."""
    if project:
        c = cs.resolve_campaign(project)
        if c is not None:
            return cs.root_campaign_id(c.id) or c.id
    store = getattr(agent, "store", None)
    sid = getattr(agent, "session_id", None)
    if store is not None and sid:
        try:
            return (store.get_session(sid) or {}).get("project_id")
        except Exception:
            return None
    return None


@tool(
    name="record_note",
    description=(
        "Record a note from the researcher into the shared lab notebook. Use this when the "
        "user says 'note that…', 'add a note…', 'remember that…'. First tidy the phrasing for "
        "clarity — keep their meaning and any specifics (numbers, strains, stages) — then save. "
        "The note is attributed to the human and tagged to the current session and its "
        "project. kind is 'observation' (default), 'finding' (a claim) or 'question'; "
        "project names a project by shorthand or id when the note is about another one."
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
    embryos: list[str] | str | None = None,
    strains: list[str] | str | None = None,
    kind: str = "observation",
    project: str | None = None,
    title: str | None = None,
    context: dict | None = None,
) -> str:
    """Write a human-authored note into the notebook, tagged to the current
    session and its project.

    ``embryos`` and ``strains`` arrive as a list or as one comma-separated
    string, whichever the model chose. The Note makes tags of either.
    """
    agent = context.get("agent") if context else None
    cs = getattr(agent, "context_store", None) if agent else None
    if cs is None:
        return "No notebook available (context store not connected)"
    from gently.harness.memory.notebook import Author, Note, NoteKind, NoteStatus, as_tags

    try:
        note_kind = NoteKind(kind or "observation")
    except ValueError:
        note_kind = NoteKind.OBSERVATION
    session_id = getattr(agent, "session_id", None)
    project_id = _project_of(agent, cs, project)
    note = Note(
        id="",
        kind=note_kind,
        body=text,
        author=Author.HUMAN,
        title=title,
        status=NoteStatus.OPEN if note_kind == NoteKind.QUESTION else NoteStatus.CONFIRMED,
        sessions=[session_id] if session_id else [],
        projects=[project_id] if project_id else [],
        embryos=as_tags(embryos),
        strains=as_tags(strains),
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


@tool(
    name="propose_finding",
    description=(
        "File a finding of your own — a claim the data supports — into the project's "
        "ledger as PROPOSED, for the researcher to confirm, supersede or dismiss on the "
        "Projects tab. Cite the note ids it rests on in basis. Never confirm it yourself."
    ),
    category=ToolCategory.UTILITY,
    examples=[
        ToolExample(
            user_query="(after two nights at 25 °C)",
            tool_input={
                "text": (
                    "At 25 °C the twitching-to-hatching interval is about 18 % shorter than "
                    "the 20 °C baseline; two embryos, so a lead, not a result."
                ),
                "basis": ["n_4f2a", "n_9c01"],
            },
        ),
    ],
)
async def propose_finding(
    text: str,
    basis: list[str] | str | None = None,
    project: str | None = None,
    strains: list[str] | str | None = None,
    context: dict | None = None,
) -> str:
    agent = context.get("agent") if context else None
    cs = getattr(agent, "context_store", None) if agent else None
    if cs is None:
        return "No notebook available (context store not connected)"
    from gently.harness.memory.notebook import Author, Note, NoteKind, NoteStatus, as_tags

    session_id = getattr(agent, "session_id", None)
    project_id = _project_of(agent, cs, project)
    note = Note(
        id="",
        kind=NoteKind.FINDING,
        body=text,
        author=Author.AGENT,
        status=NoteStatus.PROPOSED,
        sessions=[session_id] if session_id else [],
        projects=[project_id] if project_id else [],
        strains=as_tags(strains),
        basis=as_tags(basis),
    )
    note_id = cs.notebook.write_note(note)
    try:
        from gently.core.event_bus import EventType, emit

        emit(EventType.CONTEXT_UPDATED, {"kind": "note"}, source="propose_finding")
    except Exception:
        pass
    where = f"project {project_id[:8]}" if project_id else "the notebook (no project)"
    return f"Proposed finding {note_id} in {where}; it waits for the researcher to confirm."


@tool(
    name="record_expectation",
    description=(
        "Record what you expect to see and by when, so the ledger's Predicted column "
        "can be checked against what happens: target (an embryo id or 'project'), "
        "prediction, expected_time (ISO 8601), uncertainty, basis."
    ),
    category=ToolCategory.UTILITY,
    examples=[
        ToolExample(
            user_query="(baseline says comma at ~400 min)",
            tool_input={
                "target": "embryo_3",
                "prediction": "reaches comma",
                "expected_time": "2026-10-20T14:20:00",
                "uncertainty": "±30 min",
                "basis": "715 min baseline interval",
            },
        ),
    ],
)
async def record_expectation(
    target: str,
    prediction: str,
    expected_time: str,
    uncertainty: str | None = None,
    basis: str | None = None,
    project: str | None = None,
    context: dict | None = None,
) -> str:
    import uuid
    from datetime import datetime

    agent = context.get("agent") if context else None
    cs = getattr(agent, "context_store", None) if agent else None
    if cs is None:
        return "No memory available (context store not connected)"
    from gently.harness.memory.model import Expectation

    try:
        when = datetime.fromisoformat(expected_time)
    except (TypeError, ValueError):
        return f"expected_time must be ISO 8601, got {expected_time!r}"
    exp = Expectation(
        id=str(uuid.uuid4())[:8],
        target=target,
        prediction=prediction,
        expected_time=when,
        uncertainty=uncertainty,
        basis=basis,
        session_id=getattr(agent, "session_id", None),
        project_id=_project_of(agent, cs, project),
    )
    cs.add_expectation(exp)
    return f"Expectation {exp.id} recorded: {target} {prediction} by {when:%Y-%m-%d %H:%M}."


@tool(
    name="raise_question",
    description=(
        "Put an open question into the project's ledger — something the data cannot "
        "settle yet. The researcher answers or closes it on the Projects tab."
    ),
    category=ToolCategory.UTILITY,
    examples=[
        ToolExample(
            user_query="(e5 arrested at 2-fold)",
            tool_input={"content": "Why did e5 arrest at 2-fold on 29 Sep?"},
        ),
    ],
)
async def raise_question(
    content: str,
    project: str | None = None,
    context: dict | None = None,
) -> str:
    import uuid

    agent = context.get("agent") if context else None
    cs = getattr(agent, "context_store", None) if agent else None
    if cs is None:
        return "No memory available (context store not connected)"
    from gently.harness.memory.model import Question

    q = Question(
        id=str(uuid.uuid4())[:8],
        content=content,
        session_id=getattr(agent, "session_id", None),
        project_id=_project_of(agent, cs, project),
    )
    cs.add_question(q)
    return f"Question {q.id} is open: {content}"


@tool(
    name="add_reference",
    description=(
        "Keep a reference on a project: a strain (as_strain=true), a gene, a paper, "
        "an ontology term or a page. Take source, key, title and url from "
        "search_references; campaign_id may be a project's id or shorthand."
    ),
    category=ToolCategory.UTILITY,
    examples=[
        ToolExample(
            user_query="Keep the rab-3 gene page on this project",
            tool_input={
                "campaign_id": "hatching-temp",
                "source": "alliance",
                "key": "WBGene00004267",
                "title": "rab-3",
                "url": "https://wormbase.org/species/c_elegans/gene/WBGene00004267",
            },
        ),
    ],
)
async def add_reference(
    campaign_id: str,
    source: str,
    key: str,
    title: str | None = None,
    url: str | None = None,
    subtitle: str | None = None,
    as_strain: bool = False,
    context: dict | None = None,
) -> str:
    agent = context.get("agent") if context else None
    cs = getattr(agent, "context_store", None) if agent else None
    if cs is None:
        return "No memory available (context store not connected)"
    c = cs.resolve_campaign(campaign_id)
    if c is None:
        return f"Project '{campaign_id}' not found."
    root = cs.root_campaign_id(c.id) or c.id
    entry = cs.add_campaign_reference(
        root,
        {
            "source": source,
            "key": key,
            "title": title,
            "url": url,
            "subtitle": subtitle,
            "added_by": "agent",
        },
        as_strain=bool(as_strain),
    )
    kind = "strain" if as_strain else "link"
    return f"Kept {kind} {entry.get('key')} ({entry.get('source')}) on project {root[:8]}."
