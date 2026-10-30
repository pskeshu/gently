"""Ask the notebook — retrieve relevant Notes and synthesize a grounded,
cited answer with Claude. Structural retrieval only (semantic recall is a
later increment). See docs/superpowers/specs/2026-06-16-shared-lab-notebook-design.md §4.
"""

from __future__ import annotations

import re

from .notebook import Note, NotebookStore, embryo_refs


def select_notes(
    store: NotebookStore,
    *,
    thread: str | None = None,
    strain: str | None = None,
    project: str | None = None,
    limit: int = 12,
) -> list[Note]:
    """Structural narrowing: scope by thread/strain/project when given, else recent.
    Returns newest-first, capped at ``limit``. Only what is in the notebook:
    a note the person set aside is out of the agent's reading."""
    notes = store.query_notes(thread=thread, strain=strain, project=project, in_notebook=True)
    return notes[:limit]


_COVERAGE = ("covered", "partial", "not_in_notebook")

# ASK_TOOL pins the structured output. No confidence field — we don't ask the
# model to self-rate (lab rule); "coverage" is a factual grounding classification.
ASK_TOOL = {
    "name": "answer_from_notebook",
    "description": "Return a grounded answer built ONLY from the provided notebook entries.",
    "input_schema": {
        "type": "object",
        "properties": {
            "answer": {"type": "string", "description": "Direct synthesis grounded in the notes."},
            "points": {
                "type": "array",
                "description": "Supporting points, each citing the note ids it rests on.",
                "items": {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string"},
                        "note_ids": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["text", "note_ids"],
                },
            },
            "suggested_next": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Concrete next experiments/moves if the question asks what to do; else empty."
                ),
            },
            "coverage": {
                "type": "string",
                "enum": list(_COVERAGE),
                "description": "How well the provided notes cover the question.",
            },
        },
        "required": ["answer", "points", "coverage"],
    },
}

_SYSTEM = (
    "You reason over a shared lab notebook. Answer ONLY from the notebook entries "
    "provided — every claim must cite the note id(s) it rests on. If the notes do "
    "not contain the answer, say so plainly and set coverage to 'not_in_notebook'. "
    "Never invent facts not in the notes. Call the answer_from_notebook tool."
)


def _render_notes(notes: list[Note]) -> str:
    lines = []
    for n in notes:
        scope = []
        if n.strains:
            scope.append("strains=" + ",".join(n.strains))
        if n.embryos:
            scope.append("embryos=" + ",".join(r["label"] for r in embryo_refs(n)))
        tag = f" [{'; '.join(scope)}]" if scope else ""
        lines.append(f"[{n.id}] ({n.kind.value}){tag} {n.body}")
    return "\n".join(lines)


def build_ask_messages(
    question: str, notes: list[Note], history: list[dict] | None = None
) -> list[dict]:
    """The conversation as the model sees it: the page's earlier turns, then
    the notes as they are now with the new question. The notes go with the
    last message, not the first, so a page picked up again later reads
    today's notes. Same-role turns in a row are joined; the API wants
    alternation."""
    msgs: list[dict] = []
    for t in history or []:
        role = "assistant" if t.get("role") == "agent" else "user"
        text = (t.get("text") or "").strip()
        if not text:
            continue
        if msgs and msgs[-1]["role"] == role:
            msgs[-1]["content"] += "\n\n" + text
        else:
            msgs.append({"role": role, "content": text})
    body = (
        "Notebook entries:\n" + _render_notes(notes) + f"\n\nQuestion: {question}\n\n"
        "Answer using only these entries, citing note ids."
    )
    if msgs and msgs[-1]["role"] == "user":
        msgs[-1]["content"] += "\n\n" + body
    else:
        msgs.append({"role": "user", "content": body})
    return msgs


# Tool-call markup the model has been seen leaking into its own field values.
_PARAM_TAG = re.compile(r"</?parameter\b[^>]*>")


def _text(value) -> str:
    """A plain string, with any leaked tool-call markup stripped."""
    if value is None:
        return ""
    return _PARAM_TAG.sub("", str(value)).strip()


def _ids(value) -> list[str]:
    if isinstance(value, str):
        value = [value]
    if not isinstance(value, list):
        return []
    return [s.strip() for s in value if isinstance(s, str) and s.strip()]


def normalize_answer(raw) -> dict:
    """Coerce whatever the model put in the tool call into the shape the page
    renders: ``answer`` a string, ``points`` a list of ``{text, note_ids}``,
    ``suggested_next`` a list of strings, ``coverage`` one of the enum.

    The schema asks for exactly that, but a garbled call still arrives as a
    normal tool_use block — seen in the wild: ``points`` as a string carrying a
    stray ``<parameter name="text">`` tag, with the point's ``text`` and
    ``note_ids`` spilled to the top level. Whatever can be salvaged is kept as
    points; nothing here raises, so the page never has to show a generic
    "something went wrong" for a shape it could have rendered."""
    if not isinstance(raw, dict):
        raw = {}
    points = raw.get("points")
    if isinstance(points, str):
        points = [points]
    out_points: list[dict] = []
    for p in points if isinstance(points, list) else []:
        if isinstance(p, dict):
            text = _text(p.get("text"))
            if text:
                out_points.append({"text": text, "note_ids": _ids(p.get("note_ids"))})
        else:
            text = _text(p)
            if text:
                out_points.append({"text": text, "note_ids": []})
    spilled = _text(raw.get("text"))  # a point whose fields landed at the top level
    if spilled:
        out_points.append({"text": spilled, "note_ids": _ids(raw.get("note_ids"))})
    nexts = raw.get("suggested_next")
    if isinstance(nexts, str):
        nexts = [nexts]
    out_next = [_text(s) for s in nexts if _text(s)] if isinstance(nexts, list) else []
    coverage = raw.get("coverage")
    if coverage not in _COVERAGE:
        coverage = "partial" if out_points else "not_in_notebook"
    return {
        "answer": _text(raw.get("answer")),
        "points": out_points,
        "suggested_next": out_next,
        "coverage": coverage,
    }


async def answer_question(
    client, model: str, question: str, notes: list[Note], history: list[dict] | None = None
) -> dict:
    """Force the ask tool and return its input, normalized to the rendered
    shape (see ``normalize_answer``). ``history`` is the page's earlier turns,
    dicts with ``role`` (person | agent) and ``text``. Short-circuits (no API
    call) when there are no notes to ground on."""
    if not notes:
        return {
            "answer": "The notebook doesn't cover this yet.",
            "points": [],
            "suggested_next": [],
            "coverage": "not_in_notebook",
        }
    resp = await client.messages.create(
        model=model,
        max_tokens=2048,
        system=_SYSTEM,
        tools=[ASK_TOOL],
        tool_choice={"type": "tool", "name": ASK_TOOL["name"]},
        messages=build_ask_messages(question, notes, history),
    )
    for block in resp.content:
        if getattr(block, "type", None) == "tool_use":
            return normalize_answer(block.input)
    return {"answer": "", "points": [], "suggested_next": [], "coverage": "not_in_notebook"}
