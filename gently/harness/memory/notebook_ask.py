"""Ask the notebook — retrieve relevant Notes and synthesize a grounded,
cited answer with Claude. Structural retrieval only (semantic recall is a
later increment). See docs/superpowers/specs/2026-06-16-shared-lab-notebook-design.md §4.
"""

from __future__ import annotations

from .notebook import Note, NotebookStore


def select_notes(
    store: NotebookStore,
    *,
    thread: str | None = None,
    strain: str | None = None,
    limit: int = 12,
) -> list[Note]:
    """Structural narrowing: scope by thread/strain when given, else recent.
    Returns newest-first, capped at ``limit``."""
    notes = store.query_notes(thread=thread, strain=strain)
    return notes[:limit]


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
                "enum": ["covered", "partial", "not_in_notebook"],
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
            scope.append("embryos=" + ",".join(n.embryos))
        tag = f" [{'; '.join(scope)}]" if scope else ""
        lines.append(f"[{n.id}] ({n.kind.value}){tag} {n.body}")
    return "\n".join(lines)


def build_ask_messages(question: str, notes: list[Note]) -> list[dict]:
    body = (
        "Notebook entries:\n" + _render_notes(notes) + f"\n\nQuestion: {question}\n\n"
        "Answer using only these entries, citing note ids."
    )
    return [{"role": "user", "content": body}]


async def answer_question(client, model: str, question: str, notes: list[Note]) -> dict:
    """Force the ask tool and return its validated input dict. Short-circuits
    (no API call) when there are no notes to ground on."""
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
        messages=build_ask_messages(question, notes),
    )
    for block in resp.content:
        if getattr(block, "type", None) == "tool_use":
            return block.input
    return {"answer": "", "points": [], "suggested_next": [], "coverage": "not_in_notebook"}
