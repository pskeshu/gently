"""Notebook (shared lab notebook) read routes.

Exposes the notebook's Notes for the Notebook tab + Agent's-View live edge.
Read-only here; authoring/curation come in a later increment.
"""

from fastapi import APIRouter, HTTPException

from gently.harness.memory.notebook import Author, NoteKind, NoteStatus, note_to_dict


def _coerce(enum_cls, value):
    """Parse a query-param string into an enum; invalid/None → None (no filter)."""
    if value is None:
        return None
    try:
        return enum_cls(value)
    except ValueError:
        return None


def create_router(server) -> APIRouter:
    router = APIRouter()

    def _nb():
        cs = getattr(server, "context_store", None)
        return cs.notebook if cs is not None else None

    @router.get("/api/notebook/notes")
    async def list_notes(
        kind: str | None = None,
        author: str | None = None,
        status: str | None = None,
        strain: str | None = None,
        embryo: str | None = None,
        thread: str | None = None,
    ):
        nb = _nb()
        if nb is None:
            return {"available": False, "notes": []}
        notes = nb.query_notes(
            kind=_coerce(NoteKind, kind),
            author=_coerce(Author, author),
            status=_coerce(NoteStatus, status),
            strain=strain,
            embryo=embryo,
            thread=thread,
        )
        return {"available": True, "notes": [note_to_dict(n) for n in notes]}

    @router.get("/api/notebook/notes/{note_id}")
    async def get_note(note_id: str):
        nb = _nb()
        if nb is None:
            raise HTTPException(status_code=404, detail="notebook unavailable")
        note = nb.get_note(note_id)
        if note is None:
            raise HTTPException(status_code=404, detail="note not found")
        return note_to_dict(note)

    @router.get("/api/notebook/threads")
    async def list_threads():
        nb = _nb()
        if nb is None:
            return {"available": False, "threads": []}
        counts: dict[str, int] = {}
        for n in nb.query_notes():
            for t in n.threads:
                counts[t] = counts.get(t, 0) + 1
        threads = [{"id": t, "count": c} for t, c in sorted(counts.items())]
        return {"available": True, "threads": threads}

    return router
