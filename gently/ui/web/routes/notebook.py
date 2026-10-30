"""Notebook (shared lab notebook) read routes.

Exposes the notebook's Notes for the Notebook tab + Agent's-View live edge.
Reading for everyone; writing a note or settling a finding is a control
holder's action, the same store methods the agent's tools use.
"""

import asyncio
import logging

from fastapi import APIRouter, Body, Depends, HTTPException, Request

from gently.harness.memory.notebook import (
    Author,
    Note,
    NoteKind,
    NoteStatus,
    as_tags,
    embryo_refs,
    mentions_in,
    note_to_dict,
)
from gently.ui.web.auth import current_username, require_control

logger = logging.getLogger(__name__)


def _for_view(note: Note) -> dict:
    """A note as the page draws it: what is on disk, and each embryo with the
    session it belongs to. The page is handed the label; it does not build it."""
    out = note_to_dict(note)
    out["embryo_refs"] = embryo_refs(note)
    return out


def _shelf(value: str | None) -> bool | None:
    """The ``shelf`` query param as the store's ``in_notebook`` filter."""
    return {"notebook": True, "aside": False}.get(value or "")


def _coerce(enum_cls, value):
    """Parse a query-param string into an enum; invalid/None → None (no filter)."""
    if value is None:
        return None
    try:
        return enum_cls(value)
    except ValueError:
        return None


def person_note(server, body: dict, by: str | None) -> Note:
    """A person's note from a request body: kind observation, finding or
    question; facets as lists or comma strings. A finding written by a person
    is confirmed; the agent's findings arrive proposed through its tool.
    "@ryan …" is an ask: the people named, if the rig knows them, go in
    mentions, and the line reaches their inbox. Raises 400 on an empty body."""
    text = (body.get("body") or "").strip()
    if not text:
        raise HTTPException(status_code=400, detail="body required")
    kind = _coerce(NoteKind, body.get("kind")) or NoteKind.OBSERVATION
    status = _coerce(NoteStatus, body.get("status")) or (
        NoteStatus.OPEN if kind == NoteKind.QUESTION else NoteStatus.CONFIRMED
    )
    from gently.ui.web.routes.inbox import known_people

    mentions = list(
        dict.fromkeys(as_tags(body.get("mentions")) + mentions_in(text, known_people(server)))
    )
    return Note(
        id="",
        kind=kind,
        body=text,
        author=Author.HUMAN,
        title=(body.get("title") or None),
        status=status,
        strains=body.get("strains") or [],
        embryos=body.get("embryos") or [],
        sessions=body.get("sessions") or [],
        threads=body.get("threads") or [],
        projects=body.get("projects") or [],
        mentions=mentions,
        basis=list(body.get("basis") or []),
        by=by,
    )


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
        project: str | None = None,
        session: str | None = None,
        by: str | None = None,
        limit: int | None = None,
        shelf: str | None = None,
    ):
        """``shelf`` picks the set: ``notebook`` is what the agent reads,
        ``aside`` is what was set aside, and none is the record, every note,
        which the stream, the inbox and the project views draw from."""
        nb = _nb()
        if nb is None:
            return {"available": False, "notes": []}
        # Off the event loop: the notebook is one small file per note.
        notes = await asyncio.to_thread(
            nb.query_notes,
            kind=_coerce(NoteKind, kind),
            author=_coerce(Author, author),
            status=_coerce(NoteStatus, status),
            strain=strain,
            embryo=embryo,
            thread=thread,
            project=project,
            session=session or None,
            in_notebook=_shelf(shelf),
        )
        if by:
            notes = [n for n in notes if n.by == by]  # a person's own lines
        if limit is not None and limit >= 0:
            notes = notes[:limit]
        return {"available": True, "notes": [_for_view(n) for n in notes]}

    @router.get("/api/notebook/notes/{note_id}")
    async def get_note(note_id: str):
        nb = _nb()
        if nb is None:
            raise HTTPException(status_code=404, detail="notebook unavailable")
        note = nb.get_note(note_id)
        if note is None:
            raise HTTPException(status_code=404, detail="note not found")
        return _for_view(note)

    @router.get("/api/notebook/threads")
    async def list_threads(shelf: str | None = None):
        nb = _nb()
        if nb is None:
            return {"available": False, "threads": []}
        counts: dict[str, int] = {}
        for n in nb.query_notes(in_notebook=_shelf(shelf)):
            for t in n.threads:
                counts[t] = counts.get(t, 0) + 1
        threads = [{"id": t, "count": c} for t, c in sorted(counts.items())]
        return {"available": True, "threads": threads}

    @router.post("/api/notebook/ask")
    async def ask(
        question: str = Body(..., embed=True),
        thread: str | None = Body(None, embed=True),
        strain: str | None = Body(None, embed=True),
        project: str | None = Body(None, embed=True),
    ):
        nb = _nb()
        if nb is None:
            return {"available": False}
        from gently.harness.memory.notebook_ask import answer_question, select_notes
        from gently.settings import settings

        notes = select_notes(nb, thread=thread, strain=strain, project=project)
        try:
            client = getattr(server, "claude_async", None)
            if client is None:
                import anthropic

                client = anthropic.AsyncAnthropic()
            result = await answer_question(client, settings.models.main, question, notes)
        except Exception as e:
            # A 200 with the reason, not a 500: the page says what failed
            # (no API key, the model refusing, the network) instead of a
            # generic "something went wrong", and the log keeps the traceback.
            logger.exception("Ask the notebook failed")
            return {"available": False, "error": f"{type(e).__name__}: {e}"}
        result["available"] = True
        result["note_ids"] = [n.id for n in notes]
        return result

    @router.post("/api/notebook/notes", dependencies=[Depends(require_control)])
    async def write_note(request: Request):
        """A person's note: kind observation, finding or question; facets as
        lists or comma strings. A finding written here is confirmed, since the
        person wrote it; the agent's findings arrive proposed through its tool."""
        body = await request.json()
        nb = _nb()
        if nb is None:
            raise HTTPException(status_code=503, detail="notebook unavailable")
        note = person_note(server, body, current_username(request))
        note_id = nb.write_note(note)
        try:
            from gently.core.event_bus import EventType, emit

            emit(EventType.CONTEXT_UPDATED, {"kind": "note"}, source="notebook.write")
        except Exception:
            pass
        return {"ok": True, "note": _for_view(nb.get_note(note_id))}

    @router.post("/api/notebook/notes/{note_id}/status", dependencies=[Depends(require_control)])
    async def set_note_status(note_id: str, request: Request):
        """Settle a note: confirm a proposed finding, answer a question, or set
        it superseded (dismissed and kept). Superseding with a successor is the
        store's supersede_note; this is the plain status change."""
        status = (await request.json()).get("status")
        nb = _nb()
        if nb is None:
            raise HTTPException(status_code=503, detail="notebook unavailable")
        note = nb.get_note(note_id)
        if note is None:
            raise HTTPException(status_code=404, detail="note not found")
        st = _coerce(NoteStatus, status)
        if st is None:
            raise HTTPException(status_code=400, detail=f"unknown status {status!r}")
        note.status = st
        nb.write_note(note)
        try:
            from gently.core.event_bus import EventType, emit

            emit(EventType.CONTEXT_UPDATED, {"kind": "note"}, source="notebook.status")
        except Exception:
            pass
        return {"ok": True, "note": _for_view(nb.get_note(note_id))}

    def _changed(source: str) -> None:
        try:
            from gently.core.event_bus import EventType, emit

            emit(EventType.CONTEXT_UPDATED, {"kind": "note"}, source=source)
        except Exception:
            pass

    @router.post(
        "/api/notebook/notes/{note_id}/in_notebook", dependencies=[Depends(require_control)]
    )
    async def set_in_notebook(note_id: str, request: Request):
        """Set a note aside from the notebook (``{"value": false}``) or put it
        back (``true``). The line stays in the stream and the inbox either
        way; what changes is whether the agent reads it."""
        value = (await request.json()).get("value")
        if not isinstance(value, bool):
            raise HTTPException(status_code=400, detail="value must be true or false")
        nb = _nb()
        if nb is None:
            raise HTTPException(status_code=503, detail="notebook unavailable")
        note = nb.set_in_notebook(note_id, value)
        if note is None:
            raise HTTPException(status_code=404, detail="note not found")
        _changed("notebook.in_notebook")
        return {"ok": True, "note": _for_view(note)}

    @router.delete("/api/notebook/notes/{note_id}", dependencies=[Depends(require_control)])
    async def delete_note(note_id: str):
        """Take a note off the page. Its file is kept under removed/, so the
        reply says it is restorable and the page can offer Undo."""
        nb = _nb()
        if nb is None:
            raise HTTPException(status_code=503, detail="notebook unavailable")
        note = nb.get_note(note_id)
        if note is None or not nb.delete_note(note_id):
            raise HTTPException(status_code=404, detail="note not found")
        _changed("notebook.delete")
        return {"ok": True, "restorable": True, "note": _for_view(note)}

    @router.post("/api/notebook/notes/{note_id}/restore", dependencies=[Depends(require_control)])
    async def restore_note(note_id: str):
        """Undo a delete: the note comes back exactly as it was."""
        nb = _nb()
        if nb is None:
            raise HTTPException(status_code=503, detail="notebook unavailable")
        note = nb.restore_note(note_id)
        if note is None:
            raise HTTPException(status_code=404, detail="nothing to restore under that id")
        _changed("notebook.restore")
        return {"ok": True, "note": _for_view(note)}

    return router
