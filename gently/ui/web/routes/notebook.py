"""Notebook (shared lab notebook) read routes.

Exposes the notebook's Notes for the Notebook tab + Agent's-View live edge.
Reading for everyone; writing a note or settling a finding is a control
holder's action, the same store methods the agent's tools use.
"""

import asyncio

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


def _for_view(note: Note) -> dict:
    """A note as the page draws it: what is on disk, and each embryo with the
    session it belongs to. The page is handed the label; it does not build it."""
    out = note_to_dict(note)
    out["embryo_refs"] = embryo_refs(note)
    return out


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
        project: str | None = None,
        session: str | None = None,
        by: str | None = None,
        limit: int | None = None,
    ):
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
        client = getattr(server, "claude_async", None)
        if client is None:
            import anthropic

            client = anthropic.AsyncAnthropic()
        result = await answer_question(client, settings.models.main, question, notes)
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
        text = (body.get("body") or "").strip()
        if not text:
            raise HTTPException(status_code=400, detail="body required")
        kind = _coerce(NoteKind, body.get("kind")) or NoteKind.OBSERVATION
        status = _coerce(NoteStatus, body.get("status")) or (
            NoteStatus.OPEN if kind == NoteKind.QUESTION else NoteStatus.CONFIRMED
        )
        # "@ryan …" is an ask: the people named, if the rig knows them, go in
        # mentions, and the line reaches their inbox.
        from gently.ui.web.routes.inbox import known_people

        mentions = list(
            dict.fromkeys(as_tags(body.get("mentions")) + mentions_in(text, known_people(server)))
        )
        note = Note(
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
            by=current_username(request),
        )
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

    return router
