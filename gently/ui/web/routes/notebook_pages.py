"""Pages of the notebook: a saved query over the notes with a conversation on
it. See gently/harness/memory/notebook_pages.py for what a page is.

Reading a page is for everyone. Saving or deleting a page, keeping a reply as
a finding, and writing a line from a page are a control holder's actions.
Asking on a page records the turn, but asking is how a viewer reads too, so
it is open like ``/api/notebook/ask``.
"""

import asyncio
import logging

from fastapi import APIRouter, Body, Depends, HTTPException, Request

from gently.harness.memory.notebook_pages import (
    Page,
    Turn,
    clean_query,
    list_pages,
    note_from_reply,
    page_notes,
    page_to_dict,
    pages_of,
    resolve_page,
)
from gently.ui.web.auth import current_username, require_control

from .notebook import _for_view, person_note

logger = logging.getLogger(__name__)


def _changed(source: str) -> None:
    try:
        from gently.core.event_bus import EventType, emit

        emit(EventType.CONTEXT_UPDATED, {"kind": "note"}, source=source)
    except Exception:
        pass


def create_router(server) -> APIRouter:
    router = APIRouter()

    def _stores():
        cs = getattr(server, "context_store", None)
        nb = cs.notebook if cs is not None else None
        if nb is None:
            raise HTTPException(status_code=503, detail="notebook unavailable")
        return cs, nb

    def _page(cs, nb, page_id: str) -> Page:
        page = resolve_page(cs, nb, page_id)
        if page is None:
            raise HTTPException(status_code=404, detail="no such page")
        return page

    def _view(nb, page: Page) -> dict:
        notes = page_notes(nb, page)
        return {"page": page_to_dict(page), "notes": [_for_view(n) for n in notes]}

    @router.get("/api/notebook/pages")
    async def get_pages():
        cs = getattr(server, "context_store", None)
        nb = cs.notebook if cs is not None else None
        if nb is None:
            return {"available": False, "pages": []}

        def _list():
            out = []
            for p in list_pages(cs, nb):
                d = page_to_dict(p)
                d["count"] = len(page_notes(nb, p))
                d["turn_count"] = len(p.turns)
                d.pop("turns", None)
                out.append(d)
            return out

        return {"available": True, "pages": await asyncio.to_thread(_list)}

    @router.post("/api/notebook/pages", dependencies=[Depends(require_control)])
    async def save_page(request: Request):
        """A page of your own: a name and a query over the facets."""
        body = await request.json()
        cs, nb = _stores()
        name = (body.get("name") or "").strip()
        if not name:
            raise HTTPException(status_code=400, detail="name required")
        page = Page(
            id="", name=name, query=clean_query(body.get("query")), by=current_username(request)
        )
        pages_of(nb).save(page)
        return {"ok": True, **_view(nb, page)}

    @router.get("/api/notebook/pages/{page_id}")
    async def get_page(page_id: str):
        cs, nb = _stores()
        page = _page(cs, nb, page_id)
        return {"available": True, **await asyncio.to_thread(_view, nb, page)}

    @router.delete("/api/notebook/pages/{page_id}", dependencies=[Depends(require_control)])
    async def delete_page(page_id: str):
        """Deleting a saved page loses its conversation and nothing else. On
        a built-in, this clears the conversation; the page itself stays."""
        cs, nb = _stores()
        page = _page(cs, nb, page_id)
        pages_of(nb).delete(page_id)
        return {"ok": True, "builtin": page.builtin}

    @router.post("/api/notebook/pages/{page_id}/ask")
    async def ask_on_page(request: Request, page_id: str, question: str = Body(..., embed=True)):
        """A turn of the page's conversation: the question goes to the model
        with the page's notes as they are now and the earlier turns; both the
        question and the reply are kept on the page."""
        from gently.harness.memory.notebook_ask import answer_question
        from gently.settings import settings

        cs, nb = _stores()
        page = _page(cs, nb, page_id)
        q = (question or "").strip()
        if not q:
            raise HTTPException(status_code=400, detail="question required")
        notes = page_notes(nb, page)
        history = [{"role": t.role, "text": t.text} for t in page.turns]
        try:
            client = getattr(server, "claude_async", None)
            if client is None:
                import anthropic

                client = anthropic.AsyncAnthropic()
            result = await answer_question(client, settings.models.main, q, notes, history)
        except Exception as e:
            logger.exception("Ask on page %s failed", page_id)
            return {"ok": False, "error": f"{type(e).__name__}: {e}", **_view(nb, page)}
        page.turns.append(Turn(role="person", text=q, by=current_username(request)))
        page.turns.append(
            Turn(
                role="agent",
                text=result.get("answer") or "",
                note_ids=sorted(
                    {i for p in result.get("points", []) for i in p.get("note_ids", [])}
                ),
                points=list(result.get("points", [])),
                coverage=result.get("coverage"),
            )
        )
        pages_of(nb).save(page)
        return {"ok": True, **_view(nb, page)}

    @router.post(
        "/api/notebook/pages/{page_id}/turns/{index}/keep",
        dependencies=[Depends(require_control)],
    )
    async def keep_reply(page_id: str, index: int):
        """File an agent reply as a finding: proposed until confirmed,
        resting on the notes it cited, where the page is."""
        cs, nb = _stores()
        page = _page(cs, nb, page_id)
        if index < 0 or index >= len(page.turns) or page.turns[index].role != "agent":
            raise HTTPException(status_code=404, detail="no such reply")
        turn = page.turns[index]
        if not turn.kept_as:
            if not turn.text.strip():
                raise HTTPException(status_code=400, detail="nothing to keep")
            turn.kept_as = nb.write_note(note_from_reply(page, turn))
            pages_of(nb).save(page)
            _changed("notebook.page.keep")
        return {"ok": True, "note": _for_view(nb.get_note(turn.kept_as)), **_view(nb, page)}

    @router.post(
        "/api/notebook/pages/{page_id}/turns/clear", dependencies=[Depends(require_control)]
    )
    async def clear_turns(page_id: str):
        """Start the page's conversation over. The notes are untouched."""
        cs, nb = _stores()
        page = _page(cs, nb, page_id)
        page.turns = []
        if page.builtin:
            pages_of(nb).delete(page_id)  # a built-in with no talk needs no file
        else:
            pages_of(nb).save(page)
        return {"ok": True, **_view(nb, page)}

    @router.post("/api/notebook/pages/{page_id}/notes", dependencies=[Depends(require_control)])
    async def write_line(page_id: str, request: Request):
        """A person's line written from a page: filed where the page is
        (its project, thread, strain or session), so it is on the page."""
        body = await request.json()
        cs, nb = _stores()
        page = _page(cs, nb, page_id)
        q = page.query
        filed = {
            "projects": [q["project"]] if q.get("project") else [],
            "threads": [q["thread"]] if q.get("thread") else [],
            "strains": [q["strain"]] if q.get("strain") else [],
            "sessions": [q["session"]] if q.get("session") else [],
        }
        note = person_note(server, {**filed, **(body or {})}, current_username(request))
        note_id = nb.write_note(note)
        _changed("notebook.page.write")
        return {"ok": True, "note": _for_view(nb.get_note(note_id)), **_view(nb, page)}

    return router
