"""The asks addressed to a person: lines on sessions and projects that name
them with @.

Nothing here is a second record. An ask is a notebook note with the person
in its ``mentions``; whether it is done is read off what that person wrote
on the same session or project afterwards, a line of their own or a tool
note from saving marks. The one thing written is which asks they have
opened: ``agent/people/{id}/seen.yaml``, one small file per person. So the
list survives restarts and browsers, and nothing in it can go stale.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Body, Request

from gently.core.file_store import _read_yaml, _write_yaml
from gently.harness.memory.notebook import Note, note_to_dict
from gently.ui.web.auth import current_username


def known_people(server: Any) -> list[str]:
    """Ids a line can ask: the account names when there are accounts, else
    the operators this rig has seen. The same list /api/auth/users shows."""
    from gently.ui.web.accounts import get_account_store
    from gently.ui.web.operators import list_operators

    acc = get_account_store()
    if acc is not None and acc.has_users():
        return [str(u["username"]) for u in acc.list_users()]
    return [str(u["id"]) for u in list_operators(getattr(server, "gently_store", None))]


def _seen_path(cs: Any, pid: str) -> Path:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", pid)
    return Path(cs.agent_dir) / "people" / safe / "seen.yaml"


def read_seen(cs: Any, pid: str) -> set[str]:
    doc = _read_yaml(_seen_path(cs, pid)) or {}
    return {str(x) for x in (doc.get("seen") or [])}


def mark_seen(cs: Any, pid: str, ids: list[str]) -> set[str]:
    """Remember that this person has opened these asks. Ids of notes that no
    longer exist stay in the file; they match nothing and cost nothing."""
    seen = read_seen(cs, pid) | {str(x) for x in ids if x}
    _write_yaml(
        _seen_path(cs, pid),
        {"seen": sorted(seen), "at": datetime.now().isoformat(timespec="seconds")},
    )
    return seen


def _item(store: Any, cs: Any, n: Note) -> dict:
    sid = n.sessions[0] if n.sessions else None
    sess = (store.get_session(sid) or {}) if (sid and store is not None) else {}
    pid = n.projects[0] if n.projects else None
    if pid is None and sid and store is not None:
        links = store.session_links(sid)
        pid = links[0].get("id") if links else None
    c = cs.get_campaign(pid) if pid else None
    return {
        "note": note_to_dict(n),
        "session": {"id": sid, "name": sess.get("name")} if sid else None,
        "project": {"id": c.id, "name": c.shorthand or c.description} if c is not None else None,
    }


def inbox_for(store: Any, cs: Any, pid: str) -> dict:
    """What names this person, newest first, in three piles. New: not yet
    opened. Open: opened, and nothing written by them on that session or
    project since. Done: they wrote there after the ask. An ask with no
    place to answer is done once opened."""
    nb = cs.notebook
    asks = nb.query_notes(person=pid)
    seen = read_seen(cs, pid)
    # ponytail: a scan for what this person wrote; a `by` index on the notebook when it grows.
    mine = [n for n in nb.query_notes() if n.by == pid]

    def answered(a: Note) -> bool:
        if a.sessions:
            where = set(a.sessions)
            return any(n.updated_at > a.created_at and set(n.sessions) & where for n in mine)
        if a.projects:
            where = set(a.projects)
            return any(n.updated_at > a.created_at and set(n.projects) & where for n in mine)
        return a.id in seen

    piles: dict[str, list[dict]] = {"new": [], "open": [], "done": []}
    for a in asks:
        pile = "done" if answered(a) else ("open" if a.id in seen else "new")
        piles[pile].append(_item(store, cs, a))
    piles["done"] = piles["done"][:20]
    return {"me": pid, **piles}


def create_router(server: Any) -> APIRouter:
    router = APIRouter()

    def _stores():
        cs = getattr(server, "context_store", None)
        bridge = getattr(server, "agent_bridge", None)
        agent = getattr(bridge, "agent", None) if bridge is not None else None
        store = (getattr(agent, "store", None) if agent is not None else None) or getattr(
            server, "gently_store", None
        )
        return store, cs

    @router.get("/api/inbox")
    async def get_inbox(request: Request):
        """The asks that name whoever this request is."""
        store, cs = _stores()
        me = current_username(request)
        if cs is None or not me:
            return {"me": me, "new": [], "open": [], "done": []}
        return inbox_for(store, cs, me)

    @router.post("/api/inbox/seen")
    async def post_seen(request: Request, body: dict = Body(default={})):  # noqa: B008
        """This person opened these asks. Returns the inbox as it now stands."""
        store, cs = _stores()
        me = current_username(request)
        if cs is None or not me:
            return {"me": me, "new": [], "open": [], "done": []}
        mark_seen(cs, me, [str(x) for x in (body.get("ids") or [])])
        return inbox_for(store, cs, me)

    return router
