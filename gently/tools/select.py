"""The selector: which sessions, and the label that choice implies.

Every way of pooling across sessions is this one function with a different
field filled in. A project is a root campaign; a phase is a child campaign a
session was filed under; a tag is a word in the session's own ``session.yaml``.
Fields combine by intersection, and the label joins the parts.
"""

from typing import Any

from fastapi import APIRouter, Body, Depends, HTTPException

from gently.ui.web.auth import require_control


def select_sessions(
    store: Any,
    cs: Any,
    *,
    project: str | None = None,
    phase: str | None = None,
    tag: str | None = None,
) -> dict:
    """``{"label": str, "sessions": [SessionInfo]}``, newest first.

    ``project`` and ``phase`` take a campaign id or shorthand. A phase that is
    itself a root behaves as a project. Unknown ids select nothing."""
    label: list[str] = []
    ids: set[str] | None = None

    def narrow(found: set[str]) -> None:
        nonlocal ids
        ids = found if ids is None else ids & found

    root = None
    if project:
        c = cs.resolve_campaign(project) if cs is not None else None
        root = cs.root_campaign_id(c.id) if c is not None else None
        label.append(c.shorthand or c.description if c is not None else project)
        narrow({s["session_id"] for s in store.list_sessions(project_id=root)} if root else set())
    if phase:
        # By id anywhere; by shorthand only among the given project's phases,
        # since the store resolves shorthands for roots alone.
        c = cs.get_campaign(phase) if cs is not None else None
        if c is None and root:
            c = next(
                (
                    k
                    for k in cs.get_subcampaigns(root)
                    if (k.shorthand or "").lower() == phase.lower()
                ),
                None,
            )
        label.append(c.shorthand or c.description if c is not None else phase)
        found: set[str] = set()
        if c is not None:
            for s in store.list_sessions():
                if any(x.get("phase") == c.id for x in (s.get("projects") or [])):
                    found.add(s["session_id"])
            found |= {si.session_id for si in cs.get_sessions_for_campaign(c.id)}
        narrow(found)
    if tag:
        label.append(tag)
        narrow({s["session_id"] for s in store.list_sessions() if tag in (s.get("tags") or [])})
    if ids is None:
        return {"label": "", "sessions": []}
    sessions = [s for s in store.list_sessions() if s["session_id"] in ids]
    return {"label": " · ".join(label), "sessions": sessions}


def create_router(server) -> APIRouter:
    router = APIRouter()

    def _store():
        bridge = getattr(server, "agent_bridge", None)
        agent = getattr(bridge, "agent", None) if bridge is not None else None
        st = (getattr(agent, "store", None) if agent is not None else None) or getattr(
            server, "gently_store", None
        )
        if st is None:
            raise HTTPException(status_code=503, detail="Store not available")
        return st

    @router.get("/api/select/sessions")
    async def select(project: str | None = None, phase: str | None = None, tag: str | None = None):
        store = _store()
        out = select_sessions(
            store, getattr(server, "context_store", None), project=project, phase=phase, tag=tag
        )
        out["sessions"] = [
            {
                "session_id": s["session_id"],
                "name": s.get("name") or s["session_id"],
                "created_at": s.get("created_at"),
                "project_id": s.get("project_id"),
                "tags": s.get("tags") or [],
            }
            for s in out["sessions"]
        ]
        return out

    @router.put("/api/sessions/{sid}/tags", dependencies=[Depends(require_control)])
    async def put_tags(sid: str, body: dict = Body(default={})):  # noqa: B008
        store = _store()
        tags = body.get("tags")
        if not isinstance(tags, list):
            raise HTTPException(status_code=422, detail="tags must be a list")
        info = store.set_session_tags(sid, [str(t) for t in tags])
        if info is None:
            raise HTTPException(status_code=404, detail="Session not found")
        return {"tags": info.get("tags") or []}

    return router
