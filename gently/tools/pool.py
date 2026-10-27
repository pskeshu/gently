"""The pool: a session tool's rows across the sessions a selector picks.

One function for the browser's figures and the agent's tools alike. Each
row carries its session, a night label that is unique within the pool, and
the selector's label as its condition. Which tools can be pooled, and how a
session answers for each, is the table here.
"""

from typing import Any

from fastapi import APIRouter, HTTPException

from gently.tools.select import select_sessions
from gently.tools.twitch_hatch import SULSTON_TWITCH_TO_HATCH_MIN, session_report

POOLERS = {"twitch_hatch": session_report}
REFERENCES: dict[str, dict[str, float]] = {"twitch_hatch": {"Sulston": SULSTON_TWITCH_TO_HATCH_MIN}}


def pool(
    store: Any,
    cs: Any,
    tool: str,
    *,
    project: str | None = None,
    phase: str | None = None,
    tag: str | None = None,
) -> dict:
    pooler = POOLERS.get(tool)
    if pooler is None:
        raise KeyError(f"no pooler for tool {tool!r}; known: {sorted(POOLERS)}")
    picked = select_sessions(store, cs, project=project, phase=phase, tag=tag)
    # Nights often share a name; a column must not. Add the id when they do.
    names = [s.get("name") or s["session_id"] for s in picked["sessions"]]
    rows: list[dict] = []
    sessions: list[dict] = []
    for s in picked["sessions"]:
        sid = s["session_id"]
        name = s.get("name") or sid
        if names.count(name) > 1:
            name = f"{name} {sid[:4]}"
        rep = pooler(store, sid)
        sessions.append(
            {"session_id": sid, "name": name, "created_at": s.get("created_at"), "n": len(rep)}
        )
        for i, r in enumerate(rep, 1):
            rows.append(
                {**r, "n": i, "session_id": sid, "session_name": name, "condition": picked["label"]}
            )
    return {
        "label": picked["label"],
        "sessions": sessions,
        "rows": rows,
        "reference": REFERENCES.get(tool, {}),
        "sulston_min": SULSTON_TWITCH_TO_HATCH_MIN if tool == "twitch_hatch" else None,
    }


def create_router(server) -> APIRouter:
    router = APIRouter()

    @router.get("/api/tools/{tool}/pool")
    async def pool_route(
        tool: str, project: str | None = None, phase: str | None = None, tag: str | None = None
    ):
        """Every session the selector picks, the tool's rows pooled."""
        bridge = getattr(server, "agent_bridge", None)
        agent = getattr(bridge, "agent", None) if bridge is not None else None
        store = (getattr(agent, "store", None) if agent is not None else None) or getattr(
            server, "gently_store", None
        )
        if store is None:
            raise HTTPException(status_code=503, detail="Store not available")
        try:
            return pool(
                store,
                getattr(server, "context_store", None),
                tool,
                project=project,
                phase=phase,
                tag=tag,
            )
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e

    return router
