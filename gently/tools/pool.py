"""The pool: a session tool's rows across the sessions a selector picks.

One function for the browser's figures and the agent's tools alike. Each
row carries its session, a session label that is unique within the pool, and
the selector's label as its condition. Which tools can be pooled, and how a
session answers for each, is the table here.
"""

import csv
import io
import re
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

from gently.tools.select import select_sessions
from gently.tools.twitch_hatch import SULSTON_TWITCH_TO_HATCH_MIN, session_report

POOLERS = {"twitch_hatch": session_report}
REFERENCES: dict[str, dict[str, float]] = {"twitch_hatch": {"Sulston": SULSTON_TWITCH_TO_HATCH_MIN}}
# What says which row this is, ahead of what the tool measured.
FIRST = ("session_id", "session_name", "role", "condition", "n")


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
    root = None
    if project and cs is not None:
        c = cs.resolve_campaign(project)
        root = cs.root_campaign_id(c.id) if c is not None else None
    # Sessions often share a name; a column must not. Add the id when they do.
    names = [s.get("name") or s["session_id"] for s in picked["sessions"]]
    rows: list[dict] = []
    sessions: list[dict] = []
    for s in picked["sessions"]:
        sid = s["session_id"]
        name = s.get("name") or sid
        if names.count(name) > 1:
            name = f"{name} {sid[:4]}"
        rep = pooler(store, sid)
        link = (
            next((x for x in (s.get("projects") or []) if x.get("id") == root), None)
            if root
            else None
        )
        role = (link or {}).get("role") or "unassigned"
        sessions.append(
            {
                "session_id": sid,
                "name": name,
                "created_at": s.get("created_at"),
                "n": len(rep),
                "role": role,
            }
        )
        for i, r in enumerate(rep, 1):
            rows.append(
                {
                    **r,
                    "n": i,
                    "session_id": sid,
                    "session_name": name,
                    "role": role,
                    "condition": picked["label"],
                }
            )
    return {
        "label": picked["label"],
        "sessions": sessions,
        "rows": rows,
        "reference": REFERENCES.get(tool, {}),
        "sulston_min": SULSTON_TWITCH_TO_HATCH_MIN if tool == "twitch_hatch" else None,
    }


def pool_csv(pooled: dict) -> str:
    """The pooled rows as CSV: which session, its role and the row's number
    first, then the tool's own columns in the order a row gives them. A
    missing value is an empty cell."""
    rows = pooled.get("rows") or []
    keys = list(FIRST)
    for r in rows:
        keys.extend(k for k in r if k not in keys)
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=keys, extrasaction="ignore", lineterminator="\n")
    w.writeheader()
    for r in rows:
        w.writerow({k: "" if r.get(k) is None else r.get(k) for k in keys})
    return buf.getvalue()


def _filename(label: str, tool: str) -> str:
    return f"{re.sub(r'[^A-Za-z0-9]+', '-', label).strip('-') or 'pool'}-{tool}.csv"


def create_router(server) -> APIRouter:
    router = APIRouter()

    @router.get("/api/tools/{tool}/pool")
    async def pool_route(
        tool: str,
        project: str | None = None,
        phase: str | None = None,
        tag: str | None = None,
        format: str | None = None,
    ):
        """Every session the selector picks, the tool's rows pooled; as CSV
        with ``format=csv``, for a spreadsheet or a script."""
        bridge = getattr(server, "agent_bridge", None)
        agent = getattr(bridge, "agent", None) if bridge is not None else None
        store = (getattr(agent, "store", None) if agent is not None else None) or getattr(
            server, "gently_store", None
        )
        if store is None:
            raise HTTPException(status_code=503, detail="Store not available")
        try:
            out = pool(
                store,
                getattr(server, "context_store", None),
                tool,
                project=project,
                phase=phase,
                tag=tag,
            )
        except KeyError as e:
            raise HTTPException(status_code=404, detail=str(e)) from e
        if format == "csv":
            return Response(
                pool_csv(out),
                media_type="text/csv; charset=utf-8",
                headers={
                    "Content-Disposition": f'attachment; filename="{_filename(out["label"], tool)}"'
                },
            )
        return out

    return router
