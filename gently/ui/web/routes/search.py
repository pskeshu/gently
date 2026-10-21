"""One search for the operator and the agent.

``GET /api/search/{source}?q=`` answers for one source at a time, so the add
popover can fire the sources in parallel and draw each group when it lands.
The same dispatcher serves the agent's ``search_references`` tool.
"""

from fastapi import APIRouter, HTTPException

from gently.harness.plan_mode.tools.research import SEARCH_SOURCES, search


def create_router(server) -> APIRouter:
    router = APIRouter()

    @router.get("/api/search/sources")
    async def sources():
        return {"sources": list(SEARCH_SOURCES)}

    @router.get("/api/search/{source}")
    async def search_one(source: str, q: str = "", limit: int = 8):
        if source not in SEARCH_SOURCES:
            raise HTTPException(status_code=404, detail=f"unknown source {source!r}")
        q = (q or "").strip()
        if len(q) < 2:
            return {"source": source, "status": "empty", "took_ms": 0, "results": [], "error": None}
        return await search(
            source,
            q,
            cs=getattr(server, "context_store", None),
            store=getattr(server, "gently_store", None),
            limit=max(1, min(limit, 25)),
        )

    return router
