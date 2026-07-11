"""Tactic Library route.

Returns the saved tactic library from FileContextStore (``server.context_store``).
Never raises a 500 — returns an empty list when the store is absent or has no
saved tactics.
"""

from fastapi import APIRouter


def create_router(server) -> APIRouter:
    router = APIRouter()

    @router.get("/api/tactic_library")
    async def get_tactic_library():
        cs = getattr(server, "context_store", None)
        if cs is None:
            return {"tactics": []}
        try:
            tactics = cs.list_tactics()
        except Exception:
            tactics = []
        if not tactics:
            return {"tactics": []}
        return {"tactics": tactics}

    return router
