"""Operation Plan route.

Returns the agent-authored Operation Plan for a session.
The plan is stored in FileContextStore (``server.context_store``) keyed by
session_id.  ``session_id="current"`` resolves to the newest session via the
FileStore (``server.gently_store``).
"""

from fastapi import APIRouter, HTTPException


def create_router(server) -> APIRouter:
    router = APIRouter()

    def _resolve_session(session_id: str) -> str:
        if session_id == "current":
            store = getattr(server, "gently_store", None)
            if store is None:
                raise HTTPException(
                    status_code=503, detail="FileStore not configured on viz server"
                )
            sessions = store.list_sessions()
            if not sessions:
                raise HTTPException(status_code=404, detail="No sessions in store")
            session_id = sessions[0].get("session_id")
        return session_id

    @router.get("/api/operation_plan/{session_id}")
    async def get_operation_plan(session_id: str):
        real_id = _resolve_session(session_id)
        cs = getattr(server, "context_store", None)
        if cs is None:
            return {"session_id": real_id, "available": False, "plan": None}
        try:
            plan = cs.get_operation_plan(real_id)
        except Exception:
            plan = None
        if plan is None:
            return {"session_id": real_id, "available": False, "plan": None}
        return {"session_id": real_id, "available": True, "plan": plan}

    return router
