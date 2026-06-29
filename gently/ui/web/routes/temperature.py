"""Read-only temperature history for the live graph (backfill on mount/reload).

Live updates ride the TEMPERATURE_UPDATE event channel; this route is backfill only.
Mirrors routes/experiments.py session resolution.
"""

import urllib.parse

from fastapi import APIRouter, HTTPException, Request


def create_router(server) -> APIRouter:
    router = APIRouter()

    def _resolve_session(session_id: str):
        store = getattr(server, "gently_store", None)
        if store is None:
            raise HTTPException(status_code=503, detail="FileStore not configured on viz server")
        if session_id == "current":
            sessions = store.list_sessions()
            if not sessions:
                raise HTTPException(status_code=404, detail="No sessions in store")
            session_id = sessions[0].get("session_id")
        if store._session_dir(session_id) is None:
            raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")
        return session_id

    @router.get("/api/temperature/{session_id}/history")
    async def get_history(session_id: str, request: Request):
        # Parse `since` from raw query string using unquote (not unquote_plus) so that
        # timezone offsets like +00:00 are preserved. Standard FastAPI query-param
        # parsing applies unquote_plus, which converts + to a space.
        raw_qs = request.scope.get("query_string", b"").decode()
        since = None
        for part in raw_qs.split("&"):
            if "=" in part:
                k, v = part.split("=", 1)
                if urllib.parse.unquote(k) == "since":
                    since = urllib.parse.unquote(v)
                    break

        real_id = _resolve_session(session_id)
        store = server.gently_store
        samples = store.read_temperature_log(real_id, since=since)
        return {"session_id": real_id, "samples": samples}

    return router
