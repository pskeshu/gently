"""Experiment routes - strategy snapshot for the Experiment overview tab."""

import logging

from fastapi import APIRouter, HTTPException

from ..strategy_snapshot import build_strategy_snapshot

logger = logging.getLogger(__name__)


def create_router(server) -> APIRouter:
    router = APIRouter()

    def _resolve_session(session_id: str):
        """Look up a session_id on the server's FileStore and return
        ``(real_session_id, session_dir)``. The literal ``"current"``
        resolves to the most-recently-touched session (the one the agent
        is most likely working on)."""
        store = getattr(server, "gently_store", None)
        if store is None:
            raise HTTPException(
                status_code=503,
                detail="FileStore not configured on viz server",
            )

        if session_id == "current":
            sessions = store.list_sessions()
            if not sessions:
                raise HTTPException(
                    status_code=404,
                    detail="No sessions in store",
                )
            session_id = sessions[0].get("session_id")
            if not session_id:
                raise HTTPException(
                    status_code=500,
                    detail="Most-recent session has no session_id",
                )

        session_dir = store._session_dir(session_id)
        if session_dir is None or not session_dir.exists():
            raise HTTPException(
                status_code=404,
                detail=f"Session not found: {session_id}",
            )
        return session_id, session_dir

    @router.get("/api/experiments/{session_id}/strategy")
    async def get_strategy(session_id: str):
        """Return the strategy snapshot for the named session.

        Pass ``session_id="current"`` to resolve to the most-recent
        session in the store.
        """
        real_id, session_dir = _resolve_session(session_id)
        try:
            return build_strategy_snapshot(session_dir, real_id)
        except Exception as e:
            logger.exception("Failed to build strategy snapshot for %s", real_id)
            raise HTTPException(
                status_code=500,
                detail=f"Failed to build strategy: {e}",
            ) from e

    return router
