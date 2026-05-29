"""Session routes - list, retrieve, and resume saved sessions."""

import logging

from fastapi import APIRouter, Depends, HTTPException

from gently.ui.web.auth import require_control

logger = logging.getLogger(__name__)


def create_router(server) -> APIRouter:
    router = APIRouter()

    def _file_store():
        """The live FileStore (current Gently3 layout), via the agent."""
        bridge = getattr(server, "agent_bridge", None)
        if bridge is not None and getattr(bridge, "agent", None) is not None:
            st = getattr(bridge.agent, "store", None)
            if st is not None:
                return st
        return getattr(server, "gently_store", None)

    def _active_session_id():
        bridge = getattr(server, "agent_bridge", None)
        agent = bridge.agent if bridge is not None else None
        return getattr(agent, "session_id", None) if agent is not None else None

    @router.get("/api/sessions")
    async def list_sessions():
        """List available sessions (from the live FileStore)."""
        store = _file_store()
        if store is None:
            return {"sessions": []}
        active_id = _active_session_id()
        sessions = []
        try:
            for s in store.list_sessions():
                sid = s.get("session_id")
                try:
                    count = len(store.list_embryos(sid) or [])
                except Exception:
                    count = 0
                sessions.append({
                    "session_id": sid,
                    "name": s.get("name") or sid,
                    "created_at": s.get("created_at", ""),
                    "last_active": s.get("last_active", ""),
                    "embryo_count": count,
                    "description": s.get("description", ""),
                    "active": sid == active_id,
                })
        except Exception as e:
            logger.warning("Failed to list sessions from FileStore: %s", e)
        return {"sessions": sessions}

    @router.post("/api/sessions/{session_id}/resume",
                 dependencies=[Depends(require_control)])
    async def resume_session(session_id: str):
        """Switch the live agent to a different saved session.

        Reuses the same machinery as CLI resume (saves the current session,
        loads the target's embryos + conversation). Then nudges all browser
        clients to reload so they pick up the new session's state and
        transcript.
        """
        bridge = getattr(server, "agent_bridge", None)
        agent = bridge.agent if bridge is not None else None
        if agent is None:
            raise HTTPException(status_code=503, detail="Agent not ready")
        store = getattr(agent, "store", None)
        if store is None or store.get_session(session_id) is None:
            raise HTTPException(status_code=404, detail="Session not found")
        if session_id == getattr(agent, "session_id", None):
            return {"ok": True, "session_id": session_id, "active": True,
                    "note": "already active"}
        try:
            ok = agent.resume_session(session_id)
        except Exception as e:
            logger.exception("Session resume failed")
            raise HTTPException(status_code=500, detail=f"resume failed: {e}")
        if not ok:
            raise HTTPException(status_code=500, detail="resume returned false")
        # Rehydrate the viz image store from disk so the resumed session's
        # projections/filmstrips show (pixels load lazily from the FileStore).
        rehydrated = 0
        try:
            rehydrated = server.rehydrate_session(session_id)
        except Exception:
            logger.exception("rehydrate_session failed")
        # Tell every connected browser to reload — they'll reconnect to the
        # new session's state (embryos, transcript, rehydrated imagery).
        try:
            await server.manager.broadcast({"type": "session_changed",
                                            "session_id": session_id})
        except Exception:
            pass
        return {"ok": True, "session_id": session_id, "active": True,
                "rehydrated_projections": rehydrated}

    @router.get("/api/sessions/{session_id}")
    async def get_session(session_id: str):
        """Get session state for review, from the live FileStore.

        Maps the FileStore session snapshot onto the shape the Sessions review
        view expects (embryo_states / conversation). detection_history isn't
        reconstructed here (per-timepoint predictions live elsewhere).
        """
        store = _file_store()
        if store is None:
            raise HTTPException(status_code=503, detail="Store not available")
        info = store.get_session(session_id)
        if info is None:
            raise HTTPException(status_code=404, detail="Session not found")
        snapshot = store.load_session_snapshot(session_id) or {}
        experiment = snapshot.get("experiment_data", {}) or {}
        return {
            "session_id": session_id,
            "name": info.get("name") or session_id,
            "description": info.get("description", ""),
            "created_at": info.get("created_at", ""),
            "last_active": info.get("last_active", ""),
            "embryo_states": experiment.get("embryos", {}) or {},
            "conversation": snapshot.get("conversation_history", []) or [],
            "detection_history": {},
        }

    return router
