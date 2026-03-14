"""Session routes - list and retrieve saved sessions."""

import json
import logging

from fastapi import APIRouter, HTTPException

logger = logging.getLogger(__name__)


def create_router(server) -> APIRouter:
    router = APIRouter()

    @router.get("/api/sessions")
    async def list_sessions():
        """List available sessions with metadata"""
        sessions = []
        if server.sessions_dir.exists():
            for path in server.sessions_dir.glob("*.json"):
                try:
                    with open(path, encoding='utf-8') as f:
                        data = json.load(f)
                    sessions.append({
                        'session_id': data.get('session_id', path.stem),
                        'name': data.get('name', path.stem),
                        'created_at': data.get('created_at', ''),
                        'last_active': data.get('last_active', ''),
                        'embryo_count': len(data.get('embryo_states', {})),
                        'description': data.get('description', '')
                    })
                except Exception as e:
                    logger.warning(f"Failed to read session {path}: {e}")
        # Sort by created_at descending (newest first)
        sessions.sort(key=lambda x: x.get('created_at', ''), reverse=True)
        return {'sessions': sessions}

    @router.get("/api/sessions/{session_id}")
    async def get_session(session_id: str):
        """Get full session state for review"""
        path = server.sessions_dir / f"{session_id}.json"
        if not path.exists():
            raise HTTPException(status_code=404, detail="Session not found")
        try:
            with open(path, encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load session: {e}")

    return router
