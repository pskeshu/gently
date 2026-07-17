"""Session routes - list, retrieve, and resume saved sessions."""

import logging
import time
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

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
                sessions.append(
                    {
                        "session_id": sid,
                        "name": s.get("name") or sid,
                        "created_at": s.get("created_at", ""),
                        "last_active": s.get("last_active", ""),
                        "embryo_count": count,
                        "description": s.get("description", ""),
                        "active": sid == active_id,
                    }
                )
        except Exception as e:
            logger.warning("Failed to list sessions from FileStore: %s", e)
        return {"sessions": sessions}

    @router.get("/api/home/recent-images")
    async def recent_images(limit: int = 8, scan: int = 200):
        """Latest projection per embryo, aggregated across recent sessions.

        Unlike /api/snapshots (in-memory, current session only), this walks the
        FileStore on disk so the home page can show imagery from *previous*
        sessions. Cheap by construction: recent session IDs come from folder
        names (no session.yaml parse), embryo IDs from directory names (no
        embryo.yaml parse), timepoints from a filename glob (no pixel decode),
        and the walk stops as soon as `limit` images are collected.

        `scan` is the *budget* of most-recent sessions to walk while hunting for
        images, NOT a hard window — empty/aborted sessions (common at the head:
        a rig accrues many no-capture sessions) are skipped nearly for free
        (one iterdir each), so the default is generous enough to reach older
        sessions that actually hold projections. Both bounds are clamped so a
        crafted ?scan=/?limit= can't turn this unauthenticated read into an
        unbounded scan. Returns components; the client builds the (encoded) URL.
        """
        store = _file_store()
        if store is None:
            return {"images": []}
        limit = max(1, min(int(limit), 48))
        scan = max(1, min(int(scan), 500))
        out = []
        try:
            for sid in store.recent_session_ids(scan) or []:
                try:
                    eids = store.list_embryo_ids(sid)
                except Exception:
                    eids = []
                sname = None  # parsed lazily, only if this session contributes
                for eid in eids:
                    try:
                        tps = store.list_projection_timepoints(sid, eid) or []
                    except Exception:
                        tps = []
                    if not tps:
                        continue
                    if sname is None:
                        try:
                            info = store.get_session(sid)
                        except Exception:
                            info = None
                        sname = (info.get("name") if info else None) or sid
                    out.append(
                        {
                            "session_id": sid,
                            "session_name": sname,
                            "embryo_id": eid,
                            "timepoint": int(max(tps)),
                        }
                    )
                    if len(out) >= limit:
                        break
                if len(out) >= limit:
                    break
        except Exception as e:
            logger.warning("recent_images failed: %s", e)
        return {"images": out[:limit]}

    @router.get("/api/sessions/{session_id}/projection")
    async def get_session_projection(session_id: str, embryo: str, t: int):
        """Serve a saved JPEG projection from any session on disk.

        Path-traversal safe: the resolved file must live inside the session's
        own directory, so a crafted `embryo` (e.g. '../..') can't escape.
        """
        store = _file_store()
        if store is None:
            raise HTTPException(status_code=503, detail="Store not available")
        path = store.get_projection_path(session_id, embryo, t)
        if path is None:
            raise HTTPException(status_code=404, detail="Projection not found")
        try:
            sd = store._session_dir(session_id)
            resolved = Path(path).resolve()
            # Component-wise ancestor check (not str.startswith, which would
            # let a sibling like `<sd>_evil` slip through the prefix match).
            sd_resolved = Path(sd).resolve() if sd is not None else None
            if sd_resolved is None or sd_resolved not in resolved.parents:
                raise HTTPException(status_code=404, detail="Not found")
        except HTTPException:
            raise
        except Exception:
            raise HTTPException(status_code=404, detail="Not found") from None
        try:
            st = resolved.stat()
            etag = f'"{int(st.st_mtime)}-{st.st_size}"'
        except OSError:
            etag = None
        headers = {"Cache-Control": "private, max-age=60"}
        if etag:
            headers["ETag"] = etag
        return FileResponse(str(resolved), media_type="image/jpeg", headers=headers)

    @router.post("/api/sessions/{session_id}/resume", dependencies=[Depends(require_control)])
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
            return {
                "ok": True,
                "session_id": session_id,
                "active": True,
                "note": "already active",
            }
        try:
            ok = agent.resume_session(session_id)
        except Exception as e:
            logger.exception("Session resume failed")
            raise HTTPException(status_code=500, detail=f"resume failed: {e}") from e
        if not ok:
            raise HTTPException(status_code=500, detail="resume returned false")
        # Rehydrate the viz image store from disk so the resumed session's
        # projections/filmstrips show (pixels load lazily from the FileStore).
        rehydrated = 0
        try:
            rehydrated = server.rehydrate_session(session_id)
        except Exception:
            logger.exception("rehydrate_session failed")
        # Resuming is an in-app action — the operator is already past the entry
        # gate, so never bounce them back to /launch to re-answer hardware /
        # assistant. gate_passed is in-memory and resets on any backend restart,
        # which is exactly what made a resume-to-view land on the launch gate.
        server.gate_passed = True
        # Also skip the "what are we doing today?" landing overlay on the reload
        # below — resuming existing work isn't starting fresh. A timestamp (not a
        # one-shot bool) so EVERY client the broadcast reloads skips the landing,
        # not just whichever one hits the index route first.
        server._resumed_at = time.monotonic()
        # Tell every connected browser to reload — they'll reconnect to the
        # new session's state (embryos, transcript, rehydrated imagery).
        try:
            await server.manager.broadcast({"type": "session_changed", "session_id": session_id})
        except Exception:
            pass
        return {
            "ok": True,
            "session_id": session_id,
            "active": True,
            "rehydrated_projections": rehydrated,
        }

    @router.get("/api/sessions/{session_id}/plans")
    async def get_session_plans(session_id: str):
        """Plan items linked to a session, via the context store."""
        cs = getattr(server, "context_store", None)
        if cs is None:
            return {"plans": []}
        try:
            items = cs.get_plan_items_for_session(session_id)
        except Exception as e:
            logger.warning("get_plan_items_for_session failed for %s: %s", session_id, e)
            return {"plans": []}
        return {
            "plans": [
                {
                    "id": item.id,
                    "title": item.title,
                    "campaign_id": item.campaign_id,
                    "status": item.status.value,
                }
                for item in items
            ]
        }

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
