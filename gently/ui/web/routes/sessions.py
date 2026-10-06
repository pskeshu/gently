"""Session routes - list, retrieve, and resume saved sessions."""

import asyncio
import logging
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse

from gently.ui.web.auth import require_control

logger = logging.getLogger(__name__)


def _open_in_file_manager(path: Path) -> None:
    """Show ``path`` in the OS file manager. Kept for the routes that were
    here first; ``gently/core/reveal.py`` is where it is done."""
    from gently.core import reveal

    reveal.open_folder(path)


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

        def gather() -> list[dict]:
            out = []
            for s in store.list_sessions():
                sid = s.get("session_id")
                # Directory names and one checkpoint per session, no image
                # decoded: what there is before a restore, in the list itself.
                held = _what_a_session_holds(store, sid) or {}
                out.append(
                    {
                        "session_id": sid,
                        "name": s.get("name") or sid,
                        "created_at": s.get("created_at", ""),
                        "last_active": s.get("last_active", ""),
                        "embryo_count": held.get("embryo_count", 0),
                        "timepoints": held.get("timepoints", 0),
                        "last_image_at": held.get("last_image_at"),
                        "run": held.get("run"),
                        "last_run": held.get("last_run"),
                        "dic_frames": _dic_count(store, sid),
                        "description": s.get("description", ""),
                        "active": sid == active_id,
                        "advanced_diagnostics": bool(
                            (s.get("metadata") or {}).get("advanced_diagnostics")
                        ),
                    }
                )
            return out

        try:
            sessions = await asyncio.to_thread(gather)
        except Exception as e:
            logger.warning("Failed to list sessions from FileStore: %s", e)
            sessions = []
        return {"sessions": sessions}

    def _dic_count(store, sid: str) -> int:
        try:
            return len(store.list_snapshots(sid, "dic") or [])
        except Exception:
            return 0

    def _checkpoint(store, sid: str) -> dict:
        """The session's timelapse.yaml, or {}."""
        try:
            folder = store._session_dir(sid)
            path = Path(folder) / "timelapse.yaml" if folder is not None else None
            if path is not None and path.is_file():
                import yaml

                doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
                return doc if isinstance(doc, dict) else {}
        except Exception:
            logger.debug("checkpoint of %s could not be read", sid, exc_info=True)
        return {}

    def _what_a_session_holds(store, sid: str) -> dict | None:
        """What there is to carry on with in a session, read from its folder:
        directory names and one small checkpoint, no image decoded. None for
        a session with no embryo in it."""
        try:
            embryo_ids = store.list_embryo_ids(sid)
        except Exception:
            embryo_ids = []
        if not embryo_ids:
            return None
        timepoints = 0
        last_image: float | None = None
        for eid in embryo_ids:
            try:
                tps = store.list_projection_timepoints(sid, eid) or []
            except Exception:
                tps = []
            if not tps:
                continue
            timepoints += len(tps)
            try:
                newest = store.get_projection_path(sid, eid, max(tps))
                if newest is not None:
                    taken = newest.stat().st_mtime
                    last_image = taken if last_image is None else max(last_image, taken)
            except OSError:
                pass

        # ``run`` is only an interrupted run — the launch gate offers those to
        # carry on. ``last_run`` is whatever the checkpoint says, for the
        # Sessions tab to describe a session before it is restored.
        run = None
        last_run = None
        state = _checkpoint(store, sid)
        if state:
            rows = state.get("embryos") or {}
            going = [k for k, v in rows.items() if not (v or {}).get("is_complete")]
            last_run = {
                "status": state.get("status"),
                "embryos_going": len(going),
                "saved_at": state.get("saved_at"),
                "started_at": state.get("started_at"),
                "rounds": state.get("current_round"),
                "total_timepoints": state.get("total_timepoints"),
                "interval_seconds": state.get("base_interval_seconds"),
            }
            if state.get("status") in ("running", "paused") and going:
                run = dict(last_run, status="interrupted")

        info = store.get_session(sid) or {}
        return {
            "session_id": sid,
            "name": info.get("name") or sid,
            "created_at": info.get("created_at", ""),
            "embryo_count": len(embryo_ids),
            "timepoints": timepoints,
            "last_image_at": (
                datetime.fromtimestamp(last_image).isoformat(timespec="seconds")
                if last_image is not None
                else None
            ),
            "run": run,
            "last_run": last_run,
        }

    @router.get("/api/sessions/resumable")
    async def resumable_sessions(limit: int = 3, scan: int = 40):
        """The latest sessions there is something to carry on with, newest
        first, for the launch gate. A session with no embryo is not offered:
        a rig accrues many that were opened and closed.

        Declared before ``/api/sessions/{session_id}``, which would take
        "resumable" for an id.
        """
        store = _file_store()
        if store is None:
            return {"sessions": [], "active": None}
        limit = max(1, min(int(limit), 8))
        scan = max(1, min(int(scan), 200))
        active = _active_session_id()

        def gather() -> list[dict]:
            out: list[dict] = []
            for sid in store.recent_session_ids(scan) or []:
                held = _what_a_session_holds(store, sid)
                if held is None:
                    continue
                held["active"] = sid == active
                out.append(held)
                if len(out) >= limit:
                    break
            return out

        try:
            sessions = await asyncio.to_thread(gather)
        except Exception as e:
            logger.warning("Failed to list resumable sessions: %s", e)
            sessions = []
        return {"sessions": sessions, "active": active}

    @router.get("/api/home/recent-images")
    async def recent_images(limit: int = 8, scan: int = 200, sessions: int = 0):
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

        `sessions` stops the walk after that many sessions have contributed,
        so Home can show the last few sessions whole, embryo by embryo,
        instead of the first eight images wherever they fall. Each image
        says which session it is from and when that session was, and how
        many timepoints its embryo has.
        """
        store = _file_store()
        if store is None:
            return {"images": []}
        limit = max(1, min(int(limit), 48))
        scan = max(1, min(int(scan), 500))
        sessions = max(0, min(int(sessions), 12))
        contributed = 0
        out = []
        try:
            for sid in store.recent_session_ids(scan) or []:
                try:
                    eids = store.list_embryo_ids(sid)
                except Exception:
                    eids = []
                sname = None  # parsed lazily, only if this session contributes
                screated = None
                if sessions and contributed >= sessions:
                    break
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
                        screated = (info.get("created_at") if info else None) or None
                        contributed += 1
                    out.append(
                        {
                            "session_id": sid,
                            "session_name": sname,
                            "session_created_at": screated,
                            "embryo_id": eid,
                            "timepoint": int(max(tps)),
                            "timepoints": len(tps),
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

    @router.post("/api/sessions/{session_id}/open-folder", dependencies=[Depends(require_control)])
    async def open_session_folder(session_id: str):
        """Open the session's folder — where every image it took lives — in
        the OS file manager. The folder is the store's own for that session;
        no path comes from the client."""
        store = _file_store()
        if store is None:
            raise HTTPException(status_code=503, detail="No session store")
        try:
            folder = store._session_dir(session_id)
        except Exception:
            folder = None
        if folder is None or not Path(folder).is_dir():
            raise HTTPException(status_code=404, detail="Session folder not found")
        try:
            _open_in_file_manager(Path(folder))
        except Exception as exc:
            logger.warning("could not open %s in the file manager: %s", folder, exc)
            raise HTTPException(status_code=502, detail=f"could not open folder: {exc}") from exc
        return {"opened": True, "session_id": session_id, "path": str(folder)}

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
        held = await asyncio.to_thread(_what_a_session_holds, store, session_id)
        embryos, dic_frames = await asyncio.to_thread(_what_to_look_at, store, session_id)
        return {
            "session_id": session_id,
            "name": info.get("name") or session_id,
            "description": info.get("description", ""),
            "created_at": info.get("created_at", ""),
            "last_active": info.get("last_active", ""),
            "run": (held or {}).get("run"),
            "last_run": (held or {}).get("last_run"),
            "acquisition": store.get_acquisition_plan(session_id),
            "embryos": embryos,
            "dic_frames": dic_frames,
            "embryo_states": experiment.get("embryos", {}) or {},
            "conversation": snapshot.get("conversation_history", []) or [],
            "detection_history": {},
        }

    def _what_to_look_at(store, sid: str) -> tuple[list[dict], list[dict]]:
        """Per embryo: its latest projection and last predicted stage. Per
        DIC frame: where to fetch it. Paths never leave the server; the
        client gets URLs the routes below resolve through the store."""
        embryos: list[dict] = []
        states = ((store.load_session_snapshot(sid) or {}).get("experiment_data", {}) or {}).get(
            "embryos", {}
        ) or {}
        for e in store.list_embryos(sid) or []:
            eid = e.get("embryo_id")
            try:
                tps = store.list_projection_timepoints(sid, eid) or []
            except Exception:
                tps = []
            latest = max(tps) if tps else None
            try:
                preds = store.get_predictions(sid, eid) or []
            except Exception:
                preds = []
            last = preds[-1] if preds else None
            st = states.get(eid) or {}
            pos = e.get("position_coarse") or (
                {"x": e.get("position_x"), "y": e.get("position_y")}
                if e.get("position_x") is not None
                else None
            )
            embryos.append(
                {
                    "embryo_id": eid,
                    "nickname": e.get("nickname"),
                    "role": e.get("role"),
                    "strain": e.get("strain"),
                    "position": pos,
                    "timepoints": len(tps),
                    "latest_timepoint": latest,
                    "thumbnail": (
                        f"/api/sessions/{sid}/projection?embryo={eid}&t={latest}"
                        if latest is not None
                        else None
                    ),
                    "stage": (last or {}).get("predicted_stage") or st.get("current_stage"),
                    "stage_confidence": (last or {}).get("confidence"),
                    "is_complete": bool(st.get("is_complete")),
                    "predictions": [
                        {
                            "timepoint": p.get("timepoint"),
                            "stage": p.get("predicted_stage"),
                            "confidence": p.get("confidence"),
                        }
                        for p in preds
                    ],
                }
            )
        frames: list[dict] = []
        try:
            recs = store.list_snapshots(sid, "dic") or []
        except Exception:
            recs = []
        for rec in recs:
            fp = rec.get("file_path")
            if not fp:
                continue
            stem = Path(fp).stem
            meta = rec.get("metadata") or {}
            frames.append(
                {
                    "stem": stem,
                    "frame": meta.get("frame"),
                    "captured_at": meta.get("captured_at") or rec.get("captured_at"),
                    "url": f"/api/sessions/{sid}/snapshot/{stem}.png",
                }
            )
        return embryos, frames

    @router.get("/api/sessions/{session_id}/snapshot/{stem}.png")
    async def session_snapshot_png(session_id: str, stem: str, max: int | None = None):
        """One of the session's filed snapshots (DIC overview) as a PNG,
        ``?max=N`` for a thumbnail. Found through the store's own listing,
        never from a path in the request."""
        from gently.ui.web.routes.dic import tiff_png_response

        store = _file_store()
        if store is None:
            raise HTTPException(status_code=503, detail="Store not available")
        try:
            recs = store.list_snapshots(session_id) or []
        except Exception:
            recs = []
        rec = next((r for r in recs if Path(r.get("file_path") or "").stem == stem), None)
        if rec is None:
            raise HTTPException(status_code=404, detail=f"no snapshot {stem!r} in this session")
        return tiff_png_response(Path(rec["file_path"]), stem, max)

    return router
