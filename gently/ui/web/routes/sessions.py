"""Session routes - list, retrieve, and resume saved sessions."""

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Request, Response
from fastapi.responses import FileResponse

from gently.core.session_summary import interaction_stats, signal_counts
from gently.core.session_summary import read_checkpoint as _checkpoint
from gently.core.session_summary import what_a_session_holds as _what_a_session_holds
from gently.ui.web.auth import require_control

logger = logging.getLogger(__name__)


def _open_in_file_manager(path: Path) -> None:
    """Show ``path`` in the OS file manager. Kept for the routes that were
    here first; ``gently/core/reveal.py`` is where it is done."""
    from gently.core import reveal

    reveal.open_folder(path)


def _fmt_interval(sec) -> str:
    try:
        s = float(sec)
    except (TypeError, ValueError):
        return ""
    if s <= 0:
        return ""
    if s < 60:
        return f"{s:g} s"
    if s < 3600:
        return f"{round(s / 60)} min"
    h = s / 3600
    return f"{h:g} h" if h == int(h) else f"{h:.1f} h"


def derive_session_name(
    *,
    created_at: str | None,
    embryo_count: int = 0,
    last_run: dict | None = None,
    acquisition: dict | None = None,
    embryos: list[dict] | None = None,
) -> str:
    """A name from the facts: count, cadence, when, how it ended. What an
    unnamed session is called in the list, and what the model's suggestion
    falls back to. 'Oct 4 overnight' says more than a hex id."""
    parts: list[str] = []
    if embryo_count:
        parts.append(f"{embryo_count} embryo{'s' if embryo_count != 1 else ''}")
    interval = (acquisition or {}).get("interval_seconds") or (last_run or {}).get(
        "interval_seconds"
    )
    if interval:
        parts.append(f"every {_fmt_interval(interval)}")
    when = ""
    started = (last_run or {}).get("started_at") or created_at
    try:
        dt = datetime.fromisoformat(str(started)) if started else None
    except ValueError:
        dt = None
    if dt is not None:
        when = f"{dt.strftime('%b')} {dt.day}"  # not %-d: the rig runs Windows
        if dt.hour >= 17 or dt.hour < 5:
            when += " overnight"
    if when:
        parts.append(when)
    outcome = ""
    status = str((last_run or {}).get("status") or "")
    going = (last_run or {}).get("embryos_going")
    if embryos and all(e.get("is_complete") for e in embryos):
        outcome = "all complete"
    elif status in ("running", "paused") and going:
        outcome = "cut short"
    elif status in ("completed", "complete") or (status and going == 0):
        outcome = "complete"
    elif status == "stopped":
        outcome = "stopped"
    if outcome:
        parts.append(outcome)
    return " · ".join(parts) if parts else "Unnamed session"


def _first_user_words(history: list[dict], limit: int = 600) -> str:
    """What the operator asked for, in their words, for the model to name
    the session by. Text blocks only; tool traffic is not a request."""
    out: list[str] = []
    for m in history:
        if m.get("role") != "user":
            continue
        c = m.get("content")
        if isinstance(c, str):
            out.append(c)
        elif isinstance(c, list):
            out.extend(
                b.get("text", "") for b in c if isinstance(b, dict) and b.get("type") == "text"
            )
        if sum(len(x) for x in out) >= limit:
            break
    return " / ".join(x.strip() for x in out if x.strip())[:limit]


def _ask_model_for_a_name(client, summary: dict) -> tuple[str, str] | None:
    """One short call on the project's fast model: a name and a line for the
    session. None on anything but a clean JSON answer; the caller falls back
    to the derived name."""
    import json

    from gently.settings import settings

    response = client.with_options(timeout=20.0).messages.create(
        model=settings.models.fast,
        max_tokens=200,
        system=(
            "You name light-sheet microscopy sessions for the biologist who ran them. "
            'Reply with JSON only, no prose: {"name": ..., "description": ...}. '
            "The name is at most six words, specific to what was imaged and why, with no date "
            "(the interface shows it) and without the word 'session'. The description is one "
            "sentence of at most 25 words: what was imaged, how, and how it ended."
        ),
        messages=[{"role": "user", "content": json.dumps(summary, default=str)}],
    )
    text = "".join(b.text for b in response.content if getattr(b, "type", "") == "text").strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text[text.find("{") :] if "{" in text else text
    try:
        doc = json.loads(text[text.find("{") : text.rfind("}") + 1])
    except (ValueError, TypeError):
        return None
    name = str(doc.get("name") or "").strip()
    desc = str(doc.get("description") or "").strip()
    if not name:
        return None
    return name[:120], desc[:600]


_EXPORTS: dict[str, dict] = {}
# The movie being made, or the last one: one at a time, from any folder.
_MOVIES: dict = {"state": "idle"}
_SIZE_CACHE: dict[str, tuple[float, int]] = {}
_SIZE_TTL_S = 900.0


def folder_bytes(
    folder: Path | None, *, fresh: bool = False, cached_only: bool = False
) -> int | None:
    """What a session folder occupies on disk, summed over every file.
    Cached for fifteen minutes per folder unless ``fresh``: a walk reads
    metadata only, but the list asks for every session at once.

    ``cached_only`` answers from the cache, stale or not, and None when the
    folder has never been walked: the list says what it knows at once and
    the tab asks ``/api/sessions/sizes`` for the rest."""
    if folder is None:
        return None
    import os
    import time

    key = str(folder)
    now = time.monotonic()
    hit = _SIZE_CACHE.get(key)
    if hit and not fresh and (cached_only or now - hit[0] < _SIZE_TTL_S):
        return hit[1]
    if cached_only and not fresh:
        return None
    total = 0
    try:
        for root, _dirs, files in os.walk(folder):
            for name in files:
                try:
                    total += os.stat(os.path.join(root, name)).st_size
                except OSError:
                    continue
    except OSError:
        return None
    _SIZE_CACHE[key] = (now, total)
    return total


def create_router(server) -> APIRouter:
    router = APIRouter()

    def _session_tools(store, sid: str) -> list[str]:
        from gently.tools import session_tools

        try:
            return session_tools(store, sid)
        except Exception:
            return []

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

    def _plan_items_by_session(cs) -> dict[str, list[dict]]:
        """session id -> the plan items that ran it, from one pass over the plans."""
        out: dict[str, list[dict]] = {}
        if cs is None:
            return out
        try:
            items = cs.get_plan_items()
        except Exception:
            return out
        roots: dict[str, str | None] = {}
        for it in items:
            if it.campaign_id not in roots:
                try:
                    roots[it.campaign_id] = cs.root_campaign_id(it.campaign_id)
                except Exception:
                    roots[it.campaign_id] = None
            for sid in it.session_ids or []:
                out.setdefault(sid, []).append(
                    {
                        "id": it.id,
                        "title": it.title,
                        "campaign_id": it.campaign_id,
                        "root_id": roots[it.campaign_id],
                        "status": it.status.value,
                        "origin": getattr(it, "origin", "agent"),
                    }
                )
        return out

    def _stage_ontology() -> dict:
        """Stage name -> WormBase life-stage id, from the organism plugin."""
        try:
            from gently.organisms import get_organism

            org = get_organism()
            mod = getattr(org, "stages", None) or __import__(
                f"{org.__name__}.stages", fromlist=["WBLS"]
            )
            return dict(getattr(mod, "WBLS", {}) or {})
        except Exception:
            return {}

    @router.get("/api/sessions")
    async def list_sessions(
        project_id: str | None = None, unfiled: bool = False, detail: bool | None = None
    ):
        """List available sessions (from the live FileStore).

        ``project_id`` narrows to one project's sessions and ``unfiled`` to
        those in none. Either adds what the project view shows: errors and
        warnings from the event capture, the agent's turns and corrections;
        ``detail=0`` leaves those out for a caller that only counts rows.
        Every row carries its project and the plan items it served.
        """
        store = _file_store()
        if store is None:
            return {"sessions": []}
        active_id = _active_session_id()
        cs = getattr(server, "context_store", None)
        detail = (project_id is not None or unfiled) if detail is None else detail

        def gather() -> list[dict]:
            out = []
            served = _plan_items_by_session(cs)
            for s in store.list_sessions(project_id=project_id, unfiled=unfiled):
                sid = s.get("session_id")
                # Directory names and one checkpoint per session, no image
                # decoded: what there is before a restore, in the list itself.
                held = _what_a_session_holds(store, sid) or {}
                extra: dict = {}
                if detail:
                    folder = store._session_dir(sid)
                    if folder is not None:
                        extra.update(signal_counts(Path(folder) / "events.jsonl"))
                        extra.update(interaction_stats(Path(folder) / "interaction_log.jsonl"))
                out.append(
                    {
                        "session_id": sid,
                        "project_id": s.get("project_id"),
                        "projects": s.get("projects") or [],
                        "operators": [str(x) for x in (s.get("operators") or [])],
                        "tags": s.get("tags") or [],
                        "plan_items": served.get(sid, []),
                        **extra,
                        "name": s.get("name") or sid,
                        "created_at": s.get("created_at", ""),
                        "last_active": s.get("last_active", ""),
                        "embryo_count": held.get("embryo_count", 0),
                        "timepoints": held.get("timepoints", 0),
                        "last_image_at": held.get("last_image_at"),
                        "run": held.get("run"),
                        "last_run": held.get("last_run"),
                        "bf_frames": _bf_count(store, sid),
                        # The walk over a session's files is the one slow read
                        # left here: a cached size comes now, the rest arrive
                        # through /api/sessions/sizes once the list is drawn.
                        # Never a walk from the list, the live session's included:
                        # its row shows the last size counted, the detail counts afresh.
                        "bytes": folder_bytes(store._session_dir(sid), cached_only=True),
                        "brightfield_references": _reference_count(store, sid),
                        "suggested_name": derive_session_name(
                            created_at=s.get("created_at"),
                            embryo_count=held.get("embryo_count", 0),
                            last_run=held.get("last_run"),
                        ),
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

    def _reference_count(store, sid: str) -> int:
        """Complete dark+flat reference sets the session holds."""
        try:
            from gently.app.brightfield import list_records

            return sum(1 for r in list_records(store, sid) if r.get("dark") and r.get("flat"))
        except Exception:
            return 0

    def _bf_count(store, sid: str) -> int:
        """By file name: parsing every sidecar of every session took a minute."""
        try:
            count = getattr(store, "count_snapshots", None)
            n = count(sid, "bf") if callable(count) else None
            if isinstance(n, int):
                return n
            return len(store.list_snapshots(sid, "bf") or [])
        except Exception:
            return 0

    @router.get("/api/sessions/sizes")
    async def session_sizes(ids: str = ""):
        """What the named sessions occupy on disk, walked now.

        The list leaves ``bytes`` null for a session whose size is not
        cached; the tab asks here in batches after it has drawn, so a rig
        with hundreds of sessions shows its list at once. Declared before
        ``/api/sessions/{session_id}``, which would take "sizes" for an id.
        """
        store = _file_store()
        wanted = [sid for sid in (ids or "").split(",") if sid][:64]
        if store is None or not wanted:
            return {"sizes": {}}
        active_id = _active_session_id()

        def gather() -> dict[str, int | None]:
            return {
                sid: folder_bytes(store._session_dir(sid), fresh=sid == active_id) for sid in wanted
            }

        return {"sizes": await asyncio.to_thread(gather)}

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
                info = store.get_session(sid) or {}
                held["project_id"] = info.get("project_id")
                held["operators"] = [str(x) for x in (info.get("operators") or [])]
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

    def _switch(agent, session_id: str | None) -> dict:
        """Make a session live through the agent's one switching path. A
        live run is a 409: the operator stops it first, on purpose."""
        try:
            return agent.switch_session(session_id)
        except RuntimeError as e:
            raise HTTPException(status_code=409, detail=str(e)) from e
        except Exception as e:
            logger.exception("Session switch failed")
            raise HTTPException(status_code=500, detail=f"switch failed: {e}") from e

    @router.post("/api/sessions/new", dependencies=[Depends(require_control)])
    async def new_session(request: Request):
        """Open a fresh session from inside the app — until now that meant
        closing Gently and opening it again. The current session is saved;
        the embryo list, transcript and per-session logs start empty; a
        live run refuses (409). Every browser reloads onto it."""
        bridge = getattr(server, "agent_bridge", None)
        agent = bridge.agent if bridge is not None else None
        if agent is None:
            raise HTTPException(status_code=503, detail="Agent not ready")
        got = _switch(agent, None)
        sid = got.get("session_id")
        try:
            server.rehydrate_session(sid)
        except Exception:
            logger.debug("rehydrate of a fresh session failed", exc_info=True)
        _remember_operator(request, sid)
        server.gate_passed = True
        try:
            await server.manager.broadcast({"type": "session_changed", "session_id": sid})
        except Exception:
            pass
        return {"ok": True, "session_id": sid, "previous": got.get("previous")}

    def _remember_operator(request: Request, sid: str | None) -> None:
        """A session remembers who opened or resumed it."""
        from gently.ui.web.auth import current_username

        try:
            store = _file_store()
            who = current_username(request)
            if sid and store is not None and who:
                store.add_session_operator(sid, who)
        except Exception:
            logger.debug("could not stamp the operator on %s", sid, exc_info=True)

    @router.post("/api/sessions/{session_id}/resume", dependencies=[Depends(require_control)])
    async def resume_session(session_id: str, request: Request):
        """Make a saved session live: new images, the transcript and the
        stage targets belong to it from here on. Viewing a session needs
        none of this — the Sessions tab reads the folder. The current
        session is saved first; a live run refuses (409). Then every
        browser is told to reload onto the new session.
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
        got = _switch(agent, session_id)
        if not got.get("ok"):
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
        _remember_operator(request, session_id)
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

    @router.post("/api/sessions/{session_id}/project", dependencies=[Depends(require_control)])
    async def file_session(session_id: str, request: Request):
        """File a session under a project (``campaign_id``, a project or one of
        its phases, optionally with the ``plan_item_id`` it served), move it
        (``move: true``), or take it out of its project (``campaign_id: null``)."""
        body = await request.json()
        store = _file_store()
        cs = getattr(server, "context_store", None)
        if store is None or store.get_session(session_id) is None:
            raise HTTPException(status_code=404, detail="Session not found")
        from gently.app.filing import file_session_under, unfile_session
        from gently.ui.web.auth import current_username

        campaign_id = body.get("campaign_id")
        if not campaign_id:
            return {
                "ok": True,
                "unfiled": unfile_session(store, cs, session_id, body.get("from") or None),
                "project_id": (store.get_session(session_id) or {}).get("project_id"),
            }
        if cs is None:
            raise HTTPException(status_code=503, detail="Context store not available")
        try:
            result = file_session_under(
                store,
                cs,
                session_id,
                campaign_id,
                body.get("plan_item_id") or None,
                move=bool(body.get("move")),
                role=body.get("role") or None,
                by=current_username(request) or None,
            )
        except KeyError:
            raise HTTPException(status_code=404, detail="Campaign not found") from None
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e)) from None
        return {"ok": True, **result}

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
        view expects (embryo_states / conversation). Every read happens in one
        worker thread: a session's folder is many small files, and the event
        loop has other people's requests to answer.
        """
        store = _file_store()
        if store is None:
            raise HTTPException(status_code=503, detail="Store not available")
        cs = getattr(server, "context_store", None)
        active = _active_session_id()

        def gather() -> dict | None:
            info = store.get_session(session_id)
            if info is None:
                return None
            snapshot = store.load_session_snapshot(session_id) or {}
            experiment = snapshot.get("experiment_data", {}) or {}
            held = _what_a_session_holds(store, session_id) or {}
            embryos, bf_frames = _what_to_look_at(store, session_id, snapshot)
            acquisition = store.get_acquisition_plan(session_id)
            _add_dose(embryos, _checkpoint(store, session_id))
            return {
                "session_id": session_id,
                "project_id": info.get("project_id"),
                "projects": store.links_of(info),
                "plan_items": _plan_items_by_session(cs).get(session_id, []),
                "stage_ontology": _stage_ontology(),
                "name": info.get("name"),
                "suggested_name": derive_session_name(
                    created_at=info.get("created_at"),
                    embryo_count=len(embryos),
                    last_run=held.get("last_run"),
                    acquisition=acquisition,
                    embryos=embryos,
                ),
                "description": info.get("description", ""),
                "created_at": info.get("created_at", ""),
                "last_active": info.get("last_active", ""),
                "run": held.get("run"),
                "last_run": held.get("last_run"),
                # A fresh count only for the live session, whose folder is growing;
                # a finished session's size is what the cache last counted.
                "bytes": folder_bytes(store._session_dir(session_id), fresh=session_id == active),
                "acquisition": acquisition,
                "embryos": embryos,
                "bf_frames": bf_frames,
                "tools": _session_tools(store, session_id),
                "operators": [str(x) for x in (info.get("operators") or [])],
                "edited": info.get("edited"),
                "removed_embryos": _removed(store, session_id),
                "embryo_states": experiment.get("embryos", {}) or {},
                "conversation": snapshot.get("conversation_history", []) or [],
                "detection_history": {},
            }

        data = await asyncio.to_thread(gather)
        if data is None:
            raise HTTPException(status_code=404, detail="Session not found")
        return data

    def _what_to_look_at(
        store, sid: str, snapshot: dict | None = None
    ) -> tuple[list[dict], list[dict]]:
        """Per embryo: its latest projection and last predicted stage. Per
        brightfield frame: where to fetch it. Paths never leave the server; the
        client gets URLs the routes below resolve through the store."""
        embryos: list[dict] = []
        if snapshot is None:
            snapshot = store.load_session_snapshot(sid) or {}
        states = (snapshot.get("experiment_data", {}) or {}).get("embryos", {}) or {}
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
                    "projection_timepoints": tps,
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
        from gently.app.brightfield import overview_frames

        return embryos, [
            {**f, "url": f"/api/sessions/{sid}/snapshot/{f['stem']}.png"}
            for f in overview_frames(store, sid)
        ]

    def _add_dose(embryos: list[dict], state: dict) -> None:
        """Each embryo's light dose so far against its budget, from the
        checkpoint: the main reason not to carry an embryo on."""
        rows = state.get("embryos") or {}
        base = state.get("dose_budget_base_ms")
        for e in embryos:
            row = rows.get(e["embryo_id"]) or {}
            e["dose_ms"] = row.get("total_exposure_ms")
            budget = None
            if base:
                try:
                    from gently.harness.roles import get_role

                    budget = float(base) * float(
                        get_role(e.get("role") or "test").photodose_budget_multiplier
                    )
                except Exception:
                    budget = float(base)
            e["dose_budget_ms"] = budget

    def _removed(store, sid: str) -> list[dict]:
        """The embryos set aside, with why: what you ask before resuming."""
        try:
            return [
                {
                    "embryo_id": r.get("embryo_id"),
                    "nickname": r.get("nickname"),
                    "removed_at": r.get("removed_at"),
                    "reason": r.get("reason") or r.get("removed_by") or r.get("by"),
                    "timepoints": r.get("timepoints") or r.get("projections"),
                }
                for r in store.list_removed_embryos(sid) or []
            ]
        except Exception:
            return []

    @router.patch("/api/sessions/{session_id}", dependencies=[Depends(require_control)])
    async def rename_session(session_id: str, body: dict, request: Request):
        """Give a session a name and a line about it. Written to its
        session.yaml; the folder keeps its slug."""
        store = _file_store()
        if store is None:
            raise HTTPException(status_code=503, detail="Store not available")
        name = body.get("name")
        description = body.get("description")
        if name is not None and not isinstance(name, str):
            raise HTTPException(status_code=400, detail="name must be a string")
        if description is not None and not isinstance(description, str):
            raise HTTPException(status_code=400, detail="description must be a string")
        if name is not None and len(name) > 120:
            raise HTTPException(status_code=400, detail="name is too long (120 characters)")
        if description is not None and len(description) > 600:
            raise HTTPException(status_code=400, detail="description is too long (600 chars)")
        info = store.update_session(session_id, name=name, description=description)
        from gently.ui.web.auth import current_username

        store.note_session_edit(session_id, current_username(request), "name")
        if info is None:
            raise HTTPException(status_code=404, detail="Session not found")
        return {
            "session_id": session_id,
            "name": info.get("name"),
            "description": info.get("description"),
        }

    @router.post("/api/sessions/{session_id}/suggest-name", dependencies=[Depends(require_control)])
    async def suggest_session_name(session_id: str):
        """A name and a line for the session, from what its folder holds.
        The model writes it when the agent has an API key; otherwise the
        name is derived from the counts, the cadence, the date and the
        outcome. Nothing is written — the operator saves or edits it."""
        store = _file_store()
        if store is None:
            raise HTTPException(status_code=503, detail="Store not available")
        info = store.get_session(session_id)
        if info is None:
            raise HTTPException(status_code=404, detail="Session not found")
        held = await asyncio.to_thread(_what_a_session_holds, store, session_id) or {}
        embryos, _frames = await asyncio.to_thread(_what_to_look_at, store, session_id)
        acquisition = store.get_acquisition_plan(session_id)
        derived = derive_session_name(
            created_at=info.get("created_at"),
            embryo_count=len(embryos),
            last_run=held.get("last_run"),
            acquisition=acquisition,
            embryos=embryos,
        )
        bridge = getattr(server, "agent_bridge", None)
        agent = bridge.agent if bridge is not None else None
        client = getattr(agent, "claude", None) if getattr(agent, "api_enabled", False) else None
        if client is None:
            return {"name": derived, "description": None, "source": "derived"}
        snapshot = store.load_session_snapshot(session_id) or {}
        summary = {
            "created_at": info.get("created_at"),
            "embryos": [
                {
                    "id": e["embryo_id"],
                    "nickname": e.get("nickname"),
                    "role": e.get("role"),
                    "strain": e.get("strain"),
                    "timepoints": e.get("timepoints"),
                    "last_stage": e.get("stage"),
                    "complete": e.get("is_complete"),
                }
                for e in embryos
            ],
            "acquisition": acquisition,
            "run": held.get("last_run"),
            "bf_frames": _bf_count(store, session_id),
            "operator_said": _first_user_words(snapshot.get("conversation_history") or []),
        }
        try:
            got = await asyncio.to_thread(_ask_model_for_a_name, client, summary)
        except Exception:
            logger.info("name suggestion fell back to the derived name", exc_info=True)
            got = None
        if not got:
            return {"name": derived, "description": None, "source": "derived"}
        return {"name": got[0], "description": got[1], "source": "model"}

    # ---- export, for Fiji and for finding the experiment again -------------
    # One job per session at a time, in a thread; the page polls.

    @router.post("/api/sessions/{session_id}/export", dependencies=[Depends(require_control)])
    async def start_export(session_id: str, body: dict | None = None):
        """Begin exporting the session: one folder per embryo, files named so
        a sort is time order, the record beside them (see gently.core.export).
        ``{"dest": "<folder>"}`` puts it somewhere other than <root>/exports;
        ``{"crops": false}`` skips cutting each embryo out of the brightfield frames."""
        import threading

        from gently.core.export import export_session

        store = _file_store()
        if store is None:
            raise HTTPException(status_code=503, detail="Store not available")
        if store.get_session(session_id) is None:
            raise HTTPException(status_code=404, detail="Session not found")
        job = _EXPORTS.get(session_id)
        if job and job.get("state") == "running":
            return job
        dest = (body or {}).get("dest")
        crops = bool((body or {}).get("crops", True))
        dest_path = Path(str(dest)).expanduser() if dest else None
        if dest_path is not None and not dest_path.is_absolute():
            raise HTTPException(status_code=400, detail="dest must be an absolute folder path")
        job = {
            "session_id": session_id,
            "state": "running",
            "done": 0,
            "total": 0,
            "current": "",
            "path": None,
            "error": None,
            "started_at": datetime.now().isoformat(timespec="seconds"),
        }
        _EXPORTS[session_id] = job

        def progress(done: int, total: int, what: str) -> None:
            job["done"], job["total"], job["current"] = done, total, what

        def run() -> None:
            try:
                out = export_session(store, session_id, dest_path, progress, crops=crops)
                job["path"] = str(out)
                job["state"] = "done"
            except Exception as exc:
                logger.exception("export of %s failed", session_id)
                job["error"] = str(exc)
                job["state"] = "error"

        threading.Thread(target=run, name=f"export-{session_id}", daemon=True).start()
        return job

    @router.get("/api/sessions/{session_id}/export")
    async def export_status(session_id: str):
        """Where the export stands, and where it went when done."""
        store = _file_store()
        default = (
            str(Path(store.root) / "exports")
            if store is not None and getattr(store, "root", None) is not None
            else None
        )
        job = _EXPORTS.get(session_id) or {"session_id": session_id, "state": "idle"}
        return dict(job, default_dest=default)

    # ---- movies, from any folder ------------------------------------------
    # Two kinds with two behaviours. A brightfield movie is a folder of frames over
    # time, with the flat divided out where the references are beside them.
    # A SPIM movie is a folder of volumes: max projections over time, or
    # every slice stack by stack. The folder is the operator's own (an export
    # on a share, a session's volumes), so it is given, not looked up.

    @router.post("/api/movies", dependencies=[Depends(require_control)])
    async def start_movie(body: dict):
        """``{"kind": "bf", "folder": ..., "corrected": true}``,
        ``{"kind": "spim", "folder": ..., "view": "projection"|"slices"}``, or
        ``{"kind": "crops", "folder": ..., "session_id": ...}``: each embryo
        cut out of a folder of brightfield frames, where the session's Operate
        marking says the embryos are."""
        import threading

        from gently.core.export import (
            _bf_frames,
            _volume_files,
            bf_crops,
            bf_movie,
            find_embryo_boxes,
            marking_seeds,
            orient_seeds,
            spim_movie,
        )

        kind = str((body or {}).get("kind") or "")
        if kind not in ("bf", "spim", "crops"):
            raise HTTPException(status_code=400, detail="kind must be 'bf', 'spim' or 'crops'")
        if _MOVIES.get("state") == "running":
            raise HTTPException(status_code=409, detail="A movie is already being made")
        folder = (body or {}).get("folder")
        if not folder:
            raise HTTPException(status_code=400, detail="Which folder?")
        path = Path(str(folder).strip().strip('"')).expanduser()
        if not path.is_absolute():
            raise HTTPException(status_code=400, detail="folder must be an absolute path")
        if not path.is_dir():
            raise HTTPException(status_code=404, detail=f"Folder not found: {path}")

        steps: list[dict] = []
        seeds: dict = {}
        if kind == "crops":
            if not _bf_frames(path) and _bf_frames(path / "bf"):
                path = path / "bf"
            if not _bf_frames(path):
                raise HTTPException(
                    status_code=400,
                    detail="No brightfield frames (bf_f*.tif or bf.csv) in that folder",
                )
            sid = str((body or {}).get("session_id") or "")
            store = _file_store()
            if not sid or store is None or store.get_session(sid) is None:
                raise HTTPException(status_code=400, detail="Which session marked the embryos?")
            seeds = marking_seeds(store, sid, store.list_embryos(sid) or [])
            if not seeds:
                raise HTTPException(
                    status_code=409,
                    detail="This session has no embryo marking (Operate tab), so there is "
                    "nothing to say where the embryos are.",
                )
            steps.append({"crops": True})
        elif kind == "bf":
            if not _bf_frames(path) and _bf_frames(path / "bf"):
                path = path / "bf"
            if not _bf_frames(path):
                raise HTTPException(
                    status_code=400,
                    detail="No brightfield frames (bf_f*.tif or bf.csv) in that folder",
                )
            steps.append({"corrected": False})
            if bool((body or {}).get("corrected", True)):
                steps.append({"corrected": True})
        else:
            if not _volume_files(path) and _volume_files(path / "volumes"):
                path = path / "volumes"
            if not _volume_files(path):
                raise HTTPException(status_code=400, detail="No volumes (.tif) in that folder")
            view = str((body or {}).get("view") or "projection")
            if view not in ("projection", "slices"):
                raise HTTPException(status_code=400, detail="view must be 'projection' or 'slices'")
            steps.append({"view": view})

        job: dict[str, Any] = {
            "kind": kind,
            "folder": str(path),
            "state": "running",
            "done": 0,
            "total": 0,
            "current": "",
            "outputs": [],
            "note": None,
            "error": None,
            "started_at": datetime.now().isoformat(timespec="seconds"),
        }
        # The thread writes into the very dict the status route reads.
        _MOVIES.clear()
        _MOVIES.update(job)
        job = _MOVIES

        def run() -> None:
            try:
                base = 0
                for step in steps:

                    def progress(i: int, n: int, what: str) -> None:
                        job["total"] = n * len(steps)
                        job["done"] = base + i  # noqa: B023
                        job["current"] = what.split(" ")[0]

                    out: Path | None
                    if kind == "crops":
                        placed = orient_seeds(path, seeds)
                        out = bf_crops(path, find_embryo_boxes(path, placed), progress=progress)
                    elif kind == "bf":
                        out = bf_movie(path, progress=progress, corrected=step["corrected"])
                        if out is None and step["corrected"]:
                            job["note"] = (
                                "no dark and flat beside the frames, so no corrected movie"
                            )
                    else:
                        out = spim_movie(path, view=step["view"], progress=progress)
                    if out is not None:
                        job["outputs"].append(out.name)
                    base = job["done"]
                job["done"] = job["total"] = max(job["total"], job["done"])
                job["state"] = "done" if job["outputs"] else "error"
                if not job["outputs"]:
                    job["error"] = job["note"] or "Nothing could be written"
            except Exception as exc:
                logger.exception("movie of %s failed", path)
                job["error"] = str(exc)
                job["state"] = "error"

        threading.Thread(target=run, name=f"movie-{kind}", daemon=True).start()
        return job

    @router.get("/api/movies")
    async def movie_status():
        """The movie being made, or the last one."""
        return dict(_MOVIES)

    @router.get("/api/sessions/{session_id}/snapshot/{stem}.png")
    async def session_snapshot_png(session_id: str, stem: str, max: int | None = None):
        """One of the session's filed snapshots (brightfield overview) as a PNG,
        ``?max=N`` for a thumbnail. Found through the store's own listing,
        never from a path in the request."""
        from gently.ui.web.routes.bf import tiff_png_response

        store = _file_store()
        if store is None:
            raise HTTPException(status_code=503, detail="Store not available")

        def find() -> dict | None:
            one = getattr(store, "get_snapshot", None)
            if callable(one):
                got = one(session_id, stem)
                if isinstance(got, dict) or got is None:
                    return got
            try:
                recs = store.list_snapshots(session_id) or []
            except Exception:
                recs = []
            return next((r for r in recs if Path(r.get("file_path") or "").stem == stem), None)

        rec = await asyncio.to_thread(find)
        if rec is None:
            raise HTTPException(status_code=404, detail=f"no snapshot {stem!r} in this session")
        return await asyncio.to_thread(tiff_png_response, Path(rec["file_path"]), stem, max)

    @router.get("/api/sessions/{session_id}/snapshot/{stem}.u16")
    async def session_snapshot_u16(
        session_id: str, stem: str, scale: int = 1, max: int | None = None
    ):
        """A session's snapshot as uint16 bytes (see gently.core.u16): the
        numbers, for a viewer that draws its own contrast and reads values."""
        from gently.ui.web.routes.bf import tiff_u16_response

        store = _file_store()
        if store is None:
            raise HTTPException(status_code=503, detail="Store not available")

        def find() -> dict | None:
            one = getattr(store, "get_snapshot", None)
            if callable(one):
                got = one(session_id, stem)
                if isinstance(got, dict) or got is None:
                    return got
            try:
                recs = store.list_snapshots(session_id) or []
            except Exception:
                recs = []
            return next((r for r in recs if Path(r.get("file_path") or "").stem == stem), None)

        rec = await asyncio.to_thread(find)
        if rec is None:
            raise HTTPException(status_code=404, detail=f"no snapshot {stem!r} in this session")
        return await asyncio.to_thread(
            tiff_u16_response, Path(rec["file_path"]), stem, scale, None, max
        )

    def _volume_u16(
        session_id: str, embryo: str, t: int, z: int | None, scale: int, max_dim: int | None
    ) -> Response:
        """One plane of a volume TIFF, or its projection, as uint16 bytes."""
        import tifffile

        from gently.core.u16 import MEDIA_TYPE, u16_payload

        store = _file_store()
        path = store.get_volume_path(session_id, embryo, t) if store is not None else None
        if path is None:
            raise HTTPException(status_code=404, detail="Volume not found")
        extra: dict = {}
        if z is None:
            # The projection written at acquisition (gently.core.imaging): the
            # three-view layout in counts, read in a few milliseconds. An older
            # session has only the JPEG, so its counts are made from the volume
            # once, here, and kept beside it for the next time.
            from gently.core.imaging import (
                counts_projection,
                read_counts_projection,
                write_counts_projection,
            )

            stored = path.parent.parent / "projections" / f"t{t:04d}.tif"
            if stored.exists():
                arr, layout = read_counts_projection(stored)
            else:
                with tifffile.TiffFile(str(path)) as tf:
                    vol = np.asarray(tf.asarray(maxworkers=8))
                arr, layout = counts_projection(vol)
                try:
                    write_counts_projection(arr, layout, stored)
                except Exception as exc:
                    logger.debug("counts projection not kept for %s: %s", stored, exc)
            extra = {"Projection": "max", "Views": layout.get("views")}
            if isinstance(layout.get("xy"), list) and len(layout["xy"]) == 2:
                extra["XY-Height"], extra["XY-Width"] = layout["xy"]
        else:
            # Volumes are written one page per plane, zlib-compressed, so a plane
            # is one page decoded on its own. A file that holds the stack in one
            # page (a small or foreign TIFF) is read whole.
            with tifffile.TiffFile(str(path)) as tf:
                series = tf.series[0]
                shape = tuple(series.shape)
                planes = int(np.prod(shape[:-2])) if len(shape) > 2 else 1
                pages = series.pages
                if z < 0 or z >= planes:
                    raise HTTPException(
                        status_code=404, detail=f"no plane {z}; the volume has {planes}"
                    )
                if planes > 1 and len(pages) == planes and (pg := pages[z]) is not None:
                    arr = np.asarray(pg.asarray())
                else:
                    arr = np.asarray(series.asarray()).reshape(-1, *shape[-2:])[z]
                extra = {"Planes": planes, "Plane": z}
        body, headers = u16_payload(arr, scale, extra, max_dim=max_dim)
        return Response(
            body, media_type=MEDIA_TYPE, headers={"Cache-Control": "private, max-age=60", **headers}
        )

    @router.get("/api/sessions/{session_id}/embryos/{embryo}/volumes/{t}/projection.u16")
    async def volume_projection_u16(
        session_id: str, embryo: str, t: int, scale: int = 1, max: int | None = None
    ):
        """The maximum projection of a stored volume, as the numbers."""
        return await asyncio.to_thread(_volume_u16, session_id, embryo, t, None, scale, max)

    @router.get("/api/sessions/{session_id}/embryos/{embryo}/volumes/{t}/slice/{z}.u16")
    async def volume_slice_u16(
        session_id: str, embryo: str, t: int, z: int, scale: int = 1, max: int | None = None
    ):
        """One plane of a stored volume, as the numbers."""
        return await asyncio.to_thread(_volume_u16, session_id, embryo, t, z, scale, max)

    return router
