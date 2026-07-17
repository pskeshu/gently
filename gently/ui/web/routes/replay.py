"""Session replay — rrweb ingest + post-hoc player.

The recorder (``static/js/replay-recorder.js``) POSTs event batches here and
this module appends them to per-session JSONL files in the file store:

    sessions/{session}/ui-replay/rrweb-{tab}.jsonl   full rrweb event stream
    sessions/{session}/ui-replay/actions.jsonl       semantic action log
    sessions/{session}/ui-replay/meta.yaml           tabs seen, user agents

Batches that arrive while no session is active land in an unassigned bucket
(``{storage}/ui-replay/unassigned-{YYYYMMDD}/``) so nothing is dropped.

Detached by design (docs/superpowers/specs/2026-07-13-session-replay-design.md):
nothing else in gently imports this module, and the ingest path never raises
into the UI — failures return JSON errors the recorder treats as backoff
signals. Ingest is unauthenticated like the viewing surface: recording must
cover view-role users, and the instrument is localhost-first.
"""

import asyncio
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse, JSONResponse

from gently.core.file_store import _write_yaml
from gently.settings import settings

logger = logging.getLogger(__name__)

_TAB_RE = re.compile(r"^[a-f0-9]{4,16}$")
_UNASSIGNED_RE = re.compile(r"^unassigned-\d{8}$")
_MAX_BATCH_BYTES = 32 * 1024 * 1024
# meta.yaml is only rewritten when a new (dir, tab) pair first appears.
_seen_tabs: set[tuple[str, str]] = set()
# (dir, tab) pairs whose rrweb file has hit the per-tab size cap — so the
# "capped" marker is logged once, not on every subsequent ingest.
_capped_tabs: set[tuple[str, str]] = set()
# Retention prune runs once per process, lazily on the first ingest (by which
# point the store is up), off the event loop.
_pruned = False


def _store(server) -> Any:
    return getattr(server, "gently_store", None)


def _active_replay_dir(server) -> tuple[Path | None, str | None]:
    """The active session's ui-replay dir, else the day's unassigned bucket."""
    store = _store(server)
    if store is None:
        return None, None
    sid: str | None = None
    try:
        sid = server._current_session_id()
    except Exception:  # noqa: BLE001 — never let session lookup break ingest
        sid = None
    if sid:
        sd = store._session_dir(sid)
        if sd is not None and sd.exists():
            return sd / "ui-replay", sid
    day = datetime.now().strftime("%Y%m%d")
    return Path(store.root) / "ui-replay" / f"unassigned-{day}", None


def _resolve_replay_dir(server, session_id: str) -> Path | None:
    """Map a player-facing id (session id or unassigned bucket) to its dir."""
    store = _store(server)
    if store is None:
        return None
    if _UNASSIGNED_RE.match(session_id):
        d = Path(store.root) / "ui-replay" / session_id
        return d if d.exists() else None
    sd = store._session_dir(session_id)
    if sd is None:
        return None
    d = sd / "ui-replay"
    return d if d.exists() else None


def _list_recordings(server) -> list[dict[str, Any]]:
    """Every id (session or unassigned bucket) that has replay data."""
    store = _store(server)
    if store is None:
        return []
    out: list[dict[str, Any]] = []
    index: dict[str, str] = dict(getattr(store, "_index", {}) or {})
    for sid, folder in sorted(index.items(), key=lambda kv: kv[1], reverse=True):
        d = Path(store.root) / "sessions" / folder / "ui-replay"
        if d.is_dir():
            out.append({"id": sid, "folder": folder, **_dir_stats(d)})
    unassigned = Path(store.root) / "ui-replay"
    if unassigned.is_dir():
        for d in sorted(unassigned.iterdir(), reverse=True):
            if d.is_dir() and _UNASSIGNED_RE.match(d.name):
                out.append({"id": d.name, "folder": d.name, **_dir_stats(d)})
    return out


def _dir_stats(d: Path) -> dict[str, Any]:
    tabs = sorted(p.stem.replace("rrweb-", "") for p in d.glob("rrweb-*.jsonl"))
    size = sum(p.stat().st_size for p in d.glob("*.jsonl"))
    return {"tabs": tabs, "bytes": size}


def _append_lines(path: Path, records: list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")


def _update_meta(base: Path, tab: str, user_agent: str) -> None:
    key = (str(base), tab)
    if key in _seen_tabs:
        return
    _seen_tabs.add(key)
    meta_path = base / "meta.yaml"
    meta: dict[str, Any] = {}
    if meta_path.exists():
        try:
            import yaml

            meta = yaml.safe_load(meta_path.read_text(encoding="utf-8")) or {}
        except Exception:  # noqa: BLE001 — corrupt meta must not break ingest
            meta = {}
    tabs = meta.setdefault("tabs", {})
    tabs.setdefault(
        tab,
        {"first_seen": datetime.now().isoformat(), "user_agent": user_agent[:200]},
    )
    _write_yaml(meta_path, meta)


def _all_replay_dirs(store) -> list[Path]:
    """Every ui-replay dir on disk (per-session + unassigned buckets)."""
    root = Path(store.root)
    dirs = [d for d in (root / "sessions").glob("*/ui-replay") if d.is_dir()]
    unassigned = root / "ui-replay"
    if unassigned.is_dir():
        dirs += [d for d in unassigned.iterdir() if d.is_dir() and _UNASSIGNED_RE.match(d.name)]
    return dirs


def _prune_recordings(server) -> None:
    """Keep total rrweb footprint under the configured budget by deleting the
    oldest recordings first. Conservative: always keeps the newest few, skips
    the active session, and never lets a failure escape into ingest."""
    store = _store(server)
    if store is None:
        return
    budget = int(settings.ui.replay_total_budget_mb * 1024 * 1024)
    active: str | None = None
    try:
        active = server._current_session_id()
    except Exception:  # noqa: BLE001
        active = None
    entries: list[tuple[float, int, Path]] = []
    for d in _all_replay_dirs(store):
        try:
            files = list(d.glob("*.jsonl"))
            size = sum(f.stat().st_size for f in files)
            mtime = max((f.stat().st_mtime for f in files), default=0.0)
        except OSError:
            continue
        # Never prune the active session's live recording.
        if active and active in str(d):
            continue
        entries.append((mtime, size, d))
    total = sum(s for _, s, _ in entries)
    if total <= budget:
        return
    entries.sort(key=lambda e: e[0])  # oldest first
    keep_newest = 3
    prunable = entries[: max(0, len(entries) - keep_newest)]
    import shutil

    for _mtime, size, d in prunable:
        if total <= budget:
            break
        try:
            shutil.rmtree(d)
            total -= size
            logger.info("replay: pruned old recording %s (%.0f MB)", d.name, size / 1048576)
        except OSError:
            logger.debug("replay: could not prune %s", d, exc_info=True)


def _read_jsonl(path: Path) -> list[Any]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def create_router(server) -> APIRouter:
    router = APIRouter()

    # One Jinja global so every template can conditionally include the
    # recorder — the flag is the only coupling between replay and the pages.
    try:
        server.templates.env.globals["replay_enabled"] = settings.ui.replay
        server.templates.env.globals["replay_fidelity"] = settings.ui.replay_fidelity
    except Exception:  # noqa: BLE001 — a missing templates attr is not fatal
        pass

    @router.post("/replay/ingest")
    async def ingest(request: Request):
        if not settings.ui.replay:
            return JSONResponse({"error": "replay disabled"}, status_code=403)
        try:
            length = int(request.headers.get("content-length") or 0)
        except ValueError:
            length = 0
        if length > _MAX_BATCH_BYTES:
            return JSONResponse({"error": "batch too large"}, status_code=413)
        try:
            batch = await request.json()
        except Exception:  # noqa: BLE001
            return JSONResponse({"error": "bad json"}, status_code=400)
        tab = str(batch.get("tab") or "")
        if not _TAB_RE.match(tab):
            return JSONResponse({"error": "bad tab id"}, status_code=400)

        base, sid = _active_replay_dir(server)
        if base is None:
            return JSONResponse({"error": "no store"}, status_code=503)

        rrweb_events = batch.get("rrweb") or []
        actions = list(batch.get("actions") or [])
        for a in actions:
            if isinstance(a, dict):
                a["tab"] = tab
        gap = batch.get("gap")
        if gap:
            actions.append(
                {
                    "t": datetime.now().isoformat(),
                    "action": "gap",
                    "tab": tab,
                    "params": gap,
                }
            )

        # Prune old recordings to the total budget once per process, off the
        # event loop (the store is up by the time ingest first fires).
        global _pruned
        if not _pruned:
            _pruned = True
            asyncio.create_task(asyncio.to_thread(_prune_recordings, server))

        cap_bytes = int(settings.ui.replay_max_tab_mb * 1024 * 1024)

        def _write() -> None:
            rrweb_path = base / f"rrweb-{tab}.jsonl"
            if rrweb_events:
                try:
                    over_cap = rrweb_path.exists() and rrweb_path.stat().st_size >= cap_bytes
                except OSError:
                    over_cap = False
                if over_cap:
                    # A single tab hit the size cap — stop growing its rrweb
                    # stream (the live map/telemetry re-render at poll rate can
                    # run this to gigabytes). Keep the small action log going;
                    # mark the truncation once.
                    key = (str(base), tab)
                    if key not in _capped_tabs:
                        _capped_tabs.add(key)
                        actions.append(
                            {
                                "t": datetime.now().isoformat(),
                                "action": "rrweb-capped",
                                "tab": tab,
                                "params": {"cap_mb": settings.ui.replay_max_tab_mb},
                            }
                        )
                        logger.warning(
                            "replay: rrweb cap (%.0f MB) hit for tab %s — dropping further frames",
                            settings.ui.replay_max_tab_mb,
                            tab,
                        )
                else:
                    _append_lines(rrweb_path, rrweb_events)
            if actions:
                _append_lines(base / "actions.jsonl", actions)
            _update_meta(base, tab, request.headers.get("user-agent", ""))

        try:
            await asyncio.to_thread(_write)
        except Exception:  # noqa: BLE001 — degrade recording, never the app
            logger.exception("replay ingest write failed")
            return JSONResponse({"error": "write failed"}, status_code=500)
        return {"ok": True, "session": sid}

    # ---- post-hoc player (read-only; available even when recording is off) --

    @router.get("/replay", response_class=HTMLResponse)
    async def replay_index(request: Request):
        return server.templates.TemplateResponse(
            request,
            "replay.html",
            {"session_id": None, "recordings": _list_recordings(server)},
        )

    @router.get("/replay/api/recordings")
    async def api_recordings():
        return {"recordings": _list_recordings(server)}

    @router.get("/replay/api/{session_id}/tabs")
    async def api_tabs(session_id: str):
        d = _resolve_replay_dir(server, session_id)
        if d is None:
            return JSONResponse({"error": "no replay data"}, status_code=404)
        tabs = []
        for p in sorted(d.glob("rrweb-*.jsonl")):
            tabs.append(
                {
                    "tab": p.stem.replace("rrweb-", ""),
                    "bytes": p.stat().st_size,
                    "events": sum(1 for _ in open(p, encoding="utf-8")),
                }
            )
        return {"tabs": tabs}

    @router.get("/replay/api/{session_id}/events")
    async def api_events(session_id: str, tab: str, start: int = 0, end: int = 0):
        """rrweb events for one tab, optionally clipped to [start, end] ms."""
        d = _resolve_replay_dir(server, session_id)
        if d is None:
            return JSONResponse({"error": "no replay data"}, status_code=404)
        path = d / f"rrweb-{tab}.jsonl"
        if not _TAB_RE.match(tab) or not path.exists():
            return JSONResponse({"error": "unknown tab"}, status_code=404)
        events = await asyncio.to_thread(_read_jsonl, path)
        if start or end:
            events = [
                e
                for e in events
                if isinstance(e, dict)
                and (not start or e.get("timestamp", 0) >= start)
                and (not end or e.get("timestamp", 0) <= end)
            ]
        return {"events": events}

    @router.get("/replay/api/{session_id}/actions")
    async def api_actions(session_id: str):
        d = _resolve_replay_dir(server, session_id)
        if d is None:
            return JSONResponse({"error": "no replay data"}, status_code=404)
        path = d / "actions.jsonl"
        actions = await asyncio.to_thread(_read_jsonl, path) if path.exists() else []
        return {"actions": actions}

    @router.get("/replay/{session_id}", response_class=HTMLResponse)
    async def replay_player(request: Request, session_id: str):
        if _resolve_replay_dir(server, session_id) is None:
            return JSONResponse({"error": "no replay data"}, status_code=404)
        return server.templates.TemplateResponse(
            request,
            "replay.html",
            {"session_id": session_id, "recordings": []},
        )

    return router
