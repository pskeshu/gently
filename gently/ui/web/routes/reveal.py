"""Show a thing Gently keeps: in the file manager, or in Fiji.

One route, for every button that says "open folder", "show file" or "open in
Fiji". A request names *what* (a session, an embryo, a timepoint, a DIC
frame, a calibration image, the logs) and the store says where that is.
**No path comes from the browser**, and nothing in a request is joined into
one: ids are looked up, and a run or a frame is matched against the ones
that exist.

The window opens on the machine Gently runs on. Under the desktop shell that
is the operator's own screen. A browser on another computer is told the path
instead, to copy: a window opened for it would open on the microscope, in
front of whoever is sitting there.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from gently.core import reveal as os_reveal
from gently.ui.web.auth import require_control

logger = logging.getLogger(__name__)

# What can be shown, and whether it is a file. Said to the page by /about,
# so the page and the route cannot disagree about the list.
WHAT: dict[str, str] = {
    "storage": "folder",
    "session": "folder",
    "embryo": "folder",
    "timepoint": "file",
    "volume": "file",
    "projection": "file",
    "snapshots": "folder",
    "dic": "file",
    "calibration_run": "folder",
    "calibration_image": "file",
    "removed": "folder",
    "recordings": "folder",
    "logs": "folder",
    "agent": "folder",
    "config": "folder",
    "settings_history": "file",
}

ACTIONS = ("show", "fiji", "path")

# The repo's config folder: settings.local.yml, launch.local.json, hardware.yaml.
_CONFIG_DIR = Path(__file__).resolve().parents[4] / "config"


class RevealRequest(BaseModel):
    what: str
    action: str = "show"
    session_id: str | None = None
    embryo_id: str | None = None
    timepoint: int | None = None
    run: str | None = None
    n: int | None = None
    stem: str | None = None


def _missing(detail: str) -> HTTPException:
    return HTTPException(status_code=404, detail=detail)


def fiji_path() -> Path | None:
    """Fiji, where the rig's settings say it is or where it is usually found."""
    from gently.settings import settings

    return os_reveal.find_fiji(getattr(settings.ui, "fiji_path", "") or None)


def create_router(server) -> APIRouter:
    router = APIRouter()

    def _store():
        bridge = getattr(server, "agent_bridge", None)
        agent = getattr(bridge, "agent", None) if bridge is not None else None
        store = getattr(agent, "store", None) if agent is not None else None
        return store if store is not None else getattr(server, "gently_store", None)

    def _current_session() -> str | None:
        bridge = getattr(server, "agent_bridge", None)
        agent = getattr(bridge, "agent", None) if bridge is not None else None
        sid = getattr(agent, "session_id", None) if agent is not None else None
        return sid if isinstance(sid, str) and sid else None

    def _session_dir(store, req: RevealRequest) -> tuple[Path, str]:
        sid = req.session_id or _current_session()
        if not sid:
            raise _missing("There is no session to look in")
        folder = store._session_dir(sid)
        if folder is None or not Path(folder).is_dir():
            raise _missing(f"Session {sid} has no folder")
        return Path(folder), sid

    def _embryo(store, req: RevealRequest) -> tuple[Path, str, str]:
        sd, sid = _session_dir(store, req)
        if not req.embryo_id:
            raise HTTPException(status_code=422, detail="Which embryo?")
        try:
            folder = store._embryo_dir_for_session(sd, req.embryo_id)
        except ValueError as e:
            raise HTTPException(status_code=422, detail=str(e)) from None
        if not folder.is_dir():
            raise _missing(f"{req.embryo_id} has no folder in session {sid}")
        return folder, sid, req.embryo_id

    def _timepoint(req: RevealRequest) -> int:
        if req.timepoint is None or req.timepoint < 0:
            raise HTTPException(status_code=422, detail="Which timepoint?")
        return int(req.timepoint)

    def resolve(req: RevealRequest) -> Path:
        """Where the thing is. Raises 404 when it is not on disk."""
        what = req.what
        if what not in WHAT:
            raise HTTPException(status_code=422, detail=f"Gently does not keep a {what!r}")

        if what == "config":
            return _CONFIG_DIR
        if what == "settings_history":
            from gently.core import settings_history

            return settings_history.path()

        store = _store()
        if store is None:
            raise HTTPException(status_code=503, detail="The data folder is not open yet")
        root = Path(store.root)

        if what == "storage":
            return root
        if what == "logs":
            return root / "logs"
        if what == "agent":
            return root / "agent"

        if what in ("session", "snapshots", "removed", "recordings"):
            sd, _ = _session_dir(store, req)
            sub = {"snapshots": "snapshots", "removed": "removed", "recordings": "ui-replay"}
            return sd / sub[what] if what in sub else sd

        if what == "embryo":
            return _embryo(store, req)[0]

        if what in ("timepoint", "volume", "projection"):
            _, sid, eid = _embryo(store, req)
            tp = _timepoint(req)
            volume = store.get_volume_path(sid, eid, tp) if what != "projection" else None
            if volume is not None:
                return Path(volume)
            # A timepoint is its volume. Where the volume is not on this disk
            # (it was moved off, or never came) its projection stands in.
            projection = store.get_projection_path(sid, eid, tp) if what != "volume" else None
            if projection is not None:
                return Path(projection)
            raise _missing(f"{eid} t{tp} is not on disk in session {sid}")

        if what == "dic":
            _, sid = _session_dir(store, req)
            for rec in store.list_snapshots(sid, "dic"):
                fp = rec.get("file_path")
                if fp and Path(fp).stem == req.stem:
                    return Path(fp)
            raise _missing(f"No DIC frame {req.stem!r} in session {sid}")

        # calibration_run, calibration_image
        _, sid, eid = _embryo(store, req)
        run = store.calibration_record_dir(sid, eid, req.run or "")
        if run is None:
            raise _missing(f"No calibration run {req.run!r} for {eid}")
        if what == "calibration_run":
            return Path(run)
        from gently.core.calibration_record import long_path, read_frames

        folder = long_path(run)
        line = next((f for f in read_frames(folder) if f.get("n") == req.n), None)
        if line is None:
            raise _missing(f"No image {req.n} in run {req.run!r}")
        # The name comes from the run's own index, and must be one of its files.
        found = next(
            (p for p in folder.rglob("*") if p.is_file() and p.name == Path(line["file"]).name),
            None,
        )
        if found is None:
            raise _missing(f"Image {req.n} of run {req.run!r} is no longer on disk")
        return found

    def _readable(path: Path) -> str:
        """The path as the operator reads it, and as a file manager takes it."""
        from gently.core.calibration_record import short_path

        return str(short_path(path))

    @router.get("/api/reveal/about")
    async def about(request: Request):
        """What this browser can ask for: whether a window would open where
        it can be seen, and whether Fiji is there to open an image in."""
        host = request.client.host if request.client else None
        fiji = await asyncio.to_thread(fiji_path)
        return {
            "local": os_reveal.is_local(host),
            "file_manager": os_reveal.file_manager_name(),
            "fiji": {"found": fiji is not None, "path": str(fiji) if fiji else None},
            "what": WHAT,
        }

    @router.post("/api/reveal", dependencies=[Depends(require_control)])
    async def reveal(req: RevealRequest, request: Request) -> dict[str, Any]:
        if req.action not in ACTIONS:
            raise HTTPException(status_code=422, detail=f"Unknown action {req.action!r}")
        path = await asyncio.to_thread(resolve, req)
        if not path.exists():
            raise _missing(f"{_readable(path)} is not on disk")
        shown = _readable(path)
        answer: dict[str, Any] = {
            "path": shown,
            "kind": "folder" if path.is_dir() else "file",
            "what": req.what,
            "action": req.action,
            "opened": False,
        }
        if req.action == "path":
            return answer

        host = request.client.host if request.client else None
        if not os_reveal.is_local(host):
            # Not an error: the page copies the path for them instead.
            answer["reason"] = "remote"
            return answer

        try:
            if req.action == "fiji":
                if path.is_dir() or path.suffix.lower() not in os_reveal.IMAGE_SUFFIXES:
                    raise HTTPException(
                        status_code=422, detail=f"{path.name} is not an image Fiji opens"
                    )
                fiji = await asyncio.to_thread(fiji_path)
                if fiji is None:
                    raise HTTPException(
                        status_code=409,
                        detail="Fiji was not found on this computer. Say where it is in "
                        "Settings › System › Fiji.",
                    )
                await asyncio.to_thread(os_reveal.open_in_fiji, Path(shown), fiji)
                answer["fiji"] = str(fiji)
            else:
                await asyncio.to_thread(os_reveal.show, Path(shown))
        except HTTPException:
            raise
        except Exception as exc:
            logger.warning("could not %s %s: %s", req.action, shown, exc)
            raise HTTPException(status_code=502, detail=f"Could not open it: {exc}") from exc
        answer["opened"] = True
        return answer

    return router
