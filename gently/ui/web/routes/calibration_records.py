"""What each calibration looked at: the runs of the live session, and their images.

A calibration's exposures and plots are kept with the embryo (see
``gently/core/calibration_record.py``). These routes are how the browser finds
them again after the run, or after a restart: the list of runs, one run's
images, an image as a PNG, and the folder opened in the file manager.

A run and an image are found through the store's own listing and the run's own
``frames.jsonl``. Nothing in a request is ever joined into a path.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, Response

from gently.ui.web.auth import require_control

logger = logging.getLogger(__name__)

# A filed image never changes.
_IMMUTABLE = {"Cache-Control": "public, max-age=86400, immutable"}


def create_router(server) -> APIRouter:
    router = APIRouter()

    def _store():
        bridge = getattr(server, "agent_bridge", None)
        agent = getattr(bridge, "agent", None) if bridge is not None else None
        store = getattr(agent, "store", None) if agent is not None else None
        return store if store is not None else getattr(server, "gently_store", None)

    def _session_id() -> str | None:
        bridge = getattr(server, "agent_bridge", None)
        agent = getattr(bridge, "agent", None) if bridge is not None else None
        sid = getattr(agent, "session_id", None) if agent is not None else None
        return sid if isinstance(sid, str) else None

    def _run_dir(embryo_id: str, run: str) -> Path:
        store, sid = _store(), _session_id()
        if store is None or not sid:
            raise HTTPException(status_code=503, detail="No session")
        try:
            folder = store.calibration_record_dir(sid, embryo_id, run)
        except Exception:
            folder = None
        if folder is None:
            raise HTTPException(status_code=404, detail=f"no calibration run {run!r}")
        from gently.core.calibration_record import long_path

        return long_path(folder)

    def _with_urls(embryo_id: str, run: str, frames: list[dict]) -> list[dict]:
        base = f"/api/calibration/records/{embryo_id}/{run}/image"
        return [{**f, "url": f"{base}/{f.get('n')}.png"} for f in frames]

    @router.get("/api/calibration/records")
    async def list_records(embryo_id: str | None = None):
        """The calibration runs recorded in this session, oldest first."""
        store, sid = _store(), _session_id()
        if store is None or not sid:
            return {"records": [], "session_id": sid}
        try:
            records = store.list_calibration_records(sid, embryo_id)
        except Exception:
            logger.debug("calibration record listing failed", exc_info=True)
            records = []
        return {"records": records, "session_id": sid}

    @router.get("/api/calibration/records/{embryo_id}/{run}")
    async def one_record(embryo_id: str, run: str):
        """One run: what was asked, what came of it, and every image it kept."""
        from gently.core.calibration_record import read_frames, read_record

        folder = _run_dir(embryo_id, run)
        head = read_record(folder) or {}
        head["images"] = _with_urls(embryo_id, run, read_frames(folder))
        return head

    @router.get("/api/calibration/records/{embryo_id}/{run}/image/{n}.png")
    async def one_image(embryo_id: str, run: str, n: int, max: int | None = None):
        """Image ``n`` of a run as PNG; ``?max=N`` bounds the longer side."""
        from gently.core.calibration_record import read_frames

        folder = _run_dir(embryo_id, run)
        line = next((f for f in read_frames(folder) if f.get("n") == n), None)
        if line is None:
            raise HTTPException(status_code=404, detail=f"no image {n} in {run!r}")
        # The name comes from the run's own index, and must be one of its files.
        path = next(
            (p for p in folder.rglob("*") if p.is_file() and p.name == Path(line["file"]).name),
            None,
        )
        if path is None:
            raise HTTPException(status_code=404, detail=f"image {n} is no longer on disk")
        limit = int(max) if max and max > 0 else 0
        is_png = path.suffix.lower() == ".png"
        try:
            if is_png and not limit:
                return Response(path.read_bytes(), media_type="image/png", headers=_IMMUTABLE)
            from PIL import Image

            from gently.core.imaging import downsample_mean, normalize_to_uint8

            picture: Image.Image
            if is_png:
                picture = Image.open(path)
                picture.thumbnail((limit, limit))
            else:
                import tifffile

                arr = tifffile.imread(str(path))
                if arr.ndim > 2:
                    arr = arr.reshape(-1, *arr.shape[-2:])[0]
                if limit:
                    arr = downsample_mean(arr, limit)
                picture = Image.fromarray(normalize_to_uint8(arr))
            out = io.BytesIO()
            picture.save(out, format="PNG")
        except HTTPException:
            raise
        except Exception as exc:
            logger.exception("calibration image render failed for %s", path)
            raise HTTPException(
                status_code=502, detail=f"could not render image {n}: {exc}"
            ) from exc
        return Response(out.getvalue(), media_type="image/png", headers=_IMMUTABLE)

    @router.post(
        "/api/calibration/records/{embryo_id}/{run}/open-folder",
        dependencies=[Depends(require_control)],
    )
    async def open_folder(embryo_id: str, run: str):
        """Open a run's folder in the file manager, on the machine Gently runs on."""
        from gently.core.calibration_record import short_path
        from gently.ui.web.routes import sessions as sessions_routes

        # The ordinary form of the path: what a file manager is given, and
        # what the operator reads.
        folder = short_path(_run_dir(embryo_id, run))
        try:
            sessions_routes._open_in_file_manager(folder)
        except Exception as exc:
            raise HTTPException(status_code=502, detail=f"could not open folder: {exc}") from exc
        return {"opened": True, "path": str(folder)}

    return router
