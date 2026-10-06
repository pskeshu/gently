"""The DIC overview frames of the live session, as a list and as PNGs.

The overview channel files one TIFF per round under the session's
``snapshots/`` (see ``TimelapseOrchestrator._capture_dic_overview``). The
event that announces each frame carries a thumbnail, which is enough to show
it landing; it is not enough to LOOK at it — "it appears more like an icon
than a clickable image". These two routes are what a click opens: the full
frame, rendered from the file on disk, and the list a page that loaded after
the run started hydrates from.

Frames are found through the store's own listing, never from a path in the
request, so ``{stem}`` can only ever name a file the session filed.
"""

from __future__ import annotations

import io
import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException, Response

logger = logging.getLogger(__name__)

# Cached hard: a filed frame never changes.
_IMMUTABLE = {"Cache-Control": "public, max-age=86400, immutable"}


def create_router(server) -> APIRouter:
    router = APIRouter()

    def _session_id() -> str | None:
        bridge = getattr(server, "agent_bridge", None)
        agent = bridge.agent if bridge is not None else None
        return getattr(agent, "session_id", None) if agent else None

    def _records() -> list[dict]:
        store = getattr(server, "gently_store", None)
        sid = _session_id()
        if store is None or not sid:
            return []
        try:
            return list(store.list_snapshots(sid, "dic"))
        except Exception:
            logger.debug("DIC frame listing failed", exc_info=True)
            return []

    def _stem(rec: dict) -> str | None:
        fp = rec.get("file_path")
        return Path(fp).stem if fp else None

    def _reference_records() -> list[dict]:
        store = getattr(server, "gently_store", None)
        sid = _session_id()
        if store is None or not sid:
            return []
        try:
            from gently.app.brightfield import list_records

            return list_records(store, sid)
        except Exception:
            logger.debug("reference listing failed", exc_info=True)
            return []

    def _session_dir() -> Path | None:
        store = getattr(server, "gently_store", None)
        sid = _session_id()
        if store is None or not sid:
            return None
        sd = store._session_dir(sid)
        return Path(sd) if sd is not None else None

    def _correction_for(rec: dict, records: list[dict]) -> tuple[Path, Path] | None:
        """The dark and flat on disk this frame can be corrected with, or None."""
        from gently.app.brightfield import references_for_frame, resolve_reference_paths

        sd = _session_dir()
        if sd is None:
            return None
        meta = rec.get("metadata") or {}
        return resolve_reference_paths(sd, references_for_frame(meta, records))

    @router.get("/api/dic/frames")
    async def list_dic_frames():
        """Every overview frame the live session has filed, oldest first, with
        the light it was taken under and whether a dark and flat exist for it."""
        frames = []
        records = _reference_records()
        for rec in _records():
            stem = _stem(rec)
            if not stem:
                continue
            meta = rec.get("metadata") or {}
            frames.append(
                {
                    "stem": stem,
                    "frame": meta.get("frame"),
                    "round": meta.get("round"),
                    "position": meta.get("position"),
                    "captured_at": meta.get("captured_at") or rec.get("captured_at"),
                    "width": rec.get("width"),
                    "height": rec.get("height"),
                    "exposure_ms": meta.get("exposure_ms"),
                    "light": meta.get("light"),
                    "led_intensity_pct": meta.get("led_intensity_pct"),
                    "correctable": _correction_for(rec, records) is not None,
                    "url": f"/api/dic/frames/{stem}.png",
                }
            )
        return {"frames": frames, "count": len(frames)}

    @router.get("/api/dic/frames/{stem}.png")
    async def dic_frame_png(stem: str, max: int | None = None, corrected: bool = False):
        """One overview frame as PNG; ``?max=N`` bounds the longer side for a
        thumbnail; ``?corrected=1`` divides the session's dark and flat out
        first (404 if the frame has none)."""
        rec = next((r for r in _records() if _stem(r) == stem), None)
        if rec is None:
            raise HTTPException(status_code=404, detail=f"no DIC frame {stem!r} in this session")
        refs = None
        if corrected:
            refs = _correction_for(rec, _reference_records())
            if refs is None:
                raise HTTPException(
                    status_code=404, detail=f"no dark and flat for frame {stem!r} in this session"
                )
        return tiff_png_response(Path(rec["file_path"]), stem, max, correction=refs)

    return router


def tiff_png_response(
    path: Path,
    stem: str,
    max: int | None = None,
    correction: tuple[Path, Path] | None = None,
) -> Response:
    """A filed TIFF as a PNG response; ``max`` bounds the longer side for a
    thumbnail; ``correction`` is the (dark, flat) pair to divide out first.
    Shared by the live-session DIC routes and the Sessions tab's per-session
    snapshot route."""
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"frame {stem!r} is no longer on disk")
    try:
        import tifffile
        from PIL import Image

        from gently.core.imaging import downsample_mean, normalize_to_uint8

        arr = tifffile.imread(str(path))
        if arr.ndim > 2:  # a stack or a colour plane: the first 2D frame
            arr = arr.reshape(-1, *arr.shape[-2:])[0]
        if correction is not None:
            from gently.app.brightfield import correct

            dark = tifffile.imread(str(correction[0]))
            flat = tifffile.imread(str(correction[1]))
            if dark.shape == arr.shape and flat.shape == arr.shape:
                arr = correct(arr, dark, flat)
            else:
                raise ValueError("the dark and flat are not the frame's size")
        if max and max > 0:
            # Averaged, not sampled: see downsample_mean.
            arr = downsample_mean(arr, int(max))
        png = io.BytesIO()
        Image.fromarray(normalize_to_uint8(arr)).save(png, format="PNG")
    except Exception as exc:
        logger.exception("frame render failed for %s", path)
        raise HTTPException(status_code=502, detail=f"could not render {stem!r}: {exc}") from exc
    return Response(png.getvalue(), media_type="image/png", headers=_IMMUTABLE)


# ``max`` is shadowed by the query parameter above, on purpose — it is the
# name the URL uses. The builtin is kept under its own name for the arithmetic.
