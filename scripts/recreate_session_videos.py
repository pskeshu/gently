"""
Rebuild per-embryo and grid timelapse videos from a Gently3 session on disk.

Reads pre-rendered projections + meta + perception predictions and renders:

  - one MP4 per embryo at variable playback rate (5-min frames linger,
    1-min frames go faster, burst frames flash by) with timecode, cadence
    label, and predicted stage overlay
  - one synchronized grid MP4 across all embryos sharing a unified global
    timeline (cells stay dark until that embryo's first frame)

Usage
-----
    python scripts/recreate_session_videos.py
        [--session SESSION_OR_FOLDER]   # default: latest under D:/Gently3/sessions/
        [--out DIR]                      # default: <session>/_recap/
        [--storage DIR]                  # default: D:/Gently3
        [--speedup FACTOR]               # default: 600 (real_seconds / video_seconds)
        [--fps N]                        # default: 30
        [--no-perception]                # skip stage overlays
        [--no-grid] [--no-per-embryo]    # skip one or the other
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, cast

import cv2
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont

logger = logging.getLogger("recreate_videos")

# ---------------------------------------------------------------------------
# Presentation palette — dark, projector-friendly. All BGR (cv2 native).
# ---------------------------------------------------------------------------
COL_BG = (22, 17, 14)  # #0E1116 — main background
COL_PANEL = (38, 31, 26)  # #1A1F26 — card surface
COL_PANEL_HI = (64, 52, 45)  # #2D3440 — borders
COL_TEXT = (235, 233, 229)  # off-white primary
COL_TEXT_DIM = (175, 162, 156)  # secondary
COL_ACCENT = (250, 165, 96)  # sky blue (BGR)
COL_DIVIDER = (80, 70, 60)

# Cadence regimes — interval (seconds) → (label, BGR color, regime key)
CADENCE_5MIN = ("5 MIN", (137, 211, 52), "5min")  # emerald (#34D399)
CADENCE_1MIN = ("1 MIN", (36, 191, 251), "1min")  # amber (#FBBF24)
CADENCE_BURST = ("BURST", (113, 113, 248), "burst")  # red (#F87171)
CADENCE_OTHER = ("---", (160, 160, 160), "other")


def cadence_for_interval(interval_s: float) -> tuple[str, tuple[int, int, int], str]:
    if interval_s <= 5.0:
        return CADENCE_BURST
    if interval_s <= 90.0:
        return CADENCE_1MIN
    if interval_s <= 600.0:
        return CADENCE_5MIN
    return CADENCE_OTHER


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
@dataclass
class FrameRecord:
    timepoint: int
    jpg_path: Path
    acquired_at: datetime
    role: str | None = None
    predicted_stage: str | None = None
    interval_to_next_s: float = 0.0  # filled later
    cadence_label: str = "---"
    cadence_color: tuple[int, int, int] = (160, 160, 160)


@dataclass
class EmbryoData:
    embryo_id: str
    frames: list[FrameRecord] = field(default_factory=list)


@dataclass
class BurstData:
    embryo_id: str
    request_id: str
    burst_dir: Path
    manifest: dict[str, Any] = field(default_factory=dict)
    frames: list[FrameRecord] = field(default_factory=list)


def _read_meta(meta_path: Path) -> dict:
    try:
        with meta_path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as exc:
        logger.warning("Could not read %s: %s", meta_path, exc)
        return {}


def _load_predictions(predictions_path: Path) -> dict[int, str]:
    """Map timepoint -> predicted_stage from predictions.jsonl (most-recent wins)."""
    out: dict[int, str] = {}
    if not predictions_path.exists():
        return out
    with predictions_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            tp = rec.get("timepoint")
            stage = rec.get("predicted_stage")
            if tp is not None and stage:
                out[int(tp)] = str(stage)
    return out


def load_embryo(embryo_dir: Path, *, use_perception: bool = True) -> EmbryoData | None:
    projections_dir = embryo_dir / "projections"
    volumes_dir = embryo_dir / "volumes"
    if not projections_dir.is_dir():
        return None

    predictions = _load_predictions(embryo_dir / "predictions.jsonl") if use_perception else {}

    frames: list[FrameRecord] = []
    for jpg in sorted(projections_dir.glob("t*.jpg")):
        stem = jpg.stem  # e.g. "t0003"
        try:
            tp = int(stem.lstrip("t"))
        except ValueError:
            continue

        meta = _read_meta(volumes_dir / f"{stem}.meta.yaml")
        acquired_raw = meta.get("acquired_at")
        if not acquired_raw:
            # Fallback to file mtime — better than dropping the frame.
            acquired_at = datetime.fromtimestamp(jpg.stat().st_mtime)
        else:
            acquired_at = datetime.fromisoformat(str(acquired_raw))

        frames.append(
            FrameRecord(
                timepoint=tp,
                jpg_path=jpg,
                acquired_at=acquired_at,
                role=meta.get("metadata", {}).get("role"),
                predicted_stage=predictions.get(tp),
            )
        )

    if not frames:
        return None

    frames.sort(key=lambda f: f.acquired_at)
    # interval-to-next (seconds); last frame inherits the previous interval.
    for i in range(len(frames) - 1):
        frames[i].interval_to_next_s = (
            frames[i + 1].acquired_at - frames[i].acquired_at
        ).total_seconds()
    if len(frames) >= 2:
        frames[-1].interval_to_next_s = frames[-2].interval_to_next_s
    else:
        frames[-1].interval_to_next_s = 300.0  # default 5min if only one frame
    for fr in frames:
        label, color, _ = cadence_for_interval(fr.interval_to_next_s)
        fr.cadence_label = label
        fr.cadence_color = color

    return EmbryoData(embryo_id=embryo_dir.name, frames=frames)


def _looks_like_full_session(path: Path) -> bool:
    """A 'real' session has session.yaml and embryos/. Burst-only shadow dirs don't."""
    return path.is_dir() and (path / "session.yaml").exists() and (path / "embryos").is_dir()


def load_burst(burst_dir: Path) -> BurstData | None:
    """Load one burst (request_id subfolder) from disk.

    Layout written by BurstAcquisition.run:
        bursts/{request_id}/burst.yaml
        bursts/{request_id}/t{NNNN}.tif
        bursts/{request_id}/t{NNNN}.meta.yaml
        bursts/{request_id}/projections/t{NNNN}.jpg
        bursts/{request_id}/burst.mp4   # may be absent if codec failed
    """
    if not burst_dir.is_dir():
        return None
    proj_dir = burst_dir / "projections"
    if not proj_dir.is_dir():
        return None

    manifest: dict[str, Any] = {}
    manifest_path = burst_dir / "burst.yaml"
    if manifest_path.exists():
        try:
            with manifest_path.open("r", encoding="utf-8") as f:
                manifest = yaml.safe_load(f) or {}
        except Exception as exc:
            logger.warning("Could not read %s: %s", manifest_path, exc)

    embryo_id = manifest.get("embryo_id") or burst_dir.parent.parent.name
    request_id = manifest.get("request_id") or burst_dir.name

    frames: list[FrameRecord] = []
    for jpg in sorted(proj_dir.glob("t*.jpg")):
        stem = jpg.stem
        try:
            idx = int(stem.lstrip("t"))
        except ValueError:
            continue
        meta = _read_meta(burst_dir / f"{stem}.meta.yaml")
        acquired_raw = meta.get("acquired_at")
        if acquired_raw:
            acquired_at = datetime.fromisoformat(str(acquired_raw))
        else:
            acquired_at = datetime.fromtimestamp(jpg.stat().st_mtime)
        frames.append(
            FrameRecord(
                timepoint=idx,
                jpg_path=jpg,
                acquired_at=acquired_at,
                role="burst",
                predicted_stage=None,
            )
        )

    if not frames:
        return None

    frames.sort(key=lambda f: f.acquired_at)
    for i in range(len(frames) - 1):
        frames[i].interval_to_next_s = (
            frames[i + 1].acquired_at - frames[i].acquired_at
        ).total_seconds()
    if len(frames) >= 2:
        frames[-1].interval_to_next_s = frames[-2].interval_to_next_s
    else:
        frames[-1].interval_to_next_s = 1.0
    for fr in frames:
        label, color, _ = cadence_for_interval(fr.interval_to_next_s)
        fr.cadence_label = label
        fr.cadence_color = color

    return BurstData(
        embryo_id=embryo_id,
        request_id=request_id,
        burst_dir=burst_dir,
        manifest=manifest,
        frames=frames,
    )


def discover_bursts(session_dir: Path) -> list[BurstData]:
    """Find every burst across all embryos under this session."""
    out: list[BurstData] = []
    for embryo_dir in sorted((session_dir / "embryos").glob("embryo_*")):
        bursts_root = embryo_dir / "bursts"
        if not bursts_root.is_dir():
            continue
        for sub in sorted(bursts_root.iterdir()):
            if not sub.is_dir():
                continue
            burst = load_burst(sub)
            if burst is not None:
                out.append(burst)
    return out


def find_session(storage: Path, requested: str | None) -> Path:
    sessions_root = storage / "sessions"
    if not sessions_root.is_dir():
        raise FileNotFoundError(f"Sessions root not found: {sessions_root}")

    # _index.yaml is the source of truth for short-id -> folder-name mapping.
    index: dict[str, str] = {}
    index_path = sessions_root / "_index.yaml"
    if index_path.exists():
        try:
            with index_path.open("r", encoding="utf-8") as f:
                index = yaml.safe_load(f) or {}
        except Exception as exc:
            logger.warning("Could not read %s: %s", index_path, exc)

    if requested:
        # 1. Exact short-id lookup via the index.
        if requested in index:
            candidate = sessions_root / str(index[requested])
            if _looks_like_full_session(candidate):
                return candidate
        # 2. Direct folder name.
        direct = sessions_root / requested
        if _looks_like_full_session(direct):
            return direct
        # 3. Substring match — prefer full sessions over shadow dirs.
        matches = [p for p in sessions_root.iterdir() if p.is_dir() and requested in p.name]
        full = [p for p in matches if _looks_like_full_session(p)]
        pool = full or matches
        if not pool:
            raise FileNotFoundError(f"No session matches {requested!r}")
        return sorted(pool, key=lambda p: p.stat().st_mtime)[-1]

    # Latest full session by mtime.
    candidates = [p for p in sessions_root.iterdir() if _looks_like_full_session(p)]
    if not candidates:
        raise FileNotFoundError(f"No full sessions under {sessions_root}")
    return sorted(candidates, key=lambda p: p.stat().st_mtime)[-1]


# ---------------------------------------------------------------------------
# Frame rendering
# ---------------------------------------------------------------------------
HEADER_H = 80  # px above the projection (per-embryo header)
FOOTER_H = 28  # px below the projection (cadence bar + spacing)
GRID_TITLE_H = 110  # px at top of grid
GRID_GUTTER = 14  # px between cells
GRID_MARGIN = 18  # px around grid edge


# --- TrueType font loading (Segoe UI on Windows, fall back to Inter / DejaVu) ---
_FONT_CANDIDATES = [
    "C:/Windows/Fonts/segoeuib.ttf",  # Segoe UI Bold
    "C:/Windows/Fonts/segoeui.ttf",
    "C:/Windows/Fonts/Inter-Regular.ttf",
    "C:/Windows/Fonts/calibri.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]
_FONT_REGULAR_CANDIDATES = [
    "C:/Windows/Fonts/segoeui.ttf",
    "C:/Windows/Fonts/calibri.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
]


def _load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = _FONT_CANDIDATES if bold else _FONT_REGULAR_CANDIDATES
    for path in candidates:
        if Path(path).exists():
            try:
                return ImageFont.truetype(path, size=size)
            except Exception:
                continue
    return cast(ImageFont.FreeTypeFont, ImageFont.load_default())


_font_cache: dict[tuple[int, bool], ImageFont.FreeTypeFont] = {}


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    key = (size, bold)
    if key not in _font_cache:
        _font_cache[key] = _load_font(size, bold)
    return _font_cache[key]


def _bgr_to_rgb(c: tuple[int, int, int]) -> tuple[int, int, int]:
    return (c[2], c[1], c[0])


def _draw_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    *,
    size: int = 16,
    bold: bool = False,
    color: tuple[int, int, int] = COL_TEXT,
    anchor: str = "la",
) -> None:
    """Draw RGB text via PIL. ``color`` is BGR (project convention)."""
    draw.text(xy, text, font=font(size, bold=bold), fill=_bgr_to_rgb(color), anchor=anchor)


def _draw_rounded(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    radius: int,
    *,
    fill=None,
    outline=None,
    width: int = 1,
) -> None:
    fill_rgb = _bgr_to_rgb(fill) if fill else None
    outline_rgb = _bgr_to_rgb(outline) if outline else None
    draw.rounded_rectangle(box, radius=radius, fill=fill_rgb, outline=outline_rgb, width=width)


def _format_elapsed(seconds: float) -> str:
    if seconds < 0:
        seconds = 0
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"T+{h:02d}:{m:02d}:{s:02d}"


VIEW_CROP: str = "a"  # "a" | "b" | "both"


def _load_projection_bgr(jpg_path: Path) -> np.ndarray | None:
    img = cv2.imread(str(jpg_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    if VIEW_CROP in ("a", "b"):
        # diSPIM projections are side-by-side: A on the left, B on the right.
        w = img.shape[1]
        mid = w // 2
        img = img[:, :mid] if VIEW_CROP == "a" else img[:, mid:]
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def _proj_to_pil(proj: np.ndarray, target_size: tuple[int, int]) -> Image.Image:
    """Resize a BGR ndarray projection and return as PIL RGB Image."""
    resized = cv2.resize(proj, target_size, interpolation=cv2.INTER_AREA)
    return Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))


def _decorate_embryo_frame(
    proj: np.ndarray,
    *,
    proj_w: int,
    proj_h: int,
    elapsed_s: float,
    embryo_id: str,
    cadence_label: str,
    cadence_color: tuple[int, int, int],
    stage: str | None,
    stage_changed: bool,
    session_label: str | None = None,
) -> np.ndarray:
    """Compose header + projection (with subtle card frame) + cadence bar."""
    margin = 28
    width = proj_w + 2 * margin
    height = HEADER_H + proj_h + FOOTER_H + 2 * margin

    canvas = Image.new("RGB", (width, height), color=_bgr_to_rgb(COL_BG))
    draw = ImageDraw.Draw(canvas)

    # --- Header panel ---
    header_box = (margin, margin, width - margin, margin + HEADER_H - 14)
    _draw_rounded(draw, header_box, radius=10, fill=COL_PANEL, outline=COL_PANEL_HI, width=1)

    # Big timecode (left)
    _draw_text(
        draw,
        (margin + 18, margin + 12),
        _format_elapsed(elapsed_s),
        size=28,
        bold=True,
        color=COL_ACCENT,
    )
    _draw_text(
        draw,
        (margin + 18, margin + 44),
        embryo_id.replace("_", " ").upper(),
        size=13,
        color=COL_TEXT_DIM,
        bold=True,
    )
    if session_label:
        _draw_text(
            draw,
            (margin + 160, margin + 47),
            session_label,
            size=12,
            color=COL_TEXT_DIM,
        )

    # Cadence pill (right top)
    pill_w, pill_h = 110, 30
    pill_x1 = width - margin - 18 - pill_w
    pill_y1 = margin + 12
    _draw_rounded(
        draw,
        (pill_x1, pill_y1, pill_x1 + pill_w, pill_y1 + pill_h),
        radius=8,
        fill=cadence_color,
    )
    _draw_text(
        draw,
        (pill_x1 + pill_w // 2, pill_y1 + pill_h // 2),
        cadence_label,
        size=14,
        bold=True,
        color=(20, 20, 20),
        anchor="mm",
    )

    # Stage chip (right bottom of header)
    if stage:
        stage_txt = stage.upper()
        chip_w = max(80, 14 + 9 * len(stage_txt))
        chip_x1 = width - margin - 18 - chip_w
        chip_y1 = pill_y1 + pill_h + 6
        chip_fill = COL_ACCENT if stage_changed else COL_PANEL_HI
        chip_text_col = (20, 20, 20) if stage_changed else COL_TEXT
        _draw_rounded(
            draw,
            (chip_x1, chip_y1, chip_x1 + chip_w, chip_y1 + 22),
            radius=6,
            fill=chip_fill,
        )
        _draw_text(
            draw,
            (chip_x1 + chip_w // 2, chip_y1 + 11),
            stage_txt,
            size=12,
            bold=True,
            color=chip_text_col,
            anchor="mm",
        )

    # --- Projection card ---
    proj_x = margin
    proj_y = margin + HEADER_H
    pil_proj = _proj_to_pil(proj, (proj_w, proj_h))
    # Subtle outer border around the projection
    _draw_rounded(
        draw,
        (proj_x - 2, proj_y - 2, proj_x + proj_w + 2, proj_y + proj_h + 2),
        radius=6,
        outline=COL_PANEL_HI,
        width=2,
    )
    canvas.paste(pil_proj, (proj_x, proj_y))

    # --- Footer cadence bar ---
    bar_y = proj_y + proj_h + 10
    _draw_rounded(draw, (proj_x, bar_y, proj_x + proj_w, bar_y + 6), radius=3, fill=cadence_color)

    return cv2.cvtColor(np.array(canvas), cv2.COLOR_RGB2BGR)


# ---------------------------------------------------------------------------
# Variable-rate writer (frame-replication)
# ---------------------------------------------------------------------------
@dataclass
class VideoParams:
    fps: int = 30
    speedup: float = 600.0  # real_seconds rendered into 1 video second
    min_frame_dt_s: float = 0.06  # don't flash burst frames
    max_frame_dt_s: float = 0.40  # don't drag slow frames


def _video_dt_for_real_interval(interval_s: float, vp: VideoParams) -> float:
    raw = interval_s / vp.speedup
    return max(vp.min_frame_dt_s, min(vp.max_frame_dt_s, raw))


def _open_writer(path: Path, size: tuple[int, int], fps: int):
    """Try mp4v then avc1 then fall back to AVI codecs (matches video_maker.py)."""
    codecs = (
        ("mp4v", ".mp4"),
        ("avc1", ".mp4"),
        ("XVID", ".avi"),
        ("MJPG", ".avi"),
    )
    w, h = size
    for codec, ext in codecs:
        out_path = path.with_suffix(ext)
        fourcc = cv2.VideoWriter_fourcc(*codec)  # type: ignore[attr-defined]  # cv2 stubs omit this runtime attr
        writer = cv2.VideoWriter(str(out_path), fourcc, fps, (w, h), isColor=True)
        if writer.isOpened():
            return writer, out_path, codec
        writer.release()
    return None, path, None


# ---------------------------------------------------------------------------
# Per-embryo video
# ---------------------------------------------------------------------------
def _embryo_proj_dims() -> tuple[int, int]:
    """2x-scale the cropped projection for crisp text proportion on slides."""
    if VIEW_CROP in ("a", "b"):
        return 1024, 512  # 2x of native 512x256
    return 1024, 256  # native widescreen for both-views


def render_embryo_video(
    emb: EmbryoData,
    out_path: Path,
    vp: VideoParams,
    *,
    session_label: str | None = None,
) -> Path | None:
    proj_w, proj_h = _embryo_proj_dims()
    margin = 28
    canvas_w = proj_w + 2 * margin
    canvas_h = HEADER_H + proj_h + FOOTER_H + 2 * margin

    writer, real_out, codec = _open_writer(out_path, (canvas_w, canvas_h), vp.fps)
    if writer is None:
        logger.error("[%s] no working codec — skipped", emb.embryo_id)
        return None

    t0 = emb.frames[0].acquired_at
    prev_stage: str | None = None
    last_proj: np.ndarray | None = None
    written = 0

    for fr in emb.frames:
        proj = _load_projection_bgr(fr.jpg_path)
        if proj is None:
            logger.warning(
                "[%s] could not read %s — reusing previous",
                emb.embryo_id,
                fr.jpg_path.name,
            )
            if last_proj is None:
                continue
            proj = last_proj
        else:
            last_proj = proj

        stage_changed = bool(fr.predicted_stage and fr.predicted_stage != prev_stage)
        elapsed_s = (fr.acquired_at - t0).total_seconds()

        def render(
            _changed: bool,
            _proj=proj,
            _elapsed=elapsed_s,
            _fr=fr,
        ) -> np.ndarray:
            return _decorate_embryo_frame(
                _proj,
                proj_w=proj_w,
                proj_h=proj_h,
                elapsed_s=_elapsed,
                embryo_id=emb.embryo_id,
                cadence_label=_fr.cadence_label,
                cadence_color=_fr.cadence_color,
                stage=_fr.predicted_stage,
                stage_changed=_changed,
                session_label=session_label,
            )

        video_dt = _video_dt_for_real_interval(fr.interval_to_next_s, vp)
        repeat = max(1, int(round(video_dt * vp.fps)))
        # First frame holds the "STAGE CHANGED" highlight; remaining frames don't.
        decorated_hi = render(stage_changed)
        decorated_lo = render(False) if stage_changed else decorated_hi
        for k in range(repeat):
            writer.write(decorated_hi if k == 0 else decorated_lo)
            written += 1

        if fr.predicted_stage:
            prev_stage = fr.predicted_stage

    writer.release()
    logger.info(
        "[%s] wrote %s (%d video frames, %d input frames, codec=%s)",
        emb.embryo_id,
        real_out,
        written,
        len(emb.frames),
        codec,
    )
    return real_out


# ---------------------------------------------------------------------------
# Burst-only videos
# ---------------------------------------------------------------------------
def render_burst_video(
    burst: BurstData,
    out_path: Path,
    vp: VideoParams,
    *,
    session_label: str | None = None,
) -> Path | None:
    """Render one MP4 for a single burst (request_id) — slower default speedup
    so the embryo actually looks alive."""
    # Reuse the embryo-video composition by wrapping as EmbryoData. The header
    # already shows BURST in the cadence pill since interval_to_next_s < 5 s.
    pseudo = EmbryoData(
        embryo_id=f"{burst.embryo_id}  ·  {burst.request_id}",
        frames=burst.frames,
    )
    return render_embryo_video(pseudo, out_path, vp, session_label=session_label)


# ---------------------------------------------------------------------------
# Grid video (real-time synchronized)
# ---------------------------------------------------------------------------
GRID_COLS_BY_N = {
    1: 1,
    2: 2,
    3: 3,
    4: 2,
    5: 3,
    6: 3,
    7: 4,
    8: 4,
    9: 3,
    10: 5,
    11: 4,
    12: 4,
}


def _grid_layout(n: int) -> tuple[int, int]:
    cols = GRID_COLS_BY_N.get(n, max(1, int(np.ceil(np.sqrt(n)))))
    rows = int(np.ceil(n / cols))
    return rows, cols


def _grid_cell_dims() -> tuple[int, int]:
    """Pick cell projection size so view-A-only and both-views both look right."""
    if VIEW_CROP in ("a", "b"):
        return 540, 270  # 2:1 — matches cropped single view
    return 600, 150  # 4:1 — matches both-views native aspect


CELL_HEADER_H = 38
CELL_FOOTER_H = 6


def _build_global_timeline(embryos: list[EmbryoData]) -> list[datetime]:
    times = set()
    for emb in embryos:
        for fr in emb.frames:
            times.add(fr.acquired_at)
    return sorted(times)


def _most_recent_index(frames: list[FrameRecord], t: datetime) -> int:
    # Linear scan is fine for ~60 frames per embryo.
    idx = -1
    for i, fr in enumerate(frames):
        if fr.acquired_at <= t:
            idx = i
        else:
            break
    return idx


def _draw_cell(
    canvas: Image.Image,
    top_left: tuple[int, int],
    *,
    cell_w: int,
    cell_h: int,
    proj: np.ndarray | None,
    embryo_id: str,
    cadence_color: tuple[int, int, int],
    cadence_label: str,
    stage: str | None,
) -> None:
    """Paint a single cell onto the grid canvas at top_left."""
    x, y = top_left
    total_h = CELL_HEADER_H + cell_h + CELL_FOOTER_H
    draw = ImageDraw.Draw(canvas)

    # Card background + outer border
    _draw_rounded(
        draw,
        (x, y, x + cell_w, y + total_h),
        radius=12,
        fill=COL_PANEL,
        outline=COL_PANEL_HI,
        width=1,
    )

    # Header band
    _draw_text(
        draw,
        (x + 14, y + CELL_HEADER_H // 2),
        embryo_id.replace("_", " ").upper(),
        size=14,
        bold=True,
        color=COL_TEXT,
        anchor="lm",
    )

    if stage:
        _draw_text(
            draw,
            (x + cell_w // 2, y + CELL_HEADER_H // 2),
            stage.upper(),
            size=12,
            bold=True,
            color=COL_ACCENT,
            anchor="mm",
        )

    # Cadence pill (right)
    pill_w, pill_h = 78, 22
    pill_x1 = x + cell_w - 14 - pill_w
    pill_y1 = y + (CELL_HEADER_H - pill_h) // 2
    _draw_rounded(
        draw,
        (pill_x1, pill_y1, pill_x1 + pill_w, pill_y1 + pill_h),
        radius=6,
        fill=cadence_color,
    )
    _draw_text(
        draw,
        (pill_x1 + pill_w // 2, pill_y1 + pill_h // 2),
        cadence_label,
        size=11,
        bold=True,
        color=(20, 20, 20),
        anchor="mm",
    )

    # Projection area
    proj_y = y + CELL_HEADER_H
    if proj is None:
        # "Waiting" placeholder
        _draw_rounded(
            draw,
            (x + 6, proj_y, x + cell_w - 6, proj_y + cell_h),
            radius=6,
            fill=(48, 40, 34),
        )
        _draw_text(
            draw,
            (x + cell_w // 2, proj_y + cell_h // 2),
            "WAITING",
            size=14,
            bold=True,
            color=COL_TEXT_DIM,
            anchor="mm",
        )
    else:
        pil_proj = _proj_to_pil(proj, (cell_w, cell_h))
        canvas.paste(pil_proj, (x, proj_y))

    # Bottom cadence accent
    bar_y = proj_y + cell_h
    _draw_rounded(draw, (x + 6, bar_y, x + cell_w - 6, bar_y + 4), radius=2, fill=cadence_color)


def render_grid_video(
    embryos: list[EmbryoData],
    out_path: Path,
    vp: VideoParams,
    *,
    session_label: str | None = None,
) -> Path | None:
    if not embryos:
        return None
    timeline = _build_global_timeline(embryos)
    if len(timeline) < 2:
        logger.warning("Grid: not enough timepoints for a movie (need ≥ 2)")
        return None

    rows, cols = _grid_layout(len(embryos))
    cell_w, cell_h = _grid_cell_dims()
    cell_total_h = CELL_HEADER_H + cell_h + CELL_FOOTER_H

    grid_w = cols * cell_w + (cols - 1) * GRID_GUTTER
    grid_h = rows * cell_total_h + (rows - 1) * GRID_GUTTER
    canvas_w = grid_w + 2 * GRID_MARGIN
    canvas_h = GRID_TITLE_H + grid_h + 2 * GRID_MARGIN

    writer, real_out, codec = _open_writer(out_path, (canvas_w, canvas_h), vp.fps)
    if writer is None:
        logger.error("Grid: no working codec — skipped")
        return None

    cached_proj: dict[str, np.ndarray | None] = {emb.embryo_id: None for emb in embryos}
    cached_record_idx: dict[str, int] = {emb.embryo_id: -1 for emb in embryos}

    t0 = timeline[0]
    written = 0
    cadence_priority = {"burst": 3, "1min": 2, "5min": 1, "other": 0}

    for tick_idx, t in enumerate(timeline):
        for emb in embryos:
            i = _most_recent_index(emb.frames, t)
            if i < 0:
                continue
            if i != cached_record_idx[emb.embryo_id]:
                proj = _load_projection_bgr(emb.frames[i].jpg_path)
                if proj is not None:
                    cached_proj[emb.embryo_id] = proj
                    cached_record_idx[emb.embryo_id] = i

        canvas = Image.new("RGB", (canvas_w, canvas_h), color=_bgr_to_rgb(COL_BG))
        draw = ImageDraw.Draw(canvas)

        # --- Title bar ---
        title_box = (
            GRID_MARGIN,
            GRID_MARGIN,
            canvas_w - GRID_MARGIN,
            GRID_MARGIN + GRID_TITLE_H - 16,
        )
        _draw_rounded(draw, title_box, radius=12, fill=COL_PANEL, outline=COL_PANEL_HI, width=1)

        elapsed_s = (t - t0).total_seconds()
        _draw_text(
            draw,
            (GRID_MARGIN + 22, GRID_MARGIN + 18),
            _format_elapsed(elapsed_s),
            size=36,
            bold=True,
            color=COL_ACCENT,
        )
        _draw_text(
            draw,
            (GRID_MARGIN + 22, GRID_MARGIN + 58),
            "elapsed",
            size=12,
            color=COL_TEXT_DIM,
            bold=True,
        )
        if session_label:
            _draw_text(
                draw,
                (GRID_MARGIN + 200, GRID_MARGIN + 64),
                session_label,
                size=12,
                color=COL_TEXT_DIM,
            )

        # Active cadence pill (right)
        active = None
        active_pri = -1
        for emb in embryos:
            i = cached_record_idx[emb.embryo_id]
            if i < 0:
                continue
            fr = emb.frames[i]
            if (t - fr.acquired_at).total_seconds() > max(fr.interval_to_next_s, 30) * 1.5:
                continue
            _, color, key = cadence_for_interval(fr.interval_to_next_s)
            pri = cadence_priority.get(key, 0)
            if pri > active_pri:
                active_pri = pri
                active = (fr.cadence_label, fr.cadence_color)
        if active is not None:
            label, color = active
            pill_w, pill_h = 200, 44
            pill_x1 = canvas_w - GRID_MARGIN - 22 - pill_w
            pill_y1 = GRID_MARGIN + (GRID_TITLE_H - 16 - pill_h) // 2
            _draw_rounded(
                draw,
                (pill_x1, pill_y1, pill_x1 + pill_w, pill_y1 + pill_h),
                radius=10,
                fill=color,
            )
            _draw_text(
                draw,
                (pill_x1 + pill_w // 2, pill_y1 + pill_h // 2),
                f"CADENCE  {label}",
                size=16,
                bold=True,
                color=(20, 20, 20),
                anchor="mm",
            )

        # --- Cells ---
        for idx, emb in enumerate(embryos):
            r = idx // cols
            c = idx % cols
            x = GRID_MARGIN + c * (cell_w + GRID_GUTTER)
            y = GRID_MARGIN + GRID_TITLE_H + r * (cell_total_h + GRID_GUTTER)

            i = cached_record_idx[emb.embryo_id]
            if i < 0:
                _draw_cell(
                    canvas,
                    (x, y),
                    cell_w=cell_w,
                    cell_h=cell_h,
                    proj=None,
                    embryo_id=emb.embryo_id,
                    cadence_color=COL_PANEL_HI,
                    cadence_label="--",
                    stage=None,
                )
            else:
                fr = emb.frames[i]
                _draw_cell(
                    canvas,
                    (x, y),
                    cell_w=cell_w,
                    cell_h=cell_h,
                    proj=cached_proj[emb.embryo_id],
                    embryo_id=emb.embryo_id,
                    cadence_color=fr.cadence_color,
                    cadence_label=fr.cadence_label,
                    stage=fr.predicted_stage,
                )

        frame_bgr = cv2.cvtColor(np.array(canvas), cv2.COLOR_RGB2BGR)

        if tick_idx < len(timeline) - 1:
            interval_s = (timeline[tick_idx + 1] - t).total_seconds()
        else:
            interval_s = 300.0
        video_dt = _video_dt_for_real_interval(interval_s, vp)
        repeat = max(1, int(round(video_dt * vp.fps)))
        for _ in range(repeat):
            writer.write(frame_bgr)
            written += 1

    writer.release()
    logger.info(
        "Grid: wrote %s (%d video frames, %d timeline ticks, codec=%s)",
        real_out,
        written,
        len(timeline),
        codec,
    )
    return real_out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--session",
        default=None,
        help="Session folder name or partial id (default: latest)",
    )
    parser.add_argument(
        "--storage",
        default="D:/Gently3",
        help="Gently storage root (default: D:/Gently3)",
    )
    parser.add_argument("--out", default=None, help="Output directory (default: <session>/_recap)")
    parser.add_argument(
        "--speedup",
        type=float,
        default=600.0,
        help="real_seconds rendered into one video second (default: 600)",
    )
    parser.add_argument("--fps", type=int, default=30, help="Playback FPS (default: 30)")
    parser.add_argument(
        "--min-frame-dt",
        type=float,
        default=0.06,
        help="Min seconds per input frame (default: 0.06)",
    )
    parser.add_argument(
        "--max-frame-dt",
        type=float,
        default=0.40,
        help="Max seconds per input frame (default: 0.40)",
    )
    parser.add_argument(
        "--no-perception", action="store_true", help="Skip predicted-stage overlays"
    )
    parser.add_argument("--no-grid", action="store_true", help="Skip the synchronized grid video")
    parser.add_argument("--no-per-embryo", action="store_true", help="Skip per-embryo videos")
    parser.add_argument(
        "--bursts",
        action="store_true",
        help="Also render one MP4 per burst (one per request_id)",
    )
    parser.add_argument(
        "--bursts-only",
        action="store_true",
        help="Render burst MP4s only — skip timelapse + grid",
    )
    parser.add_argument(
        "--burst-speedup",
        type=float,
        default=10.0,
        help="Speedup for burst videos only (default: 10 — near-real-time)",
    )
    parser.add_argument(
        "--view",
        choices=("a", "b", "both"),
        default="a",
        help="Which diSPIM view to render (default: a)",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args(argv)

    global VIEW_CROP
    VIEW_CROP = args.view

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    storage = Path(args.storage)
    session_dir = find_session(storage, args.session)
    out_dir = Path(args.out) if args.out else session_dir / "_recap"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Session: %s", session_dir)
    logger.info("Output:  %s", out_dir)
    logger.info("View:    %s  |  Speedup: %.0fx real-time", VIEW_CROP, args.speedup)
    session_label = session_dir.name

    embryo_dirs = sorted((session_dir / "embryos").glob("embryo_*"))
    if not embryo_dirs:
        logger.error("No embryos found under %s/embryos", session_dir)
        return 1

    embryos: list[EmbryoData] = []
    for ed in embryo_dirs:
        emb = load_embryo(ed, use_perception=not args.no_perception)
        if emb is None:
            logger.warning("Skipping %s (no projections)", ed.name)
            continue
        logger.info(
            "Loaded %s: %d frames (%s → %s)",
            emb.embryo_id,
            len(emb.frames),
            emb.frames[0].acquired_at.isoformat(timespec="seconds"),
            emb.frames[-1].acquired_at.isoformat(timespec="seconds"),
        )
        embryos.append(emb)

    if not embryos:
        logger.error("No usable embryo data found")
        return 1

    vp = VideoParams(
        fps=args.fps,
        speedup=args.speedup,
        min_frame_dt_s=args.min_frame_dt,
        max_frame_dt_s=args.max_frame_dt,
    )

    render_timelapse = not args.bursts_only
    if render_timelapse and not args.no_per_embryo:
        for emb in embryos:
            render_embryo_video(
                emb, out_dir / f"{emb.embryo_id}.mp4", vp, session_label=session_label
            )

    if render_timelapse and not args.no_grid:
        render_grid_video(embryos, out_dir / "grid.mp4", vp, session_label=session_label)

    if args.bursts or args.bursts_only:
        bursts = discover_bursts(session_dir)
        if not bursts:
            logger.warning("No bursts found under %s/embryos/*/bursts/", session_dir)
        else:
            burst_out_dir = out_dir / "bursts"
            burst_out_dir.mkdir(parents=True, exist_ok=True)
            burst_vp = VideoParams(
                fps=vp.fps,
                speedup=args.burst_speedup,
                min_frame_dt_s=vp.min_frame_dt_s,
                max_frame_dt_s=max(vp.max_frame_dt_s, 0.6),
            )
            logger.info(
                "Rendering %d burst(s) at %.0fx speedup",
                len(bursts),
                args.burst_speedup,
            )
            for burst in bursts:
                # request_id already contains the embryo_id; uuid8 suffix keeps it unique.
                out_name = f"{burst.request_id}.mp4"
                render_burst_video(
                    burst,
                    burst_out_dir / out_name,
                    burst_vp,
                    session_label=session_label,
                )

    logger.info("Done. Videos written to %s", out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
