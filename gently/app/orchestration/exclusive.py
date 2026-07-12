"""
Exclusive acquisition primitives — operations that take over the whole
acquisition queue while they run.

Burst (60 frames @ 1Hz / ASAP per TestEmbryo) is the first instance.
Future: long-exposure single frames, recalibration mid-run, focus sweeps,
photoactivation, FRAP.

While an ExclusiveAcquisition is in flight, the orchestrator's normal
per-embryo due loop yields the wheel and won't acquire from other
embryos. Their ``next_due_at`` keeps advancing — they may end up
"overdue" — and the priority queue picks them up ASAP after the
exclusive op completes.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from gently.app.temperature_sampler import temperature_stamp

logger = logging.getLogger(__name__)


@dataclass
class ExclusiveResult:
    """Outcome of an ExclusiveAcquisition.run()."""

    success: bool
    target_embryo_id: str
    request_id: str
    frames_captured: int = 0
    duration_s: float = 0.0
    output_path: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)
    error: str | None = None


class ExclusiveAcquisition(ABC):
    """Base class for operations that pause the global queue.

    Subclasses must implement ``run(orchestrator) -> ExclusiveResult``.
    The orchestrator handles setting / clearing ``_burst_in_progress`` and
    pausing / resuming other embryos around the call.
    """

    #: Human-readable kind name, for events / UI / persistence.
    kind: str = "exclusive"

    def __init__(self, target_embryo_id: str, request_id: str | None = None):
        self.target_embryo_id = target_embryo_id
        self.request_id = request_id or _make_request_id(self.kind, target_embryo_id)

    @abstractmethod
    async def run(self, orchestrator) -> ExclusiveResult: ...


def _make_request_id(kind: str, embryo_id: str) -> str:
    import uuid

    return f"{kind}_{embryo_id}_{uuid.uuid4().hex[:8]}"


# ----------------------------------------------------------------------
# BurstAcquisition — the experiment's 60-frame movie at 1 Hz / ASAP
# ----------------------------------------------------------------------


class BurstAcquisition(ExclusiveAcquisition):
    """One-embryo burst: ``frames`` rapid acquisitions at ``mode`` cadence.

    Parameters
    ----------
    target_embryo_id : str
        Which embryo to image. Must be in ``orchestrator._embryo_states``.
    frames : int
        Total frames to capture (default 60).
    mode : "1hz" | "asap"
        Cadence policy. ``1hz`` sleeps to the next 1-second tick;
        ``asap`` chains acquisitions back-to-back as fast as the
        hardware can deliver.
    num_slices : int
        Per-frame z-slices. Default 1 (snap mode is the fastest way to
        approach 1 Hz on this hardware).
    """

    kind = "burst"

    def __init__(
        self,
        target_embryo_id: str,
        *,
        frames: int = 60,
        mode: str = "1hz",
        num_slices: int = 1,
        request_id: str | None = None,
        temperature_provider=None,
        laser_config: str | None = None,
        tactic_id: str | None = None,
    ):
        super().__init__(target_embryo_id=target_embryo_id, request_id=request_id)
        self.frames = frames
        self.mode = mode if mode in ("1hz", "asap") else "1hz"
        self.num_slices = num_slices
        self._temperature_provider = temperature_provider
        self._laser_config = laser_config
        self._tactic_id = tactic_id

    async def run(self, orchestrator) -> ExclusiveResult:
        from gently.core import EventType

        embryo = orchestrator._embryo_states.get(self.target_embryo_id)
        if embryo is None:
            return ExclusiveResult(
                success=False,
                target_embryo_id=self.target_embryo_id,
                request_id=self.request_id,
                error=f"Embryo {self.target_embryo_id!r} not in active timelapse",
            )

        # Phase 10 dedicated burst events (Phase 7 originally rode
        # STATUS_CHANGED; now we have first-class types).
        orchestrator._emit_event(
            EventType.BURST_START,
            {
                "embryo_id": self.target_embryo_id,
                "request_id": self.request_id,
                "frames": self.frames,
                "mode": self.mode,
                "phase": getattr(self, "_phase", None),
                "tactic_id": self._tactic_id,
            },
        )

        # Hardware kwargs from the embryo's calibration (mirrors _acquire_embryo)
        cal = embryo.calibration or {}
        galvo_amplitude = cal.get("galvo_amplitude", 0.5)
        galvo_center = cal.get("galvo_center", 0.0)
        piezo_amplitude = cal.get("piezo_amplitude", 25.0)
        piezo_center = cal.get("piezo_center", 50.0)

        # Move to embryo position once at the start of the burst.
        pos = embryo.stage_position or {}
        if pos.get("x") is not None and pos.get("y") is not None:
            try:
                await orchestrator.client.move_to_position(pos["x"], pos["y"])
            except Exception as e:
                logger.warning("Burst move-to failed for %s: %s", self.target_embryo_id, e)

        # Single device-layer plan holds the MMCore lock and pause_state_updates
        # for the entire burst, eliminating the per-frame state-poller race.
        # Progress events are approximated locally on a timer (see _progress_ticker)
        # because the plan only returns when all frames are done.
        loop_start = datetime.now()
        progress_task = asyncio.create_task(
            self._progress_ticker(orchestrator, EventType.BURST_FRAME)
        )
        try:
            result = await orchestrator.client.acquire_burst(
                frames=self.frames,
                mode=self.mode,
                num_slices=self.num_slices,
                exposure_ms=embryo.exposure_ms,
                galvo_amplitude=galvo_amplitude,
                galvo_center=galvo_center,
                piezo_amplitude=piezo_amplitude,
                piezo_center=piezo_center,
                laser_power_488_pct=getattr(embryo, "laser_power_488_pct", None),
                laser_config=self._laser_config,
            )
        except Exception as e:
            logger.error("Burst failed for %s: %s", self.target_embryo_id, e)
            result = {"success": False, "error": str(e), "frames": []}
        finally:
            progress_task.cancel()
            try:
                await progress_task
            except (asyncio.CancelledError, Exception):
                pass

        frames_data = result.get("frames") or []
        frames_captured: list[np.ndarray] = [
            np.asarray(f["volume"]) for f in frames_data if f.get("volume") is not None
        ]

        # Prefer the plan's measured timing; fall back to wall-clock if absent.
        duration_s = float(result.get("duration_s") or 0.0)
        if duration_s <= 0:
            duration_s = (datetime.now() - loop_start).total_seconds()
        sustained_hz = float(result.get("sustained_hz") or 0.0)
        if sustained_hz <= 0 and duration_s > 0:
            sustained_hz = len(frames_captured) / duration_s

        # Persist per-frame TIFFs + projections + manifest BEFORE the MP4 attempt
        # so a codec hiccup can't lose the raw burst data.
        burst_dir = _persist_burst_to_disk(
            orchestrator=orchestrator,
            embryo=embryo,
            embryo_id=self.target_embryo_id,
            request_id=self.request_id,
            mode=self.mode,
            frames_requested=self.frames,
            frames_data=frames_data,
            loop_start=loop_start,
            duration_s=duration_s,
            sustained_hz=sustained_hz,
            galvo_amplitude=galvo_amplitude,
            galvo_center=galvo_center,
            piezo_amplitude=piezo_amplitude,
            piezo_center=piezo_center,
            laser_power_488_pct=getattr(embryo, "laser_power_488_pct", None),
            temperature_provider=self._temperature_provider,
        )

        # Generate MP4 (derivative artifact; safe to fail).
        mp4_path = await _maybe_generate_mp4(
            burst_dir=burst_dir,
            embryo_id=self.target_embryo_id,
            request_id=self.request_id,
            frames=frames_captured,
        )

        success = bool(result.get("success")) and len(frames_captured) > 0

        orchestrator._emit_event(
            EventType.BURST_COMPLETE,
            {
                "embryo_id": self.target_embryo_id,
                "request_id": self.request_id,
                "frames_captured": len(frames_captured),
                "duration_s": duration_s,
                "sustained_hz": sustained_hz,
                "mp4_path": mp4_path,
                "tactic_id": self._tactic_id,
            },
        )

        return ExclusiveResult(
            success=success,
            target_embryo_id=self.target_embryo_id,
            request_id=self.request_id,
            frames_captured=len(frames_captured),
            duration_s=duration_s,
            output_path=mp4_path,
            extra={"sustained_hz": sustained_hz, "mode": self.mode},
            error=result.get("error") if not success else None,
        )

    async def _progress_ticker(self, orchestrator, frame_event_type):
        """Approximate per-frame progress for the UI while the plan runs.

        The device-layer plan returns only when all frames are done, so we
        can't observe per-frame completion here. For 1 Hz mode the cadence
        is known; for ASAP we tick at 1 s as a best-guess. Cancelled by
        ``run`` once the plan returns.
        """
        target_dt = 1.0  # 1 s tick for both 1hz and asap modes
        tick_interval = 5.0  # match the old "every 5th frame" cadence
        start = datetime.now()
        while True:
            await asyncio.sleep(tick_interval)
            elapsed = (datetime.now() - start).total_seconds()
            approx_idx = min(self.frames - 1, int(elapsed / target_dt))
            orchestrator._emit_event(
                frame_event_type,
                {
                    "embryo_id": self.target_embryo_id,
                    "request_id": self.request_id,
                    "frame_idx": approx_idx,
                    "total_frames": self.frames,
                    "approximate": True,
                },
            )


def _resolve_burst_dir(orchestrator, embryo_id: str, request_id: str) -> Path | None:
    """Return ``bursts/{request_id}/`` under the embryo's session folder.

    Uses ``FileStore._session_dir`` so the short session_id resolves to the full
    ``YYYYMMDD_HHMM_slug_id8`` folder — historic bursts wrote to a shadow
    ``sessions/{short_id}/`` path because of a direct ``root/sessions/{id}``
    join here.
    """
    store = getattr(orchestrator, "_store", None)
    sid = getattr(orchestrator, "_session_id", None)
    if store is None or sid is None:
        return None
    session_dir: Path | None = None
    for attr in ("_session_dir", "session_dir"):
        fn = getattr(store, attr, None)
        if callable(fn):
            try:
                session_dir = fn(sid)
            except Exception:
                session_dir = None
            if session_dir is not None:
                break
    if session_dir is None:
        # Last-resort fallback: previous behaviour (will write to the shadow folder).
        logger.warning(
            "FileStore has no session_dir resolver; falling back to root/sessions/%s",
            sid,
        )
        session_dir = store.root / "sessions" / sid

    burst_dir = session_dir / "embryos" / embryo_id / "bursts" / request_id
    burst_dir.mkdir(parents=True, exist_ok=True)
    return burst_dir


def _persist_burst_to_disk(
    *,
    orchestrator,
    embryo,
    embryo_id: str,
    request_id: str,
    mode: str,
    frames_requested: int,
    frames_data: list[dict[str, Any]],
    loop_start: datetime,
    duration_s: float,
    sustained_hz: float,
    galvo_amplitude: float,
    galvo_center: float,
    piezo_amplitude: float,
    piezo_center: float,
    laser_power_488_pct: float | None,
    temperature_provider=None,
) -> Path | None:
    """Save per-frame TIFFs + meta + projections + a burst.yaml manifest.

    Best-effort: any per-frame failure is logged and skipped, the rest still
    gets written. Returns the burst directory (or ``None`` if it could not be
    resolved).
    """
    burst_dir = _resolve_burst_dir(orchestrator, embryo_id, request_id)
    if burst_dir is None:
        logger.warning("Burst persistence skipped: no store/session bound")
        return None

    try:
        import tifffile
    except ImportError:
        logger.warning("Burst persistence skipped: tifffile not available")
        return burst_dir

    try:
        import yaml as _yaml
    except ImportError:
        _yaml = None

    proj_dir = burst_dir / "projections"
    proj_dir.mkdir(exist_ok=True)

    # Compute temperature stamp once for the whole burst (all frames share the
    # same reading — the sampler captures at ~1 Hz so per-frame variation is
    # sub-resolution anyway).
    _temp = temperature_stamp(temperature_provider() if temperature_provider else None)

    # Position recorded for the manifest.
    pos = getattr(embryo, "stage_position", {}) or {}
    sid = getattr(orchestrator, "_session_id", None)

    saved_frames: list[dict[str, Any]] = []
    for i, fr in enumerate(frames_data, start=1):
        vol = fr.get("volume")
        if vol is None:
            continue
        arr = np.asarray(vol)
        tif_path = burst_dir / f"t{i:04d}.tif"
        meta_path = burst_dir / f"t{i:04d}.meta.yaml"
        proj_path = proj_dir / f"t{i:04d}.jpg"

        # Real per-frame acquisition time from the Bluesky event doc; fall back
        # to even spacing across the measured burst duration.
        epoch = fr.get("acquired_at_epoch")
        if epoch is not None:
            acquired_at = datetime.fromtimestamp(float(epoch)).isoformat()
        else:
            if duration_s > 0 and len(frames_data) > 1:
                offset = (i - 1) * duration_s / max(1, len(frames_data) - 1)
            else:
                offset = (i - 1) * (1.0 if mode == "1hz" else 0.0)
            acquired_at = datetime.fromtimestamp(loop_start.timestamp() + offset).isoformat()

        try:
            tifffile.imwrite(str(tif_path), arr)
        except Exception as exc:
            logger.warning("[%s] burst frame %d TIFF write failed: %s", embryo_id, i, exc)
            continue

        # Projection via the same helper used for regular volumes.
        try:
            from gently.core.imaging import generate_jpeg_projection

            generate_jpeg_projection(arr, proj_path)
        except Exception as exc:
            logger.debug("[%s] burst frame %d projection failed: %s", embryo_id, i, exc)

        meta = {
            "session_id": sid,
            "embryo_id": embryo_id,
            "request_id": request_id,
            "frame_index": i,
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
            "acquired_at": acquired_at,
            "metadata": {
                "num_slices": int(getattr(embryo, "num_slices", 1))
                if hasattr(embryo, "num_slices")
                else None,
                "exposure_ms": float(getattr(embryo, "exposure_ms", 0.0))
                if hasattr(embryo, "exposure_ms")
                else None,
                "acquisition_mode": "burst",
                "burst_mode": mode,
                "laser_power_488_pct": laser_power_488_pct,
                "role": "burst",
                "temperature": _temp,
            },
        }
        if _yaml is not None:
            try:
                with meta_path.open("w", encoding="utf-8") as f:
                    _yaml.safe_dump(meta, f, sort_keys=False)
            except Exception as exc:
                logger.debug("[%s] burst frame %d meta write failed: %s", embryo_id, i, exc)
        saved_frames.append(
            {
                "frame_index": i,
                "tif": tif_path.name,
                "projection": f"projections/{proj_path.name}",
                "acquired_at": acquired_at,
            }
        )

    manifest = {
        "request_id": request_id,
        "session_id": sid,
        "embryo_id": embryo_id,
        "mode": mode,
        "frames_requested": frames_requested,
        "frames_captured": len(saved_frames),
        "started_at": loop_start.isoformat(),
        "duration_s": duration_s,
        "sustained_hz": sustained_hz,
        "embryo_position": {"x": pos.get("x"), "y": pos.get("y")},
        "laser_power_488_pct": laser_power_488_pct,
        "temperature": _temp,
        "scan": {
            "galvo_amplitude": galvo_amplitude,
            "galvo_center": galvo_center,
            "piezo_amplitude": piezo_amplitude,
            "piezo_center": piezo_center,
        },
        "frames": saved_frames,
    }
    if _yaml is not None:
        try:
            with (burst_dir / "burst.yaml").open("w", encoding="utf-8") as f:
                _yaml.safe_dump(manifest, f, sort_keys=False)
        except Exception as exc:
            logger.warning("[%s] burst manifest write failed: %s", embryo_id, exc)

    logger.info(
        "[%s] persisted %d/%d burst frames -> %s",
        embryo_id,
        len(saved_frames),
        frames_requested,
        burst_dir,
    )
    return burst_dir


async def _maybe_generate_mp4(
    *,
    burst_dir: Path | None,
    embryo_id: str,
    request_id: str,
    frames: list[np.ndarray],
) -> str | None:
    """Best-effort MP4 generation using OpenCV's VideoWriter.

    Mirrors the codec-fallback pattern in :mod:`gently.app.video_maker`
    (mp4v → avc1 → XVID → MJPG). Returns ``None`` if no frames, no codec
    opens, or the burst directory wasn't resolved — the burst still
    succeeds either way.
    """
    if not frames or burst_dir is None:
        return None

    try:
        import cv2  # type: ignore
    except ImportError:
        logger.warning("MP4 generation skipped: cv2 not available")
        return None

    try:
        mp4_path = burst_dir / "burst.mp4"

        # Reduce 3D frames to 2D max-projections, normalize to uint8, and
        # convert to 3-channel BGR for VideoWriter.
        proj_frames: list[np.ndarray] = []
        for f in frames:
            v = np.squeeze(f)
            if v.ndim == 4:
                v = v[0]
            if v.ndim == 3:
                v = np.max(v, axis=0)
            if v.ndim != 2:
                continue
            if v.dtype != np.uint8:
                lo, hi = float(v.min()), float(v.max())
                if hi > lo:
                    v = ((v.astype(np.float32) - lo) / (hi - lo) * 255).astype(np.uint8)
                else:
                    v = np.zeros_like(v, dtype=np.uint8)
            proj_frames.append(cv2.cvtColor(v, cv2.COLOR_GRAY2BGR))

        if not proj_frames:
            return None

        height, width = proj_frames[0].shape[:2]
        codecs = (
            ("mp4v", ".mp4"),
            ("avc1", ".mp4"),
            ("XVID", ".avi"),
            ("MJPG", ".avi"),
        )
        writer = None
        chosen_codec = None
        chosen_path = mp4_path
        for codec, ext in codecs:
            test_path = mp4_path.with_suffix(ext)
            fourcc = cv2.VideoWriter_fourcc(*codec)  # type: ignore[attr-defined]  # cv2 stubs omit VideoWriter_fourcc
            w = cv2.VideoWriter(str(test_path), fourcc, 10, (width, height), isColor=True)
            if w.isOpened():
                writer = w
                chosen_codec = codec
                chosen_path = test_path
                break
            w.release()

        if writer is None:
            logger.warning("MP4 generation skipped: no working codec (tried mp4v/avc1/XVID/MJPG)")
            return None

        for frame in proj_frames:
            writer.write(frame)
        writer.release()
        logger.info(
            "Wrote burst movie: %s (%d frames, codec=%s)",
            chosen_path,
            len(proj_frames),
            chosen_codec,
        )
        return str(chosen_path)
    except Exception as e:
        logger.warning("MP4 generation failed: %s", e)
        return None
