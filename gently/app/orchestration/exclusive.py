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
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ExclusiveResult:
    """Outcome of an ExclusiveAcquisition.run()."""
    success: bool
    target_embryo_id: str
    request_id: str
    frames_captured: int = 0
    duration_s: float = 0.0
    output_path: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


class ExclusiveAcquisition(ABC):
    """Base class for operations that pause the global queue.

    Subclasses must implement ``run(orchestrator) -> ExclusiveResult``.
    The orchestrator handles setting / clearing ``_burst_in_progress`` and
    pausing / resuming other embryos around the call.
    """

    #: Human-readable kind name, for events / UI / persistence.
    kind: str = "exclusive"

    def __init__(self, target_embryo_id: str, request_id: Optional[str] = None):
        self.target_embryo_id = target_embryo_id
        self.request_id = request_id or _make_request_id(self.kind, target_embryo_id)

    @abstractmethod
    async def run(self, orchestrator) -> ExclusiveResult:
        ...


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
        request_id: Optional[str] = None,
    ):
        super().__init__(target_embryo_id=target_embryo_id, request_id=request_id)
        self.frames = frames
        self.mode = mode if mode in ("1hz", "asap") else "1hz"
        self.num_slices = num_slices

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
        orchestrator._emit_event(EventType.BURST_START, {
            "embryo_id": self.target_embryo_id,
            "request_id": self.request_id,
            "frames": self.frames,
            "mode": self.mode,
        })

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
        frames_captured: List[np.ndarray] = [
            np.asarray(f["volume"]) for f in frames_data if f.get("volume") is not None
        ]

        # Prefer the plan's measured timing; fall back to wall-clock if absent.
        duration_s = float(result.get("duration_s") or 0.0)
        if duration_s <= 0:
            duration_s = (datetime.now() - loop_start).total_seconds()
        sustained_hz = float(result.get("sustained_hz") or 0.0)
        if sustained_hz <= 0 and duration_s > 0:
            sustained_hz = len(frames_captured) / duration_s

        # Generate MP4 from captured frames (best-effort).
        mp4_path = await _maybe_generate_mp4(
            orchestrator=orchestrator,
            embryo_id=self.target_embryo_id,
            request_id=self.request_id,
            frames=frames_captured,
        )

        success = bool(result.get("success")) and len(frames_captured) > 0

        orchestrator._emit_event(EventType.BURST_COMPLETE, {
            "embryo_id": self.target_embryo_id,
            "request_id": self.request_id,
            "frames_captured": len(frames_captured),
            "duration_s": duration_s,
            "sustained_hz": sustained_hz,
            "mp4_path": mp4_path,
        })

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
            orchestrator._emit_event(frame_event_type, {
                "embryo_id": self.target_embryo_id,
                "request_id": self.request_id,
                "frame_idx": approx_idx,
                "total_frames": self.frames,
                "approximate": True,
            })


async def _maybe_generate_mp4(
    *,
    orchestrator,
    embryo_id: str,
    request_id: str,
    frames: List[np.ndarray],
) -> Optional[str]:
    """Best-effort MP4 generation using OpenCV's VideoWriter.

    Mirrors the codec-fallback pattern in :mod:`gently.app.video_maker`
    (mp4v → avc1 → XVID → MJPG). Returns ``None`` if no frames, no codec
    opens, or the session isn't backed by a store — the burst still
    succeeds either way.
    """
    if not frames:
        return None
    if not (getattr(orchestrator, "_store", None) and getattr(orchestrator, "_session_id", None)):
        return None

    try:
        import cv2  # type: ignore
    except ImportError:
        logger.warning("MP4 generation skipped: cv2 not available")
        return None

    try:
        from pathlib import Path
        burst_dir = (
            orchestrator._store.root / "sessions" / orchestrator._session_id
            / "embryos" / embryo_id / "bursts"
        )
        burst_dir.mkdir(parents=True, exist_ok=True)
        mp4_path = burst_dir / f"{request_id}.mp4"

        # Reduce 3D frames to 2D max-projections, normalize to uint8, and
        # convert to 3-channel BGR for VideoWriter.
        proj_frames: List[np.ndarray] = []
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
            ('mp4v', '.mp4'),
            ('avc1', '.mp4'),
            ('XVID', '.avi'),
            ('MJPG', '.avi'),
        )
        writer = None
        chosen_codec = None
        chosen_path = mp4_path
        for codec, ext in codecs:
            test_path = mp4_path.with_suffix(ext)
            fourcc = cv2.VideoWriter_fourcc(*codec)
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
            chosen_path, len(proj_frames), chosen_codec,
        )
        return str(chosen_path)
    except Exception as e:
        logger.warning("MP4 generation failed: %s", e)
        return None
