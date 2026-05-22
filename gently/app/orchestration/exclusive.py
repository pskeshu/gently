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

        target_dt = 1.0 if self.mode == "1hz" else 0.0
        start = datetime.now()

        # Phase 10 dedicated burst events (Phase 7 originally rode
        # STATUS_CHANGED; now we have first-class types).
        orchestrator._emit_event(EventType.BURST_START, {
            "embryo_id": self.target_embryo_id,
            "request_id": self.request_id,
            "frames": self.frames,
            "mode": self.mode,
        })

        # Collect captured volumes for MP4 generation at the end.
        frames_captured: List[np.ndarray] = []
        frame_paths: List[str] = []

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

        # Tight acquisition loop.
        loop_start = datetime.now()
        for i in range(self.frames):
            tick_start = datetime.now()
            try:
                result = await orchestrator.client.acquire_volume(
                    num_slices=self.num_slices,
                    exposure_ms=embryo.exposure_ms,
                    galvo_amplitude=galvo_amplitude,
                    galvo_center=galvo_center,
                    piezo_amplitude=piezo_amplitude,
                    piezo_center=piezo_center,
                    laser_power_488_pct=getattr(embryo, "laser_power_488_pct", None),
                )
            except Exception as e:
                logger.error("Burst frame %d failed for %s: %s", i, self.target_embryo_id, e)
                continue

            vol = result.get("volume")
            if vol is not None:
                frames_captured.append(np.asarray(vol))
            vp = result.get("volume_path")
            if vp:
                frame_paths.append(str(vp))

            # Periodic event for UI / persistence (every 5th frame to keep noise down).
            if i % 5 == 0:
                orchestrator._emit_event(EventType.BURST_FRAME, {
                    "embryo_id": self.target_embryo_id,
                    "request_id": self.request_id,
                    "frame_idx": i,
                    "total_frames": self.frames,
                })

            # Cadence pacing.
            if target_dt > 0:
                elapsed = (datetime.now() - tick_start).total_seconds()
                wait = target_dt - elapsed
                if wait > 0:
                    await asyncio.sleep(wait)

        duration_s = (datetime.now() - loop_start).total_seconds()
        sustained_hz = (len(frames_captured) / duration_s) if duration_s > 0 else 0.0

        # Generate MP4 from captured frames (best-effort).
        mp4_path = await _maybe_generate_mp4(
            orchestrator=orchestrator,
            embryo_id=self.target_embryo_id,
            request_id=self.request_id,
            frames=frames_captured,
        )

        orchestrator._emit_event(EventType.BURST_COMPLETE, {
            "embryo_id": self.target_embryo_id,
            "request_id": self.request_id,
            "frames_captured": len(frames_captured),
            "duration_s": duration_s,
            "sustained_hz": sustained_hz,
            "mp4_path": mp4_path,
        })

        return ExclusiveResult(
            success=True,
            target_embryo_id=self.target_embryo_id,
            request_id=self.request_id,
            frames_captured=len(frames_captured),
            duration_s=duration_s,
            output_path=mp4_path,
            extra={"sustained_hz": sustained_hz, "mode": self.mode},
        )


async def _maybe_generate_mp4(
    *,
    orchestrator,
    embryo_id: str,
    request_id: str,
    frames: List[np.ndarray],
) -> Optional[str]:
    """Best-effort MP4 generation. Returns None if no frames or imageio
    isn't available — the burst still succeeds either way."""
    if not frames:
        return None
    if not (getattr(orchestrator, "_store", None) and getattr(orchestrator, "_session_id", None)):
        return None

    try:
        from pathlib import Path
        burst_dir = (
            orchestrator._store.root / "sessions" / orchestrator._session_id
            / "embryos" / embryo_id / "bursts"
        )
        burst_dir.mkdir(parents=True, exist_ok=True)
        mp4_path = burst_dir / f"{request_id}.mp4"

        # Reduce 3D frames to 2D max-projections for the movie.
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
            proj_frames.append(v)

        if not proj_frames:
            return None

        try:
            import imageio.v2 as imageio
        except ImportError:
            import imageio  # type: ignore
        imageio.mimwrite(str(mp4_path), proj_frames, fps=10)
        logger.info("Wrote burst MP4: %s (%d frames)", mp4_path, len(proj_frames))
        return str(mp4_path)
    except Exception as e:
        logger.warning("MP4 generation failed: %s", e)
        return None
