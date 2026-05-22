"""
Timelapse Orchestrator for Adaptive Multi-Embryo Acquisition

Manages background timelapse acquisition with:
- Per-embryo stop conditions (e.g., hatching detection)
- Dynamic interval adjustment
- Non-blocking operation (agent stays responsive)
- Event-driven status updates
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, TYPE_CHECKING
import traceback

import numpy as np

from gently.core import EventType, get_event_bus
from gently.core.imaging import (
    projection_three_view,
    compute_crop_bounds,
    apply_crop_bounds,
    image_to_base64,
    normalize_to_uint8,
)
from gently.settings import settings
from gently.harness.error_log import GlobalErrorLog
from gently.organisms import get_organism

# Re-export models for backward compatibility
from .timelapse_models import (
    StopConditionType,
    IntervalRule,
    StopCondition,
    TimelapseStatus,
    TimelapseState,
)
from gently.harness.state import EmbryoState

if TYPE_CHECKING:
    from gently.core.file_store import FileStore

# Default trace directory
TRACE_BASE_PATH = settings.storage.traces_dir

logger = logging.getLogger(__name__)


class TimelapseOrchestrator:
    """
    Background timelapse manager

    Runs independently of agent conversation. The agent can:
    - Query status anytime via get_status()
    - Modify parameters via modify_embryo()
    - Stop embryos or entire timelapse via stop()

    Events are emitted for UI updates without blocking.
    """

    def __init__(
        self,
        microscope_client,
        experiment_state,
        perceiver=None,
        on_volume_callback: Optional[Callable] = None,
        session_id: Optional[str] = None,
        store: Optional["FileStore"] = None,
    ):
        """
        Parameters
        ----------
        microscope_client : QueueServerClient
            Client for hardware control
        experiment_state : ExperimentState
            Shared experiment state
        perceiver : gently_perception.Perceiver, optional
            VLM-based perception system for stage classification
        on_volume_callback : callable, optional
            Called after each volume: on_volume_callback(embryo_id, timepoint, volume)
        session_id : str, optional
            Session identifier for trace file storage
        store : FileStore, optional
            Unified data store for persisting perception predictions
        """
        self.client = microscope_client
        self.experiment = experiment_state
        self.perceiver = perceiver
        self.on_volume_callback = on_volume_callback

        # Trace file storage (writes JSON files to disk)
        self._session_id = session_id
        self._trace_dir: Optional[Path] = None

        # Unified store for perception persistence
        self._store = store
        self._perception_run_id: Optional[int] = None

        # Event bus for status updates
        self._event_bus = get_event_bus()

        # Timelapse state
        # Holds references to EmbryoState objects from self.experiment.embryos
        # for embryos currently active in this timelapse. Same objects, no
        # duplication of state.
        self._embryo_states: Dict[str, EmbryoState] = {}
        self._status = TimelapseStatus.IDLE
        self._started_at: Optional[datetime] = None
        self._total_timepoints = 0
        self._current_round = 0
        self._error_message: Optional[str] = None

        # Round-based scheduling (global timing for all embryos)
        self._base_interval_seconds: float = 120.0
        self._total_pause_duration: timedelta = timedelta(0)
        self._pause_start: Optional[datetime] = None

        # Control
        self._acquisition_task: Optional[asyncio.Task] = None
        self._stop_requested = False

        # In-flight perception tasks. Acquisition fires perception tasks
        # as create_task() and moves on to the next embryo immediately, so
        # round-to-round throughput is bounded by volume acquisition time,
        # not VLM API latency. Tasks are tracked here so (a) stop() can
        # await them before returning and (b) we can cap concurrency if
        # the Claude API rate limit becomes a problem.
        self._perception_tasks: Set[asyncio.Task] = set()

        # Interval adjustment rules
        self._interval_rules: List[IntervalRule] = []
        self._applied_rules: Dict[str, Set[str]] = {}  # embryo_id -> set of applied rule names

        # Async cadence state (Phase 4). Single-burst exclusion handled by
        # _burst_in_progress: while set to an embryo_id, _run_loop skips
        # all other embryos and runs only the burst executor (Phase 7).
        self._burst_in_progress: Optional[str] = None

        # Global error log for cross-embryo hardware error correlation
        self.global_error_log = GlobalErrorLog()

    async def start(
        self,
        embryo_ids: List[str],
        stop_condition: str = "manual",
        base_interval_seconds: float = 120.0,
        condition_value: Any = None,
    ) -> str:
        """
        Start timelapse in background

        Returns immediately with status message. The timelapse runs
        asynchronously and can be monitored via get_status().

        Parameters
        ----------
        embryo_ids : list of str
            Embryo IDs to image (None = all active embryos)
        stop_condition : str
            One of: "manual", "hatching", "comma", "timepoints:N", "duration:Xh"
        base_interval_seconds : float
            Default interval between acquisitions
        condition_value : any, optional
            Value for stop condition (e.g., number of timepoints)

        Returns
        -------
        str
            Status message
        """
        if self._status == TimelapseStatus.RUNNING:
            return "Timelapse already running. Use stop() first or modify_embryo() to change parameters."

        # Parse stop condition
        stop_cond = self._parse_stop_condition(stop_condition, condition_value)

        # Get embryo list
        if not embryo_ids:
            embryo_ids = [e.id for e in self.experiment.embryos.values() if not e.should_skip]

        if not embryo_ids:
            return "No embryos to image. Add embryos first."

        # Validate embryos exist
        missing = [eid for eid in embryo_ids if eid not in self.experiment.embryos]
        if missing:
            return f"Embryos not found: {missing}"

        # Initialize timelapse runtime state on EmbryoState references.
        # No separate state object — orchestrator and agent share the same
        # EmbryoState instances.
        from gently.harness.roles import REGISTRY as ROLE_REGISTRY
        now = datetime.now()
        self._embryo_states = {}
        for eid in embryo_ids:
            embryo = self.experiment.embryos[eid]
            embryo.stop_condition = stop_cond
            embryo.is_complete = False
            embryo.completion_reason = None
            embryo.error_count = 0
            embryo.last_error = None
            embryo.detection_triggered_at = None
            embryo.detection_type = None
            embryo.no_object_since_timepoint = None

            # Async cadence init: per-embryo interval defaults to the role's
            # default_cadence_seconds (e.g. 300s for test/calibration), falling
            # back to base_interval_seconds, then to 300s. An explicit
            # interval_seconds already on the embryo (e.g. from a prior
            # modify_parameters call) is preserved.
            if embryo.interval_seconds is None:
                role_def = ROLE_REGISTRY.get(embryo.role)
                if role_def is not None:
                    embryo.interval_seconds = role_def.default_cadence_seconds
                else:
                    embryo.interval_seconds = base_interval_seconds
            embryo.cadence_phase = "normal"
            embryo.next_due_at = now  # image immediately on first tick
            self._embryo_states[eid] = embryo

        # Burst state is per-session; clear at start.
        self._burst_in_progress = None

        # Initialize trace directory for file-based persistence
        if self._session_id:
            self._trace_dir = TRACE_BASE_PATH / self._session_id
            self._trace_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Trace file storage enabled: {self._trace_dir}")

        # Create a perception run in the unified store
        self._perception_run_id = None
        if self._store and self._session_id and self.perceiver:
            try:
                self._perception_run_id = self._store.create_perception_run(
                    session_id=self._session_id,
                    name=f"timelapse_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    method="vlm_stage_classification",
                    model_name=settings.models.perception,
                    source="live",
                    config={"stop_condition": stop_condition, "interval": base_interval_seconds},
                )
                logger.info(f"Created perception run {self._perception_run_id} in FileStore")
            except Exception as e:
                logger.warning(f"Failed to create perception run in store: {e}")

        # Initialize round-based scheduling (global timing for all embryos)
        self._base_interval_seconds = base_interval_seconds
        self._started_at = datetime.now()
        self._total_pause_duration = timedelta(0)
        self._pause_start = None

        # Reset state (but preserve total timepoints from existing embryos)
        self._status = TimelapseStatus.RUNNING
        self._total_timepoints = sum(e.timepoints_acquired for e in self._embryo_states.values())
        self._current_round = -1  # Will become 0 on first round
        self._stop_requested = False
        self._error_message = None

        # Start background task
        self._acquisition_task = asyncio.create_task(self._run_loop())

        # Emit event
        self._emit_event(EventType.ACQUISITION_STARTED, {
            'embryo_ids': embryo_ids,
            'stop_condition': stop_condition,
            'interval_seconds': base_interval_seconds,
        })

        logger.info(f"Started timelapse for {len(embryo_ids)} embryos")

        return (
            f"Started timelapse for {len(embryo_ids)} embryos\n"
            f"Stop condition: {stop_condition}\n"
            f"Interval: {base_interval_seconds}s\n"
            f"Use get_timelapse_status to monitor progress."
        )

    def _parse_stop_condition(
        self,
        condition: str,
        value: Any = None
    ) -> StopCondition:
        """Parse stop condition string into StopCondition object"""
        condition = condition.lower().strip()

        # Handle legacy keyword forms
        if condition in ("until_hatching",):
            condition = "hatching"
        elif condition in ("until_comma",):
            condition = "comma"
        elif condition == "fixed_timepoints" and value is not None:
            return StopCondition.fixed_timepoints(int(value))
        elif condition == "duration" and value is not None:
            return StopCondition.duration_hours(float(value))

        try:
            return StopCondition.parse(condition)
        except ValueError:
            logger.warning(f"Unknown stop condition '{condition}', using manual")
            return StopCondition.manual()

    def _get_round_scheduled_time(self, round_num: int) -> datetime:
        """
        Get when a round should start, accounting for pauses.

        Parameters
        ----------
        round_num : int
            Round number (0-indexed)

        Returns
        -------
        datetime
            Scheduled time for the round
        """
        base_time = self._started_at + timedelta(seconds=round_num * self._base_interval_seconds)
        return base_time + self._total_pause_duration

    def _get_elapsed_active_time(self) -> float:
        """Get elapsed time excluding pauses, in seconds"""
        now = datetime.now()
        elapsed = (now - self._started_at).total_seconds()
        pause_seconds = self._total_pause_duration.total_seconds()
        return elapsed - pause_seconds

    # ------------------------------------------------------------------
    # Async cadence helpers
    # ------------------------------------------------------------------

    def _is_eligible(self, embryo) -> bool:
        """An embryo is eligible for the due-loop pick if it's active.

        ``paused`` phase skips it (over-budget, manual pause, or another
        embryo's burst is in flight). ``burst`` phase means an exclusive
        burst-executor handles it — the normal due-loop should skip too.
        """
        if embryo.is_complete:
            return False
        if embryo.should_skip:
            return False
        phase = getattr(embryo, "cadence_phase", "normal")
        if phase in ("paused", "burst"):
            return False
        return True

    def _pick_next_due(self) -> tuple:
        """Pick the most-overdue eligible embryo.

        Returns
        -------
        (embryo, wait_seconds) : tuple
            ``embryo`` is the EmbryoState ready to acquire now (next_due_at
            <= now), or None if no embryo is due. When None, ``wait_seconds``
            is how long until the soonest-due embryo (capped at 5s for
            responsiveness). When (embryo, 0.0), acquire immediately.
        """
        now = datetime.now()
        eligible = [e for e in self._embryo_states.values() if self._is_eligible(e)]
        if not eligible:
            return None, 5.0  # nothing to do; check back later

        # Embryos with no next_due_at yet: schedule them immediately.
        for e in eligible:
            if e.next_due_at is None:
                e.next_due_at = now

        # Find the smallest next_due_at
        soonest = min(eligible, key=lambda e: e.next_due_at)
        if soonest.next_due_at <= now:
            return soonest, 0.0

        wait_s = (soonest.next_due_at - now).total_seconds()
        return None, min(wait_s, 5.0)

    def _reschedule(self, embryo, *, from_now: bool = True) -> None:
        """Set ``embryo.next_due_at`` to now + interval_seconds.

        ``from_now=True`` (default) anchors at the actual current time —
        the right choice when acquisitions can take noticeable time. If
        you'd rather anchor at the previous due-time (catch up on missed
        rounds after a burst), pass False.
        """
        interval_s = embryo.interval_seconds or self._base_interval_seconds or 300.0
        anchor = datetime.now() if from_now else (embryo.next_due_at or datetime.now())
        embryo.next_due_at = anchor + timedelta(seconds=interval_s)

    def transition_cadence(
        self,
        embryo,
        *,
        new_phase: Optional[str] = None,
        new_interval_seconds: Optional[float] = None,
        reschedule: bool = True,
        reason: Optional[str] = None,
    ) -> None:
        """Public API for cadence transitions (Phase 5 / agent tools use this).

        Mutates the embryo, schedules its next acquisition, and emits an
        ``EMBRYO_CADENCE_CHANGED`` event so downstream UI / persistence
        stays consistent.
        """
        old_phase = getattr(embryo, "cadence_phase", "normal")
        old_interval = embryo.interval_seconds

        changed = False
        if new_phase is not None and new_phase != old_phase:
            embryo.cadence_phase = new_phase
            changed = True
        if new_interval_seconds is not None and new_interval_seconds != old_interval:
            embryo.interval_seconds = new_interval_seconds
            changed = True

        if not changed:
            return

        if reschedule:
            self._reschedule(embryo)

        self._emit_event(EventType.EMBRYO_CADENCE_CHANGED, {
            "embryo_id": embryo.id,
            "old_phase": old_phase,
            "new_phase": embryo.cadence_phase,
            "old_interval_s": old_interval,
            "new_interval_s": embryo.interval_seconds,
            "next_due_at": embryo.next_due_at.isoformat() if embryo.next_due_at else None,
            "reason": reason,
        })

    async def _finalize_timelapse(self):
        """Drain perception tasks, log trace count, emit completion event."""
        if self._perception_tasks:
            pending = list(self._perception_tasks)
            logger.info(
                f"Draining {len(pending)} perception task(s) before "
                f"completing timelapse..."
            )
            _done, still_pending = await asyncio.wait(pending, timeout=60.0)
            if still_pending:
                logger.warning(
                    f"{len(still_pending)} perception task(s) did not finish "
                    f"in 60s - proceeding to completion anyway"
                )
            self._perception_tasks.clear()

        if self._trace_dir:
            try:
                trace_count = len(list(self._trace_dir.glob("*.json")))
                logger.info(f"Trace files saved: {trace_count} files in {self._trace_dir}")
            except Exception:
                pass

        self._finalize_perception_run("completed")

        self._emit_event(EventType.ACQUISITION_COMPLETED, {
            "total_timepoints": self._total_timepoints,
            "duration_minutes": (
                (datetime.now() - self._started_at).total_seconds() / 60
                if self._started_at else 0
            ),
        })

    async def _run_loop(self):
        """Async per-embryo acquisition loop (Phase 4).

        Priority-queue model: each tick, pick the most-overdue eligible
        embryo and acquire it. Each embryo carries its own
        ``next_due_at`` / ``interval_seconds`` / ``cadence_phase`` —
        independent cadences interleave naturally.

        Exclusive burst (Phase 7): while ``_burst_in_progress`` is set,
        the loop yields the wheel to the burst executor and skips normal
        acquisitions until burst completes.
        """
        try:
            while not self._stop_requested:
                # Handle pause (whole-session)
                if self._status == TimelapseStatus.PAUSED:
                    await asyncio.sleep(1)
                    continue

                # Exclusive burst (Phase 7 wires the executor). For now,
                # if the flag is set, just yield and check back.
                if self._burst_in_progress is not None:
                    await asyncio.sleep(0.2)
                    continue

                # All embryos complete?
                active_count = sum(
                    1 for e in self._embryo_states.values()
                    if not e.is_complete and not e.should_skip
                )
                if active_count == 0:
                    self._status = TimelapseStatus.COMPLETED
                    await self._finalize_timelapse()
                    logger.info("Timelapse completed - all embryos finished")
                    break

                # Pick the next due embryo (or wait if none ready)
                embryo, wait_s = self._pick_next_due()
                if embryo is None:
                    await asyncio.sleep(max(0.1, wait_s))
                    continue

                # Acquire this embryo. Each acquisition is awaited here —
                # the embryo's next_due_at is advanced AFTER acquisition
                # lands, so we don't pile up overdue items if a single
                # acquisition takes longer than the interval.
                if self._stop_requested:
                    break
                acquire_start = datetime.now()
                await self._acquire_embryo(embryo, round_time=acquire_start)

                # Logical acquisition counter (replaces _current_round for
                # backward-compat with UI consumers that read it).
                self._current_round += 1

                # Reschedule (anchor at "now" so a slow acquisition doesn't
                # cause immediate retrigger).
                self._reschedule(embryo, from_now=True)

                # Small yield between back-to-back picks so other tasks
                # (perception, event handlers) get cycles.
                await asyncio.sleep(0.05)

                # Periodic cleanup every ~10 acquisitions.
                if self._store and self._current_round > 0 and self._current_round % 10 == 0:
                    try:
                        self._store.cleanup_incoming()
                    except Exception:
                        logger.debug("cleanup_incoming failed", exc_info=True)

        except asyncio.CancelledError:
            logger.info("Timelapse cancelled")
            self._status = TimelapseStatus.IDLE
        except Exception as e:
            logger.error(f"Timelapse error: {e}\n{traceback.format_exc()}")
            self._status = TimelapseStatus.FAILED
            self._error_message = str(e)
            self._finalize_perception_run("failed", error_message=str(e))
            self._emit_event(EventType.ACQUISITION_FAILED, {
                'error': str(e),
            })

    async def _acquire_embryo(self, embryo_state: EmbryoState, round_time: datetime = None):
        """Acquire a single volume for one embryo

        Parameters
        ----------
        embryo_state : EmbryoState
            Embryo state to acquire
        round_time : datetime, optional
            Shared timestamp for all embryos in this round (keeps them in sync)
        """
        embryo_id = embryo_state.id
        embryo = self.experiment.embryos.get(embryo_id)

        if not embryo:
            embryo_state.is_complete = True
            embryo_state.completion_reason = "embryo_removed"
            return

        try:
            # Move to embryo position
            pos = embryo.stage_position
            if pos and pos.get('x') is not None:
                await self.client.move_to_position(pos['x'], pos['y'])

            # Get calibration parameters
            cal = embryo.calibration or {}
            galvo_amplitude = cal.get('galvo_amplitude', 0.5)
            galvo_center = cal.get('galvo_center', 0.0)
            piezo_amplitude = cal.get('piezo_amplitude', 25.0)
            piezo_center = cal.get('piezo_center', 50.0)

            # Acquire based on mode (volume or snap)
            acquisition_mode = getattr(embryo, 'acquisition_mode', 'volume')

            if acquisition_mode == 'snap':
                # Single 2D lightsheet image
                result = await self.client.capture_lightsheet_image(
                    piezo_position=piezo_center,
                    galvo_position=galvo_center,
                )
                num_frames = 1
                exposure_ms = 50.0  # Default snap exposure
            else:
                # Full 3D volume (default)
                result = await self.client.acquire_volume(
                    num_slices=embryo.num_slices,
                    exposure_ms=embryo.exposure_ms,
                    galvo_amplitude=galvo_amplitude,
                    galvo_center=galvo_center,
                    piezo_amplitude=piezo_amplitude,
                    piezo_center=piezo_center,
                    laser_power_488_pct=getattr(embryo, 'laser_power_488_pct', None),
                )
                num_frames = embryo.num_slices
                exposure_ms = embryo.exposure_ms

            if result.get('success'):
                # Update state
                embryo_state.timepoints_acquired += 1
                embryo_state.error_count = 0
                self._total_timepoints += 1

                # Use round_time for consistent timestamps across all embryos
                acquisition_timestamp = round_time if round_time else datetime.now()

                # Update embryo state
                embryo.timepoints_acquired = embryo_state.timepoints_acquired
                # Record light exposure (includes updating last_imaged)
                embryo.record_exposure(
                    exposure_ms=exposure_ms,
                    num_frames=num_frames,
                    timestamp=acquisition_timestamp
                )

                # Note: VOLUME_ACQUIRED event is emitted by the callback (agent.on_volume_acquired)
                # to avoid duplicate events and include more metadata

                # Callback for volume/image processing
                volume_data = None
                volume_uids = None  # Track UIDs from storage for perception events
                if self.on_volume_callback:
                    # Get data - 'volume' for volume mode, 'image' for snap mode
                    data = result.get('volume') if acquisition_mode == 'volume' else result.get('image')
                    if data is not None:
                        # Ensure data is numpy array
                        if not isinstance(data, np.ndarray):
                            data = np.array(data)
                        # For snap mode (2D), add Z dimension so store_volume works
                        if acquisition_mode == 'snap' and data.ndim == 2:
                            data = data[np.newaxis, ...]  # Add Z dimension: (Y,X) -> (1,Y,X)
                        volume_data = data
                        # Pass volume_path if available (zero-copy from device)
                        volume_path = result.get('volume_path')
                        # Callback may return UIDs from storage
                        callback_result = await self.on_volume_callback(
                            embryo_id,
                            embryo_state.timepoints_acquired,
                            data,
                            volume_path=volume_path,
                        )
                        if isinstance(callback_result, dict):
                            volume_uids = callback_result

                # Fire perception as a background task instead of awaiting
                # it. This unblocks the next embryo's acquisition so the
                # filmstrip gets fresh volumes every few seconds, not every
                # 20-40 seconds. Perception still runs to completion and
                # still emits DETECTOR_EVALUATED events; it just no longer
                # gates the acquisition loop.
                if self.perceiver and volume_data is not None:
                    perception_task = asyncio.create_task(
                        self._run_perception(
                            embryo_id=embryo_id,
                            timepoint=embryo_state.timepoints_acquired,
                            volume=volume_data,
                            embryo_state=embryo_state,
                            volume_uids=volume_uids,
                        )
                    )
                    self._perception_tasks.add(perception_task)
                    # Remove the task from the set when it finishes so the
                    # set doesn't grow unbounded over a long timelapse.
                    perception_task.add_done_callback(self._perception_tasks.discard)

                # Check stop condition (based on acquisition state, not
                # perception - we'll re-check inside _run_perception too)
                await self._check_stop_condition(embryo_state)

                logger.debug(
                    f"Acquired t={embryo_state.timepoints_acquired} for {embryo_id}"
                )

            else:
                embryo_state.error_count += 1
                embryo_state.last_error = result.get('error', 'Unknown error')

                # Log to global error log for cross-embryo correlation
                self.global_error_log.log_error(
                    round_number=self._current_round,
                    embryo_id=embryo_id,
                    timepoint=embryo_state.timepoints_acquired,
                    error_type="acquisition",
                    message=embryo_state.last_error
                )

                # Stop after too many errors
                if embryo_state.error_count >= 3:
                    embryo_state.is_complete = True
                    embryo_state.completion_reason = f"errors: {embryo_state.last_error}"

        except Exception as e:
            embryo_state.error_count += 1
            embryo_state.last_error = str(e)
            logger.error(
                f"Acquisition failed for {embryo_id} t{embryo_state.timepoints_acquired}: "
                f"{e}\n{traceback.format_exc()}"
            )

            # Log to global error log
            self.global_error_log.log_error(
                round_number=self._current_round,
                embryo_id=embryo_id,
                timepoint=embryo_state.timepoints_acquired,
                error_type="acquisition_exception",
                message=str(e),
                exception=e
            )

    async def _check_stop_condition(self, embryo_state: EmbryoState):
        """
        Check if ANY stop condition is met (OR logic for composite conditions).

        Supports both single conditions and composite conditions.
        """
        # Check all conditions (primary + additional) with OR logic
        for cond in embryo_state.stop_condition.all_conditions():
            reason = self._evaluate_single_condition(cond, embryo_state)
            if reason:
                embryo_state.is_complete = True
                embryo_state.completion_reason = reason
                logger.info(f"Embryo {embryo_state.id} stopped: {reason}")
                return  # Stop on first matching condition

    def _evaluate_single_condition(
        self,
        cond: StopCondition,
        embryo_state: EmbryoState
    ) -> Optional[str]:
        """
        Evaluate a single stop condition.

        Parameters
        ----------
        cond : StopCondition
            Condition to evaluate
        embryo_state : EmbryoState
            Current embryo state

        Returns
        -------
        str or None
            Completion reason if condition is met, None otherwise
        """
        if cond.condition_type == StopConditionType.MANUAL:
            # Never auto-stop
            return None

        elif cond.condition_type == StopConditionType.FIXED_TIMEPOINTS:
            if embryo_state.timepoints_acquired >= cond.value:
                return f"reached {cond.value} timepoints"

        elif cond.condition_type == StopConditionType.DURATION:
            elapsed_hours = (datetime.now() - self._started_at).total_seconds() / 3600
            if elapsed_hours >= cond.value:
                return f"reached {cond.value}h duration"

        elif cond.condition_type in (StopConditionType.STAGE_BASED,
                                      StopConditionType.HATCHING,
                                      StopConditionType.COMMA_STAGE):
            # Generic stage-based stop: check if current stage is in target set
            target = cond.target_stages or set()

            # Check perception system for current stage
            if self.perceiver:
                session = self.perceiver.get_session(embryo_state.id)
                if session:
                    current_stage = session.current_stage
                    if current_stage and current_stage in target:
                        # First time detecting target stage — record the timepoint
                        if embryo_state.detection_triggered_at is None:
                            embryo_state.detection_triggered_at = embryo_state.timepoints_acquired
                            embryo_state.detection_type = f"stage:{current_stage}"
                            logger.info(
                                f"Target stage '{current_stage}' detected for {embryo_state.id} "
                                f"at t{embryo_state.timepoints_acquired}, "
                                f"will acquire {cond.confirm_timepoints} more confirmation timepoints"
                            )

                        # Check if we've acquired enough confirmation timepoints
                        timepoints_since_detection = (
                            embryo_state.timepoints_acquired - embryo_state.detection_triggered_at
                        )
                        if timepoints_since_detection >= cond.confirm_timepoints:
                            if cond.confirm_timepoints > 0:
                                return (
                                    f"stage '{current_stage}' detected + "
                                    f"{cond.confirm_timepoints} confirmation timepoints"
                                )
                            return f"target stage reached (current: {current_stage})"

                    # Also catch terminal-stage completion the upper block
                    # might miss: e.g. target={hatching} but perception now
                    # reports 'hatched'. Session has no is_complete() — we
                    # check terminal-stage membership directly.
                    organism = get_organism()
                    if (current_stage
                            and current_stage in organism.TERMINAL_STAGES
                            and target & organism.TERMINAL_STAGES):
                        return f"terminal stage '{current_stage}' reached (perception)"

            # Fallback: check legacy hatching_status (for manual marking)
            organism = get_organism()
            if target & organism.TERMINAL_STAGES:
                embryo = self.experiment.embryos.get(embryo_state.id)
                if embryo:
                    hatched_via_status = embryo.hatching_status.get('hatched', False)
                    if hatched_via_status:
                        return "terminal stage detected (manual)"

        return None

    def get_status(self) -> TimelapseState:
        """
        Get current timelapse status

        Can be called anytime, even while timelapse is running.

        Returns
        -------
        TimelapseState
            Current state with all embryo details
        """
        # Async cadence: "next round" is meaningless; report the soonest
        # next_due_at across active embryos so UI consumers still get a
        # useful "next acquisition" estimate.
        next_round_time = None
        seconds_until_next = None

        if self._status == TimelapseStatus.RUNNING and self._embryo_states:
            due_times = [
                e.next_due_at for e in self._embryo_states.values()
                if not e.is_complete
                and not e.should_skip
                and getattr(e, "cadence_phase", "normal") != "paused"
                and e.next_due_at is not None
            ]
            if due_times:
                next_round_time = min(due_times)
                seconds_until_next = max(
                    0, (next_round_time - datetime.now()).total_seconds()
                )

        return TimelapseState(
            status=self._status,
            started_at=self._started_at,
            embryos=self._embryo_states.copy(),
            total_timepoints=self._total_timepoints,
            current_round=self._current_round,
            interval_seconds=self._base_interval_seconds,
            next_round_time=next_round_time,
            seconds_until_next_round=seconds_until_next,
            error_message=self._error_message,
        )

    async def add_embryo(
        self,
        embryo_id: str,
        stop_condition: Optional[str] = None,
        condition_value: Any = None,
    ) -> str:
        """
        Add an embryo to a running timelapse (round-based architecture)

        Parameters
        ----------
        embryo_id : str
            Embryo to add
        stop_condition : str, optional
            Stop condition (defaults to same as other embryos, or "manual")
        condition_value : any, optional
            Value for stop condition

        Returns
        -------
        str
            Confirmation message
        """
        if self._status != TimelapseStatus.RUNNING:
            return "No timelapse running. Use start_adaptive_timelapse first."

        if embryo_id in self._embryo_states:
            return f"Embryo '{embryo_id}' already in timelapse"

        # Validate embryo exists
        if embryo_id not in self.experiment.embryos:
            return f"Embryo '{embryo_id}' not found in experiment"

        # Determine stop condition - default to matching other embryos or manual
        if stop_condition:
            stop_cond = self._parse_stop_condition(stop_condition, condition_value)
        else:
            # Use the same stop condition as existing embryos
            existing_states = list(self._embryo_states.values())
            if existing_states:
                stop_cond = existing_states[0].stop_condition
            else:
                stop_cond = StopCondition.manual()

        # Add embryo to the timelapse: set runtime fields on its EmbryoState
        # and register the reference in _embryo_states.
        from gently.harness.roles import REGISTRY as ROLE_REGISTRY
        embryo = self.experiment.embryos[embryo_id]
        embryo.stop_condition = stop_cond
        embryo.is_complete = False
        embryo.completion_reason = None
        embryo.error_count = 0
        embryo.last_error = None
        embryo.detection_triggered_at = None
        embryo.detection_type = None
        embryo.no_object_since_timepoint = None

        # Async cadence init for the newcomer.
        if embryo.interval_seconds is None:
            role_def = ROLE_REGISTRY.get(embryo.role)
            embryo.interval_seconds = (
                role_def.default_cadence_seconds if role_def is not None
                else self._base_interval_seconds
            )
        embryo.cadence_phase = "normal"
        embryo.next_due_at = datetime.now()  # picked up on next loop tick

        self._embryo_states[embryo_id] = embryo

        logger.info(
            f"Added {embryo_id} to timelapse "
            f"(interval={embryo.interval_seconds}s, role={embryo.role})"
        )

        return (
            f"Added {embryo_id} to timelapse\n"
            f"  Stop condition: {stop_cond.condition_type.value}\n"
            f"  Interval: {embryo.interval_seconds}s (role: {embryo.role})\n"
            f"  Will be picked up on next due-loop tick."
        )

    async def remove_embryo(self, embryo_id: str, mark_complete: bool = True) -> str:
        """
        Remove an embryo from a running timelapse

        Parameters
        ----------
        embryo_id : str
            Embryo to remove
        mark_complete : bool
            If True, mark as complete (preserves history). If False, remove entirely.

        Returns
        -------
        str
            Confirmation message
        """
        if self._status != TimelapseStatus.RUNNING:
            return "No timelapse running."

        if embryo_id not in self._embryo_states:
            return f"Embryo '{embryo_id}' not in timelapse"

        emb_state = self._embryo_states[embryo_id]
        timepoints = emb_state.timepoints_acquired

        if mark_complete:
            # Mark as complete so it's excluded from future rounds
            emb_state.is_complete = True
            emb_state.completion_reason = "removed by user"
            logger.info(f"Marked {embryo_id} as complete (removed from timelapse)")
            return (
                f"Removed {embryo_id} from timelapse\n"
                f"  Timepoints acquired: {timepoints}\n"
                f"  Status: marked complete"
            )
        else:
            # Remove entirely from tracking
            del self._embryo_states[embryo_id]
            logger.info(f"Removed {embryo_id} from timelapse tracking entirely")
            return (
                f"Removed {embryo_id} from timelapse\n"
                f"  Timepoints acquired: {timepoints}\n"
                f"  Status: removed from tracking"
            )

    def modify_interval(self, new_interval_seconds: float) -> str:
        """
        Broadcast a new acquisition interval to every embryo in this timelapse.

        Async cadence: each embryo carries its own ``interval_seconds`` and
        ``next_due_at``. This convenience method updates every embryo at once
        (including paused / fast / burst phases — they all get the new
        interval, though paused embryos stay paused until separately resumed).
        For per-embryo changes, use ``transition_cadence(embryo, ...)`` or
        the agent tool ``modify_parameters``.

        Parameters
        ----------
        new_interval_seconds : float
            New interval in seconds (must be >= 1)
        """
        if new_interval_seconds < 1:
            return "Error: Interval must be at least 1 second"

        old_base = self._base_interval_seconds
        self._base_interval_seconds = new_interval_seconds
        changed = []
        for eid, embryo in self._embryo_states.items():
            old_per = embryo.interval_seconds
            self.transition_cadence(
                embryo,
                new_interval_seconds=new_interval_seconds,
                reason="modify_interval (broadcast)",
            )
            changed.append((eid, old_per, new_interval_seconds))

        logger.info(
            "Broadcast interval: base %ss -> %ss; %d embryos rescheduled",
            old_base, new_interval_seconds, len(changed),
        )
        return (
            f"Interval changed to {new_interval_seconds}s across "
            f"{len(changed)} embryo(s); each rescheduled on next loop tick."
        )

    async def modify_embryo(
        self,
        embryo_id: str,
        stop_condition: Optional[str] = None,
        condition_value: Any = None,
    ) -> str:
        """
        Modify parameters for a single embryo during timelapse

        Note: Interval is now global for all embryos. Use modify_interval() to change it.

        Parameters
        ----------
        embryo_id : str
            Embryo to modify
        stop_condition : str, optional
            New stop condition
        condition_value : any, optional
            Value for stop condition

        Returns
        -------
        str
            Confirmation message
        """
        if embryo_id not in self._embryo_states:
            return f"Embryo '{embryo_id}' not in timelapse"

        estate = self._embryo_states[embryo_id]

        changes = []

        if stop_condition is not None:
            new_cond = self._parse_stop_condition(stop_condition, condition_value)
            estate.stop_condition = new_cond
            changes.append(f"stop condition: {stop_condition}")

        if not changes:
            return f"No changes specified for {embryo_id}. Note: use modify_interval() to change acquisition interval."

        return f"Modified {embryo_id}: {', '.join(changes)}"

    async def stop_embryo(self, embryo_id: str, reason: str = "user_request") -> str:
        """
        Stop imaging a specific embryo

        Parameters
        ----------
        embryo_id : str
            Embryo to stop
        reason : str
            Reason for stopping

        Returns
        -------
        str
            Confirmation message
        """
        if embryo_id not in self._embryo_states:
            return f"Embryo '{embryo_id}' not in timelapse"

        estate = self._embryo_states[embryo_id]
        estate.is_complete = True
        estate.completion_reason = reason

        return f"Stopped imaging {embryo_id} (reason: {reason})"

    async def stop(self, reason: str = "user_request") -> str:
        """
        Stop the entire timelapse

        Parameters
        ----------
        reason : str
            Reason for stopping

        Returns
        -------
        str
            Confirmation message
        """
        if self._status != TimelapseStatus.RUNNING:
            return "No timelapse running"

        self._stop_requested = True

        # Cancel the task
        if self._acquisition_task:
            self._acquisition_task.cancel()
            try:
                await self._acquisition_task
            except asyncio.CancelledError:
                pass

        # Drain in-flight perception tasks. Acquisition is now fire-and-
        # forget, so at stop time there may still be a few VLM calls in
        # the air. Give them a bounded window to finish gracefully so
        # DETECTOR_EVALUATED events and trace files don't get dropped.
        if self._perception_tasks:
            pending = list(self._perception_tasks)
            logger.info(f"Waiting for {len(pending)} in-flight perception task(s) to finish...")
            try:
                done, still_pending = await asyncio.wait(pending, timeout=60.0)
                if still_pending:
                    logger.warning(
                        f"{len(still_pending)} perception task(s) did not finish "
                        f"in 60s - cancelling to unblock stop"
                    )
                    for t in still_pending:
                        t.cancel()
                    # Swallow cancellation errors
                    await asyncio.gather(*still_pending, return_exceptions=True)
            except Exception as e:
                logger.warning(f"Error draining perception tasks: {e}")
            self._perception_tasks.clear()

        self._status = TimelapseStatus.IDLE

        # Log trace file count
        if self._trace_dir:
            trace_count = len(list(self._trace_dir.glob("*.json")))
            logger.info(f"Trace files saved: {trace_count} files in {self._trace_dir}")

        self._finalize_perception_run("stopped")

        # Emit stop event for viz server and other listeners
        get_event_bus().publish(
            EventType.ACQUISITION_STOPPED,
            {
                "reason": reason,
                "total_timepoints": self._total_timepoints,
                "embryo_count": len(self._embryo_states),
            },
            source="timelapse_orchestrator"
        )

        return f"Timelapse stopped (reason: {reason}). Acquired {self._total_timepoints} total timepoints."

    async def pause(self) -> str:
        """Pause the timelapse"""
        if self._status != TimelapseStatus.RUNNING:
            return "No timelapse running"

        self._pause_start = datetime.now()  # Track when pause started
        self._status = TimelapseStatus.PAUSED

        return "Timelapse paused. Use resume() to continue."

    async def resume(self) -> str:
        """Resume a paused timelapse"""
        if self._status != TimelapseStatus.PAUSED:
            return "Timelapse not paused"

        # Track pause duration to exclude from scheduling
        if self._pause_start:
            self._total_pause_duration += datetime.now() - self._pause_start
            self._pause_start = None

        self._status = TimelapseStatus.RUNNING

        return "Timelapse resumed."

    def add_interval_rule(self, rule: IntervalRule):
        """
        Add an interval adjustment rule

        Parameters
        ----------
        rule : IntervalRule
            Rule to add
        """
        self._interval_rules.append(rule)
        logger.info(f"Added interval rule: {rule.name}")

    def add_speedup_on_stage(
        self,
        stage_name: str,
        new_interval_seconds: float = 30.0,
        embryo_ids: Optional[List[str]] = None,
    ):
        """
        Add a rule to speed up imaging when a stage is reached

        Parameters
        ----------
        stage_name : str
            Stage that triggers speedup (e.g., "3fold", "pretzel", "comma")
        new_interval_seconds : float
            New interval after stage reached
        embryo_ids : list, optional
            Only apply to these embryos (None = all)
        """
        rule = IntervalRule(
            name=f"speedup_on_{stage_name}",
            trigger_stage=stage_name,
            new_interval_seconds=new_interval_seconds,
            applies_to=embryo_ids,
            one_time=True,
        )
        self.add_interval_rule(rule)

    def add_pre_terminal_speedup(self, fast_interval: float = 30.0):
        """
        Add automatic speedup when the organism's pre-terminal stage is detected.

        Uses get_organism().PRE_TERMINAL_SPEEDUP_STAGE to determine the trigger
        stage (e.g., "pretzel" for C. elegans).

        Parameters
        ----------
        fast_interval : float
            Interval to use after pre-terminal stage detection (default 30s)
        """
        organism = get_organism()
        stage = organism.PRE_TERMINAL_SPEEDUP_STAGE
        self.add_speedup_on_stage(stage, fast_interval)
        logger.info(f"Added pre-terminal speedup: {fast_interval}s after {stage} detection")

    # Backward-compatible alias
    def add_pre_hatching_speedup(self, fast_interval: float = 30.0):
        """Alias for add_pre_terminal_speedup (backward compatibility)."""
        self.add_pre_terminal_speedup(fast_interval)

    def _check_interval_rules(
        self,
        embryo_id: str,
        detector_name: Optional[str] = None,
        stage: Optional[str] = None,
    ):
        """
        Check if any interval rules should apply

        Parameters
        ----------
        embryo_id : str
            Embryo to check
        detector_name : str, optional
            Detector that just fired
        stage : str, optional
            Stage that was detected
        """
        if embryo_id not in self._embryo_states:
            return

        estate = self._embryo_states[embryo_id]

        # Get already applied rules for this embryo
        if embryo_id not in self._applied_rules:
            self._applied_rules[embryo_id] = set()
        applied = self._applied_rules[embryo_id]

        for rule in self._interval_rules:
            # Skip if already applied (for one-time rules)
            if rule.one_time and rule.name in applied:
                continue

            # Check if rule matches
            if rule.matches(
                embryo_id=embryo_id,
                detector_name=detector_name,
                stage=stage,
            ):
                # Async cadence: mutate ONLY the embryos targeted by the rule.
                # rule.applies_to=None means "all embryos in this timelapse";
                # otherwise scope to the listed ids.
                target_ids = rule.applies_to or list(self._embryo_states.keys())
                applied.add(rule.name)

                for tid in target_ids:
                    target = self._embryo_states.get(tid)
                    if target is None or target.is_complete:
                        continue
                    old_interval = target.interval_seconds
                    self.transition_cadence(
                        target,
                        new_interval_seconds=rule.new_interval_seconds,
                        reason=f"rule:{rule.name}",
                    )
                    logger.info(
                        "Applied interval rule '%s' (triggered by %s) to %s: "
                        "%ss -> %ss",
                        rule.name, embryo_id, tid, old_interval, rule.new_interval_seconds,
                    )

    def _finalize_perception_run(self, status: str = "completed", error_message: str = None):
        """Mark the perception run as finished in FileStore."""
        if self._store and self._perception_run_id is not None:
            try:
                self._store.complete_perception_run(
                    self._perception_run_id,
                    status=status,
                    error_message=error_message,
                )
                logger.info(f"Perception run {self._perception_run_id} marked as {status}")
            except Exception as e:
                logger.warning(f"Failed to finalize perception run: {e}")

    def _emit_event(self, event_type: EventType, data: Dict):
        """Emit event to event bus"""
        self._event_bus.publish(
            event_type=event_type,
            data=data,
            source="timelapse_orchestrator",
        )

    # How often to recheck embryos marked as no_object (in timepoints)
    NO_OBJECT_RECHECK_INTERVAL = 10

    def _serialize_result(self, embryo_id: str, timepoint: int, result, session=None) -> dict:
        """Serialize a perception result to a dict (shared by trace, store, and events)."""
        data = {
            "session_id": self._session_id,
            "embryo_id": embryo_id,
            "timepoint": timepoint,
            "timestamp": datetime.now().isoformat(),
            "predicted_stage": result.stage,
            "reasoning": result.reasoning,
            "raw_response": result.raw_response,
        }
        if session:
            data["stability"] = session.stability
        return data

    def _volume_to_b64(self, volume) -> tuple:
        """Extract View A from volume and return (view_a, image_b64) or (None, None)."""
        view_a = volume[0] if volume.ndim == 4 else volume

        if view_a.ndim == 3:
            z_depth, height, width = view_a.shape
            if width > height * 2:
                view_a = view_a[:, :, :width // 2]
            bounds = compute_crop_bounds(view_a)
            cropped = apply_crop_bounds(view_a, bounds)
            three_view_img, _ = projection_three_view(cropped)
            return view_a, image_to_base64(three_view_img)
        elif view_a.ndim == 2:
            img = normalize_to_uint8(view_a)
            return view_a, image_to_base64(img)
        else:
            logger.warning(f"Unexpected volume dimensions: {volume.ndim}")
            return None, None

    async def _run_perception(
        self,
        embryo_id: str,
        timepoint: int,
        volume,
        embryo_state: EmbryoState,
        volume_uids: dict = None,
    ):
        """Run perception on acquired volume and emit results."""
        # Skip perception for no_object embryos (except periodic rechecks)
        if embryo_state.no_object_since_timepoint is not None:
            timepoints_since = timepoint - embryo_state.no_object_since_timepoint
            if timepoints_since % self.NO_OBJECT_RECHECK_INTERVAL != 0:
                next_recheck = embryo_state.no_object_since_timepoint + (
                    (timepoints_since // self.NO_OBJECT_RECHECK_INTERVAL + 1)
                    * self.NO_OBJECT_RECHECK_INTERVAL
                )
                self._emit_event(EventType.DETECTOR_EVALUATED, {
                    'embryo_id': embryo_id, 'timepoint': timepoint,
                    'detector_name': 'perception', 'stage': 'no_object',
                    'reasoning': f"Skipped (empty field). Rechecking in {next_recheck - timepoint} timepoints.",
                    'skipped': True,
                })
                return
            else:
                logger.info(f"Rechecking no_object embryo {embryo_id} at t={timepoint}")

        try:
            # Volume → projection → base64
            view_a, image_b64 = self._volume_to_b64(volume)
            if view_a is None:
                return

            # Run perception via gently_perception.Perceiver
            result = await self.perceiver(
                embryo_id=embryo_id,
                timepoint=timepoint,
                image_b64=image_b64,
                timestamp=datetime.now(),
            )

            # Track no_object state
            if result.stage == "no_object":
                if embryo_state.no_object_since_timepoint is None:
                    embryo_state.no_object_since_timepoint = timepoint
                    logger.info(f"Embryo {embryo_id} marked as no_object at t={timepoint}")
            elif embryo_state.no_object_since_timepoint is not None:
                logger.info(f"Embryo {embryo_id} object found at t={timepoint}, resuming perception")
                embryo_state.no_object_since_timepoint = None

            # Get session for temporal context
            session = self.perceiver.get_session(embryo_id)

            # Serialize result once, reuse for trace file, store, and events
            result_data = self._serialize_result(embryo_id, timepoint, result, session)

            # Persist trace to JSON file
            if self._trace_dir:
                try:
                    self._write_trace_file(embryo_id, timepoint, result_data)
                except Exception as e:
                    logger.warning(f"Failed to write trace file: {e}")

            # Persist prediction to FileStore
            if self._store and self._perception_run_id and self._session_id:
                try:
                    self._store.store_prediction(
                        run_id=self._perception_run_id,
                        session_id=self._session_id,
                        embryo_id=embryo_id,
                        timepoint=timepoint,
                        predicted_stage=result.stage,
                        reasoning=result.reasoning,
                        trace_data=result_data,
                    )
                except Exception as e:
                    logger.warning(f"Failed to store prediction: {e}")

            # Build and emit perception event
            event_data = {
                'embryo_id': embryo_id, 'timepoint': timepoint,
                'detector_name': 'perception',
                'stage': result.stage,
                'reasoning': result.reasoning,
            }
            if volume_uids:
                event_data['volume_uid'] = volume_uids.get('volume_uid')
                event_data['projection_uid'] = volume_uids.get('projection_uid')
            if session:
                event_data['stability'] = session.stability
                summary = session.summary()
                if summary.get('temporal'):
                    from dataclasses import asdict
                    event_data['temporal_analysis'] = asdict(summary['temporal'])

            self._emit_event(EventType.DETECTOR_EVALUATED, event_data)

            if result.stage in ("hatching", "hatched"):
                self._emit_event(EventType.HATCHING_DETECTED, {
                    'embryo_id': embryo_id, 'timepoint': timepoint,
                    'detector_name': 'hatching', 'stage': result.stage,
                })

            # Check interval rules based on stage
            self._check_interval_rules(embryo_id=embryo_id, stage=result.stage)

            logger.info(
                f"[{embryo_id}] T{timepoint}: stage={result.stage}, "
                f"stability={session.stability if session else '?'}"
            )

        except Exception as e:
            logger.error(f"Perception failed for {embryo_id}: {e}")

    def _write_trace_file(self, embryo_id: str, timepoint: int, trace_data: dict) -> Path:
        """Write perception trace to JSON file."""
        filename = f"{embryo_id}_T{timepoint:04d}.json"
        file_path = self._trace_dir / filename
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(trace_data, f, indent=2, ensure_ascii=False)
        logger.debug(f"Wrote trace: {file_path.name}")
        return file_path
