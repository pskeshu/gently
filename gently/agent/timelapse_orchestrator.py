"""
Timelapse Orchestrator for Adaptive Multi-Embryo Acquisition

Manages background timelapse acquisition with:
- Per-embryo stop conditions (e.g., hatching detection)
- Dynamic interval adjustment
- Non-blocking operation (copilot stays responsive)
- Event-driven status updates
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set
import traceback

from ..core import EventType, get_event_bus

logger = logging.getLogger(__name__)


class StopConditionType(Enum):
    """Types of stop conditions for embryo acquisition"""
    MANUAL = "manual"                    # Stop only when user says
    HATCHING = "hatching"                # Stop when hatching detected
    COMMA_STAGE = "comma_stage"          # Stop at comma stage
    FIXED_TIMEPOINTS = "fixed_timepoints"  # Stop after N timepoints
    DURATION = "duration"                # Stop after X hours
    ALL_HATCHED = "all_hatched"          # Stop when all embryos hatch


@dataclass
class IntervalRule:
    """
    Rule for automatically adjusting acquisition interval

    Triggers when a condition is met (detector fires, stage reached, etc.)
    """
    name: str
    trigger_detector: Optional[str] = None  # Detector name that triggers this rule
    trigger_stage: Optional[str] = None     # Stage name that triggers (comma, pretzel, etc.)
    trigger_timepoint: Optional[int] = None # Timepoint that triggers
    new_interval_seconds: float = 30.0      # New interval when triggered
    applies_to: Optional[List[str]] = None  # Embryo IDs (None = all)
    one_time: bool = True                   # Only apply once per embryo

    def matches(
        self,
        embryo_id: str,
        detector_name: Optional[str] = None,
        stage: Optional[str] = None,
        timepoint: Optional[int] = None,
    ) -> bool:
        """Check if this rule should trigger"""
        # Check embryo filter
        if self.applies_to and embryo_id not in self.applies_to:
            return False

        # Check trigger conditions
        if self.trigger_detector and detector_name == self.trigger_detector:
            return True
        if self.trigger_stage and stage == self.trigger_stage:
            return True
        if self.trigger_timepoint and timepoint >= self.trigger_timepoint:
            return True

        return False


@dataclass
class StopCondition:
    """Configuration for when to stop imaging an embryo"""
    condition_type: StopConditionType
    value: Any = None  # e.g., number of timepoints, hours, etc.

    @classmethod
    def until_hatching(cls) -> 'StopCondition':
        return cls(StopConditionType.HATCHING)

    @classmethod
    def until_comma(cls) -> 'StopCondition':
        return cls(StopConditionType.COMMA_STAGE)

    @classmethod
    def fixed_timepoints(cls, n: int) -> 'StopCondition':
        return cls(StopConditionType.FIXED_TIMEPOINTS, value=n)

    @classmethod
    def duration_hours(cls, hours: float) -> 'StopCondition':
        return cls(StopConditionType.DURATION, value=hours)

    @classmethod
    def manual(cls) -> 'StopCondition':
        return cls(StopConditionType.MANUAL)


@dataclass
class EmbryoAcquisitionState:
    """State for a single embryo in the timelapse"""
    embryo_id: str
    interval_seconds: float = 120.0
    timepoints_acquired: int = 0
    last_acquired: Optional[datetime] = None
    stop_condition: StopCondition = field(default_factory=StopCondition.manual)
    is_complete: bool = False
    completion_reason: Optional[str] = None
    error_count: int = 0
    last_error: Optional[str] = None

    @property
    def next_acquisition_time(self) -> Optional[datetime]:
        """When this embryo should next be acquired"""
        if self.is_complete or self.last_acquired is None:
            return None
        return self.last_acquired + timedelta(seconds=self.interval_seconds)

    @property
    def seconds_until_next(self) -> Optional[float]:
        """Seconds until next acquisition (None if complete or never acquired)"""
        next_time = self.next_acquisition_time
        if next_time is None:
            return None
        return max(0, (next_time - datetime.now()).total_seconds())


class TimelapseStatus(Enum):
    """Overall timelapse status"""
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class TimelapseState:
    """Current state of the timelapse"""
    status: TimelapseStatus
    started_at: Optional[datetime]
    embryos: Dict[str, EmbryoAcquisitionState]
    total_timepoints: int = 0
    current_round: int = 0
    next_embryo: Optional[str] = None
    next_acquisition_in: Optional[float] = None
    error_message: Optional[str] = None

    def to_dict(self) -> Dict:
        """Serialize for display"""
        active = [e for e in self.embryos.values() if not e.is_complete]
        completed = [e for e in self.embryos.values() if e.is_complete]

        return {
            'status': self.status.value,
            'started_at': self.started_at.isoformat() if self.started_at else None,
            'duration_minutes': (datetime.now() - self.started_at).total_seconds() / 60 if self.started_at else 0,
            'total_timepoints': self.total_timepoints,
            'current_round': self.current_round,
            'active_embryos': len(active),
            'completed_embryos': len(completed),
            'next_embryo': self.next_embryo,
            'next_acquisition_in_seconds': self.next_acquisition_in,
            'embryo_details': {
                eid: {
                    'timepoints': e.timepoints_acquired,
                    'is_complete': e.is_complete,
                    'completion_reason': e.completion_reason,
                    'interval_seconds': e.interval_seconds,
                    'seconds_until_next': e.seconds_until_next,
                }
                for eid, e in self.embryos.items()
            },
            'error': self.error_message,
        }


class TimelapseOrchestrator:
    """
    Background timelapse manager

    Runs independently of copilot conversation. The copilot can:
    - Query status anytime via get_status()
    - Modify parameters via modify_embryo()
    - Stop embryos or entire timelapse via stop()

    Events are emitted for UI updates without blocking.
    """

    def __init__(
        self,
        microscope_client,
        experiment_state,
        detection_queue=None,
        on_volume_callback: Optional[Callable] = None,
    ):
        """
        Parameters
        ----------
        microscope_client : QueueServerClient
            Client for hardware control
        experiment_state : ExperimentState
            Shared experiment state
        detection_queue : DetectionQueue, optional
            For running detectors on acquired volumes
        on_volume_callback : callable, optional
            Called after each volume: on_volume_callback(embryo_id, timepoint, volume)
        """
        self.client = microscope_client
        self.experiment = experiment_state
        self.detection_queue = detection_queue
        self.on_volume_callback = on_volume_callback

        # Event bus for status updates
        self._event_bus = get_event_bus()

        # Timelapse state
        self._embryo_states: Dict[str, EmbryoAcquisitionState] = {}
        self._status = TimelapseStatus.IDLE
        self._started_at: Optional[datetime] = None
        self._total_timepoints = 0
        self._current_round = 0
        self._error_message: Optional[str] = None

        # Control
        self._acquisition_task: Optional[asyncio.Task] = None
        self._paused = False
        self._stop_requested = False

        # Interval adjustment rules
        self._interval_rules: List[IntervalRule] = []
        self._applied_rules: Dict[str, Set[str]] = {}  # embryo_id -> set of applied rule names

        # Callbacks for detection results
        self._detection_callbacks: Dict[str, Callable] = {}

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

        # Initialize embryo states (preserve existing timepoint counts)
        self._embryo_states = {}
        for eid in embryo_ids:
            embryo = self.experiment.embryos[eid]
            self._embryo_states[eid] = EmbryoAcquisitionState(
                embryo_id=eid,
                interval_seconds=embryo.interval_seconds or base_interval_seconds,
                stop_condition=stop_cond,
                timepoints_acquired=embryo.timepoints_acquired,  # Preserve count!
                last_acquired=embryo.last_imaged,  # Preserve last acquisition time
            )

        # Reset state (but preserve total timepoints from existing embryos)
        self._status = TimelapseStatus.RUNNING
        self._started_at = datetime.now()
        self._total_timepoints = sum(e.timepoints_acquired for e in self._embryo_states.values())
        self._current_round = 0
        self._paused = False
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

        if condition == "manual":
            return StopCondition.manual()
        elif condition == "hatching" or condition == "until_hatching":
            return StopCondition.until_hatching()
        elif condition == "comma" or condition == "comma_stage" or condition == "until_comma":
            return StopCondition.until_comma()
        elif condition.startswith("timepoints:"):
            n = int(condition.split(":")[1])
            return StopCondition.fixed_timepoints(n)
        elif condition.startswith("duration:"):
            h = float(condition.split(":")[1].rstrip("h"))
            return StopCondition.duration_hours(h)
        elif value is not None and condition == "fixed_timepoints":
            return StopCondition.fixed_timepoints(int(value))
        elif value is not None and condition == "duration":
            return StopCondition.duration_hours(float(value))
        else:
            logger.warning(f"Unknown stop condition '{condition}', using manual")
            return StopCondition.manual()

    async def _run_loop(self):
        """Main acquisition loop - runs in background"""
        try:
            while not self._stop_requested:
                # Check if all embryos are complete
                active_embryos = [
                    e for e in self._embryo_states.values()
                    if not e.is_complete
                ]

                if not active_embryos:
                    self._status = TimelapseStatus.COMPLETED
                    self._emit_event(EventType.ACQUISITION_COMPLETED, {
                        'total_timepoints': self._total_timepoints,
                        'duration_minutes': (datetime.now() - self._started_at).total_seconds() / 60,
                    })
                    logger.info("Timelapse completed - all embryos finished")
                    break

                # Handle pause
                if self._paused:
                    await asyncio.sleep(1)
                    continue

                # Get next embryo to acquire (by next acquisition time)
                self._current_round += 1
                embryos_this_round = self._get_embryos_for_round()

                if not embryos_this_round:
                    # Wait until next embryo is ready
                    wait_time = self._get_min_wait_time()
                    if wait_time and wait_time > 0:
                        await asyncio.sleep(min(wait_time, 10))  # Check every 10s max
                    else:
                        await asyncio.sleep(1)
                    continue

                # Acquire each ready embryo
                for embryo_state in embryos_this_round:
                    if self._stop_requested:
                        break

                    await self._acquire_embryo(embryo_state)

                    # Small delay between embryos
                    await asyncio.sleep(0.5)

        except asyncio.CancelledError:
            logger.info("Timelapse cancelled")
            self._status = TimelapseStatus.IDLE
        except Exception as e:
            logger.error(f"Timelapse error: {e}\n{traceback.format_exc()}")
            self._status = TimelapseStatus.FAILED
            self._error_message = str(e)
            self._emit_event(EventType.ACQUISITION_FAILED, {
                'error': str(e),
            })

    def _get_embryos_for_round(self) -> List[EmbryoAcquisitionState]:
        """Get embryos that are ready for acquisition now"""
        now = datetime.now()
        ready = []

        for embryo_state in self._embryo_states.values():
            if embryo_state.is_complete:
                continue

            # First acquisition or time has elapsed
            if embryo_state.last_acquired is None:
                ready.append(embryo_state)
            elif embryo_state.next_acquisition_time <= now:
                ready.append(embryo_state)

        return ready

    def _get_min_wait_time(self) -> Optional[float]:
        """Get minimum time until any embryo is ready"""
        wait_times = []
        for embryo_state in self._embryo_states.values():
            if embryo_state.is_complete:
                continue
            secs = embryo_state.seconds_until_next
            if secs is not None:
                wait_times.append(secs)

        return min(wait_times) if wait_times else None

    async def _acquire_embryo(self, embryo_state: EmbryoAcquisitionState):
        """Acquire a single volume for one embryo"""
        embryo_id = embryo_state.embryo_id
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
                )
                num_frames = embryo.num_slices
                exposure_ms = embryo.exposure_ms

            if result.get('success'):
                # Update state
                embryo_state.timepoints_acquired += 1
                embryo_state.last_acquired = datetime.now()
                embryo_state.error_count = 0
                self._total_timepoints += 1

                # Update embryo state
                embryo.timepoints_acquired = embryo_state.timepoints_acquired
                # Record light exposure (includes updating last_imaged)
                embryo.record_exposure(
                    exposure_ms=exposure_ms,
                    num_frames=num_frames,
                    timestamp=embryo_state.last_acquired
                )

                # Note: VOLUME_ACQUIRED event is emitted by the callback (copilot.on_volume_acquired)
                # to avoid duplicate events and include more metadata

                # Callback for volume/image processing
                if self.on_volume_callback:
                    # Get data - 'volume' for volume mode, 'image' for snap mode
                    data = result.get('volume') if acquisition_mode == 'volume' else result.get('image')
                    if data is not None:
                        # Ensure data is numpy array
                        import numpy as np
                        if not isinstance(data, np.ndarray):
                            data = np.array(data)
                        # For snap mode (2D), add Z dimension so store_volume works
                        if acquisition_mode == 'snap' and data.ndim == 2:
                            data = data[np.newaxis, ...]  # Add Z dimension: (Y,X) -> (1,Y,X)
                        await self.on_volume_callback(
                            embryo_id,
                            embryo_state.timepoints_acquired,
                            data
                        )

                # Check stop condition
                await self._check_stop_condition(embryo_state)

                logger.debug(
                    f"Acquired t={embryo_state.timepoints_acquired} for {embryo_id}"
                )

            else:
                embryo_state.error_count += 1
                embryo_state.last_error = result.get('error', 'Unknown error')
                logger.warning(f"Acquisition failed for {embryo_id}: {embryo_state.last_error}")

                # Stop after too many errors
                if embryo_state.error_count >= 3:
                    embryo_state.is_complete = True
                    embryo_state.completion_reason = f"errors: {embryo_state.last_error}"

        except Exception as e:
            embryo_state.error_count += 1
            embryo_state.last_error = str(e)
            logger.error(f"Error acquiring {embryo_id}: {e}")

    async def _check_stop_condition(self, embryo_state: EmbryoAcquisitionState):
        """Check if embryo should stop based on its condition"""
        cond = embryo_state.stop_condition

        if cond.condition_type == StopConditionType.MANUAL:
            # Never auto-stop
            pass

        elif cond.condition_type == StopConditionType.FIXED_TIMEPOINTS:
            if embryo_state.timepoints_acquired >= cond.value:
                embryo_state.is_complete = True
                embryo_state.completion_reason = f"reached {cond.value} timepoints"

        elif cond.condition_type == StopConditionType.DURATION:
            elapsed_hours = (datetime.now() - self._started_at).total_seconds() / 3600
            if elapsed_hours >= cond.value:
                embryo_state.is_complete = True
                embryo_state.completion_reason = f"reached {cond.value}h duration"

        elif cond.condition_type == StopConditionType.HATCHING:
            # Check if hatching was detected (check both places)
            embryo = self.experiment.embryos.get(embryo_state.embryo_id)
            if embryo:
                # Check hatching_status (legacy)
                hatched_via_status = embryo.hatching_status.get('hatched', False)
                # Check detection_results (from detector system)
                hatched_via_detector = embryo.was_detected('hatching')

                if hatched_via_status or hatched_via_detector:
                    embryo_state.is_complete = True
                    embryo_state.completion_reason = "hatching detected"
                    self._emit_event(EventType.HATCHING_DETECTED, {
                        'embryo_id': embryo_state.embryo_id,
                        'timepoint': embryo_state.timepoints_acquired,
                    })
                    logger.info(f"Embryo {embryo_state.embryo_id} stopped: hatching detected")

        elif cond.condition_type == StopConditionType.COMMA_STAGE:
            # Check if comma stage was detected
            embryo = self.experiment.embryos.get(embryo_state.embryo_id)
            if embryo:
                # Use the was_detected helper from EmbryoState
                if embryo.was_detected('comma') or embryo.was_detected('comma_stage'):
                    embryo_state.is_complete = True
                    embryo_state.completion_reason = "comma stage detected"
                    logger.info(f"Embryo {embryo_state.embryo_id} stopped: comma stage detected")

    def get_status(self) -> TimelapseState:
        """
        Get current timelapse status

        Can be called anytime, even while timelapse is running.

        Returns
        -------
        TimelapseState
            Current state with all embryo details
        """
        # Calculate next acquisition
        next_embryo = None
        next_time = None

        if self._status == TimelapseStatus.RUNNING:
            for eid, estate in self._embryo_states.items():
                if estate.is_complete:
                    continue
                secs = estate.seconds_until_next
                if secs is not None and (next_time is None or secs < next_time):
                    next_time = secs
                    next_embryo = eid

        return TimelapseState(
            status=self._status,
            started_at=self._started_at,
            embryos=self._embryo_states.copy(),
            total_timepoints=self._total_timepoints,
            current_round=self._current_round,
            next_embryo=next_embryo,
            next_acquisition_in=next_time,
            error_message=self._error_message,
        )

    async def add_embryo(
        self,
        embryo_id: str,
        interval_seconds: Optional[float] = None,
        stop_condition: Optional[str] = None,
        condition_value: Any = None,
    ) -> str:
        """
        Add an embryo to a running timelapse

        Parameters
        ----------
        embryo_id : str
            Embryo to add
        interval_seconds : float, optional
            Interval for this embryo (defaults to 120s)
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

        embryo = self.experiment.embryos[embryo_id]

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

        # Determine interval
        if interval_seconds is None:
            interval_seconds = embryo.interval_seconds or 120.0

        # Add new embryo state
        self._embryo_states[embryo_id] = EmbryoAcquisitionState(
            embryo_id=embryo_id,
            interval_seconds=interval_seconds,
            stop_condition=stop_cond,
        )

        logger.info(f"Added {embryo_id} to timelapse with interval {interval_seconds}s")

        return (
            f"✓ Added {embryo_id} to timelapse\n"
            f"  Interval: {interval_seconds}s\n"
            f"  Stop condition: {stop_cond.condition_type.value}\n"
            f"  Will start imaging on next round"
        )

    async def modify_embryo(
        self,
        embryo_id: str,
        interval_seconds: Optional[float] = None,
        stop_condition: Optional[str] = None,
        condition_value: Any = None,
    ) -> str:
        """
        Modify parameters for a single embryo during timelapse

        Parameters
        ----------
        embryo_id : str
            Embryo to modify
        interval_seconds : float, optional
            New interval
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

        if interval_seconds is not None:
            old_interval = estate.interval_seconds
            estate.interval_seconds = interval_seconds
            changes.append(f"interval: {old_interval}s -> {interval_seconds}s")

        if stop_condition is not None:
            new_cond = self._parse_stop_condition(stop_condition, condition_value)
            estate.stop_condition = new_cond
            changes.append(f"stop condition: {stop_condition}")

        if not changes:
            return f"No changes specified for {embryo_id}"

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

        self._status = TimelapseStatus.IDLE

        return f"Timelapse stopped (reason: {reason}). Acquired {self._total_timepoints} total timepoints."

    async def pause(self) -> str:
        """Pause the timelapse"""
        if self._status != TimelapseStatus.RUNNING:
            return "No timelapse running"

        self._paused = True
        self._status = TimelapseStatus.PAUSED

        return "Timelapse paused. Use resume() to continue."

    async def resume(self) -> str:
        """Resume a paused timelapse"""
        if self._status != TimelapseStatus.PAUSED:
            return "Timelapse not paused"

        self._paused = False
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

    def add_speedup_on_detection(
        self,
        detector_name: str,
        new_interval_seconds: float = 30.0,
        embryo_ids: Optional[List[str]] = None,
    ):
        """
        Add a rule to speed up imaging when a detector fires

        Parameters
        ----------
        detector_name : str
            Detector that triggers speedup (e.g., "pretzel", "comma")
        new_interval_seconds : float
            New interval after detection
        embryo_ids : list, optional
            Only apply to these embryos (None = all)
        """
        rule = IntervalRule(
            name=f"speedup_on_{detector_name}",
            trigger_detector=detector_name,
            new_interval_seconds=new_interval_seconds,
            applies_to=embryo_ids,
            one_time=True,
        )
        self.add_interval_rule(rule)

    def add_pre_hatching_speedup(self, fast_interval: float = 30.0):
        """
        Add automatic speedup when pretzel/3-fold stage is detected

        This is a convenience method for the common "speed up near hatching" use case.
        When the pretzel detector fires, the interval is reduced to capture hatching.

        Parameters
        ----------
        fast_interval : float
            Interval to use after pretzel detection (default 30s)
        """
        self.add_speedup_on_detection("pretzel", fast_interval)
        logger.info(f"Added pre-hatching speedup: {fast_interval}s after pretzel detection")

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
                timepoint=estate.timepoints_acquired,
            ):
                old_interval = estate.interval_seconds
                estate.interval_seconds = rule.new_interval_seconds
                applied.add(rule.name)

                logger.info(
                    f"Applied interval rule '{rule.name}' to {embryo_id}: "
                    f"{old_interval}s -> {rule.new_interval_seconds}s"
                )

                # Emit event
                self._emit_event(EventType.STATUS_CHANGED, {
                    'embryo_id': embryo_id,
                    'change': 'interval_adjusted',
                    'rule': rule.name,
                    'old_interval': old_interval,
                    'new_interval': rule.new_interval_seconds,
                })

    def on_detection_result(
        self,
        embryo_id: str,
        detector_name: str,
        result: Dict
    ):
        """
        Handle detection result from detector system

        Called by copilot when a detector fires. Can trigger
        stop conditions or interval adjustments.

        Parameters
        ----------
        embryo_id : str
            Embryo that was analyzed
        detector_name : str
            Name of detector that fired
        result : dict
            Detection result
        """
        if embryo_id not in self._embryo_states:
            return

        estate = self._embryo_states[embryo_id]

        # Check interval adjustment rules first (if detected)
        if result.get('detected'):
            self._check_interval_rules(
                embryo_id=embryo_id,
                detector_name=detector_name,
            )

        # Check if this triggers a stop condition
        if detector_name == 'hatching' and result.get('detected'):
            if estate.stop_condition.condition_type == StopConditionType.HATCHING:
                estate.is_complete = True
                estate.completion_reason = "hatching detected"
                logger.info(f"Stopping {embryo_id}: hatching detected")

        elif detector_name == 'comma_stage' and result.get('detected'):
            if estate.stop_condition.condition_type == StopConditionType.COMMA_STAGE:
                estate.is_complete = True
                estate.completion_reason = "comma stage detected"
                logger.info(f"Stopping {embryo_id}: comma stage detected")

    def _emit_event(self, event_type: EventType, data: Dict):
        """Emit event to event bus"""
        self._event_bus.publish(
            event_type=event_type,
            data=data,
            source="timelapse_orchestrator",
        )
