"""
Timelapse Orchestrator for Adaptive Multi-Embryo Acquisition

Manages background timelapse acquisition with:
- Per-embryo stop conditions (e.g., hatching detection)
- Dynamic interval adjustment
- Non-blocking operation (copilot stays responsive)
- Event-driven status updates
"""

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, TYPE_CHECKING
import traceback

from ..core import EventType, get_event_bus
from .error_log import GlobalErrorLog

if TYPE_CHECKING:
    pass

# Default trace directory
TRACE_BASE_PATH = Path("D:/Gently/traces")

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
    """
    Configuration for when to stop imaging an embryo.

    Supports composite conditions with OR logic via additional_conditions.
    If ANY condition is met (primary or additional), the embryo will stop.

    For detection-based conditions (HATCHING, COMMA_STAGE), confirm_timepoints
    specifies how many additional timepoints to acquire after detection before
    actually stopping - useful to verify the detection is real.
    """
    condition_type: StopConditionType
    value: Any = None  # e.g., number of timepoints, hours, etc.
    confirm_timepoints: int = 0  # Extra timepoints to acquire after detection
    additional_conditions: List['StopCondition'] = field(default_factory=list)

    def add_condition(self, condition: 'StopCondition') -> None:
        """
        Add another stop condition (OR logic).

        The embryo will stop when ANY condition is met.

        Parameters
        ----------
        condition : StopCondition
            Additional condition to add
        """
        self.additional_conditions.append(condition)

    def all_conditions(self) -> List['StopCondition']:
        """
        Get all conditions including self (flattened).

        Returns
        -------
        List[StopCondition]
            Primary condition plus all additional conditions
        """
        return [self] + self.additional_conditions

    def describe(self) -> str:
        """
        Human-readable description of the stop condition(s).

        Returns
        -------
        str
            Description like "hatching OR duration:10h"
        """
        def _describe_single(cond: 'StopCondition') -> str:
            confirm_suffix = f"+{cond.confirm_timepoints}tp" if cond.confirm_timepoints > 0 else ""
            if cond.condition_type == StopConditionType.MANUAL:
                return "manual"
            elif cond.condition_type == StopConditionType.HATCHING:
                return f"hatching{confirm_suffix}"
            elif cond.condition_type == StopConditionType.COMMA_STAGE:
                return f"comma_stage{confirm_suffix}"
            elif cond.condition_type == StopConditionType.FIXED_TIMEPOINTS:
                return f"{cond.value} timepoints"
            elif cond.condition_type == StopConditionType.DURATION:
                return f"{cond.value}h duration"
            else:
                return str(cond.condition_type.value)

        descriptions = [_describe_single(self)]
        for cond in self.additional_conditions:
            descriptions.append(_describe_single(cond))

        return " OR ".join(descriptions)

    @classmethod
    def until_hatching(cls, confirm_timepoints: int = 0) -> 'StopCondition':
        """
        Stop when hatching is detected.

        Parameters
        ----------
        confirm_timepoints : int
            Extra timepoints to acquire after detection to confirm (default 0)
        """
        return cls(StopConditionType.HATCHING, confirm_timepoints=confirm_timepoints)

    @classmethod
    def until_comma(cls, confirm_timepoints: int = 0) -> 'StopCondition':
        """
        Stop when comma stage is detected.

        Parameters
        ----------
        confirm_timepoints : int
            Extra timepoints to acquire after detection to confirm (default 0)
        """
        return cls(StopConditionType.COMMA_STAGE, confirm_timepoints=confirm_timepoints)

    @classmethod
    def fixed_timepoints(cls, n: int) -> 'StopCondition':
        return cls(StopConditionType.FIXED_TIMEPOINTS, value=n)

    @classmethod
    def duration_hours(cls, hours: float) -> 'StopCondition':
        return cls(StopConditionType.DURATION, value=hours)

    @classmethod
    def manual(cls) -> 'StopCondition':
        return cls(StopConditionType.MANUAL)

    @classmethod
    def composite(cls, *conditions: 'StopCondition') -> 'StopCondition':
        """
        Create a composite stop condition from multiple conditions (OR logic).

        Parameters
        ----------
        *conditions : StopCondition
            Multiple conditions to combine

        Returns
        -------
        StopCondition
            First condition with others added as additional

        Examples
        --------
        >>> StopCondition.composite(
        ...     StopCondition.until_hatching(),
        ...     StopCondition.duration_hours(10)
        ... )
        # Stops on hatching OR after 10 hours
        """
        if not conditions:
            return cls.manual()

        primary = conditions[0]
        for cond in conditions[1:]:
            primary.add_condition(cond)
        return primary

    @classmethod
    def parse(cls, spec: str) -> 'StopCondition':
        """
        Parse a stop condition specification string.

        Supports composite conditions with | separator.

        Parameters
        ----------
        spec : str
            Specification like "hatching", "duration:10", "hatching|duration:10"

        Returns
        -------
        StopCondition
            Parsed condition(s)
        """
        def _parse_single(s: str) -> 'StopCondition':
            s = s.strip().lower()

            # Check for confirmation timepoints suffix: "hatching+3" or "comma+5"
            confirm_timepoints = 0
            if '+' in s:
                base, confirm_str = s.rsplit('+', 1)
                try:
                    confirm_timepoints = int(confirm_str)
                    s = base
                except ValueError:
                    pass  # Not a valid number, treat as part of the string

            if s == 'manual':
                return cls.manual()
            elif s == 'hatching':
                return cls.until_hatching(confirm_timepoints=confirm_timepoints)
            elif s in ('comma', 'comma_stage'):
                return cls.until_comma(confirm_timepoints=confirm_timepoints)
            elif s.startswith('timepoints:'):
                n = int(s.split(':')[1])
                return cls.fixed_timepoints(n)
            elif s.startswith('duration:'):
                hours_str = s.split(':')[1]
                # Handle "10h" or "10" format
                if hours_str.endswith('h'):
                    hours_str = hours_str[:-1]
                hours = float(hours_str)
                return cls.duration_hours(hours)
            else:
                raise ValueError(f"Unknown stop condition: {s}")

        parts = spec.split('|')
        conditions = [_parse_single(p) for p in parts]
        return cls.composite(*conditions)


@dataclass
class EmbryoAcquisitionState:
    """
    State for a single embryo in the timelapse.

    Note: Timing is now handled globally by the orchestrator's round-based
    scheduling. Per-embryo timing fields are kept for backward compatibility
    but the orchestrator uses global round timing for synchronization.
    """
    embryo_id: str
    timepoints_acquired: int = 0
    stop_condition: StopCondition = field(default_factory=StopCondition.manual)
    is_complete: bool = False
    completion_reason: Optional[str] = None
    error_count: int = 0
    last_error: Optional[str] = None
    # Track when detection first occurred (for confirm_timepoints feature)
    detection_triggered_at: Optional[int] = None  # Timepoint when detection first fired
    detection_type: Optional[str] = None  # What was detected (hatching, comma, etc.)
    # Track no_object state for skipping perception
    no_object_since_timepoint: Optional[int] = None  # Timepoint when no_object was first returned


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
    interval_seconds: float = 120.0
    next_round_time: Optional[datetime] = None
    seconds_until_next_round: Optional[float] = None
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
            'interval_seconds': self.interval_seconds,
            'next_round_time': self.next_round_time.isoformat() if self.next_round_time else None,
            'seconds_until_next_round': self.seconds_until_next_round,
            'active_embryos': len(active),
            'completed_embryos': len(completed),
            'embryo_details': {
                eid: {
                    'timepoints': e.timepoints_acquired,
                    'is_complete': e.is_complete,
                    'completion_reason': e.completion_reason,
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
        perception_manager=None,
        on_volume_callback: Optional[Callable] = None,
        session_id: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        microscope_client : QueueServerClient
            Client for hardware control
        experiment_state : ExperimentState
            Shared experiment state
        perception_manager : PerceptionManager, optional
            VLM-based perception system for stage classification
        on_volume_callback : callable, optional
            Called after each volume: on_volume_callback(embryo_id, timepoint, volume)
        session_id : str, optional
            Session identifier for trace file storage
        """
        self.client = microscope_client
        self.experiment = experiment_state
        self.perception_manager = perception_manager
        self.on_volume_callback = on_volume_callback

        # Trace file storage (writes JSON files to disk)
        self._session_id = session_id
        self._trace_dir: Optional[Path] = None

        # Event bus for status updates
        self._event_bus = get_event_bus()

        # Timelapse state
        self._embryo_states: Dict[str, EmbryoAcquisitionState] = {}
        self._status = TimelapseStatus.IDLE
        self._started_at: Optional[datetime] = None
        self._total_timepoints = 0
        self._current_round = 0
        self._error_message: Optional[str] = None

        # Round-based scheduling (global timing for all embryos)
        self._base_interval_seconds: float = 120.0
        self._timelapse_start_time: Optional[datetime] = None
        self._total_pause_duration: timedelta = timedelta(0)
        self._pause_start: Optional[datetime] = None

        # Control
        self._acquisition_task: Optional[asyncio.Task] = None
        self._paused = False
        self._stop_requested = False

        # Interval adjustment rules
        self._interval_rules: List[IntervalRule] = []
        self._applied_rules: Dict[str, Set[str]] = {}  # embryo_id -> set of applied rule names

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

        # Initialize embryo states (preserve existing timepoint counts)
        self._embryo_states = {}
        for eid in embryo_ids:
            embryo = self.experiment.embryos[eid]
            self._embryo_states[eid] = EmbryoAcquisitionState(
                embryo_id=eid,
                stop_condition=stop_cond,
                timepoints_acquired=embryo.timepoints_acquired,  # Preserve count!
            )

        # Initialize trace directory for file-based persistence
        if self._session_id:
            self._trace_dir = TRACE_BASE_PATH / self._session_id
            self._trace_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Trace file storage enabled: {self._trace_dir}")

        # Initialize round-based scheduling (global timing for all embryos)
        self._base_interval_seconds = base_interval_seconds
        self._timelapse_start_time = datetime.now()
        self._total_pause_duration = timedelta(0)
        self._pause_start = None

        # Reset state (but preserve total timepoints from existing embryos)
        self._status = TimelapseStatus.RUNNING
        self._started_at = self._timelapse_start_time
        self._total_timepoints = sum(e.timepoints_acquired for e in self._embryo_states.values())
        self._current_round = -1  # Will become 0 on first round
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
        base_time = self._timelapse_start_time + timedelta(seconds=round_num * self._base_interval_seconds)
        return base_time + self._total_pause_duration

    def _get_elapsed_active_time(self) -> float:
        """Get elapsed time excluding pauses, in seconds"""
        now = datetime.now()
        elapsed = (now - self._timelapse_start_time).total_seconds()
        pause_seconds = self._total_pause_duration.total_seconds()
        return elapsed - pause_seconds

    async def _run_loop(self):
        """Main acquisition loop - runs in background with round-based scheduling"""
        try:
            while not self._stop_requested:
                # Handle pause
                if self._paused:
                    await asyncio.sleep(1)
                    continue

                # Calculate which round we should be on based on elapsed active time
                elapsed_active = self._get_elapsed_active_time()
                target_round = int(elapsed_active // self._base_interval_seconds)

                # If we haven't reached next round yet, wait
                if target_round <= self._current_round:
                    next_round_time = self._get_round_scheduled_time(self._current_round + 1)
                    wait_seconds = (next_round_time - datetime.now()).total_seconds()
                    if wait_seconds > 0:
                        await asyncio.sleep(min(wait_seconds, 5))  # Check every 5s max
                    else:
                        await asyncio.sleep(0.5)
                    continue

                # Check if all embryos are complete
                active_embryos = [e for e in self._embryo_states.values() if not e.is_complete]
                if not active_embryos:
                    self._status = TimelapseStatus.COMPLETED

                    # Log trace file count
                    if self._trace_dir:
                        trace_count = len(list(self._trace_dir.glob("*.json")))
                        logger.info(f"Trace files saved: {trace_count} files in {self._trace_dir}")

                    self._emit_event(EventType.ACQUISITION_COMPLETED, {
                        'total_timepoints': self._total_timepoints,
                        'duration_minutes': (datetime.now() - self._started_at).total_seconds() / 60,
                    })
                    logger.info("Timelapse completed - all embryos finished")
                    break

                # Execute round - ALL active embryos together
                self._current_round = target_round
                round_time = self._get_round_scheduled_time(target_round)

                logger.info(f"Starting round {target_round} (scheduled: {round_time.strftime('%H:%M:%S')})")

                for embryo_state in active_embryos:
                    if self._stop_requested:
                        break
                    await self._acquire_embryo(embryo_state, round_time=round_time)
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

    async def _acquire_embryo(self, embryo_state: EmbryoAcquisitionState, round_time: datetime = None):
        """Acquire a single volume for one embryo

        Parameters
        ----------
        embryo_state : EmbryoAcquisitionState
            Embryo state to acquire
        round_time : datetime, optional
            Shared timestamp for all embryos in this round (keeps them in sync)
        """
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

                # Note: VOLUME_ACQUIRED event is emitted by the callback (copilot.on_volume_acquired)
                # to avoid duplicate events and include more metadata

                # Callback for volume/image processing
                volume_data = None
                volume_uids = None  # Track UIDs from storage for perception events
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
                        volume_data = data
                        # Callback may return UIDs from storage
                        callback_result = await self.on_volume_callback(
                            embryo_id,
                            embryo_state.timepoints_acquired,
                            data
                        )
                        if isinstance(callback_result, dict):
                            volume_uids = callback_result

                # Run perception on acquired volume
                if self.perception_manager and volume_data is not None:
                    await self._run_perception(
                        embryo_id=embryo_id,
                        timepoint=embryo_state.timepoints_acquired,
                        volume=volume_data,
                        embryo_state=embryo_state,
                        volume_uids=volume_uids,
                    )

                # Check stop condition
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

            # Log to global error log
            self.global_error_log.log_error(
                round_number=self._current_round,
                embryo_id=embryo_id,
                timepoint=embryo_state.timepoints_acquired,
                error_type="acquisition_exception",
                message=str(e),
                exception=e
            )

    async def _check_stop_condition(self, embryo_state: EmbryoAcquisitionState):
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
                logger.info(f"Embryo {embryo_state.embryo_id} stopped: {reason}")
                return  # Stop on first matching condition

    def _evaluate_single_condition(
        self,
        cond: StopCondition,
        embryo_state: EmbryoAcquisitionState
    ) -> Optional[str]:
        """
        Evaluate a single stop condition.

        Parameters
        ----------
        cond : StopCondition
            Condition to evaluate
        embryo_state : EmbryoAcquisitionState
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

        elif cond.condition_type == StopConditionType.HATCHING:
            # Check perception system for hatching/hatched status
            if self.perception_manager:
                session = self.perception_manager.get_session(embryo_state.embryo_id)
                if session and session.is_complete():
                    return "hatching complete (perception)"

            # Fallback: check legacy hatching_status (for manual marking)
            embryo = self.experiment.embryos.get(embryo_state.embryo_id)
            if embryo:
                hatched_via_status = embryo.hatching_status.get('hatched', False)
                if hatched_via_status:
                    return "hatching detected (manual)"

        elif cond.condition_type == StopConditionType.COMMA_STAGE:
            # Check perception system for comma stage
            if self.perception_manager:
                session = self.perception_manager.get_session(embryo_state.embryo_id)
                if session:
                    current_stage = session.get_current_stage()
                    # Comma or later stages (embryo has passed comma)
                    if current_stage in ('comma', '1.5fold', '2fold', '3fold', 'hatched'):
                        # First time detecting comma - record the timepoint
                        if embryo_state.detection_triggered_at is None:
                            embryo_state.detection_triggered_at = embryo_state.timepoints_acquired
                            embryo_state.detection_type = "comma_stage"
                            logger.info(
                                f"Comma stage detected for {embryo_state.embryo_id} at t{embryo_state.timepoints_acquired}, "
                                f"will acquire {cond.confirm_timepoints} more confirmation timepoints"
                            )

                        # Check if we've acquired enough confirmation timepoints
                        timepoints_since_detection = embryo_state.timepoints_acquired - embryo_state.detection_triggered_at
                        if timepoints_since_detection >= cond.confirm_timepoints:
                            if cond.confirm_timepoints > 0:
                                return f"comma stage detected + {cond.confirm_timepoints} confirmation timepoints"
                            return f"comma stage detected (current: {current_stage})"

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
        # Calculate next round timing
        next_round_time = None
        seconds_until_next = None

        if self._status == TimelapseStatus.RUNNING and self._timelapse_start_time:
            next_round_time = self._get_round_scheduled_time(self._current_round + 1)
            seconds_until_next = max(0, (next_round_time - datetime.now()).total_seconds())

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

        # Add new embryo state (round-based: no per-embryo interval)
        self._embryo_states[embryo_id] = EmbryoAcquisitionState(
            embryo_id=embryo_id,
            stop_condition=stop_cond,
        )

        logger.info(f"Added {embryo_id} to timelapse (will join round {self._current_round + 1})")

        return (
            f"Added {embryo_id} to timelapse\n"
            f"  Stop condition: {stop_cond.condition_type.value}\n"
            f"  Global interval: {self._base_interval_seconds}s\n"
            f"  Will start imaging on round {self._current_round + 1}"
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
        Change the global acquisition interval (affects all embryos).

        The change takes effect on the next round.

        Parameters
        ----------
        new_interval_seconds : float
            New interval in seconds (must be >= 10)

        Returns
        -------
        str
            Confirmation message
        """
        if new_interval_seconds < 10:
            return "Error: Interval must be at least 10 seconds"

        old_interval = self._base_interval_seconds
        self._base_interval_seconds = new_interval_seconds
        logger.info(f"Interval changed from {old_interval}s to {new_interval_seconds}s")

        return f"Interval changed from {old_interval}s to {new_interval_seconds}s (takes effect next round)"

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

        self._status = TimelapseStatus.IDLE

        # Log trace file count
        if self._trace_dir:
            trace_count = len(list(self._trace_dir.glob("*.json")))
            logger.info(f"Trace files saved: {trace_count} files in {self._trace_dir}")

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

        self._paused = True
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

    def add_pre_hatching_speedup(self, fast_interval: float = 30.0):
        """
        Add automatic speedup when 3fold stage is detected

        This is a convenience method for the common "speed up near hatching" use case.
        When the perception system detects 3fold stage, the interval is reduced to
        capture hatching at higher temporal resolution.

        Parameters
        ----------
        fast_interval : float
            Interval to use after 3fold detection (default 30s)
        """
        self.add_speedup_on_stage("3fold", fast_interval)
        logger.info(f"Added pre-hatching speedup: {fast_interval}s after 3fold detection")

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
                # Round-based: interval rules now modify the global interval
                old_interval = self._base_interval_seconds
                self._base_interval_seconds = rule.new_interval_seconds
                applied.add(rule.name)

                logger.info(
                    f"Applied interval rule '{rule.name}' (triggered by {embryo_id}): "
                    f"global interval {old_interval}s -> {rule.new_interval_seconds}s"
                )

                # Emit event
                self._emit_event(EventType.STATUS_CHANGED, {
                    'embryo_id': embryo_id,
                    'change': 'global_interval_adjusted',
                    'rule': rule.name,
                    'old_interval': old_interval,
                    'new_interval': rule.new_interval_seconds,
                })

    def _emit_event(self, event_type: EventType, data: Dict):
        """Emit event to event bus"""
        self._event_bus.publish(
            event_type=event_type,
            data=data,
            source="timelapse_orchestrator",
        )

    # How often to recheck embryos marked as no_object (in timepoints)
    NO_OBJECT_RECHECK_INTERVAL = 10

    async def _run_perception(
        self,
        embryo_id: str,
        timepoint: int,
        volume,
        embryo_state: EmbryoAcquisitionState,
        volume_uids: dict = None,
    ):
        """
        Run perception on acquired volume.

        Creates a dual-view projection image (top + side MIPs from View A)
        and sends to VLM for stage classification.

        If the embryo was previously marked as no_object, perception is skipped
        except for periodic rechecks (every NO_OBJECT_RECHECK_INTERVAL timepoints).

        Parameters
        ----------
        embryo_id : str
            Embryo identifier
        timepoint : int
            Current timepoint number
        volume : ndarray
            Volume data - can be:
            - 4D: (Views, Z, Y, X) - uses View A (index 0)
            - 3D: (Z, Y, X) - may have dual-view in X dimension
            - 2D: (Y, X) - single projection
        embryo_state : EmbryoAcquisitionState
            Current acquisition state
        """
        # Check if we should skip perception for no_object embryos
        if embryo_state.no_object_since_timepoint is not None:
            timepoints_since_no_object = timepoint - embryo_state.no_object_since_timepoint
            if timepoints_since_no_object % self.NO_OBJECT_RECHECK_INTERVAL != 0:
                # Calculate next recheck timepoint
                next_recheck = embryo_state.no_object_since_timepoint + (
                    (timepoints_since_no_object // self.NO_OBJECT_RECHECK_INTERVAL + 1)
                    * self.NO_OBJECT_RECHECK_INTERVAL
                )
                timepoints_until_recheck = next_recheck - timepoint

                logger.debug(
                    f"Skipping perception for {embryo_id} t={timepoint} "
                    f"(no_object since t={embryo_state.no_object_since_timepoint}, "
                    f"next recheck at t={next_recheck})"
                )

                # Emit skipped event for viz server
                self._emit_event(EventType.DETECTOR_EVALUATED, {
                    'embryo_id': embryo_id,
                    'timepoint': timepoint,
                    'detector_name': 'perception',
                    'stage': 'no_object',
                    'is_hatching': False,
                    'confidence': 1.0,
                    'reasoning': f"Skipped (empty field). Rechecking in {timepoints_until_recheck} timepoint{'s' if timepoints_until_recheck != 1 else ''}.",
                    'is_transitional': False,
                    'transition_between': None,
                    'skipped': True,  # Flag to indicate this was skipped
                })
                return
            else:
                logger.info(
                    f"Rechecking no_object embryo {embryo_id} at t={timepoint}"
                )

        try:
            import numpy as np
            from .perception.projection import (
                projection_three_view,
                compute_crop_bounds,
                apply_crop_bounds,
                image_to_base64,
                normalize_image,
            )

            # Handle 4D volumes: extract View A (first view)
            if volume.ndim == 4:
                # Shape: (Views, Z, Y, X)
                view_a = volume[0]  # Extract View A -> (Z, Y, X)
            else:
                view_a = volume

            # Handle 3D volumes
            if view_a.ndim == 3:
                z_depth, height, width = view_a.shape

                # Check if width contains dual-view data (X = 2*width, views side-by-side)
                # diSPIM format has width roughly 4x height when dual-view
                if width > height * 2:
                    # Extract View A (left half)
                    view_a = view_a[:, :, :width // 2]

                # Auto-crop to embryo region
                bounds = compute_crop_bounds(view_a)
                cropped = apply_crop_bounds(view_a, bounds)

                # Generate three-view projection
                three_view_img, _ = projection_three_view(cropped)
                image_b64 = image_to_base64(three_view_img)

            elif view_a.ndim == 2:
                # Single 2D image - use as-is
                img = normalize_image(view_a)
                image_b64 = image_to_base64(img)
            else:
                logger.warning(f"Unexpected volume dimensions: {volume.ndim}")
                return

            # Run perception (pass view_a volume for view_embryo tool support)
            result = await self.perception_manager.process_image(
                embryo_id=embryo_id,
                timepoint=timepoint,
                image_b64=image_b64,
                volume=view_a,
            )

            # Track no_object state for skipping future perception
            if result.stage == "no_object":
                if embryo_state.no_object_since_timepoint is None:
                    embryo_state.no_object_since_timepoint = timepoint
                    logger.info(
                        f"Embryo {embryo_id} marked as no_object at t={timepoint}, "
                        f"will skip perception until recheck at t={timepoint + self.NO_OBJECT_RECHECK_INTERVAL}"
                    )
            else:
                # Object found - clear no_object state if it was set
                if embryo_state.no_object_since_timepoint is not None:
                    logger.info(
                        f"Embryo {embryo_id} now has object (stage={result.stage}) at t={timepoint}, "
                        f"resuming normal perception (was no_object since t={embryo_state.no_object_since_timepoint})"
                    )
                    embryo_state.no_object_since_timepoint = None

            # Persist trace to JSON file (if trace directory is available)
            if self._trace_dir:
                try:
                    self._write_trace_file(embryo_id, timepoint, result)
                except Exception as persist_error:
                    logger.warning(f"Failed to write trace file: {persist_error}")

            # Emit perception event for viz server
            event_data = {
                'embryo_id': embryo_id,
                'timepoint': timepoint,
                'detector_name': 'perception',
                'stage': result.stage,
                'is_hatching': result.is_hatching,
                'confidence': result.confidence,
                'reasoning': result.reasoning,
                'is_transitional': result.is_transitional,
                'transition_between': result.transition_between,
            }

            # Include volume/projection UIDs for viz server image linking
            if volume_uids:
                event_data['volume_uid'] = volume_uids.get('volume_uid')
                event_data['projection_uid'] = volume_uids.get('projection_uid')

            # Add observed features if available
            if result.observed_features:
                event_data['observed_features'] = {
                    'shape': result.observed_features.shape,
                    'curvature': result.observed_features.curvature,
                    'shell_status': result.observed_features.shell_status,
                    'body_segments': result.observed_features.body_segments_visible,
                    'emergence': result.observed_features.emergence,
                }

            # Add contrastive reasoning if available
            if result.contrastive_reasoning:
                event_data['contrastive_reasoning'] = {
                    'why_not_previous': result.contrastive_reasoning.why_not_previous_stage,
                    'why_not_next': result.contrastive_reasoning.why_not_next_stage,
                }

            # Add reasoning trace if available (from interleaved reasoning)
            if result.reasoning_trace:
                event_data['reasoning_trace'] = result.reasoning_trace.to_dict()

            # Add temporal analysis if available (for detecting arrested/stalled embryos)
            session = self.perception_manager.sessions.get(embryo_id)
            if session:
                temporal = session.compute_temporal_analysis()
                if temporal:
                    event_data['temporal_analysis'] = temporal.to_dict()

            self._emit_event(EventType.DETECTOR_EVALUATED, event_data)

            # Check for hatching event
            if result.is_hatching:
                self._emit_event(EventType.HATCHING_DETECTED, {
                    'embryo_id': embryo_id,
                    'timepoint': timepoint,
                    'detector_name': 'hatching',
                    'stage': result.stage,
                    'confidence': result.confidence,
                })

            # Check interval rules based on stage
            self._check_interval_rules(
                embryo_id=embryo_id,
                stage=result.stage,
            )

            logger.info(
                f"[{embryo_id}] T{timepoint}: stage={result.stage}, "
                f"hatching={result.is_hatching}, confidence={result.confidence:.0%}"
            )

        except Exception as e:
            logger.error(f"Perception failed for {embryo_id}: {e}")
            # Don't raise - perception failure shouldn't stop acquisition

    def _write_trace_file(self, embryo_id: str, timepoint: int, result) -> Path:
        """
        Write perception trace to JSON file.

        File format: {embryo_id}_T{timepoint:04d}.json
        Location: D:/Gently/traces/{session_id}/

        Parameters
        ----------
        embryo_id : str
            Embryo identifier
        timepoint : int
            Timepoint number
        result : PerceptionResult
            Perception result with stage, confidence, and traces

        Returns
        -------
        Path
            Path to the written trace file
        """
        timestamp = datetime.now()

        # Build trace data
        trace_data = {
            # Identifiers
            "session_id": self._session_id,
            "embryo_id": embryo_id,
            "timepoint": timepoint,
            "timestamp": timestamp.isoformat(),

            # Results
            "predicted_stage": result.stage,
            "confidence": result.confidence,
            "reasoning": result.reasoning,
            "is_hatching": getattr(result, 'is_hatching', False),
            "is_transitional": getattr(result, 'is_transitional', False),
            "transition_between": getattr(result, 'transition_between', None),
        }

        # Add observed features if available
        if hasattr(result, 'observed_features') and result.observed_features:
            trace_data["observed_features"] = {
                'shape': result.observed_features.shape,
                'curvature': result.observed_features.curvature,
                'shell_status': result.observed_features.shell_status,
                'body_segments': getattr(result.observed_features, 'body_segments_visible', None),
                'emergence': result.observed_features.emergence,
            }

        # Add contrastive reasoning if available
        if hasattr(result, 'contrastive_reasoning') and result.contrastive_reasoning:
            trace_data["contrastive_reasoning"] = {
                'why_not_previous': result.contrastive_reasoning.why_not_previous_stage,
                'why_not_next': result.contrastive_reasoning.why_not_next_stage,
            }

        # Add full reasoning trace if available
        if hasattr(result, 'reasoning_trace') and result.reasoning_trace:
            trace_data["reasoning_trace"] = result.reasoning_trace.to_dict()

        # Add verification info if available
        if hasattr(result, 'verification_triggered') and result.verification_triggered:
            trace_data["verification"] = {
                'triggered': True,
                'result': result.verification_result.to_dict() if result.verification_result else None,
            }

        # Write JSON file
        filename = f"{embryo_id}_T{timepoint:04d}.json"
        file_path = self._trace_dir / filename

        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(trace_data, f, indent=2, ensure_ascii=False)

        logger.debug(f"Wrote trace: {file_path.name}")
        return file_path
