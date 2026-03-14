"""
Data models for the Timelapse Orchestrator.

Defines stop conditions, interval rules, embryo acquisition state,
and timelapse status — all pure data with no dependency on the orchestrator.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Set

from gently.organisms import get_organism


class StopConditionType(Enum):
    """Types of stop conditions for embryo acquisition"""
    MANUAL = "manual"                    # Stop only when user says
    STAGE_BASED = "stage_based"          # Stop when any of target stages reached
    FIXED_TIMEPOINTS = "fixed_timepoints"  # Stop after N timepoints
    DURATION = "duration"                # Stop after X hours
    ALL_TERMINAL = "all_terminal"        # Stop when all embryos reach terminal stage
    # Legacy aliases (kept for backward compatibility with serialized data)
    HATCHING = "hatching"
    COMMA_STAGE = "comma_stage"


@dataclass
class IntervalRule:
    """
    Rule for automatically adjusting acquisition interval.

    Triggers when a condition is met (detector fires, stage reached, etc.)
    """
    name: str
    trigger_detector: Optional[str] = None  # Detector name that triggers this rule
    trigger_stage: Optional[str] = None     # Stage name that triggers (comma, pretzel, etc.)
    new_interval_seconds: float = 30.0      # New interval when triggered
    applies_to: Optional[List[str]] = None  # Embryo IDs (None = all)
    one_time: bool = True                   # Only apply once per embryo

    def matches(
        self,
        embryo_id: str,
        detector_name: Optional[str] = None,
        stage: Optional[str] = None,
    ) -> bool:
        """Check if this rule should trigger"""
        if self.applies_to and embryo_id not in self.applies_to:
            return False

        if self.trigger_detector and detector_name == self.trigger_detector:
            return True
        if self.trigger_stage and stage == self.trigger_stage:
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
    target_stages: Optional[Set[str]] = None  # Stages that satisfy STAGE_BASED condition
    confirm_timepoints: int = 0  # Extra timepoints to acquire after detection
    additional_conditions: List['StopCondition'] = field(default_factory=list)

    def add_condition(self, condition: 'StopCondition') -> None:
        """Add another stop condition (OR logic)."""
        self.additional_conditions.append(condition)

    def all_conditions(self) -> List['StopCondition']:
        """Get all conditions including self (flattened)."""
        return [self] + self.additional_conditions

    def describe(self) -> str:
        """Human-readable description of the stop condition(s)."""
        def _describe_single(cond: 'StopCondition') -> str:
            confirm_suffix = f"+{cond.confirm_timepoints}tp" if cond.confirm_timepoints > 0 else ""
            if cond.condition_type == StopConditionType.MANUAL:
                return "manual"
            elif cond.condition_type in (StopConditionType.STAGE_BASED,
                                         StopConditionType.HATCHING,
                                         StopConditionType.COMMA_STAGE):
                stages_str = ",".join(sorted(cond.target_stages)) if cond.target_stages else "?"
                return f"stages({stages_str}){confirm_suffix}"
            elif cond.condition_type == StopConditionType.FIXED_TIMEPOINTS:
                return f"{cond.value} timepoints"
            elif cond.condition_type == StopConditionType.DURATION:
                return f"{cond.value}h duration"
            elif cond.condition_type == StopConditionType.ALL_TERMINAL:
                return "all_terminal"
            else:
                return str(cond.condition_type.value)

        descriptions = [_describe_single(self)]
        for cond in self.additional_conditions:
            descriptions.append(_describe_single(cond))

        return " OR ".join(descriptions)

    @classmethod
    def until_hatching(cls, confirm_timepoints: int = 0) -> 'StopCondition':
        """Stop when hatching is detected (backward-compatible convenience method)."""
        organism = get_organism()
        return cls(
            StopConditionType.STAGE_BASED,
            target_stages=organism.STOP_CONDITIONS["hatching"],
            confirm_timepoints=confirm_timepoints,
        )

    @classmethod
    def until_comma(cls, confirm_timepoints: int = 0) -> 'StopCondition':
        """Stop when comma stage is detected (backward-compatible convenience method)."""
        organism = get_organism()
        return cls(
            StopConditionType.STAGE_BASED,
            target_stages=organism.STOP_CONDITIONS["comma"],
            confirm_timepoints=confirm_timepoints,
        )

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
        """Create a composite stop condition from multiple conditions (OR logic)."""
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
        Organism-defined stop condition names (from STOP_CONDITIONS dict)
        are resolved automatically.

        Parameters
        ----------
        spec : str
            Specification like "hatching", "duration:10", "hatching|duration:10"
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
                    pass

            if s == 'manual':
                return cls.manual()
            elif s.startswith('timepoints:'):
                n = int(s.split(':')[1])
                return cls.fixed_timepoints(n)
            elif s.startswith('duration:'):
                hours_str = s.split(':')[1]
                if hours_str.endswith('h'):
                    hours_str = hours_str[:-1]
                hours = float(hours_str)
                return cls.duration_hours(hours)
            else:
                organism = get_organism()
                lookup = s.replace("_stage", "")
                if lookup in organism.STOP_CONDITIONS:
                    return cls(
                        StopConditionType.STAGE_BASED,
                        target_stages=organism.STOP_CONDITIONS[lookup],
                        confirm_timepoints=confirm_timepoints,
                    )
                raise ValueError(
                    f"Unknown stop condition: {s}. "
                    f"Available: manual, timepoints:N, duration:Xh, "
                    f"{', '.join(organism.STOP_CONDITIONS.keys())}"
                )

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
    detection_triggered_at: Optional[int] = None
    detection_type: Optional[str] = None
    no_object_since_timepoint: Optional[int] = None


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
