"""
Data models for the Timelapse Orchestrator.

Defines stop conditions, interval rules, embryo acquisition state,
and timelapse status — all pure data with no dependency on the orchestrator.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any

from gently.organisms import get_organism


class StopConditionType(Enum):
    """Types of stop conditions for embryo acquisition"""

    MANUAL = "manual"  # Stop only when user says
    STAGE_BASED = "stage_based"  # Stop when any of target stages reached
    FIXED_TIMEPOINTS = "fixed_timepoints"  # Stop after N timepoints
    DURATION = "duration"  # Stop after X hours
    ALL_TERMINAL = "all_terminal"  # Stop when all embryos reach terminal stage
    # Phase 8: stop when every role='test' embryo has hatched
    # (via the dopaminergic detector setting hatching_status.hatched=True).
    ALL_TEST_HATCHED = "all_test_hatched"
    # Legacy aliases (kept for backward compatibility with serialized data)
    HATCHING = "hatching"
    COMMA_STAGE = "comma_stage"


@dataclass
class IntervalRule:
    """
    Rule for automatically adjusting acquisition interval.

    Triggers when a condition is met (detector fires, stage reached, etc.)
    Scope: a match for embryo X applies the cadence change to embryo X only.
    ``applies_to`` filters which embryos this rule listens to — it is not a
    fan-out target list.
    """

    name: str
    trigger_detector: str | None = None  # Detector name that triggers this rule
    trigger_stage: str | None = None  # Stage name that triggers (comma, pretzel, etc.)
    new_interval_seconds: float = 30.0  # New interval when triggered
    applies_to: list[str] | None = None  # Embryo IDs this rule listens to (None = all)
    confirm_timepoints: int = 0  # require N consecutive trigger matches before firing
    one_time: bool = True  # Only apply once per embryo

    def matches(
        self,
        embryo_id: str,
        detector_name: str | None = None,
        stage: str | None = None,
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
class PowerRule:
    """
    Rule for automatically adjusting per-line laser power.

    Phase 5 reactive control. Sticky-monotonic with floor/ceiling, designed
    for the dopaminergic-reporter experiment's "ramp down on saturation"
    pattern: trigger on intensity_level == SATURATING, step ``laser_power_488_pct``
    DOWN by ``step_pct`` until ``floor_pct``, never going up.

    The hard safety limit at DiSPIMLightSource.POWER_LIMITS_PCT is the
    bottom-line bound; this is the soft control on top.
    """

    name: str
    wavelength: int = 488
    trigger_detector: str | None = None
    trigger_intensity_levels: list[str] | None = None  # e.g. ["SATURATING"]
    trigger_stage: str | None = None
    step_pct: float = 1.0  # how much to change per firing
    floor_pct: float = 2.0  # never go below
    ceiling_pct: float = 6.0  # never go above
    direction: str = "down"  # "down" (sticky-downward) or "up"
    applies_to: list[str] | None = None
    confirm_timepoints: int = 0  # require N consecutive firings before applying
    one_time: bool = False  # default: fire repeatedly for ramps

    def matches(
        self,
        embryo_id: str,
        detector_name: str | None = None,
        stage: str | None = None,
        intensity_level: str | None = None,
    ) -> bool:
        if self.applies_to and embryo_id not in self.applies_to:
            return False
        if self.trigger_detector and detector_name != self.trigger_detector:
            return False
        # Predicates are OR-combined when multiple are set, AND-combined
        # with the detector filter above (already applied).
        if self.trigger_intensity_levels and intensity_level in self.trigger_intensity_levels:
            return True
        if self.trigger_stage and stage == self.trigger_stage:
            return True
        # If only trigger_detector was set (no further filter), require an
        # explicit predicate match to avoid firing on every detection.
        return False

    def next_power(self, current_pct: float) -> float:
        """Return the post-step power %, clipped to [floor, ceiling]."""
        if self.direction == "down":
            return max(self.floor_pct, current_pct - self.step_pct)
        return min(self.ceiling_pct, current_pct + self.step_pct)


@dataclass
class BurstRule:
    """Rule for auto-queuing a one-shot burst acquisition.

    Fires when an embryo's detection matches all specified predicates
    (intensity + structure). One-time-per-embryo semantics are enforced
    downstream by ``queue_burst`` (which gates on ``_burst_applied``),
    so this rule doesn't need its own ``one_time`` flag.
    """

    name: str
    trigger_detector: str | None = None
    trigger_intensity_levels: list[str] | None = None  # AND-combined predicate
    trigger_structure_qualities: list[str] | None = None  # AND-combined predicate
    frames: int = 60
    mode: str = "1hz"  # "1hz" | "asap"
    num_slices: int = 1
    applies_to: list[str] | None = None  # listen-filter
    confirm_timepoints: int = 0  # require N consecutive matches before firing

    def matches(
        self,
        embryo_id: str,
        detector_name: str | None = None,
        intensity_level: str | None = None,
        structure_quality: str | None = None,
    ) -> bool:
        if self.applies_to and embryo_id not in self.applies_to:
            return False
        if self.trigger_detector and detector_name != self.trigger_detector:
            return False
        # AND-combine all specified predicates. At least one must be set
        # to avoid firing on every detection.
        if self.trigger_intensity_levels:
            if intensity_level not in self.trigger_intensity_levels:
                return False
        if self.trigger_structure_qualities:
            if structure_quality not in self.trigger_structure_qualities:
                return False
        if not self.trigger_intensity_levels and not self.trigger_structure_qualities:
            return False
        return True


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
    target_stages: set[str] | None = None  # Stages that satisfy STAGE_BASED condition
    confirm_timepoints: int = 0  # Extra timepoints to acquire after detection
    additional_conditions: list["StopCondition"] = field(default_factory=list)

    def add_condition(self, condition: "StopCondition") -> None:
        """Add another stop condition (OR logic)."""
        self.additional_conditions.append(condition)

    def all_conditions(self) -> list["StopCondition"]:
        """Get all conditions including self (flattened)."""
        return [self] + self.additional_conditions

    def describe(self) -> str:
        """Human-readable description of the stop condition(s)."""

        def _describe_single(cond: "StopCondition") -> str:
            confirm_suffix = f"+{cond.confirm_timepoints}tp" if cond.confirm_timepoints > 0 else ""
            if cond.condition_type == StopConditionType.MANUAL:
                return "manual"
            elif cond.condition_type in (
                StopConditionType.STAGE_BASED,
                StopConditionType.HATCHING,
                StopConditionType.COMMA_STAGE,
            ):
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
    def until_hatching(cls, confirm_timepoints: int = 0) -> "StopCondition":
        """Stop when hatching is detected (backward-compatible convenience method)."""
        organism = get_organism()
        return cls(
            StopConditionType.STAGE_BASED,
            target_stages=organism.STOP_CONDITIONS["hatching"],
            confirm_timepoints=confirm_timepoints,
        )

    @classmethod
    def until_comma(cls, confirm_timepoints: int = 0) -> "StopCondition":
        """Stop when comma stage is detected (backward-compatible convenience method)."""
        organism = get_organism()
        return cls(
            StopConditionType.STAGE_BASED,
            target_stages=organism.STOP_CONDITIONS["comma"],
            confirm_timepoints=confirm_timepoints,
        )

    @classmethod
    def fixed_timepoints(cls, n: int) -> "StopCondition":
        return cls(StopConditionType.FIXED_TIMEPOINTS, value=n)

    @classmethod
    def duration_hours(cls, hours: float) -> "StopCondition":
        return cls(StopConditionType.DURATION, value=hours)

    @classmethod
    def manual(cls) -> "StopCondition":
        return cls(StopConditionType.MANUAL)

    @classmethod
    def all_test_hatched(cls, confirm_timepoints: int = 0) -> "StopCondition":
        """Stop when EVERY role='test' embryo's ``hatching_status.hatched``
        flag is True (set by the dopaminergic detector / Phase 2 path)."""
        return cls(
            StopConditionType.ALL_TEST_HATCHED,
            confirm_timepoints=confirm_timepoints,
        )

    @classmethod
    def composite(cls, *conditions: "StopCondition") -> "StopCondition":
        """Create a composite stop condition from multiple conditions (OR logic)."""
        if not conditions:
            return cls.manual()
        primary = conditions[0]
        for cond in conditions[1:]:
            primary.add_condition(cond)
        return primary

    @classmethod
    def parse(cls, spec: str) -> "StopCondition":
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

        def _parse_single(s: str) -> "StopCondition":
            s = s.strip().lower()

            # Check for confirmation timepoints suffix: "hatching+3" or "comma+5"
            confirm_timepoints = 0
            if "+" in s:
                base, confirm_str = s.rsplit("+", 1)
                try:
                    confirm_timepoints = int(confirm_str)
                    s = base
                except ValueError:
                    pass

            if s == "manual":
                return cls.manual()
            elif s in ("all_test_hatched", "test_hatched"):
                return cls.all_test_hatched(confirm_timepoints=confirm_timepoints)
            elif s.startswith("timepoints:"):
                n = int(s.split(":")[1])
                return cls.fixed_timepoints(n)
            elif s.startswith("duration:"):
                hours_str = s.split(":")[1]
                if hours_str.endswith("h"):
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

        parts = spec.split("|")
        conditions = [_parse_single(p) for p in parts]
        return cls.composite(*conditions)


# EmbryoAcquisitionState was removed in Phase 1.5 of the consolidation.
# Per-embryo runtime fields (stop_condition, is_complete, completion_reason,
# error_count, last_error, detection_triggered_at, detection_type,
# no_object_since_timepoint, timepoints_acquired) now live on EmbryoState
# (gently/harness/state.py). The orchestrator's `_embryo_states` dict holds
# direct references to the agent's EmbryoState instances — one source of truth.


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
    started_at: datetime | None
    # Dict of embryo_id -> EmbryoState reference (from agent.experiment.embryos).
    # Typed Any to avoid importing harness/ from models/ (dependency direction).
    embryos: dict[str, Any]
    total_timepoints: int = 0
    current_round: int = 0
    interval_seconds: float = 120.0
    next_round_time: datetime | None = None
    seconds_until_next_round: float | None = None
    error_message: str | None = None

    def to_dict(self) -> dict:
        """Serialize for display"""
        active = [e for e in self.embryos.values() if not e.is_complete]
        completed = [e for e in self.embryos.values() if e.is_complete]

        return {
            "status": self.status.value,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "duration_minutes": (datetime.now() - self.started_at).total_seconds() / 60
            if self.started_at
            else 0,
            "total_timepoints": self.total_timepoints,
            "current_round": self.current_round,
            "interval_seconds": self.interval_seconds,
            "next_round_time": self.next_round_time.isoformat() if self.next_round_time else None,
            "seconds_until_next_round": self.seconds_until_next_round,
            "active_embryos": len(active),
            "completed_embryos": len(completed),
            "embryo_details": {
                eid: {
                    "timepoints": e.timepoints_acquired,
                    "is_complete": e.is_complete,
                    "completion_reason": e.completion_reason,
                }
                for eid, e in self.embryos.items()
            },
            "error": self.error_message,
        }
