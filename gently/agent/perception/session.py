"""
Simple perception session - just observations over time.

No probability distributions, no state machines, no weighted averaging.
Just: what did the VLM see at each timepoint?
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, TYPE_CHECKING

import numpy as np

from .stages import STAGES, STAGE_CRITERIA, TRANSITION_ZONES, DevelopmentalStage


@dataclass
class Observation:
    """A single observation from the VLM."""

    timepoint: int
    timestamp: datetime
    stage: str  # see stages.py for valid values
    is_hatching: bool
    confidence: float
    reasoning: str  # VLM's explanation

    # New fields for improved perception
    is_transitional: bool = False
    transition_between: Optional[List[str]] = None  # e.g., ["comma", "1.5fold"]


@dataclass
class TemporalAnalysis:
    """
    Temporal metrics computed from observation history.

    Used to detect developmental arrest (dead embryos) and provide
    temporal context to the VLM for more informed classification.
    """

    # Current state
    current_stage: Optional[str]
    time_in_current_stage_min: float  # Minutes at current stage
    observations_in_current_stage: int  # Number of observations at this stage

    # Expected durations (from stages.py)
    expected_duration_min: Optional[float]  # Expected duration for current stage
    overtime_ratio: float  # time_in_stage / expected_duration (>1.0 = overtime)

    # Progression metrics
    last_stage_change_timepoint: Optional[int]  # Timepoint when stage last changed
    timepoints_since_change: int  # How many timepoints at same stage
    total_session_duration_min: float  # Total session duration

    # Arrest indicators
    is_potentially_arrested: bool  # True if significantly overtime
    arrest_confidence: float  # 0.0-1.0, higher = more likely arrested
    arrest_reason: Optional[str]  # Human-readable explanation

    # Transitional state tracking
    is_currently_transitional: bool = False  # Last observation was transitional
    consecutive_transitional_count: int = 0  # How many transitional observations in a row
    transition_between: Optional[List[str]] = None  # Current transition if any
    last_confidence: float = 1.0  # Confidence of last observation
    suggest_tool_use: bool = False  # Hint that tool use would be helpful

    def to_dict(self) -> Dict[str, Any]:
        return {
            "current_stage": self.current_stage,
            "time_in_current_stage_min": round(self.time_in_current_stage_min, 1),
            "observations_in_current_stage": self.observations_in_current_stage,
            "expected_duration_min": self.expected_duration_min,
            "overtime_ratio": round(self.overtime_ratio, 2),
            "last_stage_change_timepoint": self.last_stage_change_timepoint,
            "timepoints_since_change": self.timepoints_since_change,
            "total_session_duration_min": round(self.total_session_duration_min, 1),
            "is_potentially_arrested": self.is_potentially_arrested,
            "arrest_confidence": round(self.arrest_confidence, 2),
            "arrest_reason": self.arrest_reason,
            "is_currently_transitional": self.is_currently_transitional,
            "consecutive_transitional_count": self.consecutive_transitional_count,
            "transition_between": self.transition_between,
            "last_confidence": round(self.last_confidence, 2),
            "suggest_tool_use": self.suggest_tool_use,
        }


@dataclass
class PerceptionSession:
    """
    Simple session tracking observations over time.

    No complex belief state - just a list of what the VLM said.
    """

    embryo_id: str
    created_at: datetime = field(default_factory=datetime.now)

    # Just a list of observations
    observations: List[Observation] = field(default_factory=list)

    # Track first hatching detection
    hatching_started_at: Optional[int] = None
    hatching_complete_at: Optional[int] = None

    # Image cache for interleaved reasoning (timepoint -> base64 image)
    # Stores recent images so VLM can request previous timepoints
    _image_cache: Dict[int, str] = field(default_factory=dict)
    _image_cache_max_size: int = 10  # Keep last N images

    def add_observation(
        self,
        timepoint: int,
        stage: str,
        is_hatching: bool,
        confidence: float = 0.0,
        reasoning: str = "",
        is_transitional: bool = False,
        transition_between: Optional[List[str]] = None,
    ) -> None:
        """Add a new observation."""
        obs = Observation(
            timepoint=timepoint,
            timestamp=datetime.now(),
            stage=stage,
            is_hatching=is_hatching,
            confidence=confidence,
            reasoning=reasoning,
            is_transitional=is_transitional,
            transition_between=transition_between,
        )
        self.observations.append(obs)

        # Track hatching milestones
        if is_hatching and self.hatching_started_at is None:
            self.hatching_started_at = timepoint

        if stage == "hatched" and self.hatching_complete_at is None:
            self.hatching_complete_at = timepoint

    def get_recent_observations(self, n: int = 3) -> List[Observation]:
        """Get the N most recent observations."""
        return self.observations[-n:] if self.observations else []

    def store_image(self, timepoint: int, image_b64: str) -> None:
        """Store an image in the cache for potential future reference."""
        self._image_cache[timepoint] = image_b64

        # Evict oldest images if cache is full
        if len(self._image_cache) > self._image_cache_max_size:
            oldest_tp = min(self._image_cache.keys())
            del self._image_cache[oldest_tp]

    def get_image(self, timepoint: int) -> Optional[str]:
        """Get a stored image by timepoint."""
        return self._image_cache.get(timepoint)

    def get_previous_image(self, current_timepoint: int, offset: int = 1) -> Optional[tuple]:
        """
        Get image from a previous timepoint.

        Returns (timepoint, image_b64) or None if not available.
        """
        target_tp = current_timepoint - offset
        image = self._image_cache.get(target_tp)
        if image:
            return (target_tp, image)
        return None

    def get_available_timepoints(self) -> List[int]:
        """Get list of timepoints with cached images."""
        return sorted(self._image_cache.keys())

    def get_current_stage(self) -> Optional[str]:
        """Get the most recently observed stage."""
        if self.observations:
            return self.observations[-1].stage
        return None

    def is_complete(self) -> bool:
        """Check if embryo has hatched."""
        return self.hatching_complete_at is not None

    def compute_temporal_analysis(self) -> Optional[TemporalAnalysis]:
        """
        Compute temporal metrics from observation history.

        Called before each VLM perception call to provide temporal context.
        Detects potential developmental arrest based on time at current stage.

        Returns:
            TemporalAnalysis with metrics, or None if no observations yet.
        """
        if not self.observations:
            return None

        current_stage = self.get_current_stage()
        now = datetime.now()

        # Find when current stage started (walk backwards through observations)
        stage_start_time = None
        stage_start_timepoint = None
        observations_at_stage = 0

        for obs in reversed(self.observations):
            if obs.stage == current_stage:
                stage_start_time = obs.timestamp
                stage_start_timepoint = obs.timepoint
                observations_at_stage += 1
            else:
                break

        # Calculate time in current stage
        if stage_start_time:
            time_in_stage = (now - stage_start_time).total_seconds() / 60.0
        else:
            time_in_stage = 0.0

        # Get expected duration for this stage from STAGE_CRITERIA
        stage_info = STAGE_CRITERIA.get(current_stage, {})
        expected_duration = stage_info.get("typical_duration_min")

        # Calculate overtime ratio
        if expected_duration and expected_duration > 0:
            overtime_ratio = time_in_stage / expected_duration
        else:
            overtime_ratio = 0.0

        # Total session duration
        total_duration = (now - self.created_at).total_seconds() / 60.0

        # Calculate timepoints since last stage change
        if stage_start_timepoint is not None and self.observations:
            current_tp = self.observations[-1].timepoint
            timepoints_since_change = current_tp - stage_start_timepoint
        else:
            timepoints_since_change = 0

        # Arrest detection logic
        is_arrested = False
        arrest_confidence = 0.0
        arrest_reason = None

        # Arrest criteria:
        # 1. Early stage > 225 min (2.5x expected 90 min)
        # 2. Any stage > 3x expected duration
        # 3. 30+ consecutive observations at same early-stage
        if current_stage == "early":
            if time_in_stage > 225:  # 2.5x of 90 min
                is_arrested = True
                arrest_confidence = min(1.0, (time_in_stage - 225) / 100)
                arrest_reason = f"Early stage for {time_in_stage:.0f} min (expected ~90 min)"
        elif current_stage == "arrested":
            # Already classified as arrested
            is_arrested = True
            arrest_confidence = 1.0
            arrest_reason = "Previously classified as arrested"
        elif expected_duration:
            threshold = expected_duration * 3.0
            if time_in_stage > threshold:
                is_arrested = True
                arrest_confidence = min(1.0, (time_in_stage - threshold) / (expected_duration * 2))
                arrest_reason = f"{current_stage} stage for {time_in_stage:.0f} min (expected ~{expected_duration:.0f} min)"

        # Also check observation count (in case timing is off due to gaps)
        if observations_at_stage >= 30 and current_stage in ("early", "comma"):
            if not is_arrested:
                is_arrested = True
                arrest_reason = f"No progression for {observations_at_stage} consecutive observations"
            arrest_confidence = max(arrest_confidence, 0.8)

        # Compute transitional state tracking
        last_obs = self.observations[-1] if self.observations else None
        is_transitional = last_obs.is_transitional if last_obs else False
        transition_between = last_obs.transition_between if last_obs else None
        last_confidence = last_obs.confidence if last_obs else 1.0

        # Count consecutive transitional observations
        consecutive_transitional = 0
        for obs in reversed(self.observations):
            if obs.is_transitional:
                consecutive_transitional += 1
            else:
                break

        # Determine if we should suggest tool use
        suggest_tool_use = False
        if is_transitional:
            suggest_tool_use = True  # Always suggest for transitional
        elif last_confidence < 0.7:
            suggest_tool_use = True  # Low confidence needs reference
        elif observations_at_stage <= 2:
            suggest_tool_use = True  # Early in a stage, good to verify
        elif consecutive_transitional >= 3:
            suggest_tool_use = True  # Stuck in transitional state

        # Stricter arrest detection for stuck transitions (>45 min is concerning)
        if consecutive_transitional >= 10:
            if not is_arrested:
                is_arrested = True
                arrest_reason = f"Stuck in transitional state for {consecutive_transitional} observations"
            arrest_confidence = max(arrest_confidence, 0.6)

        return TemporalAnalysis(
            current_stage=current_stage,
            time_in_current_stage_min=time_in_stage,
            observations_in_current_stage=observations_at_stage,
            expected_duration_min=expected_duration,
            overtime_ratio=overtime_ratio,
            last_stage_change_timepoint=stage_start_timepoint,
            timepoints_since_change=timepoints_since_change,
            total_session_duration_min=total_duration,
            is_potentially_arrested=is_arrested,
            arrest_confidence=arrest_confidence,
            arrest_reason=arrest_reason,
            is_currently_transitional=is_transitional,
            consecutive_transitional_count=consecutive_transitional,
            transition_between=transition_between,
            last_confidence=last_confidence,
            suggest_tool_use=suggest_tool_use,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for JSON storage."""
        return {
            "embryo_id": self.embryo_id,
            "created_at": self.created_at.isoformat(),
            "observations": [
                {
                    "timepoint": o.timepoint,
                    "timestamp": o.timestamp.isoformat(),
                    "stage": o.stage,
                    "is_hatching": o.is_hatching,
                    "confidence": o.confidence,
                    "reasoning": o.reasoning,
                    "is_transitional": o.is_transitional,
                    "transition_between": o.transition_between,
                }
                for o in self.observations
            ],
            "hatching_started_at": self.hatching_started_at,
            "hatching_complete_at": self.hatching_complete_at,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PerceptionSession":
        """Deserialize from JSON."""
        session = cls(
            embryo_id=data["embryo_id"],
            created_at=datetime.fromisoformat(data["created_at"]),
            hatching_started_at=data.get("hatching_started_at"),
            hatching_complete_at=data.get("hatching_complete_at"),
        )
        session.observations = [
            Observation(
                timepoint=o["timepoint"],
                timestamp=datetime.fromisoformat(o["timestamp"]),
                stage=o["stage"],
                is_hatching=o["is_hatching"],
                confidence=o.get("confidence", 0.0),
                reasoning=o.get("reasoning", ""),
                is_transitional=o.get("is_transitional", False),
                transition_between=o.get("transition_between"),
            )
            for o in data.get("observations", [])
        ]
        return session


@dataclass
class ObservedFeatures:
    """Features observed in the image before classification."""

    shape: str = ""
    curvature: str = ""
    shell_status: str = ""
    body_segments_visible: str = ""
    emergence: str = ""


@dataclass
class ContrastiveReasoning:
    """Why this is NOT adjacent stages."""

    why_not_previous_stage: str = ""
    why_not_next_stage: str = ""


@dataclass
class ReasoningStep:
    """A single step in the interleaved reasoning process."""

    step_type: str  # "initial_analysis", "tool_call", "tool_result", "final_decision"
    content: str  # The reasoning text or tool call details
    timestamp: datetime = field(default_factory=datetime.now)

    # For tool calls
    tool_name: Optional[str] = None  # "view_previous_timepoint", "view_reference_example"
    tool_input: Optional[Dict[str, Any]] = None  # Tool parameters
    tool_result_summary: Optional[str] = None  # Brief description of what was shown

    # For image references (so viz can display them)
    image_timepoint: Optional[int] = None  # Which timepoint's image was viewed
    image_type: Optional[str] = None  # "previous_timepoint", "reference_example"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step_type": self.step_type,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "tool_name": self.tool_name,
            "tool_input": self.tool_input,
            "tool_result_summary": self.tool_result_summary,
            "image_timepoint": self.image_timepoint,
            "image_type": self.image_type,
        }


@dataclass
class ReasoningTrace:
    """Complete trace of the interleaved reasoning process."""

    steps: List[ReasoningStep] = field(default_factory=list)
    total_tool_calls: int = 0
    tools_used: List[str] = field(default_factory=list)

    def add_step(self, step: ReasoningStep) -> None:
        self.steps.append(step)
        if step.step_type == "tool_call" and step.tool_name:
            self.total_tool_calls += 1
            if step.tool_name not in self.tools_used:
                self.tools_used.append(step.tool_name)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "steps": [s.to_dict() for s in self.steps],
            "total_tool_calls": self.total_tool_calls,
            "tools_used": self.tools_used,
        }


@dataclass
class PerceptionResult:
    """Result from a single perception call."""

    stage: str
    is_hatching: bool
    confidence: float
    reasoning: str

    # New fields for improved perception
    observed_features: Optional[ObservedFeatures] = None
    contrastive_reasoning: Optional[ContrastiveReasoning] = None
    is_transitional: bool = False
    transition_between: Optional[List[str]] = None  # e.g., ["comma", "1.5fold"]

    # Interleaved reasoning trace (for observability)
    reasoning_trace: Optional[ReasoningTrace] = None

    # For automation
    should_stop: bool = False  # True if hatched
