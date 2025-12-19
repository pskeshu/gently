"""
Simple perception session - just observations over time.

No probability distributions, no state machines, no weighted averaging.
Just: what did the VLM see at each timepoint?
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional


@dataclass
class Observation:
    """A single observation from the VLM."""

    timepoint: int
    timestamp: datetime
    stage: str  # early, comma, pretzel, 3fold, hatching, hatched
    is_hatching: bool
    confidence: float
    reasoning: str  # VLM's explanation


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

    def add_observation(
        self,
        timepoint: int,
        stage: str,
        is_hatching: bool,
        confidence: float = 0.0,
        reasoning: str = "",
    ) -> None:
        """Add a new observation."""
        obs = Observation(
            timepoint=timepoint,
            timestamp=datetime.now(),
            stage=stage,
            is_hatching=is_hatching,
            confidence=confidence,
            reasoning=reasoning,
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

    def get_current_stage(self) -> Optional[str]:
        """Get the most recently observed stage."""
        if self.observations:
            return self.observations[-1].stage
        return None

    def is_complete(self) -> bool:
        """Check if embryo has hatched."""
        return self.hatching_complete_at is not None

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
            )
            for o in data.get("observations", [])
        ]
        return session


@dataclass
class PerceptionResult:
    """Result from a single perception call."""

    stage: str
    is_hatching: bool
    confidence: float
    reasoning: str

    # For automation
    should_stop: bool = False  # True if hatched
