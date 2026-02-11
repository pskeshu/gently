"""
Shared types for the agent core.

Extracted from daemon/types.py and daemon/clock.py so that the agent core
can stand alone without the daemon architecture.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from ..context import ContextUpdates


class ThinkTrigger(Enum):
    """What triggered a thinking cycle."""
    INTERVAL = "interval"
    EVENT = "event"
    SURPRISE = "surprise"
    WATCHPOINT = "watchpoint"
    USER = "user"
    ESCALATION = "escalation"
    EXPECTATION = "expectation"


class ThinkingMode(Enum):
    """Depth of thinking."""
    FAST = "fast"
    MODERATE = "moderate"
    DEEP = "deep"


def model_name_for_mode(mode: ThinkingMode) -> str:
    """Get the Claude model name for a thinking mode."""
    mapping = {
        ThinkingMode.FAST: "claude-haiku-4-5-20251001",
        ThinkingMode.MODERATE: "claude-sonnet-4-5-20250929",
        ThinkingMode.DEEP: "claude-opus-4-6",
    }
    return mapping[mode]


@dataclass
class WorldState:
    """
    Current state of the world as seen by the agent.

    Sampled fresh each thinking cycle.
    """
    current_time: datetime = field(default_factory=datetime.now)
    user_present: bool = False
    microscope_status: Optional[Dict[str, Any]] = None
    recent_events: List[Any] = field(default_factory=list)
    session_id: Optional[str] = None
    context_richness: float = 0.5

    def summary(self) -> str:
        """One-line summary of world state."""
        parts = [f"time={self.current_time.strftime('%H:%M:%S')}"]
        if self.user_present:
            parts.append("user_present")
        if self.microscope_status:
            status = self.microscope_status.get("status", "unknown")
            parts.append(f"microscope={status}")
        parts.append(f"recent_events={len(self.recent_events)}")
        parts.append(f"richness={self.context_richness:.2f}")
        return " ".join(parts)


@dataclass
class ThinkResult:
    """Result from a thinking cycle."""
    actions: List[Dict[str, Any]] = field(default_factory=list)
    context_updates: ContextUpdates = field(default_factory=ContextUpdates)
    reasoning: str = ""
    observations_noted: List[str] = field(default_factory=list)
    model_used: str = ""
    duration_ms: float = 0.0
    mode: ThinkingMode = ThinkingMode.MODERATE
    trigger: ThinkTrigger = ThinkTrigger.INTERVAL
