"""
Context Store — The agent's mind.

Separate from GentlyStore (raw data), ContextStore holds understanding:
- Intentions: campaigns, projects, session focus
- Understanding: embryo states, learnings
- Observations: synthesized notes, not raw data
- Expectations: predictions about what will happen
- Attention: watchpoints, open questions

This is the scaffolding that enables the agent to remember across
its finite context window.
"""

from .model import (
    Campaign,
    Project,
    SessionIntent,
    Learning,
    Observation,
    Expectation,
    Watchpoint,
    Question,
    EmbryoUnderstanding,
    Intentions,
    Understanding,
    Attention,
    Context,
    ContextUpdates,
    # Enums
    Significance,
    Confidence,
    Status,
    ExpectationStatus,
    WatchpointStatus,
    QuestionStatus,
)
from .store import ContextStore
from .serialization import context_to_dict, context_to_json, context_summary

__all__ = [
    # Dataclasses
    "Campaign",
    "Project",
    "SessionIntent",
    "Learning",
    "Observation",
    "Expectation",
    "Watchpoint",
    "Question",
    "EmbryoUnderstanding",
    "Intentions",
    "Understanding",
    "Attention",
    "Context",
    "ContextUpdates",
    # Enums
    "Significance",
    "Confidence",
    "Status",
    "ExpectationStatus",
    "WatchpointStatus",
    "QuestionStatus",
    # Store
    "ContextStore",
    # Serialization
    "context_to_dict",
    "context_to_json",
    "context_summary",
]
