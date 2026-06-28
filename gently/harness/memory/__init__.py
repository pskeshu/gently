"""
Context Store — The agent's mind.

Separate from FileStore (raw data), FileContextStore holds understanding:
- Intentions: campaigns, projects, session focus
- Understanding: embryo states, learnings
- Observations: synthesized notes, not raw data
- Expectations: predictions about what will happen
- Attention: watchpoints, open questions

This is the scaffolding that enables the agent to remember across
its finite context window.
"""

from .model import (
    Attention,
    BenchSpec,
    Campaign,
    Confidence,
    Context,
    ContextUpdates,
    EmbryoUnderstanding,
    Expectation,
    ExpectationStatus,
    ImagingSpec,
    Intentions,
    Learning,
    Observation,
    PlanItem,
    PlanItemStatus,
    PlanItemType,
    PlannedSession,
    PlannedSessionStatus,
    Project,
    Question,
    QuestionStatus,
    SessionIntent,
    # Enums
    Significance,
    Status,
    Understanding,
    Watchpoint,
    WatchpointStatus,
)
from .store import (
    ContextStore,
)

try:
    from .file_store import FileContextStore
except ImportError:
    FileContextStore = None  # type: ignore[assignment, misc]
from .gap_assessment import ContextGapReport, Gap, GapLayer, GapSeverity, assess_gaps
from .onboarding import (
    OnboardingMessage,
    generate_onboarding_messages,
    process_onboarding_response,
)
from .serialization import context_summary, context_to_dict, context_to_json
from .startup_wizard import StartupWizard

__all__ = [
    # Dataclasses
    "Campaign",
    "Project",
    "SessionIntent",
    "PlannedSession",
    "PlanItem",
    "ImagingSpec",
    "BenchSpec",
    # Plan enums
    "PlanItemStatus",
    "PlanItemType",
    # Other dataclasses
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
    "PlannedSessionStatus",
    "ExpectationStatus",
    "WatchpointStatus",
    "QuestionStatus",
    # Store
    "ContextStore",
    "FileContextStore",
    # Serialization
    "context_to_dict",
    "context_to_json",
    "context_summary",
    # Gap Assessment
    "assess_gaps",
    "ContextGapReport",
    "Gap",
    "GapLayer",
    "GapSeverity",
    # Onboarding
    "generate_onboarding_messages",
    "process_onboarding_response",
    "OnboardingMessage",
    # Startup Wizard
    "StartupWizard",
]
