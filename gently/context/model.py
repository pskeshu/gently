"""
Context model — dataclasses representing the agent's mind.

These are the things the agent "knows" and "believes", not raw data.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional
from enum import Enum


class Significance(str, Enum):
    """How important something is."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Confidence(str, Enum):
    """How confident we are in a belief."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Status(str, Enum):
    """Generic status for things that can be active/completed."""
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"


class PlannedSessionStatus(str, Enum):
    """Status for planned imaging sessions."""
    PLANNED = "planned"
    ACTIVE = "active"        # Currently in progress
    COMPLETED = "completed"
    SKIPPED = "skipped"      # Decided not to do it
    CANCELLED = "cancelled"


class ExpectationStatus(str, Enum):
    """Status for expectations/predictions."""
    PENDING = "pending"
    CONFIRMED = "confirmed"
    SURPRISED = "surprised"
    EXPIRED = "expired"


class WatchpointStatus(str, Enum):
    """Status for watchpoints."""
    ACTIVE = "active"
    TRIGGERED = "triggered"
    RESOLVED = "resolved"


class QuestionStatus(str, Enum):
    """Status for open questions."""
    OPEN = "open"
    INVESTIGATING = "investigating"
    RESOLVED = "resolved"


# ---------------------------------------------------------------------------
# Intentions: Why are we doing this?
# ---------------------------------------------------------------------------

@dataclass
class Campaign:
    """
    A research campaign — a long-running goal spanning multiple sessions.

    Campaigns form a hierarchy: a campaign can have subcampaigns (children)
    or grow into a supercampaign (by acquiring children over time). A session
    can contribute to multiple campaigns simultaneously.

    Example: "Capture 50 hatching events from wild-type embryos"
    """
    id: str
    description: str  # Natural language, as the researcher said it
    shorthand: Optional[str] = None  # Short label: "temp-division", "hatching-50"
    summary: Optional[str] = None  # Agent-rephrased structured summary
    target: Optional[str] = None  # Measurable goal: "50 hatching events"
    progress: Optional[str] = None  # Current state: "23/50"
    parent_id: Optional[str] = None  # Parent campaign (for hierarchy)
    status: Status = Status.ACTIVE
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    @property
    def display_name(self) -> str:
        """Short display name: shorthand if available, else truncated description."""
        if self.shorthand:
            return self.shorthand
        return self.description[:50] + ("..." if len(self.description) > 50 else "")


@dataclass
class Project:
    """
    A project within a campaign — a discrete piece of work.

    Example: "Optimize imaging parameters for early stages"
    """
    id: str
    description: str
    campaign_id: Optional[str] = None
    status: Status = Status.ACTIVE
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class SessionIntent:
    """
    What this session is about.

    Tracks planned intent vs what actually happened.
    A session can belong to multiple campaigns (linked via session_campaigns).
    """
    session_id: str
    planned_intent: Optional[str] = None  # What was planned
    actual_summary: Optional[str] = None  # What happened
    campaign_ids: List[str] = field(default_factory=list)  # Linked campaigns
    created_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None


@dataclass
class PlannedSession:
    """
    A scheduled imaging session on the project calendar.

    Created days or weeks ahead. Links to campaigns. Can carry
    acquisition parameters from a previous session so you can
    say "use same settings as last Tuesday."

    When the researcher sits down and starts imaging, the wizard
    can match the planned session to the actual session and
    pre-populate intent + parameters.
    """
    id: str
    title: Optional[str] = None  # "N2 baseline imaging round 3"
    notes: Optional[str] = None  # Free-form: what to do, what to watch for
    scheduled_date: Optional[str] = None  # ISO date: "2026-02-15"
    scheduled_time: Optional[str] = None  # ISO time: "14:00" (optional)
    estimated_duration_minutes: Optional[int] = None
    acquisition_params: Optional[Dict[str, Any]] = None  # From previous session
    source_session_id: Optional[str] = None  # "use params from this session"
    status: PlannedSessionStatus = PlannedSessionStatus.PLANNED
    session_id: Optional[str] = None  # Linked actual session once started
    campaign_ids: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    @property
    def display_title(self) -> str:
        if self.title:
            return self.title
        if self.notes:
            return self.notes[:40] + ("..." if len(self.notes) > 40 else "")
        return f"Session on {self.scheduled_date or '(unscheduled)'}"


@dataclass
class Intentions:
    """Collection of the agent's intentions at multiple levels."""
    campaigns: List[Campaign] = field(default_factory=list)
    projects: List[Project] = field(default_factory=list)
    planned_sessions: List[PlannedSession] = field(default_factory=list)
    current_focus: Optional[str] = None
    session_intent: Optional[SessionIntent] = None


# ---------------------------------------------------------------------------
# Understanding: What do we believe?
# ---------------------------------------------------------------------------

@dataclass
class Learning:
    """
    Something the agent has learned from observations.

    Example: "Batch 7 develops 20% faster than average"
    """
    id: str
    content: str  # Human-readable insight
    confidence: Confidence = Confidence.MEDIUM
    basis: Optional[str] = None  # What observations support this
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class EmbryoUnderstanding:
    """
    What the agent understands about a specific embryo.

    This is synthesized understanding, not raw data.
    """
    embryo_id: str
    current_stage: Optional[str] = None
    stage_confidence: Optional[Confidence] = None
    health_assessment: Optional[str] = None
    notes: List[str] = field(default_factory=list)
    last_observed: Optional[datetime] = None

    # Tracking flags
    is_tracked: bool = True
    is_hatched: bool = False
    needs_attention: bool = False
    attention_reason: Optional[str] = None


@dataclass
class Understanding:
    """The agent's overall understanding of the experiment."""
    embryo_states: Dict[str, EmbryoUnderstanding] = field(default_factory=dict)
    learnings: List[Learning] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Observations: What have we seen? (synthesized, not raw)
# ---------------------------------------------------------------------------

@dataclass
class Observation:
    """
    A synthesized observation about the experiment.

    Not raw data — a meaningful note about what happened.
    """
    id: str
    timestamp: datetime
    type: str  # stage_transition, anomaly, session_summary, milestone
    content: str  # Human-readable description
    embryo_id: Optional[str] = None
    significance: Significance = Significance.MEDIUM
    session_id: Optional[str] = None
    gently_refs: Optional[Dict[str, Any]] = None  # References to GentlyStore data
    relates_to: Optional[List[str]] = None  # Related goals/observations


# ---------------------------------------------------------------------------
# Expectations: What do we predict?
# ---------------------------------------------------------------------------

@dataclass
class Expectation:
    """
    A prediction about what will happen.

    Example: "embryo_3 will reach comma stage by 14:30"
    """
    id: str
    target: str  # What this is about (embryo_id, etc)
    prediction: str  # "will reach comma stage"
    expected_time: datetime
    uncertainty: Optional[str] = None  # "±30 minutes"
    basis: Optional[str] = None  # Why we expect this
    status: ExpectationStatus = ExpectationStatus.PENDING
    created_at: datetime = field(default_factory=datetime.now)
    resolved_at: Optional[datetime] = None


# ---------------------------------------------------------------------------
# Attention: What should we watch?
# ---------------------------------------------------------------------------

@dataclass
class Watchpoint:
    """
    Something to watch for.

    Example: Watch embryo_3 for "approaching hatching"
    """
    id: str
    target: str  # "embryo_3"
    condition: str  # "approaching hatching"
    priority: Significance = Significance.MEDIUM
    status: WatchpointStatus = WatchpointStatus.ACTIVE
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class Question:
    """
    An open question the agent is tracking.

    Example: "Why is batch 7 slower than batch 6?"
    """
    id: str
    content: str
    status: QuestionStatus = QuestionStatus.OPEN
    resolution: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    resolved_at: Optional[datetime] = None


@dataclass
class Attention:
    """What the agent is watching/thinking about."""
    watchpoints: List[Watchpoint] = field(default_factory=list)
    open_questions: List[Question] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Full Context
# ---------------------------------------------------------------------------

@dataclass
class Context:
    """
    The full context loaded for the agent each thinking cycle.

    This is the agent's "working memory" for a single think.
    """
    intentions: Intentions = field(default_factory=Intentions)
    understanding: Understanding = field(default_factory=Understanding)
    observations: List[Observation] = field(default_factory=list)
    expectations: List[Expectation] = field(default_factory=list)
    attention: Attention = field(default_factory=Attention)

    @property
    def active_campaigns(self) -> List[Campaign]:
        """Get active campaigns."""
        return [c for c in self.intentions.campaigns if c.status == Status.ACTIVE]

    @property
    def pending_expectations(self) -> List[Expectation]:
        """Get pending expectations."""
        return [e for e in self.expectations if e.status == ExpectationStatus.PENDING]

    @property
    def active_watchpoints(self) -> List[Watchpoint]:
        """Get active watchpoints."""
        return [w for w in self.attention.watchpoints if w.status == WatchpointStatus.ACTIVE]


# ---------------------------------------------------------------------------
# Context Updates (from agent response)
# ---------------------------------------------------------------------------

@dataclass
class ContextUpdates:
    """
    Batch updates to apply to the context store.

    The agent returns these after thinking.
    """
    # New items to add
    new_observations: List[Observation] = field(default_factory=list)
    new_expectations: List[Expectation] = field(default_factory=list)
    new_watchpoints: List[Watchpoint] = field(default_factory=list)
    new_learnings: List[Learning] = field(default_factory=list)
    new_questions: List[Question] = field(default_factory=list)

    # Status updates
    resolved_expectations: Dict[str, ExpectationStatus] = field(default_factory=dict)
    triggered_watchpoints: List[str] = field(default_factory=list)
    resolved_questions: Dict[str, str] = field(default_factory=dict)  # id -> resolution

    # Understanding updates
    embryo_updates: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    # Campaign/project progress
    campaign_progress: Dict[str, str] = field(default_factory=dict)  # id -> progress

    # Focus update
    new_focus: Optional[str] = None
