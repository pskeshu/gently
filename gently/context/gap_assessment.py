"""
Gap assessment — the daemon inspects its own mind and identifies what's missing.

Run at every startup. Produces a ContextGap report that drives onboarding:
which questions to ask, what ingestion to suggest, how much conversation
is needed before the daemon can be a useful partner.
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

from .store import ContextStore

logger = logging.getLogger(__name__)


class GapLayer(str, Enum):
    """Which context layer has a gap."""
    LAB = "lab"           # Layer 1: lab identity, setup, organism
    CAMPAIGN = "campaign"  # Layer 2: research direction, goals
    SESSION = "session"    # Layer 3: today's intent
    REALTIME = "realtime"  # Layer 4: current observations (never gapped)


class GapSeverity(str, Enum):
    """How critical this gap is."""
    EMPTY = "empty"        # Nothing at all — full onboarding needed
    THIN = "thin"          # Some context exists but insufficient
    ADEQUATE = "adequate"  # Enough to function, could be richer


@dataclass
class Gap:
    """A single identified gap in the daemon's knowledge."""
    layer: GapLayer
    severity: GapSeverity
    description: str
    suggested_action: str  # What the daemon should do about it


@dataclass
class ContextGapReport:
    """
    Result of gap assessment.

    The daemon uses this to decide what onboarding is needed.
    """
    gaps: List[Gap] = field(default_factory=list)
    readiness: float = 0.0  # 0.0 (blank) to 1.0 (fully oriented)
    needs_lab_onboarding: bool = False
    needs_campaign: bool = False
    needs_session_intent: bool = False
    active_campaign_id: Optional[str] = None
    active_campaign_description: Optional[str] = None
    session_count: int = 0  # How many past sessions exist
    learning_count: int = 0

    @property
    def is_first_launch(self) -> bool:
        return self.needs_lab_onboarding

    @property
    def has_campaign(self) -> bool:
        return self.active_campaign_id is not None

    @property
    def conversation_weight(self) -> str:
        """How much conversation is needed: heavy, moderate, light, none."""
        if self.needs_lab_onboarding:
            return "heavy"
        if self.needs_campaign:
            return "moderate"
        if self.needs_session_intent:
            return "light"
        return "none"


def assess_gaps(context_store: ContextStore) -> ContextGapReport:
    """
    Inspect the context store and identify what's missing.

    Parameters
    ----------
    context_store : ContextStore
        The daemon's mind to inspect.

    Returns
    -------
    ContextGapReport
        Assessment of what's missing and what to do about it.
    """
    report = ContextGapReport()
    readiness_score = 0.0

    # --- Layer 1: Lab identity ---
    learnings = context_store.get_learnings(limit=100)
    report.learning_count = len(learnings)

    # Check for lab-level knowledge (learnings about the setup, organism, etc.)
    lab_learnings = [
        l for l in learnings
        if l.basis and any(
            kw in l.basis.lower()
            for kw in ("lab", "setup", "microscope", "organism", "onboarding", "identity")
        )
    ]

    if not learnings:
        report.needs_lab_onboarding = True
        report.gaps.append(Gap(
            layer=GapLayer.LAB,
            severity=GapSeverity.EMPTY,
            description="No learnings at all — this appears to be a first launch.",
            suggested_action="Conduct lab onboarding conversation.",
        ))
    elif not lab_learnings:
        report.gaps.append(Gap(
            layer=GapLayer.LAB,
            severity=GapSeverity.THIN,
            description="Have learnings but none about lab identity/setup.",
            suggested_action="Ask about lab setup and research program.",
        ))
        readiness_score += 0.1
    else:
        readiness_score += 0.25

    # --- Layer 2: Campaign ---
    campaigns = context_store.get_active_campaigns()

    if not campaigns:
        report.needs_campaign = True
        # Check if there were ever campaigns (completed ones)
        all_campaigns_row = context_store._conn.execute(
            "SELECT COUNT(*) as cnt FROM campaigns"
        ).fetchone()
        total_campaigns = all_campaigns_row["cnt"] if all_campaigns_row else 0

        if total_campaigns == 0:
            report.gaps.append(Gap(
                layer=GapLayer.CAMPAIGN,
                severity=GapSeverity.EMPTY,
                description="No campaigns ever created — no research direction.",
                suggested_action="Ask about research goals or suggest ingesting papers.",
            ))
        else:
            report.gaps.append(Gap(
                layer=GapLayer.CAMPAIGN,
                severity=GapSeverity.THIN,
                description="Past campaigns exist but none active.",
                suggested_action="Ask if starting new work or resuming.",
            ))
            readiness_score += 0.1
    else:
        report.active_campaign_id = campaigns[0].id
        report.active_campaign_description = campaigns[0].description
        readiness_score += 0.25

    # --- Layer 3: Session intent ---
    current_intent = context_store.get_current_session_intent()

    if not current_intent:
        report.needs_session_intent = True

        # Count past sessions for context
        session_rows = context_store._conn.execute(
            "SELECT COUNT(*) as cnt FROM session_intents"
        ).fetchone()
        report.session_count = session_rows["cnt"] if session_rows else 0

        if report.session_count == 0:
            report.gaps.append(Gap(
                layer=GapLayer.SESSION,
                severity=GapSeverity.EMPTY,
                description="No session history at all.",
                suggested_action="Establish session intent after campaign context.",
            ))
        else:
            report.gaps.append(Gap(
                layer=GapLayer.SESSION,
                severity=GapSeverity.THIN,
                description=f"{report.session_count} past sessions, but no intent for current.",
                suggested_action="Quick check-in: continuing campaign or starting fresh?",
            ))
            readiness_score += 0.1
    else:
        readiness_score += 0.25

    # --- Expectations and watchpoints (indicators of engaged state) ---
    expectations = context_store.get_pending_expectations()
    watchpoints = context_store.get_active_watchpoints()
    questions = context_store.get_open_questions()

    if expectations:
        readiness_score += 0.1
    if watchpoints:
        readiness_score += 0.1
    if questions:
        readiness_score += 0.05

    report.readiness = min(readiness_score, 1.0)

    # Log the assessment
    gap_count = len(report.gaps)
    if gap_count == 0:
        logger.info(f"Gap assessment: no gaps found (readiness={report.readiness:.2f})")
    else:
        logger.info(
            f"Gap assessment: {gap_count} gaps found "
            f"(readiness={report.readiness:.2f}, "
            f"conversation_weight={report.conversation_weight})"
        )
        for gap in report.gaps:
            logger.info(f"  [{gap.layer.value}] {gap.severity.value}: {gap.description}")

    return report
