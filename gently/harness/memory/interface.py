"""
AgentMemory — The agent's interface to its persistent memory.

Wraps ContextStore with agent-facing query and formatting methods.
Used by tools (memory_tools.py), prompt builder (prompt_manager.py),
and session briefing (agent_bridge.py).

This is the single source of truth for how the agent accesses its
accumulated knowledge across sessions.
"""

import logging
from typing import List, Optional

logger = logging.getLogger(__name__)


def _friendly_name(campaign) -> str:
    """Human-readable campaign name: description first, shorthand in parens."""
    desc = campaign.description
    if campaign.shorthand:
        return f"{desc} ({campaign.shorthand})"
    return desc


def _short_name(campaign) -> str:
    """Compact name for inline mentions: description truncated."""
    return campaign.description[:50] + ("..." if len(campaign.description) > 50 else "")


class AgentMemory:
    """The agent's interface to its persistent memory.

    Wraps ContextStore with agent-facing query and formatting methods.
    All formatting methods return markdown strings ready for tool output
    or prompt injection.
    """

    def __init__(self, context_store, session_id: str = None):
        self.store = context_store
        self.session_id = session_id

    # ------------------------------------------------------------------
    # Prompt layer — lightweight awareness for system prompt
    # ------------------------------------------------------------------

    def get_awareness_summary(self) -> str:
        """Lightweight memory summary for the system prompt (~100 tokens).

        Shows root campaigns (not phases), human-readable names, counts,
        and session-campaign link. Returns '' if context store is empty.
        """
        try:
            root_campaigns = self.store.get_root_campaigns()
            focus = self.store.get_state("current_focus")
            learnings = self.store.get_learnings(limit=100)
            observations = self.store.get_recent_observations(limit=100)
            questions = self.store.get_open_questions()

            # Nothing to report
            if not root_campaigns and not learnings and not observations:
                return ""

            lines = [
                "# Your Memory",
                "You have persistent memory from previous sessions. "
                "Use recall tools to access details.",
            ]

            # Root campaigns with progress
            if root_campaigns:
                campaign_parts = []
                for c in root_campaigns[:5]:
                    name = _short_name(c)
                    try:
                        status = self.store.get_plan_status(c.id)
                        if status["total"] > 0:
                            campaign_parts.append(
                                f'"{name}" ({status["completed"]}/{status["total"]})'
                            )
                        else:
                            campaign_parts.append(f'"{name}"')
                    except Exception:
                        campaign_parts.append(f'"{name}"')
                lines.append(f"- Active campaigns: {', '.join(campaign_parts)}")

            # Current focus
            if focus:
                lines.append(f"- Current focus: {focus}")

            # Session-campaign link
            if self.session_id:
                try:
                    session_campaigns = self.store.get_campaigns_for_session(
                        self.session_id
                    )
                    if session_campaigns:
                        names = [_short_name(c) for c in session_campaigns[:2]]
                        lines.append(
                            f'- This session: linked to {", ".join(f"{n!r}" for n in names)}'
                        )
                except Exception:
                    pass

            # Counts
            count_parts = []
            if learnings:
                count_parts.append(f"{len(learnings)} learnings")
            if observations:
                count_parts.append(f"{len(observations)} observations")
            if questions:
                count_parts.append(f"{len(questions)} open questions")
            if count_parts:
                lines.append(f"- {', '.join(count_parts)} available")

            # Available tools
            lines.append(
                "- Tools: recall_campaigns, recall_learnings, "
                "recall_observations, recall_context, query_lab_history"
            )

            return "\n".join(lines)
        except Exception as e:
            logger.debug(f"Could not build memory awareness: {e}")
            return ""

    # ------------------------------------------------------------------
    # Briefing layer — auto-briefing at session start
    # ------------------------------------------------------------------

    def get_session_briefing(self, campaign_id: str = None) -> str:
        """Generate a session briefing for new sessions.

        If campaign_id is provided (or session is linked to a campaign),
        focuses on that campaign's data. Otherwise gives a broad overview.

        Returns '' if no data to brief on.
        """
        try:
            # Determine focus campaign
            focus_campaign = None
            if campaign_id:
                focus_campaign = self.store.get_campaign(campaign_id)
            elif self.session_id:
                try:
                    linked = self.store.get_campaigns_for_session(self.session_id)
                    if linked:
                        focus_campaign = linked[0]
                except Exception:
                    pass

            if focus_campaign:
                return self._briefing_for_campaign(focus_campaign)
            else:
                return self._briefing_broad()
        except Exception as e:
            logger.debug(f"Could not generate session briefing: {e}")
            return ""

    def _briefing_for_campaign(self, campaign) -> str:
        """Campaign-focused briefing."""
        lines = [f"## Session Briefing — {_friendly_name(campaign)}"]

        if campaign.target:
            lines.append(f"**Target**: {campaign.target}")
        if campaign.progress:
            lines.append(f"**Progress**: {campaign.progress}")

        # Phases (subcampaigns)
        try:
            phases = self.store.get_subcampaigns(campaign.id)
            if phases:
                lines.append("\n**Phases**:")
                for p in phases:
                    try:
                        ps = self.store.get_plan_status(p.id)
                        items_str = f" ({ps['completed']}/{ps['total']} items)" if ps["total"] > 0 else ""
                    except Exception:
                        items_str = ""
                    lines.append(f"  - {_friendly_name(p)}{items_str}")
        except Exception:
            pass

        # Plan status (root campaign)
        try:
            status = self.store.get_plan_status(campaign.id)
            if status["total"] > 0:
                lines.append(
                    f"\n**Plan**: {status['completed']}/{status['total']} items done, "
                    f"{status['in_progress']} in progress"
                )
                if status.get("next_actions"):
                    lines.append("**Next actions**:")
                    for item in status["next_actions"][:5]:
                        lines.append(f"  - {item.title}")
                if status.get("pending_decisions"):
                    lines.append("**Pending decisions**:")
                    for item in status["pending_decisions"][:3]:
                        lines.append(f"  - {item.title}")
        except Exception:
            pass

        # Relevant learnings
        learnings = self.store.get_learnings(limit=50)
        if learnings:
            lines.append("\n**Recent learnings**:")
            for l in learnings[:5]:
                conf = l.confidence.value if l.confidence else "?"
                lines.append(f"  - [{conf}] {l.content[:150]}")

        # Other active root campaigns (brief mention)
        root_campaigns = self.store.get_root_campaigns()
        others = [c for c in root_campaigns if c.id != campaign.id]
        if others:
            names = [_short_name(c) for c in others[:3]]
            lines.append(f"\n**Other active campaigns**: {', '.join(names)}")

        # Open questions
        questions = self.store.get_open_questions()
        if questions:
            lines.append("\n**Open questions**:")
            for q in questions[:3]:
                lines.append(f"  - {q.content[:100]}")

        return "\n".join(lines) if len(lines) > 1 else ""

    def _briefing_broad(self) -> str:
        """Broad briefing when no campaign is linked — kept minimal."""
        root_campaigns = self.store.get_root_campaigns()
        learnings = self.store.get_learnings(limit=100)

        if not root_campaigns and not learnings:
            return ""

        parts = []
        if root_campaigns:
            parts.append(f"{len(root_campaigns)} campaigns")
        if learnings:
            parts.append(f"{len(learnings)} learnings")

        return f"{', '.join(parts)} loaded. Use recall tools for details."

    # ------------------------------------------------------------------
    # Recall layer — used by tools and briefing
    # ------------------------------------------------------------------

    def recall_campaigns(self, status: str = "active") -> str:
        """List campaigns as a hierarchy: root campaigns with phases nested."""
        if status == "all":
            root_campaigns = self.store.get_root_campaigns(status=None)
        else:
            root_campaigns = self.store.get_root_campaigns(status=status)

        if not root_campaigns:
            return f"No {status} campaigns found."

        lines = [f"## Campaigns ({status})"]
        for c in root_campaigns:
            status_val = c.status.value
            lines.append(f"\n### {_friendly_name(c)} [{status_val}]")
            if c.target:
                lines.append(f"  **Target**: {c.target}")
            if c.progress:
                lines.append(f"  **Progress**: {c.progress}")

            try:
                plan = self.store.get_plan_status(c.id)
                if plan["total"] > 0:
                    lines.append(
                        f"  **Plan**: {plan['completed']}/{plan['total']} done, "
                        f"{plan['in_progress']} in progress"
                    )
            except Exception:
                pass

            # Phases (subcampaigns)
            try:
                phases = self.store.get_subcampaigns(c.id)
                if phases:
                    lines.append("  **Phases**:")
                    for p in phases:
                        try:
                            ps = self.store.get_plan_status(p.id)
                            items_str = f" ({ps['completed']}/{ps['total']} items)" if ps["total"] > 0 else ""
                        except Exception:
                            items_str = ""
                        lines.append(f"    - {_friendly_name(p)}{items_str}")
            except Exception:
                pass

            lines.append(f"  id: {c.id}")

        return "\n".join(lines)

    def recall_learnings(self, query: str = None, limit: int = 20) -> str:
        """Search or list learnings."""
        learnings = self.store.get_learnings(limit=max(limit, 50))

        if query:
            query_lower = query.lower()
            terms = query_lower.split()
            learnings = [
                l
                for l in learnings
                if any(
                    term in l.content.lower() or (l.basis and term in l.basis.lower())
                    for term in terms
                )
            ]

        learnings = learnings[:limit]

        if not learnings:
            msg = f"No learnings found matching '{query}'." if query else "No learnings recorded yet."
            return msg

        header = f"## Learnings matching '{query}'" if query else "## Recent Learnings"
        lines = [header]
        for l in learnings:
            conf = l.confidence.value if l.confidence else "?"
            lines.append(f"\n- [{conf}] {l.content}")
            if l.basis:
                lines.append(f"  _Basis_: {l.basis[:200]}")
            lines.append(f"  _{l.created_at.strftime('%Y-%m-%d %H:%M')}_")

        return "\n".join(lines)

    def recall_observations(
        self, query: str = None, embryo_id: str = None, limit: int = 20
    ) -> str:
        """Search or list observations."""
        if embryo_id:
            observations = self.store.get_observations_for_embryo(embryo_id)
        else:
            observations = self.store.get_recent_observations(limit=max(limit, 50))

        if query:
            query_lower = query.lower()
            terms = query_lower.split()
            observations = [
                o
                for o in observations
                if any(term in o.content.lower() for term in terms)
            ]

        observations = observations[:limit]

        if not observations:
            msg = f"No observations found matching '{query}'." if query else "No observations recorded yet."
            return msg

        header = "## Observations"
        if query:
            header += f" matching '{query}'"
        if embryo_id:
            header += f" for {embryo_id}"
        lines = [header]
        for o in observations:
            sig = o.significance.value if o.significance else "?"
            time_str = o.timestamp.strftime("%Y-%m-%d %H:%M")
            embryo_tag = f" [{o.embryo_id}]" if o.embryo_id else ""
            lines.append(f"\n- [{time_str}] [{sig}]{embryo_tag} {o.content[:200]}")
            if o.type:
                lines.append(f"  _Type_: {o.type}")

        return "\n".join(lines)

    def recall_full_context(self, campaign_id: str = None) -> str:
        """Full context snapshot — the 'catch me up' method.

        If campaign_id provided (or session is linked), focuses there.
        Otherwise returns broad overview.
        """
        # Determine focus campaign
        focus_campaign = None
        if campaign_id:
            try:
                focus_campaign = self.store.get_campaign(campaign_id)
            except Exception:
                pass
        elif self.session_id:
            try:
                linked = self.store.get_campaigns_for_session(self.session_id)
                if linked:
                    focus_campaign = linked[0]
            except Exception:
                pass

        lines = ["## Full Context Snapshot"]

        # Focus
        focus = self.store.get_state("current_focus")
        if focus:
            lines.append(f"\n**Current focus**: {focus}")

        # Root campaigns with phases
        root_campaigns = self.store.get_root_campaigns()
        if root_campaigns:
            lines.append("\n### Active Campaigns")
            for c in root_campaigns:
                name = _friendly_name(c)
                is_focus = " ← this session" if (focus_campaign and c.id == focus_campaign.id) else ""
                progress = f" — {c.progress}" if c.progress else ""
                try:
                    status = self.store.get_plan_status(c.id)
                    if status["total"] > 0:
                        lines.append(
                            f"- **{name}**{progress} "
                            f"({status['completed']}/{status['total']} items){is_focus}"
                        )
                    else:
                        lines.append(f"- **{name}**{progress}{is_focus}")
                except Exception:
                    lines.append(f"- **{name}**{progress}{is_focus}")

                # Phases
                try:
                    phases = self.store.get_subcampaigns(c.id)
                    if phases:
                        for p in phases:
                            try:
                                ps = self.store.get_plan_status(p.id)
                                items_str = f" ({ps['completed']}/{ps['total']})" if ps["total"] > 0 else ""
                            except Exception:
                                items_str = ""
                            lines.append(f"  - {_short_name(p)}{items_str}")
                except Exception:
                    pass

        # Plan items for focus campaign
        if focus_campaign:
            try:
                status = self.store.get_plan_status(focus_campaign.id)
                if status.get("next_actions"):
                    lines.append(f"\n### Next Actions ({_short_name(focus_campaign)})")
                    for item in status["next_actions"][:5]:
                        lines.append(f"- {item.title}")
                if status.get("pending_decisions"):
                    lines.append(f"\n### Pending Decisions ({_short_name(focus_campaign)})")
                    for item in status["pending_decisions"][:3]:
                        lines.append(f"- {item.title}")
            except Exception:
                pass

        # Learnings
        learnings = self.store.get_learnings(limit=10)
        if learnings:
            lines.append("\n### Recent Learnings")
            for l in learnings[:10]:
                conf = l.confidence.value if l.confidence else "?"
                lines.append(f"- [{conf}] {l.content[:150]}")

        # Expectations
        expectations = self.store.get_pending_expectations()
        if expectations:
            lines.append("\n### Pending Expectations")
            for e in expectations[:5]:
                time_str = e.expected_time.strftime("%Y-%m-%d %H:%M") if e.expected_time else "?"
                lines.append(f"- {e.target}: {e.prediction} (by {time_str})")

        # Watchpoints
        watchpoints = self.store.get_active_watchpoints()
        if watchpoints:
            lines.append("\n### Active Watchpoints")
            for w in watchpoints[:5]:
                lines.append(f"- {w.target}: {w.condition}")

        # Questions
        questions = self.store.get_open_questions()
        if questions:
            lines.append("\n### Open Questions")
            for q in questions[:5]:
                lines.append(f"- {q.content[:100]}")

        return "\n".join(lines) if len(lines) > 1 else "No context data recorded yet."

    # ------------------------------------------------------------------
    # Session-campaign awareness
    # ------------------------------------------------------------------

    def get_session_campaigns(self) -> list:
        """Get campaigns linked to the current session."""
        if not self.session_id:
            return []
        try:
            return self.store.get_campaigns_for_session(self.session_id)
        except Exception:
            return []
