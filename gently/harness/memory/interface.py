"""
AgentMemory — The agent's interface to its persistent memory.

Wraps ContextStore with agent-facing query and formatting methods.
Used by tools (memory_tools.py), prompt builder (prompt_manager.py),
and session briefing (agent_bridge.py).

This is the single source of truth for how the agent accesses its
accumulated knowledge across sessions.
"""

import logging

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

    def __init__(self, context_store, session_id: str | None = None):
        self.store = context_store
        self.session_id = session_id
        # Set by startup flow after resolve_plan_context()
        self.active_plan_item_id: str | None = None

    # ------------------------------------------------------------------
    # Plan context resolution — called at startup
    # ------------------------------------------------------------------

    def resolve_plan_context(self) -> tuple[str | None, list]:
        """Determine which plan item to activate for this session.

        Scans all active campaigns for unblocked imaging items.

        Returns
        -------
        (active_item_id, candidates)
            If exactly one unblocked imaging item exists, active_item_id is
            set and candidates is a single-element list.
            If multiple exist, active_item_id is None and candidates has them all.
            If none, both are empty/None.
        """
        try:
            from .model import PlanItemType

            root_campaigns = self.store.get_root_campaigns(status="active")
            imaging_candidates = []
            for campaign in root_campaigns:
                unblocked = self.store.get_unblocked_plan_items(campaign.id)
                for item in unblocked:
                    if item.type == PlanItemType.IMAGING:
                        spec = self.store.resolve_imaging_spec(item)
                        imaging_candidates.append((item, spec, campaign))

            if len(imaging_candidates) == 1:
                item, spec, campaign = imaging_candidates[0]
                return item.id, imaging_candidates
            elif len(imaging_candidates) > 1:
                return None, imaging_candidates
            else:
                return None, []
        except Exception as e:
            logger.debug(f"Plan context resolution failed: {e}")
            return None, []

    def format_imaging_spec_summary(self, spec) -> str:
        """One-line summary of an ImagingSpec for briefings."""
        parts = []
        if spec.strain:
            parts.append(spec.strain)
        if spec.temperature_c:
            parts.append(f"{spec.temperature_c}\u00b0C")
        if spec.num_slices:
            parts.append(f"{spec.num_slices} slices")
        if spec.exposure_ms:
            parts.append(f"{spec.exposure_ms}ms")
        if spec.stop_condition:
            parts.append(f"until {spec.stop_condition}")
        if spec.interval_s:
            parts.append(f"every {spec.interval_s}s")
        return ", ".join(parts) if parts else "no spec"

    def format_imaging_spec_block(self, spec) -> str:
        """Multi-line ImagingSpec for system prompt injection."""
        lines = []
        if spec.strain:
            lines.append(f"  Strain: {spec.strain}")
        if spec.genotype:
            lines.append(f"  Genotype: {spec.genotype}")
        if spec.reporter:
            lines.append(f"  Reporter: {spec.reporter}")
        if spec.temperature_c:
            lines.append(f"  Temperature: {spec.temperature_c}\u00b0C")
        if spec.num_embryos:
            lines.append(f"  Target embryos: {spec.num_embryos}")
        if spec.num_slices:
            lines.append(f"  Slices: {spec.num_slices}")
        if spec.exposure_ms:
            lines.append(f"  Exposure: {spec.exposure_ms}ms")
        if spec.laser_wavelength_nm:
            laser = f"{spec.laser_wavelength_nm}nm"
            if spec.laser_power_pct:
                laser += f" at {spec.laser_power_pct}%"
            lines.append(f"  Laser: {laser}")
        if spec.interval_s:
            lines.append(f"  Interval: {spec.interval_s}s")
        if spec.adaptive_intervals:
            for stage, interval in spec.adaptive_intervals.items():
                lines.append(f"  Interval ({stage}): {interval}s")
        if spec.stop_condition:
            lines.append(f"  Stop condition: {spec.stop_condition}")
        if spec.estimated_duration_h:
            lines.append(f"  Estimated duration: {spec.estimated_duration_h}h")
        if spec.detectors:
            lines.append(f"  Detectors: {', '.join(spec.detectors)}")
        if spec.success_criteria:
            lines.append(f"  Success criteria: {spec.success_criteria}")
        return "\n".join(lines)

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
                    session_campaigns = self.store.get_campaigns_for_session(self.session_id)
                    if session_campaigns:
                        names = [_short_name(c) for c in session_campaigns[:2]]
                        lines.append(
                            f"- This session: linked to {', '.join(f'{n!r}' for n in names)}"
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

            # Active plan item — full spec in prompt so agent knows
            # what it's executing without needing to call recall tools
            if self.active_plan_item_id:
                try:
                    item = self.store.get_plan_item(self.active_plan_item_id)
                    if item:
                        spec = self.store.resolve_imaging_spec(item)
                        campaign = self.store.get_campaign(item.campaign_id)
                        campaign_name = _short_name(campaign) if campaign else "?"
                        # Walk to the root campaign for the overall goal (the item's
                        # campaign may be a phase under it).
                        root = campaign
                        seen_ids: set[str] = set()
                        while root and root.parent_id and root.parent_id not in seen_ids:
                            seen_ids.add(root.id)
                            root = self.store.get_campaign(root.parent_id)
                        lines.append(f"\n## Active Plan Item: {item.title}")
                        if root and root.target:
                            lines.append(f"Goal of the investigation: {root.target}")
                        if campaign and root and campaign.id != root.id:
                            lines.append(f"Phase: {campaign_name}")
                        lines.append(f"Campaign: {_short_name(root) if root else campaign_name}")
                        lines.append(f"Status: {item.status.value}")
                        if spec:
                            lines.append(self.format_imaging_spec_block(spec))
                        # What's next — the items/gates this run unblocks.
                        try:
                            root_id = root.id if root else item.campaign_id
                            nxt = [
                                u
                                for u in self.store.get_unblocked_plan_items(root_id)
                                if u.id != item.id
                            ][:3]
                            if nxt:
                                bits = []
                                for u in nxt:
                                    is_dp = u.type.value == "decision_point"
                                    bits.append(u.title + (" (decision point)" if is_dp else ""))
                                lines.append("Next up: " + "; ".join(bits))
                        except Exception:
                            pass
                        lines.append(
                            "\nYou're executing this item within the plan above — use the "
                            "spec to configure and run, and keep the goal and what's next in "
                            "mind (you can make go/no-go calls). The user expects these settings."
                        )
                except Exception:
                    pass

            # Available tools
            lines.append(
                "\n- Tools: recall_campaigns, recall_learnings, "
                "recall_observations, recall_context, query_lab_history"
            )

            return "\n".join(lines)
        except Exception as e:
            logger.debug(f"Could not build memory awareness: {e}")
            return ""

    # ------------------------------------------------------------------
    # Briefing layer — auto-briefing at session start
    # ------------------------------------------------------------------

    def get_session_briefing(self, campaign_id: str | None = None) -> str:
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
                        items_str = (
                            f" ({ps['completed']}/{ps['total']} items)" if ps["total"] > 0 else ""
                        )
                    except Exception:
                        items_str = ""
                    lines.append(f"  - {_friendly_name(p)}{items_str}")
        except Exception:
            pass

        # Plan status (root campaign)
        try:
            from .model import PlanItemType

            status = self.store.get_plan_status(campaign.id)
            if status["total"] > 0:
                lines.append(
                    f"\n**Plan**: {status['completed']}/{status['total']} items done, "
                    f"{status['in_progress']} in progress"
                )
                if status.get("next_actions"):
                    lines.append("**Next actions**:")
                    for item in status["next_actions"][:5]:
                        if item.type == PlanItemType.IMAGING:
                            spec = self.store.resolve_imaging_spec(item)
                            spec_summary = self.format_imaging_spec_summary(spec) if spec else ""
                            suffix = f" — {spec_summary}" if spec_summary else ""
                            lines.append(f"  - [imaging] {item.title}{suffix}")
                        else:
                            lines.append(f"  - [{item.type.value}] {item.title}")
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
            for learning in learnings[:5]:
                conf = learning.confidence.value if learning.confidence else "?"
                lines.append(f"  - [{conf}] {learning.content[:150]}")

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
        """Broad briefing when no campaign is linked.

        If there are actionable imaging plan items, surfaces them so the
        agent can set plan context early in the session.
        """
        root_campaigns = self.store.get_root_campaigns()
        learnings = self.store.get_learnings(limit=100)

        if not root_campaigns and not learnings:
            return ""

        lines = []

        # Surface actionable imaging items from the plan
        active_id, candidates = self.resolve_plan_context()
        if candidates:
            if len(candidates) == 1:
                item, spec, campaign = candidates[0]
                spec_summary = self.format_imaging_spec_summary(spec) if spec else "no spec"
                lines.append(f"## Next up: {item.title}")
                lines.append(f"Campaign: {_friendly_name(campaign)}")
                if spec:
                    lines.append(self.format_imaging_spec_block(spec))
                lines.append("")
                lines.append(
                    "This is the only unblocked imaging task. "
                    "Plan context has been set automatically."
                )
            else:
                lines.append("## Ready to image")
                lines.append(f"{len(candidates)} imaging tasks are unblocked:")
                lines.append("")
                for item, spec, campaign in candidates:
                    spec_summary = self.format_imaging_spec_summary(spec) if spec else "no spec"
                    lines.append(f"- **{item.title}** — {spec_summary}")
                    lines.append(f"  Campaign: {_short_name(campaign)}")
                lines.append("")
                lines.append("Which imaging task are you working on today?")
        else:
            # No plan items — minimal briefing
            parts = []
            if root_campaigns:
                parts.append(f"{len(root_campaigns)} campaigns")
            if learnings:
                parts.append(f"{len(learnings)} learnings")
            lines.append(f"{', '.join(parts)} loaded. Use recall tools for details.")

        return "\n".join(lines)

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
                            items_str = (
                                f" ({ps['completed']}/{ps['total']} items)"
                                if ps["total"] > 0
                                else ""
                            )
                        except Exception:
                            items_str = ""
                        lines.append(f"    - {_friendly_name(p)}{items_str}")
            except Exception:
                pass

            lines.append(f"  id: {c.id}")

        return "\n".join(lines)

    def recall_learnings(self, query: str | None = None, limit: int = 20) -> str:
        """Search or list learnings."""
        learnings = self.store.get_learnings(limit=max(limit, 50))

        if query:
            query_lower = query.lower()
            terms = query_lower.split()
            learnings = [
                learning
                for learning in learnings
                if any(
                    term in learning.content.lower()
                    or (learning.basis and term in learning.basis.lower())
                    for term in terms
                )
            ]

        learnings = learnings[:limit]

        if not learnings:
            msg = (
                f"No learnings found matching '{query}'." if query else "No learnings recorded yet."
            )
            return msg

        header = f"## Learnings matching '{query}'" if query else "## Recent Learnings"
        lines = [header]
        for learning in learnings:
            conf = learning.confidence.value if learning.confidence else "?"
            lines.append(f"\n- [{conf}] {learning.content}")
            if learning.basis:
                lines.append(f"  _Basis_: {learning.basis[:200]}")
            lines.append(f"  _{learning.created_at.strftime('%Y-%m-%d %H:%M')}_")

        return "\n".join(lines)

    def recall_observations(
        self, query: str | None = None, embryo_id: str | None = None, limit: int = 20
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
                o for o in observations if any(term in o.content.lower() for term in terms)
            ]

        observations = observations[:limit]

        if not observations:
            msg = (
                f"No observations found matching '{query}'."
                if query
                else "No observations recorded yet."
            )
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

    def recall_full_context(self, campaign_id: str | None = None) -> str:
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
                is_focus = (
                    " ← this session" if (focus_campaign and c.id == focus_campaign.id) else ""
                )
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
                                items_str = (
                                    f" ({ps['completed']}/{ps['total']})" if ps["total"] > 0 else ""
                                )
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
            for learning in learnings[:10]:
                conf = learning.confidence.value if learning.confidence else "?"
                lines.append(f"- [{conf}] {learning.content[:150]}")

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
