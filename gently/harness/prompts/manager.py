"""
PromptManager - System prompt construction and context summarization.

Extracted from agent.py to separate prompt-building logic from
conversation mechanics and session persistence.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Any

from gently.settings import settings

from ..plan_mode.prompt import build_plan_prompt
from ..resolution_mode.prompt import build_resolution_prompt
from ..tools.registry import get_tool_registry
from .templates import build_system_prompt

logger = logging.getLogger(__name__)


class PromptManager:
    """
    Builds system prompts, selects tools per mode, and generates
    context summaries for session awareness.

    Does not hold back-references to agent — receives data as parameters.
    """

    def __init__(self, claude_client, model):
        self.claude = claude_client
        self.model = model

        # Context summary caching
        self._context_summary_cache: str | None = None
        self._context_summary_time: datetime | None = None
        self._context_summary_ttl: int = 300  # 5 minutes

        # Memory awareness caching
        self._memory_awareness_cache: str | None = None
        self._memory_awareness_time: datetime | None = None
        self._memory_awareness_ttl: int = 600  # 10 minutes

        # Set by agent after construction
        self.context_store = None
        self.memory = None  # AgentMemory instance

    def update_system_prompt(
        self, experiment, client, mode: str, context_summary: str | None = None, perceiver=None
    ) -> str:
        """
        Rebuild system prompt with current experiment state and connection status.

        Parameters
        ----------
        experiment : ExperimentState
            Current experiment state
        client : MicroscopeClient or None
            Microscope client for connection status
        mode : str
            Current mode ("run" or "plan")
        context_summary : str, optional
            AI-generated context summary

        Returns
        -------
        str
            The built system prompt
        """
        memory_awareness = self.get_cached_memory_awareness()

        if mode == "resolution":
            return build_resolution_prompt(
                context_summary=context_summary,
                memory_awareness=memory_awareness,
            )

        if mode == "plan":
            active_plan = self.get_active_plan_summary()
            return build_plan_prompt(
                context_summary=context_summary,
                active_plan_summary=active_plan,
                memory_awareness=memory_awareness,
            )

        # Execution mode
        if client:
            connection_status = {
                "device_layer": client.is_connected,
                "sam_detection": client.has_sam,
            }
        else:
            connection_status = None

        return build_system_prompt(
            experiment,
            connection_status,
            context_summary,
            memory_awareness=memory_awareness,
            microscope=client,
            perceiver=perceiver,
        )

    def get_tools_for_mode(self, mode: str, has_microscope: bool) -> list:
        """
        Get the Claude tool schemas for the given mode.

        Parameters
        ----------
        mode : str
            Current mode ("run" or "plan")
        has_microscope : bool
            Whether microscope is connected

        Returns
        -------
        list
            Tool schemas for Claude API
        """
        registry = get_tool_registry()
        if mode == "resolution":
            # Resolution mode is paperwork — no microscope, no acquisition,
            # no plan editing. Just memory recall, lifecycle decisions,
            # spec application, and the escape-hatch listing.
            resolution_tool_names = {
                # Lifecycle (transitions out of resolution mode)
                "attach_session_to_plan",
                "mark_session_standalone",
                "detach_session_from_plan",
                "mark_plan_item_status",
                # Plan application
                "apply_plan_acquisition_spec",
                # Context recall
                "recall_campaigns",
                "recall_learnings",
                "recall_observations",
                "recall_context",
                "recall_sibling_sessions",
                "summarize_campaign_history",
                # Escape hatches
                "list_imaging_candidates",
                "ask_user_choice",
            }
            all_tools = registry.get_claude_schemas(has_microscope=False)
            return [t for t in all_tools if t["name"] in resolution_tool_names]
        if mode == "plan":
            plan_tool_names = {
                "create_campaign",
                "create_plan_item",
                "update_plan_item",
                "link_plan_items",
                "propose_plan",
                "get_plan_status",
                "get_plan_item",
                "move_plan_item",
                "delete_plan_item",
                "reorder_plan_items",
                "update_phase",
                "delete_phase",
                "export_plan",
                "query_lab_history",
                "check_hardware_capability",
                "search_literature",
                "search_strains",
                "validate_plan",
                "batch_update_status",
                "batch_update_spec",
                "snapshot_plan",
                "list_plan_versions",
                "restore_plan_version",
                "save_plan_template",
                "list_templates",
                "apply_template",
                "ask_user_choice",
            }
            all_tools = registry.get_claude_schemas(has_microscope=False)
            return [t for t in all_tools if t["name"] in plan_tool_names]
        else:
            return registry.get_claude_schemas(has_microscope=has_microscope)

    def get_cached_memory_awareness(self) -> str:
        """Get memory awareness summary with caching (10-minute TTL)."""
        if not self.memory:
            return ""
        now = datetime.now()
        if (
            self._memory_awareness_cache is None
            or self._memory_awareness_time is None
            or (now - self._memory_awareness_time).total_seconds() > self._memory_awareness_ttl
        ):
            self._memory_awareness_cache = self.memory.get_awareness_summary()
            self._memory_awareness_time = now
        return self._memory_awareness_cache

    def get_active_plan_summary(self) -> str | None:
        """Get a summary of the active experimental plan, if any."""
        if not self.context_store:
            return None
        try:
            campaigns = self.context_store.get_root_campaigns()
            if not campaigns:
                return None
            lines = []
            for campaign in campaigns:
                status = self.context_store.get_plan_status(campaign.id)
                if status["total"] == 0:
                    continue
                lines.append(
                    f"Campaign: {campaign.description}"
                    f" ({status['completed']}/{status['total']} items done)"
                )
                if status["next_actions"]:
                    lines.append(
                        "  Next: " + ", ".join(a.title for a in status["next_actions"][:3])
                    )
                if status["pending_decisions"]:
                    lines.append(
                        "  Decisions pending: "
                        + ", ".join(d.title for d in status["pending_decisions"])
                    )
            return "\n".join(lines) if lines else None
        except Exception:
            return None

    def gather_context_data(self, experiment, timelapse_orch, timeline_mgr) -> dict:
        """
        Gather raw context data for summarization.

        Parameters
        ----------
        experiment : ExperimentState
            Current experiment state
        timelapse_orch : TimelapseOrchestrator or None
            Timelapse orchestrator for status
        timeline_mgr : TimelineManager or None
            Timeline manager for recent events

        Returns
        -------
        dict
            Context data including timelapse status, events, and detections
        """
        data: dict[str, Any] = {
            "current_time": datetime.now().isoformat(),
            "timelapse_status": None,
            "recent_events": [],
            "recent_detections": [],
            "detection_reasoning": [],
        }

        if timelapse_orch:
            try:
                status = timelapse_orch.get_status()
                data["timelapse_status"] = {
                    "state": status.status.value if status.status else "unknown",
                    "total_timepoints": status.total_timepoints or 0,
                    "started_at": status.started_at.isoformat() if status.started_at else None,
                    "embryo_count": len(status.embryos) if status.embryos else 0,
                }
            except Exception as e:
                logger.debug(f"Could not get timelapse status: {e}")

        if timeline_mgr:
            try:
                events = timeline_mgr.get_events(limit=20, session_id="current")
                data["recent_events"] = [
                    {
                        "type": e.event_subtype,
                        "time": e.timestamp.isoformat(),
                        "embryo": e.embryo_id,
                        "detector": e.detector_name,
                        "timepoint": e.timepoint,
                        "confidence": e.confidence,
                    }
                    for e in events
                ]
            except Exception as e:
                logger.debug(f"Could not get timeline events: {e}")

        try:
            for embryo_id, embryo_state in experiment.embryos.items():
                if not hasattr(embryo_state, "detection_results"):
                    continue
                for detector_name, results in embryo_state.detection_results.items():
                    recent_results = results[-3:] if len(results) > 3 else results
                    for r in recent_results:
                        if r.get("detected"):
                            data["recent_detections"].append(
                                {
                                    "detector": detector_name,
                                    "embryo": embryo_id,
                                    "timepoint": r.get("timepoint"),
                                    "confidence": r.get("confidence"),
                                }
                            )
                            if r.get("reasoning"):
                                data["detection_reasoning"].append(
                                    {
                                        "detector": detector_name,
                                        "embryo": embryo_id,
                                        "timepoint": r.get("timepoint"),
                                        "reasoning": r.get("reasoning")[:500],
                                    }
                                )
        except Exception as e:
            logger.debug(f"Could not get detection results: {e}")

        return data

    async def generate_context_summary(self, experiment, timelapse_orch, timeline_mgr) -> str:
        """
        Generate concise context summary using Haiku.

        Parameters
        ----------
        experiment : ExperimentState
        timelapse_orch : TimelapseOrchestrator or None
        timeline_mgr : TimelineManager or None

        Returns
        -------
        str
            Brief context summary (2-3 sentences)
        """
        raw_data = self.gather_context_data(experiment, timelapse_orch, timeline_mgr)

        has_timelapse = raw_data["timelapse_status"] is not None
        has_events = len(raw_data["recent_events"]) > 0
        has_detections = len(raw_data["recent_detections"]) > 0

        if not (has_timelapse or has_events or has_detections):
            return ""

        prompt = f"""Summarize the current microscopy session state in 2-3 sentences for
another AI assistant. Focus on: timelapse status (is it running, completed, or idle?),
time since last activity, and notable detections. Be factual and concise.

Raw session data:
{json.dumps(raw_data, indent=2, default=str)}

Write a brief status summary. Examples:
- "Timelapse completed 10h ago with 233 timepoints. Hatching was detected at timepoints
  175-193 with HIGH confidence."
- "Timelapse is currently running for embryo_1 at timepoint 45. No detections yet."
- "No active timelapse. Last session had 50 timepoints, with comma stage detected at t=30."
"""

        try:
            response = await asyncio.to_thread(
                self.claude.messages.create,
                model=settings.models.fast,
                max_tokens=150,
                messages=[{"role": "user", "content": prompt}],
            )
            return response.content[0].text.strip()
        except Exception as e:
            logger.warning(f"Failed to generate context summary: {e}")
            return ""

    async def get_cached_context_summary(self, experiment, timelapse_orch, timeline_mgr) -> str:
        """
        Get context summary with caching (5-minute TTL).

        Parameters
        ----------
        experiment : ExperimentState
        timelapse_orch : TimelapseOrchestrator or None
        timeline_mgr : TimelineManager or None

        Returns
        -------
        str
            Cached or newly generated context summary
        """
        now = datetime.now()
        if (
            self._context_summary_cache is None
            or self._context_summary_time is None
            or (now - self._context_summary_time).total_seconds() > self._context_summary_ttl
        ):
            self._context_summary_cache = await self.generate_context_summary(
                experiment, timelapse_orch, timeline_mgr
            )
            self._context_summary_time = now
        return self._context_summary_cache

    def invalidate_context_cache(self):
        """Invalidate the context summary and memory awareness caches."""
        self._context_summary_cache = None
        self._context_summary_time = None
        self._memory_awareness_cache = None
        self._memory_awareness_time = None

    def get_cached_system_prompt(self, system_prompt: str) -> list:
        """Get system prompt formatted for Anthropic prompt caching.

        Parameters
        ----------
        system_prompt : str
            The system prompt text

        Returns
        -------
        list
            Prompt formatted with cache_control for prompt caching
        """
        return [
            {
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral", "ttl": "1h"},
            }
        ]
