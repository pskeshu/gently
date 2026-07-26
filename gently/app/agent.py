"""
Main Microscopy Agent implementation

Thin coordinator that delegates to:
- ConversationManager: LLM calls, tool execution, token tracking
- SessionManager: session persistence and lifecycle
- PromptManager: prompt construction and context summarization

Integrated with:
- Event Bus for async message passing between components
- FileStore for unified file-based data persistence
"""

import asyncio
import logging
import os
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import anthropic
import numpy as np

from ..exceptions import StorageError
from ..settings import settings

if TYPE_CHECKING:
    from ..ui.web.server import VisualizationServer
    from .bottom_camera_monitor import BottomCameraStreamMonitor
    from .device_state_monitor import DeviceStateMonitor
    from .lightsheet_monitor import LightSheetStreamMonitor
    from .operation_plan_updater import OperationPlanUpdater
    from .temperature_sampler import TemperatureSampler

from gently_perception import Perceiver

from ..core import EventType, emit, get_event_bus
from ..core.file_store import FileStore
from ..harness.conversation import ConversationManager
from ..harness.orchestration.plan_synthesis import PlanLibrary, PlanSynthesizer, PlanValidator
from ..harness.prompts.manager import PromptManager
from ..harness.session.interaction_logger import InteractionLogger
from ..harness.session.manager import SessionManager
from ..harness.session.timeline import TimelineManager
from ..harness.state import ExperimentState
from ..harness.tools.registry import get_tool_registry

# Import tools package to trigger @tool decorator registration
from . import tools as _tools  # noqa: F401
from .orchestration.timelapse import TimelapseOrchestrator

logger = logging.getLogger(__name__)

# Shown when the agent is launched in UI-only mode (--no-api). The web UI is
# fully browsable, but anything that would call Claude is disabled.
_NO_API_NOTICE = (
    "The agent is running in **UI-only mode** (`--no-api`), so it can't "
    "respond — no Anthropic API calls are made. You can browse the interface, "
    "view saved sessions, and explore the UI. To enable chat, perception, and "
    "plan generation, restart without `--no-api` and with `ANTHROPIC_API_KEY` set."
)


class MicroscopyAgent:
    """
    Conversational AI agent for microscopy experiments

    This is the main class that orchestrates:
    - Conversation with user via Claude API
    - Experiment state management
    - Plan generation and validation
    - Image analysis with Claude Vision
    - Dynamic parameter adaptation
    """

    def __init__(
        self,
        api_key: str | None = None,
        storage_path: Path = Path("./experiment_data"),
        model: str = settings.models.main,
        microscope_client=None,
        session_id: str | None = None,
        store: FileStore | None = None,
        no_api: bool = False,
    ):
        """
        Parameters
        ----------
        api_key : str, optional
            Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
        storage_path : Path
            Where to store experiment data and images
        model : str
            Claude model to use
        microscope_client : MicroscopeClient, optional
            RPC client for microscope server. Required for hardware control.
        session_id : str, optional
            Session ID to resume. If None, creates new session.
        store : FileStore
            Unified file-based data store. Required.
        no_api : bool
            UI-only mode: skip Anthropic API calls entirely. The full agent and
            its sub-components are still constructed (so the web UI boots), but
            message handling short-circuits with a clear notice instead of
            calling Claude. Useful for browsing the UI without an API key.
        """
        if store is None:
            raise ValueError("FileStore is required. Pass store=FileStore(path) to agent.")

        # UI-only mode: no real API calls. We still build the client object (so
        # all sub-components that hold a reference work), but fall back to a
        # placeholder key so construction never fails when no key is set, and
        # the message entry points refuse to call Claude.
        self.api_enabled = not no_api

        # Shared API client. No interleaved-thinking beta header: it's GA on the
        # 4.6+ models and obsolete on Fable 5 (always-on thinking); the header is
        # dropped so it can't conflict with the new model family.
        self.claude = anthropic.Anthropic(
            api_key=api_key
            or os.getenv("ANTHROPIC_API_KEY")
            or ("no-api-mode" if no_api else None),
        )
        self.model = model

        # Mode: "run" (default), "plan" (experimental design), or
        # "resolution" (figuring out what this session is for at startup).
        # Resolution mode is entered explicitly by the startup wiring when
        # there are unblocked plan items that need a decision; it is not
        # the default for resumed sessions.
        self.mode: str = "run"

        # Context store (agent's mind — set via set_context_store)
        self.context_store: Any | None = None

        # Experiment state
        self.experiment = ExperimentState()

        # Storage path (for legacy compatibility, use store.root going forward)
        self.storage_path = Path(storage_path)

        # Unified store (FileStore) — single source of truth
        self.store = store

        # System prompt (rebuilt by PromptManager, stored here for save_session)
        self.system_prompt: str = ""

        # Plan synthesis
        self.plan_synthesizer = PlanSynthesizer(
            plan_library=PlanLibrary(), validator=PlanValidator()
        )

        # Event bus for async messaging (must be before perception manager)
        self._event_bus = get_event_bus()

        # Broadcast the embryo list whenever it mutates. Hooked through the
        # state object's observer so add/remove/nickname/restore all publish
        # without each call site having to remember.
        self.experiment.on_embryos_changed = self._publish_embryos_update

        # Perception system (gently-perception harness)
        self.perceiver = Perceiver()

        # Microscope interface (hardware abstraction)
        self.microscope = microscope_client

        # Backward-compat alias
        self.client = self.microscope

        # Callbacks
        self.on_message_callback: Callable | None = None
        self.choice_handler: Callable | None = None

        # Serializes conversation turns: user turns and autonomous wake turns
        # must not interleave on the shared conversation_history.
        self._turn_lock = asyncio.Lock()

        # Autonomy backstop: while a wake turn runs, _autonomous_active is True
        # and the registry refuses these irreversible tools (they require a
        # human). User turns are unaffected. _wake_choice_factory is set by the
        # web bridge so ASK-mode wake turns can round-trip an approval picker.
        self._autonomous_active = False
        self._autonomous_blocked_tools = frozenset(
            {
                "set_laser_power",
                "remove_embryo",
                "stop_timelapse",
            }
        )
        self._wake_choice_factory = None
        self._wake_choice_discard = None

        # Interaction logger for structured logging (research data collection)
        self.interaction_logger: InteractionLogger | None = None

        # Event capture — durable log of every EventBus event during this
        # session. Substrate for offline replay / shadow-mode A/B of
        # candidate orchestrator architectures.
        self.event_capture = None

        # Decision log — what production decided at each turn (tool calls,
        # response text, prompt hash). Pairs with event capture so a
        # candidate replay can be diffed against production turn-by-turn.
        self.decision_log = None

        # Timelapse orchestrator (initialized when microscope connected)
        self.timelapse_orchestrator: TimelapseOrchestrator | None = None

        # Timeline manager for tracking events
        self.timeline_manager: TimelineManager | None = None

        # Visualization server for real-time feedback
        self.viz_server: VisualizationServer | None = None

        # Device-state monitor (bridges device-layer SSE → EventBus)
        self.device_state_monitor: DeviceStateMonitor | None = None
        # Session-scoped temperature sampler — polls device layer, persists readings.
        self.temperature_sampler: TemperatureSampler | None = None
        # Bus-subscriber that transitions plan tactics when execution events fire.
        self.operation_plan_updater: OperationPlanUpdater | None = None
        # Opt-in bottom-camera stream bridge — created when viz starts, but
        # left unstarted until the operator clicks "Start camera" in the UI.
        self.bottom_camera_monitor: BottomCameraStreamMonitor | None = None
        # Opt-in lightsheet stream bridge — same lifecycle as bottom_camera_monitor.
        self.lightsheet_monitor: LightSheetStreamMonitor | None = None

        # ===== Create delegate managers =====

        # Conversation manager (LLM calls, tool execution, token tracking)
        self.conversation = ConversationManager(
            client=self.claude,
            model=self.model,
            tool_registry=get_tool_registry(),
        )
        # Wire tool execution context
        self.conversation._tool_context = {
            "agent": self,
            "client": getattr(self, "microscope", None),
            "microscope": getattr(self, "microscope", None),
            "databroker": getattr(self, "databroker", None),
        }

        # Session manager (persistence)
        self.sessions = SessionManager(
            store=self.store,
            storage_path=self.storage_path,
        )

        # Prompt manager (prompt construction, context summarization)
        self.prompts = PromptManager(
            claude_client=self.claude,
            model=self.model,
        )

        # Initialize or resume session
        if session_id:
            success, history = self.sessions._resume_session(session_id, self.experiment)
            if success:
                self.conversation.conversation_history = history
            self._emit_event(
                EventType.SESSION_RESTORED,
                {
                    "session_id": session_id,
                    "embryo_count": len(self.experiment.embryos),
                    "message_count": len(self.conversation.conversation_history),
                },
            )
        else:
            self.sessions.create_session()
            self._emit_event(
                EventType.SESSION_STARTED,
                {
                    "session_id": self.sessions.session_id,
                },
            )

        # Initialize interaction logger (for research data collection)
        self._init_interaction_logger()

        # Start event capture into the session folder so offline replay /
        # shadow-mode testing has a durable input stream. Filters out the
        # high-volume telemetry types (DEVICE_STATE_UPDATE / BOTTOM_CAMERA_FRAME)
        # by default so a long timelapse doesn't bury the meaningful events.
        self._init_event_capture()

        # Open the per-session production decision log and hand it to the
        # conversation manager so each Claude round-trip is captured.
        self._init_decision_log()

        # Wire interaction logger and choice handler to conversation manager
        self.conversation.interaction_logger = self.interaction_logger
        self.conversation.choice_handler = self.choice_handler

        # Initialize timelapse orchestrator (if microscope connected)
        self._init_timelapse_orchestrator()

        # Initialize timeline manager (subscribes to event bus)
        self._init_timeline_manager()

        # Subscribe to CV result events for EmbryoState integration
        self._subscribe_to_cv_events()

        # Decision-moment wake-router (opt-in, default OFF). Wakes the agent on
        # wake-worthy perception/lifecycle events so it can adapt acquisition
        # autonomously; enabled via the set_autonomy tool.
        try:
            from gently.app.wake_router import WakeRouter

            self.wake_router: WakeRouter | None = WakeRouter(self, self._event_bus)
        except Exception:
            logger.exception("Failed to init wake-router")
            self.wake_router = None

        # Build initial system prompt
        self._update_system_prompt()

    # ===== Backward-Compat Delegation Properties =====

    @property
    def session_id(self) -> str | None:
        """Get current session ID (None before a session is created)."""
        return self.sessions.session_id

    @property
    def _session_id(self) -> str | None:
        """Internal session ID (backward compat)."""
        return self.sessions._session_id

    @_session_id.setter
    def _session_id(self, value):
        self.sessions._session_id = value

    @property
    def conversation_history(self) -> list[dict]:
        """Get conversation history."""
        return self.conversation.conversation_history

    @conversation_history.setter
    def conversation_history(self, value):
        self.conversation.conversation_history = value

    @property
    def total_input_tokens(self) -> int:
        return self.conversation.total_input_tokens

    @property
    def total_output_tokens(self) -> int:
        return self.conversation.total_output_tokens

    @property
    def api_call_count(self) -> int:
        return self.conversation.api_call_count

    @property
    def cache_creation_tokens(self) -> int:
        return self.conversation.cache_creation_tokens

    @property
    def cache_read_tokens(self) -> int:
        return self.conversation.cache_read_tokens

    @property
    def token_usage_summary(self) -> str:
        return self.conversation.token_usage_summary

    @property
    def current_context_tokens(self) -> int:
        return self.conversation.current_context_tokens

    # ===== Mode Management =====

    def set_context_store(self, context_store) -> None:
        """Attach the context store (agent's mind) to the agent."""
        self.context_store = context_store
        self.conversation.context_store = context_store
        self.prompts.context_store = context_store
        # Create agent memory harness
        from ..harness.memory.interface import AgentMemory

        self.memory = AgentMemory(context_store, session_id=self.session_id)
        self.prompts.memory = self.memory

    def enter_plan_mode(self) -> str:
        """Switch to plan mode (experimental design)."""
        if self.mode == "plan":
            return "Already in plan mode."
        self.mode = "plan"
        import gently.harness.plan_mode.tools  # noqa: F401

        self._update_system_prompt()
        emit(EventType.STATUS_CHANGED, {"field": "agent_mode", "value": "plan"}, source="agent")
        logger.info("Entered plan mode")
        return "Switched to plan mode. I'm now your experimental design collaborator."

    def enter_resolution_mode(self) -> str:
        """Switch into resolution mode at session start.

        The agent's job in this mode is to figure out what this session
        is for — continuing an existing plan item, resuming an interrupted
        session, starting standalone, or designing a new plan — and to
        call one of the resolution lifecycle tools to record that choice
        and transition into the right next mode.
        """
        if self.mode == "resolution":
            return "Already in resolution mode."
        self.mode = "resolution"
        if self.prompts:
            self.prompts.invalidate_context_cache()
        self._update_system_prompt()
        emit(
            EventType.STATUS_CHANGED,
            {"field": "agent_mode", "value": "resolution"},
            source="agent",
        )
        logger.info("Entered resolution mode")
        return "Resolution mode active. Determining what this session is for."

    def exit_resolution_mode(self, outcome: str | None = None) -> str:
        """Leave resolution mode for run mode.

        Called by resolution tools (attach_session_to_plan,
        mark_session_standalone, etc.) once the user has confirmed
        what this session is. ``outcome`` is a one-line summary the
        caller wants surfaced (the tool already returned its own
        description; this is just for logging).
        """
        if self.mode != "resolution":
            return ""
        self.mode = "run"
        if self.prompts:
            self.prompts.invalidate_context_cache()
        self._update_system_prompt()
        emit(
            EventType.STATUS_CHANGED,
            {"field": "agent_mode", "value": "run"},
            source="agent",
        )
        if outcome:
            logger.info(f"Exited resolution mode: {outcome}")
        else:
            logger.info("Exited resolution mode")
        return ""

    def exit_plan_mode(self) -> str:
        """Switch back to run mode.

        Resolves plan context so that any newly created imaging items
        are picked up as the active plan item automatically.
        """
        if self.mode == "run":
            return "Already in run mode."
        self.mode = "run"

        # Resolve plan context from the freshly created/updated plan
        result = ""
        if self.memory:
            active_id, candidates = self.memory.resolve_plan_context()
            if active_id:
                self.experiment.active_plan_item_id = active_id
                self.memory.active_plan_item_id = active_id
                item = self.context_store.get_plan_item(active_id) if self.context_store else None
                title = item.title if item else active_id
                result = f"Back to run mode. Active plan item: {title}"

                # Link session to campaign
                if item and self.session_id and self.context_store:
                    try:
                        self.context_store.link_session_campaign(
                            self.session_id,
                            item.campaign_id,
                        )
                    except Exception:
                        pass

                # Seed the Operation Plan from the plan item's tactics outline
                # (idempotent: no-op if plan already has active/done tactics).
                try:
                    from gently.app.tools.operation_plan_seed import (
                        seed_operation_plan_from_plan_item,
                    )

                    if self.session_id is not None:
                        seed_operation_plan_from_plan_item(self.context_store, self.session_id)
                except Exception:
                    logger.exception("operation-plan seeding failed")
            elif candidates:
                titles = [c[0].title for c in candidates]
                listing = ", ".join(titles[:5])
                result = (
                    f"Back to run mode. {len(candidates)} imaging tasks "
                    f"ready: {listing}. Which one are you working on?"
                )
            else:
                result = "Back to run mode."

            self.prompts.invalidate_context_cache()

        self._update_system_prompt()
        emit(EventType.STATUS_CHANGED, {"field": "agent_mode", "value": "run"}, source="agent")
        logger.info("Exited plan mode")
        return result

    # ===== Prompt & System Prompt =====

    def _update_system_prompt(self, context_summary: str | None = None):
        """Rebuild system prompt via PromptManager."""
        self.system_prompt = self.prompts.update_system_prompt(
            self.experiment,
            self.client,
            self.mode,
            context_summary,
            perceiver=getattr(self, "perceiver", None),
        )

    def _get_active_plan_summary(self) -> str | None:
        """Delegation shim for agent bridge access."""
        return self.prompts.get_active_plan_summary()

    def _get_tools_for_mode(self) -> list:
        """Get tools for current mode."""
        return self.prompts.get_tools_for_mode(self.mode, self._has_microscope())

    def _get_cached_system_prompt(self):
        """Get cached system prompt."""
        return self.prompts.get_cached_system_prompt(self.system_prompt)

    def invalidate_context_cache(self):
        """Invalidate the context summary cache."""
        self.prompts.invalidate_context_cache()

    # ===== Session Delegation =====

    def save_session(self) -> bool:
        """Save current session state."""
        return self.sessions.save_session(
            self.experiment, self.conversation.conversation_history, self.system_prompt
        )

    def _auto_save(self):
        """Auto-save session (non-blocking, silent on error)."""
        self.sessions.auto_save(
            self.experiment, self.conversation.conversation_history, self.system_prompt
        )

    def list_sessions(self) -> list[dict]:
        """List available sessions."""
        return self.sessions.list_sessions()

    def resume_session(self, session_id: str) -> bool:
        """Resume a session (public interface for CLI)."""
        return self.sessions.resume_session(
            session_id, self.experiment, self.conversation, self._update_system_prompt
        )

    # ===== Init Helpers =====

    def _init_interaction_logger(self):
        """Initialize the interaction logger for structured logging."""
        try:
            self.interaction_logger = InteractionLogger(
                storage_path=self.storage_path,
                session_id=self.session_id or "unknown",
                model=self.model,
            )
        except Exception as e:
            logging.getLogger(__name__).warning(f"Failed to init interaction logger: {e}")
            self.interaction_logger = None

    def _init_event_capture(self):
        """Open the per-session events.jsonl capture.

        Resolves the session folder via FileStore._session_dir so the log
        sits next to session.yaml / interaction_log.jsonl. Silent no-op
        when the session folder can't be resolved (e.g. test harness with
        a stripped-down agent) — replay just won't have a log to read.
        """
        from gently.eval import EventCapture

        try:
            session_dir = None
            sid = self.session_id
            if self.store is not None and sid:
                session_dir = self.store._session_dir(sid)
            if session_dir is None:
                logging.getLogger(__name__).debug(
                    "EventCapture: no session dir for %s — skipping", sid
                )
                return
            path = session_dir / "events.jsonl"
            self.event_capture = EventCapture(path)
            self.event_capture.start(self._event_bus)
        except Exception:
            logging.getLogger(__name__).exception("Failed to init event capture")
            self.event_capture = None

    def stop_event_capture(self):
        """Flush + close the events.jsonl. Idempotent; safe at shutdown."""
        if self.event_capture is not None:
            try:
                self.event_capture.stop()
            except Exception:
                logging.getLogger(__name__).exception("EventCapture stop failed")
            self.event_capture = None

    def _init_decision_log(self):
        """Open the per-session decisions.jsonl and wire it into conversation.

        Each call to ConversationManager.call_claude writes one Decision
        row (success or error) describing what production decided for the
        user turn. Shadow candidates write their own rows into separate
        logs and the two are diffed offline.
        """
        from gently.eval import DecisionLog

        try:
            session_dir = None
            sid = self.session_id
            if self.store is not None and sid:
                session_dir = self.store._session_dir(sid)
            if session_dir is None:
                logging.getLogger(__name__).debug(
                    "DecisionLog: no session dir for %s — skipping", sid
                )
                return
            path = session_dir / "decisions.jsonl"
            self.decision_log = DecisionLog(path)
            self.decision_log.open()
            self.conversation.decision_log = self.decision_log
        except Exception:
            logging.getLogger(__name__).exception("Failed to init decision log")
            self.decision_log = None

    def stop_decision_log(self):
        """Flush + close the decisions.jsonl. Idempotent; safe at shutdown."""
        if self.decision_log is not None:
            try:
                self.decision_log.close()
            except Exception:
                logging.getLogger(__name__).exception("DecisionLog close failed")
            self.decision_log = None
            if hasattr(self, "conversation") and self.conversation is not None:
                self.conversation.decision_log = None

    def _init_timelapse_orchestrator(self):
        """Initialize the timelapse orchestrator if microscope is connected."""
        if not self._has_microscope():
            return

        try:
            self.timelapse_orchestrator = TimelapseOrchestrator(
                microscope_client=self.client,
                experiment_state=self.experiment,
                perceiver=self.perceiver,
                on_volume_callback=self.on_volume_acquired,
                session_id=self.session_id,
                store=self.store,
                claude_client=self.claude,
                temperature_provider=lambda: (
                    self.temperature_sampler.latest if self.temperature_sampler else None
                ),
            )
        except Exception as e:
            logging.getLogger(__name__).warning(f"Failed to init timelapse orchestrator: {e}")
            self.timelapse_orchestrator = None

    async def _start_hardware_monitors(self):
        """Start the live hardware telemetry monitors (idempotent).

        Called at boot when the microscope is already connected, and by the
        device-layer watcher when the layer comes up mid-session. Each monitor
        is created only if absent, so repeated calls are cheap no-ops.
        """
        if self.microscope is None:
            return
        if self.device_state_monitor is None:
            try:
                from .device_state_monitor import DeviceStateMonitor

                self.device_state_monitor = DeviceStateMonitor(self.microscope)
                await self.device_state_monitor.start()
                logger.info("Device-state monitor started")
            except Exception as e:
                logger.warning(f"Failed to start device-state monitor: {e}")
                self.device_state_monitor = None
        if self.temperature_sampler is None:
            try:
                from .temperature_sampler import TemperatureSampler

                self.temperature_sampler = TemperatureSampler(
                    self.microscope, self.store, lambda: self.session_id
                )
                await self.temperature_sampler.start()
                logger.info("Temperature sampler started")
            except Exception as e:
                logger.warning(f"Failed to start temperature sampler: {e}")
                self.temperature_sampler = None
        # Stream bridges are constructed but left unstarted — streaming is
        # opt-in via explicit operator actions in the UI.
        if self.bottom_camera_monitor is None:
            try:
                from .bottom_camera_monitor import BottomCameraStreamMonitor

                self.bottom_camera_monitor = BottomCameraStreamMonitor(self.microscope)
                logger.info("Bottom-camera monitor ready (not started)")
            except Exception as e:
                logger.warning(f"Failed to construct bottom-camera monitor: {e}")
                self.bottom_camera_monitor = None
        if self.lightsheet_monitor is None:
            try:
                from .lightsheet_monitor import LightSheetStreamMonitor

                self.lightsheet_monitor = LightSheetStreamMonitor(self.microscope)
                logger.info("Lightsheet monitor ready (not started)")
            except Exception as e:
                logger.warning(f"Failed to construct lightsheet monitor: {e}")
                self.lightsheet_monitor = None

    async def _stop_hardware_monitors(self):
        """Stop + drop the live hardware telemetry monitors (idempotent).

        Mirror of ``_start_hardware_monitors`` — used both on shutdown and when
        the device layer goes away mid-session so telemetry stops cleanly.
        """
        for attr in (
            "bottom_camera_monitor",
            "lightsheet_monitor",
            "device_state_monitor",
            "temperature_sampler",
        ):
            mon = getattr(self, attr, None)
            if mon is not None:
                try:
                    await mon.stop()
                except Exception:
                    logger.exception(f"Failed to stop {attr}")
                setattr(self, attr, None)

    async def attach_hardware(self):
        """Wire the agent to a now-connected microscope (idempotent).

        Configures the device session (volume staging dir), registers the
        microscope tools, and starts the live telemetry monitors. Called by the
        device-layer watcher when the layer reaches 'ready' — so hardware brought
        up mid-session (from the launch gate or the Devices panel) becomes fully
        usable without a relaunch. The timelapse orchestrator already holds the
        same client object, so it needs no re-wiring here.
        """
        if self.client is None:
            return
        try:
            await self.client.configure_device_session(str(self.store.incoming_dir))
        except Exception as e:
            logger.warning(f"configure_device_session failed on attach: {e}")
        try:
            from gently.harness.microscope import register_microscope_tools

            register_microscope_tools(self.client)
        except Exception as e:
            logger.debug(f"microscope tool registration skipped on attach: {e}")
        await self._start_hardware_monitors()

    async def detach_hardware(self):
        """Tear down live hardware wiring when the device layer goes away.

        Stops the telemetry monitors. Registered microscope tools are left in
        the schema on purpose — they return clear "not connected" errors rather
        than vanishing (which would make the model hallucinate tool calls). The
        timelapse orchestrator is kept too; its client simply reads disconnected.
        Idempotent.
        """
        await self._stop_hardware_monitors()

    def _init_timeline_manager(self):
        """Initialize the timeline manager for event tracking."""
        try:
            # Per-session timeline.jsonl — points the TimelineManager at the
            # FileStore-indexed session folder so each session gets its own
            # event log. Falls back to ``<root>/sessions/<id>`` for the
            # legacy bare-id layout when the session isn't indexed yet.
            sid = self.sessions._session_id
            session_path = None
            if sid:
                session_path = self.store._session_dir(sid)
                if session_path is None:
                    session_path = self.store.root / "sessions" / sid
            if session_path is None:
                # No session yet — keep events in memory only by passing None.
                logging.getLogger(__name__).debug(
                    "TimelineManager started without persistence (no session_id yet)"
                )
            self.timeline_manager = TimelineManager(
                storage_path=session_path,
                max_events=1000,
                session_id=sid,
            )
            self.timeline_manager.start()
        except Exception as e:
            logging.getLogger(__name__).warning(f"Failed to init timeline manager: {e}")
            self.timeline_manager = None

    def _subscribe_to_cv_events(self):
        """Subscribe to CV subagent result events for EmbryoState integration."""
        try:
            self._cv_subscriptions = []

            def on_cv_result(event):
                try:
                    data = event.data
                    embryo_id = data.get("embryo_id")
                    result = data.get("result", {})
                    result_type = data.get("result_type", "analysis")

                    if embryo_id and embryo_id in self.experiment.embryos:
                        embryo = self.experiment.embryos[embryo_id]
                        structured = result.get("structured", result)
                        embryo.add_cv_result(result_type, structured)
                        logger.info(f"Updated {embryo_id} with CV {result_type} result")
                        self._auto_save()
                except Exception as e:
                    logger.warning(f"Error handling CV result event: {e}")

            unsub = self._event_bus.subscribe(EventType.CV_RESULT_READY, on_cv_result)
            self._cv_subscriptions.append(unsub)

            def on_stage_detected(event):
                try:
                    data = event.data
                    embryo_id = data.get("embryo_id")
                    if embryo_id and embryo_id in self.experiment.embryos:
                        embryo = self.experiment.embryos[embryo_id]
                        embryo.add_cv_result(
                            "stage_classification",
                            {
                                "stage": data.get("stage"),
                                "confidence": data.get("confidence"),
                                "nuclei_count": data.get("nuclei_count"),
                                "timepoint": data.get("timepoint"),
                            },
                        )
                except Exception as e:
                    logger.warning(f"Error handling stage detected event: {e}")

            unsub = self._event_bus.subscribe(EventType.STAGE_DETECTED, on_stage_detected)
            self._cv_subscriptions.append(unsub)

            def on_perception(event):
                # Bridge the perception loop's DETECTOR_EVALUATED into EmbryoState so
                # the prompt/display developmental stage reflects the live Perceiver.
                # (The STAGE_DETECTED wiring above is never emitted by the perception
                # path — this closes that long-standing gap.) Record only on an
                # actual stage CHANGE to keep cv_analyses a clean transition log and
                # avoid per-timepoint disk/cache churn; live stability/timing is read
                # straight from the Perceiver by the prompt snapshot + pull tool.
                try:
                    data = event.data
                    if data.get("skipped") or data.get("detector_name") != "perception":
                        return  # ignore recheck-skips and role=test pseudo-stages
                    embryo_id = data.get("embryo_id")
                    stage = data.get("stage")
                    # 'no_object' is an empty-field sentinel, not a developmental
                    # stage — don't mirror it into latest_developmental_stage.
                    if (
                        not stage
                        or stage == "no_object"
                        or not embryo_id
                        or embryo_id not in self.experiment.embryos
                    ):
                        return
                    embryo = self.experiment.embryos[embryo_id]
                    if stage == getattr(embryo, "latest_developmental_stage", None):
                        return  # steady state — nothing new to mirror
                    embryo.add_cv_result(
                        "stage_classification",
                        {
                            "stage": stage,
                            "timepoint": data.get("timepoint"),
                            "stability": data.get("stability"),
                            "temporal_analysis": data.get("temporal_analysis"),
                            "detector_name": "perception",
                        },
                    )
                    self.invalidate_context_cache()
                    self._auto_save()
                    logger.info(
                        "Perception: %s -> stage %s (t%s)", embryo_id, stage, data.get("timepoint")
                    )
                except Exception as e:
                    logger.warning(f"Error handling perception event: {e}")

            unsub = self._event_bus.subscribe(EventType.DETECTOR_EVALUATED, on_perception)
            self._cv_subscriptions.append(unsub)

            logger.debug("Subscribed to CV result events")

        except Exception as e:
            logger.warning(f"Failed to subscribe to CV events: {e}")
            self._cv_subscriptions = []

    # ===== Visualization Server Methods =====

    async def start_viz_server(
        self, port: int = settings.network.viz_port, ssl_certfile=None, ssl_keyfile=None
    ):
        """Start the visualization server for real-time feedback."""
        if self.viz_server is not None:
            logger.info("Visualization server already running")
            return

        try:
            from ..ui.web.server import VisualizationServer

            self.viz_server = VisualizationServer(
                port=port,
                event_bus=self._event_bus,
                gently_store=self.store,
                ssl_certfile=ssl_certfile,
                ssl_keyfile=ssl_keyfile,
            )
            await self.viz_server.start()
            scheme = "https" if ssl_certfile else "http"
            logger.info(f"Visualization server started at {scheme}://localhost:{port}")
            # Seed the viz tracker with current experiment state. The
            # event-bus replay path covers most cases, but in-memory event
            # history can roll past EMBRYO_DETECTED emissions from session
            # resume that happened in __init__ before this viz_server was
            # ever instantiated. Direct seeding makes the device-map
            # embryo overlay reliable across all startup orderings.
            try:
                tracker = getattr(self.viz_server, "timelapse_tracker", None)
                if tracker is not None and hasattr(tracker, "seed_from_experiment"):
                    n = tracker.seed_from_experiment(self.experiment)
                    if n:
                        logger.info(
                            "Seeded viz tracker with %d embryo(s) from experiment.",
                            n,
                        )
            except Exception as e:
                logger.debug("seed_from_experiment failed (non-fatal): %s", e)
        except ImportError as e:
            logger.warning(f"FastAPI not available for viz server: {e}")
            self.viz_server = None
        except Exception as e:
            logger.warning(f"Failed to start viz server: {e}")
            self.viz_server = None

        # Hardware telemetry monitors — meaningful only once the microscope
        # client is actually connected. If the device layer comes up later, the
        # device-layer watcher calls attach_hardware() → _start_hardware_monitors().
        # (With the always-created client, self.microscope is non-None even when
        # offline, so gate on the live connection, not on the object's existence.)
        if self.microscope is not None and getattr(self.microscope, "is_connected", False):
            await self._start_hardware_monitors()

        # Operation-plan updater is NOT hardware-gated (it needs the context
        # store, not the microscope) — start it whenever the store is present.
        if self.operation_plan_updater is None and self.context_store is not None:
            try:
                from .operation_plan_updater import OperationPlanUpdater

                self.operation_plan_updater = OperationPlanUpdater(
                    self.context_store, lambda: self.session_id
                )
                await self.operation_plan_updater.start()
                logger.info("Operation-plan updater started")
            except Exception as e:
                logger.warning(f"Failed to start operation-plan updater: {e}")
                self.operation_plan_updater = None

    async def stop_viz_server(self):
        """Stop the visualization server if running."""
        await self._stop_hardware_monitors()
        if self.operation_plan_updater is not None:
            try:
                await self.operation_plan_updater.stop()
            except Exception:
                logger.exception("Failed to stop operation-plan updater")
            self.operation_plan_updater = None
        if self.viz_server is not None:
            await self.viz_server.stop()
            self.viz_server = None
            logger.info("Visualization server stopped")

    def push_viz(
        self,
        array: np.ndarray,
        uid: str,
        data_type: str = "image",
        metadata: dict[str, Any] | None = None,
    ):
        """Non-blocking push of image to visualization server."""
        if self.viz_server is None:
            return

        try:
            asyncio.create_task(self.viz_server.push_image(array, uid, data_type, metadata or {}))
        except RuntimeError:
            pass
        except Exception as e:
            logger.debug(f"Failed to push image to viz server: {e}")

    # ===== Event Helpers =====

    def _has_microscope(self) -> bool:
        """Check if microscope server connection is available.

        Returns True if a client exists (even if temporarily disconnected)
        so that microscope tools remain in the Claude tool schema. The tools
        themselves check connectivity and return an error if the device layer
        is unreachable, which is better than Claude hallucinating XML tool
        calls when the tools are missing from the schema.
        """
        return self.client is not None

    def _emit_event(self, event_type: EventType, data: dict | None = None):
        """Emit an event to the event bus."""
        self._event_bus.publish(
            event_type=event_type,
            data=data or {},
            source="agent",
        )

    def _publish_embryos_update(self) -> None:
        """Broadcast the current embryo list as an EMBRYOS_UPDATE event.

        Wired into ExperimentState.on_embryos_changed at agent init so every
        add / remove / restore / nickname change snaps a fresh full-list
        snapshot onto the bus. The viz server's wildcard subscription forwards
        it straight to connected browsers — that's how the Devices > Map page
        learns about embryos without a poll loop.
        """
        if self._event_bus is None:
            return
        try:
            embryos = [e.to_dict() for e in self.experiment.embryos.values()]
        except Exception:
            logger.exception("Failed to serialise embryos for EMBRYOS_UPDATE")
            return
        payload = {
            "embryos": embryos,
            "count": len(embryos),
            "session_id": getattr(self, "session_id", None),
        }
        try:
            self._event_bus.publish(
                event_type=EventType.EMBRYOS_UPDATE,
                data=payload,
                source="agent.experiment",
            )
        except Exception:
            logger.exception("Failed to publish EMBRYOS_UPDATE")

    def _mark_significant_action(self, action_type: str):
        """Mark that a significant action occurred (triggers auto-save)."""
        self._auto_save()
        self._emit_event(
            EventType.SESSION_SAVED,
            {
                "session_id": self.sessions._session_id,
                "action_type": action_type,
            },
        )

    # ===== Public Message API =====

    async def handle_message(self, user_message: str) -> str:
        """
        Main entry point for user interaction.

        Parameters
        ----------
        user_message : str
            Message from user

        Returns
        -------
        str
            Response from agent
        """
        if quick_response := self.conversation.try_quick_response(
            user_message,
            self.experiment,
            self.mode,
            self.enter_plan_mode,
            self.exit_plan_mode,
        ):
            return quick_response

        if not self.api_enabled:
            return _NO_API_NOTICE

        # Update system prompt with current state and context awareness
        context_summary = await self.prompts.get_cached_context_summary(
            self.experiment, self.timelapse_orchestrator, self.timeline_manager
        )
        self._update_system_prompt(context_summary)

        # Add user message to history
        self.conversation.conversation_history.append({"role": "user", "content": user_message})

        tools = self._get_tools_for_mode()
        cached_prompt = self._get_cached_system_prompt()

        return await self.conversation.call_claude(
            user_message, cached_prompt, tools, self.mode, self._auto_save
        )

    async def handle_message_stream(self, user_message: str, autonomous: bool = False):
        """
        Handle message with streaming response.

        Yields chunks as they arrive from Claude API.
        Supports asend() for sending values back (e.g., user choice selections).

        Parameters
        ----------
        user_message : str
            Message from user
        autonomous : bool
            When True (wake turns only), sets _autonomous_active after the
            turn-lock is acquired so the registry backstop never fires while a
            user turn is still holding the lock.

        Yields
        ------
        dict
            Chunks with 'type' and data
        """
        if quick_response := self.conversation.try_quick_response(
            user_message,
            self.experiment,
            self.mode,
            self.enter_plan_mode,
            self.exit_plan_mode,
        ):
            yield {"type": "text", "text": quick_response}
            return

        if not self.api_enabled:
            yield {"type": "text", "text": _NO_API_NOTICE}
            return

        # Hold the turn-lock for the whole streamed turn so an autonomous wake
        # turn cannot interleave on the shared conversation_history.
        lock = getattr(self, "_turn_lock", None)
        acquired = False
        if lock is not None:
            await lock.acquire()
            acquired = True
        if autonomous:
            self._autonomous_active = True
        try:
            context_summary = await self.prompts.get_cached_context_summary(
                self.experiment, self.timelapse_orchestrator, self.timeline_manager
            )
            self._update_system_prompt(context_summary)

            self.conversation.conversation_history.append({"role": "user", "content": user_message})

            tools = self._get_tools_for_mode()
            cached_prompt = self._get_cached_system_prompt()

            inner_gen = self.conversation.call_claude_stream(
                cached_prompt,
                tools,
                tool_label_fn=self.conversation.tool_label,
                auto_save_fn=self._auto_save,
            )
            sent_value = None

            try:
                while True:
                    if sent_value is None:
                        chunk = await inner_gen.__anext__()
                    else:
                        chunk = await inner_gen.asend(sent_value)
                    sent_value = yield chunk
            except StopAsyncIteration:
                return
        finally:
            if autonomous:
                self._autonomous_active = False
            if acquired and lock is not None:
                lock.release()

    async def run_wake_turn(
        self, wake_note: str, trigger: str | None = None, interactive: bool = False
    ):
        """Drive one autonomous (no-user) turn for the wake-router.

        Runs through the normal streaming pipeline (so it acquires the turn-lock
        and is recorded to conversation history / auto-saved). Brackets the turn
        with an 'autonomous_start' (carrying the wake trigger) and a synthesized
        'stream_end' so it streams to the web chat distinctly. Passes
        autonomous=True to handle_message_stream, which sets _autonomous_active
        after acquiring the turn-lock so the registry backstop refuses
        irreversible tools only while this turn holds the lock.
        When interactive (ASK mode) a choice_request round-trips through the
        operator; otherwise it is auto-cancelled. Run mode only.
        """
        if self.mode != "run":
            logger.info("Wake turn skipped — agent not in run mode (mode=%s)", self.mode)
            return ""

        async def _emit(chunk):
            cb = getattr(self, "on_message_callback", None)
            if cb is None:
                return
            try:
                res = cb(chunk)
                if asyncio.iscoroutine(res):
                    await res
            except Exception:
                logger.debug("on_message_callback failed for wake chunk", exc_info=True)

        await _emit({"type": "autonomous_start", "trigger": trigger or ""})
        text_parts = []
        agen = self.handle_message_stream(wake_note, autonomous=True)
        sent_value = None
        try:
            while True:
                try:
                    if sent_value is None:
                        chunk = await agen.__anext__()
                    else:
                        chunk = await agen.asend(sent_value)
                        sent_value = None
                except StopAsyncIteration:
                    break
                ctype = chunk.get("type") if isinstance(chunk, dict) else None
                if ctype == "text":
                    text_parts.append(chunk.get("text", ""))
                if ctype == "choice_request":
                    # Resolve via the operator (ASK) or auto-cancel (AUTO).
                    sent_value = await self._resolve_wake_choice(chunk, _emit, interactive)
                    continue  # don't re-emit the raw choice_request
                await _emit(chunk)
        except Exception:
            logger.exception("run_wake_turn error")
        finally:
            try:
                # If the lock was acquired, closing the generator triggers
                # handle_message_stream's finally, resetting _autonomous_active
                # and releasing the turn-lock.
                await agen.aclose()
            except Exception:
                pass
            await _emit({"type": "stream_end"})
        summary = "".join(text_parts).strip()
        if summary:
            logger.info("Autonomous wake turn result: %s", summary[:500])
        return summary

    async def _resolve_wake_choice(self, chunk, emit, interactive):
        """Resolve a choice_request raised during a wake turn.

        AUTO (or no operator channel) -> 'cancelled'. ASK -> register a future via
        the web choice-factory, broadcast the picker to clients, and await the
        operator's selection (timeout -> 'skip' so an unanswered picker can't hold
        the turn-lock forever)."""
        choice_data = chunk.get("choice_data", {}) if isinstance(chunk, dict) else {}
        factory = getattr(self, "_wake_choice_factory", None)
        if not interactive or factory is None:
            logger.info(
                "Wake picker auto-cancelled (interactive=%s, channel=%s)",
                interactive,
                factory is not None,
            )
            return "cancelled"
        try:
            future = factory(choice_data)  # registers future + sets request_id
        except Exception:
            logger.exception("wake choice factory failed")
            return "cancelled"
        request_id = choice_data.get("request_id", "")
        await emit({**chunk, "origin": "wake", "request_id": request_id})
        from gently.app.wake_router import ASK_TIMEOUT_SEC

        try:
            selected = await asyncio.wait_for(future, timeout=ASK_TIMEOUT_SEC)
        except asyncio.TimeoutError:
            logger.info("Wake ASK timed out (%.0fs) -> skip", ASK_TIMEOUT_SEC)
            selected = "skip"
        except asyncio.CancelledError:
            # The picker future was cancelled (e.g. the operator disconnected) —
            # treat as a cancelled proposal so the turn finishes cleanly.
            logger.info("Wake ASK future cancelled -> cancelled")
            selected = "cancelled"
        except Exception:
            selected = "cancelled"
        finally:
            # Don't leak the future in the router-scoped registry on timeout/cancel.
            discard = getattr(self, "_wake_choice_discard", None)
            if discard is not None and request_id:
                try:
                    discard(request_id)
                except Exception:
                    pass
        return selected or "skip"

    async def get_tool_call(self, user_message: str) -> dict | None:
        """Dry-run tool call (for benchmarking)."""
        context_summary = await self.prompts.get_cached_context_summary(
            self.experiment, self.timelapse_orchestrator, self.timeline_manager
        )
        self._update_system_prompt(context_summary)
        tools = self._get_tools_for_mode()
        cached_prompt = self._get_cached_system_prompt()
        return await self.conversation.get_tool_call(user_message, cached_prompt, tools)

    # === Experiment Management Methods ===

    def load_embryos_from_database(self, database: dict):
        """Load embryos from calibration database."""
        if "embryos" not in database:
            return

        for embryo_id, embryo_data in database["embryos"].items():
            position = embryo_data.get("stage_position_after_centering_um", {})
            calibration = embryo_data.get("calibration", {})

            self.experiment.add_embryo(
                embryo_id=embryo_id,
                position=position,
                calibration=calibration,
                uid=embryo_data.get("uid"),
            )

        self._update_system_prompt()

    def import_embryos_from_session(self, session_id: str, clear_existing: bool = False) -> dict:
        """
        Import embryos from another session into the current experiment.

        Reads embryo data from the store's database (primary) and falls back
        to the JSON snapshot if needed.

        Parameters
        ----------
        session_id : str
            Session ID to import embryos from
        clear_existing : bool
            If True, clear existing embryos before importing

        Returns
        -------
        dict
            Import result with 'success', 'imported', 'skipped', 'errors'
        """
        # Primary source: embryos table in DB
        db_embryos = self.store.list_embryos(session_id) if self.store else []

        # Build a unified dict keyed by embryo_id
        embryo_states: dict[str, Any] = {}
        for row in db_embryos:
            eid = row.get("embryo_id", "")
            if not eid:
                continue
            src_role = row.get("role")
            if not src_role:
                logger.warning(
                    "Embryo %s in source session has no role; defaulting to "
                    "'unassigned'. Re-mark before starting acquisition.",
                    eid,
                )
                src_role = "unassigned"
            coarse = row.get("position_coarse") or {}
            fine = row.get("position_fine") or {}
            embryo_states[eid] = {
                # stage_position remains for legacy consumers of this snapshot.
                # It carries the resolved (fine ?? coarse) view; add_embryo()
                # downstream treats it as coarse, but the explicit
                # position_fine field below will override that on restore.
                "stage_position": dict(fine) if fine else dict(coarse),
                "position_coarse": dict(coarse),
                "position_fine": dict(fine),
                "calibration": row.get("calibration") or {},
                "uid": row.get("embryo_uid"),
                "user_label": row.get("nickname"),
                "role": src_role,
            }

        # Fallback: JSON snapshot (legacy path)
        if not embryo_states:
            session_data = self.store.load_session_snapshot(session_id)
            if session_data:
                embryo_states = session_data.get("embryo_states", {})

        if not embryo_states:
            return {
                "success": False,
                "error": "No embryos found in session",
                "imported": [],
                "skipped": [],
            }

        if clear_existing:
            self.experiment.embryos.clear()
            self.experiment.notify_embryos_changed()

        imported = []
        skipped = []
        errors = []

        for embryo_id, embryo_data in embryo_states.items():
            if embryo_id in self.experiment.embryos and not clear_existing:
                skipped.append(embryo_id)
                continue

            try:
                # Prefer explicit coarse/fine when the snapshot has them
                # (FileStore path); fall back to flat stage_position for the
                # legacy JSON-snapshot path which only carries the resolved view.
                position_coarse = embryo_data.get("position_coarse")
                position_fine = embryo_data.get("position_fine")
                if position_coarse is None and position_fine is None:
                    position_coarse = embryo_data.get("stage_position", {})
                calibration = embryo_data.get("calibration", {})
                source_uid = embryo_data.get("uid") or f"{session_id}_{embryo_id}"

                self.experiment.add_embryo(
                    embryo_id=embryo_id,
                    position=position_coarse or {},
                    position_fine=position_fine or {},
                    calibration=calibration,
                    user_label=embryo_data.get("user_label"),
                    uid=source_uid,
                    role=embryo_data.get("role") or "unassigned",
                )

                embryo = self.experiment.embryos[embryo_id]
                embryo.nickname = embryo_data.get("nickname")
                embryo.interval_seconds = embryo_data.get("interval_seconds")
                embryo.num_slices = embryo_data.get("num_slices", 50)
                embryo.exposure_ms = embryo_data.get("exposure_ms", 10.0)
                embryo.priority = embryo_data.get("priority", "normal")
                embryo.acquisition_mode = embryo_data.get("acquisition_mode", "volume")

                # Light budget import. Prefer fields already on embryo_data
                # (future schema may persist these directly on embryo.yaml);
                # otherwise reconstruct by walking volumes/*.meta.yaml in
                # the source session. TODO: when dose-tracking lands as a
                # first-class persisted field (per-embryo dose_log.jsonl +
                # `dose:` summary on embryo.yaml), this fallback should be
                # removed and the import should fail loudly if dose is
                # missing.
                dose = self._compute_imported_dose(session_id, embryo_id)
                embryo.exposure_count = embryo_data.get("exposure_count") or dose["exposure_count"]
                embryo.total_exposure_ms = (
                    embryo_data.get("total_exposure_ms") or dose["total_exposure_ms"]
                )
                embryo.timepoints_acquired = (
                    embryo_data.get("timepoints_acquired") or dose["exposure_count"]
                )

                last_imaged_str = embryo_data.get("last_imaged")
                if last_imaged_str:
                    try:
                        embryo.last_imaged = datetime.fromisoformat(last_imaged_str)
                    except (ValueError, TypeError):
                        embryo.last_imaged = dose["last_imaged"]
                else:
                    embryo.last_imaged = dose["last_imaged"]

                imported.append(embryo_id)

            except Exception as e:
                errors.append(f"{embryo_id}: {str(e)}")

        self._update_system_prompt()
        self._mark_significant_action("embryo_import")

        return {
            "success": len(imported) > 0,
            "imported": imported,
            "skipped": skipped,
            "errors": errors,
            "source_session": session_id,
        }

    def _compute_imported_dose(self, source_session_id: str, embryo_id: str) -> dict:
        """Reconstruct an embryo's realized 488 nm photodose from the source
        session's per-volume meta files.

        Hack used by import_embryos_from_session — exposure history is not
        currently persisted on the FileStore embryo record, so we walk
        ``embryos/{id}/volumes/*.meta.yaml`` and sum
        ``num_slices * exposure_ms`` per acquisition. This captures normal
        acquisitions, calibration sub-acquisitions, and burst frames so
        long as each writes a meta file (which the existing acquisition
        path does).

        TODO: replace with reading a persisted ``dose:`` block from
        embryo.yaml once dose-tracking is first-class.
        """
        from datetime import datetime
        from pathlib import Path

        import yaml

        result: dict[str, Any] = {
            "exposure_count": 0,
            "total_exposure_ms": 0.0,
            "last_imaged": None,
        }

        if not self.store:
            return result

        # FileStore exposes _session_dir(session_id) → resolved Path.
        session_dir_fn = getattr(self.store, "_session_dir", None)
        sd = session_dir_fn(source_session_id) if callable(session_dir_fn) else None
        if sd is None:
            return result

        vols_dir = Path(sd) / "embryos" / embryo_id / "volumes"
        if not vols_dir.is_dir():
            return result

        latest = None
        for meta_path in sorted(vols_dir.glob("*.meta.yaml")):
            try:
                doc = yaml.safe_load(meta_path.read_text()) or {}
            except Exception:
                continue
            md = doc.get("metadata") or {}
            num_slices = md.get("num_slices")
            if num_slices is None:
                shape = doc.get("shape") or []
                num_slices = shape[0] if shape else 0
            exposure_ms = md.get("exposure_ms") or 0.0
            try:
                result["total_exposure_ms"] += float(num_slices) * float(exposure_ms)
            except (TypeError, ValueError):
                pass
            result["exposure_count"] += 1
            acq = doc.get("acquired_at")
            if acq and (latest is None or acq > latest):
                latest = acq

        if latest:
            try:
                result["last_imaged"] = datetime.fromisoformat(latest)
            except (ValueError, TypeError):
                pass

        return result

    async def on_volume_acquired(
        self, embryo_id: str, timepoint: int, volume_data, volume_path=None
    ):
        """Callback when a volume is acquired."""
        embryo = self.experiment.embryos.get(embryo_id)
        if not embryo:
            return

        if hasattr(volume_data, "read_volume"):
            volume = volume_data.read_volume()
        else:
            volume = volume_data

        stored_path = None
        if self.store and self.session_id:
            try:
                self.store.register_embryo(
                    self.session_id,
                    embryo_id,
                    position_coarse=embryo.position_coarse or None,
                    position_fine=embryo.position_fine or None,
                    calibration=embryo.calibration,
                    role=embryo.role,
                )
                acq_metadata = {
                    "num_slices": embryo.num_slices,
                    "exposure_ms": embryo.exposure_ms,
                    "interval_seconds": embryo.interval_seconds,
                    "acquisition_mode": embryo.acquisition_mode,
                    "laser_power_488_pct": embryo.laser_power_488_pct,
                    "role": embryo.role,
                    "calibration": embryo.calibration,
                }
                if volume_path is not None:
                    stored_path = self.store.register_volume(
                        self.session_id,
                        embryo_id,
                        timepoint,
                        incoming_path=Path(volume_path),
                        metadata=acq_metadata,
                        volume_data=volume,
                    )
                else:
                    stored_path = self.store.put_volume(
                        self.session_id,
                        embryo_id,
                        timepoint,
                        volume,
                        metadata=acq_metadata,
                    )
            except StorageError:
                raise
            except Exception as e:
                logger.error(f"FileStore write failed: {e}")

        session_prefix = f"{self.session_id[:8]}_" if self.session_id else ""
        volume_uid = f"volume_{session_prefix}{embryo_id}_t{timepoint:04d}"
        projection_uid = f"proj_{session_prefix}{embryo_id}_t{timepoint:04d}"

        if self.viz_server and volume is not None:
            try:
                from gently.core.imaging import (
                    apply_crop_bounds,
                    compute_crop_bounds,
                    projection_three_view,
                )

                view_a = volume[0] if volume.ndim == 4 else volume

                if view_a.ndim == 3:
                    # view_a is already one view (4D handled on the line above).
                    # No aspect-ratio splitting: native frames are 4:1.
                    bounds = compute_crop_bounds(view_a)
                    cropped = apply_crop_bounds(view_a, bounds)
                    three_view_img, _ = projection_three_view(cropped)
                else:
                    three_view_img = view_a.astype(np.float32)
                    if three_view_img.max() > three_view_img.min():
                        three_view_img = (
                            (three_view_img - three_view_img.min())
                            / (three_view_img.max() - three_view_img.min())
                            * 255
                        )
                    three_view_img = three_view_img.astype(np.uint8)

                self.push_viz(
                    array=three_view_img,
                    uid=projection_uid,
                    data_type="volume_projection",
                    metadata={
                        "embryo_id": embryo_id,
                        "timepoint": timepoint,
                        "shape": list(volume.shape),
                        "projection_uid": projection_uid,
                        "volume_uid": volume_uid,
                        "projection_type": "three_view",
                    },
                )
            except Exception as e:
                logger.warning(f"Failed to push to viz: {e}")

        self._emit_event(
            EventType.VOLUME_ACQUIRED,
            {
                "embryo_id": embryo_id,
                "timepoint": timepoint,
                "volume_uid": volume_uid,
                "projection_uid": projection_uid,
                "volume_path": str(stored_path) if stored_path else None,
                "shape": list(volume.shape),
            },
        )

        return {
            "volume_uid": volume_uid,
            "projection_uid": projection_uid,
        }

    def should_stop_experiment(self) -> bool:
        """Check if experiment should stop (e.g., all embryos hatched)."""
        if not self.experiment.embryos:
            return False
        return all(e.should_skip for e in self.experiment.embryos.values())

    def get_embryo_acquisition_order(self) -> list[str]:
        """Get embryo acquisition order based on priority."""
        high = [
            e.id
            for e in self.experiment.embryos.values()
            if e.priority == "high" and not e.should_skip
        ]
        normal = [
            e.id
            for e in self.experiment.embryos.values()
            if e.priority == "normal" and not e.should_skip
        ]
        low = [
            e.id
            for e in self.experiment.embryos.values()
            if e.priority == "low" and not e.should_skip
        ]
        return high + normal + low

    def decide_parameters(self, embryo_id: str, timepoint: int) -> dict:
        """Get current acquisition parameters for embryo."""
        embryo = self.experiment.embryos.get(embryo_id)
        if not embryo:
            return {"num_slices": 50, "exposure_ms": 10.0}
        return {"num_slices": embryo.num_slices, "exposure_ms": embryo.exposure_ms}

    def decide_next_interval(self, timepoint: int) -> float:
        """Decide interval until next timepoint."""
        active_embryos = [e for e in self.experiment.embryos.values() if not e.should_skip]
        if not active_embryos:
            return 120.0
        return min(
            (e.interval_seconds for e in active_embryos if e.interval_seconds is not None),
            default=120.0,
        )

    # === Perception System Integration ===

    async def check_blank_image(
        self,
        volume: np.ndarray,
        embryo_id: str,
    ) -> bool:
        """Check if an image appears blank using Claude Vision."""
        try:
            volume = np.squeeze(volume)
            max_proj = np.max(volume, axis=0) if volume.ndim == 3 else volume

            if np.std(max_proj) < 1.0 or np.max(max_proj) < 10:
                logger.warning(f"[BLANK_CHECK] {embryo_id}: Numerical check indicates blank image")
                return True

            import base64
            import io

            from PIL import Image

            if max_proj.max() > 0:
                normalized = (max_proj / max_proj.max() * 255).astype(np.uint8)
            else:
                return True

            img = Image.fromarray(normalized)
            buffer = io.BytesIO()
            img.save(buffer, format="PNG")
            b64_image = base64.b64encode(buffer.getvalue()).decode()

            prompt = """\
Look at this microscopy image. Is this a VALID microscopy image or a BLANK/CORRUPTED image?

A BLANK or CORRUPTED image shows:
- Mostly uniform gray/black with no structure
- No visible biological features
- Static noise only
- Hardware artifacts (stripes, patterns) without actual sample

A VALID image shows:
- Visible biological structure (embryo, cells, etc.)
- Even if the embryo is small or faint, there should be clear structure

Respond with ONLY: VALID or BLANK"""

            response = await asyncio.to_thread(
                lambda: self.claude.messages.create(
                    model=settings.models.fast,
                    max_tokens=10,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {"type": "text", "text": prompt},
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": "image/png",
                                        "data": b64_image,
                                    },
                                },
                            ],
                        }
                    ],
                )
            )

            block = cast(anthropic.types.TextBlock, response.content[0])
            result = block.text.strip().upper()
            is_blank = "BLANK" in result

            if is_blank:
                logger.warning(f"[BLANK_CHECK] {embryo_id}: Claude Vision detected blank image")

            return is_blank

        except (
            anthropic.APIConnectionError,
            anthropic.RateLimitError,
            anthropic.APIStatusError,
        ) as e:
            logger.error(f"[BLANK_CHECK] Claude API error for {embryo_id}: {e}")
            return False
        except Exception as e:
            logger.error(f"[BLANK_CHECK] Error checking {embryo_id}: {e}")
            return False
