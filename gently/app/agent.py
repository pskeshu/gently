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
from typing import Dict, List, Optional, Callable, Any, TYPE_CHECKING
from datetime import datetime
from pathlib import Path

import anthropic
import numpy as np

from ..exceptions import StorageError, AgentError
from ..settings import settings

if TYPE_CHECKING:
    from ..ui.web.server import VisualizationServer

logger = logging.getLogger(__name__)

from ..harness.state import ExperimentState, EmbryoState, ImageRecord
from ..harness.orchestration.plan_synthesis import PlanSynthesizer, PlanLibrary, PlanValidator
from ..harness.tools.registry import get_tool_registry
from gently_perception import Perceiver

# Import tools package to trigger @tool decorator registration
from . import tools as _tools  # noqa: F401
from ..harness.session.interaction_logger import InteractionLogger
from .orchestration.timelapse import TimelapseOrchestrator
from ..harness.session.timeline import TimelineManager
from ..core import EventType, get_event_bus, emit
from ..core.file_store import FileStore

from ..harness.conversation import ConversationManager
from ..harness.session.manager import SessionManager
from ..harness.prompts.manager import PromptManager


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
        api_key: Optional[str] = None,
        storage_path: Path = Path("./experiment_data"),
        model: str = settings.models.main,
        microscope_client=None,
        session_id: Optional[str] = None,
        store: FileStore = None,
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
        """
        if store is None:
            raise ValueError("FileStore is required. Pass store=FileStore(path) to agent.")

        # API client with interleaved thinking support
        self.claude = anthropic.Anthropic(
            api_key=api_key or os.getenv("ANTHROPIC_API_KEY"),
            default_headers={"anthropic-beta": "interleaved-thinking-2025-05-14"}
        )
        self.model = model

        # Mode: "run" (default), "plan" (experimental design), or
        # "resolution" (figuring out what this session is for at startup).
        # Resolution mode is entered explicitly by the startup wiring when
        # there are unblocked plan items that need a decision; it is not
        # the default for resumed sessions.
        self.mode: str = "run"

        # Context store (agent's mind — set via set_context_store)
        self.context_store: Optional[Any] = None

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
            plan_library=PlanLibrary(),
            validator=PlanValidator()
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
        self.on_message_callback: Optional[Callable] = None
        self.choice_handler: Optional[Callable] = None

        # Interaction logger for structured logging (research data collection)
        self.interaction_logger: Optional[InteractionLogger] = None

        # Event capture — durable log of every EventBus event during this
        # session. Substrate for offline replay / shadow-mode A/B of
        # candidate orchestrator architectures.
        self.event_capture = None

        # Decision log — what production decided at each turn (tool calls,
        # response text, prompt hash). Pairs with event capture so a
        # candidate replay can be diffed against production turn-by-turn.
        self.decision_log = None

        # Timelapse orchestrator (initialized when microscope connected)
        self.timelapse_orchestrator: Optional[TimelapseOrchestrator] = None

        # Timeline manager for tracking events
        self.timeline_manager: Optional[TimelineManager] = None

        # Visualization server for real-time feedback
        self.viz_server: Optional["VisualizationServer"] = None

        # Device-state monitor (bridges device-layer SSE → EventBus)
        self.device_state_monitor = None
        # Opt-in bottom-camera stream bridge — created when viz starts, but
        # left unstarted until the operator clicks "Start camera" in the UI.
        self.bottom_camera_monitor = None

        # ===== Create delegate managers =====

        # Conversation manager (LLM calls, tool execution, token tracking)
        self.conversation = ConversationManager(
            client=self.claude,
            model=self.model,
            tool_registry=get_tool_registry(),
        )
        # Wire tool execution context
        self.conversation._tool_context = {
            'agent': self,
            'client': getattr(self, 'microscope', None),
            'microscope': getattr(self, 'microscope', None),
            'databroker': getattr(self, 'databroker', None),
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
            self._emit_event(EventType.SESSION_RESTORED, {
                'session_id': session_id,
                'embryo_count': len(self.experiment.embryos),
                'message_count': len(self.conversation.conversation_history),
            })
        else:
            self.sessions.create_session()
            self._emit_event(EventType.SESSION_STARTED, {
                'session_id': self.sessions.session_id,
            })

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

        # Build initial system prompt
        self._update_system_prompt()

    # ===== Backward-Compat Delegation Properties =====

    @property
    def session_id(self) -> str:
        """Get current session ID."""
        return self.sessions.session_id

    @property
    def _session_id(self) -> Optional[str]:
        """Internal session ID (backward compat)."""
        return self.sessions._session_id

    @_session_id.setter
    def _session_id(self, value):
        self.sessions._session_id = value

    @property
    def conversation_history(self) -> List[Dict]:
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

    def exit_resolution_mode(self, outcome: str = None) -> str:
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
                item = self.context_store.get_plan_item(active_id)
                title = item.title if item else active_id
                result = f"Back to run mode. Active plan item: {title}"

                # Link session to campaign
                if item and self.session_id and self.context_store:
                    try:
                        self.context_store.link_session_campaign(
                            self.session_id, item.campaign_id,
                        )
                    except Exception:
                        pass
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

    def _update_system_prompt(self, context_summary: str = None):
        """Rebuild system prompt via PromptManager."""
        self.system_prompt = self.prompts.update_system_prompt(
            self.experiment, self.client, self.mode, context_summary
        )

    def _get_active_plan_summary(self) -> Optional[str]:
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

    def list_sessions(self) -> List[Dict]:
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
                    "EventCapture: no session dir for %s — skipping", sid)
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
                    "DecisionLog: no session dir for %s — skipping", sid)
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
            )
        except Exception as e:
            logging.getLogger(__name__).warning(f"Failed to init timelapse orchestrator: {e}")
            self.timelapse_orchestrator = None

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
                        embryo.add_cv_result("stage_classification", {
                            "stage": data.get("stage"),
                            "confidence": data.get("confidence"),
                            "nuclei_count": data.get("nuclei_count"),
                            "timepoint": data.get("timepoint"),
                        })
                except Exception as e:
                    logger.warning(f"Error handling stage detected event: {e}")

            unsub = self._event_bus.subscribe(EventType.STAGE_DETECTED, on_stage_detected)
            self._cv_subscriptions.append(unsub)

            logger.debug("Subscribed to CV result events")

        except Exception as e:
            logger.warning(f"Failed to subscribe to CV events: {e}")
            self._cv_subscriptions = []

    # ===== Visualization Server Methods =====

    async def start_viz_server(self, port: int = settings.network.viz_port, ssl_certfile=None, ssl_keyfile=None):
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

        # Start device-state monitor: streams device-layer SSE onto the EventBus,
        # which the viz server already wildcard-subscribes to. Safe to skip
        # silently if the microscope client isn't ready — the user just won't
        # see live readouts until next session.
        if self.microscope is not None and self.device_state_monitor is None:
            try:
                from .device_state_monitor import DeviceStateMonitor
                self.device_state_monitor = DeviceStateMonitor(self.microscope)
                await self.device_state_monitor.start()
                logger.info("Device-state monitor started")
            except Exception as e:
                logger.warning(f"Failed to start device-state monitor: {e}")
                self.device_state_monitor = None

        # Construct the bottom-camera monitor — but don't start it. Streaming
        # is opt-in and waits for an explicit operator action via the
        # /api/devices/bottom_camera/stream/start route.
        if self.microscope is not None and self.bottom_camera_monitor is None:
            try:
                from .bottom_camera_monitor import BottomCameraStreamMonitor
                self.bottom_camera_monitor = BottomCameraStreamMonitor(self.microscope)
                logger.info("Bottom-camera monitor ready (not started)")
            except Exception as e:
                logger.warning(f"Failed to construct bottom-camera monitor: {e}")
                self.bottom_camera_monitor = None

    async def stop_viz_server(self):
        """Stop the visualization server if running."""
        if self.bottom_camera_monitor is not None:
            try:
                await self.bottom_camera_monitor.stop()
            except Exception:
                logger.exception("Failed to stop bottom-camera monitor")
            self.bottom_camera_monitor = None
        if self.device_state_monitor is not None:
            try:
                await self.device_state_monitor.stop()
            except Exception:
                logger.exception("Failed to stop device-state monitor")
            self.device_state_monitor = None
        if self.viz_server is not None:
            await self.viz_server.stop()
            self.viz_server = None
            logger.info("Visualization server stopped")

    def push_viz(
        self,
        array: np.ndarray,
        uid: str,
        data_type: str = "image",
        metadata: Optional[Dict[str, Any]] = None,
    ):
        """Non-blocking push of image to visualization server."""
        if self.viz_server is None:
            return

        try:
            asyncio.create_task(
                self.viz_server.push_image(array, uid, data_type, metadata or {})
            )
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

    def _emit_event(self, event_type: EventType, data: Optional[Dict] = None):
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
        self._emit_event(EventType.SESSION_SAVED, {
            'session_id': self.sessions._session_id,
            'action_type': action_type,
        })

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
            user_message, self.experiment, self.mode,
            self.enter_plan_mode, self.exit_plan_mode,
        ):
            return quick_response

        # Update system prompt with current state and context awareness
        context_summary = await self.prompts.get_cached_context_summary(
            self.experiment, self.timelapse_orchestrator, self.timeline_manager
        )
        self._update_system_prompt(context_summary)

        # Add user message to history
        self.conversation.conversation_history.append({
            "role": "user",
            "content": user_message
        })

        tools = self._get_tools_for_mode()
        cached_prompt = self._get_cached_system_prompt()

        return await self.conversation.call_claude(
            user_message, cached_prompt, tools, self.mode, self._auto_save
        )

    async def handle_message_stream(self, user_message: str):
        """
        Handle message with streaming response.

        Yields chunks as they arrive from Claude API.
        Supports asend() for sending values back (e.g., user choice selections).

        Parameters
        ----------
        user_message : str
            Message from user

        Yields
        ------
        dict
            Chunks with 'type' and data
        """
        if quick_response := self.conversation.try_quick_response(
            user_message, self.experiment, self.mode,
            self.enter_plan_mode, self.exit_plan_mode,
        ):
            yield {'type': 'text', 'text': quick_response}
            return

        context_summary = await self.prompts.get_cached_context_summary(
            self.experiment, self.timelapse_orchestrator, self.timeline_manager
        )
        self._update_system_prompt(context_summary)

        self.conversation.conversation_history.append({
            "role": "user",
            "content": user_message
        })

        tools = self._get_tools_for_mode()
        cached_prompt = self._get_cached_system_prompt()

        inner_gen = self.conversation.call_claude_stream(
            cached_prompt, tools,
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

    async def get_tool_call(self, user_message: str) -> Optional[Dict]:
        """Dry-run tool call (for benchmarking)."""
        context_summary = await self.prompts.get_cached_context_summary(
            self.experiment, self.timelapse_orchestrator, self.timeline_manager
        )
        self._update_system_prompt(context_summary)
        tools = self._get_tools_for_mode()
        cached_prompt = self._get_cached_system_prompt()
        return await self.conversation.get_tool_call(user_message, cached_prompt, tools)

    # === Experiment Management Methods ===

    def load_embryos_from_database(self, database: Dict):
        """Load embryos from calibration database."""
        if 'embryos' not in database:
            return

        for embryo_id, embryo_data in database['embryos'].items():
            position = embryo_data.get('stage_position_after_centering_um', {})
            calibration = embryo_data.get('calibration', {})

            self.experiment.add_embryo(
                embryo_id=embryo_id,
                position=position,
                calibration=calibration,
                uid=embryo_data.get('uid'),
            )

        self._update_system_prompt()

    def import_embryos_from_session(self, session_id: str, clear_existing: bool = False) -> Dict:
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
        embryo_states = {}
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
                embryo_states = session_data.get('embryo_states', {})

        if not embryo_states:
            return {
                'success': False,
                'error': "No embryos found in session",
                'imported': [],
                'skipped': [],
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
                position_coarse = embryo_data.get('position_coarse')
                position_fine = embryo_data.get('position_fine')
                if position_coarse is None and position_fine is None:
                    position_coarse = embryo_data.get('stage_position', {})
                calibration = embryo_data.get('calibration', {})
                source_uid = embryo_data.get('uid') or f"{session_id}_{embryo_id}"

                self.experiment.add_embryo(
                    embryo_id=embryo_id,
                    position=position_coarse or {},
                    position_fine=position_fine or {},
                    calibration=calibration,
                    user_label=embryo_data.get('user_label'),
                    uid=source_uid,
                    role=embryo_data.get('role') or 'unassigned',
                )

                embryo = self.experiment.embryos[embryo_id]
                embryo.nickname = embryo_data.get('nickname')
                embryo.interval_seconds = embryo_data.get('interval_seconds')
                embryo.num_slices = embryo_data.get('num_slices', 50)
                embryo.exposure_ms = embryo_data.get('exposure_ms', 10.0)
                embryo.priority = embryo_data.get('priority', 'normal')
                embryo.acquisition_mode = embryo_data.get('acquisition_mode', 'volume')

                # Light budget import. Prefer fields already on embryo_data
                # (future schema may persist these directly on embryo.yaml);
                # otherwise reconstruct by walking volumes/*.meta.yaml in
                # the source session. TODO: when dose-tracking lands as a
                # first-class persisted field (per-embryo dose_log.jsonl +
                # `dose:` summary on embryo.yaml), this fallback should be
                # removed and the import should fail loudly if dose is
                # missing.
                dose = self._compute_imported_dose(session_id, embryo_id)
                embryo.exposure_count = (
                    embryo_data.get('exposure_count')
                    or dose['exposure_count']
                )
                embryo.total_exposure_ms = (
                    embryo_data.get('total_exposure_ms')
                    or dose['total_exposure_ms']
                )
                embryo.timepoints_acquired = (
                    embryo_data.get('timepoints_acquired')
                    or dose['exposure_count']
                )

                last_imaged_str = embryo_data.get('last_imaged')
                if last_imaged_str:
                    try:
                        embryo.last_imaged = datetime.fromisoformat(last_imaged_str)
                    except (ValueError, TypeError):
                        embryo.last_imaged = dose['last_imaged']
                else:
                    embryo.last_imaged = dose['last_imaged']

                imported.append(embryo_id)

            except Exception as e:
                errors.append(f"{embryo_id}: {str(e)}")

        self._update_system_prompt()
        self._mark_significant_action("embryo_import")

        return {
            'success': len(imported) > 0,
            'imported': imported,
            'skipped': skipped,
            'errors': errors,
            'source_session': session_id,
        }

    def _compute_imported_dose(self, source_session_id: str, embryo_id: str) -> Dict:
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
        import yaml
        from datetime import datetime
        from pathlib import Path

        result = {
            'exposure_count': 0,
            'total_exposure_ms': 0.0,
            'last_imaged': None,
        }

        if not self.store:
            return result

        # FileStore exposes _session_dir(session_id) → resolved Path.
        session_dir_fn = getattr(self.store, '_session_dir', None)
        sd = session_dir_fn(source_session_id) if callable(session_dir_fn) else None
        if sd is None:
            return result

        vols_dir = Path(sd) / 'embryos' / embryo_id / 'volumes'
        if not vols_dir.is_dir():
            return result

        latest = None
        for meta_path in sorted(vols_dir.glob('*.meta.yaml')):
            try:
                doc = yaml.safe_load(meta_path.read_text()) or {}
            except Exception:
                continue
            md = doc.get('metadata') or {}
            num_slices = md.get('num_slices')
            if num_slices is None:
                shape = doc.get('shape') or []
                num_slices = shape[0] if shape else 0
            exposure_ms = md.get('exposure_ms') or 0.0
            try:
                result['total_exposure_ms'] += float(num_slices) * float(exposure_ms)
            except (TypeError, ValueError):
                pass
            result['exposure_count'] += 1
            acq = doc.get('acquired_at')
            if acq and (latest is None or acq > latest):
                latest = acq

        if latest:
            try:
                result['last_imaged'] = datetime.fromisoformat(latest)
            except (ValueError, TypeError):
                pass

        return result

    async def on_volume_acquired(self, embryo_id: str, timepoint: int,
                                volume_data, volume_path=None):
        """Callback when a volume is acquired."""
        embryo = self.experiment.embryos.get(embryo_id)
        if not embryo:
            return

        if hasattr(volume_data, 'read_volume'):
            volume = volume_data.read_volume()
        else:
            volume = volume_data

        stored_path = None
        if self.store and self.session_id:
            try:
                self.store.register_embryo(
                    self.session_id, embryo_id,
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
                        self.session_id, embryo_id, timepoint,
                        incoming_path=Path(volume_path),
                        metadata=acq_metadata,
                        volume_data=volume,
                    )
                else:
                    stored_path = self.store.put_volume(
                        self.session_id, embryo_id, timepoint, volume,
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
                    projection_three_view,
                    compute_crop_bounds,
                    apply_crop_bounds,
                )

                view_a = volume[0] if volume.ndim == 4 else volume

                if view_a.ndim == 3:
                    z_depth, height, width = view_a.shape
                    if width > height * 2:
                        view_a = view_a[:, :, :width // 2]
                    bounds = compute_crop_bounds(view_a)
                    cropped = apply_crop_bounds(view_a, bounds)
                    three_view_img, _ = projection_three_view(cropped)
                else:
                    three_view_img = view_a.astype(np.float32)
                    if three_view_img.max() > three_view_img.min():
                        three_view_img = (three_view_img - three_view_img.min()) / (three_view_img.max() - three_view_img.min()) * 255
                    three_view_img = three_view_img.astype(np.uint8)

                self.push_viz(
                    array=three_view_img,
                    uid=projection_uid,
                    data_type="volume_projection",
                    metadata={
                        'embryo_id': embryo_id,
                        'timepoint': timepoint,
                        'shape': list(volume.shape),
                        'projection_uid': projection_uid,
                        'volume_uid': volume_uid,
                        'projection_type': 'three_view',
                    }
                )
            except Exception as e:
                logger.warning(f"Failed to push to viz: {e}")

        self._emit_event(EventType.VOLUME_ACQUIRED, {
            'embryo_id': embryo_id,
            'timepoint': timepoint,
            'volume_uid': volume_uid,
            'projection_uid': projection_uid,
            'volume_path': str(stored_path) if stored_path else None,
            'shape': list(volume.shape),
        })

        return {
            'volume_uid': volume_uid,
            'projection_uid': projection_uid,
        }

    def should_stop_experiment(self) -> bool:
        """Check if experiment should stop (e.g., all embryos hatched)."""
        if not self.experiment.embryos:
            return False
        return all(e.should_skip for e in self.experiment.embryos.values())

    def get_embryo_acquisition_order(self) -> List[str]:
        """Get embryo acquisition order based on priority."""
        high = [e.id for e in self.experiment.embryos.values() if e.priority == "high" and not e.should_skip]
        normal = [e.id for e in self.experiment.embryos.values() if e.priority == "normal" and not e.should_skip]
        low = [e.id for e in self.experiment.embryos.values() if e.priority == "low" and not e.should_skip]
        return high + normal + low

    def decide_parameters(self, embryo_id: str, timepoint: int) -> Dict:
        """Get current acquisition parameters for embryo."""
        embryo = self.experiment.embryos.get(embryo_id)
        if not embryo:
            return {'num_slices': 50, 'exposure_ms': 10.0}
        return {
            'num_slices': embryo.num_slices,
            'exposure_ms': embryo.exposure_ms
        }

    def decide_next_interval(self, timepoint: int) -> float:
        """Decide interval until next timepoint."""
        active_embryos = [e for e in self.experiment.embryos.values() if not e.should_skip]
        if not active_embryos:
            return 120.0
        return min(e.interval_seconds for e in active_embryos)

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

            import io
            import base64
            from PIL import Image

            if max_proj.max() > 0:
                normalized = (max_proj / max_proj.max() * 255).astype(np.uint8)
            else:
                return True

            img = Image.fromarray(normalized)
            buffer = io.BytesIO()
            img.save(buffer, format='PNG')
            b64_image = base64.b64encode(buffer.getvalue()).decode()

            prompt = """Look at this microscopy image. Is this a VALID microscopy image or a BLANK/CORRUPTED image?

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
                self.claude.messages.create,
                model=settings.models.fast,
                max_tokens=10,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": b64_image
                            }
                        }
                    ]
                }]
            )

            result = response.content[0].text.strip().upper()
            is_blank = "BLANK" in result

            if is_blank:
                logger.warning(f"[BLANK_CHECK] {embryo_id}: Claude Vision detected blank image")

            return is_blank

        except (anthropic.APIConnectionError, anthropic.RateLimitError, anthropic.APIStatusError) as e:
            logger.error(f"[BLANK_CHECK] Claude API error for {embryo_id}: {e}")
            return False
        except Exception as e:
            logger.error(f"[BLANK_CHECK] Error checking {embryo_id}: {e}")
            return False
