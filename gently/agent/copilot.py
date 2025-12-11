"""
Main Microscopy Copilot implementation

Now integrated with:
- Event Bus for async message passing between components
- Session Manager for persistence and resume
- Data Store for UID-based data flow
"""

import asyncio
import json
import logging
import os
from typing import Dict, List, Optional, Callable, Any, TYPE_CHECKING
from datetime import datetime
from pathlib import Path

import anthropic
import numpy as np

if TYPE_CHECKING:
    from ..visualization.server import VisualizationServer

logger = logging.getLogger(__name__)

from .state import ExperimentState, EmbryoState, ImageRecord
from .image_manager import ImageManager
from .plan_synthesis import PlanSynthesizer, PlanLibrary, PlanValidator
from .prompts import build_system_prompt, build_context_message
from .tool_registry import get_tool_registry
from .detector_registry import DetectorRegistry
from .detection_queue import DetectionQueue
from .detection_verifier import DetectionVerifier
from .interaction_logger import InteractionLogger
from .timelapse_orchestrator import TimelapseOrchestrator
from .timeline import TimelineManager
from ..session import SessionManager
from ..core import EventType, get_event_bus, emit


class MicroscopyCopilot:
    """
    Conversational AI copilot for microscopy experiments

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
        model: str = "claude-opus-4-5-20251101",
        microscope_client=None,
        session_id: Optional[str] = None,
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
        """
        # API client with interleaved thinking support
        self.claude = anthropic.Anthropic(
            api_key=api_key or os.getenv("ANTHROPIC_API_KEY"),
            default_headers={"anthropic-beta": "interleaved-thinking-2025-05-14"}
        )
        self.model = model

        # Conversation state
        self.conversation_history: List[Dict] = []
        self.system_prompt: str = ""

        # Experiment state
        self.experiment = ExperimentState()

        # Storage path
        self.storage_path = Path(storage_path)

        # Image management
        self.image_manager = ImageManager(
            storage_path=self.storage_path / "images",
            history_length=10
        )

        # Plan synthesis
        self.plan_synthesizer = PlanSynthesizer(
            plan_library=PlanLibrary(),
            validator=PlanValidator()
        )

        # Detector system
        self.detector_registry = DetectorRegistry(
            storage_path=self.storage_path / "detector_registry.json"
        )
        self.detection_queue = DetectionQueue(
            registry=self.detector_registry,
            image_manager=self.image_manager,
            claude_client=self.claude,
            model=self.model,
            on_detection_callback=self._on_detection_fired
        )

        # Detection verifier (challenger agent for AUTO mode)
        self.detection_verifier = DetectionVerifier(
            claude_client=self.claude,
            model="claude-haiku-4-5-20251001"  # Use Haiku for speed/cost
        )

        # Session management
        self.session_manager = SessionManager(
            sessions_dir=self.storage_path / "sessions",
            auto_save=True
        )

        # Hardware interface via RPC client
        self.client = microscope_client

        # Databroker (optional, for data catalog integration)
        # Get from client if available
        self.databroker = getattr(microscope_client, '_db', None) if microscope_client else None

        # Callbacks
        self.on_message_callback: Optional[Callable] = None
        self.choice_handler: Optional[Callable] = None  # For interactive choice UI

        # Event bus for async messaging
        self._event_bus = get_event_bus()

        # Interaction logger for structured logging (research data collection)
        self.interaction_logger: Optional[InteractionLogger] = None

        # Timelapse orchestrator (initialized when microscope connected)
        self.timelapse_orchestrator: Optional[TimelapseOrchestrator] = None

        # Timeline manager for tracking events
        self.timeline_manager: Optional[TimelineManager] = None

        # Visualization server for real-time feedback
        self.viz_server: Optional["VisualizationServer"] = None

        # Token usage tracking for session
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0
        self.api_call_count: int = 0
        self.cache_creation_tokens: int = 0  # Tokens written to cache
        self.cache_read_tokens: int = 0  # Tokens read from cache (90% cheaper)

        # Context summary caching (for state awareness)
        self._context_summary_cache: Optional[str] = None
        self._context_summary_time: Optional[datetime] = None
        self._context_summary_ttl: int = 300  # 5 minutes

        # Initialize or resume session
        if session_id:
            self.resume_session(session_id)
        else:
            self.session_manager.create_session()
            self._emit_event(EventType.SESSION_STARTED, {
                'session_id': self.session_id,
            })

        # Set session-specific storage path for images
        if self.session_id:
            self.image_manager.set_session(self.session_id)

        # Initialize interaction logger (for research data collection)
        self._init_interaction_logger()

        # Initialize timelapse orchestrator (if microscope connected)
        self._init_timelapse_orchestrator()

        # Initialize timeline manager (subscribes to event bus)
        self._init_timeline_manager()

        # Subscribe to CV result events for EmbryoState integration
        self._subscribe_to_cv_events()

        # Build initial system prompt
        self._update_system_prompt()

    def _update_system_prompt(self, context_summary: str = None):
        """
        Rebuild system prompt with current experiment state and connection status.

        Parameters
        ----------
        context_summary : str, optional
            AI-generated context summary for session awareness.
            If None, no context section is included in the prompt.
        """
        # Build connection status
        if self.client:
            connection_status = {
                'queue_server': self.client.is_connected,
                'sam_server': self.client.has_sam,
                'databroker': self.client.has_databroker and self.client.is_connected,
            }
        else:
            connection_status = None  # Offline mode

        self.system_prompt = build_system_prompt(
            self.experiment, connection_status, context_summary
        )

    def _gather_context_data(self) -> dict:
        """
        Gather raw context data for summarization.

        Collects timelapse status, recent events, and detection results
        to provide situational awareness to the LLM.

        Returns
        -------
        dict
            Context data including timelapse status, events, and detections
        """
        import json
        data = {
            'current_time': datetime.now().isoformat(),
            'timelapse_status': None,
            'recent_events': [],
            'recent_detections': [],
            'detection_reasoning': [],  # Include vision API reasoning
        }

        # Timelapse status
        if self.timelapse_orchestrator:
            try:
                status = self.timelapse_orchestrator.get_status()
                # get_status() returns TimelapseState object, not dict
                data['timelapse_status'] = {
                    'state': status.status.value if status.status else 'unknown',
                    'total_timepoints': status.total_timepoints or 0,
                    'started_at': status.started_at.isoformat() if status.started_at else None,
                    'embryo_count': len(status.embryos) if status.embryos else 0,
                }
            except Exception as e:
                logger.debug(f"Could not get timelapse status: {e}")

        # Recent timeline events (last 20)
        if self.timeline_manager:
            try:
                events = self.timeline_manager.get_events(limit=20, session_id='current')
                data['recent_events'] = [
                    {
                        'type': e.event_subtype,
                        'time': e.timestamp.isoformat(),
                        'embryo': e.embryo_id,
                        'detector': e.detector_name,
                        'timepoint': e.timepoint,
                        'confidence': e.confidence,
                    }
                    for e in events
                ]
            except Exception as e:
                logger.debug(f"Could not get timeline events: {e}")

        # Recent detection results with reasoning (from embryo states)
        try:
            for embryo_id, embryo_state in self.experiment.embryos.items():
                if not hasattr(embryo_state, 'detection_results'):
                    continue
                for detector_name, results in embryo_state.detection_results.items():
                    # Get last 3 results per detector
                    recent_results = results[-3:] if len(results) > 3 else results
                    for r in recent_results:
                        if r.get('detected'):
                            data['recent_detections'].append({
                                'detector': detector_name,
                                'embryo': embryo_id,
                                'timepoint': r.get('timepoint'),
                                'confidence': r.get('confidence'),
                            })
                            # Include reasoning if available
                            if r.get('reasoning'):
                                data['detection_reasoning'].append({
                                    'detector': detector_name,
                                    'embryo': embryo_id,
                                    'timepoint': r.get('timepoint'),
                                    'reasoning': r.get('reasoning')[:500],  # Truncate long reasoning
                                })
        except Exception as e:
            logger.debug(f"Could not get detection results: {e}")

        return data

    async def _generate_context_summary(self) -> str:
        """
        Generate concise context summary using Haiku.

        Calls Claude Haiku to summarize raw context data into
        a brief, actionable summary for the main LLM.

        Returns
        -------
        str
            Brief context summary (2-3 sentences)
        """
        import json
        raw_data = self._gather_context_data()

        # Skip if nothing interesting
        has_timelapse = raw_data['timelapse_status'] is not None
        has_events = len(raw_data['recent_events']) > 0
        has_detections = len(raw_data['recent_detections']) > 0

        if not (has_timelapse or has_events or has_detections):
            return ""

        prompt = f"""Summarize the current microscopy session state in 2-3 sentences for another AI assistant.
Focus on: timelapse status (is it running, completed, or idle?), time since last activity, and notable detections.
Be factual and concise.

Raw session data:
{json.dumps(raw_data, indent=2, default=str)}

Write a brief status summary. Examples:
- "Timelapse completed 10h ago with 233 timepoints. Hatching was detected at timepoints 175-193 with HIGH confidence."
- "Timelapse is currently running for embryo_1 at timepoint 45. No detections yet."
- "No active timelapse. Last session had 50 timepoints, with comma stage detected at t=30."
"""

        try:
            response = await asyncio.to_thread(
                self.claude.messages.create,
                model="claude-haiku-4-5-20251001",
                max_tokens=150,
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text.strip()
        except Exception as e:
            logger.warning(f"Failed to generate context summary: {e}")
            return ""

    async def _get_cached_context_summary(self) -> str:
        """
        Get context summary with caching.

        Caches the context summary for 5 minutes to avoid
        excessive API calls during rapid interactions.

        Returns
        -------
        str
            Cached or newly generated context summary
        """
        now = datetime.now()
        if (self._context_summary_cache is None or
            self._context_summary_time is None or
            (now - self._context_summary_time).total_seconds() > self._context_summary_ttl):
            self._context_summary_cache = await self._generate_context_summary()
            self._context_summary_time = now
        return self._context_summary_cache

    def invalidate_context_cache(self):
        """Invalidate the context summary cache to force regeneration."""
        self._context_summary_cache = None
        self._context_summary_time = None

    def _get_cached_system_prompt(self):
        """Get system prompt formatted for Anthropic prompt caching.

        Returns the system prompt as a list with cache_control to enable
        prompt caching, which dramatically reduces token costs for the
        ~4K token system prompt on subsequent API calls.

        Uses 1-hour TTL since microscopy sessions have gaps during
        timelapse acquisition (5-min default would expire too quickly).
        """
        return [
            {
                "type": "text",
                "text": self.system_prompt,
                "cache_control": {"type": "ephemeral", "ttl": "1h"}
            }
        ]

    def _track_token_usage(self, response):
        """Track token usage from API response, including cache metrics"""
        if hasattr(response, 'usage'):
            usage = response.usage
            self.total_input_tokens += usage.input_tokens
            self.total_output_tokens += usage.output_tokens
            self.api_call_count += 1

            # Track cache metrics if available (prompt caching)
            self.cache_creation_tokens += getattr(usage, 'cache_creation_input_tokens', 0)
            self.cache_read_tokens += getattr(usage, 'cache_read_input_tokens', 0)

    @property
    def current_context_tokens(self) -> int:
        """Estimate current context window size in tokens (what Claude sees per call)"""
        # System prompt
        system_tokens = len(self.system_prompt) // 4 if self.system_prompt else 0

        # Tool schemas (roughly constant)
        tool_tokens = 10000  # ~10K tokens for 65 tools

        # Conversation history - handle various content types safely
        conv_chars = 0
        for msg in self.conversation_history:
            content = msg.get('content', '')
            if isinstance(content, str):
                conv_chars += len(content)
            elif isinstance(content, list):
                # Content can be a list of content blocks
                for block in content:
                    if isinstance(block, dict):
                        # Text block or other dict-based content
                        conv_chars += len(str(block.get('text', '')))
                    elif hasattr(block, 'text'):
                        # Anthropic SDK content block (TextBlock, etc.)
                        conv_chars += len(str(block.text))
                    else:
                        # Fallback: estimate from string repr
                        conv_chars += len(str(block))
            else:
                # Fallback for other types
                conv_chars += len(str(content))

        conv_tokens = conv_chars // 4

        return system_tokens + tool_tokens + conv_tokens

    @property
    def token_usage_summary(self) -> str:
        """Get human-readable token usage summary"""
        # Note: input_tokens is SEPARATE from cache tokens (per Anthropic API)
        # Total = input_tokens + cache_read + cache_created + output_tokens
        cache_read = self.cache_read_tokens
        cache_created = self.cache_creation_tokens
        total_input = self.total_input_tokens + cache_read + cache_created
        total = total_input + self.total_output_tokens

        # Cost estimate (Claude Sonnet pricing with 1-hour cache)
        input_cost = self.total_input_tokens * 0.003 / 1000  # $3/M input
        cache_read_cost = cache_read * 0.0003 / 1000  # $0.30/M cached (90% off)
        cache_write_cost = cache_created * 0.006 / 1000  # $6/M cache write (2x for 1h TTL)
        output_cost = self.total_output_tokens * 0.015 / 1000  # $15/M output
        total_cost = input_cost + cache_read_cost + cache_write_cost + output_cost

        # Calculate savings from caching (what it would cost without cache)
        cost_without_cache = (total_input * 0.003 + self.total_output_tokens * 0.015) / 1000
        savings = cost_without_cache - total_cost

        summary = (
            f"Tokens: {total:,} total ({total_input:,} in, {self.total_output_tokens:,} out) | "
            f"API calls: {self.api_call_count} | Est. cost: ${total_cost:.3f}"
        )
        if cache_read > 0:
            summary += f" (cache saved ${savings:.3f})"
        return summary

    async def _call_api_with_retry(self, api_func, *args, max_retries=3, **kwargs):
        """
        Call an API function with retry logic for transient errors.

        Parameters
        ----------
        api_func : callable
            The API function to call
        max_retries : int
            Maximum number of retry attempts
        *args, **kwargs
            Arguments to pass to the API function

        Returns
        -------
        The API response

        Raises
        ------
        Exception
            If all retries fail
        """
        from anthropic import APIStatusError

        retry_delay = 1.0

        for attempt in range(max_retries):
            try:
                return await asyncio.to_thread(api_func, *args, **kwargs)
            except APIStatusError as e:
                error_type = getattr(e, 'body', {})
                if isinstance(error_type, dict):
                    error_type = error_type.get('error', {}).get('type', '')

                # Retry on overloaded or rate limit errors
                is_retryable = (
                    error_type in ('overloaded_error', 'rate_limit_error') or
                    'overloaded' in str(e).lower() or
                    'rate_limit' in str(e).lower()
                )

                if is_retryable and attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                    logger.warning(f"API error ({error_type}), retrying in {wait_time:.1f}s (attempt {attempt + 1}/{max_retries})")
                    await asyncio.sleep(wait_time)
                    continue

                # Re-raise if not retryable or out of retries
                raise

        raise RuntimeError(f"API call failed after {max_retries} retries")

    def _init_interaction_logger(self):
        """Initialize the interaction logger for structured logging"""
        try:
            self.interaction_logger = InteractionLogger(
                storage_path=self.storage_path,
                session_id=self.session_id or "unknown",
                model=self.model,
            )
        except Exception as e:
            # Don't fail if logger can't be initialized
            import logging
            logging.getLogger(__name__).warning(f"Failed to init interaction logger: {e}")
            self.interaction_logger = None

    def _init_timelapse_orchestrator(self):
        """Initialize the timelapse orchestrator if microscope is connected"""
        if not self._has_microscope():
            return

        try:
            self.timelapse_orchestrator = TimelapseOrchestrator(
                microscope_client=self.client,
                experiment_state=self.experiment,
                detection_queue=self.detection_queue,
                on_volume_callback=self.on_volume_acquired,
            )
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"Failed to init timelapse orchestrator: {e}")
            self.timelapse_orchestrator = None

    def _init_timeline_manager(self):
        """Initialize the timeline manager for event tracking"""
        try:
            # Store timeline in sessions directory
            timeline_path = self.session_manager.sessions_dir
            self.timeline_manager = TimelineManager(
                storage_path=timeline_path,
                max_events=1000,
                session_id=self.session_id,  # Tag events with current session
            )
            self.timeline_manager.start()
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"Failed to init timeline manager: {e}")
            self.timeline_manager = None

    def _subscribe_to_cv_events(self):
        """
        Subscribe to CV subagent result events for EmbryoState integration.

        When the CV agent completes analysis, it publishes CV_RESULT_READY events.
        This handler updates the corresponding EmbryoState with the results,
        making them accessible via /embryos and persistent across sessions.
        """
        try:
            # Store unsubscribe functions for cleanup
            self._cv_subscriptions = []

            # Subscribe to CV_RESULT_READY for direct result handling
            def on_cv_result(event):
                """Handle CV result ready event"""
                try:
                    data = event.data
                    embryo_id = data.get("embryo_id")
                    result = data.get("result", {})
                    result_type = data.get("result_type", "analysis")

                    if embryo_id and embryo_id in self.experiment.embryos:
                        embryo = self.experiment.embryos[embryo_id]

                        # Extract structured result
                        structured = result.get("structured", result)

                        # Add to EmbryoState
                        embryo.add_cv_result(result_type, structured)

                        logger.info(f"Updated {embryo_id} with CV {result_type} result")

                        # Auto-save session
                        if self.session_manager:
                            self._save_session_state()

                except Exception as e:
                    logger.warning(f"Error handling CV result event: {e}")

            unsub = self._event_bus.subscribe(EventType.CV_RESULT_READY, on_cv_result)
            self._cv_subscriptions.append(unsub)

            # Also subscribe to specific result types for backwards compatibility
            def on_stage_detected(event):
                """Handle stage detection event"""
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

    async def start_viz_server(self, port: int = 8080):
        """
        Start the visualization server for real-time feedback.

        Opens a web dashboard at http://localhost:{port} for viewing
        live images during calibration and acquisition.

        Parameters
        ----------
        port : int
            Port for web server (default: 8080)
        """
        if self.viz_server is not None:
            logger.info("Visualization server already running")
            return

        try:
            from ..visualization.server import VisualizationServer

            self.viz_server = VisualizationServer(
                port=port,
                event_bus=self._event_bus,
            )
            await self.viz_server.start()
            logger.info(f"Visualization server started at http://localhost:{port}")

            # Wire up CV subagent's segmentation module to push results to viz
            try:
                from ..cv_subagent.tools.segmentation import set_viz_server
                set_viz_server(self.viz_server)
                logger.info("CV subagent segmentation wired to viz server")
            except ImportError:
                logger.debug("CV subagent not available for viz server wiring")
        except ImportError as e:
            logger.warning(f"FastAPI not available for viz server: {e}")
            self.viz_server = None
        except Exception as e:
            logger.warning(f"Failed to start viz server: {e}")
            self.viz_server = None

    async def stop_viz_server(self):
        """Stop the visualization server if running."""
        if self.viz_server is not None:
            # Clear CV subagent segmentation reference
            try:
                from ..cv_subagent.tools.segmentation import set_viz_server
                set_viz_server(None)
            except ImportError:
                pass

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
        """
        Non-blocking push of image to visualization server.

        Uses asyncio.create_task() for fire-and-forget behavior.
        Copilot doesn't wait for viz server to process the image.

        Parameters
        ----------
        array : np.ndarray
            Image or volume to push (2D or 3D). 3D volumes are max-projected.
        uid : str
            Unique identifier for this image
        data_type : str
            Type: 'calibration', 'focus_sweep', 'volume_projection', 'edge_detection', etc.
        metadata : dict, optional
            Additional metadata (embryo_id, galvo, piezo, score, etc.)
        """
        if self.viz_server is None:
            return  # No viz server, silently skip

        try:
            loop = asyncio.get_running_loop()
            asyncio.create_task(
                self.viz_server.push_image(array, uid, data_type, metadata or {})
            )
        except RuntimeError:
            # No running event loop - shouldn't happen in async context
            pass
        except Exception as e:
            # Fire-and-forget: never block copilot for viz errors
            logger.debug(f"Failed to push image to viz server: {e}")

    def _has_microscope(self) -> bool:
        """Check if microscope server connection is available"""
        return self.client is not None

    # ===== Session Management Methods =====

    def resume_session(self, session_id: str) -> bool:
        """
        Resume a previous session

        Restores conversation history, experiment state, and detector configs.

        Parameters
        ----------
        session_id : str
            Session ID to resume

        Returns
        -------
        bool
            True if resumed successfully
        """
        session = self.session_manager.load_session(session_id)
        if not session:
            return False

        # Get state to restore
        state = self.session_manager.sync_to_copilot()

        # Restore conversation history
        self.conversation_history = state.get('conversation_history', [])

        # Restore experiment state
        experiment_data = state.get('experiment_data', {})
        embryo_states = state.get('embryo_states', {})

        # Restore embryos
        for embryo_id, embryo_data in embryo_states.items():
            self.experiment.add_embryo(
                embryo_id=embryo_id,
                position=embryo_data.get('stage_position', {}),
                calibration=embryo_data.get('calibration', {}),
                user_label=embryo_data.get('user_label'),
            )
            # Restore additional embryo state
            embryo = self.experiment.embryos[embryo_id]
            embryo.nickname = embryo_data.get('nickname')
            embryo.interval_seconds = embryo_data.get('interval_seconds')  # None = use timelapse default
            embryo.num_slices = embryo_data.get('num_slices', 50)
            embryo.exposure_ms = embryo_data.get('exposure_ms', 10.0)
            embryo.priority = embryo_data.get('priority', 'normal')
            embryo.timepoints_acquired = embryo_data.get('timepoints_acquired', 0)
            embryo.should_skip = embryo_data.get('should_skip', False)
            embryo.skip_reason = embryo_data.get('skip_reason')

        # Restore detection history
        detection_history = state.get('detection_history', {})
        for embryo_id, detections in detection_history.items():
            if embryo_id in self.experiment.embryos:
                for det in detections:
                    detector_name = det.pop('detector', 'unknown')
                    self.experiment.embryos[embryo_id].add_detection_result(
                        detector_name, det
                    )

        # Restore detector configs (if detector registry supports it)
        detector_configs = state.get('detector_configs', {})
        # Note: detector registry already persists to its own file,
        # so we don't need to restore from session state

        # Update image storage path for this session
        self.image_manager.set_session(session_id)

        # Emit session restored event
        self._emit_event(EventType.SESSION_RESTORED, {
            'session_id': session_id,
            'embryo_count': len(embryo_states),
            'message_count': len(self.conversation_history),
        })

        return True

    def _auto_save(self):
        """Auto-save session (non-blocking, silent on error)"""
        try:
            self.session_manager.update_state(
                conversation=self.conversation_history,
                experiment=self.experiment.to_dict(),
                system_prompt=self.system_prompt,
            )
            self.session_manager.save_session()
        except Exception:
            pass  # Silent fail for auto-save

    def save_session(self) -> bool:
        """
        Save current session state

        Returns
        -------
        bool
            True if saved successfully
        """
        # Sync current state to session
        self.session_manager.sync_from_copilot(
            conversation_history=self.conversation_history,
            experiment=self.experiment,
            detector_registry=self.detector_registry,
            system_prompt=self.system_prompt,
        )
        return self.session_manager.save_session()

    def list_sessions(self) -> List[Dict]:
        """
        List available sessions

        Returns
        -------
        list of dict
            Session summaries
        """
        return self.session_manager.list_sessions()

    def _emit_event(self, event_type: EventType, data: Optional[Dict] = None):
        """
        Emit an event to the event bus

        Parameters
        ----------
        event_type : EventType
            Type of event to emit
        data : dict, optional
            Event payload
        """
        self._event_bus.publish(
            event_type=event_type,
            data=data or {},
            source="copilot",
        )

    def _mark_significant_action(self, action_type: str):
        """
        Mark that a significant action occurred

        Triggers sync and auto-save.

        Parameters
        ----------
        action_type : str
            Type of action (acquisition, detection, calibration, etc.)
        """
        # Sync current state to session
        self.session_manager.sync_from_copilot(
            conversation_history=self.conversation_history,
            experiment=self.experiment,
            detector_registry=self.detector_registry,
            system_prompt=self.system_prompt,
        )
        # Trigger auto-save
        self.session_manager.mark_significant_action(action_type)

        # Emit session saved event
        self._emit_event(EventType.SESSION_SAVED, {
            'session_id': self.session_id,
            'action_type': action_type,
        })

    @property
    def session_id(self) -> Optional[str]:
        """Get current session ID"""
        return self.session_manager.session_id

    async def handle_message(self, user_message: str) -> str:
        """
        Main entry point for user interaction

        Parameters
        ----------
        user_message : str
            Message from user

        Returns
        -------
        str
            Response from copilot
        """
        # Try quick response first (no API call)
        if quick_response := self._try_quick_response(user_message):
            return quick_response

        # Complex query - use Claude
        return await self._call_claude(user_message)

    async def handle_message_stream(self, user_message: str):
        """
        Handle message with streaming response

        Yields chunks as they arrive from Claude API.
        Supports asend() for sending values back (e.g., user choice selections).

        Parameters
        ----------
        user_message : str
            Message from user

        Yields
        ------
        dict
            Chunks with 'type' and data:
            - {'type': 'text', 'text': '...'}
            - {'type': 'tool_call', 'tool_name': '...', 'tool_input': {...}, 'duration': 0.5}
            - {'type': 'choice_request', 'choice_data': {...}} - requires asend() with selection
        """
        # Try quick response first (no API call)
        if quick_response := self._try_quick_response(user_message):
            yield {'type': 'text', 'text': quick_response}
            return

        # Update system prompt with current state and context awareness
        context_summary = await self._get_cached_context_summary()
        self._update_system_prompt(context_summary)

        # Add user message to history
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })

        # Stream from Claude - manually propagate asend() values
        inner_gen = self._call_claude_stream()
        sent_value = None

        try:
            while True:
                if sent_value is None:
                    chunk = await inner_gen.__anext__()
                else:
                    chunk = await inner_gen.asend(sent_value)
                # Yield chunk and capture any sent value
                sent_value = yield chunk
        except StopAsyncIteration:
            return

    def _try_quick_response(self, message: str) -> Optional[str]:
        """
        Answer simple queries from state without LLM call

        Parameters
        ----------
        message : str
            User message

        Returns
        -------
        str or None
            Quick response if possible, None if Claude is needed
        """
        message_lower = message.lower()

        # Status query
        if "status" in message_lower and len(message.split()) < 5:
            return self.experiment.get_summary()

        # Simple commands
        if message_lower in ["stop", "pause", "halt"]:
            if self.run_engine:
                self.run_engine.halt()
            self.experiment.acquisition_status = "paused"
            return "Acquisition paused. What would you like to do next?"

        # No quick response available
        return None

    def _should_use_thinking(self, message: str) -> bool:
        """
        Determine if extended thinking should be enabled for this message.

        Auto-triggers for:
        - Explicit "think/thinking" in message
        - Calibration operations
        - Plan generation / timelapse setup
        - Image/volume analysis
        - Complex multi-step queries
        """
        import re
        msg_lower = message.lower()

        # Explicit thinking request
        if re.search(r'\bthink(ing)?\b', message, re.IGNORECASE):
            return True

        # Calibration operations
        if re.search(r'\bcalibrat', msg_lower):
            return True

        # Plan generation and timelapse
        if re.search(r'\b(plan|timelapse|time-lapse|acquisition)\b', msg_lower):
            return True

        # Image/volume analysis
        if re.search(r'\b(analy[sz]e|look at|check|inspect|review).*(image|volume|embryo)', msg_lower):
            return True

        # Complex queries - multiple embryos or steps
        if re.search(r'\b(all|every|each)\s+(embryo|sample)', msg_lower):
            return True
        if re.search(r'\b(first|then|after|next|finally)\b.*\b(first|then|after|next|finally)\b', msg_lower):
            return True

        # Troubleshooting / problem solving
        if re.search(r'\b(why|problem|issue|error|wrong|fail|debug|troubleshoot)', msg_lower):
            return True

        return False

    async def _call_claude(self, user_message: str) -> str:
        """
        Call Claude API with full context and tool access

        Parameters
        ----------
        user_message : str
            User message (append --think to enable extended thinking)

        Returns
        -------
        str
            Claude's response
        """
        import time
        import re
        start_time = time.time()

        # Auto-enable extended thinking for complex operations
        use_thinking = self._should_use_thinking(user_message)

        # Start interaction logging
        interaction = None
        if self.interaction_logger:
            interaction = self.interaction_logger.start_interaction(
                user_prompt=user_message,
                system_state={
                    'embryos': {eid: e.to_dict() for eid, e in self.experiment.embryos.items()},
                    'acquisition_status': self.experiment.acquisition_status,
                }
            )

        # Update system prompt with current state and context awareness
        context_summary = await self._get_cached_context_summary()
        self._update_system_prompt(context_summary)

        # Add user message to history
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })

        error_occurred = None

        try:
            # Build API call kwargs
            api_kwargs = {
                "model": self.model,
                "system": self._get_cached_system_prompt(),
                "messages": self.conversation_history,
                "tools": get_tool_registry().get_claude_schemas(has_microscope=self._has_microscope()),
                "max_tokens": 16000 if use_thinking else 4096,
            }
            # Enable extended thinking if --think flag was used
            if use_thinking:
                api_kwargs["thinking"] = {"type": "enabled", "budget_tokens": 10000}

            # Call Claude with tools (with retry for transient errors)
            # Use cached system prompt to reduce token costs
            response = await self._call_api_with_retry(
                self.claude.messages.create,
                **api_kwargs
            )
            self._track_token_usage(response)

            # Process tool calls
            while response.stop_reason == "tool_use":
                tool_results = await self._execute_tools_with_logging(
                    response.content, interaction
                )

                # Continue conversation with tool results
                self.conversation_history.append({
                    "role": "assistant",
                    "content": response.content
                })
                self.conversation_history.append({
                    "role": "user",
                    "content": tool_results
                })

                # Get next response (with retry for transient errors)
                # Reuse api_kwargs but update messages
                api_kwargs["messages"] = self.conversation_history
                response = await self._call_api_with_retry(
                    self.claude.messages.create,
                    **api_kwargs
                )
                self._track_token_usage(response)

            # Extract text response
            assistant_message = ""
            for block in response.content:
                if hasattr(block, 'text'):
                    assistant_message += block.text

            # Add to history
            self.conversation_history.append({
                "role": "assistant",
                "content": response.content
            })

        except Exception as e:
            import traceback
            error_occurred = str(e)
            error_tb = traceback.format_exc()
            assistant_message = f"Error: {error_occurred}"

            # Log error
            if interaction and self.interaction_logger:
                self.interaction_logger.complete_interaction(
                    interaction=interaction,
                    assistant_response=assistant_message,
                    total_duration_seconds=time.time() - start_time,
                    error=error_occurred,
                    error_traceback=error_tb,
                )
            raise

        # Complete interaction logging
        if interaction and self.interaction_logger:
            self.interaction_logger.complete_interaction(
                interaction=interaction,
                assistant_response=assistant_message,
                total_duration_seconds=time.time() - start_time,
            )

        # Auto-save after conversation turn
        self._auto_save()

        return assistant_message

    async def get_tool_call(self, user_message: str) -> Optional[Dict]:
        """
        Get what tool Claude would call without executing it (dry-run mode).

        Used for benchmarking tool selection accuracy. Makes a real API call
        but doesn't execute the selected tool.

        Parameters
        ----------
        user_message : str
            User query to analyze

        Returns
        -------
        dict or None
            Tool call info: {name, input, input_tokens, output_tokens, latency_ms}
            None if Claude doesn't call a tool
        """
        import time
        start_time = time.time()

        # Update system prompt with current state and context awareness
        context_summary = await self._get_cached_context_summary()
        self._update_system_prompt(context_summary)

        # Build messages (don't modify conversation history)
        messages = self.conversation_history.copy()
        messages.append({
            "role": "user",
            "content": user_message
        })

        try:
            # Build API call kwargs
            api_kwargs = {
                "model": self.model,
                "system": self._get_cached_system_prompt(),
                "messages": messages,
                "tools": get_tool_registry().get_claude_schemas(has_microscope=self._has_microscope()),
                "max_tokens": 4096,
            }

            # Call Claude
            response = await self._call_api_with_retry(
                self.claude.messages.create,
                **api_kwargs
            )

            latency_ms = (time.time() - start_time) * 1000

            # Extract token usage
            input_tokens = getattr(response.usage, 'input_tokens', 0)
            output_tokens = getattr(response.usage, 'output_tokens', 0)

            # Find tool use block
            for block in response.content:
                if block.type == "tool_use":
                    return {
                        "name": block.name,
                        "input": block.input,
                        "input_tokens": input_tokens,
                        "output_tokens": output_tokens,
                        "latency_ms": latency_ms,
                    }

            # No tool called
            return None

        except Exception as e:
            logger.error(f"Error in get_tool_call: {e}")
            raise

    async def _execute_tools_with_logging(self, content_blocks, interaction) -> List[Dict]:
        """
        Execute Claude's tool calls with interaction logging

        Parameters
        ----------
        content_blocks : list
            Content blocks from Claude response (may include tool uses)
        interaction : InteractionRecord
            Current interaction being logged

        Returns
        -------
        list of dict
            Tool result content blocks
        """
        import time
        import json
        from .tools.interaction_tools import CHOICE_RESPONSE_TYPE

        results = []

        for block in content_blocks:
            if block.type == "tool_use":
                start_time = time.time()
                is_error = False
                error_message = None

                try:
                    result = await self._execute_single_tool(block.name, block.input)

                    # Check if result is a choice request that needs UI handling
                    if self.choice_handler and isinstance(result, str):
                        try:
                            choice_data = json.loads(result)
                            if isinstance(choice_data, dict) and choice_data.get("_type") == CHOICE_RESPONSE_TYPE:
                                # Call choice handler to get user selection
                                user_selection = await self.choice_handler(choice_data)
                                result = user_selection  # Replace with actual selection
                        except (json.JSONDecodeError, TypeError):
                            pass  # Not a choice request, use original result

                except Exception as e:
                    result = f"Error: {str(e)}"
                    is_error = True
                    error_message = str(e)

                duration = time.time() - start_time

                # Log tool call
                if interaction and self.interaction_logger:
                    self.interaction_logger.record_tool_call(
                        interaction=interaction,
                        tool_name=block.name,
                        tool_input=block.input,
                        result=result,
                        duration_seconds=duration,
                        is_error=is_error,
                        error_message=error_message,
                    )

                results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result,
                    "is_error": is_error,
                })

        return results

    async def _call_claude_stream(self):
        """
        Call Claude API with streaming enabled

        Yields
        ------
        dict
            Chunks as they arrive from Claude
        """
        import time
        from anthropic import APIStatusError

        # Stream events (run entire streaming in thread since SDK is sync)
        def stream_and_collect():
            events = []
            response_content = []
            final_message = None

            with self.claude.messages.stream(
                model=self.model,
                system=self._get_cached_system_prompt(),
                messages=self.conversation_history,
                tools=get_tool_registry().get_claude_schemas(has_microscope=self._has_microscope()),
                max_tokens=4096
            ) as stream:
                for event in stream:
                    events.append(event)
                final_message = stream.get_final_message()

            return events, final_message

        # Run streaming in thread with retry logic for transient errors
        max_retries = 3
        retry_delay = 1.0

        for attempt in range(max_retries):
            try:
                events, final_message = await asyncio.to_thread(stream_and_collect)
                # Track token usage from streaming response
                self._track_token_usage(final_message)
                break  # Success, exit retry loop
            except APIStatusError as e:
                error_type = getattr(e, 'body', {})
                if isinstance(error_type, dict):
                    error_type = error_type.get('error', {}).get('type', '')

                # Retry on overloaded or rate limit errors
                if error_type in ('overloaded_error', 'rate_limit_error') or 'overloaded' in str(e).lower():
                    if attempt < max_retries - 1:
                        wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                        logger.warning(f"API overloaded, retrying in {wait_time:.1f}s (attempt {attempt + 1}/{max_retries})")
                        yield {'type': 'text', 'text': f"\n*[API busy, retrying in {wait_time:.0f}s...]*\n"}
                        await asyncio.sleep(wait_time)
                        continue
                # Re-raise if not retryable or out of retries
                raise
        else:
            # All retries exhausted - should not reach here, but just in case
            raise RuntimeError("API overloaded after multiple retries")

        # Process events and yield text
        for event in events:
            if event.type == "content_block_delta":
                if hasattr(event.delta, 'text'):
                    yield {'type': 'text', 'text': event.delta.text}

        response_content = final_message.content

        # Process tool calls if any
        if final_message.stop_reason == "tool_use":
            # Execute tools and collect results
            tool_results = []
            for block in response_content:
                if hasattr(block, 'type') and block.type == "tool_use":
                    start_time = time.time()

                    # Yield tool start notification BEFORE execution
                    yield {
                        'type': 'tool_start',
                        'tool_name': block.name,
                        'tool_input': block.input,
                    }

                    # Execute tool
                    try:
                        tool_result = await self._execute_single_tool(block.name, block.input)

                        # Check if result is a choice request that needs UI handling
                        if isinstance(tool_result, str):
                            try:
                                from .tools.interaction_tools import CHOICE_RESPONSE_TYPE
                                choice_data = json.loads(tool_result)
                                if isinstance(choice_data, dict) and choice_data.get("_type") == CHOICE_RESPONSE_TYPE:
                                    # Yield choice data for CLI to handle directly
                                    # CLI will run picker and send result back via asend()
                                    user_selection = yield {
                                        'type': 'choice_request',
                                        'choice_data': choice_data
                                    }
                                    tool_result = user_selection or "cancelled"
                            except (json.JSONDecodeError, TypeError):
                                pass  # Not a choice request, use original result

                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": tool_result
                        })
                    except Exception as e:
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": f"Error: {str(e)}",
                            "is_error": True
                        })

                    # Yield tool call info
                    yield {
                        'type': 'tool_call',
                        'tool_name': block.name,
                        'tool_input': block.input,
                        'duration': time.time() - start_time,
                    }

            self.conversation_history.append({
                "role": "assistant",
                "content": response_content
            })
            self.conversation_history.append({
                "role": "user",
                "content": tool_results
            })

            # Auto-save after tool execution
            self._auto_save()

            # Get next response (recursively stream)
            # IMPORTANT: Don't use `async for` here because it doesn't propagate asend() values.
            # We need to manually iterate and forward asend() values for choice_request handling.
            recursive_gen = self._call_claude_stream()
            sent_value = None
            try:
                while True:
                    if sent_value is None:
                        chunk = await recursive_gen.__anext__()
                    else:
                        chunk = await recursive_gen.asend(sent_value)
                        sent_value = None
                    # Yield chunk and capture any sent value from caller
                    sent_value = yield chunk
            except StopAsyncIteration:
                pass

        else:
            # No tool calls - add final message to history
            self.conversation_history.append({
                "role": "assistant",
                "content": response_content
            })

            # Auto-save after conversation turn
            self._auto_save()

    async def _execute_tools(self, content_blocks) -> List[Dict]:
        """
        Execute Claude's tool calls

        Parameters
        ----------
        content_blocks : list
            Content blocks from Claude response (may include tool uses)

        Returns
        -------
        list of dict
            Tool result content blocks
        """
        results = []

        for block in content_blocks:
            if block.type == "tool_use":
                try:
                    result = await self._execute_single_tool(block.name, block.input)
                    results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result
                    })
                except Exception as e:
                    results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": f"Error: {str(e)}",
                        "is_error": True
                    })

        return results

    async def _execute_single_tool(self, tool_name: str, tool_input: Dict) -> str:
        """Execute a single tool call using the tool registry"""
        registry = get_tool_registry()

        # Build execution context
        context = {
            'copilot': self,
            'client': getattr(self, 'client', None),
            'databroker': getattr(self, 'databroker', None),
        }

        # Execute via registry
        return await registry.execute(tool_name, tool_input, context)

    # === Experiment Management Methods ===

    def load_embryos_from_database(self, database: Dict):
        """
        Load embryos from calibration database

        Parameters
        ----------
        database : dict
            Embryo database with positions and calibrations
        """
        if 'embryos' not in database:
            return

        for embryo_id, embryo_data in database['embryos'].items():
            position = embryo_data.get('stage_position_after_centering_um', {})
            calibration = embryo_data.get('calibration', {})

            self.experiment.add_embryo(
                embryo_id=embryo_id,
                position=position,
                calibration=calibration
            )

        # Update system prompt with new embryos
        self._update_system_prompt()

    def import_embryos_from_session(self, session_id: str, clear_existing: bool = False) -> Dict:
        """
        Import embryos from another session into the current experiment.

        This allows starting a fresh session while preserving embryo positions
        and calibration data from a previous session.

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
        import json

        # Find session file
        session_file = self.session_manager.sessions_dir / f"{session_id}.json"
        if not session_file.exists():
            return {
                'success': False,
                'error': f"Session not found: {session_id}",
                'imported': [],
                'skipped': [],
            }

        try:
            with open(session_file, 'r') as f:
                session_data = json.load(f)
        except Exception as e:
            return {
                'success': False,
                'error': f"Failed to read session: {e}",
                'imported': [],
                'skipped': [],
            }

        # Get embryo states from session
        embryo_states = session_data.get('embryo_states', {})
        if not embryo_states:
            return {
                'success': False,
                'error': "No embryos found in session",
                'imported': [],
                'skipped': [],
            }

        # Optionally clear existing embryos
        if clear_existing:
            self.experiment.embryos.clear()

        imported = []
        skipped = []
        errors = []

        for embryo_id, embryo_data in embryo_states.items():
            # Skip if already exists and not clearing
            if embryo_id in self.experiment.embryos and not clear_existing:
                skipped.append(embryo_id)
                continue

            try:
                # Extract position and calibration
                position = embryo_data.get('stage_position', {})
                calibration = embryo_data.get('calibration', {})

                # Add embryo
                self.experiment.add_embryo(
                    embryo_id=embryo_id,
                    position=position,
                    calibration=calibration,
                    user_label=embryo_data.get('user_label'),
                )

                # Restore additional state
                embryo = self.experiment.embryos[embryo_id]
                embryo.nickname = embryo_data.get('nickname')
                embryo.interval_seconds = embryo_data.get('interval_seconds')  # None = use timelapse default
                embryo.num_slices = embryo_data.get('num_slices', 50)
                embryo.exposure_ms = embryo_data.get('exposure_ms', 10.0)
                embryo.priority = embryo_data.get('priority', 'normal')
                embryo.acquisition_mode = embryo_data.get('acquisition_mode', 'volume')

                # Preserve exposure tracking across sessions (cumulative for phototoxicity)
                embryo.exposure_count = embryo_data.get('exposure_count', 0)
                embryo.total_exposure_ms = embryo_data.get('total_exposure_ms', 0.0)
                embryo.timepoints_acquired = embryo_data.get('timepoints_acquired', 0)

                # Restore last_imaged if available
                last_imaged_str = embryo_data.get('last_imaged')
                if last_imaged_str:
                    try:
                        embryo.last_imaged = datetime.fromisoformat(last_imaged_str)
                    except (ValueError, TypeError):
                        embryo.last_imaged = None
                else:
                    # Handle legacy data: if exposure recorded but no last_imaged, clear inconsistent data
                    if embryo.exposure_count > 0:
                        # Legacy session without proper tracking - reset to avoid confusion
                        embryo.exposure_count = 0
                        embryo.total_exposure_ms = 0.0

                # Note: Detection results are NOT imported - this is a fresh start

                imported.append(embryo_id)

            except Exception as e:
                errors.append(f"{embryo_id}: {str(e)}")

        # Update system prompt with new embryos
        self._update_system_prompt()

        # Mark as significant action to trigger auto-save
        self._mark_significant_action("embryo_import")

        return {
            'success': len(imported) > 0,
            'imported': imported,
            'skipped': skipped,
            'errors': errors,
            'source_session': session_id,
        }

    async def on_volume_acquired(self, embryo_id: str, timepoint: int, volume_data):
        """
        Callback when a volume is acquired

        Parameters
        ----------
        embryo_id : str
            Embryo ID
        timepoint : int
            Timepoint number
        volume_data : np.ndarray or device
            Volume data (either numpy array or device to read from)
        """
        embryo = self.experiment.embryos.get(embryo_id)
        if not embryo:
            return

        # Get volume as numpy array
        if hasattr(volume_data, 'read_volume'):
            volume = volume_data.read_volume()
        else:
            volume = volume_data

        # Store image
        record = self.image_manager.store_volume(embryo, timepoint, volume)

        # Push to viz server
        if self.viz_server and volume is not None:
            try:
                max_proj = np.max(volume, axis=0)
                self.push_viz(
                    array=max_proj,
                    uid=f"volume_{embryo_id}_t{timepoint:04d}",
                    data_type="volume_projection",
                    metadata={
                        'embryo_id': embryo_id,
                        'timepoint': timepoint,
                        'shape': list(volume.shape),
                    }
                )
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(f"Failed to push to viz: {e}")

        # Emit volume acquired event
        self._emit_event(EventType.VOLUME_ACQUIRED, {
            'embryo_id': embryo_id,
            'timepoint': timepoint,
            'volume_uid': record.volume_uid,
            'projection_uid': record.projection_uid,
            'shape': list(volume.shape),
        })

        # Run detectors on newly acquired volume
        await self.run_detectors_on_volume(embryo_id, timepoint)

    def should_stop_experiment(self) -> bool:
        """Check if experiment should stop (e.g., all embryos hatched)"""
        if not self.experiment.embryos:
            return False

        # Stop if all embryos are skipped
        all_skipped = all(e.should_skip for e in self.experiment.embryos.values())
        return all_skipped

    def get_embryo_acquisition_order(self) -> List[str]:
        """
        Get embryo acquisition order based on priority

        Returns
        -------
        list of str
            Embryo IDs in acquisition order (high priority first)
        """
        # Group by priority
        high = [e.id for e in self.experiment.embryos.values() if e.priority == "high" and not e.should_skip]
        normal = [e.id for e in self.experiment.embryos.values() if e.priority == "normal" and not e.should_skip]
        low = [e.id for e in self.experiment.embryos.values() if e.priority == "low" and not e.should_skip]

        return high + normal + low

    def decide_parameters(self, embryo_id: str, timepoint: int) -> Dict:
        """
        Get current acquisition parameters for embryo

        This is called by the Bluesky plan to get parameters.
        In the future, this could implement more sophisticated
        adaptive logic.

        Parameters
        ----------
        embryo_id : str
            Embryo ID
        timepoint : int
            Current timepoint

        Returns
        -------
        dict
            Parameters: num_slices, exposure_ms, etc.
        """
        embryo = self.experiment.embryos.get(embryo_id)
        if not embryo:
            # Default parameters
            return {
                'num_slices': 50,
                'exposure_ms': 10.0
            }

        return {
            'num_slices': embryo.num_slices,
            'exposure_ms': embryo.exposure_ms
        }

    def decide_next_interval(self, timepoint: int) -> float:
        """
        Decide interval until next timepoint

        Can be adaptive based on experiment state.

        Parameters
        ----------
        timepoint : int
            Current timepoint

        Returns
        -------
        float
            Interval in seconds
        """
        # For now, use minimum interval of active embryos
        active_embryos = [e for e in self.experiment.embryos.values() if not e.should_skip]

        if not active_embryos:
            return 120.0  # Default 2 minutes

        return min(e.interval_seconds for e in active_embryos)

    # === Detector System Integration ===

    async def _on_detection_fired(self, detector, embryo_id: str, result):
        """
        Callback when a detector fires (detected=True with sufficient confidence)

        Parameters
        ----------
        detector : Detector
            Detector that fired
        embryo_id : str
            Embryo ID
        result : DetectionResult
            Detection result
        """
        from .detector import DetectionMode

        # Get embryo
        embryo = self.experiment.embryos.get(embryo_id)
        if not embryo:
            return

        # Emit detection triggered event
        self._emit_event(EventType.DETECTION_TRIGGERED, {
            'detector_name': detector.name,
            'embryo_id': embryo_id,
            'detected': result.detected,
            'confidence': result.confidence.value if result.confidence else None,
            'timepoint': result.timepoint,
        })

        # Emit specific event for hatching detection
        if detector.name == 'hatching' and result.detected:
            self._emit_event(EventType.HATCHING_DETECTED, {
                'embryo_id': embryo_id,
                'timepoint': result.timepoint,
                'confidence': result.confidence.value if result.confidence else None,
            })

        # Notify timelapse orchestrator of detection result
        if self.timelapse_orchestrator:
            self.timelapse_orchestrator.on_detection_result(
                embryo_id=embryo_id,
                detector_name=detector.name,
                result={
                    'detected': result.detected,
                    'confidence': result.confidence.value if result.confidence else None,
                    'timepoint': result.timepoint,
                }
            )

        # Handle based on action mode
        if detector.actions.mode == DetectionMode.PASSIVE:
            # Just log, no action
            pass

        elif detector.actions.mode == DetectionMode.RECOMMEND:
            # Generate recommendation message
            message = detector.actions.get_recommendation_message(detector.name, embryo_id)

            # Add to conversation as a system message (or could trigger WebSocket notification)
            print(f"\n[DETECTOR FIRED] {message}\n")

            # Could also add to a pending recommendations queue
            # that gets presented to user next time they interact

        elif detector.actions.mode == DetectionMode.AUTO:
            # For critical actions (stop_timelapse), verify detection first
            if detector.actions.stop_timelapse:
                # Run challenger/verifier agent
                logger.info(f"[VERIFIER] Running verification for {detector.name} detection on {embryo_id}")

                try:
                    verification = await self.detection_verifier.verify(
                        detector=detector,
                        embryo_state=embryo,
                        original_result=result,
                        timepoint=result.timepoint or 0,
                    )

                    # Log verification result
                    self._emit_event(EventType.DETECTION_TRIGGERED, {
                        'detector_name': detector.name,
                        'embryo_id': embryo_id,
                        'action': 'verification_completed',
                        'verification': verification.to_dict(),
                    })

                    if not verification.consensus:
                        # Verification failed - downgrade to RECOMMEND mode
                        logger.warning(
                            f"[VERIFIER] Verification failed for {detector.name}: {verification.consensus_reasoning}"
                        )
                        # Show user the disagreement instead of auto-acting
                        print(f"\n[VERIFICATION FAILED] {detector.name} detection on {embryo_id}")
                        print(f"  Consensus: {verification.consensus_reasoning}")
                        print(f"  Please manually verify and use /stop_timelapse if confirmed.\n")

                        self._emit_event(EventType.DETECTION_TRIGGERED, {
                            'detector_name': detector.name,
                            'embryo_id': embryo_id,
                            'action': 'verification_failed',
                            'reason': verification.consensus_reasoning,
                        })
                        return  # Don't proceed with auto-action

                    # Verification passed - proceed with stop
                    logger.info(f"[VERIFIER] Verification passed for {detector.name}, proceeding with stop")

                except Exception as e:
                    logger.error(f"[VERIFIER] Verification error: {e}")
                    # On error, fall back to RECOMMEND mode rather than auto-stop
                    print(f"\n[VERIFICATION ERROR] Could not verify {detector.name} detection: {e}")
                    print(f"  Please manually verify and use /stop_timelapse if confirmed.\n")
                    return

                # Verified - stop timelapse
                if self.timelapse_orchestrator:
                    logger.info(f"[AUTO-ACTION] Stopping timelapse for {embryo_id} due to verified {detector.name} detection")
                    await self.timelapse_orchestrator.stop_embryo(embryo_id, reason=f"{detector.name} detected (verified)")
                    self._emit_event(EventType.DETECTION_TRIGGERED, {
                        'detector_name': detector.name,
                        'embryo_id': embryo_id,
                        'action': 'stopped_timelapse',
                        'verified': True,
                    })

            # Apply parameter changes (non-critical, no verification needed)
            if detector.actions.parameter_changes:
                for param, value in detector.actions.parameter_changes.items():
                    if hasattr(embryo, param):
                        setattr(embryo, param, value)
                logger.info(f"[AUTO-ACTION] Applied changes to {embryo_id}: {detector.actions.parameter_changes}")

    async def run_detectors_on_volume(self, embryo_id: str, timepoint: int):
        """
        Run all enabled detectors on a newly acquired volume

        This should be called after each volume is acquired and stored.

        Parameters
        ----------
        embryo_id : str
            Embryo ID
        timepoint : int
            Timepoint number
        """
        embryo = self.experiment.embryos.get(embryo_id)
        if not embryo:
            return

        # Run detection queue
        results = await self.detection_queue.run_detectors(embryo, timepoint)

        return results

