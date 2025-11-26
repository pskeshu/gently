"""
Main Microscopy Copilot implementation

Now integrated with:
- Event Bus for async message passing between components
- Session Manager for persistence and resume
- Data Store for UID-based data flow
"""

import asyncio
import json
import os
from typing import Dict, List, Optional, Callable
from datetime import datetime
from pathlib import Path

import anthropic

from .state import ExperimentState, EmbryoState, ImageRecord
from .image_manager import ImageManager
from .plan_synthesis import PlanSynthesizer, PlanLibrary, PlanValidator
from .prompts import build_system_prompt, build_context_message
from .tool_registry import get_tool_registry
from .detector_registry import DetectorRegistry
from .detection_queue import DetectionQueue
from .interaction_logger import InteractionLogger
from .timelapse_orchestrator import TimelapseOrchestrator
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
        model: str = "claude-sonnet-4-5-20250929",
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
        # API client
        self.claude = anthropic.Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))
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

        # Session management
        self.session_manager = SessionManager(
            sessions_dir=self.storage_path / "sessions",
            auto_save=True
        )

        # Hardware interface via RPC client
        self.client = microscope_client

        # Databroker (optional, for data catalog integration)
        self.databroker = None

        # Callbacks
        self.on_message_callback: Optional[Callable] = None

        # Event bus for async messaging
        self._event_bus = get_event_bus()

        # Interaction logger for structured logging (research data collection)
        self.interaction_logger: Optional[InteractionLogger] = None

        # Timelapse orchestrator (initialized when microscope connected)
        self.timelapse_orchestrator: Optional[TimelapseOrchestrator] = None

        # Initialize or resume session
        if session_id:
            self.resume_session(session_id)
        else:
            self.session_manager.create_session()
            self._emit_event(EventType.SESSION_STARTED, {
                'session_id': self.session_id,
            })

        # Initialize interaction logger (for research data collection)
        self._init_interaction_logger()

        # Initialize timelapse orchestrator (if microscope connected)
        self._init_timelapse_orchestrator()

        # Build initial system prompt
        self._update_system_prompt()

    def _update_system_prompt(self):
        """Rebuild system prompt with current experiment state"""
        self.system_prompt = build_system_prompt(self.experiment)

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
            embryo.interval_seconds = embryo_data.get('interval_seconds', 120)
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
            - {'type': 'tool_result', 'result': '...'}
        """
        # Try quick response first (no API call)
        if quick_response := self._try_quick_response(user_message):
            yield {'type': 'text', 'text': quick_response}
            return

        # Update system prompt with current state
        self._update_system_prompt()

        # Add user message to history
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })

        # Stream from Claude
        async for chunk in self._call_claude_stream():
            yield chunk

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

    async def _call_claude(self, user_message: str) -> str:
        """
        Call Claude API with full context and tool access

        Parameters
        ----------
        user_message : str
            User message

        Returns
        -------
        str
            Claude's response
        """
        import time
        start_time = time.time()

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

        # Update system prompt with current state
        self._update_system_prompt()

        # Add user message to history
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })

        error_occurred = None

        try:
            # Call Claude with tools
            response = await asyncio.to_thread(
                self.claude.messages.create,
                model=self.model,
                system=self.system_prompt,
                messages=self.conversation_history,
                tools=get_tool_registry().get_claude_schemas(has_microscope=self._has_microscope()),
                max_tokens=4096
            )

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

                # Get next response
                response = await asyncio.to_thread(
                    self.claude.messages.create,
                    model=self.model,
                    system=self.system_prompt,
                    messages=self.conversation_history,
                    tools=get_tool_registry().get_claude_schemas(has_microscope=self._has_microscope()),
                    max_tokens=4096
                )

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
        results = []

        for block in content_blocks:
            if block.type == "tool_use":
                start_time = time.time()
                is_error = False
                error_message = None

                try:
                    result = await self._execute_single_tool(block.name, block.input)
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

        # Stream events (run entire streaming in thread since SDK is sync)
        def stream_and_collect():
            events = []
            response_content = []
            final_message = None

            with self.claude.messages.stream(
                model=self.model,
                system=self.system_prompt,
                messages=self.conversation_history,
                tools=get_tool_registry().get_claude_schemas(has_microscope=self._has_microscope()),
                max_tokens=4096
            ) as stream:
                for event in stream:
                    events.append(event)
                final_message = stream.get_final_message()

            return events, final_message

        # Run streaming in thread
        events, final_message = await asyncio.to_thread(stream_and_collect)

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

                    # Execute tool
                    try:
                        tool_result = await self._execute_single_tool(block.name, block.input)
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
            async for chunk in self._call_claude_stream():
                yield chunk

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
            # Automatically apply parameter changes
            if detector.actions.parameter_changes:
                for param, value in detector.actions.parameter_changes.items():
                    if hasattr(embryo, param):
                        setattr(embryo, param, value)

                print(f"\n[AUTO-ACTION] Applied changes to {embryo_id}: {detector.actions.parameter_changes}\n")

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

