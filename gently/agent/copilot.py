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

        # Callbacks
        self.on_message_callback: Optional[Callable] = None

        # Event bus for async messaging
        self._event_bus = get_event_bus()

        # Initialize or resume session
        if session_id:
            self.resume_session(session_id)
        else:
            self.session_manager.create_session()
            self._emit_event(EventType.SESSION_STARTED, {
                'session_id': self.session_id,
            })

        # Build initial system prompt
        self._update_system_prompt()

    def _update_system_prompt(self):
        """Rebuild system prompt with current experiment state"""
        self.system_prompt = build_system_prompt(self.experiment)

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
        # Update system prompt with current state
        self._update_system_prompt()

        # Add user message to history
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })

        # Call Claude with tools
        response = await asyncio.to_thread(
            self.claude.messages.create,
            model=self.model,
            system=self.system_prompt,
            messages=self.conversation_history,
            tools=get_tool_registry().get_claude_schemas(),
            max_tokens=4096
        )

        # Process tool calls
        while response.stop_reason == "tool_use":
            tool_results = await self._execute_tools(response.content)

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
                tools=get_tool_registry().get_claude_schemas(),
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

        return assistant_message

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
                tools=get_tool_registry().get_claude_schemas(),
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

            # Get next response (recursively stream)
            async for chunk in self._call_claude_stream():
                yield chunk

        else:
            # No tool calls - add final message to history
            self.conversation_history.append({
                "role": "assistant",
                "content": response_content
            })

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

    async def _tool_generate_plan(self, tool_input: Dict) -> str:
        """Generate Bluesky plan"""
        goal = tool_input['goal']
        embryo_ids = tool_input['embryo_ids']
        plan_type = tool_input.get('plan_type', 'adaptive_timelapse')
        params = tool_input.get('parameters', {})

        # Set intelligent defaults
        if 'interval_seconds' not in params:
            params['interval_seconds'] = 120  # 2 minutes default
        if 'num_timepoints' not in params:
            params['num_timepoints'] = 500  # ~16 hours at 2min intervals
        if 'num_slices' not in params:
            params['num_slices'] = 50
        if 'exposure_ms' not in params:
            params['exposure_ms'] = 10.0

        # Generate plan
        plan_code = self.plan_synthesizer.synthesize(
            goal=goal,
            embryo_ids=embryo_ids,
            params=params,
            plan_type=plan_type
        )

        # Store plan in experiment
        self.experiment.current_plan_name = goal
        self.experiment.plan_history.append({
            'goal': goal,
            'embryo_ids': embryo_ids,
            'parameters': params,
            'plan_type': plan_type,
            'generated_at': datetime.now().isoformat(),
            'code': plan_code
        })

        return f"Generated {plan_type} plan:\n\nGoal: {goal}\nEmbryos: {embryo_ids}\nParameters: {json.dumps(params, indent=2)}\n\nPlan validated and ready to execute."

    def _tool_query_embryo(self, tool_input: Dict) -> str:
        """Query embryo status"""
        embryo_id = tool_input['embryo_id']

        # Try to find embryo (handles "embryo 3" -> "embryo_003" conversion)
        embryo = self.experiment.get_embryo_by_any_name(embryo_id)

        if not embryo:
            return f"Embryo '{embryo_id}' not found. Available embryos: {list(self.experiment.embryos.keys())}"

        return json.dumps(embryo.to_dict(), indent=2)

    async def _tool_analyze_volume(self, tool_input: Dict) -> str:
        """Analyze volume with Claude Vision"""
        embryo_id = tool_input['embryo_id']
        analysis_prompt = tool_input['analysis_prompt']
        use_context = tool_input.get('use_recent_context', False)
        timepoint = tool_input.get('timepoint', None)

        embryo = self.experiment.get_embryo_by_any_name(embryo_id)
        if not embryo:
            return f"Embryo '{embryo_id}' not found"

        # Get image
        if timepoint is not None:
            # Find specific timepoint
            image_record = next(
                (img for img in embryo.recent_images if img.timepoint == timepoint),
                None
            )
            if not image_record:
                return f"No image found for timepoint {timepoint}"
        else:
            # Use latest
            image_record = self.image_manager.get_latest_image(embryo)
            if not image_record:
                return f"No images available for {embryo_id}"

        # Build message content
        content = []

        if use_context and len(embryo.recent_images) > 1:
            content.append({
                "type": "text",
                "text": f"Here are the last {len(embryo.recent_images)} timepoints for {embryo_id} (for temporal context):"
            })
            content.extend(self.image_manager.get_recent_context(embryo, num_images=5))

        # Add latest image
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": image_record.max_projection_b64
            }
        })

        # Add question
        content.append({
            "type": "text",
            "text": analysis_prompt
        })

        # Call Claude Vision
        response = await asyncio.to_thread(
            self.claude.messages.create,
            model=self.model,
            max_tokens=1024,
            messages=[{"role": "user", "content": content}]
        )

        analysis_result = response.content[0].text

        # Cache result
        embryo.custom_classifications[analysis_prompt] = {
            "result": analysis_result,
            "timepoint": image_record.timepoint,
            "timestamp": datetime.now().isoformat()
        }

        return analysis_result

    def _tool_modify_parameters(self, tool_input: Dict) -> str:
        """Modify embryo acquisition parameters"""
        embryo_id = tool_input['embryo_id']
        changes = tool_input['changes']
        reason = tool_input['reason']

        embryo = self.experiment.get_embryo_by_any_name(embryo_id)
        if not embryo:
            return f"Embryo '{embryo_id}' not found"

        # Apply changes
        old_params = {
            'interval_seconds': embryo.interval_seconds,
            'num_slices': embryo.num_slices,
            'exposure_ms': embryo.exposure_ms,
            'priority': embryo.priority
        }

        if 'interval_seconds' in changes:
            embryo.interval_seconds = changes['interval_seconds']
        if 'num_slices' in changes:
            embryo.num_slices = changes['num_slices']
        if 'exposure_ms' in changes:
            embryo.exposure_ms = changes['exposure_ms']
        if 'priority' in changes:
            embryo.priority = changes['priority']

        return f"Modified {embryo_id} parameters:\nReason: {reason}\n\nChanges:\n{json.dumps(changes, indent=2)}\n\nPrevious: {json.dumps(old_params, indent=2)}"

    def _tool_experiment_summary(self) -> str:
        """Get full experiment summary"""
        return self.experiment.get_summary()

    def _tool_skip_embryo(self, tool_input: Dict) -> str:
        """Skip embryo in future acquisitions"""
        embryo_id = tool_input['embryo_id']
        reason = tool_input['reason']

        embryo = self.experiment.get_embryo_by_any_name(embryo_id)
        if not embryo:
            return f"Embryo '{embryo_id}' not found"

        embryo.should_skip = True
        embryo.skip_reason = reason

        return f"Marked {embryo_id} to skip. Reason: {reason}"

    def _tool_resume_embryo(self, tool_input: Dict) -> str:
        """Resume skipped embryo"""
        embryo_id = tool_input['embryo_id']

        embryo = self.experiment.get_embryo_by_any_name(embryo_id)
        if not embryo:
            return f"Embryo '{embryo_id}' not found"

        embryo.should_skip = False
        embryo.skip_reason = None

        return f"Resumed imaging {embryo_id}"

    def _tool_assign_nickname(self, tool_input: Dict) -> str:
        """Assign nickname to embryo"""
        embryo_id = tool_input['embryo_id']
        nickname = tool_input['nickname']

        embryo = self.experiment.get_embryo_by_any_name(embryo_id)
        if not embryo:
            return f"Embryo '{embryo_id}' not found"

        old_nickname = embryo.nickname
        embryo.nickname = nickname

        if old_nickname:
            return f"Renamed {embryo_id}: '{old_nickname}' → '{nickname}'"
        else:
            return f"Nicknamed {embryo_id} as '{nickname}'"

    # === Detector Management Tool Methods ===

    def _tool_list_detectors(self, tool_input: Dict) -> str:
        """List all detectors"""
        filter_type = tool_input.get('filter', 'all')

        if filter_type == 'enabled':
            detectors = self.detector_registry.list_enabled()
        elif filter_type == 'disabled':
            all_detectors = self.detector_registry.list_all()
            detectors = [d for d in all_detectors if not d.enabled]
        else:
            detectors = self.detector_registry.list_all()

        if not detectors:
            return f"No {filter_type} detectors found."

        # Format detector list
        lines = [f"Detectors ({len(detectors)} {filter_type}):"]
        lines.append("")

        for detector in detectors:
            status = "✓ enabled" if detector.enabled else "✗ disabled"
            mode = detector.actions.mode.value
            lines.append(f"• {detector.name}: {status}")
            lines.append(f"  Description: {detector.description}")
            lines.append(f"  Action mode: {mode}")
            lines.append(f"  Runs: {detector.run_count}, Detections: {detector.detection_count}")

            if detector.conditions.min_timepoint:
                lines.append(f"  Min timepoint: {detector.conditions.min_timepoint}")

            if detector.actions.parameter_changes:
                lines.append(f"  Parameter changes: {detector.actions.parameter_changes}")

            lines.append("")

        return "\n".join(lines)

    async def _tool_add_detector(self, tool_input: Dict) -> str:
        """Add new detector"""
        from .detector import Detector, DetectorConditions, DetectorActions, DetectionMode, ConfidenceLevel

        name = tool_input['name']

        # Check if already exists
        if self.detector_registry.get(name):
            return f"Detector '{name}' already exists. Use a different name or remove the existing one first."

        # Create from preset if specified
        if 'preset' in tool_input:
            detector = self.detector_registry.create_preset_detector(tool_input['preset'])
            if not detector:
                return f"Unknown preset: {tool_input['preset']}"

            # Override name if custom name provided
            detector.name = name

        else:
            # Create custom detector
            description = tool_input.get('description', f"Custom detector: {name}")
            detection_prompt = tool_input.get('detection_prompt')

            if not detection_prompt:
                return "Error: Either 'preset' or 'detection_prompt' must be provided"

            detector = Detector(
                name=name,
                description=description,
                detection_prompt=detection_prompt,
                use_temporal_context=True,
                temporal_context_size=5,
                confidence_threshold=ConfidenceLevel.MEDIUM
            )

        # Apply conditions
        if 'min_timepoint' in tool_input:
            detector.conditions.min_timepoint = tool_input['min_timepoint']

        # Apply action mode
        if 'action_mode' in tool_input:
            detector.actions.mode = DetectionMode(tool_input['action_mode'])

        if 'parameter_changes' in tool_input:
            detector.actions.parameter_changes = tool_input['parameter_changes']

        # Add to registry
        success = self.detector_registry.add(detector)

        if success:
            # Trigger session auto-save after adding detector
            self._mark_significant_action("detector_config")
            return f"Detector '{name}' added successfully!\n\nDetails:\n- Description: {detector.description}\n- Action mode: {detector.actions.mode.value}\n- Min timepoint: {detector.conditions.min_timepoint or 'none'}\n\nThe detector is now enabled and will run on all future volumes."
        else:
            return f"Failed to add detector '{name}'"

    async def _tool_generate_detector_prompt(self, tool_input: Dict) -> str:
        """Generate detection prompt using Claude"""
        detector_description = tool_input['detector_description']
        context = tool_input.get('context', '')

        # Use Claude to generate an optimal detection prompt
        prompt_generation_request = f"""I need to create a detection prompt for a C. elegans embryo imaging system that uses Claude Vision API.

The detector should detect: {detector_description}

{f"Additional context: {context}" if context else ""}

Please generate an optimal detection prompt that:
1. Clearly describes what to look for in the image(s)
2. Includes key visual characteristics specific to C. elegans embryos
3. Accounts for temporal context if relevant (multiple timepoints shown)
4. Instructs Claude to focus on the LATEST/CURRENT image
5. Requests a structured response in this format:
   DETECTED: [YES/NO]
   CONFIDENCE: [HIGH/MEDIUM/LOW]
   REASONING: [brief explanation]

The prompt should be clear, specific, and leverage your knowledge of C. elegans developmental biology.

Generate the detection prompt now:"""

        # Call Claude
        response = await asyncio.to_thread(
            self.claude.messages.create,
            model=self.model,
            max_tokens=1024,
            messages=[{"role": "user", "content": prompt_generation_request}]
        )

        generated_prompt = response.content[0].text

        return f"Generated detection prompt for '{detector_description}':\n\n{generated_prompt}\n\nYou can now use this prompt to create a detector with the add_detector tool."

    async def _tool_test_detector(self, tool_input: Dict) -> str:
        """Test detector on embryo"""
        detector_name = tool_input['detector_name']
        embryo_id = tool_input['embryo_id']
        timepoint = tool_input.get('timepoint')

        embryo = self.experiment.get_embryo_by_any_name(embryo_id)
        if not embryo:
            return f"Embryo '{embryo_id}' not found"

        detector = self.detector_registry.get(detector_name)
        if not detector:
            return f"Detector '{detector_name}' not found"

        # Test detector
        result = await self.detection_queue.test_detector(detector_name, embryo, timepoint)

        if not result:
            return f"Could not test detector (no images available for {embryo_id})"

        # Format result
        status = "✓ DETECTED" if result.detected else "✗ NOT DETECTED"
        output = f"Test result for '{detector_name}' on {embryo_id}:\n\n"
        output += f"Status: {status}\n"
        output += f"Confidence: {result.confidence.value if result.confidence else 'N/A'}\n"
        output += f"Reasoning: {result.reasoning}\n"
        output += f"API duration: {result.api_duration:.2f}s\n"

        if result.error:
            output += f"\nError: {result.error_message}"

        return output

    def _tool_enable_disable_detector(self, tool_input: Dict) -> str:
        """Enable or disable detector"""
        detector_name = tool_input['detector_name']
        enabled = tool_input['enabled']

        if enabled:
            success = self.detector_registry.enable(detector_name)
            action = "enabled"
        else:
            success = self.detector_registry.disable(detector_name)
            action = "disabled"

        if success:
            return f"Detector '{detector_name}' {action}"
        else:
            return f"Detector '{detector_name}' not found"

    def _tool_remove_detector(self, tool_input: Dict) -> str:
        """Remove detector"""
        detector_name = tool_input['detector_name']

        success = self.detector_registry.remove(detector_name)

        if success:
            # Trigger session auto-save after removing detector
            self._mark_significant_action("detector_config")
            return f"Detector '{detector_name}' removed"
        else:
            return f"Detector '{detector_name}' not found"

    def _tool_get_detection_summary(self) -> str:
        """Get detection summary"""
        summary = self.detection_queue.get_detection_summary(self.experiment.embryos)

        # Format summary
        lines = ["Detection Summary:", ""]

        # Per-detector summary
        lines.append("=== By Detector ===")
        for detector_name, detector_info in summary['detectors'].items():
            lines.append(f"\n{detector_name}:")
            lines.append(f"  Enabled: {detector_info['enabled']}")
            lines.append(f"  Total runs: {detector_info['total_runs']}")
            lines.append(f"  Total detections: {detector_info['total_detections']}")

            if detector_info['embryos_detected']:
                lines.append(f"  Detected in embryos:")
                for embryo_det in detector_info['embryos_detected']:
                    lines.append(f"    - {embryo_det['embryo_id']} at t{embryo_det['timepoint']:04d} ({embryo_det['confidence']})")

        # Per-embryo summary
        lines.append("\n=== By Embryo ===")
        for embryo_id, embryo_info in summary['embryos'].items():
            lines.append(f"\n{embryo_id}:")
            for detector_name, det_info in embryo_info['detections'].items():
                if det_info['detected']:
                    lines.append(f"  ✓ {detector_name}: detected at t{det_info['timepoint']:04d} ({det_info['confidence']})")
                elif det_info['timepoint'] is not None:
                    lines.append(f"  ✗ {detector_name}: not yet detected (last check: t{det_info['timepoint']:04d})")

        return "\n".join(lines)

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

    # === Device Control Tool Executors ===

    async def _tool_calibrate_embryo(self, tool_input: Dict) -> str:
        """Run calibration for single embryo via microscope server"""
        if not self._has_microscope():
            return "Error: Not connected to microscope server. Start the server and reconnect."

        embryo_id = tool_input['embryo_id']
        piezo_positions = tool_input.get('piezo_positions', [40.0, 60.0])

        # Get embryo
        embryo = self.experiment.get_embryo_by_any_name(embryo_id)
        if not embryo:
            return f"Error: Embryo '{embryo_id}' not found"

        try:
            # Move to embryo position first if we have one
            if embryo.position:
                x, y = embryo.position.get('x', 0), embryo.position.get('y', 0)
                await self.client.move_to_position(x, y)

            # Run calibration via server
            results = await self.client.calibrate_piezo_galvo(piezo_positions)

            # Store calibration in embryo
            if results and 'calibration' in results:
                embryo.calibration = results['calibration']

            # Trigger session auto-save after calibration
            self._mark_significant_action("calibration")

            return f"✓ Calibration complete for {embryo_id}\n" \
                   f"Slope: {results['calibration']['slope']:.6e} deg/µm\n" \
                   f"Offset: {results['calibration']['offset']:.6f} deg\n" \
                   f"RMSE: {results['calibration']['rmse']:.6e} deg"

        except Exception as e:
            import traceback
            return f"Error during calibration: {str(e)}\n{traceback.format_exc()}"

    async def _tool_acquire_volume(self, tool_input: Dict) -> str:
        """Acquire single volume for embryo via microscope server"""
        if not self._has_microscope():
            return "Error: Not connected to microscope server. Start the server and reconnect."

        embryo_id = tool_input['embryo_id']
        num_slices = tool_input.get('num_slices', 50)
        exposure_ms = tool_input.get('exposure_ms', 10.0)
        save = tool_input.get('save', True)

        # Get embryo
        embryo = self.experiment.get_embryo_by_any_name(embryo_id)
        if not embryo:
            return f"Error: Embryo '{embryo_id}' not found"

        try:
            # Move to embryo position first if we have one
            if embryo.position:
                x, y = embryo.position.get('x', 0), embryo.position.get('y', 0)
                await self.client.move_to_position(x, y)

            # Get calibration parameters if available
            if embryo.calibration:
                galvo_center = embryo.calibration.get('offset', 0.0)
                galvo_amplitude = embryo.calibration.get('galvo_amplitude', 0.5)
                piezo_center = embryo.calibration.get('piezo_center', 50.0)
                piezo_amplitude = embryo.calibration.get('piezo_amplitude', 25.0)
            else:
                galvo_center = 0.0
                galvo_amplitude = 0.5
                piezo_center = 50.0
                piezo_amplitude = 25.0

            # Acquire volume via server
            results = await self.client.acquire_volume(
                num_slices=num_slices,
                exposure_ms=exposure_ms,
                galvo_amplitude=galvo_amplitude,
                galvo_center=galvo_center,
                piezo_amplitude=piezo_amplitude,
                piezo_center=piezo_center,
            )

            volume = results.get('volume')
            if volume is None:
                return f"Error: Failed to acquire volume"

            # Get current timepoint
            timepoint = embryo.last_acquisition_timepoint + 1

            # Store volume
            if save:
                self.image_manager.store_volume(embryo, timepoint, volume)
                embryo.last_acquisition_timepoint = timepoint
                embryo.last_acquisition_time = datetime.now()

                # Run detectors
                await self.run_detectors_on_volume(embryo_id, timepoint)

            # Trigger session auto-save after acquisition
            self._mark_significant_action("acquisition")

            return f"✓ Volume acquired for {embryo_id}\n" \
                   f"Shape: {volume.shape}\n" \
                   f"Slices: {num_slices}, Exposure: {exposure_ms} ms\n" \
                   f"Timepoint: {timepoint}" + \
                   (f"\nSaved to storage" if save else "")

        except Exception as e:
            import traceback
            return f"Error during acquisition: {str(e)}\n{traceback.format_exc()}"

    async def _tool_move_to_embryo(self, tool_input: Dict) -> str:
        """Move stage to embryo position via microscope server"""
        if not self._has_microscope():
            return "Error: Not connected to microscope server. Start the server and reconnect."

        embryo_id = tool_input['embryo_id']

        # Get embryo
        embryo = self.experiment.get_embryo_by_any_name(embryo_id)
        if not embryo:
            return f"Error: Embryo '{embryo_id}' not found"

        if not embryo.position:
            return f"Error: Embryo '{embryo_id}' has no stored position. Run calibration first."

        try:
            x, y = embryo.position.get('x', 0), embryo.position.get('y', 0)
            await self.client.move_to_position(x, y)

            return f"✓ Moved to {embryo_id}\n" \
                   f"Position: ({x:.2f}, {y:.2f}) µm"

        except Exception as e:
            import traceback
            return f"Error moving to embryo: {str(e)}\n{traceback.format_exc()}"

    async def _tool_start_multi_embryo_timelapse(self, tool_input: Dict) -> str:
        """Start multi-embryo time-lapse acquisition"""
        if not self._has_microscope():
            return "Error: Not connected to microscope server."

        # Get parameters
        embryo_ids = tool_input.get('embryo_ids')
        if not embryo_ids:
            # Use all active embryos
            embryo_ids = [e.id for e in self.experiment.embryos.values() if not e.should_skip]

        if not embryo_ids:
            return "Error: No embryos available for acquisition"

        num_timepoints = tool_input.get('num_timepoints', 500)
        interval_seconds = tool_input.get('interval_seconds', 120)
        num_slices = tool_input.get('num_slices', 50)
        exposure_ms = tool_input.get('exposure_ms', 10.0)
        enable_detectors = tool_input.get('enable_detectors', True)

        # Update experiment state
        self.experiment.acquisition_status = "running"
        self.experiment.start_time = datetime.now()

        return f"✓ Starting multi-embryo time-lapse acquisition\n" \
               f"Embryos: {', '.join(embryo_ids)}\n" \
               f"Timepoints: {num_timepoints}\n" \
               f"Interval: {interval_seconds} seconds\n" \
               f"Slices: {num_slices}, Exposure: {exposure_ms} ms\n" \
               f"Detectors: {'enabled' if enable_detectors else 'disabled'}\n\n" \
               f"Note: This will start acquisition in the background. " \
               f"The actual workflow needs to be implemented in workflow_manager.py (Phase 3)."

    async def _tool_pause_acquisition(self) -> str:
        """Pause acquisition"""
        if not self._has_microscope():
            return "Error: Not connected to microscope server."

        if self.experiment.acquisition_status != "running":
            return f"Cannot pause: acquisition is {self.experiment.acquisition_status}"

        try:
            await self.client.pause_acquisition()
            self.experiment.acquisition_status = "paused"
            return "✓ Acquisition paused. Use resume_acquisition to continue."
        except Exception as e:
            return f"Error pausing acquisition: {str(e)}"

    async def _tool_resume_acquisition(self) -> str:
        """Resume acquisition"""
        if not self._has_microscope():
            return "Error: Not connected to microscope server."

        if self.experiment.acquisition_status != "paused":
            return f"Cannot resume: acquisition is {self.experiment.acquisition_status}"

        try:
            await self.client.resume_acquisition()
            self.experiment.acquisition_status = "running"
            return "✓ Acquisition resumed"
        except Exception as e:
            return f"Error resuming acquisition: {str(e)}"

    async def _tool_detect_embryos(self, tool_input: Dict) -> str:
        """Detect embryos using SAM + Claude Vision via microscope server"""
        if not self._has_microscope():
            return "Error: Not connected to microscope server."

        auto_calibrate = tool_input.get('auto_calibrate', False)
        min_confidence = tool_input.get('min_confidence', 0.7)
        use_claude_review = tool_input.get('use_claude_review', False)
        exposure_ms = tool_input.get('exposure_ms', None)
        brightness_percentile = tool_input.get('brightness_percentile', 99.0)
        min_area = tool_input.get('min_area', 5000)
        max_area = tool_input.get('max_area', 150000)

        try:
            if exposure_ms:
                print(f"\nUsing {exposure_ms} ms exposure for better contrast...")
            print(f"\nBrightness detection: percentile={brightness_percentile}, area={min_area}-{max_area}")
            print(f"Claude review: {'enabled' if use_claude_review else 'disabled'}")
            print("\nDetecting embryos via brightness + SAM...")

            # Run detection on server (brightness + SAM + image capture all happen server-side)
            results = await self.client.detect_embryos(
                pixel_size_um=6.5,
                objective_mag=4.0,
                use_claude_review=use_claude_review,
                min_confidence=min_confidence,
                exposure_ms=exposure_ms,
                brightness_percentile=brightness_percentile,
                min_area=min_area,
                max_area=max_area
            )

            if 'error' in results:
                error_msg = results['error']
                traceback_info = results.get('traceback', '')
                # Return clear error without triggering AI interpretation
                return f"""Server error during embryo detection:
  {error_msg}

Technical details (if available):
{traceback_info if traceback_info else '  None'}

Troubleshooting:
  • Check the server terminal for detailed error logs
  • Ensure bottom_camera and xy_stage devices are available
  • Verify SAM model file exists: sam_vit_b_01ec64.pth"""

            # Load detected embryos into experiment
            embryos = results.get('embryos', [])
            for embryo in embryos:
                embryo_id = f"embryo_{embryo['embryo_id']:03d}"
                self.experiment.add_embryo(
                    embryo_id=embryo_id,
                    position={'x': embryo['stage_x_um'], 'y': embryo['stage_y_um']},
                    calibration={}  # Empty, will be filled during calibration
                )

            # Trigger session auto-save after detection
            self._mark_significant_action("detection")

            # Format response
            response = f"""✓ Detected {len(embryos)} embryos using SAM + Claude Vision

Detection Summary:
  Initial (SAM): {results.get('initial_detections', len(embryos))}
  Final (after Claude review): {results.get('final_detections', len(embryos))}

"""
            if results.get('verification'):
                response += f"Claude verification: {results['verification'].get('verification_summary', 'Verified')}\n\n"

            response += "Detected embryo positions:\n"

            for embryo in embryos:
                response += f"  {embryo['embryo_id']}: ({embryo['stage_x_um']:.1f}, {embryo['stage_y_um']:.1f}) µm"
                if 'pixel_x' in embryo:
                    response += f" [pixel: ({embryo['pixel_x']:.0f}, {embryo['pixel_y']:.0f})]"
                if 'confidence' in embryo:
                    response += f" conf: {embryo['confidence']:.2f}"
                response += "\n"

            response += f"""
✓ Loaded {len(embryos)} embryos into experiment

Next steps:
  • Say "looks good" to proceed
  • Say "remove detection X" to correct false positives
  • Say "calibrate all" to start calibration workflow"""

            if auto_calibrate:
                response += "\n\nAuto-calibration enabled - would start calibration here (not yet implemented)"

            return response

        except Exception as e:
            import traceback
            return f"Error detecting embryos: {str(e)}\n{traceback.format_exc()}"

    async def _tool_manual_mark_embryos(self, tool_input: Dict) -> str:
        """Manually mark embryos by clicking on image"""
        if not self._has_microscope():
            return "Error: Not connected to microscope server."

        exposure_ms = tool_input.get('exposure_ms', None)

        try:
            print("\nManual embryo marking...")
            if exposure_ms:
                print(f"  Using {exposure_ms} ms exposure for better contrast...")
            print("  A matplotlib window will open - click on embryo centers.")
            print("  Close the window when done.\n")

            # Run manual marking
            results = await self.client.manual_mark_embryos(
                pixel_size_um=6.5,
                objective_mag=4.0,
                exposure_ms=exposure_ms
            )

            if 'error' in results:
                return f"Error: {results['error']}"

            # Load marked embryos into experiment
            embryos = results.get('embryos', [])
            for embryo in embryos:
                embryo_id = f"embryo_{embryo['id']:03d}"
                self.experiment.add_embryo(
                    embryo_id=embryo_id,
                    position={'x': embryo['stage_x'], 'y': embryo['stage_y']},
                    calibration={}
                )

            # Trigger session auto-save after marking embryos
            self._mark_significant_action("embryo_change")

            # Format response
            response = f"""✓ Manually marked {len(embryos)} embryos

Marked positions:
"""
            for embryo in embryos:
                response += f"  embryo_{embryo['id']:03d}: ({embryo['stage_x']:.1f}, {embryo['stage_y']:.1f}) µm\n"

            response += f"""
✓ Loaded {len(embryos)} embryos into experiment

Next steps:
  • Say "calibrate all" to start calibration workflow
  • Say "show embryos" to see the list"""

            return response

        except Exception as e:
            import traceback
            return f"Error during manual marking: {str(e)}\n{traceback.format_exc()}"

    async def _tool_view_image(self, tool_input: Dict) -> str:
        """View bottom camera image"""
        if not self._has_microscope():
            return "Error: Not connected to microscope server."

        title = tool_input.get('title', 'Bottom Camera Image')
        exposure_ms = tool_input.get('exposure_ms', None)
        save_only = tool_input.get('save_only', False)

        try:
            if exposure_ms:
                print(f"\nUsing {exposure_ms} ms exposure for better contrast...")

            # Generate save path
            save_path = None
            if save_only:
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = f"bottom_camera_{timestamp}.png"
                print(f"\nSaving image to {save_path}...")
            else:
                print("\nCapturing and viewing image...")

            result = await self.client.view_image(
                title=title,
                exposure_ms=exposure_ms,
                save_path=save_path,
                show=not save_only
            )

            if 'error' in result:
                return f"Error: {result['error']}"

            if save_only and result.get('saved_to'):
                return f"✓ Image saved to: {result['saved_to']}"
            else:
                return "✓ Image displayed in matplotlib window"

        except Exception as e:
            return f"Error viewing image: {str(e)}"

    async def _tool_capture_lightsheet(self, tool_input: Dict) -> str:
        """Capture a single lightsheet image"""
        if not self._has_microscope():
            return "Error: Not connected to microscope server."

        piezo_position = tool_input.get('piezo_position', 50.0)
        galvo_position = tool_input.get('galvo_position', 0.0)
        save_only = tool_input.get('save_only', False)

        try:
            print(f"\nCapturing lightsheet image...")
            print(f"  Piezo: {piezo_position} µm, Galvo: {galvo_position} V")

            result = await self.client.capture_lightsheet_image(
                piezo_position=piezo_position,
                galvo_position=galvo_position
            )

            if 'error' in result:
                return f"Error: {result['error']}"

            image = result.get('image')
            if image is None:
                return "Error: No image returned"

            # Save image
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"lightsheet_{timestamp}.png"

            # Normalize and save
            import numpy as np
            from PIL import Image as PILImage
            img_norm = image.astype(np.float32)
            img_norm = (img_norm - img_norm.min()) / (img_norm.max() - img_norm.min()) * 255
            PILImage.fromarray(img_norm.astype(np.uint8)).save(save_path)
            print(f"  Saved to: {save_path}")

            if not save_only:
                # Display with matplotlib
                import matplotlib
                matplotlib.use('TkAgg')
                import matplotlib.pyplot as plt
                fig, ax = plt.subplots(figsize=(10, 10))
                ax.imshow(image, cmap='gray')
                ax.set_title(f"Lightsheet: piezo={piezo_position}µm, galvo={galvo_position}V")
                plt.show()

            return f"""✓ Lightsheet image captured
  Piezo: {piezo_position} µm
  Galvo: {galvo_position} V
  Image shape: {image.shape}
  Saved to: {save_path}"""

        except Exception as e:
            import traceback
            return f"Error capturing lightsheet: {str(e)}\n{traceback.format_exc()}"

    async def _tool_show_detected_embryos(self, tool_input: Dict) -> str:
        """Show detected embryos with bounding boxes"""
        if not self._has_microscope():
            return "Error: Not connected to microscope server."

        save_to_file = tool_input.get('save_to_file', False)
        save_only = tool_input.get('save_only', False)

        try:
            # Generate save path if requested
            save_path = None
            if save_to_file or save_only:
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = f"detected_embryos_{timestamp}.png"

            if save_only:
                print(f"\nSaving embryo visualization to {save_path}...")
            else:
                print("\nShowing detected embryos with bounding boxes...")

            result = await self.client.view_detected_embryos(
                save_path=save_path,
                show=not save_only
            )

            if 'error' in result:
                return f"Error: {result['error']}"

            num_embryos = result.get('num_embryos', '?')
            if save_only:
                return f"✓ Saved {num_embryos} embryos visualization to: {result.get('saved_to', save_path)}"
            else:
                response = f"✓ Displayed {num_embryos} embryos with bounding boxes"
                if save_path and result.get('saved_to'):
                    response += f"\nAlso saved to: {result['saved_to']}"
                return response

        except Exception as e:
            import traceback
            return f"Error showing embryos: {str(e)}\n{traceback.format_exc()}"

    async def _tool_set_led(self, tool_input: Dict) -> str:
        """Set LED state"""
        if not self._has_microscope():
            return "Error: Not connected to microscope server."

        state = tool_input.get('state', 'Closed')

        try:
            result = await self.client.set_led(state)

            if result.get('success'):
                return f"LED set to '{state}'"
            else:
                return f"Error setting LED: {result.get('error', 'Unknown error')}"

        except Exception as e:
            return f"Error setting LED: {str(e)}"

    async def _tool_get_led_status(self, tool_input: Dict) -> str:
        """Get LED status"""
        if not self._has_microscope():
            return "Error: Not connected to microscope server."

        try:
            result = await self.client.get_led_status()

            if result.get('success'):
                current = result.get('current_state', 'unknown')
                available = result.get('available_configs', [])
                group = result.get('group_name', 'unknown')

                return (f"LED Status:\n"
                        f"  Current state: {current}\n"
                        f"  ConfigGroup: {group}\n"
                        f"  Available configs: {available}")
            else:
                return f"Error getting LED status: {result.get('error', 'Unknown error')}"

        except Exception as e:
            return f"Error getting LED status: {str(e)}"

    # =========================================================================
    # Databroker Tools
    # =========================================================================

    def _tool_list_runs(self, tool_input: Dict) -> str:
        """List recent runs from Databroker"""
        if not self.db:
            return "Error: Databroker not configured"

        limit = tool_input.get('limit', 10)
        embryo_id = tool_input.get('embryo_id')
        plan_name = tool_input.get('plan_name')

        try:
            # Build search criteria
            search_kwargs = {}
            if embryo_id:
                search_kwargs['embryo_id'] = embryo_id
            if plan_name:
                search_kwargs['plan_name'] = plan_name

            # Query runs
            if search_kwargs:
                runs = list(self.db.search(search_kwargs))[:limit]
            else:
                runs = list(self.db[-limit:])

            if not runs:
                return "No runs found matching criteria"

            # Format output
            response = f"Found {len(runs)} runs:\n\n"
            for i, run in enumerate(runs):
                start = run.metadata['start']
                uid_short = start['uid'][:8]
                time_str = datetime.fromtimestamp(start['time']).strftime('%Y-%m-%d %H:%M:%S')
                plan = start.get('plan_name', 'unknown')

                response += f"{i+1}. [{uid_short}...] {time_str}\n"
                response += f"   Plan: {plan}\n"

                # Include relevant metadata
                for key in ['embryo_id', 'num_slices', 'exposure_ms']:
                    if key in start:
                        response += f"   {key}: {start[key]}\n"
                response += "\n"

            return response

        except Exception as e:
            return f"Error listing runs: {str(e)}"

    def _tool_get_run_data(self, tool_input: Dict) -> str:
        """Get data from a specific run"""
        if not self.db:
            return "Error: Databroker not configured"

        run_id = tool_input['run_id']
        data_keys = tool_input.get('data_keys')
        stream = tool_input.get('stream', 'primary')

        try:
            # Get run by UID or index
            if run_id.startswith('-') or run_id.isdigit():
                run = self.db[int(run_id)]
            else:
                run = self.db[run_id]

            # Get metadata
            start = run.metadata['start']
            uid_short = start['uid'][:8]
            time_str = datetime.fromtimestamp(start['time']).strftime('%Y-%m-%d %H:%M:%S')

            response = f"Run [{uid_short}...] from {time_str}\n"
            response += f"Plan: {start.get('plan_name', 'unknown')}\n\n"

            # Get available streams
            streams = list(run)
            response += f"Available streams: {streams}\n\n"

            # Read data from requested stream
            if stream in run:
                data = run[stream].read()
                available_keys = list(data.keys())
                response += f"Data keys in '{stream}': {available_keys}\n\n"

                # Filter to requested keys
                keys_to_show = data_keys if data_keys else available_keys

                for key in keys_to_show:
                    if key in data:
                        arr = data[key]
                        if hasattr(arr, 'shape'):
                            response += f"{key}: shape={arr.shape}, dtype={arr.dtype}\n"
                        elif hasattr(arr, 'values'):
                            vals = arr.values
                            if hasattr(vals, 'shape') and len(vals.shape) > 1:
                                response += f"{key}: shape={vals.shape}, dtype={vals.dtype}\n"
                            else:
                                response += f"{key}: {vals}\n"
                        else:
                            response += f"{key}: {arr}\n"
            else:
                response += f"Stream '{stream}' not found in run\n"

            return response

        except Exception as e:
            import traceback
            return f"Error getting run data: {str(e)}\n{traceback.format_exc()}"

    async def _tool_get_run_image(self, tool_input: Dict) -> str:
        """Get and optionally analyze an image from a run"""
        if not self.db:
            return "Error: Databroker not configured"

        run_id = tool_input['run_id']
        detector = tool_input.get('detector')
        analyze = tool_input.get('analyze', False)
        analysis_prompt = tool_input.get('analysis_prompt', '')

        try:
            # Get run
            if run_id.startswith('-') or run_id.isdigit():
                run = self.db[int(run_id)]
            else:
                run = self.db[run_id]

            # Get data
            data = run.primary.read()
            available_keys = list(data.keys())

            # Find detector (auto-detect if not specified)
            if detector is None:
                # Look for common detector names
                for candidate in ['bottom_camera', 'volume_scanner', 'camera', 'HamCam1']:
                    if candidate in available_keys:
                        detector = candidate
                        break
                if detector is None:
                    return f"No detector specified and couldn't auto-detect. Available keys: {available_keys}"

            if detector not in data:
                return f"Detector '{detector}' not found. Available: {available_keys}"

            # Get image
            image_data = data[detector].values
            if len(image_data.shape) > 2:
                image = image_data[0]  # First frame if multiple
            else:
                image = image_data

            # Build response
            start = run.metadata['start']
            uid_short = start['uid'][:8]
            response = f"Image from run [{uid_short}...]\n"
            response += f"Detector: {detector}\n"
            response += f"Shape: {image.shape}, dtype: {image.dtype}\n"
            response += f"Min: {image.min()}, Max: {image.max()}, Mean: {image.mean():.1f}\n"

            # Analyze with Claude Vision if requested
            if analyze and analysis_prompt:
                response += "\n--- Claude Vision Analysis ---\n"

                # Convert to base64
                import numpy as np
                from PIL import Image as PILImage
                import base64
                from io import BytesIO

                # Normalize to 8-bit
                if image.dtype == np.uint16:
                    img_8bit = ((image - image.min()) / (image.max() - image.min()) * 255).astype(np.uint8)
                else:
                    img_8bit = image.astype(np.uint8)

                pil_img = PILImage.fromarray(img_8bit)
                buffered = BytesIO()
                pil_img.save(buffered, format="PNG")
                img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

                # Call Claude Vision
                message = self.claude.messages.create(
                    model=self.model,
                    max_tokens=2000,
                    messages=[{
                        "role": "user",
                        "content": [
                            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": img_base64}},
                            {"type": "text", "text": analysis_prompt}
                        ]
                    }]
                )

                response += message.content[0].text

            return response

        except Exception as e:
            import traceback
            return f"Error getting run image: {str(e)}\n{traceback.format_exc()}"

    def _tool_search_runs(self, tool_input: Dict) -> str:
        """Search runs by metadata criteria"""
        if not self.db:
            return "Error: Databroker not configured"

        since = tool_input.get('since')
        until = tool_input.get('until')
        metadata = tool_input.get('metadata', {})
        limit = tool_input.get('limit', 20)

        try:
            from dateutil import parser as date_parser

            # Build search
            search_kwargs = dict(metadata) if metadata else {}

            # Parse time range
            if since:
                try:
                    since_time = date_parser.parse(since)
                    search_kwargs['since'] = since_time
                except:
                    pass

            if until:
                try:
                    until_time = date_parser.parse(until)
                    search_kwargs['until'] = until_time
                except:
                    pass

            # Execute search
            if search_kwargs:
                runs = list(self.db.search(search_kwargs))[:limit]
            else:
                runs = list(self.db[-limit:])

            if not runs:
                return "No runs found matching criteria"

            # Format output
            response = f"Found {len(runs)} runs:\n\n"
            for i, run in enumerate(runs):
                start = run.metadata['start']
                uid_short = start['uid'][:8]
                time_str = datetime.fromtimestamp(start['time']).strftime('%Y-%m-%d %H:%M:%S')
                plan = start.get('plan_name', 'unknown')

                response += f"{i+1}. [{uid_short}...] {time_str} - {plan}\n"

                # Show matching metadata
                for key, val in metadata.items():
                    if key in start:
                        response += f"   {key}: {start[key]}\n"

            return response

        except Exception as e:
            import traceback
            return f"Error searching runs: {str(e)}\n{traceback.format_exc()}"
