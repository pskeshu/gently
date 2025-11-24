"""
Main Microscopy Copilot implementation
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
from .tools import get_tool_definitions
from .detector_registry import DetectorRegistry
from .detection_queue import DetectionQueue


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
        run_engine=None,
        devices: Optional[Dict] = None,
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
        run_engine : RunEngine, optional
            Bluesky RunEngine for executing plans
        devices : dict, optional
            Ophyd devices (volume_scanner, xy_stage, etc.)
        """
        # API client
        self.claude = anthropic.Anthropic(api_key=api_key or os.getenv("ANTHROPIC_API_KEY"))
        self.model = model

        # Conversation state
        self.conversation_history: List[Dict] = []
        self.system_prompt: str = ""

        # Experiment state
        self.experiment = ExperimentState()

        # Image management
        self.image_manager = ImageManager(
            storage_path=storage_path / "images",
            history_length=10
        )

        # Plan synthesis
        self.plan_synthesizer = PlanSynthesizer(
            plan_library=PlanLibrary(),
            validator=PlanValidator(devices=devices)
        )

        # Detector system
        self.detector_registry = DetectorRegistry(
            storage_path=storage_path / "detector_registry.json"
        )
        self.detection_queue = DetectionQueue(
            registry=self.detector_registry,
            image_manager=self.image_manager,
            claude_client=self.claude,
            model=self.model,
            on_detection_callback=self._on_detection_fired
        )

        # Hardware interface
        self.run_engine = run_engine
        self.devices = devices or {}

        # Callbacks
        self.on_message_callback: Optional[Callable] = None

        # Build initial system prompt
        self._update_system_prompt()

    def _update_system_prompt(self):
        """Rebuild system prompt with current experiment state"""
        self.system_prompt = build_system_prompt(self.experiment)

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
            tools=get_tool_definitions(),
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
                tools=get_tool_definitions(),
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
                tools=get_tool_definitions(),
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
            # Execute tools
            for block in response_content:
                if hasattr(block, 'type') and block.type == "tool_use":
                    start_time = time.time()

                    # Execute tool
                    tool_result = await self._execute_single_tool(block.name, block.input)

                    # Yield tool call info
                    yield {
                        'type': 'tool_call',
                        'tool_name': block.name,
                        'tool_input': block.input,
                        'duration': time.time() - start_time,
                    }

            # Continue conversation with tool results
            tool_results = await self._execute_tools(response_content)

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
        """Execute a single tool call"""

        if tool_name == "generate_bluesky_plan":
            return await self._tool_generate_plan(tool_input)

        elif tool_name == "query_embryo_status":
            return self._tool_query_embryo(tool_input)

        elif tool_name == "analyze_volume":
            return await self._tool_analyze_volume(tool_input)

        elif tool_name == "modify_parameters":
            return self._tool_modify_parameters(tool_input)

        elif tool_name == "get_experiment_summary":
            return self._tool_experiment_summary()

        elif tool_name == "skip_embryo":
            return self._tool_skip_embryo(tool_input)

        elif tool_name == "resume_embryo":
            return self._tool_resume_embryo(tool_input)

        elif tool_name == "assign_nickname":
            return self._tool_assign_nickname(tool_input)

        # Detector management tools
        elif tool_name == "list_detectors":
            return self._tool_list_detectors(tool_input)

        elif tool_name == "add_detector":
            return await self._tool_add_detector(tool_input)

        elif tool_name == "generate_detector_prompt":
            return await self._tool_generate_detector_prompt(tool_input)

        elif tool_name == "test_detector":
            return await self._tool_test_detector(tool_input)

        elif tool_name == "enable_disable_detector":
            return self._tool_enable_disable_detector(tool_input)

        elif tool_name == "remove_detector":
            return self._tool_remove_detector(tool_input)

        elif tool_name == "get_detection_summary":
            return self._tool_get_detection_summary()

        else:
            raise ValueError(f"Unknown tool: {tool_name}")

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
        self.image_manager.store_volume(embryo, timepoint, volume)

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
