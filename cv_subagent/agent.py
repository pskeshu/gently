"""
CV Agent - Claude-powered agent for computer vision analysis

The CV agent receives high-level intent and autonomously determines
which CV tools to use for C. elegans embryo analysis.

Architecture:
1. Receive high-level intent (e.g., "classify embryo stage")
2. Claude plans which tools to use
3. Execute tools in sequence, enriching context
4. Use Claude Vision with rich context for visual analysis
5. Return synthesized results
"""

import asyncio
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from .config import CVSubagentConfig, CELEGANS_STAGES
from .tasks.task_queue import TaskQueue, TaskPriority, CVTask
from .events import publish_cv_agent_thinking

logger = logging.getLogger(__name__)

# Maximum iterations to prevent infinite loops
MAX_AGENT_ITERATIONS = 20


class CVAgent:
    """
    Claude-powered agent that orchestrates CV tools

    The agent:
    1. Receives high-level intent (e.g., "classify embryo stage")
    2. Plans which tools to use
    3. Executes tools, enriching context at each step
    4. Uses Claude Vision for visual analysis with rich context
    5. Returns synthesized results
    """

    SYSTEM_PROMPT = """You are a computer vision agent specialized in C. elegans embryo analysis.

Your job is to:
1. UNDERSTAND the intent (classify, track, detect anomaly, etc.)
2. PLAN which tools to use and in what order
3. ENRICH CONTEXT before using Claude Vision:
   - Frame ROI properly around the embryo
   - Get quantitative metrics (nuclei count, morphology)
   - Add scale bars and annotations to images
   - Provide numerical context in vision prompts
4. SYNTHESIZE results from multiple tools into a final answer

## Available Tools

### Embryo Discovery
- list_embryos: List all embryos in the data store
  Parameters: session_id (str, optional), limit (int)
  Returns: List of embryos with timepoint counts

- get_embryo_info: Get detailed information about a specific embryo
  Parameters: embryo_id (str)
  Returns: Timepoints, acquisition parameters, related analyses

### Data Access
- get_volume: Load volume by embryo ID and timepoint
  Parameters: embryo_id (str), timepoint (int, optional)
  Returns: volume_uid for use with other tools

- get_embryo_history: Get list of available timepoints with timestamps
  Parameters: embryo_id (str), last_n (int, optional)
  Returns: timepoints list, volume_uids

- get_latest_volume: Get the most recent volume for an embryo
  Parameters: embryo_id (str)

- get_volume_range: Load multiple consecutive volumes for temporal analysis
  Parameters: embryo_id (str), start_timepoint (int), end_timepoint (int)

- query_volumes: Query with flexible filters
  Parameters: embryo_id, session_id, channel, timepoint_min, timepoint_max

### Image Preparation
- detect_embryo_roi: Find embryo bounding box for proper framing
  Parameters: volume_uid (str), method (str: "threshold"/"otsu"/"adaptive")
  Returns: bbox [z1,y1,x1,z2,y2,x2], center, confidence

- crop_roi: Crop volume to region of interest with padding
  Parameters: volume_uid (str), bbox (list), padding_percent (float)
  Returns: cropped_uid

- prepare_for_vision: Add scale bar, annotations, and project to 2D
  Parameters: volume_uid (str), scale_bar_um (float), annotations (dict)
  Returns: image_base64 ready for vision analysis

- create_timeline_image: Create montage of multiple volumes
  Parameters: volume_uids (list), labels (list), layout (str)
  Returns: image_base64 of timeline montage

- normalize_volume: Normalize volume intensity
  Parameters: volume_uid (str), method (str: "percentile"/"minmax"/"zscore")

### Vision Analysis
- claude_vision_analyze: Analyze image with Claude Vision
  Parameters: image_base64 (str), prompt (str)
  Returns: analysis text

- classify_developmental_stage: High-level stage classification
  Parameters: image_base64 (str), nuclei_count (int), elongation_ratio (float)
  Returns: stage, confidence, reasoning

- detect_visual_anomalies: Detect abnormalities
  Parameters: image_base64 (str), expected_stage (str), expected_nuclei (int)
  Returns: anomalies_detected, anomaly_list, severity

- compare_timepoints: Compare development across timepoints
  Parameters: timeline_image_base64 (str), nuclei_counts (list)
  Returns: progression_summary, changes, division_events

### Results Storage
- store_analysis_result: Store result linked to source volume
  Parameters: volume_uid (str), result_type (str), result (dict)

- get_analysis_results: Get previous analysis results
  Parameters: embryo_id (str), result_type (str)

### Segmentation (GPU-accelerated)
- cellpose_segment_3d: Segment cells/nuclei using Cellpose
  Parameters: volume_uid (str), model_type (str: "cyto2" or "nuclei")
  Returns: num_cells, mask_uid

- stardist_segment_3d: Segment nuclei using StarDist
  Parameters: volume_uid (str)

### Morphology
- measure_morphology: Measure shape metrics from masks
  Parameters: masks_uid (str)
  Returns: elongation, circularity, solidity

## C. elegans Developmental Stages

By nuclei count:
- 1-cell: 1 nucleus (zygote)
- 2-cell: 2 nuclei
- 4-cell: 4 nuclei
- 8-cell: 8 nuclei
- ~28-cell: gastrulation begins
- gastrula: 28-100 nuclei
- comma: ~550 nuclei (elongation starts)
- 1.5-fold, 2-fold, 3-fold: increasing elongation
- pretzel: maximum folding, twitching
- hatching: movement, shell breaking

## Guidelines

1. Always start by loading the relevant volume(s) with get_volume
2. Detect and crop the embryo ROI for better framing
3. Run segmentation to get quantitative metrics (nuclei count is KEY for staging)
4. Use morphology measurements for fold stage classification (elongation ratio)
5. ALWAYS prepare images with scale bars before using Claude Vision
6. Provide rich context in vision prompts (include metrics, scale info)
7. For temporal analysis, load multiple timepoints and create timeline images
8. Store important results using store_analysis_result for future reference
9. When done analyzing, provide a final summary - do NOT call more tools

## Example Flow: "classify embryo stage"

1. get_volume(embryo_id="embryo_1") -> volume_uid
2. detect_embryo_roi(volume_uid) -> bbox
3. crop_roi(volume_uid, bbox, padding_percent=20) -> cropped_uid
4. cellpose_segment_3d(cropped_uid, model_type="nuclei") -> num_cells=24
5. measure_morphology(masks_uid) -> elongation=1.8
6. prepare_for_vision(cropped_uid, scale_bar_um=10, annotations={"nuclei": 24})
7. classify_developmental_stage(image, nuclei_count=24, elongation_ratio=1.8)
8. Final: "This is a gastrula stage embryo (92% confidence)..."

## Example Flow: "track divisions over 5 timepoints"

1. get_embryo_history(embryo_id="embryo_1", last_n=5) -> timepoints
2. get_volume_range(embryo_id, start, end) -> volume_uids
3. For each: cellpose_segment_3d -> nuclei counts [4, 4, 6, 8, 8]
4. create_timeline_image(volume_uids, labels=["t0"...])
5. compare_timepoints(timeline_image, nuclei_counts=[4,4,6,8,8])
6. Final: "Detected 2 division events between t1-t2 and t2-t3..."
"""

    def __init__(
        self,
        anthropic_api_key: Optional[str] = None,
        data_store_url: Optional[str] = None,
        task_queue: Optional[TaskQueue] = None,
        config: Optional[CVSubagentConfig] = None,
    ):
        """
        Initialize CV Agent

        Parameters
        ----------
        anthropic_api_key : str, optional
            Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
        data_store_url : str, optional
            URL for data store access
        task_queue : TaskQueue, optional
            Task queue for async processing
        config : CVSubagentConfig, optional
            Configuration options
        """
        self.api_key = anthropic_api_key or os.environ.get("ANTHROPIC_API_KEY")
        self.data_store_url = data_store_url
        self.task_queue = task_queue
        self.config = config or CVSubagentConfig()

        # Anthropic client (lazy init)
        self._client = None

        # Tool registry (lazy init)
        self._tools = None

        # Token usage tracking (for cost estimation)
        self.total_input_tokens: int = 0
        self.total_output_tokens: int = 0
        self.cache_creation_tokens: int = 0
        self.cache_read_tokens: int = 0
        self.api_call_count: int = 0

        # Register task processor if queue provided
        if self.task_queue:
            self.task_queue.register_processor("cv_analysis", self._process_task)

    @property
    def client(self):
        """Get or create Anthropic client with interleaved thinking support"""
        if self._client is None:
            try:
                import anthropic

                # Enable interleaved thinking beta for multi-step reasoning
                if self.config.enable_interleaved_thinking:
                    self._client = anthropic.Anthropic(
                        api_key=self.api_key,
                        default_headers={
                            "anthropic-beta": "interleaved-thinking-2025-05-14"
                        }
                    )
                else:
                    self._client = anthropic.Anthropic(api_key=self.api_key)
            except ImportError:
                raise RuntimeError("anthropic package not installed")
        return self._client

    def _get_cached_system_prompt(self) -> List[Dict]:
        """Get system prompt formatted for Anthropic prompt caching.

        Returns the system prompt as a list with cache_control to enable
        prompt caching, which dramatically reduces token costs for the
        large system prompt on subsequent API calls.

        Uses 1-hour TTL since CV analysis sessions may have gaps between
        embryo analyses.
        """
        return [
            {
                "type": "text",
                "text": self.SYSTEM_PROMPT,
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
    def token_usage_summary(self) -> str:
        """Get human-readable token usage summary"""
        cache_read = self.cache_read_tokens
        cache_created = self.cache_creation_tokens
        total_input = self.total_input_tokens + cache_read + cache_created
        total = total_input + self.total_output_tokens

        # Cost estimate (Claude Sonnet pricing with 1-hour cache)
        input_cost = self.total_input_tokens * 0.003 / 1000
        cache_read_cost = cache_read * 0.0003 / 1000  # 90% off
        cache_write_cost = cache_created * 0.006 / 1000  # 2x for 1h TTL
        output_cost = self.total_output_tokens * 0.015 / 1000
        total_cost = input_cost + cache_read_cost + cache_write_cost + output_cost

        summary = f"Tokens: {total:,} | API calls: {self.api_call_count} | Est. cost: ${total_cost:.3f}"
        if cache_read > 0:
            savings = (cache_read * 0.003 / 1000) - cache_read_cost
            summary += f" (cache saved ${savings:.3f})"
        return summary

    @property
    def tools(self):
        """Get or create tool registry"""
        if self._tools is None:
            from .tools.registry import CVToolRegistry
            self._tools = CVToolRegistry(
                data_store_url=self.data_store_url,
                config=self.config,
            )
        return self._tools

    def _get_tools_for_claude(self) -> List[Dict]:
        """Get tool definitions in Claude's format"""
        return self.tools.get_claude_tools_schema()

    async def submit_task(
        self,
        intent: str,
        embryo_id: str,
        timepoints: Optional[List[int]] = None,
        volume_uids: Optional[List[str]] = None,
        context: Optional[Dict[str, Any]] = None,
        priority: TaskPriority = TaskPriority.NORMAL,
    ) -> Dict[str, Any]:
        """
        Submit an analysis task

        Parameters
        ----------
        intent : str
            High-level intent
        embryo_id : str
            Embryo to analyze
        timepoints : list, optional
            Specific timepoints
        volume_uids : list, optional
            Specific volumes
        context : dict, optional
            Additional context
        priority : TaskPriority
            Task priority

        Returns
        -------
        dict
            Task submission result
        """
        # Plan the analysis
        plan = await self._create_plan(intent, embryo_id, context)

        # Create task
        params = {
            "intent": intent,
            "embryo_id": embryo_id,
            "timepoints": timepoints,
            "volume_uids": volume_uids,
            "context": context or {},
        }

        task = await self.task_queue.submit(
            task_type="cv_analysis",
            params=params,
            priority=priority,
            callback_event="ANALYSIS_COMPLETED",
            metadata={"embryo_id": embryo_id, "intent": intent},
            plan=plan,
        )

        return {
            "task_id": task.task_id,
            "status": task.status.value,
            "plan": plan,
            "estimated_time_seconds": self._estimate_time(plan),
        }

    async def _create_plan(
        self,
        intent: str,
        embryo_id: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[str]:
        """
        Create an execution plan for the intent

        Uses keyword analysis to determine likely steps.
        """
        intent_lower = intent.lower()
        plan = []

        # Always start with data loading
        plan.append(f"1. Load volume data for {embryo_id}")
        plan.append("2. Detect embryo ROI and crop")

        # Segmentation for most tasks
        if any(word in intent_lower for word in ["count", "cell", "nuclei", "segment", "track", "division", "classify", "stage"]):
            plan.append("3. Run Cellpose 3D segmentation")
            plan.append("4. Extract cell count and properties")

        # Classification
        if any(word in intent_lower for word in ["classify", "stage", "developmental"]):
            plan.append("5. Measure morphology (elongation, shape)")
            plan.append("6. Prepare image with scale bar and annotations")
            plan.append("7. Claude Vision: classify developmental stage")

        # Tracking
        if any(word in intent_lower for word in ["track", "division", "lineage"]):
            plan.append("5. Load multiple timepoints")
            plan.append("6. Segment each timepoint")
            plan.append("7. Track cells across timepoints")
            plan.append("8. Identify division events")

        # Anomaly detection
        if any(word in intent_lower for word in ["anomaly", "abnormal", "unusual"]):
            plan.append("5. Compare to expected developmental patterns")
            plan.append("6. Claude Vision: identify anomalies")

        # Always end with synthesis
        plan.append(f"{len(plan)+1}. Synthesize results and generate report")

        return plan

    def _estimate_time(self, plan: List[str]) -> float:
        """Estimate processing time based on plan"""
        time_per_step = {
            "load": 2,
            "detect": 3,
            "crop": 1,
            "cellpose": 30,
            "stardist": 20,
            "segment": 25,
            "morphology": 5,
            "track": 10,
            "vision": 5,
            "synthesize": 2,
        }

        total = 0
        for step in plan:
            step_lower = step.lower()
            for key, seconds in time_per_step.items():
                if key in step_lower:
                    total += seconds
                    break
            else:
                total += 5

        return total

    async def _process_task(self, task: CVTask) -> Dict[str, Any]:
        """
        Process a CV analysis task

        This is the main entry point for task execution.
        """
        params = task.params
        intent = params["intent"]
        embryo_id = params["embryo_id"]
        context = params.get("context", {})

        logger.info(f"Processing CV task: {task.task_id} - {intent}")

        # Update progress
        self.task_queue.update_progress(task.task_id, 0, "Starting analysis")

        try:
            # Execute the analysis using agentic loop
            result = await self._execute_analysis(
                intent=intent,
                embryo_id=embryo_id,
                timepoints=params.get("timepoints"),
                volume_uids=params.get("volume_uids"),
                context=context,
                task=task,
            )

            return result

        except Exception as e:
            logger.error(f"Analysis failed: {e}", exc_info=True)
            raise

    async def _execute_analysis(
        self,
        intent: str,
        embryo_id: str,
        timepoints: Optional[List[int]] = None,
        volume_uids: Optional[List[str]] = None,
        context: Dict[str, Any] = None,
        task: Optional[CVTask] = None,
    ) -> Dict[str, Any]:
        """
        Execute CV analysis using Claude's agentic loop

        The agent will:
        1. Interpret the intent
        2. Call appropriate tools
        3. Enrich context at each step
        4. Generate final result
        """
        context = context or {}
        start_time = datetime.now()
        tools_used = []
        thinking_steps = []  # Collect thinking blocks to show user

        # Build initial user message
        user_message = f"""Analyze the following request for C. elegans embryo analysis:

Intent: {intent}
Embryo ID: {embryo_id}
Timepoints: {timepoints if timepoints else "use latest available"}
Additional context: {json.dumps(context) if context else "none"}

Please execute the analysis by calling the appropriate tools in sequence.
Start by loading the volume data, then proceed step by step.
When you have gathered enough information, provide a final summary.
"""

        # Get tools for Claude
        claude_tools = self._get_tools_for_claude()

        # Initialize conversation
        messages = [{"role": "user", "content": user_message}]

        # Agentic loop
        iteration = 0
        final_result = None

        while iteration < MAX_AGENT_ITERATIONS:
            iteration += 1
            logger.debug(f"Agent iteration {iteration}")

            # Update progress
            if task:
                progress = min(iteration / MAX_AGENT_ITERATIONS * 80, 80)
                self.task_queue.update_progress(
                    task.task_id,
                    progress,
                    f"Processing (iteration {iteration})"
                )

            try:
                # Build API call parameters
                api_params = {
                    "model": self.config.vision_model,
                    "max_tokens": self.config.vision_max_tokens,
                    "system": self._get_cached_system_prompt(),
                    "tools": claude_tools,
                    "messages": messages,
                }

                # Add interleaved thinking for multi-step reasoning between tool calls
                if self.config.enable_interleaved_thinking:
                    api_params["thinking"] = {
                        "type": "enabled",
                        "budget_tokens": self.config.thinking_budget_tokens,
                    }

                # Call Claude with prompt caching for system prompt
                response = self.client.messages.create(**api_params)

                # Track token usage including cache metrics
                self._track_token_usage(response)

                # Process response
                assistant_content = response.content
                stop_reason = response.stop_reason

                # Stream thinking blocks in real-time via EventBus (interleaved thinking)
                task_id = task.task_id if task else "immediate"
                for block in assistant_content:
                    if hasattr(block, 'type') and block.type == "thinking":
                        thinking_text = getattr(block, 'thinking', '')
                        if thinking_text:
                            # Collect for final result
                            thinking_steps.append({
                                "iteration": iteration,
                                "thinking": thinking_text,
                            })
                            # Publish for real-time streaming to subscribers
                            publish_cv_agent_thinking(
                                task_id=task_id,
                                thinking=thinking_text,
                                iteration=iteration,
                                embryo_id=embryo_id,
                            )
                            logger.debug(f"Agent thinking (iter {iteration}): {thinking_text[:200]}...")

                # Add assistant message to conversation
                messages.append({"role": "assistant", "content": assistant_content})

                # Check if we're done
                if stop_reason == "end_turn":
                    # Extract final text response
                    for block in assistant_content:
                        if hasattr(block, 'text'):
                            final_result = block.text
                            break
                    break

                # Process tool calls
                if stop_reason == "tool_use":
                    tool_results = []

                    for block in assistant_content:
                        if block.type == "tool_use":
                            tool_name = block.name
                            tool_input = block.input
                            tool_use_id = block.id

                            logger.info(f"Executing tool: {tool_name}")
                            tools_used.append(tool_name)

                            # Execute the tool
                            try:
                                result = await self.tools.execute(tool_name, **tool_input)
                                tool_results.append({
                                    "type": "tool_result",
                                    "tool_use_id": tool_use_id,
                                    "content": json.dumps(result) if isinstance(result, dict) else str(result),
                                })
                            except Exception as e:
                                logger.error(f"Tool {tool_name} failed: {e}")
                                tool_results.append({
                                    "type": "tool_result",
                                    "tool_use_id": tool_use_id,
                                    "content": f"Error: {str(e)}",
                                    "is_error": True,
                                })

                    # Add tool results to conversation
                    messages.append({"role": "user", "content": tool_results})

            except Exception as e:
                logger.error(f"Claude API error: {e}")
                raise

        # Build final result
        processing_time = (datetime.now() - start_time).total_seconds() * 1000

        result = {
            "task_id": task.task_id if task else "immediate",
            "intent": intent,
            "embryo_id": embryo_id,
            "completed_at": datetime.now().isoformat(),
            "summary": final_result or "Analysis completed but no summary generated",
            "processing_time_ms": processing_time,
            "tools_used": list(set(tools_used)),
            "iterations": iteration,
            "thinking_steps": thinking_steps,  # Agent reasoning visible to user
        }

        if task:
            self.task_queue.update_progress(task.task_id, 100, "Complete")

        logger.info(f"Analysis completed in {iteration} iterations, {processing_time:.0f}ms")
        return result

    async def analyze_immediate(
        self,
        intent: str,
        embryo_id: str,
        timepoints: Optional[List[int]] = None,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Execute analysis immediately (synchronous, blocking)

        Use for testing or when async queue is not needed.
        """
        return await self._execute_analysis(
            intent=intent,
            embryo_id=embryo_id,
            timepoints=timepoints,
            context=context,
        )
