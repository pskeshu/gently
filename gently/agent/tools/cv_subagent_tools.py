"""
CV Subagent Tools

Tools for delegating computer vision analysis to the CV subagent service.
The CV subagent receives high-level intent and autonomously determines
which CV tools to use for C. elegans embryo analysis.

Event-Driven Communication
--------------------------
Instead of polling cv_task_status repeatedly, cv_analyze now uses event-driven
waiting. The CV agent publishes CV_RESULT_READY when analysis completes, and
the copilot subscribes to receive results automatically.
"""

import asyncio
import logging
import uuid
from typing import Dict, List, Optional

from ..tool_registry import tool, ToolCategory, ToolExample
from ..tool_helpers import require_copilot, get_embryo_or_error

logger = logging.getLogger(__name__)

# Default timeout for waiting for CV results (seconds)
CV_RESULT_TIMEOUT = 120.0


def _update_embryo_state_from_cv_result(copilot, embryo_id: str, result: Dict):
    """
    Update EmbryoState with CV analysis result.

    This is called when CV results arrive via event or direct return.
    """
    try:
        embryo = copilot.experiment.embryos.get(embryo_id)
        if embryo is None:
            logger.warning(f"Embryo {embryo_id} not found for CV result update")
            return

        # Extract structured result
        structured = result.get("structured", {})
        result_type = structured.get("result_type", "analysis")

        # Update EmbryoState based on result type
        if hasattr(embryo, 'add_cv_result'):
            embryo.add_cv_result(result_type, structured)

        # Update quick-access fields
        if result_type == "nuclei_count" and "num_nuclei" in structured:
            if hasattr(embryo, 'latest_nuclei_count'):
                embryo.latest_nuclei_count = structured["num_nuclei"]

        elif result_type == "stage_classification" and "stage" in structured:
            if hasattr(embryo, 'latest_developmental_stage'):
                embryo.latest_developmental_stage = structured["stage"]

        elif result_type == "elongation" and "elongation_ratio" in structured:
            if hasattr(embryo, 'latest_elongation_ratio'):
                embryo.latest_elongation_ratio = structured["elongation_ratio"]

        logger.info(f"Updated EmbryoState for {embryo_id} with {result_type} result")

    except Exception as e:
        logger.warning(f"Failed to update EmbryoState from CV result: {e}")


def _format_cv_result(result: Dict) -> str:
    """Format CV analysis result for display."""
    lines = []

    # Get structured data if available
    structured = result.get("structured", {})
    result_type = structured.get("result_type", "analysis")

    if result_type == "nuclei_count":
        num_nuclei = structured.get("num_nuclei", "unknown")
        lines.append(f"Nuclei count: {num_nuclei}")

    elif result_type == "stage_classification":
        stage = structured.get("stage", "unknown")
        confidence = structured.get("confidence", "")
        nuclei_count = structured.get("nuclei_count")
        lines.append(f"Developmental stage: {stage}")
        if confidence:
            lines.append(f"Confidence: {confidence}")
        if nuclei_count:
            lines.append(f"Nuclei count: {nuclei_count}")

    elif result_type == "elongation":
        ratio = structured.get("elongation_ratio", "unknown")
        hint = structured.get("stage_hint", "")
        lines.append(f"Elongation ratio: {ratio}")
        if hint:
            lines.append(f"Stage hint: {hint}")

    elif result_type == "hatching_detection":
        hatched = structured.get("hatched", False)
        confidence = structured.get("confidence", "")
        lines.append(f"Hatched: {'Yes' if hatched else 'No'}")
        if confidence:
            lines.append(f"Confidence: {confidence}")

    # Add summary if available
    summary = result.get("summary") or structured.get("summary", "")
    if summary and len(lines) == 0:
        lines.append(summary[:500])  # Truncate long summaries

    # Add metadata
    if result.get("processing_time_ms"):
        lines.append(f"Processing time: {result['processing_time_ms']:.0f}ms")
    if result.get("tools_used"):
        lines.append(f"Tools used: {', '.join(result['tools_used'][:5])}")

    return "\n".join(lines) if lines else "Analysis completed"


@tool(
    name="cv_analyze",
    description="""Delegate a computer vision analysis task to the CV subagent.

IMPORTANT: This tool requires a volume to have been acquired for the embryo
in the current session. If no volume exists, acquire one first with acquire_volume.

The CV subagent is an intelligent agent that receives high-level intent
and autonomously determines which CV tools to use (Cellpose, StarDist,
Claude Vision, etc.) to accomplish the task.

Use this for:
- Classifying embryo developmental stage
- Counting cells and nuclei
- Tracking cell divisions over time
- Detecting developmental anomalies
- Measuring morphology (elongation, shape metrics)

The CV agent will:
1. Load relevant volume data from the data store
2. Detect and crop the embryo ROI
3. Run appropriate segmentation (Cellpose/StarDist)
4. Measure morphology and other metrics
5. Use Claude Vision with rich context (scale bars, annotations)
6. Return synthesized results

This tool waits for the CV analysis to complete and returns the result directly.""",
    category=ToolCategory.ANALYSIS,
    examples=[
        ToolExample(
            user_query="Classify the developmental stage of embryo_1",
            tool_input={"embryo_id": "embryo_1", "intent": "classify developmental stage"}
        ),
        ToolExample(
            user_query="Count cells and track divisions over 5 timepoints",
            tool_input={
                "embryo_id": "embryo_1",
                "intent": "count cells and track divisions",
                "timepoints": [0, 1, 2, 3, 4]
            }
        ),
        ToolExample(
            user_query="Detect any developmental anomalies",
            tool_input={"embryo_id": "embryo_1", "intent": "detect developmental anomalies"}
        ),
    ],
)
async def cv_analyze(
    intent: str,
    embryo_id: str,
    timepoints: Optional[List[int]] = None,
    additional_context: Optional[Dict] = None,
    context: Dict = None
) -> str:
    """
    Submit a CV analysis request and wait for result via event-driven communication.

    Parameters
    ----------
    intent : str
        High-level description of what to analyze
    embryo_id : str
        ID of the embryo to analyze
    timepoints : list, optional
        Specific timepoints to analyze
    additional_context : dict, optional
        Additional context for the agent (current stage, experiment goals, etc.)
    context : dict
        Tool context (injected by system)

    Returns
    -------
    str
        Analysis result (waits for completion)
    """
    copilot, err = require_copilot(context)
    if err:
        return err

    embryo, err = get_embryo_or_error(copilot, embryo_id)
    if err:
        return err

    # Check if embryo has volume data in this session
    if not embryo.recent_images:
        return (
            f"No volume data for {embryo_id} in this session.\n"
            f"Please acquire a volume first with: acquire_volume {embryo_id}\n"
            f"Then retry the analysis."
        )

    # Get CV client
    try:
        from gently.cv_client import get_cv_client
        cv_client = get_cv_client()
    except ImportError:
        return "Error: CV client not available. Is cv_client.py installed?"

    # Get event bus for result waiting
    try:
        from gently.core.event_bus import get_event_bus, EventType
        event_bus = get_event_bus()
    except ImportError:
        event_bus = None

    try:
        # Connect if not already connected
        if cv_client._client is None:
            await cv_client.connect()

        # Build context for the agent
        cv_context = additional_context or {}

        # Add embryo info to context
        if embryo:
            cv_context["current_stage"] = getattr(embryo, "developmental_stage", None)
            cv_context["embryo_name"] = embryo.name if hasattr(embryo, "name") else embryo_id

            # Tell CV agent which timepoint to use (latest from this session)
            if embryo.recent_images:
                latest = embryo.recent_images[-1]
                cv_context["session_timepoint"] = latest.timepoint
                cv_context["session_id"] = copilot.experiment.session_id if hasattr(copilot.experiment, 'session_id') else None

        # If no specific timepoints requested, use the latest from this session
        if timepoints is None and embryo.recent_images:
            timepoints = [embryo.recent_images[-1].timepoint]

        # Set up event-based result waiting
        result_future = asyncio.get_event_loop().create_future()
        unsubscribe = None

        if event_bus is not None:
            def on_cv_result(event):
                """Handle CV result event"""
                try:
                    data = event.data
                    # Check if this result is for our embryo
                    if data.get("embryo_id") == embryo_id:
                        if not result_future.done():
                            result_future.set_result(data.get("result", data))
                except Exception as e:
                    logger.warning(f"Error handling CV result event: {e}")

            # Subscribe to result event
            unsubscribe = event_bus.subscribe(EventType.CV_RESULT_READY, on_cv_result)
            logger.debug(f"Subscribed to CV_RESULT_READY for {embryo_id}")

        # Submit analysis request
        submit_result = await cv_client.analyze(
            intent=intent,
            embryo_id=embryo_id,
            timepoints=timepoints,
            context=cv_context,
        )

        task_id = submit_result.get("task_id", "unknown")
        logger.info(f"Submitted CV analysis task: {task_id}")

        # Wait for result via event
        if event_bus is not None and unsubscribe is not None:
            try:
                result = await asyncio.wait_for(result_future, timeout=CV_RESULT_TIMEOUT)

                # Update EmbryoState with result
                _update_embryo_state_from_cv_result(copilot, embryo_id, result)

                # Format result for display
                return _format_cv_result(result)

            except asyncio.TimeoutError:
                logger.warning(f"CV analysis timed out after {CV_RESULT_TIMEOUT}s")
                return (
                    f"CV analysis task {task_id} is still running.\n"
                    f"The analysis is taking longer than expected.\n"
                    f"Use 'cv_task_status {task_id}' to check progress."
                )

            finally:
                # Unsubscribe from events
                if unsubscribe:
                    unsubscribe()
        else:
            # Fallback: no event bus, return task submission info
            plan = submit_result.get("plan", [])
            response_lines = [
                f"CV analysis task submitted: {task_id}",
                "Execution plan:",
            ]
            for step in plan:
                response_lines.append(f"  {step}")
            response_lines.append("")
            response_lines.append(f"Use 'cv_task_status {task_id}' to check progress.")
            return "\n".join(response_lines)

    except Exception as e:
        logger.error(f"CV analysis error: {e}", exc_info=True)
        return f"Error in CV analysis: {str(e)}"


@tool(
    name="cv_task_status",
    description="Check the status of a CV subagent task",
    category=ToolCategory.ANALYSIS,
)
async def cv_task_status(
    task_id: str,
    context: Dict = None
) -> str:
    """
    Check status of a CV analysis task

    Parameters
    ----------
    task_id : str
        Task ID to check
    context : dict
        Tool context

    Returns
    -------
    str
        Task status and result (if completed)
    """
    try:
        from gently.cv_client import get_cv_client
        cv_client = get_cv_client()

        if cv_client._client is None:
            await cv_client.connect()

        status = await cv_client.get_task(task_id)

        # Format response
        lines = [
            f"Task: {task_id}",
            f"Status: {status.get('status', 'unknown')}",
        ]

        if status.get("progress") is not None:
            lines.append(f"Progress: {status['progress']:.0f}%")

        if status.get("current_step"):
            lines.append(f"Current step: {status['current_step']}")

        if status.get("status") == "completed" and status.get("result"):
            lines.append("")
            lines.append("Result:")
            result = status["result"]
            if isinstance(result, dict):
                for key, value in result.items():
                    lines.append(f"  {key}: {value}")
            else:
                lines.append(f"  {result}")

        if status.get("status") == "failed" and status.get("error"):
            lines.append("")
            lines.append(f"Error: {status['error']}")

        return "\n".join(lines)

    except Exception as e:
        return f"Error checking task status: {str(e)}"


@tool(
    name="cv_service_status",
    description="Check the status of the CV subagent service",
    category=ToolCategory.ANALYSIS,
)
async def cv_service_status(context: Dict = None) -> str:
    """
    Check CV subagent service status

    Returns
    -------
    str
        Service status including GPU availability and capabilities
    """
    try:
        from gently.cv_client import get_cv_client
        cv_client = get_cv_client()

        if cv_client._client is None:
            await cv_client.connect()

        status = await cv_client.get_status()

        lines = [
            "CV Subagent Service Status",
            "=" * 30,
            f"State: {status.get('state', 'unknown')}",
            f"Host: {status.get('host', 'unknown')}:{status.get('port', 'unknown')}",
            f"GPU available: {status.get('gpu_available', False)}",
            "",
            "Task Queue:",
            f"  Active: {status.get('task_queue', {}).get('active_tasks', 0)}",
            f"  Queued: {status.get('task_queue', {}).get('queued_tasks', 0)}",
            "",
            "Capabilities:",
        ]

        for cap in status.get("capabilities", []):
            lines.append(f"  - {cap}")

        return "\n".join(lines)

    except Exception as e:
        return f"CV subagent service not available: {str(e)}"
