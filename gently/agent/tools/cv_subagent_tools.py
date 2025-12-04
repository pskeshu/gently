"""
CV Subagent Tools

Tools for delegating computer vision analysis to the CV subagent service.
The CV subagent receives high-level intent and autonomously determines
which CV tools to use for C. elegans embryo analysis.
"""

from typing import Dict, List, Optional

from ..tool_registry import tool, ToolCategory, ToolExample
from ..tool_helpers import require_copilot, get_embryo_or_error


@tool(
    name="cv_analyze",
    description="""Delegate a computer vision analysis task to the CV subagent.

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

This is an async operation - it returns a task_id for tracking.""",
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
    Submit a CV analysis request to the CV subagent

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
        Task submission result with task_id and execution plan
    """
    copilot, err = require_copilot(context)
    if err:
        return err

    embryo, err = get_embryo_or_error(copilot, embryo_id)
    if err:
        return err

    # Get CV client
    try:
        from gently.cv_client import get_cv_client
        cv_client = get_cv_client()
    except ImportError:
        return "Error: CV client not available. Is cv_client.py installed?"

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

        # Submit analysis request
        result = await cv_client.analyze(
            intent=intent,
            embryo_id=embryo_id,
            timepoints=timepoints,
            context=cv_context,
        )

        # Format response
        task_id = result.get("task_id", "unknown")
        status = result.get("status", "unknown")
        plan = result.get("plan", [])

        response_lines = [
            f"CV analysis task submitted: {task_id}",
            f"Status: {status}",
            "",
            "Execution plan:",
        ]
        for step in plan:
            response_lines.append(f"  {step}")

        response_lines.append("")
        response_lines.append(f"Use 'cv_task_status {task_id}' to check progress.")

        return "\n".join(response_lines)

    except Exception as e:
        return f"Error submitting CV analysis: {str(e)}"


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
