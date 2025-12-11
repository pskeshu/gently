"""
Timelapse Orchestration Tools

Tools for managing adaptive timelapse acquisitions.
"""

from typing import Dict, List

from ..tool_registry import tool, ToolCategory
from ..tool_helpers import (
    require_copilot, get_embryo_or_error,
    require_timelapse_orchestrator, require_developmental_tracker
)
from ..detector import (
    Detector, DetectorConditions, DetectorActions,
    DetectionMode, ConfidenceLevel
)


@tool(
    name="generate_bluesky_plan",
    description="Generate a Bluesky acquisition plan from a scientific goal",
    category=ToolCategory.EXPERIMENT,
)
async def generate_bluesky_plan(
    goal: str,
    embryo_ids: List[str],
    plan_type: str = "adaptive_timelapse",
    parameters: Dict = None,
    context: Dict = None
) -> str:
    """Generate Bluesky plan"""
    copilot = context.get('copilot')

    if not copilot:
        return "Error: No copilot context"

    try:
        result = await copilot._generate_bluesky_plan(
            goal=goal,
            embryo_ids=embryo_ids,
            plan_type=plan_type,
            parameters=parameters or {}
        )
        return result

    except Exception as e:
        return f"Error generating plan: {str(e)}"


@tool(
    name="start_adaptive_timelapse",
    description="Start an adaptive timelapse that runs in the background. Copilot remains responsive while acquisition continues.",
    category=ToolCategory.EXPERIMENT,
    requires_microscope=True,
)
async def start_adaptive_timelapse(
    embryo_ids: List[str] = None,
    stop_condition: str = "manual",
    interval_seconds: float = 120.0,
    condition_value: int = None,
    context: Dict = None
) -> str:
    """Start adaptive timelapse in background"""
    copilot, err = require_copilot(context)
    if err:
        return err

    orchestrator, err = require_timelapse_orchestrator(copilot)
    if err:
        return err

    try:
        result = await orchestrator.start(
            embryo_ids=embryo_ids,
            stop_condition=stop_condition,
            base_interval_seconds=interval_seconds,
            condition_value=condition_value,
        )
        return result
    except Exception as e:
        return f"Error starting timelapse: {str(e)}"


@tool(
    name="get_timelapse_status",
    description="Get current status of the running timelapse including per-embryo progress",
    category=ToolCategory.EXPERIMENT,
)
def get_timelapse_status(context: Dict = None) -> str:
    """Get timelapse status"""
    copilot, err = require_copilot(context)
    if err:
        return err

    orchestrator, err = require_timelapse_orchestrator(copilot)
    if err:
        return err

    state = orchestrator.get_status()
    status_dict = state.to_dict()

    lines = [
        f"Timelapse Status: {status_dict['status'].upper()}",
        ""
    ]

    if status_dict['started_at']:
        lines.append(f"Started: {status_dict['started_at']}")
        lines.append(f"Duration: {status_dict['duration_minutes']:.1f} minutes")
        lines.append(f"Total timepoints acquired: {status_dict['total_timepoints']}")
        lines.append("")

    lines.append(f"Active embryos: {status_dict['active_embryos']}")
    lines.append(f"Completed embryos: {status_dict['completed_embryos']}")
    lines.append("")

    if status_dict['next_embryo']:
        lines.append(f"Next acquisition: {status_dict['next_embryo']} in {status_dict['next_acquisition_in_seconds']:.0f}s")
        lines.append("")

    if status_dict['embryo_details']:
        lines.append("Embryo Details:")
        for eid, details in status_dict['embryo_details'].items():
            status_marker = "[done]" if details['is_complete'] else "[active]"
            lines.append(f"  {status_marker} {eid}: t={details['timepoints']} "
                        f"(interval={details['interval_seconds']}s)")
            if details['is_complete']:
                lines.append(f"      Completed: {details['completion_reason']}")

    if status_dict['error']:
        lines.append("")
        lines.append(f"Error: {status_dict['error']}")

    return "\n".join(lines)


@tool(
    name="modify_timelapse_embryo",
    description="Modify parameters for a specific embryo during a running timelapse. Note: interval is now global - use modify_timelapse_interval to change it.",
    category=ToolCategory.EXPERIMENT,
    requires_microscope=True,
)
async def modify_timelapse_embryo(
    embryo_id: str,
    stop_condition: str = None,
    condition_value: int = None,
    context: Dict = None
) -> str:
    """Modify embryo parameters during timelapse (stop condition only - interval is global)"""
    copilot, err = require_copilot(context)
    if err:
        return err

    orchestrator, err = require_timelapse_orchestrator(copilot)
    if err:
        return "No timelapse running."

    try:
        result = await orchestrator.modify_embryo(
            embryo_id=embryo_id,
            stop_condition=stop_condition,
            condition_value=condition_value,
        )
        return result
    except Exception as e:
        return f"Error modifying embryo: {str(e)}"


@tool(
    name="add_embryo_to_timelapse",
    description="Add an embryo to an already running timelapse. The embryo will use the global interval and join on the next round.",
    category=ToolCategory.EXPERIMENT,
    requires_microscope=True,
)
async def add_embryo_to_timelapse(
    embryo_id: str,
    stop_condition: str = None,
    condition_value: int = None,
    context: Dict = None
) -> str:
    """Add an embryo to a running timelapse (uses global interval)"""
    copilot, err = require_copilot(context)
    if err:
        return err

    orchestrator, err = require_timelapse_orchestrator(copilot)
    if err:
        return "No timelapse running. Use start_adaptive_timelapse first."

    try:
        result = await orchestrator.add_embryo(
            embryo_id=embryo_id,
            stop_condition=stop_condition,
            condition_value=condition_value,
        )
        return result
    except Exception as e:
        return f"Error adding embryo: {str(e)}"


@tool(
    name="stop_timelapse_embryo",
    description="Stop imaging a specific embryo in the timelapse (other embryos continue)",
    category=ToolCategory.EXPERIMENT,
    requires_microscope=True,
)
async def stop_timelapse_embryo(
    embryo_id: str,
    reason: str = "user_request",
    context: Dict = None
) -> str:
    """Stop imaging a specific embryo"""
    copilot, err = require_copilot(context)
    if err:
        return err

    orchestrator, err = require_timelapse_orchestrator(copilot)
    if err:
        return "No timelapse running."

    try:
        result = await orchestrator.stop_embryo(embryo_id, reason)
        return result
    except Exception as e:
        return f"Error stopping embryo: {str(e)}"


@tool(
    name="stop_timelapse",
    description="Stop the entire timelapse acquisition",
    category=ToolCategory.EXPERIMENT,
    requires_microscope=True,
)
async def stop_timelapse(
    reason: str = "user_request",
    context: Dict = None
) -> str:
    """Stop entire timelapse"""
    copilot, err = require_copilot(context)
    if err:
        return err

    orchestrator, err = require_timelapse_orchestrator(copilot)
    if err:
        return "No timelapse running."

    try:
        result = await orchestrator.stop(reason)
        return result
    except Exception as e:
        return f"Error stopping timelapse: {str(e)}"


@tool(
    name="pause_timelapse",
    description="Pause the timelapse acquisition (can be resumed later)",
    category=ToolCategory.EXPERIMENT,
    requires_microscope=True,
)
async def pause_timelapse(context: Dict = None) -> str:
    """Pause timelapse"""
    copilot, err = require_copilot(context)
    if err:
        return err

    orchestrator, err = require_timelapse_orchestrator(copilot)
    if err:
        return "No timelapse running."

    try:
        result = await orchestrator.pause()
        return result
    except Exception as e:
        return f"Error pausing timelapse: {str(e)}"


@tool(
    name="resume_timelapse",
    description="Resume a paused timelapse",
    category=ToolCategory.EXPERIMENT,
    requires_microscope=True,
)
async def resume_timelapse(context: Dict = None) -> str:
    """Resume timelapse"""
    copilot, err = require_copilot(context)
    if err:
        return err

    orchestrator, err = require_timelapse_orchestrator(copilot)
    if err:
        return "No timelapse to resume."

    try:
        result = await orchestrator.resume()
        return result
    except Exception as e:
        return f"Error resuming timelapse: {str(e)}"


@tool(
    name="add_stop_condition",
    description="Add an additional stop condition to a running timelapse (OR logic). E.g., add 'hatching' condition to a timelapse running with a duration limit.",
    category=ToolCategory.EXPERIMENT,
)
def add_stop_condition(
    embryo_id: str,
    condition: str,
    context: Dict = None
) -> str:
    """
    Add an additional stop condition to an embryo in a running timelapse.

    The new condition is added with OR logic - the embryo will stop when
    ANY condition is met (the original OR the new one).

    Parameters
    ----------
    embryo_id : str
        ID of the embryo to modify
    condition : str
        Stop condition to add. Supported formats:
        - "hatching" - stop when hatching detected
        - "comma" or "comma_stage" - stop at comma stage
        - "duration:10" or "duration:10h" - stop after 10 hours
        - "timepoints:100" - stop after 100 timepoints

    Examples
    --------
    - Timelapse started with "duration:10h", add hatching detection:
      add_stop_condition(embryo_id="embryo1", condition="hatching")
      Result: Embryo stops on hatching OR 10h, whichever comes first

    Returns
    -------
    str
        Confirmation message with new stop condition description
    """
    copilot, err = require_copilot(context)
    if err:
        return err

    orchestrator, err = require_timelapse_orchestrator(copilot)
    if err:
        return "No timelapse running."

    # Parse the new condition
    from ..timelapse_orchestrator import StopCondition

    try:
        new_condition = StopCondition.parse(condition)
    except ValueError as e:
        return f"Invalid stop condition: {e}"

    # Find the embryo state in the orchestrator
    embryo_state = orchestrator._embryo_states.get(embryo_id)
    if not embryo_state:
        available = list(orchestrator._embryo_states.keys())
        return f"Embryo '{embryo_id}' not found in timelapse. Available: {available}"

    # Add the condition
    embryo_state.stop_condition.add_condition(new_condition)

    # Get updated description
    new_desc = embryo_state.stop_condition.describe()

    return (
        f"Added stop condition '{condition}' to {embryo_id}\n"
        f"Stop conditions: {new_desc}"
    )


@tool(
    name="add_interval_speedup_rule",
    description="Add a rule to automatically speed up imaging when a detector fires (e.g., 'speed up to 30s when pretzel stage detected')",
    category=ToolCategory.EXPERIMENT,
)
def add_interval_speedup_rule(
    trigger_detector: str,
    new_interval_seconds: float = 30.0,
    embryo_ids: List[str] = None,
    context: Dict = None
) -> str:
    """Add interval speedup rule"""
    copilot, err = require_copilot(context)
    if err:
        return err

    orchestrator, err = require_timelapse_orchestrator(copilot)
    if err:
        return err

    orchestrator.add_speedup_on_detection(
        detector_name=trigger_detector,
        new_interval_seconds=new_interval_seconds,
        embryo_ids=embryo_ids,
    )

    msg = f"Added interval rule: speed up to {new_interval_seconds}s when '{trigger_detector}' detector fires"
    if embryo_ids:
        msg += f" (for embryos: {', '.join(embryo_ids)})"

    return msg


@tool(
    name="enable_pre_hatching_speedup",
    description="Enable automatic speedup when embryos approach hatching (triggers on pretzel/3-fold stage detection)",
    category=ToolCategory.EXPERIMENT,
)
def enable_pre_hatching_speedup(
    fast_interval_seconds: float = 30.0,
    context: Dict = None
) -> str:
    """Enable pre-hatching speedup"""
    copilot, err = require_copilot(context)
    if err:
        return err

    orchestrator, err = require_timelapse_orchestrator(copilot)
    if err:
        return err

    # Enable pretzel detector if not already enabled
    from ..detector_registry import get_detector_presets

    presets = get_detector_presets()
    if 'pretzel' in presets and not copilot.detector_registry.get('pretzel'):
        preset_data = presets['pretzel']
        detector = Detector(
            name='pretzel',
            description=preset_data['description'],
            detection_prompt=preset_data['prompt'],
            enabled=True,
            conditions=DetectorConditions(),
            actions=DetectorActions(mode=DetectionMode.AUTO),
            use_temporal_context=True,
            temporal_context_size=5,
            confidence_threshold=ConfidenceLevel.MEDIUM,
        )
        copilot.detector_registry.add(detector)

    orchestrator.add_pre_hatching_speedup(fast_interval_seconds)

    return (
        f"Enabled pre-hatching speedup:\n"
        f"  - Pretzel detector enabled\n"
        f"  - When pretzel (3-fold) detected, interval will change to {fast_interval_seconds}s\n"
        f"  - This helps capture hatching at high temporal resolution"
    )


@tool(
    name="classify_embryo_stage",
    description="Use Claude Vision to classify the current developmental stage of an embryo",
    category=ToolCategory.ANALYSIS,
)
async def classify_embryo_stage(
    embryo_id: str,
    context: Dict = None
) -> str:
    """Classify embryo stage"""
    copilot, err = require_copilot(context)
    if err:
        return err

    embryo, err = get_embryo_or_error(copilot, embryo_id)
    if err:
        return err

    if not embryo.recent_images:
        return f"No images available for {embryo_id}. Acquire a volume first."

    latest = embryo.recent_images[-1]

    # Initialize tracker if needed
    if not hasattr(copilot, 'developmental_tracker') or copilot.developmental_tracker is None:
        from ..developmental_tracker import DevelopmentalTracker
        copilot.developmental_tracker = DevelopmentalTracker(
            claude_client=copilot.claude,
            model=copilot.model,
        )

    recent = []
    for img in embryo.recent_images[-5:]:
        recent.append({
            'timepoint': img.timepoint,
            'b64_image': img.max_projection_b64,
        })

    result = copilot.developmental_tracker.classify_stage(
        image_b64=latest.max_projection_b64,
        embryo_id=embryo.id,
        timepoint=latest.timepoint,
        recent_images=recent,
    )

    lines = [
        f"Stage Classification for {embryo.id}:",
        f"  Stage: {result.stage.value}",
        f"  Confidence: {result.confidence}",
        f"  Reasoning: {result.reasoning}",
    ]

    if result.predicted_minutes_to_hatching is not None:
        hours = result.predicted_minutes_to_hatching / 60
        lines.append(f"  Predicted time to hatching: ~{hours:.1f} hours ({result.predicted_minutes_to_hatching} min)")

    return "\n".join(lines)


@tool(
    name="get_stage_history",
    description="Get the developmental stage progression history for an embryo",
    category=ToolCategory.ANALYSIS,
)
def get_stage_history(
    embryo_id: str,
    context: Dict = None
) -> str:
    """Get stage history"""
    copilot, err = require_copilot(context)
    if err:
        return err

    tracker, err = require_developmental_tracker(copilot)
    if err:
        return err

    summary = tracker.get_progression_summary(embryo_id)

    if summary['observations'] == 0:
        return f"No stage classifications for {embryo_id}. Use classify_embryo_stage first."

    lines = [
        f"Stage Progression for {embryo_id}:",
        f"  Current stage: {summary['current_stage']} ({summary['current_confidence']})",
        f"  Observations: {summary['observations']}",
        f"  Stages observed: {', '.join(summary['stages_observed'])}",
    ]

    if summary['predicted_minutes_to_hatching'] is not None:
        hours = summary['predicted_minutes_to_hatching'] / 60
        lines.append(f"  Predicted time to hatching: ~{hours:.1f} hours")

    return "\n".join(lines)


@tool(
    name="predict_hatching",
    description="Predict time-to-hatching for an embryo with confidence intervals based on developmental stage",
    category=ToolCategory.ANALYSIS,
)
def predict_hatching(
    embryo_id: str = None,
    all_embryos: bool = False,
    context: Dict = None
) -> str:
    """Predict hatching time with confidence intervals"""
    copilot, err = require_copilot(context)
    if err:
        return err

    tracker, err = require_developmental_tracker(copilot)
    if err:
        return err

    if all_embryos:
        embryo_ids = list(copilot.experiment.embryos.keys())
        predictions = tracker.get_all_predictions(embryo_ids)

        if not predictions:
            return "No predictions available. Use classify_embryo_stage on embryos first."

        lines = ["Hatching Predictions:", ""]

        for eid, pred in predictions.items():
            lines.append(f"  {eid}:")
            lines.append(f"    Current stage: {pred.current_stage.value}")
            lines.append(f"    Predicted: {pred.predicted_hours:.1f}h ({pred.predicted_minutes} min)")
            lines.append(f"    Range: {pred.range_hours[0]:.1f} - {pred.range_hours[1]:.1f}h")
            lines.append(f"    Confidence: {pred.confidence}")
            lines.append("")

        lines.append("Development Rates:")
        for eid in predictions.keys():
            rate = tracker.estimate_development_rate(eid)
            if rate:
                rate_pct = (rate - 1.0) * 100
                speed = "faster" if rate > 1.0 else "slower"
                lines.append(f"  {eid}: {abs(rate_pct):.1f}% {speed} than standard")

        return "\n".join(lines)

    else:
        if not embryo_id:
            return "Specify embryo_id or set all_embryos=True"

        pred = tracker.get_hatching_prediction(embryo_id)
        if not pred:
            return f"No prediction available for {embryo_id}. Use classify_embryo_stage first."

        lines = [
            f"Hatching Prediction for {embryo_id}:",
            f"  Current stage: {pred.current_stage.value}",
            f"  Predicted time to hatching: {pred.predicted_hours:.1f} hours ({pred.predicted_minutes} min)",
            f"  Confidence interval: {pred.range_hours[0]:.1f} - {pred.range_hours[1]:.1f} hours",
            f"  Classification confidence: {pred.confidence}",
        ]

        rate = tracker.estimate_development_rate(embryo_id)
        if rate:
            rate_pct = (rate - 1.0) * 100
            speed = "faster" if rate > 1.0 else "slower"
            lines.append(f"  Development rate: {abs(rate_pct):.1f}% {speed} than standard")

        return "\n".join(lines)
