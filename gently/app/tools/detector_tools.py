"""
Detector Management Tools

Tools for managing detectors that analyze embryo images for specific events.
"""

from typing import Dict, Optional
import json

from gently.harness.tools.registry import tool, ToolCategory
from gently.harness.tools.helpers import require_agent, get_embryo_or_error
from gently.harness.detection.detector import (
    Detector, DetectorConditions, DetectorActions,
    DetectionMode, ConfidenceLevel
)


@tool(
    name="list_detectors",
    description="List all registered detectors and their status",
    category=ToolCategory.DETECTION,
)
def list_detectors(filter: str = "all", context: Dict = None) -> str:
    """List all detectors"""
    agent = context.get('agent') if context else None
    if not agent:
        return "Error: No agent context"

    registry = agent.detector_registry

    if filter == 'enabled':
        detectors = registry.list_enabled()
    elif filter == 'disabled':
        all_detectors = registry.list_all()
        detectors = [d for d in all_detectors if not d.enabled]
    else:
        detectors = registry.list_all()

    if not detectors:
        return f"No {filter} detectors found."

    lines = [f"Detectors ({len(detectors)} {filter}):", ""]

    for detector in detectors:
        status = "enabled" if detector.enabled else "disabled"
        mode = detector.actions.mode.value
        lines.append(f"* {detector.name}: {status}")
        lines.append(f"  Description: {detector.description}")
        lines.append(f"  Action mode: {mode}")
        lines.append(f"  Runs: {detector.run_count}, Detections: {detector.detection_count}")

        if detector.conditions.min_timepoint:
            lines.append(f"  Min timepoint: {detector.conditions.min_timepoint}")

        if detector.actions.parameter_changes:
            lines.append(f"  Parameter changes: {detector.actions.parameter_changes}")

        lines.append("")

    return "\n".join(lines)


@tool(
    name="enable_disable_detector",
    description="Enable or disable a specific detector",
    category=ToolCategory.DETECTION,
)
def enable_disable_detector(
    detector_name: str,
    enabled: bool,
    context: Dict
) -> str:
    """Enable or disable detector"""
    agent, err = require_agent(context)
    if err:
        return err

    if enabled:
        success = agent.detector_registry.enable(detector_name)
        action = "enabled"
    else:
        success = agent.detector_registry.disable(detector_name)
        action = "disabled"

    if success:
        return f"Detector '{detector_name}' {action}"
    else:
        return f"Detector '{detector_name}' not found"


@tool(
    name="remove_detector",
    description="Remove a detector from the registry",
    category=ToolCategory.DETECTION,
)
def remove_detector(detector_name: str, context: Dict) -> str:
    """Remove detector"""
    agent, err = require_agent(context)
    if err:
        return err

    success = agent.detector_registry.remove(detector_name)

    if success:
        agent._mark_significant_action("detector_config")
        return f"Detector '{detector_name}' removed"
    else:
        return f"Detector '{detector_name}' not found"


@tool(
    name="add_detector",
    description="Add a new detector to the system. Use stop_timelapse=True to automatically stop imaging when detected.",
    category=ToolCategory.DETECTION,
)
def add_detector(
    name: str,
    description: str = None,
    detection_prompt: str = None,
    preset: str = None,
    action_mode: str = "passive",
    stop_timelapse: bool = False,
    parameter_changes: Dict = None,
    min_timepoint: int = None,
    context: Dict = None
) -> str:
    """Add a new detector"""
    agent, err = require_agent(context)
    if err:
        return err

    try:
        if preset:
            # Create from preset with custom name
            from gently.harness.detection.registry import get_detector_presets
            presets = get_detector_presets()
            if preset not in presets:
                available = ", ".join(presets.keys())
                return f"Unknown preset '{preset}'. Available: {available}"
            preset_data = presets[preset]
            # Use preset's stop_timelapse unless explicitly overridden
            preset_stop = preset_data.get('stop_timelapse', False)
            detector = Detector(
                name=name,
                description=preset_data['description'],
                detection_prompt=preset_data['prompt'],
                enabled=True,
                conditions=DetectorConditions(min_timepoint=min_timepoint),
                actions=DetectorActions(
                    mode=DetectionMode(action_mode),
                    stop_timelapse=stop_timelapse or preset_stop,
                    parameter_changes=parameter_changes or {}
                ),
                use_temporal_context=preset_data.get('use_temporal_context', True),
                temporal_context_size=preset_data.get('temporal_context_size', 5),
                confidence_threshold=ConfidenceLevel(preset_data.get('confidence_threshold', 'MEDIUM')),
            )
        else:
            # Create custom detector
            if not detection_prompt:
                return "Error: detection_prompt is required for custom detectors"
            detector = Detector(
                name=name,
                description=description or f"Custom detector: {name}",
                detection_prompt=detection_prompt,
                enabled=True,
                conditions=DetectorConditions(min_timepoint=min_timepoint),
                actions=DetectorActions(
                    mode=DetectionMode(action_mode),
                    stop_timelapse=stop_timelapse,
                    parameter_changes=parameter_changes or {}
                ),
            )

        agent.detector_registry.add(detector)
        agent._mark_significant_action("detector_config")
        stop_info = " (will stop timelapse)" if detector.actions.stop_timelapse else ""
        return f"Added detector '{name}' with action mode '{action_mode}'{stop_info}"

    except Exception as e:
        import traceback
        return f"Error adding detector: {str(e)}\n{traceback.format_exc()}"


@tool(
    name="enable_preset_detector",
    description="Enable a preset detector (hatching, comma, pretzel, gastrulation, first_division). Useful for setting up 'run until hatching' workflows.",
    category=ToolCategory.DETECTION,
)
def enable_preset_detector(
    preset: str,
    action_mode: str = "auto",
    min_timepoint: int = None,
    context: Dict = None
) -> str:
    """Enable a preset detector for adaptive experiments"""
    agent, err = require_agent(context)
    if err:
        return err

    from gently.harness.detection.registry import get_detector_presets

    presets = get_detector_presets()

    if preset not in presets:
        available = ", ".join(presets.keys())
        return f"Unknown preset '{preset}'. Available: {available}"

    preset_data = presets[preset]

    # Check if already exists
    existing = agent.detector_registry.get(preset)
    if existing:
        existing.enabled = True
        # Ensure preset critical settings are applied (mode and stop_timelapse)
        existing.actions.mode = DetectionMode(action_mode)
        existing.actions.stop_timelapse = preset_data.get('stop_timelapse', False)
        agent.detector_registry.save()
        return f"Enabled existing '{preset}' detector (mode={action_mode}, stop_timelapse={existing.actions.stop_timelapse})"

    # Create detector from preset
    conditions = DetectorConditions(min_timepoint=min_timepoint)
    actions = DetectorActions(
        mode=DetectionMode(action_mode),
        stop_timelapse=preset_data.get('stop_timelapse', False),
    )

    detector = Detector(
        name=preset_data['name'],
        description=preset_data['description'],
        detection_prompt=preset_data['prompt'],
        enabled=True,
        conditions=conditions,
        actions=actions,
        use_temporal_context=preset_data.get('use_temporal_context', True),
        temporal_context_size=preset_data.get('temporal_context_size', 5),
        confidence_threshold=ConfidenceLevel(preset_data.get('confidence_threshold', 'MEDIUM')),
    )

    agent.detector_registry.add(detector)
    agent._mark_significant_action("detector_config")

    return (
        f"Enabled '{preset}' detector\n"
        f"  Description: {preset_data['description']}\n"
        f"  Action mode: {action_mode}\n"
        f"  Min timepoint: {min_timepoint or 'none'}"
    )


@tool(
    name="generate_detector_prompt",
    description="Generate an optimal detection prompt from a description",
    category=ToolCategory.DETECTION,
)
async def generate_detector_prompt(
    detector_description: str,
    context_info: str = None,
    context: Dict = None
) -> str:
    """Generate detector prompt using Claude"""
    agent, err = require_agent(context)
    if err:
        return err

    try:
        prompt = await agent._generate_detector_prompt(
            description=detector_description,
            additional_context=context_info
        )
        return f"Generated prompt:\n\n{prompt}"
    except Exception as e:
        return f"Error generating prompt: {str(e)}"


@tool(
    name="test_detector",
    description="Test a detector on a specific embryo's latest image",
    category=ToolCategory.DETECTION,
)
async def test_detector(
    detector_name: str,
    embryo_id: str,
    timepoint: int = None,
    context: Dict = None
) -> str:
    """Test detector on embryo"""
    agent, err = require_agent(context)
    if err:
        return err

    detector = agent.detector_registry.get(detector_name)
    if not detector:
        return f"Detector '{detector_name}' not found"

    embryo, err = get_embryo_or_error(agent, embryo_id)
    if err:
        return err

    try:
        result = await agent._run_detector_test(
            detector=detector,
            embryo_id=embryo.id,
            timepoint=timepoint
        )
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error testing detector: {str(e)}"


@tool(
    name="query_timeline_events",
    description="Query timeline for detection and timelapse events. Use this to find when hatching was detected, when acquisitions occurred, or review recent experiment history.",
    category=ToolCategory.DETECTION,
)
def query_timeline_events(
    event_type: str = None,
    embryo_id: str = None,
    detector_name: str = None,
    limit: int = 50,
    session_id: str = "current",
    context: Dict = None
) -> str:
    """
    Query timeline for detection and timelapse events.

    Parameters
    ----------
    event_type : str, optional
        Filter by type: "detection" or "timelapse"
    embryo_id : str, optional
        Filter by embryo ID
    detector_name : str, optional
        Filter detection events by detector name (e.g., "hatching")
    limit : int
        Maximum events to return (default 50)
    session_id : str
        Session filter: "current" (default), "all", or specific session ID
    context : Dict
        Tool context with agent reference

    Returns
    -------
    str
        Formatted list of timeline events
    """
    agent, err = require_agent(context)
    if err:
        return err

    if not hasattr(agent, 'timeline_manager') or agent.timeline_manager is None:
        return "Timeline not available"

    try:
        events = agent.timeline_manager.get_events(
            event_type=event_type,
            embryo_id=embryo_id,
            session_id=session_id,
            limit=limit,
        )

        # Filter by detector_name if specified
        if detector_name and event_type == "detection":
            events = [e for e in events if e.detector_name == detector_name]

        if not events:
            filters = []
            if event_type:
                filters.append(f"type={event_type}")
            if embryo_id:
                filters.append(f"embryo={embryo_id}")
            if detector_name:
                filters.append(f"detector={detector_name}")
            filter_str = ", ".join(filters) if filters else "none"
            return f"No events found (filters: {filter_str})"

        lines = [f"Timeline Events ({len(events)} results):", ""]

        for event in events:
            timestamp = event.timestamp.strftime("%Y-%m-%d %H:%M:%S")
            line = f"[{timestamp}] {event.event_type}/{event.event_subtype}"

            if event.embryo_id:
                line += f" | embryo={event.embryo_id}"
            if event.detector_name:
                line += f" | detector={event.detector_name}"
            if event.timepoint is not None:
                line += f" | t={event.timepoint}"
            if event.confidence:
                line += f" | confidence={event.confidence}"

            lines.append(line)
            if event.description:
                lines.append(f"    {event.description}")

        return "\n".join(lines)

    except Exception as e:
        return f"Error querying timeline: {str(e)}"
