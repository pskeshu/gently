"""
Detector Management Tools

Tools for managing detectors that analyze embryo images for specific events.
"""

from typing import Dict, Optional
import json

from ..tool_registry import tool, ToolCategory
from ..tool_helpers import require_copilot, get_embryo_or_error
from ..detector import (
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
    copilot = context.get('copilot') if context else None
    if not copilot:
        return "Error: No copilot context"

    registry = copilot.detector_registry

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
    copilot, err = require_copilot(context)
    if err:
        return err

    if enabled:
        success = copilot.detector_registry.enable(detector_name)
        action = "enabled"
    else:
        success = copilot.detector_registry.disable(detector_name)
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
    copilot, err = require_copilot(context)
    if err:
        return err

    success = copilot.detector_registry.remove(detector_name)

    if success:
        copilot._mark_significant_action("detector_config")
        return f"Detector '{detector_name}' removed"
    else:
        return f"Detector '{detector_name}' not found"


@tool(
    name="add_detector",
    description="Add a new detector to the system",
    category=ToolCategory.DETECTION,
)
def add_detector(
    name: str,
    description: str = None,
    detection_prompt: str = None,
    preset: str = None,
    action_mode: str = "passive",
    parameter_changes: Dict = None,
    min_timepoint: int = None,
    context: Dict = None
) -> str:
    """Add a new detector"""
    copilot, err = require_copilot(context)
    if err:
        return err

    try:
        if preset:
            detector = copilot.detector_registry.create_from_preset(preset, name)
        else:
            from ..detector import Detector, DetectorConfig, DetectorActions, DetectorConditions, ActionMode

            actions = DetectorActions(
                mode=ActionMode(action_mode),
                parameter_changes=parameter_changes or {}
            )
            conditions = DetectorConditions(min_timepoint=min_timepoint)
            config = DetectorConfig(
                name=name,
                description=description or f"Custom detector: {name}",
                detection_prompt=detection_prompt or "",
                actions=actions,
                conditions=conditions
            )
            detector = Detector(config)
            copilot.detector_registry.register(detector)

        copilot._mark_significant_action("detector_config")
        return f"Added detector '{name}' with action mode '{action_mode}'"

    except Exception as e:
        return f"Error adding detector: {str(e)}"


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
    copilot, err = require_copilot(context)
    if err:
        return err

    from ..detector_registry import get_detector_presets

    presets = get_detector_presets()

    if preset not in presets:
        available = ", ".join(presets.keys())
        return f"Unknown preset '{preset}'. Available: {available}"

    preset_data = presets[preset]

    # Check if already exists
    existing = copilot.detector_registry.get(preset)
    if existing:
        existing.enabled = True
        copilot.detector_registry.save()
        return f"Enabled existing '{preset}' detector"

    # Create detector from preset
    conditions = DetectorConditions(min_timepoint=min_timepoint)
    actions = DetectorActions(mode=DetectionMode(action_mode))

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

    copilot.detector_registry.add(detector)
    copilot._mark_significant_action("detector_config")

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
    copilot, err = require_copilot(context)
    if err:
        return err

    try:
        prompt = await copilot._generate_detector_prompt(
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
    copilot, err = require_copilot(context)
    if err:
        return err

    detector = copilot.detector_registry.get(detector_name)
    if not detector:
        return f"Detector '{detector_name}' not found"

    embryo, err = get_embryo_or_error(copilot, embryo_id)
    if err:
        return err

    try:
        result = await copilot._run_detector_test(
            detector=detector,
            embryo_id=embryo.id,
            timepoint=timepoint
        )
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error testing detector: {str(e)}"
