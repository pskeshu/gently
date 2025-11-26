"""
Plugin-based tool implementations

This module contains tools migrated from the monolithic if/elif chain
to the new decorator-based plugin system.

Tools are automatically registered when this module is imported.
"""

from typing import Dict, List, Optional
import json

from .tool_registry import (
    tool, ToolCategory, ToolParameter,
    get_tool_registry,
)


# =============================================================================
# Experiment Management Tools
# =============================================================================

@tool(
    name="get_experiment_summary",
    description="Get a summary of the current experiment status including all embryos",
    category=ToolCategory.EXPERIMENT,
)
def get_experiment_summary(context: Dict) -> str:
    """
    Get full experiment summary

    Parameters
    ----------
    context : dict
        Execution context with copilot reference

    Returns
    -------
    str
        Experiment summary
    """
    copilot = context.get('copilot')
    if not copilot:
        return "Error: No copilot context"
    return copilot.experiment.get_summary()


@tool(
    name="query_embryo_status",
    description="Query the status of a specific embryo by ID or name",
    category=ToolCategory.EMBRYO,
)
def query_embryo_status(embryo_id: str, context: Dict) -> str:
    """
    Query embryo status

    Parameters
    ----------
    embryo_id : str
        Embryo ID, nickname, or label to query
    context : dict
        Execution context

    Returns
    -------
    str
        JSON formatted embryo status
    """
    copilot = context.get('copilot')
    if not copilot:
        return "Error: No copilot context"

    embryo = copilot.experiment.get_embryo_by_any_name(embryo_id)
    if not embryo:
        return f"Embryo '{embryo_id}' not found. Available: {list(copilot.experiment.embryos.keys())}"

    return json.dumps(embryo.to_dict(), indent=2)


@tool(
    name="skip_embryo",
    description="Mark an embryo to be skipped in future acquisitions",
    category=ToolCategory.EMBRYO,
)
def skip_embryo(embryo_id: str, reason: str, context: Dict) -> str:
    """
    Skip embryo in future acquisitions

    Parameters
    ----------
    embryo_id : str
        Embryo to skip
    reason : str
        Reason for skipping
    context : dict
        Execution context

    Returns
    -------
    str
        Confirmation message
    """
    copilot = context.get('copilot')
    if not copilot:
        return "Error: No copilot context"

    embryo = copilot.experiment.get_embryo_by_any_name(embryo_id)
    if not embryo:
        return f"Embryo '{embryo_id}' not found"

    embryo.should_skip = True
    embryo.skip_reason = reason

    return f"Marked {embryo_id} to skip. Reason: {reason}"


@tool(
    name="remove_embryo",
    description="Permanently remove an embryo from the experiment (e.g., false detection)",
    category=ToolCategory.EMBRYO,
)
def remove_embryo(embryo_id: str, context: Dict) -> str:
    """
    Remove embryo from experiment completely

    Parameters
    ----------
    embryo_id : str
        Embryo to remove
    context : dict
        Execution context

    Returns
    -------
    str
        Confirmation message
    """
    copilot = context.get('copilot')
    if not copilot:
        return "Error: No copilot context"

    # Try to find the embryo first
    embryo = copilot.experiment.get_embryo_by_any_name(embryo_id)
    if not embryo:
        return f"Embryo '{embryo_id}' not found"

    # Remove it
    actual_id = embryo.id  # Get the actual ID in case user used nickname
    if copilot.experiment.remove_embryo(actual_id):
        return f"✓ Removed {actual_id} from experiment (false detection deleted)"
    else:
        return f"Failed to remove {embryo_id}"


@tool(
    name="resume_embryo",
    description="Resume imaging a previously skipped embryo",
    category=ToolCategory.EMBRYO,
)
def resume_embryo(embryo_id: str, context: Dict) -> str:
    """
    Resume skipped embryo

    Parameters
    ----------
    embryo_id : str
        Embryo to resume
    context : dict
        Execution context

    Returns
    -------
    str
        Confirmation message
    """
    copilot = context.get('copilot')
    if not copilot:
        return "Error: No copilot context"

    embryo = copilot.experiment.get_embryo_by_any_name(embryo_id)
    if not embryo:
        return f"Embryo '{embryo_id}' not found"

    embryo.should_skip = False
    embryo.skip_reason = None

    return f"Resumed imaging {embryo_id}"


@tool(
    name="assign_nickname",
    description="Assign a memorable nickname to an embryo",
    category=ToolCategory.EMBRYO,
)
def assign_nickname(embryo_id: str, nickname: str, context: Dict) -> str:
    """
    Assign nickname to embryo

    Parameters
    ----------
    embryo_id : str
        Embryo to nickname
    nickname : str
        New nickname
    context : dict
        Execution context

    Returns
    -------
    str
        Confirmation message
    """
    copilot = context.get('copilot')
    if not copilot:
        return "Error: No copilot context"

    embryo = copilot.experiment.get_embryo_by_any_name(embryo_id)
    if not embryo:
        return f"Embryo '{embryo_id}' not found"

    old_nickname = embryo.nickname
    embryo.nickname = nickname

    if old_nickname:
        return f"Renamed {embryo_id}: '{old_nickname}' → '{nickname}'"
    else:
        return f"Nicknamed {embryo_id} as '{nickname}'"


@tool(
    name="modify_parameters",
    description="Modify acquisition parameters for a specific embryo",
    category=ToolCategory.EMBRYO,
)
def modify_parameters(
    embryo_id: str,
    changes: Dict,
    reason: str,
    context: Dict
) -> str:
    """
    Modify embryo acquisition parameters

    Parameters
    ----------
    embryo_id : str
        Embryo to modify
    changes : dict
        Parameter changes (interval_seconds, num_slices, exposure_ms, priority)
    reason : str
        Reason for the changes
    context : dict
        Execution context

    Returns
    -------
    str
        Summary of changes
    """
    copilot = context.get('copilot')
    if not copilot:
        return "Error: No copilot context"

    embryo = copilot.experiment.get_embryo_by_any_name(embryo_id)
    if not embryo:
        return f"Embryo '{embryo_id}' not found"

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

    return (f"Modified {embryo_id} parameters:\n"
            f"Reason: {reason}\n\n"
            f"Changes:\n{json.dumps(changes, indent=2)}\n\n"
            f"Previous: {json.dumps(old_params, indent=2)}")


# =============================================================================
# Hardware Control Tools (require microscope)
# =============================================================================

@tool(
    name="move_to_embryo",
    description="Move the stage to a specific embryo's position",
    category=ToolCategory.MOVEMENT,
    requires_microscope=True,
)
async def move_to_embryo(embryo_id: str, context: Dict) -> str:
    """
    Move stage to embryo position

    Parameters
    ----------
    embryo_id : str
        Embryo to move to
    context : dict
        Execution context with client

    Returns
    -------
    str
        Movement result
    """
    copilot = context.get('copilot')
    client = context.get('client')

    if not copilot:
        return "Error: No copilot context"

    embryo = copilot.experiment.get_embryo_by_any_name(embryo_id)
    if not embryo:
        return f"Embryo '{embryo_id}' not found"

    if not embryo.stage_position:
        return f"Embryo '{embryo_id}' has no stored position. Run calibration first."

    try:
        x = embryo.stage_position.get('x', 0)
        y = embryo.stage_position.get('y', 0)
        await client.move_to_position(x, y)

        return f"✓ Moved to {embryo_id}\nPosition: ({x:.2f}, {y:.2f}) µm"

    except Exception as e:
        import traceback
        return f"Error moving to embryo: {str(e)}\n{traceback.format_exc()}"


@tool(
    name="set_led",
    description="Set the LED illumination state",
    category=ToolCategory.HARDWARE,
    requires_microscope=True,
)
async def set_led(state: str, context: Dict) -> str:
    """
    Set LED state

    Parameters
    ----------
    state : str
        LED state (e.g., 'Open', 'Closed')
    context : dict
        Execution context

    Returns
    -------
    str
        Result message
    """
    client = context.get('client')

    try:
        result = await client.set_led(state)
        if result.get('success'):
            return f"LED set to '{state}'"
        else:
            return f"Error setting LED: {result.get('error', 'Unknown error')}"
    except Exception as e:
        return f"Error setting LED: {str(e)}"


@tool(
    name="get_led_status",
    description="Get the current LED illumination status",
    category=ToolCategory.HARDWARE,
    requires_microscope=True,
)
async def get_led_status(context: Dict) -> str:
    """
    Get LED status

    Parameters
    ----------
    context : dict
        Execution context

    Returns
    -------
    str
        LED status information
    """
    client = context.get('client')

    try:
        result = await client.get_led_status()
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


# =============================================================================
# Detector Management Tools
# =============================================================================

@tool(
    name="list_detectors",
    description="List all registered detectors and their status",
    category=ToolCategory.DETECTION,
)
def list_detectors(filter: str = "all", context: Dict = None) -> str:
    """
    List all detectors

    Parameters
    ----------
    filter : str
        Filter type: 'all', 'enabled', or 'disabled'
    context : dict
        Execution context

    Returns
    -------
    str
        Formatted detector list
    """
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
    """
    Enable or disable detector

    Parameters
    ----------
    detector_name : str
        Name of detector
    enabled : bool
        True to enable, False to disable
    context : dict
        Execution context

    Returns
    -------
    str
        Result message
    """
    copilot = context.get('copilot')
    if not copilot:
        return "Error: No copilot context"

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
    """
    Remove detector

    Parameters
    ----------
    detector_name : str
        Name of detector to remove
    context : dict
        Execution context

    Returns
    -------
    str
        Result message
    """
    copilot = context.get('copilot')
    if not copilot:
        return "Error: No copilot context"

    success = copilot.detector_registry.remove(detector_name)

    if success:
        copilot._mark_significant_action("detector_config")
        return f"Detector '{detector_name}' removed"
    else:
        return f"Detector '{detector_name}' not found"


# =============================================================================
# Utility function to register all tools
# =============================================================================

def register_all_tools():
    """
    Ensure all tools in this module are registered

    This is called automatically when the module is imported,
    but can also be called explicitly to re-register.
    """
    registry = get_tool_registry()
    count = len(registry)
    return f"Registered {count} tools"


# =============================================================================
# Analysis / VLM Tools
# =============================================================================

@tool(
    name="analyze_volume",
    description="Analyze an embryo volume using Claude Vision API",
    category=ToolCategory.ANALYSIS,
)
async def analyze_volume(
    embryo_id: str,
    analysis_prompt: str,
    use_recent_context: bool = False,
    timepoint: Optional[int] = None,
    context: Dict = None
) -> str:
    """Analyze embryo volume with Claude Vision"""
    copilot = context.get('copilot')
    if not copilot:
        return "Error: No copilot context"

    embryo = copilot.experiment.get_embryo_by_any_name(embryo_id)
    if not embryo:
        return f"Embryo '{embryo_id}' not found"

    try:
        result = await copilot._analyze_with_vision(
            embryo_id=embryo.embryo_id,
            prompt=analysis_prompt,
            use_context=use_recent_context,
            timepoint=timepoint
        )
        return result
    except Exception as e:
        return f"Error analyzing volume: {str(e)}"


@tool(
    name="get_detection_summary",
    description="Get summary of all detections across all embryos",
    category=ToolCategory.DETECTION,
)
def get_detection_summary(context: Dict) -> str:
    """Get detection summary"""
    copilot = context.get('copilot')
    if not copilot:
        return "Error: No copilot context"

    lines = ["Detection Summary:", ""]

    for embryo_id, embryo in copilot.experiment.embryos.items():
        if embryo.detections:
            lines.append(f"• {embryo_id}:")
            for det_name, det_info in embryo.detections.items():
                lines.append(f"  - {det_name}: detected at t={det_info.get('timepoint', '?')}")
            lines.append("")

    if len(lines) == 2:
        return "No detections recorded yet."

    return "\n".join(lines)


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
    copilot = context.get('copilot')
    if not copilot:
        return "Error: No copilot context"

    try:
        if preset:
            detector = copilot.detector_registry.create_from_preset(preset, name)
        else:
            from .detectors import Detector, DetectorConfig, DetectorActions, DetectorConditions, ActionMode

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
    """
    Enable a preset detector for adaptive experiments

    Parameters
    ----------
    preset : str
        Preset name: "hatching", "comma", "pretzel", "gastrulation", "first_division"
    action_mode : str
        Action mode: "passive" (just log), "recommend" (suggest), "auto" (apply changes)
    min_timepoint : int, optional
        Don't run detector before this timepoint (e.g., 50 to skip early timepoints)
    context : dict
        Execution context

    Returns
    -------
    str
        Confirmation message
    """
    copilot = context.get('copilot')
    if not copilot:
        return "Error: No copilot context"

    from .detector import Detector, DetectorConditions, DetectorActions, DetectionMode
    from .detector_registry import get_detector_presets

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

    from .detector import ConfidenceLevel
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
    copilot = context.get('copilot')
    if not copilot:
        return "Error: No copilot context"

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
    copilot = context.get('copilot')
    if not copilot:
        return "Error: No copilot context"

    detector = copilot.detector_registry.get(detector_name)
    if not detector:
        return f"Detector '{detector_name}' not found"

    embryo = copilot.experiment.get_embryo_by_any_name(embryo_id)
    if not embryo:
        return f"Embryo '{embryo_id}' not found"

    try:
        result = await copilot._run_detector_test(
            detector=detector,
            embryo_id=embryo.embryo_id,
            timepoint=timepoint
        )
        return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error testing detector: {str(e)}"


# =============================================================================
# Hardware Control Tools
# =============================================================================

@tool(
    name="calibrate_embryo",
    description="Run piezo-galvo calibration for a specific embryo. Moves to embryo position first, then calibrates.",
    category=ToolCategory.HARDWARE,
    requires_microscope=True,
)
async def calibrate_embryo(
    embryo_id: str,
    piezo_positions: List[float] = None,
    context: Dict = None
) -> str:
    """Calibrate embryo piezo-galvo - moves to embryo first, then calibrates"""
    copilot = context.get('copilot')
    client = context.get('client')

    if not copilot:
        return "Error: No copilot context"

    embryo = copilot.experiment.get_embryo_by_any_name(embryo_id)
    if not embryo:
        return f"Embryo '{embryo_id}' not found"

    positions = piezo_positions or [40.0, 60.0]

    try:
        # First move to embryo position
        pos = embryo.stage_position
        if pos and pos.get('x') is not None and pos.get('y') is not None:
            print(f"  Moving to {embryo.id} at ({pos['x']:.1f}, {pos['y']:.1f})...")
            await client.move_to_position(pos['x'], pos['y'])
        else:
            print(f"  Warning: No position stored for {embryo.id}, calibrating at current position")

        # Run calibration at current position
        print(f"  Running piezo-galvo calibration...")
        result = await client.calibrate_piezo_galvo(piezo_positions=positions)

        if result.get('success'):
            embryo.calibration = result.get('calibration', {})
            copilot._mark_significant_action("calibration")
            return f"✓ Calibrated {embryo.id}\nCalibration: {json.dumps(result.get('calibration', {}), indent=2)}"
        else:
            return f"Calibration failed: {result.get('error', 'Unknown error')}"

    except Exception as e:
        return f"Error calibrating embryo: {str(e)}"


@tool(
    name="acquire_volume",
    description="Acquire a single 3D volume for a specific embryo. Moves to embryo position and uses calibration data.",
    category=ToolCategory.HARDWARE,
    requires_microscope=True,
)
async def acquire_volume(
    embryo_id: str,
    num_slices: int = 50,
    exposure_ms: float = 10.0,
    context: Dict = None
) -> str:
    """Acquire single volume - moves to embryo first, uses calibration"""
    copilot = context.get('copilot')
    client = context.get('client')

    if not copilot:
        return "Error: No copilot context"

    embryo = copilot.experiment.get_embryo_by_any_name(embryo_id)
    if not embryo:
        return f"Embryo '{embryo_id}' not found"

    try:
        # Move to embryo position first
        pos = embryo.stage_position
        if pos and pos.get('x') is not None and pos.get('y') is not None:
            print(f"  Moving to {embryo.id} at ({pos['x']:.1f}, {pos['y']:.1f})...")
            await client.move_to_position(pos['x'], pos['y'])
        else:
            print(f"  Warning: No position stored for {embryo.id}, acquiring at current position")

        # Get calibration parameters (use defaults if not calibrated)
        cal = embryo.calibration or {}
        galvo_amplitude = cal.get('galvo_amplitude', 0.5)
        galvo_center = cal.get('galvo_center', 0.0)
        piezo_amplitude = cal.get('piezo_amplitude', 25.0)
        piezo_center = cal.get('piezo_center', 50.0)

        if not embryo.calibration:
            print(f"  Warning: {embryo.id} not calibrated, using default parameters")

        print(f"  Acquiring {num_slices}-slice volume...")
        result = await client.acquire_volume(
            num_slices=num_slices,
            exposure_ms=exposure_ms,
            galvo_amplitude=galvo_amplitude,
            galvo_center=galvo_center,
            piezo_amplitude=piezo_amplitude,
            piezo_center=piezo_center
        )

        if result.get('success'):
            # Update embryo state
            embryo.timepoints_acquired += 1
            from datetime import datetime
            embryo.last_imaged = datetime.now()
            return f"✓ Acquired volume for {embryo.id}\nShape: {result.get('shape', 'unknown')}"
        else:
            return f"Acquisition failed: {result.get('error', 'Unknown error')}"

    except Exception as e:
        return f"Error acquiring volume: {str(e)}"


@tool(
    name="start_multi_embryo_timelapse",
    description="Start multi-embryo time-lapse volume acquisition (NOT YET IMPLEMENTED)",
    category=ToolCategory.HARDWARE,
    requires_microscope=True,
)
async def start_multi_embryo_timelapse(
    embryo_ids: List[str] = None,
    num_timepoints: int = 500,
    interval_seconds: float = 120,
    num_slices: int = 50,
    exposure_ms: float = 10.0,
    enable_detectors: bool = True,
    context: Dict = None
) -> str:
    """Start multi-embryo timelapse - NOT YET IMPLEMENTED"""
    # TODO: Implement timelapse plan in backend
    return ("⚠ Multi-embryo timelapse is not yet implemented.\n"
            "For now, you can:\n"
            "  - Use acquire_volume to capture single volumes\n"
            "  - Manually repeat acquisitions at intervals")


@tool(
    name="pause_acquisition",
    description="Pause currently running acquisition (NOT YET IMPLEMENTED)",
    category=ToolCategory.HARDWARE,
    requires_microscope=True,
)
async def pause_acquisition(context: Dict = None) -> str:
    """Pause acquisition - NOT YET IMPLEMENTED"""
    # TODO: Implement pause in backend
    return "⚠ Pause acquisition is not yet implemented."


@tool(
    name="resume_acquisition",
    description="Resume previously paused acquisition (NOT YET IMPLEMENTED)",
    category=ToolCategory.HARDWARE,
    requires_microscope=True,
)
async def resume_acquisition(context: Dict = None) -> str:
    """Resume acquisition - NOT YET IMPLEMENTED"""
    # TODO: Implement resume in backend
    return "⚠ Resume acquisition is not yet implemented."


@tool(
    name="view_image",
    description="Capture and display the current bottom camera image (widefield view)",
    category=ToolCategory.HARDWARE,
    requires_microscope=True,
)
async def view_image(
    title: str = "Bottom Camera Image",
    exposure_ms: float = None,
    show: bool = True,
    context: Dict = None
) -> str:
    """Capture and display bottom camera image"""
    client = context.get('client')

    try:
        print(f"  Capturing bottom camera image...")
        image = await client.capture_bottom_image(exposure_ms=exposure_ms)

        if image is None or image.shape == (100, 100):
            return "Failed to capture image from bottom camera"

        if show:
            from datetime import datetime
            from pathlib import Path
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"camera_captures/bottom_camera_{timestamp}.png"
            Path("camera_captures").mkdir(exist_ok=True)

            print(f"  Displaying image...")
            view_result = await client.view_image(
                image=image,
                title=title,
                save_path=save_path,
                show=True
            )
            return f"✓ Captured bottom camera image ({image.shape[0]}x{image.shape[1]})\nSaved to: {save_path}"
        else:
            return f"✓ Captured bottom camera image ({image.shape[0]}x{image.shape[1]})"

    except Exception as e:
        return f"Error capturing image: {str(e)}"


@tool(
    name="capture_lightsheet",
    description="Capture and display a single 2D lightsheet image (one slice only). This is a COMPLETE action - do NOT follow up with acquire_volume unless explicitly asked for a 3D volume.",
    category=ToolCategory.HARDWARE,
    requires_microscope=True,
)
async def capture_lightsheet(
    piezo_position: float = 50.0,
    galvo_position: float = 0.0,
    show: bool = True,
    context: Dict = None
) -> str:
    """Capture and optionally display a single lightsheet image"""
    client = context.get('client')

    try:
        print(f"  Capturing lightsheet at piezo={piezo_position}µm, galvo={galvo_position}V...")
        result = await client.capture_lightsheet_image(
            piezo_position=piezo_position,
            galvo_position=galvo_position
        )

        if result.get('success'):
            image = result.get('image')
            run_uid = result.get('run_uid', 'unknown')

            if image is not None and show:
                # Display the image
                from datetime import datetime
                from pathlib import Path
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = f"lightsheet_captures/lightsheet_{timestamp}.png"
                Path("lightsheet_captures").mkdir(exist_ok=True)

                print(f"  Displaying lightsheet image...")
                view_result = await client.view_image(
                    image=image,
                    title=f"Lightsheet: piezo={piezo_position}µm, galvo={galvo_position}V",
                    save_path=save_path,
                    show=True
                )
                return f"✓ Captured lightsheet image at piezo={piezo_position}µm, galvo={galvo_position}V\nSaved to: {save_path}\nRun UID: {run_uid}"
            elif image is None:
                # Plan succeeded but image couldn't be retrieved
                return f"✓ Lightsheet captured at piezo={piezo_position}µm, galvo={galvo_position}V (image not displayed - databroker retrieval issue)\nRun UID: {run_uid}"
            else:
                return f"✓ Captured lightsheet at piezo={piezo_position}µm, galvo={galvo_position}V\nRun UID: {run_uid}"
        else:
            return f"Failed: {result.get('error', 'Unknown error')}"

    except Exception as e:
        return f"Error capturing lightsheet: {str(e)}"


# =============================================================================
# Detection Tools
# =============================================================================

@tool(
    name="detect_embryos",
    description="Automatically detect embryos using brightness detection + SAM refinement",
    category=ToolCategory.DETECTION,
    requires_microscope=True,
)
async def detect_embryos(
    auto_calibrate: bool = False,
    min_confidence: float = 0.7,
    use_claude_review: bool = False,
    exposure_ms: float = None,
    brightness_percentile: float = 99.0,
    min_area: int = 5000,
    max_area: int = 150000,
    context: Dict = None
) -> str:
    """Detect embryos automatically"""
    copilot = context.get('copilot')
    client = context.get('client')

    if not copilot:
        return "Error: No copilot context"

    try:
        result = await client.detect_embryos(
            min_confidence=min_confidence,
            use_claude_review=use_claude_review,
            exposure_ms=exposure_ms,
            brightness_percentile=brightness_percentile,
            min_area=min_area,
            max_area=max_area
        )

        if result.get('success'):
            embryos = result.get('embryos', [])

            # Store result for show_detected_embryos
            copilot.last_detection_result = result

            # Add to experiment
            for emb in embryos:
                # Build position dict from stage coordinates
                position = {
                    'x': emb.get('stage_x_um', emb.get('stage_x', 0)),
                    'y': emb.get('stage_y_um', emb.get('stage_y', 0))
                }
                copilot.experiment.add_embryo(
                    embryo_id=emb['embryo_id'],
                    position=position,
                    confidence=emb.get('confidence', 0.0)
                )

            if auto_calibrate and embryos:
                return f"✓ Detected {len(embryos)} embryos. Starting calibration..."
            else:
                return f"✓ Detected {len(embryos)} embryos\nUse show_detected_embryos to visualize."
        else:
            return f"Detection failed: {result.get('error', 'Unknown error')}"

    except Exception as e:
        return f"Error detecting embryos: {str(e)}"


@tool(
    name="manual_mark_embryos",
    description="Manually mark additional embryos by clicking on an image. New embryos get unique IDs that don't conflict with existing ones.",
    category=ToolCategory.DETECTION,
    requires_microscope=True,
)
async def manual_mark_embryos(
    exposure_ms: float = None,
    context: Dict = None
) -> str:
    """Manual embryo marking - adds to existing embryos with unique IDs"""
    copilot = context.get('copilot')
    client = context.get('client')

    if not copilot:
        return "Error: No copilot context"

    try:
        result = await client.manual_mark_embryos(exposure_ms=exposure_ms)

        if result.get('success'):
            embryos = result.get('embryos', [])

            if not embryos:
                return "No embryos marked. Close the window after clicking on embryo centers."

            # Find next available embryo ID
            existing_ids = set(copilot.experiment.embryos.keys())
            max_num = 0
            for eid in existing_ids:
                # Extract number from embryo_N format
                if eid.startswith('embryo_'):
                    try:
                        num = int(eid.replace('embryo_', ''))
                        max_num = max(max_num, num)
                    except ValueError:
                        pass
            next_num = max_num + 1

            # Assign new unique IDs and add to experiment
            added_ids = []
            for emb in embryos:
                new_id = f'embryo_{next_num}'
                next_num += 1

                # Build position dict from stage coordinates
                position = {
                    'x': emb.get('stage_x_um', emb.get('stage_x', 0)),
                    'y': emb.get('stage_y_um', emb.get('stage_y', 0))
                }

                # Update embryo dict with new ID for visualization
                emb['embryo_id'] = new_id

                copilot.experiment.add_embryo(
                    embryo_id=new_id,
                    position=position,
                    confidence=emb.get('confidence', 1.0)  # Manual = high confidence
                )
                added_ids.append(new_id)

            # Store result for show_detected_embryos (merge with existing if any)
            if copilot.last_detection_result and copilot.last_detection_result.get('embryos'):
                # Merge new embryos into existing detection result
                copilot.last_detection_result['embryos'].extend(embryos)
            else:
                copilot.last_detection_result = result

            return f"✓ Added {len(added_ids)} embryo(s): {', '.join(added_ids)}"
        else:
            return f"Marking failed: {result.get('error', 'Unknown error')}"

    except Exception as e:
        return f"Error: {str(e)}"


@tool(
    name="show_detected_embryos",
    description="Show detected embryos with bounding boxes. Shows only active (non-removed) embryos. Works with resumed sessions by capturing fresh image.",
    category=ToolCategory.DETECTION,
    requires_microscope=True,
)
async def show_detected_embryos(
    only_active: bool = True,
    save_to_file: bool = True,
    context: Dict = None
) -> str:
    """Show detected embryos visualization, filtered to active embryos"""
    copilot = context.get('copilot')
    client = context.get('client')

    if not copilot:
        return "Error: No copilot context"

    # Get active embryo IDs from experiment
    active_ids = set(copilot.experiment.embryos.keys()) if copilot.experiment.embryos else set()

    if not active_ids:
        return "No embryos in experiment. Run detect_embryos first."

    try:
        # Check if we have detection result with all active embryos
        result = copilot.last_detection_result
        have_all_embryos = False

        if result and result.get('embryos') and result.get('image') is not None:
            detection_ids = {e.get('embryo_id') for e in result.get('embryos', [])}
            have_all_embryos = active_ids.issubset(detection_ids)

        # If detection result is missing or incomplete, capture fresh image and calculate positions
        if not have_all_embryos:
            print(f"  Capturing fresh image for visualization...")
            image = await client.capture_bottom_image()
            if image is None or image.shape == (100, 100):
                return "Failed to capture image for visualization."

            print(f"  Reading current stage position...")
            current_stage = await client.get_stage_position()

            # Calculate pixel positions for all active embryos
            # Formula: pixel = image_center + (embryo_stage - current_stage) / um_per_pixel
            pixel_size_um = 6.5
            objective_mag = 4.0
            um_per_pixel = pixel_size_um / objective_mag  # 1.625 um/pixel

            image_center_x = image.shape[1] / 2
            image_center_y = image.shape[0] / 2

            embryos = []
            for embryo_id in active_ids:
                embryo_state = copilot.experiment.embryos[embryo_id]
                pos = embryo_state.stage_position or {}

                stage_x = pos.get('x', current_stage[0])
                stage_y = pos.get('y', current_stage[1])

                # Convert stage position to pixel position
                dx_stage = stage_x - current_stage[0]
                dy_stage = stage_y - current_stage[1]

                pixel_x = image_center_x + dx_stage / um_per_pixel
                pixel_y = image_center_y + dy_stage / um_per_pixel

                embryos.append({
                    'embryo_id': embryo_id,
                    'pixel_x': pixel_x,
                    'pixel_y': pixel_y,
                    'stage_x_um': stage_x,
                    'stage_y_um': stage_y,
                    'confidence': embryo_state.detection_confidence,
                })

            # Update detection result for future use
            copilot.last_detection_result = {
                'image': image,
                'embryos': embryos,
                'stage_position': list(current_stage),
                'success': True
            }
            result = copilot.last_detection_result
            print(f"  Calculated positions for {len(embryos)} embryos")
        else:
            # Filter existing detection result to active embryos
            embryos = [e for e in result.get('embryos', []) if e.get('embryo_id') in active_ids]
            image = result.get('image')

        if not embryos:
            return f"No embryos to display. Active: {', '.join(active_ids)}"

        # Generate save path
        from datetime import datetime
        from pathlib import Path
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = f"detection_results/detected_embryos_{timestamp}.png"
        Path("detection_results").mkdir(exist_ok=True)

        print(f"  Showing {len(embryos)} embryos with bounding boxes...")

        # Use client to view with filtered embryos
        view_result = await client.view_embryos(
            image=image,
            embryos=embryos,
            title=f"Active Embryos ({len(embryos)})",
            save_path=save_path,
            show=True
        )

        if view_result.get('success'):
            embryo_ids = [e.get('embryo_id', '?') for e in embryos]
            return f"✓ Showing {len(embryos)} active embryos: {', '.join(embryo_ids)}\nSaved to: {save_path}"
        elif view_result.get('error'):
            return f"Display error: {view_result.get('error')}"
        else:
            return f"✓ Visualization complete. Check {save_path}"

    except Exception as e:
        return f"Error showing detections: {str(e)}"


# =============================================================================
# Databroker Tools
# =============================================================================

@tool(
    name="list_runs",
    description="List recent Bluesky runs from Databroker",
    category=ToolCategory.DATA,
)
def list_runs(
    limit: int = 10,
    embryo_id: str = None,
    plan_name: str = None,
    context: Dict = None
) -> str:
    """List recent runs"""
    copilot = context.get('copilot')

    if not copilot or not copilot.databroker:
        return "Databroker not available"

    try:
        db = copilot.databroker

        # Build query
        query = {}
        if embryo_id:
            query['embryo_id'] = embryo_id
        if plan_name:
            query['plan_name'] = plan_name

        runs = list(db(**query))[:limit]

        if not runs:
            return "No runs found"

        lines = [f"Recent runs ({len(runs)}):", ""]

        for run_uid in runs:
            run = db[run_uid]
            start = run.metadata.get('start', {})
            lines.append(f"• {run_uid[:8]}...")
            lines.append(f"  Plan: {start.get('plan_name', 'unknown')}")
            lines.append(f"  Time: {start.get('time', 'unknown')}")
            if 'embryo_id' in start:
                lines.append(f"  Embryo: {start['embryo_id']}")
            lines.append("")

        return "\n".join(lines)

    except Exception as e:
        return f"Error listing runs: {str(e)}"


@tool(
    name="get_run_data",
    description="Get data from a specific Bluesky run",
    category=ToolCategory.DATA,
)
def get_run_data(
    run_id: str,
    data_keys: List[str] = None,
    stream: str = "primary",
    context: Dict = None
) -> str:
    """Get run data"""
    copilot = context.get('copilot')

    if not copilot or not copilot.databroker:
        return "Databroker not available"

    try:
        db = copilot.databroker

        # Handle relative indices
        if run_id.startswith('-'):
            run = db[int(run_id)]
        else:
            run = db[run_id]

        # Get data
        data = run.primary.read()

        if data_keys:
            data = {k: data[k] for k in data_keys if k in data}

        # Format output
        lines = [f"Run: {run.metadata['start']['uid'][:8]}...", ""]
        lines.append(f"Available keys: {list(data.keys())}")

        for key, values in data.items():
            shape = values.shape if hasattr(values, 'shape') else 'scalar'
            lines.append(f"  {key}: shape={shape}")

        return "\n".join(lines)

    except Exception as e:
        return f"Error getting run data: {str(e)}"


@tool(
    name="get_run_image",
    description="Get an image from a Bluesky run for analysis",
    category=ToolCategory.DATA,
)
async def get_run_image(
    run_id: str,
    detector: str = None,
    analyze: bool = False,
    analysis_prompt: str = None,
    context: Dict = None
) -> str:
    """Get run image"""
    copilot = context.get('copilot')

    if not copilot or not copilot.databroker:
        return "Databroker not available"

    try:
        db = copilot.databroker

        if run_id.startswith('-'):
            run = db[int(run_id)]
        else:
            run = db[run_id]

        # Get image data
        data = run.primary.read()

        # Auto-detect detector if not specified
        if not detector:
            for key in ['bottom_camera', 'camera', 'detector']:
                if key in data:
                    detector = key
                    break

        if detector not in data:
            return f"Detector '{detector}' not found. Available: {list(data.keys())}"

        image = data[detector]
        shape = image.shape if hasattr(image, 'shape') else 'unknown'

        result = f"✓ Retrieved image from {detector}\nShape: {shape}"

        if analyze and analysis_prompt:
            # Run VLM analysis
            analysis = await copilot._analyze_image_with_vision(
                image=image,
                prompt=analysis_prompt
            )
            result += f"\n\nAnalysis:\n{analysis}"

        return result

    except Exception as e:
        return f"Error getting image: {str(e)}"


@tool(
    name="search_runs",
    description="Search Databroker runs by metadata criteria",
    category=ToolCategory.DATA,
)
def search_runs(
    since: str = None,
    until: str = None,
    metadata: Dict = None,
    limit: int = 20,
    context: Dict = None
) -> str:
    """Search runs"""
    copilot = context.get('copilot')

    if not copilot or not copilot.databroker:
        return "Databroker not available"

    try:
        db = copilot.databroker

        # Build query
        query = metadata or {}

        if since:
            query['since'] = since
        if until:
            query['until'] = until

        runs = list(db(**query))[:limit]

        if not runs:
            return "No matching runs found"

        lines = [f"Found {len(runs)} runs:", ""]

        for run_uid in runs:
            run = db[run_uid]
            start = run.metadata.get('start', {})
            lines.append(f"• {run_uid[:8]}: {start.get('plan_name', 'unknown')}")

        return "\n".join(lines)

    except Exception as e:
        return f"Error searching runs: {str(e)}"


# =============================================================================
# Plan Generation Tools
# =============================================================================

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


# =============================================================================
# Timelapse Orchestration Tools
# =============================================================================

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
    """
    Start adaptive timelapse in background

    Parameters
    ----------
    embryo_ids : list of str, optional
        Embryos to image (None = all active embryos)
    stop_condition : str
        When to stop: "manual", "hatching", "comma", "timepoints:N", "duration:Xh"
    interval_seconds : float
        Default interval between acquisitions
    condition_value : int, optional
        Value for stop condition (e.g., number of timepoints)
    context : dict
        Execution context

    Returns
    -------
    str
        Status message
    """
    copilot = context.get('copilot')
    if not copilot:
        return "Error: No copilot context"

    # Check if orchestrator exists
    if not hasattr(copilot, 'timelapse_orchestrator') or copilot.timelapse_orchestrator is None:
        return "Error: Timelapse orchestrator not initialized. Check microscope connection."

    orchestrator = copilot.timelapse_orchestrator

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
    """
    Get timelapse status

    Parameters
    ----------
    context : dict
        Execution context

    Returns
    -------
    str
        Formatted status information
    """
    copilot = context.get('copilot')
    if not copilot:
        return "Error: No copilot context"

    if not hasattr(copilot, 'timelapse_orchestrator') or copilot.timelapse_orchestrator is None:
        return "Timelapse orchestrator not initialized."

    orchestrator = copilot.timelapse_orchestrator
    state = orchestrator.get_status()
    status_dict = state.to_dict()

    # Format nicely
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

    # Per-embryo details
    if status_dict['embryo_details']:
        lines.append("Embryo Details:")
        for eid, details in status_dict['embryo_details'].items():
            status_marker = "✓" if details['is_complete'] else "▶"
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
    description="Modify parameters for a specific embryo during a running timelapse",
    category=ToolCategory.EXPERIMENT,
    requires_microscope=True,
)
async def modify_timelapse_embryo(
    embryo_id: str,
    interval_seconds: float = None,
    stop_condition: str = None,
    condition_value: int = None,
    context: Dict = None
) -> str:
    """
    Modify embryo parameters during timelapse

    Parameters
    ----------
    embryo_id : str
        Embryo to modify
    interval_seconds : float, optional
        New interval between acquisitions
    stop_condition : str, optional
        New stop condition
    condition_value : int, optional
        Value for stop condition
    context : dict
        Execution context

    Returns
    -------
    str
        Confirmation message
    """
    copilot = context.get('copilot')
    if not copilot:
        return "Error: No copilot context"

    if not hasattr(copilot, 'timelapse_orchestrator') or copilot.timelapse_orchestrator is None:
        return "No timelapse running."

    orchestrator = copilot.timelapse_orchestrator

    try:
        result = await orchestrator.modify_embryo(
            embryo_id=embryo_id,
            interval_seconds=interval_seconds,
            stop_condition=stop_condition,
            condition_value=condition_value,
        )
        return result
    except Exception as e:
        return f"Error modifying embryo: {str(e)}"


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
    """
    Stop imaging a specific embryo

    Parameters
    ----------
    embryo_id : str
        Embryo to stop
    reason : str
        Reason for stopping
    context : dict
        Execution context

    Returns
    -------
    str
        Confirmation message
    """
    copilot = context.get('copilot')
    if not copilot:
        return "Error: No copilot context"

    if not hasattr(copilot, 'timelapse_orchestrator') or copilot.timelapse_orchestrator is None:
        return "No timelapse running."

    orchestrator = copilot.timelapse_orchestrator

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
    """
    Stop entire timelapse

    Parameters
    ----------
    reason : str
        Reason for stopping
    context : dict
        Execution context

    Returns
    -------
    str
        Confirmation message
    """
    copilot = context.get('copilot')
    if not copilot:
        return "Error: No copilot context"

    if not hasattr(copilot, 'timelapse_orchestrator') or copilot.timelapse_orchestrator is None:
        return "No timelapse running."

    orchestrator = copilot.timelapse_orchestrator

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
    copilot = context.get('copilot')
    if not copilot:
        return "Error: No copilot context"

    if not hasattr(copilot, 'timelapse_orchestrator') or copilot.timelapse_orchestrator is None:
        return "No timelapse running."

    try:
        result = await copilot.timelapse_orchestrator.pause()
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
    copilot = context.get('copilot')
    if not copilot:
        return "Error: No copilot context"

    if not hasattr(copilot, 'timelapse_orchestrator') or copilot.timelapse_orchestrator is None:
        return "No timelapse to resume."

    try:
        result = await copilot.timelapse_orchestrator.resume()
        return result
    except Exception as e:
        return f"Error resuming timelapse: {str(e)}"


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
    """
    Add interval speedup rule

    Parameters
    ----------
    trigger_detector : str
        Detector that triggers speedup (e.g., "pretzel", "comma")
    new_interval_seconds : float
        New interval when triggered
    embryo_ids : list, optional
        Only apply to these embryos (None = all)
    context : dict
        Execution context

    Returns
    -------
    str
        Confirmation message
    """
    copilot = context.get('copilot')
    if not copilot:
        return "Error: No copilot context"

    if not hasattr(copilot, 'timelapse_orchestrator') or copilot.timelapse_orchestrator is None:
        return "Timelapse orchestrator not initialized."

    orchestrator = copilot.timelapse_orchestrator
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
    """
    Enable pre-hatching speedup

    Parameters
    ----------
    fast_interval_seconds : float
        Interval to use after pretzel detection
    context : dict
        Execution context

    Returns
    -------
    str
        Confirmation message
    """
    copilot = context.get('copilot')
    if not copilot:
        return "Error: No copilot context"

    if not hasattr(copilot, 'timelapse_orchestrator') or copilot.timelapse_orchestrator is None:
        return "Timelapse orchestrator not initialized."

    # Enable pretzel detector if not already enabled
    from .detector import Detector, DetectorConditions, DetectorActions, DetectionMode, ConfidenceLevel
    from .detector_registry import get_detector_presets

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

    # Add the speedup rule
    copilot.timelapse_orchestrator.add_pre_hatching_speedup(fast_interval_seconds)

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
    """
    Classify embryo stage

    Parameters
    ----------
    embryo_id : str
        Embryo to classify
    context : dict
        Execution context

    Returns
    -------
    str
        Stage classification result
    """
    copilot = context.get('copilot')
    if not copilot:
        return "Error: No copilot context"

    # Get embryo
    embryo = copilot.experiment.get_embryo_by_any_name(embryo_id)
    if not embryo:
        return f"Embryo '{embryo_id}' not found"

    # Check for recent images
    if not embryo.recent_images:
        return f"No images available for {embryo_id}. Acquire a volume first."

    # Get latest image
    latest = embryo.recent_images[-1]

    # Initialize tracker if needed
    if not hasattr(copilot, 'developmental_tracker') or copilot.developmental_tracker is None:
        from .developmental_tracker import DevelopmentalTracker
        copilot.developmental_tracker = DevelopmentalTracker(
            claude_client=copilot.claude,
            model=copilot.model,
        )

    # Get recent images for context
    recent = []
    for img in embryo.recent_images[-5:]:
        recent.append({
            'timepoint': img.timepoint,
            'b64_image': img.max_projection_b64,
        })

    # Classify
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
    """
    Get stage history

    Parameters
    ----------
    embryo_id : str
        Embryo to query
    context : dict
        Execution context

    Returns
    -------
    str
        Stage history summary
    """
    copilot = context.get('copilot')
    if not copilot:
        return "Error: No copilot context"

    if not hasattr(copilot, 'developmental_tracker') or copilot.developmental_tracker is None:
        return "No stage classifications recorded yet. Use classify_embryo_stage first."

    summary = copilot.developmental_tracker.get_progression_summary(embryo_id)

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


# Auto-register on import (tools are registered by decorators)
_registered = register_all_tools()
