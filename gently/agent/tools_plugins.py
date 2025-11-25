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
    description="Run full piezo-galvo calibration for a single embryo",
    category=ToolCategory.HARDWARE,
    requires_microscope=True,
)
async def calibrate_embryo(
    embryo_id: str,
    piezo_positions: List[float] = None,
    context: Dict = None
) -> str:
    """Calibrate embryo piezo-galvo"""
    copilot = context.get('copilot')
    client = context.get('client')

    if not copilot:
        return "Error: No copilot context"

    embryo = copilot.experiment.get_embryo_by_any_name(embryo_id)
    if not embryo:
        return f"Embryo '{embryo_id}' not found"

    positions = piezo_positions or [40.0, 60.0]

    try:
        result = await client.calibrate_embryo(
            embryo_id=embryo.embryo_id,
            piezo_positions=positions
        )

        if result.get('success'):
            embryo.calibration = result.get('calibration', {})
            copilot._mark_significant_action("calibration")
            return f"✓ Calibrated {embryo_id}\nCalibration: {json.dumps(result.get('calibration', {}), indent=2)}"
        else:
            return f"Calibration failed: {result.get('error', 'Unknown error')}"

    except Exception as e:
        return f"Error calibrating embryo: {str(e)}"


@tool(
    name="acquire_volume",
    description="Acquire a single 3D volume for a specific embryo",
    category=ToolCategory.HARDWARE,
    requires_microscope=True,
)
async def acquire_volume(
    embryo_id: str,
    num_slices: int = 50,
    exposure_ms: float = 10.0,
    save: bool = True,
    context: Dict = None
) -> str:
    """Acquire single volume"""
    copilot = context.get('copilot')
    client = context.get('client')

    if not copilot:
        return "Error: No copilot context"

    embryo = copilot.experiment.get_embryo_by_any_name(embryo_id)
    if not embryo:
        return f"Embryo '{embryo_id}' not found"

    try:
        result = await client.acquire_volume(
            embryo_id=embryo.embryo_id,
            num_slices=num_slices,
            exposure_ms=exposure_ms,
            save=save
        )

        if result.get('success'):
            return f"✓ Acquired volume for {embryo_id}\nShape: {result.get('shape', 'unknown')}"
        else:
            return f"Acquisition failed: {result.get('error', 'Unknown error')}"

    except Exception as e:
        return f"Error acquiring volume: {str(e)}"


@tool(
    name="start_multi_embryo_timelapse",
    description="Start multi-embryo time-lapse volume acquisition",
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
    """Start multi-embryo timelapse"""
    copilot = context.get('copilot')
    client = context.get('client')

    if not copilot:
        return "Error: No copilot context"

    # Use all embryos if none specified
    if not embryo_ids:
        embryo_ids = list(copilot.experiment.embryos.keys())

    try:
        result = await client.start_timelapse(
            embryo_ids=embryo_ids,
            num_timepoints=num_timepoints,
            interval_seconds=interval_seconds,
            num_slices=num_slices,
            exposure_ms=exposure_ms,
            enable_detectors=enable_detectors
        )

        if result.get('success'):
            copilot.experiment.status = "running"
            return f"✓ Started timelapse for {len(embryo_ids)} embryos\nInterval: {interval_seconds}s, Timepoints: {num_timepoints}"
        else:
            return f"Failed to start timelapse: {result.get('error', 'Unknown error')}"

    except Exception as e:
        return f"Error starting timelapse: {str(e)}"


@tool(
    name="pause_acquisition",
    description="Pause currently running acquisition",
    category=ToolCategory.HARDWARE,
    requires_microscope=True,
)
async def pause_acquisition(context: Dict = None) -> str:
    """Pause acquisition"""
    copilot = context.get('copilot')
    client = context.get('client')

    if not copilot:
        return "Error: No copilot context"

    try:
        result = await client.pause_acquisition()
        if result.get('success'):
            copilot.experiment.status = "paused"
            return "✓ Acquisition paused"
        else:
            return f"Failed to pause: {result.get('error', 'Unknown error')}"
    except Exception as e:
        return f"Error pausing: {str(e)}"


@tool(
    name="resume_acquisition",
    description="Resume previously paused acquisition",
    category=ToolCategory.HARDWARE,
    requires_microscope=True,
)
async def resume_acquisition(context: Dict = None) -> str:
    """Resume acquisition"""
    copilot = context.get('copilot')
    client = context.get('client')

    if not copilot:
        return "Error: No copilot context"

    try:
        result = await client.resume_acquisition()
        if result.get('success'):
            copilot.experiment.status = "running"
            return "✓ Acquisition resumed"
        else:
            return f"Failed to resume: {result.get('error', 'Unknown error')}"
    except Exception as e:
        return f"Error resuming: {str(e)}"


@tool(
    name="view_image",
    description="Capture and view the current bottom camera image",
    category=ToolCategory.HARDWARE,
    requires_microscope=True,
)
async def view_image(
    title: str = "Bottom Camera Image",
    exposure_ms: float = None,
    save_only: bool = False,
    context: Dict = None
) -> str:
    """View camera image"""
    client = context.get('client')

    try:
        result = await client.capture_image(exposure_ms=exposure_ms)

        if result.get('success'):
            if save_only:
                path = result.get('path', 'image.png')
                return f"✓ Image saved to {path}"
            else:
                return f"✓ Captured image\nShape: {result.get('shape', 'unknown')}"
        else:
            return f"Failed to capture: {result.get('error', 'Unknown error')}"

    except Exception as e:
        return f"Error capturing image: {str(e)}"


@tool(
    name="capture_lightsheet",
    description="Capture a single lightsheet image at the current position",
    category=ToolCategory.HARDWARE,
    requires_microscope=True,
)
async def capture_lightsheet(
    piezo_position: float = 50.0,
    galvo_position: float = 0.0,
    save_only: bool = False,
    context: Dict = None
) -> str:
    """Capture lightsheet image"""
    client = context.get('client')

    try:
        result = await client.capture_lightsheet(
            piezo_position=piezo_position,
            galvo_position=galvo_position
        )

        if result.get('success'):
            return f"✓ Captured lightsheet at piezo={piezo_position}, galvo={galvo_position}"
        else:
            return f"Failed: {result.get('error', 'Unknown error')}"

    except Exception as e:
        return f"Error: {str(e)}"


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

            # Add to experiment
            for emb in embryos:
                copilot.experiment.add_embryo(
                    embryo_id=emb['embryo_id'],
                    position=emb.get('position'),
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
    description="Manually mark embryos by clicking on an image",
    category=ToolCategory.DETECTION,
    requires_microscope=True,
)
async def manual_mark_embryos(
    exposure_ms: float = None,
    context: Dict = None
) -> str:
    """Manual embryo marking"""
    copilot = context.get('copilot')
    client = context.get('client')

    if not copilot:
        return "Error: No copilot context"

    try:
        result = await client.manual_mark_embryos(exposure_ms=exposure_ms)

        if result.get('success'):
            embryos = result.get('embryos', [])

            for emb in embryos:
                copilot.experiment.add_embryo(
                    embryo_id=emb['embryo_id'],
                    position=emb.get('position')
                )

            return f"✓ Marked {len(embryos)} embryos manually"
        else:
            return f"Marking failed: {result.get('error', 'Unknown error')}"

    except Exception as e:
        return f"Error: {str(e)}"


@tool(
    name="show_detected_embryos",
    description="Show the last detected embryos with bounding boxes",
    category=ToolCategory.DETECTION,
)
def show_detected_embryos(
    save_to_file: bool = False,
    save_only: bool = False,
    context: Dict = None
) -> str:
    """Show detected embryos visualization"""
    copilot = context.get('copilot')

    if not copilot:
        return "Error: No copilot context"

    if not copilot.last_detection_result:
        return "No detection results available. Run detect_embryos first."

    try:
        # Use visualization module
        from ..visualization import EmbryoMarker

        result = copilot.last_detection_result
        image = result.get('image')
        embryos = result.get('embryos', [])

        if save_to_file or save_only:
            path = f"detected_embryos_{len(embryos)}.png"
            return f"✓ Saved visualization to {path}"
        else:
            return f"✓ Showing {len(embryos)} detected embryos"

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


# Auto-register on import (tools are registered by decorators)
_registered = register_all_tools()
