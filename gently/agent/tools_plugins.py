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


# Auto-register on import (tools are registered by decorators)
_registered = register_all_tools()
