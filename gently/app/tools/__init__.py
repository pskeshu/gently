"""
Agent Tools Package

Tools organized by category for the microscopy agent.
All tools are automatically registered via the @tool decorator when imported.
"""

# Import all tool modules to register their tools
from gently.harness.plan_mode.tools import (
    lab_context as _lab_context,  # noqa: F401
)

# query_lab_history in run mode
# Re-export helper utilities for convenience
from gently.harness.tools.helpers import (
    format_duration,
    get_embryo_or_error,
    get_timestamp_string,
    require_agent,
    require_databroker,
    require_developmental_tracker,
    require_interaction_logger,
    require_microscope,
    require_timelapse_orchestrator,
)

# Import tool registry utilities
from gently.harness.tools.registry import ToolCategory, get_tool_registry

from . import (
    acquisition_tools,
    analysis_tools,
    calibration_tools,
    detection_tools,
    experiment_tools,
    focus_tools,
    interaction_tools,
    led_tools,
    light_source_tools,
    memory_tools,
    operation_plan_tools,
    plan_execution_tools,
    resolution_tools,
    session_tools,
    stage_tools,
    tactic_library_tools,
    temperature_protocol_tools,
    temperature_tools,
    timelapse_tools,
    volume_tools,
)


def register_all_tools():
    """
    Ensure all tools in this package are registered.
    This is called automatically when the package is imported.
    """
    registry = get_tool_registry()
    return f"Registered {len(registry)} tools"


# Auto-register on import
_registered = register_all_tools()

__all__ = [
    # Helper utilities
    "format_duration",
    "get_embryo_or_error",
    "get_timestamp_string",
    "require_agent",
    "require_databroker",
    "require_developmental_tracker",
    "require_interaction_logger",
    "require_microscope",
    "require_timelapse_orchestrator",
    # Tool registry
    "ToolCategory",
    # Tool modules (imported for registration side effects)
    "acquisition_tools",
    "analysis_tools",
    "calibration_tools",
    "detection_tools",
    "experiment_tools",
    "focus_tools",
    "interaction_tools",
    "led_tools",
    "light_source_tools",
    "memory_tools",
    "operation_plan_tools",
    "plan_execution_tools",
    "resolution_tools",
    "tactic_library_tools",
    "session_tools",
    "stage_tools",
    "temperature_protocol_tools",
    "temperature_tools",
    "timelapse_tools",
    "volume_tools",
]
