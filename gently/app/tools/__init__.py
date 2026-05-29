"""
Agent Tools Package

Tools organized by category for the microscopy agent.
All tools are automatically registered via the @tool decorator when imported.
"""

# Import all tool modules to register their tools
from . import experiment_tools
from . import stage_tools
from . import led_tools
from . import light_source_tools
from . import temperature_tools
from . import calibration_tools
from . import acquisition_tools
from . import volume_tools
from . import analysis_tools
from . import timelapse_tools
from . import session_tools
from . import focus_tools
from . import interaction_tools
from . import detection_tools
from . import plan_execution_tools
from . import memory_tools
from . import resolution_tools
from gently.harness.plan_mode.tools import lab_context as _lab_context  # query_lab_history in run mode

# Import tool registry utilities
from gently.harness.tools.registry import get_tool_registry, ToolCategory

# Re-export helper utilities for convenience
from gently.harness.tools.helpers import (
    require_agent, get_embryo_or_error, require_microscope,
    require_interaction_logger, require_developmental_tracker,
    require_timelapse_orchestrator, require_databroker,
    get_timestamp_string, format_duration
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
