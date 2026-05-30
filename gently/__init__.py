"""
Gently — Agentic harnessing for microscopy.

Built around atomic microscope actions that compose into complex experimental workflows,
with an LLM-powered agent providing autonomous observation, planning, and control.
"""

# Main entry point
from .gently import Gently, create_gently

# Harness (framework)
from .harness.tools.registry import tool, ToolRegistry, ToolCategory, get_tool_registry
from .harness.memory.store import ContextStore  # legacy SQLite store (kept for backward compat)
try:
    from .harness.memory.file_store import FileContextStore
except ImportError:
    FileContextStore = None
from .harness.memory.interface import AgentMemory

# Core infrastructure
from .core import (
    EventBus,
    EventType,
    get_event_bus,
)

# Core utilities
from .core.store import GentlyStore  # legacy SQLite store (kept for backward compat)
from .core.file_store import FileStore
from .core.imaging import (
    normalize_to_uint8,
    image_to_base64,
    projection_three_view,
    render_volume_view,
    clip_and_project,
    generate_jpeg_projection,
)
from .core.coordinates import (
    pixel_to_stage_position,
    stage_to_pixel_position,
    pixel_displacement_to_stage_movement,
    get_um_per_pixel,
    DEFAULT_PIXEL_SIZE_UM,
    DEFAULT_OBJECTIVE_MAG,
)

# Analysis utilities
try:
    from .analysis.core import (
        FocusAnalysisConfig,
        FocusResult,
        FocusAlgorithm,
        FitFunction,
        calculate_focus_score,
        analyze_focus_stack,
        fit_focus_curve,
    )
    _ANALYSIS_AVAILABLE = True
except ImportError:
    _ANALYSIS_AVAILABLE = False

# Hardware (loaded via plugins, not imported directly at top level)
# Use: from gently.hardware.dispim import ...
# Or: load_hardware("dispim"); hw = get_hardware()

# Visualization (web map view replaces the retired napari marker)
try:
    from .ui.web import (
        mark_embryos_web,
        get_visualization_server,
        generate_focus_curve_plot,
        generate_calibration_summary_plot,
        generate_edge_detection_plot,
    )
    _VISUALIZATION_AVAILABLE = True
except ImportError:
    _VISUALIZATION_AVAILABLE = False

__version__ = "0.22.0.dev0"
__all__ = [
    # Main entry point
    "Gently",
    "create_gently",

    # Harness
    "tool",
    "ToolRegistry",
    "ToolCategory",
    "get_tool_registry",
    "ContextStore",  # legacy SQLite store (backward compat)
    "AgentMemory",

    # Core infrastructure
    "EventBus",
    "EventType",
    "get_event_bus",
    "GentlyStore",  # legacy SQLite store (backward compat)
    "FileStore",
    "FileContextStore",

    # Imaging
    "normalize_to_uint8",
    "image_to_base64",
    "projection_three_view",
    "render_volume_view",
    "clip_and_project",
    "generate_jpeg_projection",

    # Coordinates
    "pixel_to_stage_position",
    "stage_to_pixel_position",
    "pixel_displacement_to_stage_movement",
    "get_um_per_pixel",
    "DEFAULT_PIXEL_SIZE_UM",
    "DEFAULT_OBJECTIVE_MAG",

    # Analysis
    "FocusAnalysisConfig",
    "FocusResult",
    "FocusAlgorithm",
    "FitFunction",
    "calculate_focus_score",
    "analyze_focus_stack",
    "fit_focus_curve",
]
