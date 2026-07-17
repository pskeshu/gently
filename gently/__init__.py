"""
Gently — Agentic harnessing for microscopy.

Built around atomic microscope actions that compose into complex experimental workflows,
with an LLM-powered agent providing autonomous observation, planning, and control.
"""

# Main entry point
from .gently import Gently, create_gently
from .harness.memory.store import (
    ContextStore,
)

# legacy SQLite store (kept for backward compat)
# Harness (framework)
from .harness.tools.registry import ToolCategory, ToolRegistry, get_tool_registry, tool

try:
    from .harness.memory.file_store import FileContextStore
except ImportError:
    FileContextStore = None  # type: ignore[assignment, misc]
# Core infrastructure
from .core import (
    EventBus,
    EventType,
    get_event_bus,
)
from .core.coordinates import (
    DEFAULT_OBJECTIVE_MAG,
    DEFAULT_PIXEL_SIZE_UM,
    get_um_per_pixel,
    pixel_displacement_to_stage_movement,
    pixel_to_stage_position,
    stage_to_pixel_position,
)
from .core.file_store import FileStore
from .core.imaging import (
    clip_and_project,
    generate_jpeg_projection,
    image_to_base64,
    normalize_to_uint8,
    projection_three_view,
    render_volume_view,
)

# Core utilities
from .core.store import GentlyStore  # legacy SQLite store (kept for backward compat)
from .harness.memory.interface import AgentMemory

# Analysis utilities
try:
    from .analysis.core import (
        FitFunction,
        FocusAlgorithm,
        FocusAnalysisConfig,
        FocusResult,
        analyze_focus_stack,
        calculate_focus_score,
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
        generate_calibration_summary_plot,
        generate_edge_detection_plot,
        generate_focus_curve_plot,
        get_visualization_server,
        mark_embryos_web,
    )

    _VISUALIZATION_AVAILABLE = True
except ImportError:
    _VISUALIZATION_AVAILABLE = False

__version__ = "1.0.0.dev0"
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
    # Visualization
    "generate_calibration_summary_plot",
    "generate_edge_detection_plot",
    "generate_focus_curve_plot",
    "get_visualization_server",
    "mark_embryos_web",
]
