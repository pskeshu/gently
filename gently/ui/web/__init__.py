"""
Visualization Tools for diSPIM
==============================

Provides:
- Web map view for interactive embryo marking + role assignment
- Visualization server with real-time streaming
- Plot generation utilities for real-time feedback

The napari-based marker (``EmbryoMarker``, ``mark_embryos_napari``) was
retired in Phase 1 — use ``mark_embryos_web`` for all interactive marking.
"""

from .embryo_marker import mark_embryos_web
from .plots import (
    generate_calibration_summary_plot,
    generate_edge_detection_plot,
    generate_focus_curve_plot,
)


# Lazy import for server (requires FastAPI)
def get_visualization_server():
    from .server import VisualizationServer, create_visualization_server

    return VisualizationServer, create_visualization_server


__all__ = [
    "mark_embryos_web",
    "get_visualization_server",
    "generate_focus_curve_plot",
    "generate_calibration_summary_plot",
    "generate_edge_detection_plot",
]
