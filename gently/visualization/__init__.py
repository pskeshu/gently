"""
Visualization Tools for diSPIM
=========================================

Provides:
- Napari-based interactive visualization tools
- Web-based visualization server with real-time streaming
- Plot generation utilities for real-time feedback
"""

from .embryo_marker import EmbryoMarker, mark_embryos_napari, mark_embryos_web
from .plots import (
    generate_focus_curve_plot,
    generate_calibration_summary_plot,
    generate_edge_detection_plot,
)

# Lazy import for server (requires FastAPI)
def get_visualization_server():
    from .server import VisualizationServer, create_visualization_server
    return VisualizationServer, create_visualization_server

__all__ = [
    'EmbryoMarker',
    'mark_embryos_napari',
    'mark_embryos_web',
    'get_visualization_server',
    'generate_focus_curve_plot',
    'generate_calibration_summary_plot',
    'generate_edge_detection_plot',
]
