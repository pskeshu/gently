"""
Visualization Tools for diSPIM Microscopy
=========================================

Provides:
- Napari-based interactive visualization tools
- Web-based visualization server with real-time streaming
"""

from .embryo_marker import EmbryoMarker, mark_embryos_napari

# Lazy import for server (requires FastAPI)
def get_visualization_server():
    from .server import VisualizationServer, create_visualization_server
    return VisualizationServer, create_visualization_server

__all__ = [
    'EmbryoMarker',
    'mark_embryos_napari',
    'get_visualization_server',
]
