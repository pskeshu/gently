"""
Tools available to the CV Agent for image analysis and processing.

This module provides all the tools the CV agent can use to analyze
C. elegans embryo development, including:

- Data Access: Load volumes, query history, store results
- Preparation: ROI detection, cropping, scale bars, image prep
- Vision: Claude Vision analysis, stage classification
- Segmentation: Cellpose, StarDist for cell/nuclei detection
- Morphology: Shape metrics, elongation measurements
- Tracking: Cell tracking across timepoints, division detection
"""

from .registry import CVToolRegistry, cv_tool, ToolCategory

# Import tool modules to register them
from . import data_access
from . import preparation
from . import vision
from . import segmentation
from . import morphology
from . import tracking

# Export cache utilities for direct access
from .data_access import (
    cache_volume,
    get_cached_volume,
    get_cached_volume_info,
    clear_cache,
)

__all__ = [
    "CVToolRegistry",
    "cv_tool",
    "ToolCategory",
    # Cache utilities
    "cache_volume",
    "get_cached_volume",
    "get_cached_volume_info",
    "clear_cache",
]
