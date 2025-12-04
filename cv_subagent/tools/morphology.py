"""
Morphology Tools for CV Agent

Tools for measuring shape metrics from segmentation masks.
These measurements help classify developmental stages (e.g., elongation for fold stages).
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np

from .registry import cv_tool, ToolCategory
from .data_access import get_cached_volume

logger = logging.getLogger(__name__)


@cv_tool(
    name="measure_morphology",
    description="""Measure shape metrics from segmentation masks.

Returns morphological features important for developmental staging:
- elongation: Length/width ratio (key for comma/fold stages)
- circularity: How circular the object is (1.0 = perfect circle)
- solidity: Convex hull fill ratio (detects irregular shapes)
- major/minor axis: Principal axis lengths

Use after cellpose_segment_3d or stardist_segment_3d.""",
    category=ToolCategory.ANALYSIS,
)
def measure_morphology(
    masks_uid: str,
    include_per_cell: bool = False,
) -> Dict[str, Any]:
    """
    Measure morphology from segmentation masks

    Parameters
    ----------
    masks_uid : str
        UID of segmentation masks
    include_per_cell : bool
        Include individual cell measurements

    Returns
    -------
    dict
        embryo_metrics: Overall embryo shape metrics
        cell_metrics: Per-cell metrics (if include_per_cell)
        num_cells: Number of cells
    """
    logger.info(f"Measuring morphology from {masks_uid}")

    masks = get_cached_volume(masks_uid)
    if masks is None:
        return {"error": f"Masks {masks_uid} not found"}

    # Measure overall embryo shape (all cells combined)
    embryo_mask = masks > 0
    embryo_metrics = _measure_shape_3d(embryo_mask)

    # Measure individual cells if requested
    cell_metrics = []
    num_cells = int(masks.max())

    if include_per_cell and num_cells > 0:
        for label in range(1, num_cells + 1):
            cell_mask = masks == label
            if np.sum(cell_mask) > 0:
                metrics = _measure_shape_3d(cell_mask)
                metrics["label"] = label
                cell_metrics.append(metrics)

    # Calculate aggregate statistics
    if cell_metrics:
        volumes = [c["volume_voxels"] for c in cell_metrics]
        embryo_metrics["mean_cell_volume"] = float(np.mean(volumes))
        embryo_metrics["std_cell_volume"] = float(np.std(volumes))
        embryo_metrics["min_cell_volume"] = float(np.min(volumes))
        embryo_metrics["max_cell_volume"] = float(np.max(volumes))

    return {
        "masks_uid": masks_uid,
        "num_cells": num_cells,
        "embryo_metrics": embryo_metrics,
        "cell_metrics": cell_metrics if include_per_cell else None,
    }


@cv_tool(
    name="measure_embryo_elongation",
    description="""Measure embryo elongation ratio specifically.

The elongation ratio (length/width) is critical for distinguishing:
- Pre-elongation stages: ratio < 1.5
- Comma stage: ratio 1.5-2.0
- 1.5-fold: ratio 2.0-2.5
- 2-fold: ratio 2.5-3.5
- 3-fold: ratio > 3.5""",
    category=ToolCategory.ANALYSIS,
)
def measure_embryo_elongation(
    masks_uid: Optional[str] = None,
    volume_uid: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Measure embryo elongation ratio

    Parameters
    ----------
    masks_uid : str, optional
        UID of segmentation masks (preferred)
    volume_uid : str, optional
        UID of volume (will threshold if no masks)

    Returns
    -------
    dict
        elongation_ratio: Length/width ratio
        major_axis_length: Embryo length in voxels
        minor_axis_length: Embryo width in voxels
        stage_hint: Suggested stage based on elongation
    """
    if masks_uid:
        masks = get_cached_volume(masks_uid)
        if masks is None:
            return {"error": f"Masks {masks_uid} not found"}
        binary = masks > 0
    elif volume_uid:
        volume = get_cached_volume(volume_uid)
        if volume is None:
            return {"error": f"Volume {volume_uid} not found"}
        # Simple threshold to get embryo
        threshold = np.percentile(volume, 75)
        binary = volume > threshold
    else:
        return {"error": "Must provide masks_uid or volume_uid"}

    # Measure elongation
    metrics = _measure_shape_3d(binary)
    elongation = metrics["elongation"]

    # Provide stage hint based on elongation
    if elongation < 1.3:
        stage_hint = "pre-elongation (1-cell to gastrula)"
    elif elongation < 1.7:
        stage_hint = "early elongation (late gastrula to comma)"
    elif elongation < 2.2:
        stage_hint = "comma to 1.5-fold"
    elif elongation < 3.0:
        stage_hint = "1.5-fold to 2-fold"
    elif elongation < 4.0:
        stage_hint = "2-fold to 3-fold"
    else:
        stage_hint = "3-fold or pretzel"

    return {
        "elongation_ratio": elongation,
        "major_axis_length": metrics["major_axis_length"],
        "minor_axis_length": metrics["minor_axis_length"],
        "stage_hint": stage_hint,
        "circularity": metrics["circularity"],
    }


@cv_tool(
    name="measure_cell_sizes",
    description="Measure the size distribution of segmented cells.",
    category=ToolCategory.ANALYSIS,
)
def measure_cell_sizes(masks_uid: str) -> Dict[str, Any]:
    """
    Measure cell size distribution

    Parameters
    ----------
    masks_uid : str
        UID of segmentation masks

    Returns
    -------
    dict
        num_cells: Number of cells
        sizes: List of cell volumes in voxels
        statistics: Mean, std, min, max
        size_histogram: Binned size counts
    """
    masks = get_cached_volume(masks_uid)
    if masks is None:
        return {"error": f"Masks {masks_uid} not found"}

    num_cells = int(masks.max())
    if num_cells == 0:
        return {
            "num_cells": 0,
            "sizes": [],
            "statistics": {},
        }

    sizes = []
    for label in range(1, num_cells + 1):
        size = np.sum(masks == label)
        if size > 0:
            sizes.append(int(size))

    sizes = np.array(sizes)

    # Create histogram
    if len(sizes) > 1:
        hist, bin_edges = np.histogram(sizes, bins=min(10, len(sizes)))
        histogram = {
            "counts": hist.tolist(),
            "bin_edges": bin_edges.tolist(),
        }
    else:
        histogram = None

    return {
        "num_cells": len(sizes),
        "sizes": sizes.tolist(),
        "statistics": {
            "mean": float(sizes.mean()),
            "std": float(sizes.std()),
            "min": int(sizes.min()),
            "max": int(sizes.max()),
            "median": float(np.median(sizes)),
        },
        "size_histogram": histogram,
    }


# =============================================================================
# Helper Functions
# =============================================================================

def _measure_shape_3d(binary_mask: np.ndarray) -> Dict[str, float]:
    """
    Measure 3D shape metrics from a binary mask

    Parameters
    ----------
    binary_mask : np.ndarray
        Binary mask (True = object)

    Returns
    -------
    dict
        Shape metrics
    """
    # Volume
    volume = int(np.sum(binary_mask))
    if volume == 0:
        return {
            "volume_voxels": 0,
            "elongation": 1.0,
            "major_axis_length": 0,
            "minor_axis_length": 0,
            "circularity": 0,
            "solidity": 0,
        }

    try:
        from skimage.measure import regionprops

        # Use regionprops for accurate measurements
        props = regionprops(binary_mask.astype(np.uint8))[0]

        # Get principal axes (eigenvalues of inertia tensor)
        # Note: in 3D, we use the moments for elongation calculation
        if binary_mask.ndim == 3:
            # Project to get 2D shape for elongation
            projection = np.max(binary_mask, axis=0)
            props_2d = regionprops(projection.astype(np.uint8))
            if props_2d:
                major = props_2d[0].major_axis_length
                minor = props_2d[0].minor_axis_length
            else:
                major = minor = 1.0
        else:
            major = props.major_axis_length
            minor = props.minor_axis_length

        elongation = major / minor if minor > 0 else 1.0

        # Circularity (2D) - using max projection
        perimeter = _estimate_perimeter(np.max(binary_mask, axis=0) if binary_mask.ndim == 3 else binary_mask)
        area_2d = np.sum(np.max(binary_mask, axis=0) if binary_mask.ndim == 3 else binary_mask)
        circularity = 4 * np.pi * area_2d / (perimeter ** 2) if perimeter > 0 else 0

        # Solidity (using convex hull)
        try:
            from scipy.spatial import ConvexHull
            coords = np.argwhere(binary_mask)
            if len(coords) > 4:
                hull = ConvexHull(coords)
                solidity = volume / hull.volume if hull.volume > 0 else 1.0
            else:
                solidity = 1.0
        except:
            solidity = 1.0

        return {
            "volume_voxels": volume,
            "elongation": float(elongation),
            "major_axis_length": float(major),
            "minor_axis_length": float(minor),
            "circularity": float(min(1.0, circularity)),
            "solidity": float(min(1.0, solidity)),
        }

    except ImportError:
        # Fallback without skimage
        return _measure_shape_simple(binary_mask)


def _measure_shape_simple(binary_mask: np.ndarray) -> Dict[str, float]:
    """Simple shape measurement without skimage"""
    volume = int(np.sum(binary_mask))

    # Find bounding box
    coords = np.argwhere(binary_mask)
    if len(coords) == 0:
        return {
            "volume_voxels": 0,
            "elongation": 1.0,
            "major_axis_length": 0,
            "minor_axis_length": 0,
            "circularity": 0,
            "solidity": 0,
        }

    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    dims = maxs - mins + 1

    # For 3D, use max projection dimensions for elongation
    if len(dims) == 3:
        major = max(dims[1], dims[2])  # Y or X
        minor = min(dims[1], dims[2])
    else:
        major = max(dims)
        minor = min(dims)

    elongation = major / minor if minor > 0 else 1.0

    return {
        "volume_voxels": volume,
        "elongation": float(elongation),
        "major_axis_length": float(major),
        "minor_axis_length": float(minor),
        "circularity": 0.5,  # Unknown without perimeter
        "solidity": 0.8,  # Estimated
    }


def _estimate_perimeter(binary_2d: np.ndarray) -> float:
    """Estimate perimeter of 2D binary image"""
    try:
        from skimage.measure import perimeter
        return perimeter(binary_2d)
    except ImportError:
        # Simple edge counting
        edges = np.abs(np.diff(binary_2d.astype(int), axis=0)).sum()
        edges += np.abs(np.diff(binary_2d.astype(int), axis=1)).sum()
        return float(edges)
