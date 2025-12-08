"""
Segmentation Tools for CV Agent

Deep learning-based 3D segmentation using Cellpose and StarDist.
"""

import logging
import uuid
from typing import Any, Dict, List, Optional

import numpy as np

from .registry import cv_tool, ToolCategory, ToolExample, ToolParameter
from .data_access import get_cached_volume, cache_volume

logger = logging.getLogger(__name__)

try:
    from ..utils.gpu_manager import get_gpu_manager
    HAS_GPU_MANAGER = True
except ImportError:
    HAS_GPU_MANAGER = False


@cv_tool(
    name="cellpose_segment_3d",
    description="""Segment cells or nuclei in a 3D volume using Cellpose.

For diSPIM data, use crop_view_a=True to use only the left half (View A).
Use stitch_mode=True (default) for fast 3D segmentation.
Use downsample_factor to reduce image size for faster processing.""",
    category=ToolCategory.SEGMENTATION,
    requires_gpu=True,
    examples=[
        ToolExample("Segment nuclei", {"volume_uid": "vol_abc"}),
        ToolExample("diSPIM View A", {"volume_uid": "vol_abc", "crop_view_a": True, "downsample_factor": 2}),
    ],
    parameters=[
        ToolParameter(name="volume_uid", type="string", description="UID of volume", required=True),
        ToolParameter(name="diameter", type="number", description="Cell diameter in pixels", required=False, default=30.0),
        ToolParameter(name="stitch_mode", type="boolean", description="Use fast stitch mode", required=False, default=True),
        ToolParameter(name="stitch_threshold", type="number", description="IoU threshold for stitching", required=False, default=0.5),
        ToolParameter(name="crop_view_a", type="boolean", description="Crop to View A (left half) for diSPIM", required=False, default=False),
        ToolParameter(name="downsample_factor", type="number", description="Downsample XY factor", required=False, default=1),
        ToolParameter(name="batch_size", type="integer", description="GPU batch size", required=False, default=64),
        ToolParameter(name="flow_threshold", type="number", description="Flow error threshold", required=False, default=0.4),
        ToolParameter(name="cellprob_threshold", type="number", description="Cell probability threshold", required=False, default=0.0),
        ToolParameter(name="min_size", type="integer", description="Minimum cell size", required=False, default=15),
    ],
)
def cellpose_segment_3d(
    volume_uid: str,
    diameter: float = 30.0,
    stitch_mode: bool = True,
    stitch_threshold: float = 0.5,
    crop_view_a: bool = False,
    downsample_factor: float = 1,
    batch_size: int = 64,
    flow_threshold: float = 0.4,
    cellprob_threshold: float = 0.0,
    min_size: int = 15,
) -> Dict[str, Any]:
    """Segment cells/nuclei using Cellpose v4 API with optimizations"""
    logger.info(f"Running Cellpose on {volume_uid}")

    volume = get_cached_volume(volume_uid)
    if volume is None:
        return {"error": f"Volume {volume_uid} not found"}

    try:
        from cellpose import models

        use_gpu = HAS_GPU_MANAGER and get_gpu_manager().gpu_available if HAS_GPU_MANAGER else False

        if crop_view_a:
            mid_x = volume.shape[-1] // 2
            volume = volume[..., :mid_x]
            logger.info(f"Cropped to View A: {volume.shape}")

        effective_diameter = diameter
        if downsample_factor > 1:
            from scipy.ndimage import zoom
            volume = zoom(volume, (1, 1.0/downsample_factor, 1.0/downsample_factor), order=1)
            effective_diameter = diameter / downsample_factor
            logger.info(f"Downsampled {downsample_factor}x: {volume.shape}")

        vol_norm = volume.astype(np.float32)
        vol_norm = (vol_norm - vol_norm.min()) / (vol_norm.max() - vol_norm.min() + 1e-8)
        vol_norm = (vol_norm * 255).astype(np.uint8)

        model = models.CellposeModel(gpu=use_gpu)

        if stitch_mode:
            masks, flows, styles = model.eval(
                vol_norm, diameter=effective_diameter, do_3D=False, z_axis=0,
                stitch_threshold=stitch_threshold, batch_size=batch_size,
                flow_threshold=flow_threshold, cellprob_threshold=cellprob_threshold, min_size=min_size)
        else:
            masks, flows, styles = model.eval(
                vol_norm, diameter=effective_diameter, do_3D=True, z_axis=0,
                batch_size=batch_size, flow_threshold=flow_threshold,
                cellprob_threshold=cellprob_threshold, min_size=min_size)

        num_cells = int(masks.max())
        cells = _extract_cell_properties(masks, volume)

        mask_uid = f"cellpose_masks_{uuid.uuid4().hex[:8]}"
        cache_volume(mask_uid, masks.astype(np.uint16), {
            "source_uid": volume_uid, "num_cells": num_cells,
            "stitch_mode": stitch_mode, "crop_view_a": crop_view_a,
            "downsample_factor": downsample_factor})

        return {"num_cells": num_cells, "mask_uid": mask_uid, "cells": cells,
                "diameter_used": effective_diameter, "gpu_used": use_gpu}

    except ImportError:
        return _synthetic_segmentation(volume_uid, volume, "cellpose")
    except Exception as e:
        return {"error": str(e), "num_cells": 0}


def _extract_cell_properties(masks: np.ndarray, volume: np.ndarray) -> List[Dict[str, Any]]:
    cells = []
    try:
        from skimage.measure import regionprops
        for prop in regionprops(masks, intensity_image=volume):
            cells.append({"label": int(prop.label), "centroid": [float(c) for c in prop.centroid],
                          "volume_voxels": int(prop.area), "mean_intensity": float(prop.mean_intensity)})
    except ImportError:
        pass
    return cells


def _synthetic_segmentation(volume_uid: str, volume: np.ndarray, seg_type: str) -> Dict[str, Any]:
    from scipy import ndimage
    masks, _ = ndimage.label(volume > np.percentile(volume, 85))
    mask_uid = f"synthetic_{uuid.uuid4().hex[:8]}"
    cache_volume(mask_uid, masks.astype(np.uint16), {"source_uid": volume_uid})
    return {"num_cells": int(masks.max()), "mask_uid": mask_uid, "synthetic": True}


@cv_tool(name="get_segmentation_masks", description="Get masks by UID", category=ToolCategory.SEGMENTATION,
         examples=[ToolExample("Get masks", {"mask_uid": "m1"})])
def get_segmentation_masks(mask_uid: str) -> Dict[str, Any]:
    masks = get_cached_volume(mask_uid)
    if masks is None:
        return {"error": f"Not found: {mask_uid}"}
    return {"mask_uid": mask_uid, "shape": list(masks.shape), "num_labels": int(masks.max())}
