"""
Segmentation Tools for CV Agent

Deep learning-based 3D segmentation using Cellpose and StarDist.
"""

import asyncio
import base64
import logging
import os
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

# Viz server URL (configurable via environment variable)
VIZ_SERVER_URL = os.environ.get("VIZ_SERVER_URL", "http://localhost:8080")


def _push_segmentation_to_viz(volume: np.ndarray, masks: np.ndarray,
                               volume_uid: str, mask_uid: str, metadata: Dict = None):
    """Push 3D segmentation to viz server via HTTP POST.

    The CV subagent runs as a separate process, so we use HTTP to communicate
    with the viz server instead of direct function calls.
    """
    import requests

    try:
        # Ensure arrays are contiguous for proper serialization
        volume = np.ascontiguousarray(volume)
        masks = np.ascontiguousarray(masks)

        # Encode arrays as base64
        volume_b64 = base64.b64encode(volume.tobytes()).decode('ascii')
        masks_b64 = base64.b64encode(masks.tobytes()).decode('ascii')

        payload = {
            'volume_b64': volume_b64,
            'masks_b64': masks_b64,
            'uid': mask_uid,
            'shape': list(volume.shape),
            'dtype_vol': str(volume.dtype),
            'dtype_mask': str(masks.dtype),
            'metadata': {
                'source_uid': volume_uid,
                **(metadata or {})
            }
        }

        response = requests.post(
            f"{VIZ_SERVER_URL}/api/volumes3d",
            json=payload,
            timeout=30
        )

        if response.status_code == 200:
            logger.info(f"Pushed 3D segmentation to viz server: {mask_uid}")
        else:
            logger.warning(f"Viz server returned {response.status_code}: {response.text}")

    except requests.exceptions.ConnectionError:
        logger.debug("Viz server not available, skipping 3D volume push")
    except Exception as e:
        logger.warning(f"Failed to push segmentation to viz: {e}")


@cv_tool(
    name="cellpose_segment_3d",
    description="""Segment cells or nuclei in a 3D volume using Cellpose v4 (cpsam model).

For diSPIM data, use crop_view_a=True to use only the left half (View A).
Use stitch_mode=True (default) for fast 3D segmentation.

IMPORTANT: Cellpose v4 only has the cpsam generalist model. The diameter parameter
is critical - default 12px is for C. elegans nuclei (~5µm at 0.406µm/px).""",
    category=ToolCategory.SEGMENTATION,
    requires_gpu=True,
    examples=[
        ToolExample("Segment nuclei", {"volume_uid": "vol_abc"}),
        ToolExample("diSPIM View A", {"volume_uid": "vol_abc", "crop_view_a": True}),
        ToolExample("Larger cells", {"volume_uid": "vol_abc", "diameter": 30}),
    ],
    parameters=[
        ToolParameter(name="volume_uid", type="string", description="UID of volume", required=True),
        ToolParameter(name="diameter", type="number", description="Cell diameter in pixels (default 12 for nuclei)", required=False, default=12.0),
        ToolParameter(name="stitch_mode", type="boolean", description="Use fast stitch mode", required=False, default=True),
        ToolParameter(name="stitch_threshold", type="number", description="IoU threshold for stitching", required=False, default=0.5),
        ToolParameter(name="crop_view_a", type="boolean", description="Crop to View A (left half) for diSPIM", required=False, default=False),
        ToolParameter(name="batch_size", type="integer", description="GPU batch size", required=False, default=64),
        ToolParameter(name="flow_threshold", type="number", description="Flow error threshold", required=False, default=0.4),
        ToolParameter(name="cellprob_threshold", type="number", description="Cell probability threshold", required=False, default=0.0),
        ToolParameter(name="min_size", type="integer", description="Minimum cell size", required=False, default=15),
    ],
)
def cellpose_segment_3d(
    volume_uid: str,
    diameter: float = 12.0,
    stitch_mode: bool = True,
    stitch_threshold: float = 0.5,
    crop_view_a: bool = False,
    batch_size: int = 64,
    flow_threshold: float = 0.4,
    cellprob_threshold: float = 0.0,
    min_size: int = 15,
) -> Dict[str, Any]:
    """Segment cells/nuclei using Cellpose v4 API (cpsam model)"""
    logger.info(f"Running Cellpose (cpsam, diameter={diameter}px) on {volume_uid}")

    volume = get_cached_volume(volume_uid)
    if volume is None:
        return {"error": f"Volume {volume_uid} not found"}

    try:
        from cellpose import models

        use_gpu = HAS_GPU_MANAGER and get_gpu_manager().gpu_available if HAS_GPU_MANAGER else False

        # Ensure volume is 3D (handle 4D inputs)
        if volume.ndim == 4:
            if volume.shape[0] == 1:
                volume = volume[0]
            else:
                logger.warning(f"4D volume {volume.shape}, using first frame")
                volume = volume[0]

        if crop_view_a:
            mid_x = volume.shape[-1] // 2
            volume = volume[..., :mid_x]
            logger.info(f"Cropped to View A: {volume.shape}")

        vol_norm = volume.astype(np.float32)
        vol_norm = (vol_norm - vol_norm.min()) / (vol_norm.max() - vol_norm.min() + 1e-8)
        vol_norm = (vol_norm * 255).astype(np.uint8)

        # Cellpose v4 only has cpsam model - specify diameter explicitly
        cellpose_model = models.CellposeModel(gpu=use_gpu)

        if stitch_mode:
            masks, flows, styles = cellpose_model.eval(
                vol_norm, diameter=diameter, do_3D=False, z_axis=0,
                stitch_threshold=stitch_threshold, batch_size=batch_size,
                flow_threshold=flow_threshold, cellprob_threshold=cellprob_threshold, min_size=min_size)
        else:
            masks, flows, styles = cellpose_model.eval(
                vol_norm, diameter=diameter, do_3D=True, z_axis=0,
                batch_size=batch_size, flow_threshold=flow_threshold,
                cellprob_threshold=cellprob_threshold, min_size=min_size)

        num_cells = int(masks.max())
        cells = _extract_cell_properties(masks, volume)

        mask_uid = f"cellpose_masks_{uuid.uuid4().hex[:8]}"
        cache_volume(mask_uid, masks.astype(np.uint16), {
            "source_uid": volume_uid, "num_cells": num_cells,
            "stitch_mode": stitch_mode, "crop_view_a": crop_view_a,
            "diameter": diameter})

        # Push 3D segmentation to viz server with Z-slider
        _push_segmentation_to_viz(volume, masks, volume_uid, mask_uid, {
            "num_cells": num_cells,
            "diameter": diameter,
            "stitch_mode": stitch_mode,
            "gpu_used": use_gpu,
        })

        return {"num_cells": num_cells, "mask_uid": mask_uid, "cells": cells,
                "diameter_used": diameter, "gpu_used": use_gpu}

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
