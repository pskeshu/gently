"""
Segmentation Tools for CV Agent

Deep learning-based 3D segmentation using Cellpose and StarDist.
These tools enable the agent to count cells/nuclei and get instance masks.
"""

import logging
import uuid
from typing import Any, Dict, List, Optional

import numpy as np

from .registry import cv_tool, ToolCategory, ToolExample, ToolParameter
from .data_access import get_cached_volume, cache_volume

logger = logging.getLogger(__name__)

# Try to import GPU manager
try:
    from ..utils.gpu_manager import get_gpu_manager
    HAS_GPU_MANAGER = True
except ImportError:
    HAS_GPU_MANAGER = False
    logger.warning("GPU manager not available")


# =============================================================================
# Cellpose Segmentation
# =============================================================================

@cv_tool(
    name="cellpose_segment_3d",
    description="""Segment cells or nuclei in a 3D volume using Cellpose.

Cellpose is a deep learning model for cell segmentation. Use model_type:
- "nuclei": Best for nuclear stains (H2B-GFP, DAPI, Hoechst)
- "cyto2": Best for cytoplasmic segmentation
- "cyto": Original cytoplasm model

Returns cell count, mask UID, and cell properties (centroids, volumes).""",
    category=ToolCategory.SEGMENTATION,
    requires_gpu=True,
    examples=[
        ToolExample("Segment nuclei in volume", {"volume_uid": "vol_abc", "model_type": "nuclei"}),
        ToolExample("Segment cytoplasm with custom diameter", {"volume_uid": "vol_xyz", "model_type": "cyto2", "diameter": 30.0}),
    ],
    parameters=[
        ToolParameter(name="volume_uid", type="string", description="UID of volume to segment", required=True),
        ToolParameter(name="model_type", type="string", description="Cellpose model type",
                      required=False, default="nuclei", enum=["nuclei", "cyto2", "cyto"]),
        ToolParameter(name="diameter", type="number", description="Expected cell diameter in pixels (None for auto)", required=False),
        ToolParameter(name="flow_threshold", type="number", description="Flow error threshold (lower = stricter)", required=False, default=0.4),
        ToolParameter(name="cellprob_threshold", type="number", description="Cell probability threshold", required=False, default=0.0),
        ToolParameter(name="do_3D", type="boolean", description="Run full 3D segmentation (vs slice-by-slice)", required=False, default=True),
        ToolParameter(name="anisotropy", type="number", description="Z/XY anisotropy ratio (e.g., 2.0 if Z spacing is 2x XY)", required=False),
        ToolParameter(name="min_size", type="integer", description="Minimum cell size in voxels", required=False, default=15),
    ],
)
def cellpose_segment_3d(
    volume_uid: str,
    model_type: str = "nuclei",
    diameter: Optional[float] = None,
    flow_threshold: float = 0.4,
    cellprob_threshold: float = 0.0,
    do_3D: bool = True,
    anisotropy: Optional[float] = None,
    min_size: int = 15,
) -> Dict[str, Any]:
    """
    Segment cells/nuclei using Cellpose

    Parameters
    ----------
    volume_uid : str
        UID of volume to segment
    model_type : str
        Model type: "nuclei", "cyto2", "cyto"
    diameter : float, optional
        Expected cell diameter in pixels (None for auto)
    flow_threshold : float
        Flow error threshold (lower = stricter)
    cellprob_threshold : float
        Cell probability threshold
    do_3D : bool
        Run full 3D segmentation (vs slice-by-slice)
    anisotropy : float, optional
        Z/XY anisotropy ratio (e.g., 2.0 if Z spacing is 2x XY)
    min_size : int
        Minimum cell size in voxels

    Returns
    -------
    dict
        num_cells: Number of cells detected
        mask_uid: UID of segmentation masks (cached)
        cells: List of cell properties (centroid, volume)
        model_used: Model type used
    """
    logger.info(f"Running Cellpose {model_type} on {volume_uid}")

    volume = get_cached_volume(volume_uid)
    if volume is None:
        return {"error": f"Volume {volume_uid} not found in cache"}

    try:
        from cellpose import models

        # Get GPU manager for model caching
        if HAS_GPU_MANAGER:
            gpu_mgr = get_gpu_manager()
            use_gpu = gpu_mgr.gpu_available
        else:
            use_gpu = False

        # Load model
        logger.info(f"Loading Cellpose model: {model_type} (GPU={use_gpu})")
        model = models.Cellpose(gpu=use_gpu, model_type=model_type)

        # Normalize volume for Cellpose (expects 0-255 or 0-1)
        vol_normalized = volume.astype(np.float32)
        vol_normalized = (vol_normalized - vol_normalized.min()) / (vol_normalized.max() - vol_normalized.min() + 1e-8)

        # Run segmentation
        logger.info(f"Running segmentation (3D={do_3D}, shape={volume.shape})")

        if do_3D:
            masks, flows, styles, diams = model.eval(
                vol_normalized,
                diameter=diameter,
                flow_threshold=flow_threshold,
                cellprob_threshold=cellprob_threshold,
                do_3D=True,
                anisotropy=anisotropy,
                min_size=min_size,
            )
        else:
            # Slice-by-slice 2D segmentation
            masks = np.zeros_like(volume, dtype=np.uint16)
            for z in range(volume.shape[0]):
                mask_2d, _, _, _ = model.eval(
                    vol_normalized[z],
                    diameter=diameter,
                    flow_threshold=flow_threshold,
                    cellprob_threshold=cellprob_threshold,
                )
                masks[z] = mask_2d

        # Count cells and extract properties
        num_cells = int(masks.max())
        cells = _extract_cell_properties(masks, volume)

        # Cache masks
        mask_uid = f"cellpose_masks_{uuid.uuid4().hex[:8]}"
        cache_volume(mask_uid, masks.astype(np.uint16), {
            "source_uid": volume_uid,
            "model_type": model_type,
            "num_cells": num_cells,
            "segmentation_type": "cellpose",
        })

        logger.info(f"Cellpose found {num_cells} cells")

        return {
            "num_cells": num_cells,
            "mask_uid": mask_uid,
            "cells": cells,
            "model_used": model_type,
            "diameter_used": float(diams) if not do_3D else diameter,
            "gpu_used": use_gpu,
        }

    except ImportError:
        logger.warning("Cellpose not installed, using synthetic segmentation")
        return _synthetic_segmentation(volume_uid, volume, "cellpose", model_type)

    except Exception as e:
        logger.error(f"Cellpose segmentation failed: {e}")
        return {"error": str(e), "num_cells": 0, "mask_uid": None}


@cv_tool(
    name="stardist_segment_3d",
    description="""Segment nuclei in a 3D volume using StarDist.

StarDist is optimized for star-convex shaped objects like nuclei.
Best for dense nuclear segmentation with touching nuclei.""",
    category=ToolCategory.SEGMENTATION,
    requires_gpu=True,
    examples=[
        ToolExample("Segment nuclei with StarDist", {"volume_uid": "vol_abc"}),
        ToolExample("Use 3D model for better accuracy", {"volume_uid": "vol_xyz", "use_3d_model": True}),
    ],
    parameters=[
        ToolParameter(name="volume_uid", type="string", description="UID of volume to segment", required=True),
        ToolParameter(name="model_name", type="string", description="StarDist model name",
                      required=False, default="2D_versatile_fluo", enum=["2D_versatile_fluo", "2D_versatile_he", "3D_demo"]),
        ToolParameter(name="prob_thresh", type="number", description="Object probability threshold", required=False, default=0.5),
        ToolParameter(name="nms_thresh", type="number", description="Non-maximum suppression threshold", required=False, default=0.3),
        ToolParameter(name="n_tiles", type="array", description="Number of tiles for large images [z, y, x]", required=False),
        ToolParameter(name="use_3d_model", type="boolean", description="Use 3D StarDist model (slower but more accurate)", required=False, default=False),
    ],
)
def stardist_segment_3d(
    volume_uid: str,
    model_name: str = "2D_versatile_fluo",
    prob_thresh: float = 0.5,
    nms_thresh: float = 0.3,
    n_tiles: Optional[tuple] = None,
    use_3d_model: bool = False,
) -> Dict[str, Any]:
    """
    Segment nuclei using StarDist

    Parameters
    ----------
    volume_uid : str
        UID of volume to segment
    model_name : str
        Model name: "2D_versatile_fluo", "2D_versatile_he", "3D_demo"
    prob_thresh : float
        Object probability threshold
    nms_thresh : float
        Non-maximum suppression threshold
    n_tiles : tuple, optional
        Number of tiles for large images (z, y, x)
    use_3d_model : bool
        Use 3D StarDist model (slower but more accurate for 3D)

    Returns
    -------
    dict
        num_nuclei: Number of nuclei detected
        mask_uid: UID of segmentation masks
        nuclei: List of nuclei properties
    """
    logger.info(f"Running StarDist on {volume_uid}")

    volume = get_cached_volume(volume_uid)
    if volume is None:
        return {"error": f"Volume {volume_uid} not found in cache"}

    try:
        # Normalize volume
        vol_normalized = volume.astype(np.float32)
        vol_normalized = (vol_normalized - vol_normalized.min()) / (vol_normalized.max() - vol_normalized.min() + 1e-8)

        if use_3d_model:
            from stardist.models import StarDist3D

            logger.info(f"Loading StarDist3D model: {model_name}")
            model = StarDist3D.from_pretrained(model_name if "3D" in model_name else "3D_demo")

            # Run 3D segmentation
            labels, details = model.predict_instances(
                vol_normalized,
                prob_thresh=prob_thresh,
                nms_thresh=nms_thresh,
                n_tiles=n_tiles,
            )
        else:
            from stardist.models import StarDist2D

            logger.info(f"Loading StarDist2D model: {model_name}")
            model = StarDist2D.from_pretrained(model_name)

            # Slice-by-slice 2D segmentation
            labels = np.zeros_like(volume, dtype=np.uint16)
            max_label = 0

            for z in range(volume.shape[0]):
                labels_2d, _ = model.predict_instances(
                    vol_normalized[z],
                    prob_thresh=prob_thresh,
                    nms_thresh=nms_thresh,
                )
                # Offset labels to make unique across slices
                labels_2d[labels_2d > 0] += max_label
                max_label = labels_2d.max()
                labels[z] = labels_2d

        # Count nuclei and extract properties
        num_nuclei = int(labels.max())
        nuclei = _extract_cell_properties(labels, volume)

        # Cache masks
        mask_uid = f"stardist_masks_{uuid.uuid4().hex[:8]}"
        cache_volume(mask_uid, labels.astype(np.uint16), {
            "source_uid": volume_uid,
            "model_name": model_name,
            "num_nuclei": num_nuclei,
            "segmentation_type": "stardist",
        })

        logger.info(f"StarDist found {num_nuclei} nuclei")

        return {
            "num_nuclei": num_nuclei,
            "mask_uid": mask_uid,
            "nuclei": nuclei,
            "model_used": model_name,
            "used_3d": use_3d_model,
        }

    except ImportError:
        logger.warning("StarDist not installed, using synthetic segmentation")
        return _synthetic_segmentation(volume_uid, volume, "stardist", model_name)

    except Exception as e:
        logger.error(f"StarDist segmentation failed: {e}")
        return {"error": str(e), "num_nuclei": 0, "mask_uid": None}


# =============================================================================
# Helper Functions
# =============================================================================

def _extract_cell_properties(masks: np.ndarray, volume: np.ndarray) -> List[Dict[str, Any]]:
    """
    Extract properties for each segmented cell

    Parameters
    ----------
    masks : np.ndarray
        Label image (0 = background, 1+ = cell IDs)
    volume : np.ndarray
        Original intensity image

    Returns
    -------
    list
        List of cell property dictionaries
    """
    cells = []
    num_cells = int(masks.max())

    # Use regionprops if available
    try:
        from skimage.measure import regionprops

        props = regionprops(masks, intensity_image=volume)
        for prop in props:
            cells.append({
                "label": int(prop.label),
                "centroid": [float(c) for c in prop.centroid],
                "volume_voxels": int(prop.area),  # area is volume in 3D
                "mean_intensity": float(prop.mean_intensity),
                "bbox": [int(b) for b in prop.bbox],
            })
    except ImportError:
        # Fallback: basic centroid calculation
        for label in range(1, num_cells + 1):
            coords = np.argwhere(masks == label)
            if len(coords) > 0:
                centroid = coords.mean(axis=0)
                cells.append({
                    "label": label,
                    "centroid": [float(c) for c in centroid],
                    "volume_voxels": len(coords),
                    "mean_intensity": float(volume[masks == label].mean()),
                })

    return cells


def _synthetic_segmentation(
    volume_uid: str,
    volume: np.ndarray,
    seg_type: str,
    model_name: str,
) -> Dict[str, Any]:
    """
    Generate synthetic segmentation for testing when models unavailable

    This creates a simple threshold-based segmentation that roughly
    identifies bright objects as "cells".
    """
    logger.info("Generating synthetic segmentation (models not available)")

    try:
        from scipy import ndimage
        from skimage.measure import label

        # Simple threshold-based segmentation
        threshold = np.percentile(volume, 85)
        binary = volume > threshold

        # Label connected components
        masks, num_features = ndimage.label(binary)

        # Filter small objects
        min_size = 50
        for i in range(1, num_features + 1):
            if np.sum(masks == i) < min_size:
                masks[masks == i] = 0

        # Relabel
        masks = label(masks > 0)
        num_cells = int(masks.max())

    except ImportError:
        # Even simpler fallback
        threshold = np.percentile(volume, 85)
        masks = (volume > threshold).astype(np.uint16)
        num_cells = 1 if masks.max() > 0 else 0

    # Extract properties
    cells = _extract_cell_properties(masks, volume)

    # Cache
    mask_uid = f"synthetic_masks_{uuid.uuid4().hex[:8]}"
    cache_volume(mask_uid, masks.astype(np.uint16), {
        "source_uid": volume_uid,
        "segmentation_type": "synthetic",
        "num_cells": num_cells,
    })

    return {
        "num_cells": num_cells,
        "mask_uid": mask_uid,
        "cells": cells,
        "model_used": f"synthetic ({seg_type} unavailable)",
        "synthetic": True,
        "message": f"{seg_type} not installed, using threshold-based fallback",
    }


@cv_tool(
    name="get_segmentation_masks",
    description="Retrieve segmentation masks by UID for further analysis.",
    category=ToolCategory.SEGMENTATION,
    examples=[
        ToolExample("Get masks from previous segmentation", {"mask_uid": "mask_abc123"}),
    ],
)
def get_segmentation_masks(mask_uid: str) -> Dict[str, Any]:
    """
    Get segmentation masks from cache

    Parameters
    ----------
    mask_uid : str
        UID of the masks

    Returns
    -------
    dict
        masks: The mask array (numpy)
        num_labels: Number of unique labels
        metadata: Associated metadata
    """
    from .data_access import get_cached_volume_info

    masks = get_cached_volume(mask_uid)
    if masks is None:
        return {"error": f"Masks {mask_uid} not found"}

    info = get_cached_volume_info(mask_uid) or {}

    return {
        "mask_uid": mask_uid,
        "shape": list(masks.shape),
        "num_labels": int(masks.max()),
        "dtype": str(masks.dtype),
        "metadata": info.get("metadata", {}),
    }
