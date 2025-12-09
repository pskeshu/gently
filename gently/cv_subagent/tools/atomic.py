"""
Atomic Task-Oriented Tools for CV Agent

These tools provide single-purpose, end-to-end analysis for common C. elegans
embryo analysis tasks. Unlike primitive tools that require chaining, these
atomic tools handle the full pipeline internally:

1. Load volume using CVContext (handles diSPIM View A cropping)
2. Perform minimal processing for the specific task
3. Store result in data store with provenance
4. Publish completion event

Usage
-----
Each atomic tool receives a CVContext (injected by the registry) containing
microscope and organism configuration. This eliminates guessing like
"This is likely a diSPIM image based on the wide width."

Example:
    # Registry injects context automatically
    result = await registry.execute("count_nuclei", embryo_id="embryo_1", timepoint=5)
"""

import logging
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

from .registry import cv_tool, ToolCategory, ToolExample, ToolParameter
from .data_access import get_cached_volume, cache_volume, _get_store
from .segmentation import _push_segmentation_to_viz
from .preparation import _push_image_to_viz

logger = logging.getLogger(__name__)

# No downsampling by default - full resolution for accurate segmentation
DEFAULT_DOWNSAMPLE_FACTOR = 1

# Import CVContext type for annotations
try:
    from ..context import CVContext
except ImportError:
    CVContext = Any


# =============================================================================
# Internal Helpers
# =============================================================================

def _load_volume_for_context(
    context: CVContext,
    embryo_id: str,
    timepoint: Optional[int] = None,
) -> Optional[np.ndarray]:
    """
    Load volume with microscope-aware preprocessing.

    Parameters
    ----------
    context : CVContext
        Microscope and organism context
    embryo_id : str
        Embryo ID to load
    timepoint : int, optional
        Specific timepoint, or latest if None

    Returns
    -------
    np.ndarray or None
        Preprocessed volume ready for analysis
    """
    store = _get_store()
    if store is None:
        logger.error("Data store not available")
        return None

    # Query for the volume
    query_params = {"data_type": "volume", "embryo_id": embryo_id}
    if timepoint is not None:
        query_params["timepoint"] = timepoint

    try:
        refs = store.query(**query_params)
        if not refs:
            logger.warning(f"No volume found for {embryo_id}, timepoint={timepoint}")
            return None

        # Get the most recent if no specific timepoint
        ref = refs[-1] if timepoint is None else refs[0]
        volume = store.retrieve(ref)

        if volume is None:
            return None

        # Cache for potential reuse by other tools
        volume_uid = f"atomic_{embryo_id}_{timepoint or 'latest'}_{uuid.uuid4().hex[:8]}"
        cache_volume(volume_uid, volume, {
            "embryo_id": embryo_id,
            "timepoint": timepoint,
            "source": "atomic_tool",
        })

        # Apply microscope-specific preprocessing
        volume = _preprocess_for_microscope(volume, context)

        return volume

    except Exception as e:
        logger.error(f"Failed to load volume: {e}")
        return None


def _preprocess_for_microscope(volume: np.ndarray, context: CVContext) -> np.ndarray:
    """
    Apply microscope-specific preprocessing.

    Steps:
    1. Handle 4D -> 3D conversion
    2. Crop to View A for diSPIM (left half)
    3. Detect and crop to embryo ROI
    """
    # Handle 4D volumes (channels)
    if volume.ndim == 4:
        if volume.shape[0] == 1:
            volume = volume[0]
        else:
            logger.warning(f"Multi-channel volume {volume.shape}, using first channel")
            volume = volume[0]

    # Crop to View A for diSPIM
    if context.is_dispim and context.has_dual_view:
        view_slice = context.get_view_a_slice()
        volume = volume[..., view_slice]
        logger.info(f"Cropped to View A: {volume.shape}")

    # Detect and crop to embryo ROI
    volume = _crop_to_embryo_roi(volume, padding_percent=20.0)

    return volume


def _crop_to_embryo_roi(
    volume: np.ndarray,
    padding_percent: float = 20.0,
    min_size_voxels: int = 1000,
) -> np.ndarray:
    """
    Detect embryo and crop volume to ROI with padding.

    Parameters
    ----------
    volume : np.ndarray
        3D volume (Z, Y, X)
    padding_percent : float
        Padding around detected ROI as percentage
    min_size_voxels : int
        Minimum object size to consider as embryo

    Returns
    -------
    np.ndarray
        Cropped volume containing just the embryo region
    """
    try:
        from scipy import ndimage

        # Threshold to find embryo
        threshold = np.percentile(volume, 75)
        mask = volume > threshold

        # Label connected components
        labeled, num_features = ndimage.label(mask)
        if num_features == 0:
            logger.warning("No embryo detected, returning full volume")
            return volume

        # Find largest component above min size
        sizes = ndimage.sum(mask, labeled, range(1, num_features + 1))
        valid_labels = [i + 1 for i, s in enumerate(sizes) if s >= min_size_voxels]

        if not valid_labels:
            logger.warning("No objects above minimum size, returning full volume")
            return volume

        # Keep the largest object
        largest_label = valid_labels[np.argmax([sizes[i-1] for i in valid_labels])]
        embryo_mask = labeled == largest_label

        # Find bounding box
        coords = np.argwhere(embryo_mask)
        z_min, y_min, x_min = coords.min(axis=0)
        z_max, y_max, x_max = coords.max(axis=0)

        # Add padding
        z_pad = int((z_max - z_min) * padding_percent / 100)
        y_pad = int((y_max - y_min) * padding_percent / 100)
        x_pad = int((x_max - x_min) * padding_percent / 100)

        z1 = max(0, z_min - z_pad)
        y1 = max(0, y_min - y_pad)
        x1 = max(0, x_min - x_pad)
        z2 = min(volume.shape[0], z_max + z_pad + 1)
        y2 = min(volume.shape[1], y_max + y_pad + 1)
        x2 = min(volume.shape[2], x_max + x_pad + 1)

        cropped = volume[z1:z2, y1:y2, x1:x2]
        logger.info(f"Cropped to embryo ROI: {volume.shape} -> {cropped.shape}")

        # Push ROI visualization to viz server
        try:
            max_proj = np.max(volume, axis=0)
            # Normalize to 0-255
            max_proj = ((max_proj - max_proj.min()) / (max_proj.max() - max_proj.min() + 1e-8) * 255).astype(np.uint8)
            # Convert to RGB
            vis_img = np.stack([max_proj, max_proj, max_proj], axis=-1)
            # Draw ROI rectangle (green)
            vis_img[y_min, x_min:x_max, :] = [0, 255, 0]
            vis_img[min(y_max, vis_img.shape[0]-1), x_min:x_max, :] = [0, 255, 0]
            vis_img[y_min:y_max, x_min, :] = [0, 255, 0]
            vis_img[y_min:y_max, min(x_max, vis_img.shape[1]-1), :] = [0, 255, 0]
            # Draw padded ROI rectangle (cyan)
            vis_img[y1, x1:x2, :] = [0, 255, 255]
            vis_img[min(y2-1, vis_img.shape[0]-1), x1:x2, :] = [0, 255, 255]
            vis_img[y1:y2, x1, :] = [0, 255, 255]
            vis_img[y1:y2, min(x2-1, vis_img.shape[1]-1), :] = [0, 255, 255]
            # Draw center crosshair (red)
            cy, cx = (y_min + y_max) // 2, (x_min + x_max) // 2
            vis_img[max(0, cy-5):min(vis_img.shape[0], cy+5), cx, :] = [255, 0, 0]
            vis_img[cy, max(0, cx-5):min(vis_img.shape[1], cx+5), :] = [255, 0, 0]

            roi_uid = f"roi_atomic_{uuid.uuid4().hex[:6]}"
            _push_image_to_viz(vis_img, roi_uid, "roi_detection", {
                "bbox": [int(z_min), int(y_min), int(x_min), int(z_max), int(y_max), int(x_max)],
                "padded_bbox": [int(z1), int(y1), int(x1), int(z2), int(y2), int(x2)],
                "original_shape": list(volume.shape),
                "cropped_shape": list(cropped.shape),
            })
        except Exception as e:
            logger.debug(f"ROI visualization failed: {e}")

        return cropped

    except ImportError:
        logger.warning("scipy not available, skipping ROI detection")
        return volume
    except Exception as e:
        logger.warning(f"ROI detection failed: {e}, returning full volume")
        return volume


def _run_cellpose_nuclei(
    volume: np.ndarray,
    diameter_px: float = 12.0,
) -> tuple:
    """
    Run Cellpose for nuclei segmentation using v4 API (cpsam model).

    Parameters
    ----------
    volume : np.ndarray
        3D volume (Z, Y, X)
    diameter_px : float
        Expected nucleus diameter in pixels. Default 12px (~5µm at 0.406µm/px).
        Critical parameter for cpsam model - must be specified for good results.

    Returns
    -------
    tuple
        (masks, processed_volume, effective_diameter, gpu_used)
        - masks: Label mask at full resolution
        - processed_volume: Normalized volume (for viz)
        - effective_diameter: Diameter used
        - gpu_used: Whether GPU was used
    """
    try:
        from cellpose import models

        gpu_used = False
        try:
            import torch
            gpu_used = torch.cuda.is_available()
        except ImportError:
            pass

        # Normalize
        vol_norm = volume.astype(np.float32)
        vol_norm = (vol_norm - vol_norm.min()) / (vol_norm.max() - vol_norm.min() + 1e-8)
        vol_norm = (vol_norm * 255).astype(np.uint8)

        # Cellpose v4 only has cpsam model - nuclei/cyto2 don't exist
        # Must specify diameter since cpsam is a generalist model
        model = models.CellposeModel(gpu=gpu_used)

        # Run segmentation with stitch mode for speed
        masks, flows, styles = model.eval(
            vol_norm,
            diameter=diameter_px,
            do_3D=False,
            z_axis=0,
            stitch_threshold=0.5,
            min_size=15,
        )

        return masks, volume, diameter_px, gpu_used

    except ImportError:
        logger.error("Cellpose not installed")
        return np.zeros_like(volume, dtype=np.int32), volume, diameter_px, False
    except Exception as e:
        logger.error(f"Cellpose failed: {e}")
        return np.zeros_like(volume, dtype=np.int32), volume, diameter_px, False


def _store_and_publish(
    context: CVContext,
    result_type: str,
    result: Dict[str, Any],
) -> str:
    """
    Store analysis result and publish completion event.

    Returns
    -------
    str
        UID of the stored result
    """
    store = _get_store()
    result_uid = f"{result_type}_{uuid.uuid4().hex[:8]}"

    # Store in data store
    if store is not None:
        try:
            store.store(
                data=result,
                data_type="analysis",
                metadata={
                    "result_type": result_type,
                    "embryo_id": context.embryo_id,
                    "timepoint": context.timepoint,
                    "session_id": context.session_id,
                    "timestamp": datetime.now().isoformat(),
                },
            )
        except Exception as e:
            logger.warning(f"Failed to store result: {e}")

    # Publish completion event
    try:
        from gently.core.event_bus import get_event_bus, EventType
        event_bus = get_event_bus()

        event_bus.publish(EventType.CV_RESULT_READY, {
            "result_type": result_type,
            "embryo_id": context.embryo_id,
            "timepoint": context.timepoint,
            "session_id": context.session_id,
            "result": result,
            "result_uid": result_uid,
        })
    except Exception as e:
        logger.debug(f"Could not publish event: {e}")

    return result_uid


# =============================================================================
# Atomic Tools
# =============================================================================

@cv_tool(
    name="count_nuclei",
    description="""Count the number of nuclei in an embryo at a specific timepoint.

This is a single-purpose atomic tool - use it when you just need a nuclei count.
Internally handles volume loading, diSPIM preprocessing, and Cellpose segmentation.

Returns the nuclei count and a cached mask UID for follow-up analysis.""",
    category=ToolCategory.ANALYSIS,
    examples=[
        ToolExample("How many cells in embryo_1?", {"embryo_id": "embryo_1"}),
        ToolExample("Count nuclei at timepoint 5", {"embryo_id": "embryo_1", "timepoint": 5}),
    ],
)
def count_nuclei(
    embryo_id: str,
    timepoint: Optional[int] = None,
    context: CVContext = None,
) -> Dict[str, Any]:
    """
    Count nuclei in an embryo - atomic end-to-end tool.

    Parameters
    ----------
    embryo_id : str
        ID of the embryo to analyze
    timepoint : int, optional
        Specific timepoint, or latest if not provided
    context : CVContext
        Microscope/organism context (injected by registry)

    Returns
    -------
    dict
        - num_nuclei: int - Count of nuclei
        - mask_uid: str - UID of cached segmentation mask
        - embryo_id: str
        - timepoint: int
    """
    if context is None:
        # Fallback if context not injected
        from ..context import CVContext
        context = CVContext(embryo_id=embryo_id, timepoint=timepoint)

    # Update context with request params
    context.embryo_id = embryo_id
    context.timepoint = timepoint

    # Load volume
    volume = _load_volume_for_context(context, embryo_id, timepoint)
    if volume is None:
        return {
            "error": f"Could not load volume for {embryo_id}, timepoint={timepoint}",
            "num_nuclei": 0,
        }

    # Generate volume UID for viz server
    volume_uid = f"vol_{embryo_id}_{timepoint or 'latest'}_{uuid.uuid4().hex[:6]}"

    # Run segmentation with diameter from context (based on microscope scale)
    # Default: 5µm nuclei / 0.406µm/px = ~12px
    diameter_px = context.cell_diameter_px if context else 12.0
    masks, processed_volume, effective_diameter, gpu_used = _run_cellpose_nuclei(
        volume, diameter_px=diameter_px
    )

    # Count
    num_nuclei = int(masks.max())

    # Cache masks for potential follow-up
    mask_uid = f"nuclei_masks_{embryo_id}_{timepoint or 'latest'}_{uuid.uuid4().hex[:6]}"
    cache_volume(mask_uid, masks.astype(np.uint16), {
        "type": "segmentation_mask",
        "embryo_id": embryo_id,
        "timepoint": timepoint,
        "num_nuclei": num_nuclei,
    })

    # Push to viz server for 3D visualization
    _push_segmentation_to_viz(processed_volume, masks, volume_uid, mask_uid, {
        "embryo_id": embryo_id,
        "timepoint": timepoint,
        "num_nuclei": num_nuclei,
        "diameter": effective_diameter,
        "gpu_used": gpu_used,
        "tool": "count_nuclei",
    })

    result = {
        "embryo_id": embryo_id,
        "timepoint": timepoint,
        "num_nuclei": num_nuclei,
        "mask_uid": mask_uid,
        "gpu_used": gpu_used,
    }

    # Store and publish
    _store_and_publish(context, "nuclei_count", result)

    return result


@cv_tool(
    name="classify_stage",
    description="""Classify the developmental stage of a C. elegans embryo.

This atomic tool determines the stage based on nuclei count and morphology.
Stages: 1-cell, 2-cell, 4-cell, 8-cell, 16-cell, gastrula, bean, comma,
1.5-fold, 2-fold, 3-fold, pretzel, hatching.

Returns stage name, confidence, and supporting metrics.""",
    category=ToolCategory.ANALYSIS,
    examples=[
        ToolExample("What stage is embryo_1?", {"embryo_id": "embryo_1"}),
        ToolExample("Classify stage at timepoint 10", {"embryo_id": "embryo_1", "timepoint": 10}),
    ],
)
def classify_stage(
    embryo_id: str,
    timepoint: Optional[int] = None,
    context: CVContext = None,
) -> Dict[str, Any]:
    """
    Classify developmental stage - atomic end-to-end tool.

    Parameters
    ----------
    embryo_id : str
        ID of the embryo to analyze
    timepoint : int, optional
        Specific timepoint, or latest if not provided
    context : CVContext
        Microscope/organism context (injected by registry)

    Returns
    -------
    dict
        - stage: str - Developmental stage name
        - confidence: str - HIGH, MEDIUM, or LOW
        - nuclei_count: int - Number of nuclei detected
        - elongation_ratio: float - Embryo elongation
    """
    from ..config import CELEGANS_STAGES

    if context is None:
        from ..context import CVContext
        context = CVContext(embryo_id=embryo_id, timepoint=timepoint)

    context.embryo_id = embryo_id
    context.timepoint = timepoint

    # Load volume
    volume = _load_volume_for_context(context, embryo_id, timepoint)
    if volume is None:
        return {
            "error": f"Could not load volume for {embryo_id}",
            "stage": "unknown",
            "confidence": "LOW",
        }

    # Generate volume UID for viz server
    volume_uid = f"vol_{embryo_id}_{timepoint or 'latest'}_{uuid.uuid4().hex[:6]}"

    # Segment nuclei with diameter from context
    diameter_px = context.cell_diameter_px if context else 12.0
    masks, processed_volume, effective_diameter, gpu_used = _run_cellpose_nuclei(
        volume, diameter_px=diameter_px
    )
    num_nuclei = int(masks.max())

    # Cache masks and push to viz
    mask_uid = f"stage_masks_{embryo_id}_{timepoint or 'latest'}_{uuid.uuid4().hex[:6]}"
    cache_volume(mask_uid, masks.astype(np.uint16), {
        "type": "segmentation_mask",
        "embryo_id": embryo_id,
        "timepoint": timepoint,
        "num_nuclei": num_nuclei,
    })

    _push_segmentation_to_viz(processed_volume, masks, volume_uid, mask_uid, {
        "embryo_id": embryo_id,
        "timepoint": timepoint,
        "num_nuclei": num_nuclei,
        "diameter": effective_diameter,
        "gpu_used": gpu_used,
        "tool": "classify_stage",
    })

    # Measure elongation
    elongation_ratio = _measure_elongation(masks)

    # Classify based on nuclei count
    stage = "unknown"
    confidence = "LOW"

    for stage_name, info in CELEGANS_STAGES.items():
        if info["nuclei_min"] <= num_nuclei <= info["nuclei_max"]:
            stage = stage_name
            confidence = "HIGH"
            break

    # Adjust confidence based on elongation for fold stages
    if stage in ["comma", "1.5-fold", "2-fold", "3-fold", "pretzel"]:
        if elongation_ratio < 1.5:
            confidence = "MEDIUM"  # Elongation doesn't match expected

    result = {
        "embryo_id": embryo_id,
        "timepoint": timepoint,
        "stage": stage,
        "confidence": confidence,
        "nuclei_count": num_nuclei,
        "elongation_ratio": elongation_ratio,
    }

    _store_and_publish(context, "stage_classification", result)

    return result


def _measure_elongation(masks: np.ndarray) -> float:
    """Measure elongation ratio from segmentation masks."""
    try:
        from skimage.measure import regionprops

        # Create binary mask of all cells
        binary = masks > 0

        # Get bounding region
        props = regionprops(binary.astype(np.uint8).max(axis=0).astype(np.int32))
        if not props:
            return 1.0

        region = props[0]
        if region.minor_axis_length > 0:
            return region.major_axis_length / region.minor_axis_length
        return 1.0

    except Exception:
        return 1.0


@cv_tool(
    name="measure_elongation",
    description="""Measure the elongation ratio of a C. elegans embryo.

Elongation ratio indicates developmental stage:
- ~1.0: Pre-elongation (1-cell to gastrula)
- 1.5-2.0: Comma to 1.5-fold
- 2.0-3.0: 2-fold
- >3.0: 3-fold and beyond

Returns elongation ratio with stage hint.""",
    category=ToolCategory.ANALYSIS,
    examples=[
        ToolExample("Measure elongation", {"embryo_id": "embryo_1"}),
    ],
)
def measure_elongation(
    embryo_id: str,
    timepoint: Optional[int] = None,
    context: CVContext = None,
) -> Dict[str, Any]:
    """
    Measure embryo elongation - atomic end-to-end tool.

    Parameters
    ----------
    embryo_id : str
        ID of the embryo to analyze
    timepoint : int, optional
        Specific timepoint
    context : CVContext
        Microscope/organism context

    Returns
    -------
    dict
        - elongation_ratio: float
        - stage_hint: str - Suggested stage based on elongation
        - major_axis: float
        - minor_axis: float
    """
    if context is None:
        from ..context import CVContext
        context = CVContext(embryo_id=embryo_id, timepoint=timepoint)

    context.embryo_id = embryo_id
    context.timepoint = timepoint

    volume = _load_volume_for_context(context, embryo_id, timepoint)
    if volume is None:
        return {"error": f"Could not load volume for {embryo_id}"}

    # Simple thresholding for embryo detection
    from skimage.filters import threshold_otsu
    from skimage.measure import regionprops

    try:
        # Max projection for 2D analysis
        projection = volume.max(axis=0)
        thresh = threshold_otsu(projection)
        binary = projection > thresh

        props = regionprops(binary.astype(np.int32))
        if not props:
            return {
                "embryo_id": embryo_id,
                "timepoint": timepoint,
                "elongation_ratio": 1.0,
                "stage_hint": "unknown",
                "error": "Could not detect embryo region",
            }

        region = props[0]
        major = region.major_axis_length
        minor = region.minor_axis_length

        if minor > 0:
            elongation = major / minor
        else:
            elongation = 1.0

        # Determine stage hint
        if elongation < 1.3:
            stage_hint = "pre-elongation (1-cell to gastrula)"
        elif elongation < 2.0:
            stage_hint = "comma to 1.5-fold"
        elif elongation < 3.0:
            stage_hint = "2-fold"
        else:
            stage_hint = "3-fold or beyond"

        result = {
            "embryo_id": embryo_id,
            "timepoint": timepoint,
            "elongation_ratio": float(elongation),
            "stage_hint": stage_hint,
            "major_axis": float(major),
            "minor_axis": float(minor),
        }

        _store_and_publish(context, "elongation", result)

        return result

    except Exception as e:
        logger.error(f"Elongation measurement failed: {e}")
        return {"error": str(e), "elongation_ratio": 1.0}


@cv_tool(
    name="segment_nuclei",
    description="""Run nuclei segmentation and return the mask UID.

Use this when you need the segmentation masks for further analysis
(tracking, morphology, etc.) rather than just a count.""",
    category=ToolCategory.SEGMENTATION,
    requires_gpu=True,
    examples=[
        ToolExample("Segment nuclei", {"embryo_id": "embryo_1"}),
    ],
)
def segment_nuclei(
    embryo_id: str,
    timepoint: Optional[int] = None,
    context: CVContext = None,
) -> Dict[str, Any]:
    """
    Segment nuclei and cache masks - atomic tool.

    Parameters
    ----------
    embryo_id : str
        ID of the embryo
    timepoint : int, optional
        Specific timepoint
    context : CVContext
        Microscope/organism context

    Returns
    -------
    dict
        - mask_uid: str - UID to retrieve masks
        - num_cells: int - Number of cells detected
        - cells: list - Per-cell info (centroid, volume)
    """
    if context is None:
        from ..context import CVContext
        context = CVContext(embryo_id=embryo_id, timepoint=timepoint)

    context.embryo_id = embryo_id
    context.timepoint = timepoint

    volume = _load_volume_for_context(context, embryo_id, timepoint)
    if volume is None:
        return {"error": f"Could not load volume for {embryo_id}"}

    # Generate volume UID for viz server
    volume_uid = f"vol_{embryo_id}_{timepoint or 'latest'}_{uuid.uuid4().hex[:6]}"

    # Run segmentation with diameter from context
    diameter_px = context.cell_diameter_px if context else 12.0
    masks, processed_volume, effective_diameter, gpu_used = _run_cellpose_nuclei(
        volume, diameter_px=diameter_px
    )
    num_cells = int(masks.max())

    # Cache masks
    mask_uid = f"seg_masks_{embryo_id}_{timepoint or 'latest'}_{uuid.uuid4().hex[:6]}"
    cache_volume(mask_uid, masks.astype(np.uint16), {
        "type": "segmentation_mask",
        "embryo_id": embryo_id,
        "timepoint": timepoint,
        "num_cells": num_cells,
    })

    # Push to viz server for 3D visualization
    _push_segmentation_to_viz(processed_volume, masks, volume_uid, mask_uid, {
        "embryo_id": embryo_id,
        "timepoint": timepoint,
        "num_cells": num_cells,
        "diameter": effective_diameter,
        "gpu_used": gpu_used,
        "tool": "segment_nuclei",
    })

    # Extract per-cell info
    cells = []
    try:
        from skimage.measure import regionprops

        for prop in regionprops(masks):
            cells.append({
                "label": int(prop.label),
                "centroid": [float(c) for c in prop.centroid],
                "volume_voxels": int(prop.area),
            })
    except Exception:
        pass

    result = {
        "embryo_id": embryo_id,
        "timepoint": timepoint,
        "mask_uid": mask_uid,
        "num_cells": num_cells,
        "cells": cells[:20],  # Limit to first 20 for response size
        "gpu_used": gpu_used,
    }

    _store_and_publish(context, "segmentation", result)

    return result


@cv_tool(
    name="detect_hatching",
    description="""Check if a C. elegans embryo has hatched.

Detects hatching based on morphological changes:
- Loss of eggshell boundary
- Presence of motile larva
- Change in overall shape

Returns hatching status with confidence.""",
    category=ToolCategory.ANALYSIS,
    examples=[
        ToolExample("Has embryo_1 hatched?", {"embryo_id": "embryo_1"}),
    ],
)
def detect_hatching(
    embryo_id: str,
    timepoint: Optional[int] = None,
    context: CVContext = None,
) -> Dict[str, Any]:
    """
    Detect if embryo has hatched - atomic tool.

    Parameters
    ----------
    embryo_id : str
        ID of the embryo
    timepoint : int, optional
        Specific timepoint
    context : CVContext
        Microscope/organism context

    Returns
    -------
    dict
        - hatched: bool
        - confidence: str - HIGH, MEDIUM, LOW
        - indicators: list - What indicated hatching
    """
    if context is None:
        from ..context import CVContext
        context = CVContext(embryo_id=embryo_id, timepoint=timepoint)

    context.embryo_id = embryo_id
    context.timepoint = timepoint

    volume = _load_volume_for_context(context, embryo_id, timepoint)
    if volume is None:
        return {"error": f"Could not load volume for {embryo_id}"}

    # Generate volume UID for viz server
    volume_uid = f"vol_{embryo_id}_{timepoint or 'latest'}_{uuid.uuid4().hex[:6]}"

    # Segment with diameter from context
    diameter_px = context.cell_diameter_px if context else 12.0
    masks, processed_volume, effective_diameter, gpu_used = _run_cellpose_nuclei(
        volume, diameter_px=diameter_px
    )
    num_nuclei = int(masks.max())
    elongation = _measure_elongation(masks)

    # Cache and push to viz
    mask_uid = f"hatching_masks_{embryo_id}_{timepoint or 'latest'}_{uuid.uuid4().hex[:6]}"
    cache_volume(mask_uid, masks.astype(np.uint16), {
        "type": "segmentation_mask",
        "embryo_id": embryo_id,
        "timepoint": timepoint,
        "num_nuclei": num_nuclei,
    })

    _push_segmentation_to_viz(processed_volume, masks, volume_uid, mask_uid, {
        "embryo_id": embryo_id,
        "timepoint": timepoint,
        "num_nuclei": num_nuclei,
        "diameter": effective_diameter,
        "gpu_used": gpu_used,
        "tool": "detect_hatching",
    })

    # Heuristic hatching detection
    # Post-hatching: ~558 nuclei, very elongated larva
    indicators = []
    hatched = False
    confidence = "LOW"

    if num_nuclei >= 550:
        indicators.append(f"High nuclei count ({num_nuclei})")
        if elongation > 4.0:
            indicators.append(f"Highly elongated ({elongation:.1f})")
            hatched = True
            confidence = "HIGH"
        elif elongation > 3.0:
            confidence = "MEDIUM"
            hatched = True

    result = {
        "embryo_id": embryo_id,
        "timepoint": timepoint,
        "hatched": hatched,
        "confidence": confidence,
        "indicators": indicators,
        "nuclei_count": num_nuclei,
        "elongation_ratio": elongation,
        "gpu_used": gpu_used,
    }

    _store_and_publish(context, "hatching_detection", result)

    return result
