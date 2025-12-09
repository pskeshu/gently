"""
Image Preparation Tools for CV Agent

Tools for preparing volumes for Claude Vision analysis.
These tools implement the context enrichment pattern:
1. Detect and frame the embryo ROI
2. Crop with appropriate padding
3. Add scale bars and annotations
4. Project 3D to 2D for vision analysis
"""

import base64
import io
import logging
import os
import uuid
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .registry import cv_tool, ToolCategory, ToolExample, ToolParameter
from .data_access import get_cached_volume, cache_volume

# Cache for prepared images (base64) - separate from volume cache
_prepared_images: Dict[str, str] = {}


def cache_prepared_image(uid: str, base64_data: str):
    """Cache a prepared base64 image"""
    _prepared_images[uid] = base64_data
    # Keep cache bounded
    if len(_prepared_images) > 50:
        # Remove oldest entries
        keys = list(_prepared_images.keys())
        for k in keys[:10]:
            del _prepared_images[k]


def get_prepared_image(uid: str) -> Optional[str]:
    """Get a cached prepared image by UID"""
    return _prepared_images.get(uid)

logger = logging.getLogger(__name__)

# Viz server URL for pushing visualizations
VIZ_SERVER_URL = os.environ.get("VIZ_SERVER_URL", "http://localhost:8080")


def _push_image_to_viz(
    array: np.ndarray,
    uid: str,
    data_type: str = "cv_visualization",
    metadata: Dict = None
):
    """Push a 2D image to viz server via HTTP POST.

    The CV subagent runs as a separate process, so we use HTTP to communicate
    with the viz server instead of direct function calls.
    """
    import requests

    try:
        # Ensure array is contiguous for proper serialization
        array = np.ascontiguousarray(array)

        # Encode array as base64
        image_b64 = base64.b64encode(array.tobytes()).decode('ascii')

        payload = {
            'image_b64': image_b64,
            'uid': uid,
            'shape': list(array.shape),
            'dtype': str(array.dtype),
            'data_type': data_type,
            'metadata': metadata or {}
        }

        response = requests.post(
            f"{VIZ_SERVER_URL}/api/images",
            json=payload,
            timeout=10
        )

        if response.status_code == 200:
            logger.info(f"Pushed {data_type} to viz server: {uid}")
        else:
            logger.warning(f"Viz server returned {response.status_code}: {response.text}")

    except Exception as e:
        logger.debug(f"Viz server not available: {e}")

# Try to import image processing libraries
try:
    from PIL import Image, ImageDraw, ImageFont
    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    logger.warning("PIL not available, some preparation features limited")

try:
    from scipy import ndimage
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    logger.warning("scipy not available, using basic ROI detection")

try:
    from skimage import filters, measure, morphology
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
    logger.warning("skimage not available, using basic methods")


def _ensure_3d(volume: np.ndarray) -> np.ndarray:
    """Ensure volume is 3D (Z, Y, X) by squeezing or taking first channel/timepoint"""
    if volume.ndim == 3:
        return volume
    elif volume.ndim == 4:
        # Could be (T, Z, Y, X) or (C, Z, Y, X) - take first along dim 0
        if volume.shape[0] == 1:
            return volume[0]
        else:
            # Take first timepoint/channel
            logger.warning(f"4D volume {volume.shape}, using first frame")
            return volume[0]
    elif volume.ndim == 2:
        # Add Z dimension
        return volume[np.newaxis, :, :]
    elif volume.ndim > 4:
        # Squeeze all singleton dimensions
        squeezed = np.squeeze(volume)
        if squeezed.ndim == 3:
            return squeezed
        elif squeezed.ndim == 4:
            return squeezed[0]
        else:
            logger.warning(f"Cannot reduce {volume.shape} to 3D, taking slice")
            return volume.reshape(-1, volume.shape[-2], volume.shape[-1])[:volume.shape[-3]]
    else:
        return volume


@cv_tool(
    name="detect_embryo_roi",
    description="Detect embryo bounding box for proper framing. Returns bbox and center coordinates.",
    category=ToolCategory.PREPARATION,
    examples=[
        ToolExample("Find the embryo in volume abc", {"volume_uid": "abc"}),
        ToolExample("Detect embryo using Otsu thresholding", {"volume_uid": "xyz", "method": "otsu"}),
    ],
    parameters=[
        ToolParameter(name="volume_uid", type="string", description="UID of the volume to analyze", required=True),
        ToolParameter(name="method", type="string", description="Detection method",
                      required=False, default="threshold", enum=["threshold", "otsu", "adaptive"]),
        ToolParameter(name="min_size_voxels", type="integer", description="Minimum object size to consider as embryo",
                      required=False, default=1000),
    ],
)
def detect_embryo_roi(
    volume_uid: str,
    method: str = "threshold",
    min_size_voxels: int = 1000,
) -> Dict[str, Any]:
    """
    Find embryo bounding box in a volume

    Parameters
    ----------
    volume_uid : str
        UID of the volume to analyze
    method : str
        Detection method: "threshold", "otsu", or "adaptive"
    min_size_voxels : int
        Minimum object size to consider as embryo

    Returns
    -------
    dict
        bbox: [z1, y1, x1, z2, y2, x2] bounding box
        center: [z, y, x] center coordinates
        volume_voxels: number of voxels in detected region
        confidence: detection confidence (0-1)
    """
    logger.info(f"Detecting embryo ROI in {volume_uid}, method={method}")

    volume = get_cached_volume(volume_uid)
    if volume is None:
        return {
            "error": f"Volume {volume_uid} not found in cache",
            "bbox": None,
            "center": None,
        }

    # Ensure volume is 3D
    volume = _ensure_3d(volume)

    try:
        # Get binary mask of embryo
        if method == "otsu" and HAS_SKIMAGE:
            threshold = filters.threshold_otsu(volume)
            mask = volume > threshold
        elif method == "adaptive" and HAS_SKIMAGE:
            # Use local adaptive thresholding
            threshold = filters.threshold_local(
                volume.max(axis=0),  # 2D projection for speed
                block_size=51,
            )
            mask_2d = volume.max(axis=0) > threshold
            # Extend to 3D
            mask = np.zeros_like(volume, dtype=bool)
            for z in range(volume.shape[0]):
                mask[z] = mask_2d
        else:
            # Simple threshold method
            threshold = np.percentile(volume, 75)
            mask = volume > threshold

        # Clean up mask
        if HAS_SCIPY:
            # Remove small objects
            labeled, num_features = ndimage.label(mask)
            if num_features > 0:
                sizes = ndimage.sum(mask, labeled, range(1, num_features + 1))
                # Keep only largest object above minimum size
                valid_labels = [i + 1 for i, s in enumerate(sizes) if s >= min_size_voxels]
                if valid_labels:
                    # Keep the largest
                    largest_label = valid_labels[np.argmax([sizes[i-1] for i in valid_labels])]
                    mask = labeled == largest_label

        # Find bounding box
        coords = np.argwhere(mask)
        if len(coords) == 0:
            # No embryo found, return full volume
            return {
                "bbox": [0, 0, 0, volume.shape[0], volume.shape[1], volume.shape[2]],
                "center": [s // 2 for s in volume.shape],
                "volume_voxels": 0,
                "confidence": 0.0,
                "message": "No embryo detected, returning full volume",
            }

        z_min, y_min, x_min = coords.min(axis=0)
        z_max, y_max, x_max = coords.max(axis=0)

        # Calculate center
        center = [
            (z_min + z_max) // 2,
            (y_min + y_max) // 2,
            (x_min + x_max) // 2,
        ]

        # Calculate confidence based on object properties
        volume_voxels = np.sum(mask)
        total_voxels = np.prod(volume.shape)
        # Embryo should be a reasonable fraction of the volume
        fill_ratio = volume_voxels / total_voxels
        confidence = min(1.0, fill_ratio * 10)  # Scale up small ratios

        # Create visualization: max projection with ROI box
        try:
            max_proj = np.max(volume, axis=0)
            # Normalize to 0-255
            max_proj = ((max_proj - max_proj.min()) / (max_proj.max() - max_proj.min() + 1e-8) * 255).astype(np.uint8)
            # Convert to RGB
            vis_img = np.stack([max_proj, max_proj, max_proj], axis=-1)
            # Draw ROI rectangle (green)
            # Top and bottom edges
            vis_img[y_min, x_min:x_max, :] = [0, 255, 0]
            vis_img[min(y_max, vis_img.shape[0]-1), x_min:x_max, :] = [0, 255, 0]
            # Left and right edges
            vis_img[y_min:y_max, x_min, :] = [0, 255, 0]
            vis_img[y_min:y_max, min(x_max, vis_img.shape[1]-1), :] = [0, 255, 0]
            # Draw center crosshair (red)
            cy, cx = center[1], center[2]
            vis_img[max(0, cy-5):min(vis_img.shape[0], cy+5), cx, :] = [255, 0, 0]
            vis_img[cy, max(0, cx-5):min(vis_img.shape[1], cx+5), :] = [255, 0, 0]
            # Push to viz server
            roi_uid = f"roi_{volume_uid}_{uuid.uuid4().hex[:6]}"
            _push_image_to_viz(vis_img, roi_uid, "roi_detection", {
                "source_uid": volume_uid,
                "bbox": [int(z_min), int(y_min), int(x_min), int(z_max), int(y_max), int(x_max)],
                "confidence": float(confidence),
                "method": method,
            })
        except Exception as e:
            logger.debug(f"ROI visualization failed: {e}")

        return {
            "bbox": [int(z_min), int(y_min), int(x_min), int(z_max), int(y_max), int(x_max)],
            "center": [int(c) for c in center],
            "volume_voxels": int(volume_voxels),
            "confidence": float(confidence),
            "method": method,
        }

    except Exception as e:
        logger.error(f"ROI detection failed: {e}")
        # Return full volume as fallback
        return {
            "bbox": [0, 0, 0, volume.shape[0], volume.shape[1], volume.shape[2]],
            "center": [s // 2 for s in volume.shape],
            "volume_voxels": 0,
            "confidence": 0.0,
            "error": str(e),
        }


@cv_tool(
    name="crop_roi",
    description="Crop volume to region of interest with padding for context.",
    category=ToolCategory.PREPARATION,
    examples=[
        ToolExample("Crop volume to detected embryo region", {"volume_uid": "vol_abc", "bbox": [10, 50, 50, 40, 200, 200]}),
        ToolExample("Crop with extra padding", {"volume_uid": "vol_xyz", "bbox": [5, 30, 30, 45, 220, 220], "padding": 20}),
    ],
)
def crop_roi(
    volume_uid: str,
    bbox: List[int],
    padding_percent: float = 20.0,
) -> Dict[str, Any]:
    """
    Crop volume to bounding box with padding

    Parameters
    ----------
    volume_uid : str
        UID of volume to crop
    bbox : list
        Bounding box [z1, y1, x1, z2, y2, x2]
    padding_percent : float
        Padding to add around bbox as percentage

    Returns
    -------
    dict
        cropped_uid: UID of the cropped volume (cached)
        shape: New volume dimensions
        actual_bbox: Final bbox after padding applied
    """
    logger.info(f"Cropping {volume_uid} with {padding_percent}% padding")

    volume = get_cached_volume(volume_uid)
    if volume is None:
        return {"error": f"Volume {volume_uid} not found in cache"}

    # Ensure volume is 3D
    volume = _ensure_3d(volume)

    z1, y1, x1, z2, y2, x2 = bbox

    # Calculate padding in voxels
    z_pad = int((z2 - z1) * padding_percent / 100)
    y_pad = int((y2 - y1) * padding_percent / 100)
    x_pad = int((x2 - x1) * padding_percent / 100)

    # Apply padding with bounds checking
    z1_pad = max(0, z1 - z_pad)
    y1_pad = max(0, y1 - y_pad)
    x1_pad = max(0, x1 - x_pad)
    z2_pad = min(volume.shape[0], z2 + z_pad)
    y2_pad = min(volume.shape[1], y2 + y_pad)
    x2_pad = min(volume.shape[2], x2 + x_pad)

    # Crop
    cropped = volume[z1_pad:z2_pad, y1_pad:y2_pad, x1_pad:x2_pad].copy()

    # Cache the cropped volume
    cropped_uid = f"cropped_{uuid.uuid4().hex[:8]}"
    cache_volume(cropped_uid, cropped, {
        "source_uid": volume_uid,
        "original_bbox": bbox,
        "applied_bbox": [z1_pad, y1_pad, x1_pad, z2_pad, y2_pad, x2_pad],
        "padding_percent": padding_percent,
    })

    # Push cropped visualization to viz server
    try:
        max_proj = np.max(cropped, axis=0)
        max_proj = ((max_proj - max_proj.min()) / (max_proj.max() - max_proj.min() + 1e-8) * 255).astype(np.uint8)
        _push_image_to_viz(max_proj, cropped_uid, "cropped_roi", {
            "source_uid": volume_uid,
            "bbox": [z1_pad, y1_pad, x1_pad, z2_pad, y2_pad, x2_pad],
            "shape": list(cropped.shape),
        })
    except Exception as e:
        logger.debug(f"Crop viz push failed: {e}")

    return {
        "cropped_uid": cropped_uid,
        "shape": list(cropped.shape),
        "original_bbox": bbox,
        "actual_bbox": [z1_pad, y1_pad, x1_pad, z2_pad, y2_pad, x2_pad],
        "padding_applied": [z_pad, y_pad, x_pad],
    }


@cv_tool(
    name="prepare_for_vision",
    description="Prepare volume for Claude Vision: add scale bar, annotations, project to 2D. Returns base64 image.",
    category=ToolCategory.PREPARATION,
    examples=[
        ToolExample("Prepare volume for vision analysis", {"volume_uid": "vol_abc"}),
        ToolExample("Create mean projection with annotations", {"volume_uid": "vol_xyz", "projection": "mean", "annotations": {"stage": "gastrula"}}),
        ToolExample("Show specific slice", {"volume_uid": "vol_abc", "projection": "slice", "slice_index": 25}),
    ],
    parameters=[
        ToolParameter(name="volume_uid", type="string", description="UID of volume to prepare", required=True),
        ToolParameter(name="scale_bar_um", type="number", description="Scale bar length in micrometers", required=False, default=10.0),
        ToolParameter(name="scale_um_per_px", type="number", description="Pixel size in micrometers", required=False, default=0.5),
        ToolParameter(name="annotations", type="object", description="Annotations to overlay: {\"nuclei\": 24, \"stage\": \"gastrula\"}", required=False),
        ToolParameter(name="projection", type="string", description="Projection method for 3D to 2D",
                      required=False, default="max", enum=["max", "mean", "sum", "slice"]),
        ToolParameter(name="slice_index", type="integer", description="Slice index if projection='slice'", required=False),
        ToolParameter(name="colormap", type="string", description="Colormap name (gray, viridis, etc.)", required=False, default="gray"),
        ToolParameter(name="contrast_percentile", type="array", description="Percentile range for contrast [low, high]", required=False, default=[1, 99]),
    ],
)
def prepare_for_vision(
    volume_uid: str,
    scale_bar_um: float = 10.0,
    scale_um_per_px: float = 0.5,
    annotations: Optional[Dict[str, Any]] = None,
    projection: str = "max",
    slice_index: Optional[int] = None,
    colormap: str = "gray",
    contrast_percentile: Tuple[float, float] = (1, 99),
) -> Dict[str, Any]:
    """
    Prepare a volume for Claude Vision analysis

    Creates a 2D projection with scale bar and annotations.

    Parameters
    ----------
    volume_uid : str
        UID of volume to prepare
    scale_bar_um : float
        Scale bar length in micrometers
    scale_um_per_px : float
        Pixel size in micrometers
    annotations : dict, optional
        Annotations to overlay: {"nuclei": 24, "stage": "gastrula", ...}
    projection : str
        Projection method: "max", "mean", "sum", or "slice"
    slice_index : int, optional
        Slice index if projection="slice"
    colormap : str
        Colormap name (gray, viridis, etc.)
    contrast_percentile : tuple
        Percentile range for contrast adjustment

    Returns
    -------
    dict
        image_base64: Base64 encoded PNG image
        width: Image width in pixels
        height: Image height in pixels
        scale_bar_px: Scale bar length in pixels
    """
    logger.info(f"Preparing {volume_uid} for vision, projection={projection}")

    volume = get_cached_volume(volume_uid)
    if volume is None:
        return {"error": f"Volume {volume_uid} not found in cache"}

    if not HAS_PIL:
        return {"error": "PIL not available for image preparation"}

    # Ensure volume is 3D
    volume = _ensure_3d(volume)

    try:
        # Create 2D projection
        if projection == "max":
            image_2d = np.max(volume, axis=0)
        elif projection == "mean":
            image_2d = np.mean(volume, axis=0)
        elif projection == "sum":
            image_2d = np.sum(volume, axis=0)
        elif projection == "slice":
            idx = slice_index if slice_index is not None else volume.shape[0] // 2
            idx = min(max(0, idx), volume.shape[0] - 1)
            image_2d = volume[idx]
        else:
            image_2d = np.max(volume, axis=0)

        # Normalize to 0-255 with contrast adjustment
        p_low, p_high = contrast_percentile
        v_min = np.percentile(image_2d, p_low)
        v_max = np.percentile(image_2d, p_high)
        if v_max > v_min:
            image_norm = np.clip((image_2d - v_min) / (v_max - v_min), 0, 1)
        else:
            image_norm = np.zeros_like(image_2d)
        image_uint8 = (image_norm * 255).astype(np.uint8)

        # Create PIL image
        pil_image = Image.fromarray(image_uint8, mode='L')

        # Convert to RGB for annotations
        pil_image = pil_image.convert('RGB')

        # Add scale bar
        scale_bar_px = int(scale_bar_um / scale_um_per_px)
        draw = ImageDraw.Draw(pil_image)

        # Position scale bar in bottom-left
        margin = 10
        bar_height = 5
        bar_y = pil_image.height - margin - bar_height
        bar_x1 = margin
        bar_x2 = margin + scale_bar_px

        # Draw white bar with black outline
        draw.rectangle([bar_x1-1, bar_y-1, bar_x2+1, bar_y+bar_height+1], fill='black')
        draw.rectangle([bar_x1, bar_y, bar_x2, bar_y+bar_height], fill='white')

        # Add scale text
        scale_text = f"{scale_bar_um:.0f} µm"
        try:
            font = ImageFont.truetype("arial.ttf", 12)
        except (OSError, IOError):
            font = ImageFont.load_default()

        text_bbox = draw.textbbox((0, 0), scale_text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_x = bar_x1 + (scale_bar_px - text_width) // 2
        text_y = bar_y - 15

        # Draw text with outline for visibility
        for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
            draw.text((text_x + dx, text_y + dy), scale_text, fill='black', font=font)
        draw.text((text_x, text_y), scale_text, fill='white', font=font)

        # Add annotations if provided
        if annotations:
            annotation_lines = []
            for key, value in annotations.items():
                if isinstance(value, float):
                    annotation_lines.append(f"{key}: {value:.2f}")
                else:
                    annotation_lines.append(f"{key}: {value}")

            # Draw annotations in top-left
            y_pos = margin
            for line in annotation_lines:
                # Outline for visibility
                for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                    draw.text((margin + dx, y_pos + dy), line, fill='black', font=font)
                draw.text((margin, y_pos), line, fill='cyan', font=font)
                y_pos += 15

        # Convert to base64
        buffer = io.BytesIO()
        pil_image.save(buffer, format='PNG')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

        # Generate UID and cache the prepared image
        prepared_uid = f"prepared_{volume_uid}_{uuid.uuid4().hex[:6]}"
        cache_prepared_image(prepared_uid, image_base64)

        # Push to viz server
        try:
            vis_array = np.array(pil_image)
            _push_image_to_viz(vis_array, prepared_uid, "vision_prepared", {
                "source_uid": volume_uid,
                "projection": projection,
                "scale_bar_um": scale_bar_um,
                "annotations": list(annotations.keys()) if annotations else [],
            })
        except Exception as e:
            logger.debug(f"Vision prep viz push failed: {e}")

        # Return UID instead of base64 to avoid bloating conversation context
        return {
            "prepared_image_uid": prepared_uid,
            "width": pil_image.width,
            "height": pil_image.height,
            "scale_bar_um": scale_bar_um,
            "scale_bar_px": scale_bar_px,
            "scale_um_per_px": scale_um_per_px,
            "projection": projection,
            "annotations_added": list(annotations.keys()) if annotations else [],
            "note": "Use this prepared_image_uid with claude_vision_analyze or classify_developmental_stage",
        }

    except Exception as e:
        logger.error(f"Vision preparation failed: {e}")
        return {"error": str(e)}


@cv_tool(
    name="create_timeline_image",
    description="Create a timeline montage of multiple volumes for temporal analysis.",
    category=ToolCategory.PREPARATION,
    examples=[
        ToolExample("Create timeline of 5 volumes", {"volume_uids": ["v1", "v2", "v3", "v4", "v5"]}),
        ToolExample("Grid layout with labels", {"volume_uids": ["v1", "v2", "v3", "v4"], "labels": ["t=0", "t=1", "t=2", "t=3"], "layout": "grid"}),
    ],
    parameters=[
        ToolParameter(name="volume_uids", type="array", description="List of volume UIDs to include", required=True),
        ToolParameter(name="labels", type="array", description="Labels for each volume (e.g., timepoint labels)", required=False),
        ToolParameter(name="scale_bar_um", type="number", description="Scale bar length in micrometers", required=False, default=10.0),
        ToolParameter(name="scale_um_per_px", type="number", description="Pixel size in micrometers", required=False, default=0.5),
        ToolParameter(name="layout", type="string", description="Layout arrangement",
                      required=False, default="horizontal", enum=["horizontal", "grid"]),
        ToolParameter(name="max_width", type="integer", description="Maximum image width in pixels", required=False, default=1200),
    ],
)
def create_timeline_image(
    volume_uids: List[str],
    labels: Optional[List[str]] = None,
    scale_bar_um: float = 10.0,
    scale_um_per_px: float = 0.5,
    layout: str = "horizontal",
    max_width: int = 1200,
) -> Dict[str, Any]:
    """
    Create a timeline montage of multiple volumes

    Parameters
    ----------
    volume_uids : list
        List of volume UIDs to include
    labels : list, optional
        Labels for each volume (e.g., timepoint labels)
    scale_bar_um : float
        Scale bar length in micrometers
    scale_um_per_px : float
        Pixel size
    layout : str
        Layout: "horizontal" or "grid"
    max_width : int
        Maximum output image width

    Returns
    -------
    dict
        image_base64: Base64 encoded montage
        width, height: Image dimensions
        num_frames: Number of volumes included
    """
    logger.info(f"Creating timeline image from {len(volume_uids)} volumes")

    if not HAS_PIL:
        return {"error": "PIL not available for timeline creation"}

    if not volume_uids:
        return {"error": "No volume UIDs provided"}

    try:
        # Create projections for each volume
        projections = []
        for uid in volume_uids:
            volume = get_cached_volume(uid)
            if volume is not None:
                volume = _ensure_3d(volume)
                proj = np.max(volume, axis=0)
                # Normalize
                p_low, p_high = np.percentile(proj, [1, 99])
                if p_high > p_low:
                    proj = np.clip((proj - p_low) / (p_high - p_low), 0, 1)
                else:
                    proj = np.zeros_like(proj)
                projections.append((proj * 255).astype(np.uint8))
            else:
                logger.warning(f"Volume {uid} not found, skipping")

        if not projections:
            return {"error": "No volumes could be loaded"}

        # Calculate layout
        n_frames = len(projections)
        frame_h, frame_w = projections[0].shape

        if layout == "horizontal":
            # Scale frames to fit max_width
            total_width = frame_w * n_frames
            if total_width > max_width:
                scale = max_width / total_width
                frame_w = int(frame_w * scale)
                frame_h = int(frame_h * scale)

            canvas_w = frame_w * n_frames
            canvas_h = frame_h + 20  # Extra space for labels
        else:
            # Grid layout
            cols = int(np.ceil(np.sqrt(n_frames)))
            rows = int(np.ceil(n_frames / cols))
            canvas_w = frame_w * cols
            canvas_h = (frame_h + 20) * rows

        # Create canvas
        canvas = Image.new('RGB', (canvas_w, canvas_h), color='black')
        draw = ImageDraw.Draw(canvas)

        try:
            font = ImageFont.truetype("arial.ttf", 12)
        except (OSError, IOError):
            font = ImageFont.load_default()

        # Place frames
        for i, proj in enumerate(projections):
            pil_frame = Image.fromarray(proj, mode='L').convert('RGB')

            # Resize if needed
            if pil_frame.size != (frame_w, frame_h):
                pil_frame = pil_frame.resize((frame_w, frame_h), Image.Resampling.LANCZOS)

            if layout == "horizontal":
                x = i * frame_w
                y = 0
            else:
                col = i % cols
                row = i // cols
                x = col * frame_w
                y = row * (frame_h + 20)

            canvas.paste(pil_frame, (x, y))

            # Add label
            if labels and i < len(labels):
                label = labels[i]
            else:
                label = f"t{i}"

            draw.text((x + 5, y + frame_h + 2), label, fill='white', font=font)

        # Add scale bar on first frame
        scale_bar_px = int(scale_bar_um / scale_um_per_px)
        bar_y = frame_h - 10
        draw.rectangle([5, bar_y, 5 + scale_bar_px, bar_y + 3], fill='white')
        draw.text((5, bar_y - 12), f"{scale_bar_um:.0f}µm", fill='white', font=font)

        # Convert to base64
        buffer = io.BytesIO()
        canvas.save(buffer, format='PNG')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')

        # Push to viz server
        try:
            vis_array = np.array(canvas)
            vis_uid = f"timeline_{uuid.uuid4().hex[:8]}"
            _push_image_to_viz(vis_array, vis_uid, "timeline", {
                "num_frames": n_frames,
                "layout": layout,
                "volume_uids": volume_uids[:5],  # First 5 for metadata
                "labels": labels[:5] if labels else None,
            })
        except Exception as e:
            logger.debug(f"Timeline viz push failed: {e}")

        return {
            "image_base64": image_base64,
            "width": canvas_w,
            "height": canvas_h,
            "num_frames": n_frames,
            "layout": layout,
            "frame_size": [frame_w, frame_h],
        }

    except Exception as e:
        logger.error(f"Timeline creation failed: {e}")
        return {"error": str(e)}


@cv_tool(
    name="normalize_volume",
    description="Normalize volume intensity for consistent analysis.",
    category=ToolCategory.PREPARATION,
    examples=[
        ToolExample("Normalize with default percentile method", {"volume_uid": "vol_abc"}),
        ToolExample("Use z-score normalization", {"volume_uid": "vol_xyz", "method": "zscore"}),
    ],
    parameters=[
        ToolParameter(name="volume_uid", type="string", description="UID of volume to normalize", required=True),
        ToolParameter(name="method", type="string", description="Normalization method",
                      required=False, default="percentile", enum=["percentile", "minmax", "zscore"]),
        ToolParameter(name="percentile_range", type="array", description="Percentile range for 'percentile' method [low, high]",
                      required=False, default=[1, 99]),
    ],
)
def normalize_volume(
    volume_uid: str,
    method: str = "percentile",
    percentile_range: Tuple[float, float] = (1, 99),
) -> Dict[str, Any]:
    """
    Normalize volume intensity

    Parameters
    ----------
    volume_uid : str
        UID of volume to normalize
    method : str
        Normalization method: "percentile", "minmax", "zscore"
    percentile_range : tuple
        Percentile range for "percentile" method

    Returns
    -------
    dict
        normalized_uid: UID of normalized volume
        original_range: [min, max] of original
        normalized_range: [min, max] of result
    """
    logger.info(f"Normalizing {volume_uid}, method={method}")

    volume = get_cached_volume(volume_uid)
    if volume is None:
        return {"error": f"Volume {volume_uid} not found in cache"}

    # Ensure volume is 3D
    volume = _ensure_3d(volume)

    original_range = [float(volume.min()), float(volume.max())]

    if method == "percentile":
        p_low, p_high = percentile_range
        v_min = np.percentile(volume, p_low)
        v_max = np.percentile(volume, p_high)
        if v_max > v_min:
            normalized = np.clip((volume - v_min) / (v_max - v_min), 0, 1)
        else:
            normalized = np.zeros_like(volume, dtype=np.float32)
    elif method == "minmax":
        v_min, v_max = volume.min(), volume.max()
        if v_max > v_min:
            normalized = (volume - v_min) / (v_max - v_min)
        else:
            normalized = np.zeros_like(volume, dtype=np.float32)
    elif method == "zscore":
        mean = volume.mean()
        std = volume.std()
        if std > 0:
            normalized = (volume - mean) / std
        else:
            normalized = np.zeros_like(volume, dtype=np.float32)
    else:
        return {"error": f"Unknown normalization method: {method}"}

    # Convert to float32
    normalized = normalized.astype(np.float32)

    # Cache result
    normalized_uid = f"norm_{uuid.uuid4().hex[:8]}"
    cache_volume(normalized_uid, normalized, {
        "source_uid": volume_uid,
        "normalization_method": method,
    })

    return {
        "normalized_uid": normalized_uid,
        "shape": list(normalized.shape),
        "original_range": original_range,
        "normalized_range": [float(normalized.min()), float(normalized.max())],
        "method": method,
    }
