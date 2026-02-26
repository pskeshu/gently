"""
Projection utilities for perception system.

Generates three-view orthogonal projections from 3D volumes,
and a depth-aware alpha-composite view for 3D viewing.
Extracted from gently/dataset/explorer_server.py for shared use.
"""

from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image as PIL_Image

from gently.imaging import normalize_to_uint8
from gently.imaging import image_to_base64 as _image_to_base64

# Optional tifffile import (only needed for load_volume)
try:
    import tifffile
except ImportError:
    tifffile = None


def normalize_image(img: np.ndarray, p_low: float = 1, p_high: float = 99) -> np.ndarray:
    """Normalize image to 0-255 uint8 using percentile scaling.

    Thin wrapper around :func:`gently.imaging.normalize_to_uint8`.
    """
    return normalize_to_uint8(img, method="percentile", p_low=p_low, p_high=p_high)


def image_to_base64(img: np.ndarray, format: str = "JPEG", quality: int = 90) -> str:
    """Convert numpy array to base64-encoded image.

    Thin wrapper around :func:`gently.imaging.image_to_base64` that preserves
    the original default quality (90) and always converts to RGB.
    """
    return _image_to_base64(img, format=format, quality=quality, ensure_rgb=True)


def load_volume(path: Path) -> np.ndarray:
    """Load a volume from TIFF file, extract View A."""
    if tifffile is None:
        raise ImportError("tifffile is required for load_volume")
    vol = tifffile.imread(str(path))
    vol = np.squeeze(vol)
    if vol.ndim == 3:
        z_depth, height, width = vol.shape
        # Extract View A (left half) if dual-view format
        if width > height * 2:
            vol = vol[:, :, :width // 2]
    return vol


def compute_crop_bounds(
    volume: np.ndarray, padding: int = 20, sigma_mult: float = 3.5
) -> Tuple[int, int, int, int]:
    """
    Compute crop bounds for 3D volume using center-of-mass of bright pixels.

    Returns (y_min, y_max, x_min, x_max) bounds for cropping.
    """
    if volume.ndim != 3:
        return (0, volume.shape[0], 0, volume.shape[1])
    max_proj = np.max(volume, axis=0).astype(np.float32)
    threshold = np.percentile(max_proj, 95)
    mask = max_proj > threshold
    y_coords, x_coords = np.where(mask)
    if len(y_coords) < 10:
        return (0, volume.shape[1], 0, volume.shape[2])
    cy, cx = np.mean(y_coords), np.mean(x_coords)
    y_std = max(np.std(y_coords), 20)
    x_std = max(np.std(x_coords), 20)
    y_min = int(max(0, cy - sigma_mult * y_std - padding))
    y_max = int(min(volume.shape[1], cy + sigma_mult * y_std + padding))
    x_min = int(max(0, cx - sigma_mult * x_std - padding))
    x_max = int(min(volume.shape[2], cx + sigma_mult * x_std + padding))
    return (y_min, y_max, x_min, x_max)


def apply_crop_bounds(
    volume: np.ndarray, bounds: Tuple[int, int, int, int]
) -> np.ndarray:
    """Apply pre-computed crop bounds to a volume."""
    y_min, y_max, x_min, x_max = bounds
    return volume[
        :,
        max(0, y_min) : min(volume.shape[1], y_max),
        max(0, x_min) : min(volume.shape[2], x_max),
    ]


def projection_three_view(volume: np.ndarray) -> Tuple[np.ndarray, str]:
    """
    Generate three orthogonal views layout from a 3D volume.

    Layout:
    ┌─────────┬───┬─────────┐
    │   XY    │   │   YZ    │  (TOP ROW)
    │ (top)   │   │ (side)  │
    ├─────────┴───┴─────────┤
    │         XZ            │  (BOTTOM ROW)
    │       (front)         │
    └───────────────────────┘

    Views:
    - XY: Looking down (best for shape, curvature, folding)
    - YZ: Looking from side (best for depth, body height)
    - XZ: Looking from front (best for symmetry, coiling)

    Parameters
    ----------
    volume : np.ndarray
        3D volume with shape (Z, Y, X)

    Returns
    -------
    combined : np.ndarray
        Combined three-view image
    description : str
        Description of the layout
    """
    if volume.ndim != 3:
        return normalize_image(volume), "2D input"

    z_depth, height, width = volume.shape

    # Generate max projections along each axis
    xy_proj = normalize_image(np.max(volume, axis=0))  # Looking down (Z projection)
    xz_proj = normalize_image(np.max(volume, axis=1))  # Looking from front (Y projection)
    yz_proj = normalize_image(np.max(volume, axis=2))  # Looking from side (X projection)

    xy_h, xy_w = xy_proj.shape

    # Scale Z dimension for display (Z is typically undersampled)
    z_display_h = max(xy_h // 3, int(z_depth * 3))

    # Resize XZ projection (front view)
    pil_xz = PIL_Image.fromarray(xz_proj)
    pil_xz = pil_xz.resize((xy_w, z_display_h), PIL_Image.Resampling.LANCZOS)
    xz_scaled = np.array(pil_xz)

    # Resize YZ projection (side view) - rotated to align with XY
    yz_rotated = yz_proj.T
    z_display_w = z_display_h
    pil_yz = PIL_Image.fromarray(yz_rotated)
    pil_yz = pil_yz.resize((z_display_w, xy_h), PIL_Image.Resampling.LANCZOS)
    yz_scaled = np.array(pil_yz)

    # Combine views with separators
    sep = 3
    v_sep = np.ones((xy_h, sep), dtype=np.uint8) * 128

    # Top row: XY | separator | YZ
    top_row = np.concatenate([xy_proj, v_sep, yz_scaled], axis=1)
    total_width = top_row.shape[1]

    # Bottom row: XZ (padded to match width)
    if xz_scaled.shape[1] < total_width:
        pad = np.zeros(
            (xz_scaled.shape[0], total_width - xz_scaled.shape[1]), dtype=np.uint8
        )
        bottom_row = np.concatenate([xz_scaled, pad], axis=1)
    else:
        bottom_row = xz_scaled[:, :total_width]

    # Horizontal separator and combine
    h_sep = np.ones((sep, total_width), dtype=np.uint8) * 128
    combined = np.concatenate([top_row, h_sep, bottom_row], axis=0)

    return combined, "Three-view: [XY|YZ] top, [XZ] bottom"


def render_volume_view(
    volume: np.ndarray,
    rotation_x: float = 0,
    rotation_y: float = 0,
    threshold: float = 0.2,
) -> str:
    """
    Render a 3D volume from a specific viewing angle using alpha compositing.

    Produces a depth-aware view where you can see the embryo's shape and
    structure, not just a flat max projection.

    Parameters
    ----------
    volume : np.ndarray
        3D volume (Z, Y, X) or 4D (Views, Z, Y, X)
    rotation_x : float
        Rotation around X axis in degrees (-90 to 90)
    rotation_y : float
        Rotation around Y axis in degrees (-180 to 180)
    threshold : float
        Intensity threshold for transparency (0-1)

    Returns
    -------
    str
        Base64-encoded JPEG image
    """
    from scipy import ndimage

    # Handle 4D volumes (Views, Z, Y, X) - take first view
    if volume.ndim == 4:
        volume = volume[0]

    # Normalize to 0-1
    vol = volume.astype(np.float32)
    p1, p99 = np.percentile(vol, [1, 99])
    vol = np.clip((vol - p1) / (p99 - p1 + 1e-8), 0, 1)

    # Apply rotations
    if rotation_y != 0:
        vol = ndimage.rotate(vol, rotation_y, axes=(0, 2), reshape=False, order=1)
    if rotation_x != 0:
        vol = ndimage.rotate(vol, rotation_x, axes=(0, 1), reshape=False, order=1)

    # Alpha composite from back to front (same as Three.js stacked slices)
    z_depth = vol.shape[0]
    result = np.zeros(vol.shape[1:], dtype=np.float32)
    accumulated_alpha = np.zeros_like(result)

    for z in range(z_depth):
        slice_val = vol[z]
        # Alpha based on intensity above threshold
        alpha = np.clip((slice_val - threshold) / (1 - threshold + 1e-8), 0, 1) * 0.3

        # Front-to-back compositing
        result += slice_val * alpha * (1 - accumulated_alpha)
        accumulated_alpha += alpha * (1 - accumulated_alpha)

    # Normalize result to 0-255
    if result.max() > 0:
        result = (result / result.max() * 255).astype(np.uint8)
    else:
        result = result.astype(np.uint8)

    return _image_to_base64(result, format="JPEG", quality=85, max_dimension=800)
