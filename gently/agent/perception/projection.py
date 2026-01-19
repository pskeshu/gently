"""
Projection utilities for perception system.

Generates three-view orthogonal projections from 3D volumes.
Extracted from gently/dataset/explorer_server.py for shared use.
"""

import base64
import io
from pathlib import Path
from typing import Tuple

import numpy as np
from PIL import Image as PIL_Image

# Optional tifffile import (only needed for load_volume)
try:
    import tifffile
except ImportError:
    tifffile = None


def normalize_image(img: np.ndarray, p_low: float = 1, p_high: float = 99) -> np.ndarray:
    """Normalize image to 0-255 uint8 using percentile scaling."""
    img = img.astype(np.float32)
    vmin = np.percentile(img, p_low)
    vmax = np.percentile(img, p_high)
    if vmax > vmin:
        img = np.clip((img - vmin) / (vmax - vmin), 0, 1)
    else:
        img = np.zeros_like(img)
    return (img * 255).astype(np.uint8)


def image_to_base64(img: np.ndarray, format: str = "JPEG", quality: int = 90) -> str:
    """Convert numpy array to base64-encoded image."""
    pil_img = PIL_Image.fromarray(img)
    if pil_img.mode not in ('RGB', 'RGBA'):
        pil_img = pil_img.convert('RGB')
    buffer = io.BytesIO()
    pil_img.save(buffer, format=format, quality=quality)
    return base64.b64encode(buffer.getvalue()).decode()


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
