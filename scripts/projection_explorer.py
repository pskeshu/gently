#!/usr/bin/env python
"""
Projection Explorer - Test different volume projection methods for perception.

This module provides tools to:
1. Extract volumes from a session on demand
2. Apply different projection methods
3. Compare projections side-by-side
4. Export projections for Claude vision testing

Usage:
    # Interactive explorer with web UI
    python scripts/projection_explorer.py --session 59799c78

    # Export comparison images
    python scripts/projection_explorer.py --session 59799c78 --export --embryo embryo_1

    # List available embryos
    python scripts/projection_explorer.py --session 59799c78 --list

Projection Methods:
    - dual_view: Current TOP + SIDE max projection (baseline)
    - depth_colored: MIP with Z-depth encoded as color (new)
    - multi_slice: Montage of N representative slices (new)
    - subvolume: Top/mid/bottom third projections (new)
"""

import argparse
import base64
import io
import json
import socketserver
import sys
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import numpy as np
from scipy import ndimage

# Lazy imports
tifffile: Any = None
PIL_Image: Any = None
plt: Any = None


def ensure_dependencies():
    """Ensure required dependencies are available."""
    global tifffile, PIL_Image, plt

    try:
        import tifffile as _tifffile

        tifffile = _tifffile
    except ImportError:
        print("ERROR: tifffile required. Install: pip install tifffile")
        sys.exit(1)

    try:
        from PIL import Image as _Image

        PIL_Image = _Image
    except ImportError:
        print("ERROR: Pillow required. Install: pip install Pillow")
        sys.exit(1)

    try:
        import matplotlib
        import matplotlib.pyplot as _plt

        matplotlib.use("Agg")  # Non-interactive backend
        plt = _plt
    except ImportError:
        print("WARNING: matplotlib not available, some features disabled")
        plt = None


# =============================================================================
# Volume Loading
# =============================================================================


def discover_volumes(session_dir: Path, embryo_id: str | None = None) -> dict[str, list[Path]]:
    """Discover volume files in a session directory."""
    if not session_dir.exists():
        return {}

    tif_files = list(session_dir.glob("*.tif")) + list(session_dir.glob("*.tiff"))
    embryo_volumes: dict = {}

    for f in tif_files:
        parts = f.stem.split("_")
        if len(parts) >= 3:
            eid = f"{parts[0]}_{parts[1]}"

            try:
                timestamp_str = f"{parts[2]}_{parts[3]}" if len(parts) >= 4 else parts[2]
                timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
            except (ValueError, IndexError):
                timestamp = datetime.fromtimestamp(f.stat().st_mtime)

            if embryo_id is None or eid == embryo_id:
                if eid not in embryo_volumes:
                    embryo_volumes[eid] = []
                embryo_volumes[eid].append((timestamp, f))

    result = {}
    for eid, volumes in embryo_volumes.items():
        volumes.sort(key=lambda x: x[0])
        result[eid] = [v[1] for v in volumes]

    return result


def load_volume(path: Path, crop_bounds: tuple[int, int, int, int] | None = None) -> np.ndarray:
    """Load a volume from TIFF file, extract View A, and optionally apply crop bounds."""
    vol = tifffile.imread(str(path))
    vol = np.squeeze(vol)

    if vol.ndim == 3:
        z_depth, height, width = vol.shape
        # Extract View A (left half) if dual-view format
        if width > height * 2:
            vol = vol[:, :, : width // 2]

        # Apply crop bounds if provided
        if crop_bounds is not None:
            vol = apply_crop_bounds(vol, crop_bounds)

    return vol


# =============================================================================
# Image Utilities
# =============================================================================


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
    if pil_img.mode not in ("RGB", "RGBA"):
        pil_img = pil_img.convert("RGB")

    buffer = io.BytesIO()
    pil_img.save(buffer, format=format, quality=quality)
    return base64.b64encode(buffer.getvalue()).decode()


def resize_to_height(img: np.ndarray, target_height: int) -> np.ndarray:
    """Resize image to target height, maintaining aspect ratio."""
    pil_img = PIL_Image.fromarray(img)
    ratio = target_height / pil_img.height
    new_width = int(pil_img.width * ratio)
    pil_img = pil_img.resize((new_width, target_height), PIL_Image.Resampling.LANCZOS)
    return np.array(pil_img)


# =============================================================================
# Auto-crop utilities
# =============================================================================


def find_content_bbox(
    img: np.ndarray, threshold_percentile: float = 10, padding: int = 20
) -> tuple[int, int, int, int]:
    """
    Find bounding box of content (non-background) in image.

    Returns (y_min, y_max, x_min, x_max) with padding.
    """
    # Handle RGB images
    if img.ndim == 3:
        gray = np.mean(img, axis=2)
    else:
        gray = img.astype(np.float32)

    # Threshold to find content
    threshold = np.percentile(gray, threshold_percentile)
    mask = gray > threshold

    # Find rows and cols with content
    rows_with_content = np.any(mask, axis=1)
    cols_with_content = np.any(mask, axis=0)

    if not np.any(rows_with_content) or not np.any(cols_with_content):
        # No content found, return full image
        return 0, img.shape[0], 0, img.shape[1] if img.ndim == 2 else img.shape[1]

    y_indices = np.where(rows_with_content)[0]
    x_indices = np.where(cols_with_content)[0]

    y_min = max(0, y_indices[0] - padding)
    y_max = min(img.shape[0], y_indices[-1] + padding)
    x_min = max(0, x_indices[0] - padding)
    x_max = min(img.shape[1], x_indices[-1] + padding)

    return y_min, y_max, x_min, x_max


def auto_crop(img: np.ndarray, threshold_percentile: float = 10, padding: int = 20) -> np.ndarray:
    """Auto-crop image to content region."""
    y_min, y_max, x_min, x_max = find_content_bbox(img, threshold_percentile, padding)
    return img[y_min:y_max, x_min:x_max]


def compute_crop_bounds(
    volume: np.ndarray, padding: int = 20, sigma_mult: float = 3.5
) -> tuple[int, int, int, int]:
    """
    Compute crop bounds for 3D volume using center-of-mass of bright pixels.

    Uses top 5% brightest pixels to find embryo location and extent.
    Returns (y_min, y_max, x_min, x_max).
    """
    if volume.ndim != 3:
        return (0, volume.shape[0], 0, volume.shape[1])

    # Use max projection to find content bounds
    max_proj = np.max(volume, axis=0).astype(np.float32)

    # Use top 5% brightest pixels (the embryo)
    threshold = np.percentile(max_proj, 95)
    mask = max_proj > threshold

    y_coords, x_coords = np.where(mask)
    if len(y_coords) < 10:
        return (0, volume.shape[1], 0, volume.shape[2])  # Full volume

    # Find center of mass and spread
    cy = np.mean(y_coords)
    cx = np.mean(x_coords)
    y_std = max(np.std(y_coords), 20)  # Minimum spread
    x_std = max(np.std(x_coords), 20)

    # Crop to sigma_mult * std around center
    y_min = int(max(0, cy - sigma_mult * y_std - padding))
    y_max = int(min(volume.shape[1], cy + sigma_mult * y_std + padding))
    x_min = int(max(0, cx - sigma_mult * x_std - padding))
    x_max = int(min(volume.shape[2], cx + sigma_mult * x_std + padding))

    return (y_min, y_max, x_min, x_max)


def apply_crop_bounds(volume: np.ndarray, bounds: tuple[int, int, int, int]) -> np.ndarray:
    """Apply pre-computed crop bounds to a volume."""
    y_min, y_max, x_min, x_max = bounds
    # Clamp to actual volume dimensions
    y_min = max(0, y_min)
    y_max = min(volume.shape[1], y_max)
    x_min = max(0, x_min)
    x_max = min(volume.shape[2], x_max)
    return volume[:, y_min:y_max, x_min:x_max]


# =============================================================================
# Projection Methods
# =============================================================================


def find_outer_boundary(img: np.ndarray, percentile: float = 50) -> np.ndarray:
    """
    Find outer boundary of embryo by thresholding and extracting mask edge.

    Returns binary boundary image (0 or 255).
    """
    # Threshold to get embryo mask
    thresh = np.percentile(img, percentile)
    mask = img > thresh

    # Vectorized erosion: pixel is 1 only if all neighbors are 1
    padded = np.pad(mask, 1, mode="constant", constant_values=True)
    eroded = (
        padded[:-2, :-2]
        & padded[:-2, 1:-1]
        & padded[:-2, 2:]
        & padded[1:-1, :-2]
        & padded[1:-1, 1:-1]
        & padded[1:-1, 2:]
        & padded[2:, :-2]
        & padded[2:, 1:-1]
        & padded[2:, 2:]
    )

    # Boundary = mask AND NOT eroded
    boundary = (mask & ~eroded).astype(np.uint8) * 255
    return boundary


def overlay_edges(
    img: np.ndarray, edges: np.ndarray, color: tuple[int, int, int] = (255, 200, 0)
) -> np.ndarray:
    """Overlay edge contours on image in specified color."""
    if img.ndim == 2:
        rgb = np.stack([img, img, img], axis=-1)
    else:
        rgb = img.copy()

    edge_mask = edges > 0
    rgb[edge_mask] = color

    return rgb


def projection_dual_view(volume: np.ndarray) -> tuple[np.ndarray, str]:
    """
    Dual-view projection: TOP above, SIDE below, sharing X axis.
    Includes embryo boundary overlay using Canny edge detection.

    TOP: max projection along Z (looking down, X-Y plane)
    SIDE: max projection along Y (looking from front, X-Z plane)

    Returns:
        (image, description) tuple
    """
    if volume.ndim != 3:
        return normalize_image(volume), "2D input"

    z_depth, height, width = volume.shape

    # TOP: max along Z (axis 0) -> shape (Y, X)
    top_proj = np.max(volume, axis=0)
    # SIDE: max along Y (axis 1) -> shape (Z, X)
    side_proj = np.max(volume, axis=1)

    top_norm = normalize_image(top_proj)
    side_norm = normalize_image(side_proj)

    # Find outer boundary of embryo
    top_edges = find_outer_boundary(top_norm, percentile=50)
    side_edges = find_outer_boundary(side_norm, percentile=50)

    # Overlay edges on images
    top_rgb = overlay_edges(top_norm, top_edges, color=(255, 200, 0))
    side_rgb = overlay_edges(side_norm, side_edges, color=(255, 200, 0))

    # Scale side view to match top width and make Z dimension visible
    target_width = top_rgb.shape[1]
    side_new_h = max(height // 3, int(z_depth * 3))

    pil_side = PIL_Image.fromarray(side_rgb)
    pil_side = pil_side.resize((target_width, side_new_h), PIL_Image.Resampling.LANCZOS)
    side_scaled = np.array(pil_side)

    # Combine vertically: TOP on top, SIDE below
    sep = np.ones((3, target_width, 3), dtype=np.uint8) * 128
    combined = np.concatenate([top_rgb, sep, side_scaled], axis=0)

    return combined, "Dual-view MIP with boundary: TOP (XY) above, SIDE (XZ) below"


def projection_depth_colored(volume: np.ndarray, colormap: str = "turbo") -> tuple[np.ndarray, str]:
    """
    Depth-colored max intensity projection with side view.

    First, create a Z-depth colored volume (each slice colored by its Z position).
    Then project from TOP (along Z) and SIDE (along Y).

    Both views show the same Z-depth coloring, allowing you to trace
    structures between views using consistent colors.

    Returns:
        (RGB image, description) tuple
    """
    if volume.ndim != 3:
        gray = normalize_image(volume)
        return np.stack([gray, gray, gray], axis=-1), "2D input"

    z_depth, height, width = volume.shape

    # Get colormap
    if plt is not None:
        cmap = plt.get_cmap(colormap)
    else:
        cmap = None

    # Create Z-depth colored volume: each slice gets colored by its Z position
    # Shape: (Z, Y, X, 3)
    colored_volume = np.zeros((z_depth, height, width, 3), dtype=np.float32)

    for z in range(z_depth):
        z_norm = z / max(1, z_depth - 1)  # 0 to 1

        # Get color for this depth
        if cmap is not None:
            color = np.array(cmap(z_norm)[:3])  # RGB
        else:
            color = np.array([z_norm, 0.5, 1 - z_norm])  # Simple gradient

        # Get intensity for this slice, normalized
        slice_data = volume[z].astype(np.float32)
        slice_norm = (slice_data - slice_data.min()) / max(1, slice_data.max() - slice_data.min())

        # Color the slice: intensity * depth_color
        colored_volume[z] = slice_norm[:, :, np.newaxis] * color

    # === TOP VIEW: Max projection along Z of colored volume ===
    top_rgb = np.max(colored_volume, axis=0)  # Shape: (Y, X, 3)
    top_rgb = (top_rgb * 255).astype(np.uint8)

    # === SIDE VIEW: Max projection along Y of colored volume ===
    side_rgb = np.max(colored_volume, axis=1)  # Shape: (Z, X, 3)
    side_rgb = (side_rgb * 255).astype(np.uint8)

    # Scale side view to match top width and reasonable height
    pil_side = PIL_Image.fromarray(side_rgb)
    side_new_h = max(height // 3, int(z_depth * 3))  # Make Z dimension more visible
    pil_side = pil_side.resize((width, side_new_h), PIL_Image.Resampling.LANCZOS)
    side_scaled = np.array(pil_side)

    # Combine vertically: TOP above, SIDE below
    sep = np.ones((3, width, 3), dtype=np.uint8) * 128
    combined = np.concatenate([top_rgb, sep, side_scaled], axis=0)

    return combined, f"Z-depth colored MIP ({colormap}): TOP + SIDE views"


def projection_multi_slice(volume: np.ndarray, n_slices: int = 6) -> tuple[np.ndarray, str]:
    """
    Montage of N representative z-slices.

    Returns:
        (image montage, description) tuple
    """
    if volume.ndim != 3:
        return normalize_image(volume), "2D input"

    z_depth, height, width = volume.shape

    # Select evenly-spaced slices
    if z_depth <= n_slices:
        indices = list(range(z_depth))
    else:
        indices = np.linspace(0, z_depth - 1, n_slices, dtype=int).tolist()

    # Normalize each slice
    slices = [normalize_image(volume[i]) for i in indices]

    # Arrange in 2 rows
    n_cols = (len(slices) + 1) // 2
    n_rows = 2

    # Pad to fill grid
    while len(slices) < n_rows * n_cols:
        slices.append(np.zeros_like(slices[0]))

    # Build montage
    rows = []
    for r in range(n_rows):
        row_slices = slices[r * n_cols : (r + 1) * n_cols]
        # Add thin separator between slices
        sep = np.ones((height, 2), dtype=np.uint8) * 64
        row_with_sep = []
        for i, s in enumerate(row_slices):
            if i > 0:
                row_with_sep.append(sep)
            row_with_sep.append(s)
        rows.append(np.concatenate(row_with_sep, axis=1))

    # Add horizontal separator between rows
    row_width = rows[0].shape[1]
    h_sep = np.ones((2, row_width), dtype=np.uint8) * 64
    montage = np.concatenate([rows[0], h_sep, rows[1]], axis=0)

    z_labels = ", ".join([f"z={i}" for i in indices])
    return montage, f"Multi-slice montage ({n_slices} slices: {z_labels})"


def projection_subvolume(volume: np.ndarray) -> tuple[np.ndarray, str]:
    """
    Sub-volume projections: top third, middle third, bottom third.

    Useful for separating structures at different depths.

    Returns:
        (image with 3 projections, description) tuple
    """
    if volume.ndim != 3:
        return normalize_image(volume), "2D input"

    z_depth, height, width = volume.shape

    # Divide into thirds
    z1 = z_depth // 3
    z2 = 2 * z_depth // 3

    top_vol = volume[:z1]
    mid_vol = volume[z1:z2]
    bot_vol = volume[z2:]

    # Project each
    top_proj = (
        normalize_image(np.max(top_vol, axis=0))
        if top_vol.size > 0
        else np.zeros((height, width), dtype=np.uint8)
    )
    mid_proj = (
        normalize_image(np.max(mid_vol, axis=0))
        if mid_vol.size > 0
        else np.zeros((height, width), dtype=np.uint8)
    )
    bot_proj = (
        normalize_image(np.max(bot_vol, axis=0))
        if bot_vol.size > 0
        else np.zeros((height, width), dtype=np.uint8)
    )

    # Combine horizontally with labels
    sep = np.ones((height, 4), dtype=np.uint8) * 128
    combined = np.concatenate([top_proj, sep, mid_proj, sep, bot_proj], axis=1)

    return (
        combined,
        f"Sub-volume MIPs: TOP (z=0-{z1}) | MID (z={z1}-{z2}) | BOT (z={z2}-{z_depth})",
    )


def projection_three_view(volume: np.ndarray) -> tuple[np.ndarray, str]:
    """
    Three orthogonal views with proper axis alignment:

        [XY][YZ]    <- XY shares Y axis (vertical) with YZ
        [XZ]        <- XZ shares X axis (horizontal) with XY

    - XY: Top view (looking down Z), X horizontal, Y vertical
    - XZ: Front view (looking along Y), X horizontal, Z vertical - BELOW XY
    - YZ: Side view (looking along X), Z horizontal, Y vertical - RIGHT of XY

    Returns:
        (image with 3 views, description) tuple
    """
    if volume.ndim != 3:
        return normalize_image(volume), "2D input"

    z_depth, height, width = volume.shape

    # XY: max along Z (looking down) -> (Y, X)
    xy_proj = normalize_image(np.max(volume, axis=0))
    xy_h, xy_w = xy_proj.shape

    # XZ: max along Y (looking from front) -> (Z, X)
    xz_proj = normalize_image(np.max(volume, axis=1))

    # YZ: max along X (looking from side) -> (Z, Y)
    yz_proj = normalize_image(np.max(volume, axis=2))

    # Consistent Z-dimension sizing (same as dual_view and depth_colored)
    z_display_h = max(xy_h // 3, int(z_depth * 3))

    # Scale XZ to match XY width (same X axis) and proper Z height
    pil_xz = PIL_Image.fromarray(xz_proj)
    pil_xz = pil_xz.resize((xy_w, z_display_h), PIL_Image.Resampling.LANCZOS)
    xz_scaled = np.array(pil_xz)

    # YZ: needs Y vertical (to align with XY's Y), Z horizontal
    # yz_proj is (Z, Y), we need (Y, Z) so Y is vertical
    yz_rotated = yz_proj.T  # Now (Y, Z) - Y vertical, Z horizontal

    # Scale YZ to match XY height (same Y axis), Z dimension as width
    z_display_w = z_display_h  # Keep Z dimension consistent
    pil_yz = PIL_Image.fromarray(yz_rotated)
    pil_yz = pil_yz.resize((z_display_w, xy_h), PIL_Image.Resampling.LANCZOS)
    yz_scaled = np.array(pil_yz)

    sep = 3  # Separator thickness

    # Top row: XY | sep | YZ
    v_sep = np.ones((xy_h, sep), dtype=np.uint8) * 128
    top_row = np.concatenate([xy_proj, v_sep, yz_scaled], axis=1)

    # Bottom row: XZ | padding to align with top row
    total_width = top_row.shape[1]
    if xz_scaled.shape[1] < total_width:
        pad = np.zeros((xz_scaled.shape[0], total_width - xz_scaled.shape[1]), dtype=np.uint8)
        bottom_row = np.concatenate([xz_scaled, pad], axis=1)
    else:
        bottom_row = xz_scaled[:, :total_width]

    # Horizontal separator
    h_sep = np.ones((sep, total_width), dtype=np.uint8) * 128

    # Combine
    combined = np.concatenate([top_row, h_sep, bottom_row], axis=0)

    return combined, "Three-view: [XY|YZ] top, [XZ] bottom - axes aligned"


def render_volume_rotated_v2(
    volume: np.ndarray, angle_y: float, angle_x: float = -0.5, threshold: float = 0.12
) -> np.ndarray:
    """
    Render volume by actually rotating it in 3D, then projecting.
    Uses scipy.ndimage.rotate for true 3D rotation.
    """
    from scipy.ndimage import rotate

    # Normalize volume
    vol = volume.astype(np.float32)
    p1, p99 = np.percentile(vol, [1, 99])
    vol = np.clip((vol - p1) / (p99 - p1 + 1e-8), 0, 1)

    # Convert angles from radians to degrees
    angle_y_deg = np.degrees(angle_y)
    angle_x_deg = np.degrees(angle_x)

    # Rotate around Y axis (rotation in XZ plane, axis 1 is Y)
    # axes=(2, 0) means rotate in the X-Z plane
    rotated = rotate(vol, angle_y_deg, axes=(2, 0), reshape=False, order=1, mode="constant", cval=0)

    # Rotate around X axis (rotation in YZ plane)
    # axes=(1, 0) means rotate in the Y-Z plane
    rotated = rotate(
        rotated,
        angle_x_deg,
        axes=(1, 0),
        reshape=False,
        order=1,
        mode="constant",
        cval=0,
    )

    # Now do alpha-blended projection along Z (front to back would be back to front after rotation)
    z_depth, height, width = rotated.shape

    result = np.zeros((height, width, 3), dtype=np.float32)

    # Back-to-front compositing
    for z in range(z_depth):
        slice_img = rotated[z]

        # Compute alpha based on threshold
        alpha = np.clip((slice_img - threshold) * 2, 0, 1)

        # Composite
        for c in range(3):
            result[:, :, c] = slice_img * alpha + result[:, :, c] * (1 - alpha)

    # Convert to uint8
    result = (np.clip(result, 0, 1) * 255).astype(np.uint8)

    return result


def render_volume_rotated(
    volume: np.ndarray,
    angle_y: float,
    angle_x: float = -0.5,
    threshold: float = 0.12,
    num_slices: int = 48,
    perspective: float = 0.4,
) -> np.ndarray:
    """
    Render volume from a rotated viewpoint with parallax and perspective.

    Simulates the 3D viewer by shifting and scaling slices based on rotation,
    creating depth through parallax and perspective foreshortening.

    Parameters
    ----------
    volume : np.ndarray
        3D volume (Z, Y, X)
    angle_y : float
        Rotation around Y axis in radians (horizontal spin)
    angle_x : float
        Rotation around X axis in radians (vertical tilt)
    threshold : float
        Intensity threshold for transparency (0-1)
    num_slices : int
        Number of slices to composite
    perspective : float
        Perspective strength (0 = orthographic, higher = more perspective)

    Returns
    -------
    np.ndarray
        RGB image (H, W, 3) uint8
    """
    # Normalize volume to 0-1
    vol = volume.astype(np.float32)
    p1, p99 = np.percentile(vol, [1, 99])
    vol = np.clip((vol - p1) / (p99 - p1 + 1e-8), 0, 1)

    z_depth, height, width = vol.shape

    # Calculate the z-scale factor (exaggerate depth like in 3D viewer)
    z_scale = (z_depth / width) * 3.0

    # Output canvas size (larger to accommodate shifts and scaling)
    margin = int(max(width, height) * 0.5)
    out_h = height + 2 * margin
    out_w = width + 2 * margin
    center_y, center_x = out_h // 2, out_w // 2

    # Initialize output
    result = np.zeros((out_h, out_w, 3), dtype=np.float32)

    # Compute rotation components
    sin_y, cos_y = np.sin(angle_y), np.cos(angle_y)
    sin_x, cos_x = np.sin(angle_x), np.cos(angle_x)

    # Sample slices - sort by depth after rotation for proper compositing
    slice_data_list = []
    for i in range(num_slices):
        z_idx = int(i * z_depth / num_slices)
        z_pos = (i / num_slices - 0.5) * z_scale  # Normalized z position

        # Apply rotation to get the projected depth and shifts
        rotated_z = z_pos * cos_y
        shift_x = z_pos * sin_y * width * 1.2  # Horizontal parallax (increased for more depth)

        # Apply X rotation (tilt)
        shift_y = -rotated_z * sin_x * height * 1.0  # Vertical parallax (increased for more depth)
        depth_after_rotation = rotated_z * cos_x

        # Perspective scale: closer slices are larger, further are smaller
        # Increase perspective effect for more 3D depth
        perspective_scale = 1.0 + (depth_after_rotation * perspective * 1.5 / z_scale)
        perspective_scale = np.clip(perspective_scale, 0.5, 1.5)

        slice_data_list.append(
            {
                "z_idx": z_idx,
                "shift_x": shift_x,
                "shift_y": shift_y,
                "depth": depth_after_rotation,
                "scale": perspective_scale,
            }
        )

    # Sort by depth (back to front for proper alpha compositing)
    slice_data_list.sort(key=lambda s: s["depth"])

    # Composite slices
    for slice_info in slice_data_list:
        z_idx = slice_info["z_idx"]
        shift_x = slice_info["shift_x"]
        shift_y = slice_info["shift_y"]
        scale = slice_info["scale"]

        # Get slice
        slice_img = vol[z_idx, :, :]

        # Apply slight blur for smoothness
        slice_img = ndimage.gaussian_filter(slice_img, sigma=0.5)

        # Compute alpha - match Three.js formula: (val - threshold) * 2, capped at 1
        alpha = np.clip((slice_img - threshold) * 2, 0, 1)

        # Scale the slice for perspective effect
        if abs(scale - 1.0) > 0.01:
            new_h = int(height * scale)
            new_w = int(width * scale)
            pil_slice = PIL_Image.fromarray((slice_img * 255).astype(np.uint8))
            pil_alpha = PIL_Image.fromarray((alpha * 255).astype(np.uint8))
            pil_slice = pil_slice.resize((new_w, new_h), PIL_Image.Resampling.BILINEAR)
            pil_alpha = pil_alpha.resize((new_w, new_h), PIL_Image.Resampling.BILINEAR)
            slice_img = np.array(pil_slice).astype(np.float32) / 255
            alpha = np.array(pil_alpha).astype(np.float32) / 255
            curr_h, curr_w = new_h, new_w
        else:
            curr_h, curr_w = height, width

        # Calculate paste position (centered, then shifted)
        y_start = center_y - curr_h // 2 + int(shift_y)
        x_start = center_x - curr_w // 2 + int(shift_x)

        # Bounds checking
        src_y_start = max(0, -y_start)
        src_x_start = max(0, -x_start)
        dst_y_start = max(0, y_start)
        dst_x_start = max(0, x_start)

        src_y_end = min(curr_h, out_h - y_start)
        src_x_end = min(curr_w, out_w - x_start)
        dst_y_end = dst_y_start + (src_y_end - src_y_start)
        dst_x_end = dst_x_start + (src_x_end - src_x_start)

        if dst_y_end <= dst_y_start or dst_x_end <= dst_x_start:
            continue

        # Get the slice region
        src_slice = slice_img[src_y_start:src_y_end, src_x_start:src_x_end]
        src_alpha = alpha[src_y_start:src_y_end, src_x_start:src_x_end]

        # Alpha composite
        for c in range(3):
            result[dst_y_start:dst_y_end, dst_x_start:dst_x_end, c] = (
                src_slice * src_alpha
                + result[dst_y_start:dst_y_end, dst_x_start:dst_x_end, c] * (1 - src_alpha)
            )

    # Crop to content (remove empty margins)
    gray = np.mean(result, axis=2)
    rows = np.any(gray > 0.01, axis=1)
    cols = np.any(gray > 0.01, axis=0)
    if np.any(rows) and np.any(cols):
        y_min, y_max = np.where(rows)[0][[0, -1]]
        x_min, x_max = np.where(cols)[0][[0, -1]]
        # Add small padding
        pad = 15
        y_min = max(0, y_min - pad)
        y_max = min(out_h, y_max + pad)
        x_min = max(0, x_min - pad)
        x_max = min(out_w, x_max + pad)
        result = result[y_min:y_max, x_min:x_max]

    # Convert to uint8
    result = (np.clip(result, 0, 1) * 255).astype(np.uint8)

    return result


def projection_spin_3d(volume: np.ndarray) -> tuple[np.ndarray, str]:
    """
    Multiple 3D perspective views from different angles.

    Shows a 2x3 grid of views rotating around the volume.

    Returns:
        (image, description) tuple
    """
    if volume.ndim != 3:
        return normalize_image(volume), "2D input"

    # JS threshold 0.30 (slider=30) compares against 0-255, so 30/255=0.12 in 0-1 space
    js_threshold = 0.30
    py_threshold = js_threshold * 100 / 255  # ~0.118

    # Base tilt angle (calibrated to match 3D viewer)
    base_tilt = 0.21  # angle_x

    # Different rotation angles around Y axis (horizontal spin)
    # 6 views: front, 60°, 120°, 180° (back), 240°, 300°
    angles_y = [-0.05, 0.5, 1.0, 1.57, 2.1, 2.6]

    views = []
    for angle_y in angles_y:
        view = render_volume_rotated(
            volume,
            angle_y=angle_y,
            angle_x=base_tilt,
            threshold=py_threshold,
            perspective=0.5,
        )
        views.append(view)

    # Arrange in 2x3 grid
    # Resize views to same size
    target_h = max(v.shape[0] for v in views)
    target_w = max(v.shape[1] for v in views)

    padded = []
    for v in views:
        h, w = v.shape[:2]
        pad_h = (target_h - h) // 2
        pad_w = (target_w - w) // 2
        if v.ndim == 3:
            p = np.zeros((target_h, target_w, 3), dtype=v.dtype)
            p[pad_h : pad_h + h, pad_w : pad_w + w] = v
        else:
            p = np.zeros((target_h, target_w), dtype=v.dtype)
            p[pad_h : pad_h + h, pad_w : pad_w + w] = v
        padded.append(p)

    # Create 2x3 grid
    row1 = np.hstack(padded[0:3])
    row2 = np.hstack(padded[3:6])
    grid = np.vstack([row1, row2])

    return grid, f"6 views, threshold: {js_threshold:.2f}, tilt: {base_tilt:.2f}"


# Registry of available projection methods
PROJECTION_METHODS = {
    "dual_view": projection_dual_view,
    "depth_colored": projection_depth_colored,
    "multi_slice": projection_multi_slice,
    "subvolume": projection_subvolume,
    "three_view": projection_three_view,
    "spin_3d": projection_spin_3d,
}


# =============================================================================
# Session Manager
# =============================================================================


class SessionManager:
    """Manages volume data for a session with aggressive caching."""

    def __init__(self, session_path: Path):
        self.session_path = session_path
        self.embryo_volumes = discover_volumes(session_path)
        self.current_embryo = None
        self.current_idx = 0
        self._volume_cache: dict = {}  # (embryo, idx) -> volume
        self._projection_cache: dict = {}  # (embryo, idx, method) -> (img, desc)
        self._crop_bounds: dict = {}  # embryo -> (y_min, y_max, x_min, x_max)
        self._cache_size = 20  # Keep more volumes in memory

        if self.embryo_volumes:
            self.current_embryo = sorted(self.embryo_volumes.keys())[0]

    @property
    def embryo_list(self) -> list[str]:
        return sorted(self.embryo_volumes.keys())

    @property
    def current_volumes(self) -> list[Path]:
        if self.current_embryo:
            return self.embryo_volumes.get(self.current_embryo, [])
        return []

    @property
    def total_timepoints(self) -> int:
        return len(self.current_volumes)

    def switch_embryo(self, embryo_id: str):
        if embryo_id in self.embryo_volumes:
            self.current_embryo = embryo_id
            self.current_idx = 0

    def _get_crop_bounds(self) -> tuple[int, int, int, int] | None:
        """Get or compute crop bounds for current embryo (computed from first volume)."""
        if self.current_embryo is None:
            return None

        if self.current_embryo not in self._crop_bounds:
            # Compute bounds from first volume of this embryo
            volumes = self.current_volumes
            if not volumes:
                return None

            # Load first volume without cropping to compute bounds
            vol = tifffile.imread(str(volumes[0]))
            vol = np.squeeze(vol)
            if vol.ndim == 3:
                z_depth, height, width = vol.shape
                if width > height * 2:
                    vol = vol[:, :, : width // 2]

            bounds = compute_crop_bounds(vol, padding=20)
            self._crop_bounds[self.current_embryo] = bounds
            print(f"[ROI] Computed crop bounds for {self.current_embryo}: {bounds}")
            print(
                f"[ROI] Original volume shape: {vol.shape}, cropped region:"
                f" {bounds[1] - bounds[0]}x{bounds[3] - bounds[2]}"
            )

        return self._crop_bounds[self.current_embryo]

    def get_volume(self, idx: int) -> np.ndarray | None:
        """Load volume at index, with caching and stable crop bounds."""
        volumes = self.current_volumes
        if not volumes or idx < 0 or idx >= len(volumes):
            return None

        cache_key = (self.current_embryo, idx)
        if cache_key not in self._volume_cache:
            # Evict old entries if cache too large
            if len(self._volume_cache) > self._cache_size:
                # Remove entries far from current index
                keys_to_remove = []
                for k in self._volume_cache:
                    if k[0] != self.current_embryo or abs(k[1] - idx) > 10:
                        keys_to_remove.append(k)
                for k in keys_to_remove[: len(self._volume_cache) - self._cache_size // 2]:
                    del self._volume_cache[k]

            # Get stable crop bounds for this embryo
            crop_bounds = self._get_crop_bounds()
            vol = load_volume(volumes[idx], crop_bounds)
            print(f"[ROI] Loaded volume {idx}: shape={vol.shape}, bounds={crop_bounds}")
            self._volume_cache[cache_key] = vol

        return self._volume_cache[cache_key]

    def get_projection(self, idx: int, method: str) -> tuple[np.ndarray, str] | None:
        """Get cached projection or compute it."""
        cache_key = (self.current_embryo, idx, method)

        if cache_key not in self._projection_cache:
            volume = self.get_volume(idx)
            if volume is None:
                return None

            if method in PROJECTION_METHODS:
                result = PROJECTION_METHODS[method](volume)
                self._projection_cache[cache_key] = result

                # Limit projection cache size
                if len(self._projection_cache) > 100:
                    # Remove oldest entries
                    keys = list(self._projection_cache.keys())
                    for k in keys[:50]:
                        del self._projection_cache[k]
            else:
                return None

        return self._projection_cache[cache_key]

    def get_current_volume(self) -> np.ndarray | None:
        return self.get_volume(self.current_idx)

    def get_current_path(self) -> Path | None:
        volumes = self.current_volumes
        if volumes and 0 <= self.current_idx < len(volumes):
            return volumes[self.current_idx]
        return None

    def preload_adjacent(self, idx: int, radius: int = 2):
        """Preload volumes adjacent to current index (call from background thread)."""
        for i in range(max(0, idx - radius), min(self.total_timepoints, idx + radius + 1)):
            if i != idx:
                self.get_volume(i)  # This will cache it


# =============================================================================
# Web UI Server
# =============================================================================


class ExplorerHandler(BaseHTTPRequestHandler):
    """HTTP handler for projection explorer."""

    session_manager: SessionManager  # assigned on the class before the server starts serving
    current_methods: list[str] = ["dual_view", "depth_colored"]
    current_z_slice: int = 0  # For 3D volume viewer

    def log_message(self, format, *args):
        pass  # Suppress logging

    def do_GET(self):
        parsed = urlparse(self.path)

        try:
            if parsed.path == "/":
                self.send_html()
            elif parsed.path == "/api/projections":
                self.send_projections()
            elif parsed.path == "/api/status":
                self.send_status()
            elif parsed.path.startswith("/api/goto"):
                params = parse_qs(parsed.query)
                idx = int(params.get("idx", [0])[0])
                self.session_manager.current_idx = max(
                    0, min(idx, self.session_manager.total_timepoints - 1)
                )
                self.send_projections()
            elif parsed.path.startswith("/api/embryo"):
                params = parse_qs(parsed.query)
                embryo = params.get("name", [""])[0]
                self.session_manager.switch_embryo(embryo)
                self.send_projections()
            elif parsed.path.startswith("/api/methods"):
                params = parse_qs(parsed.query)
                methods = params.get("m", [])
                if methods:
                    # Update class variable, not instance variable
                    ExplorerHandler.current_methods = [
                        m for m in methods if m in PROJECTION_METHODS
                    ]
                self.send_projections()
            elif parsed.path.startswith("/api/zslice"):
                params = parse_qs(parsed.query)
                z = int(params.get("z", [0])[0])
                self.send_zslice(z)
            elif parsed.path == "/api/volume":
                self.send_volume()
            elif parsed.path == "/3d":
                self.send_3d_viewer()
            else:
                self.send_error(404)
        except Exception as e:
            self.send_error(500, str(e))

    def send_json(self, data):
        body = json.dumps(data).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

    def send_status(self):
        sm = self.session_manager
        volume = sm.get_current_volume()
        z_depth = volume.shape[0] if volume is not None and volume.ndim == 3 else 0
        crop_bounds = sm._crop_bounds.get(sm.current_embryo) if sm.current_embryo else None
        self.send_json(
            {
                "session": sm.session_path.name,
                "embryos": sm.embryo_list,
                "current_embryo": sm.current_embryo,
                "total": sm.total_timepoints,
                "idx": sm.current_idx,
                "methods": list(PROJECTION_METHODS.keys()),
                "active_methods": ExplorerHandler.current_methods,
                "z_depth": z_depth,
                "crop_bounds": crop_bounds,
            }
        )

    def send_zslice(self, z: int):
        sm = self.session_manager
        volume = sm.get_current_volume()
        if volume is None or volume.ndim != 3:
            self.send_json({"error": "No volume loaded"})
            return

        z_depth = volume.shape[0]
        z = max(0, min(z, z_depth - 1))
        ExplorerHandler.current_z_slice = z

        # Get the slice and normalize
        slice_img = volume[z]
        slice_norm = normalize_image(slice_img)

        # Convert to RGB
        slice_rgb = np.stack([slice_norm, slice_norm, slice_norm], axis=-1)

        self.send_json(
            {
                "z": z,
                "z_depth": z_depth,
                "data": image_to_base64(slice_rgb),
            }
        )

    def send_volume(self):
        """Send full volume as base64 for 3D rendering."""
        sm = self.session_manager
        volume = sm.get_current_volume()
        if volume is None or volume.ndim != 3:
            self.send_json({"error": "No volume loaded"})
            return

        z, h, w = volume.shape

        # Normalize to float
        vol_norm = volume.astype(np.float32)
        p1, p99 = np.percentile(vol_norm, [1, 99])
        vol_norm = np.clip((vol_norm - p1) / (p99 - p1 + 1e-8), 0, 1)

        # Apply Gaussian blur along Z axis to reduce banding at side views
        vol_norm = ndimage.gaussian_filter1d(vol_norm, sigma=1.0, axis=0)

        vol_uint8 = (vol_norm * 255).astype(np.uint8)

        # Encode as base64
        vol_bytes = vol_uint8.tobytes()
        vol_b64 = base64.b64encode(vol_bytes).decode()

        self.send_json(
            {
                "shape": [z, h, w],
                "data": vol_b64,
            }
        )

    def send_projections(self):
        sm = self.session_manager
        path = sm.get_current_path()
        idx = sm.current_idx

        # Use cached projections
        projections = []
        volume_shape = None

        for method_name in ExplorerHandler.current_methods:
            result = sm.get_projection(idx, method_name)
            if result:
                img, desc = result
                projections.append(
                    {
                        "method": method_name,
                        "description": desc,
                        "data": image_to_base64(img),
                    }
                )

        if not projections:
            self.send_json({"error": "No volume loaded"})
            return

        # Get volume shape from cache
        volume = sm.get_current_volume()
        volume_shape = list(volume.shape) if volume is not None else []

        self.send_json(
            {
                "idx": idx,
                "total": sm.total_timepoints,
                "filename": path.name if path else "",
                "embryo": sm.current_embryo,
                "embryo_list": sm.embryo_list,
                "projections": projections,
                "volume_shape": volume_shape,
            }
        )

        # Preload adjacent volumes in background
        import threading

        threading.Thread(target=sm.preload_adjacent, args=(idx,), daemon=True).start()

    def send_html(self):
        html = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Projection Explorer</title>
    <style>
        * { box-sizing: border-box; }
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: #0d1117;
            color: #c9d1d9;
            margin: 0;
            padding: 20px;
        }
        .container { max-width: 1800px; margin: 0 auto; }
        h1 { color: #58a6ff; margin-bottom: 5px; }
        .subtitle { color: #8b949e; margin-bottom: 20px; }

        .controls {
            display: flex;
            gap: 15px;
            align-items: center;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }
        .control-group {
            display: flex;
            align-items: center;
            gap: 8px;
        }
        label { color: #8b949e; font-size: 14px; }
        select, button {
            padding: 8px 12px;
            border-radius: 6px;
            border: 1px solid #30363d;
            background: #21262d;
            color: #c9d1d9;
            font-size: 14px;
        }
        button { cursor: pointer; }
        button:hover { background: #30363d; }

        .nav-controls {
            display: flex;
            align-items: center;
            gap: 10px;
            flex: 1;
        }
        .slider {
            flex: 1;
            min-width: 200px;
            accent-color: #58a6ff;
        }
        .position {
            font-family: monospace;
            min-width: 100px;
            text-align: center;
        }

        .method-toggles {
            display: flex;
            gap: 8px;
            flex-wrap: wrap;
        }
        .method-toggle {
            padding: 6px 12px;
            border-radius: 20px;
            border: 1px solid #30363d;
            background: #21262d;
            cursor: pointer;
            font-size: 13px;
            transition: all 0.2s;
        }
        .method-toggle.active {
            background: #238636;
            border-color: #238636;
        }
        .method-toggle:hover { border-color: #58a6ff; }

        .projections {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(500px, 1fr));
            gap: 20px;
        }
        .projection-card {
            background: #161b22;
            border: 1px solid #30363d;
            border-radius: 8px;
            overflow: hidden;
        }
        .projection-header {
            padding: 12px 16px;
            background: #21262d;
            border-bottom: 1px solid #30363d;
        }
        .projection-title {
            font-weight: 600;
            color: #58a6ff;
        }
        .projection-desc {
            font-size: 12px;
            color: #8b949e;
            margin-top: 4px;
        }
        .projection-image {
            padding: 10px;
            text-align: center;
            background: #000;
        }
        .projection-image img {
            max-width: 100%;
            height: auto;
        }

        .info-bar {
            display: flex;
            gap: 20px;
            padding: 10px 16px;
            background: #21262d;
            border-radius: 6px;
            margin-bottom: 20px;
            font-size: 13px;
        }
        .info-item { color: #8b949e; }
        .info-value { color: #c9d1d9; font-weight: 500; }

        .keyboard-hint {
            color: #6e7681;
            font-size: 12px;
            margin-top: 20px;
        }
        .loading {
            position: fixed;
            top: 10px;
            right: 10px;
            background: #238636;
            color: white;
            padding: 8px 16px;
            border-radius: 6px;
            font-size: 14px;
            display: none;
            z-index: 1000;
        }
        .loading.active { display: block; }
    </style>
</head>
<body>
    <div class="loading" id="loading">Loading...</div>
    <div class="container">
        <h1>Projection Explorer</h1>
        <p class="subtitle">Compare different volume projection methods for embryo perception</p>

        <div class="info-bar">
            <div class="info-item">Session: <span class="info-value" id="session">-</span></div>
            <div class="info-item">Embryo: <span class="info-value" id="embryo">-</span></div>
            <div class="info-item">Volume: <span class="info-value" id="volume-shape">-</span></div>
            <div class="info-item">ROI: <span class="info-value" id="crop-bounds">-</span></div>
            <div class="info-item">File: <span class="info-value" id="filename">-</span></div>
        </div>

        <div class="controls">
            <div class="control-group">
                <label>Embryo:</label>
                <select id="embryo-select" onchange="switchEmbryo(this.value)"></select>
            </div>

            <div class="nav-controls">
                <button onclick="prev()">← Prev</button>
                <input type="range" class="slider" id="slider" min="0" max="100"
                  value="0" oninput="debouncedGoto(this.value)">
                <button onclick="next()">Next →</button>
                <span class="position" id="position">0 / 0</span>
                <span class="stage-label" id="stage-label"
                  style="margin-left:12px;padding:2px 8px;background:#238636;
                         border-radius:4px;font-weight:bold;"></span>
            </div>
        </div>

        <div class="controls">
            <label>Methods:</label>
            <div class="method-toggles" id="method-toggles"></div>
        </div>

        <div class="projections" id="projections"></div>

        <div class="projection-card" id="viewer3d-card">
            <div class="projection-header">
                <div class="projection-title">3D Volume</div>
                <div class="projection-desc">
                    Drag: X/Y | Shift+Drag: Z | Scroll: zoom |
                    Threshold: <input type="range" id="thresh3d" min="0" max="100"
                      value="30" style="width:80px;vertical-align:middle;">
                    <span id="thresh-display"
                      style="font-family:monospace;color:#58a6ff;min-width:35px;display:inline-block;">0.30</span>
                    | <span id="angle-display"
                      style="font-family:monospace;color:#58a6ff;"
                      >angle_y: 0.00, angle_x: 0.00, angle_z: 0.00</span>
                    <button id="copy-angles-btn" onclick="copyAngles()"
                      style="margin-left:8px;padding:2px 8px;font-size:11px;cursor:pointer;
                             background:#21262d;border:1px solid #30363d;
                             color:#c9d1d9;border-radius:4px;">Copy</button>
                </div>
            </div>
            <div id="viewer3d" style="height:400px;background:#000;"></div>
        </div>

        <p class="keyboard-hint">Keyboard: ← → or A/D to navigate, 1-6 to toggle methods</p>
    </div>

    <script>
        let state = {
            idx: 0,
            total: 0,
            methods: [],
            activeMethods: [],
            currentEmbryo: null
        };

        // Ground truth stage transitions (from biologist annotations)
        const GROUND_TRUTH = {
            "embryo_1": {"early": 0, "bean": 43, "comma": 49,
                         "1.5fold": 55, "2fold": 70, "pretzel": 90},
            "embryo_2": {"early": 0, "bean": 33, "comma": 39,
                         "1.5fold": 45, "2fold": 60, "pretzel": 80},
            "embryo_3": {"early": 0, "bean": 27, "comma": 33,
                         "1.5fold": 39, "2fold": 50, "pretzel": 69},
            "embryo_4": {"early": 0, "bean": 54, "comma": 60,
                         "1.5fold": 69, "2fold": 77, "pretzel": 97}
        };

        const STAGE_ORDER = [
            "early", "bean", "comma", "1.5fold", "2fold", "pretzel", "hatching", "hatched"
        ];
        const STAGE_COLORS = {
            "early": "#6e7681",
            "bean": "#8b5cf6",
            "comma": "#3b82f6",
            "1.5fold": "#06b6d4",
            "2fold": "#10b981",
            "pretzel": "#f59e0b",
            "hatching": "#ef4444",
            "hatched": "#ec4899"
        };

        function getStageAtTimepoint(embryo, timepoint) {
            const transitions = GROUND_TRUTH[embryo];
            if (!transitions) return null;

            let currentStage = "early";
            for (const stage of STAGE_ORDER) {
                if (transitions[stage] !== undefined && timepoint >= transitions[stage]) {
                    currentStage = stage;
                }
            }
            return currentStage;
        }

        function updateStageLabel() {
            const label = document.getElementById('stage-label');
            const stage = getStageAtTimepoint(state.currentEmbryo, state.idx);
            if (stage) {
                label.textContent = stage;
                label.style.background = STAGE_COLORS[stage] || '#238636';
            } else {
                label.textContent = '';
            }
        }

        function showLoading(show) {
            document.getElementById('loading').classList.toggle('active', show);
        }

        // Debounce slider to avoid flooding requests
        let debounceTimer = null;
        function debouncedGoto(idx) {
            state.idx = parseInt(idx);
            document.getElementById('position').textContent = (state.idx + 1) + ' / ' + state.total;
            updateStageLabel();
            clearTimeout(debounceTimer);
            debounceTimer = setTimeout(() => gotoIdx(idx), 150);
        }

        async function fetchStatus() {
            showLoading(true);
            const resp = await fetch('/api/status');
            const data = await resp.json();
            showLoading(false);

            state.methods = data.methods;
            state.activeMethods = data.active_methods;

            document.getElementById('session').textContent = data.session;

            // Display crop bounds if available
            if (data.crop_bounds) {
                const [yMin, yMax, xMin, xMax] = data.crop_bounds;
                document.getElementById('crop-bounds').textContent =
                    `Y:${yMin}-${yMax} X:${xMin}-${xMax} (${yMax-yMin}×${xMax-xMin})`;
            } else {
                document.getElementById('crop-bounds').textContent = 'none';
            }

            // Populate embryo selector
            const select = document.getElementById('embryo-select');
            select.innerHTML = '';
            data.embryos.forEach(e => {
                const opt = document.createElement('option');
                opt.value = e;
                opt.textContent = e;
                if (e === data.current_embryo) opt.selected = true;
                select.appendChild(opt);
            });

            // Populate method toggles
            const toggles = document.getElementById('method-toggles');
            toggles.innerHTML = '';
            data.methods.forEach((m, i) => {
                const btn = document.createElement('button');
                btn.className = 'method-toggle'
                    + (data.active_methods.includes(m) ? ' active' : '');
                btn.textContent = (i + 1) + ': ' + m;
                btn.onclick = () => toggleMethod(m);
                toggles.appendChild(btn);
            });

            await fetchProjections();
        }

        async function fetchProjections() {
            showLoading(true);
            const resp = await fetch('/api/projections');
            const data = await resp.json();
            showLoading(false);

            if (data.error) {
                console.error(data.error);
                return;
            }

            state.idx = data.idx;
            state.total = data.total;
            state.currentEmbryo = data.embryo;

            document.getElementById('slider').max = data.total - 1;
            document.getElementById('slider').value = data.idx;
            document.getElementById('position').textContent = (data.idx + 1) + ' / ' + data.total;
            document.getElementById('embryo').textContent = data.embryo;
            document.getElementById('filename').textContent = data.filename;
            document.getElementById('volume-shape').textContent = data.volume_shape.join(' × ');
            updateStageLabel();

            // Update projections
            const container = document.getElementById('projections');
            container.innerHTML = '';

            data.projections.forEach(proj => {
                const card = document.createElement('div');
                card.className = 'projection-card';
                const descId = 'desc-' + proj.method;
                card.innerHTML = `
                    <div class="projection-header">
                        <div class="projection-title">${proj.method}</div>
                        <div class="projection-desc">
                            <span id="${descId}">${proj.description}</span>
                            <button onclick="copyText('${descId}')"
                              style="margin-left:8px;padding:2px 8px;font-size:11px;
                                     cursor:pointer;background:#21262d;border:1px solid #30363d;
                                     color:#c9d1d9;border-radius:4px;">Copy</button>
                        </div>
                    </div>
                    <div class="projection-image">
                        <img src="data:image/jpeg;base64,${proj.data}" alt="${proj.method}">
                    </div>
                `;
                container.appendChild(card);
            });

            // Reload 3D volume
            if (typeof load3DVolume === 'function') {
                load3DVolume();
            }
        }

        async function gotoIdx(idx) {
            await fetch('/api/goto?idx=' + idx);
            await fetchProjections();
        }

        async function prev() {
            await gotoIdx(Math.max(0, state.idx - 1));
        }

        async function next() {
            await gotoIdx(Math.min(state.total - 1, state.idx + 1));
        }

        async function switchEmbryo(name) {
            await fetch('/api/embryo?name=' + encodeURIComponent(name));
            await fetchProjections();
        }

        async function toggleMethod(method) {
            let methods = [...state.activeMethods];
            if (methods.includes(method)) {
                methods = methods.filter(m => m !== method);
            } else {
                methods.push(method);
            }
            if (methods.length === 0) methods = ['dual_view'];  // Keep at least one

            state.activeMethods = methods;
            await fetch('/api/methods?m=' + methods.join('&m='));

            // Update toggle buttons
            document.querySelectorAll('.method-toggle').forEach(btn => {
                const m = btn.textContent.split(': ')[1];
                btn.classList.toggle('active', methods.includes(m));
            });

            await fetchProjections();
        }

        document.addEventListener('keydown', (e) => {
            if (e.key === 'ArrowLeft' || e.key === 'a') prev();
            else if (e.key === 'ArrowRight' || e.key === 'd') next();
            else if (e.key >= '1' && e.key <= '9') {
                const idx = parseInt(e.key) - 1;
                if (idx < state.methods.length) {
                    toggleMethod(state.methods[idx]);
                }
            }
        });

        function copyAngles() {
            const threshVal = document.getElementById('thresh-display').textContent;
            const angleText = document.getElementById('angle-display').textContent;
            const fullText = 'threshold: ' + threshVal + ', ' + angleText;
            navigator.clipboard.writeText(fullText).then(() => {
                const btn = document.getElementById('copy-angles-btn');
                btn.textContent = 'Copied!';
                setTimeout(() => { btn.textContent = 'Copy'; }, 1000);
            });
        }

        function copyText(elementId) {
            const text = document.getElementById(elementId).textContent;
            navigator.clipboard.writeText(text).then(() => {
                const btn = event.target;
                btn.textContent = 'Copied!';
                setTimeout(() => { btn.textContent = 'Copy'; }, 1000);
            });
        }

        // Initial load
        fetchStatus();
    </script>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script>
        let scene3d, camera3d, renderer3d, sliceGroup;
        let axesScene, axesCamera, axesGroup;  // For orientation indicator
        let volumeData, volumeShape;
        let isDragging3d = false;
        let prevMouse3d = { x: 0, y: 0 };
        let viewer3dInitialized = false;
        let savedRotation = { x: -0.5, y: 0.5 };  // Default angled view
        let savedZoom = 0.9;  // Zoomed in to match spin_3d scale

        async function load3DVolume() {
            const resp = await fetch('/api/volume');
            const data = await resp.json();
            if (data.error) return;

            volumeShape = data.shape;
            const raw = atob(data.data);
            volumeData = new Uint8Array(raw.length);
            for (let i = 0; i < raw.length; i++) {
                volumeData[i] = raw.charCodeAt(i);
            }

            if (!viewer3dInitialized) {
                init3DViewer();
                viewer3dInitialized = true;
            } else {
                // Rebuild slices with current threshold, keep camera
                const thresh = parseInt(document.getElementById('thresh3d').value);
                buildSlices3d(32, thresh);
            }
        }

        function createSliceTex(zIndex, threshold) {
            const [zd, h, w] = volumeShape;
            const sliceSize = w * h;
            const offset = zIndex * sliceSize;
            const rgba = new Uint8Array(w * h * 4);
            for (let i = 0; i < sliceSize; i++) {
                const val = volumeData[offset + i];
                if (val > threshold) {
                    rgba[i * 4] = val;
                    rgba[i * 4 + 1] = val;
                    rgba[i * 4 + 2] = val;
                    rgba[i * 4 + 3] = Math.min(255, (val - threshold) * 2);
                } else {
                    rgba[i * 4 + 3] = 0;
                }
            }
            const tex = new THREE.DataTexture(rgba, w, h, THREE.RGBAFormat);
            tex.needsUpdate = true;
            return tex;
        }

        function buildSlices3d(numSlices, threshold) {
            if (!volumeShape) return;
            const [zd, h, w] = volumeShape;
            while (sliceGroup.children.length > 0) {
                const c = sliceGroup.children[0];
                c.geometry.dispose();
                c.material.dispose();
                sliceGroup.remove(c);
            }
            const aspect = w / h;
            const zScale = (zd / w) * 3;
            for (let i = 0; i < numSlices; i++) {
                const zIndex = Math.floor(i * zd / numSlices);
                const zPos = (i / numSlices - 0.5) * zScale;
                const tex = createSliceTex(zIndex, threshold);
                const mat = new THREE.MeshBasicMaterial({
                    map: tex, transparent: true, side: THREE.DoubleSide, depthWrite: false
                });
                const geo = new THREE.PlaneGeometry(1, 1 / aspect);
                const mesh = new THREE.Mesh(geo, mat);
                mesh.position.z = zPos;
                sliceGroup.add(mesh);
            }
        }

        function createAxisArrow(dir, color, length = 0.4) {
            const group = new THREE.Group();
            // Shaft
            const shaftGeo = new THREE.CylinderGeometry(0.02, 0.02, length * 0.8, 8);
            const shaftMat = new THREE.MeshBasicMaterial({ color: color });
            const shaft = new THREE.Mesh(shaftGeo, shaftMat);
            shaft.position.y = length * 0.4;
            group.add(shaft);
            // Head
            const headGeo = new THREE.ConeGeometry(0.05, length * 0.2, 8);
            const headMat = new THREE.MeshBasicMaterial({ color: color });
            const head = new THREE.Mesh(headGeo, headMat);
            head.position.y = length * 0.9;
            group.add(head);
            // Orient to direction
            if (dir === 'x') group.rotation.z = -Math.PI / 2;
            else if (dir === 'z') group.rotation.x = Math.PI / 2;
            return group;
        }

        function init3DViewer() {
            const container = document.getElementById('viewer3d');
            const w = container.clientWidth;
            const h = container.clientHeight;

            scene3d = new THREE.Scene();
            camera3d = new THREE.PerspectiveCamera(50, w / h, 0.1, 100);
            camera3d.position.z = savedZoom;  // Match spin_3d scale

            renderer3d = new THREE.WebGLRenderer({ antialias: true });
            renderer3d.setSize(w, h);
            renderer3d.setClearColor(0x000000);
            renderer3d.autoClear = false;  // Allow multiple render passes
            container.appendChild(renderer3d.domElement);

            sliceGroup = new THREE.Group();
            sliceGroup.rotation.x = savedRotation.x;
            sliceGroup.rotation.y = savedRotation.y;
            sliceGroup.rotation.z = 0;  // Explicitly set to 0
            sliceGroup.scale.y = -1;  // Flip Y to match dual_view orientation
            scene3d.add(sliceGroup);
            buildSlices3d(32, 30);

            // Create axes orientation indicator
            axesScene = new THREE.Scene();
            axesCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.1, 10);
            axesCamera.position.z = 2;

            axesGroup = new THREE.Group();
            axesGroup.add(createAxisArrow('x', 0xff4444));  // Red = X
            axesGroup.add(createAxisArrow('y', 0x44ff44));  // Green = Y
            axesGroup.add(createAxisArrow('z', 0x4444ff));  // Blue = Z
            axesScene.add(axesGroup);

            // Add axis labels
            const labelCanvas = document.createElement('canvas');
            labelCanvas.width = 64;
            labelCanvas.height = 64;
            const ctx = labelCanvas.getContext('2d');

            function makeLabel(text, color) {
                ctx.clearRect(0, 0, 64, 64);
                ctx.font = 'bold 48px Arial';
                ctx.fillStyle = color;
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                ctx.fillText(text, 32, 32);
                const tex = new THREE.CanvasTexture(labelCanvas);
                const mat = new THREE.SpriteMaterial({ map: tex.clone() });
                const sprite = new THREE.Sprite(mat);
                sprite.scale.set(0.15, 0.15, 1);
                return sprite;
            }

            const labelX = makeLabel('X', '#ff4444');
            labelX.position.set(0.55, 0, 0);
            axesGroup.add(labelX);

            const labelY = makeLabel('Y', '#44ff44');
            labelY.position.set(0, 0.55, 0);
            axesGroup.add(labelY);

            const labelZ = makeLabel('Z', '#4444ff');
            labelZ.position.set(0, 0, 0.55);
            axesGroup.add(labelZ);

            document.getElementById('thresh3d').addEventListener('input', (e) => {
                const threshVal = parseInt(e.target.value);
                buildSlices3d(32, threshVal);
                document.getElementById('thresh-display').textContent =
                    (threshVal / 100).toFixed(2);
            });

            renderer3d.domElement.addEventListener('mousedown', (e) => {
                isDragging3d = true;
                prevMouse3d = { x: e.clientX, y: e.clientY };
            });
            window.addEventListener('mouseup', () => isDragging3d = false);
            renderer3d.domElement.addEventListener('mousemove', (e) => {
                if (!isDragging3d) return;
                if (e.shiftKey) {
                    // Shift+drag: rotate around Z axis
                    sliceGroup.rotation.z += (e.clientX - prevMouse3d.x) * 0.01;
                } else {
                    // Normal drag: rotate around X and Y
                    sliceGroup.rotation.y += (e.clientX - prevMouse3d.x) * 0.01;
                    sliceGroup.rotation.x += (e.clientY - prevMouse3d.y) * 0.01;
                }
                savedRotation.x = sliceGroup.rotation.x;
                savedRotation.y = sliceGroup.rotation.y;
                // Sync axes orientation with volume
                axesGroup.rotation.x = sliceGroup.rotation.x;
                axesGroup.rotation.y = sliceGroup.rotation.y;
                axesGroup.rotation.z = sliceGroup.rotation.z;
                // Update angle display (including z rotation)
                document.getElementById('angle-display').textContent =
                    'angle_y: ' + sliceGroup.rotation.y.toFixed(2) +
                    ', angle_x: ' + sliceGroup.rotation.x.toFixed(2) +
                    ', angle_z: ' + sliceGroup.rotation.z.toFixed(2);
                prevMouse3d = { x: e.clientX, y: e.clientY };
            });
            renderer3d.domElement.addEventListener('wheel', (e) => {
                e.preventDefault();
                camera3d.position.z =
                    Math.max(0.5, Math.min(5, camera3d.position.z + e.deltaY * 0.002));
                savedZoom = camera3d.position.z;
            });

            // Initialize axes rotation to match saved rotation
            axesGroup.rotation.x = savedRotation.x;
            axesGroup.rotation.y = savedRotation.y;

            // Initialize angle display
            document.getElementById('angle-display').textContent =
                'angle_y: ' + savedRotation.y.toFixed(2) +
                ', angle_x: ' + savedRotation.x.toFixed(2) +
                ', angle_z: ' + sliceGroup.rotation.z.toFixed(2);

            function animate3d() {
                requestAnimationFrame(animate3d);
                const w = renderer3d.domElement.width;
                const h = renderer3d.domElement.height;

                // Render main scene (full viewport)
                renderer3d.clear();
                renderer3d.setViewport(0, 0, w, h);
                renderer3d.render(scene3d, camera3d);

                // Render axes in bottom-left corner
                const axesSize = Math.min(w, h) * 0.4;
                renderer3d.clearDepth();
                renderer3d.setViewport(10, 10, axesSize, axesSize);
                renderer3d.render(axesScene, axesCamera);

                // Reset viewport for next frame
                renderer3d.setViewport(0, 0, w, h);
            }
            animate3d();
        }

        // Load 3D after page loads
        setTimeout(load3DVolume, 500);
    </script>
</body>
</html>"""

        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", len(html))
        self.end_headers()
        self.wfile.write(html.encode())

    def send_3d_viewer(self):
        html = """<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>3D Volume Viewer</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body { background: #111; overflow: hidden; font-family: sans-serif; }
        #container { width: 100vw; height: 100vh; }
        #info {
            position: fixed;
            top: 10px;
            left: 10px;
            color: #fff;
            background: rgba(0,0,0,0.7);
            padding: 10px 15px;
            border-radius: 6px;
            font-size: 14px;
            z-index: 100;
        }
        #loading {
            position: fixed;
            top: 50%;
            left: 50%;
            transform: translate(-50%, -50%);
            color: #fff;
            font-size: 24px;
        }
        #controls {
            position: fixed;
            bottom: 20px;
            left: 50%;
            transform: translateX(-50%);
            display: flex;
            gap: 15px;
            background: rgba(0,0,0,0.7);
            padding: 15px;
            border-radius: 8px;
        }
        .control-group { display: flex; flex-direction: column; gap: 5px; }
        .control-group label { color: #aaa; font-size: 12px; }
        .control-group input { width: 120px; }
    </style>
</head>
<body>
    <div id="loading">Loading volume...</div>
    <div id="container"></div>
    <div id="info">Drag to rotate | Scroll to zoom</div>
    <div id="controls">
        <div class="control-group">
            <label>Slices: <span id="sliceVal">32</span></label>
            <input type="range" id="numSlices" min="8" max="64" value="32">
        </div>
        <div class="control-group">
            <label>Threshold: <span id="threshVal">30</span></label>
            <input type="range" id="threshold" min="0" max="100" value="30">
        </div>
    </div>

    <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
    <script>
        let scene, camera, renderer, sliceGroup;
        let volumeData, volumeShape;
        let sliceTextures = [];
        let isDragging = false;
        let prevMouse = { x: 0, y: 0 };

        async function loadVolume() {
            const resp = await fetch('/api/volume');
            const data = await resp.json();
            if (data.error) {
                document.getElementById('loading').textContent = data.error;
                return;
            }

            volumeShape = data.shape;
            const raw = atob(data.data);
            volumeData = new Uint8Array(raw.length);
            for (let i = 0; i < raw.length; i++) {
                volumeData[i] = raw.charCodeAt(i);
            }

            document.getElementById('loading').style.display = 'none';
            init3D();
        }

        function createSliceTexture(zIndex, threshold) {
            const [zd, h, w] = volumeShape;
            const sliceSize = w * h;
            const offset = zIndex * sliceSize;

            const rgba = new Uint8Array(w * h * 4);
            for (let i = 0; i < sliceSize; i++) {
                const val = volumeData[offset + i];
                if (val > threshold) {
                    rgba[i * 4] = val;
                    rgba[i * 4 + 1] = val;
                    rgba[i * 4 + 2] = val;
                    rgba[i * 4 + 3] = Math.min(255, (val - threshold) * 2);
                } else {
                    rgba[i * 4 + 3] = 0;
                }
            }

            const tex = new THREE.DataTexture(rgba, w, h, THREE.RGBAFormat);
            tex.needsUpdate = true;
            return tex;
        }

        function buildSlices(numSlices, threshold) {
            const [zd, h, w] = volumeShape;

            // Clear old slices
            while (sliceGroup.children.length > 0) {
                const child = sliceGroup.children[0];
                child.geometry.dispose();
                child.material.dispose();
                sliceGroup.remove(child);
            }

            const aspect = w / h;
            const zScale = (zd / w) * 3;  // Exaggerate Z for better depth perception

            for (let i = 0; i < numSlices; i++) {
                const zIndex = Math.floor(i * zd / numSlices);
                const zPos = (i / numSlices - 0.5) * zScale;

                const tex = createSliceTexture(zIndex, threshold);
                const mat = new THREE.MeshBasicMaterial({
                    map: tex,
                    transparent: true,
                    side: THREE.DoubleSide,
                    depthWrite: false
                });
                const geo = new THREE.PlaneGeometry(1, 1 / aspect);
                const mesh = new THREE.Mesh(geo, mat);
                mesh.position.z = zPos;
                sliceGroup.add(mesh);
            }
        }

        function init3D() {
            const [zd, h, w] = volumeShape;

            scene = new THREE.Scene();
            camera = new THREE.PerspectiveCamera(
                50, window.innerWidth / window.innerHeight, 0.1, 100
            );
            camera.position.z = 2.5;

            renderer = new THREE.WebGLRenderer({ antialias: true });
            renderer.setSize(window.innerWidth, window.innerHeight);
            renderer.setClearColor(0x111111);
            document.getElementById('container').appendChild(renderer.domElement);

            sliceGroup = new THREE.Group();
            scene.add(sliceGroup);

            buildSlices(32, 30);

            // Controls
            document.getElementById('numSlices').addEventListener('input', (e) => {
                document.getElementById('sliceVal').textContent = e.target.value;
                const thresh = parseInt(document.getElementById('threshold').value);
                buildSlices(parseInt(e.target.value), thresh);
            });

            document.getElementById('threshold').addEventListener('input', (e) => {
                document.getElementById('threshVal').textContent = e.target.value;
                const num = parseInt(document.getElementById('numSlices').value);
                buildSlices(num, parseInt(e.target.value));
            });

            // Mouse controls
            renderer.domElement.addEventListener('mousedown', (e) => {
                isDragging = true;
                prevMouse = { x: e.clientX, y: e.clientY };
            });
            window.addEventListener('mouseup', () => isDragging = false);
            window.addEventListener('mousemove', (e) => {
                if (!isDragging) return;
                const dx = e.clientX - prevMouse.x;
                const dy = e.clientY - prevMouse.y;
                sliceGroup.rotation.y += dx * 0.01;
                sliceGroup.rotation.x += dy * 0.01;
                prevMouse = { x: e.clientX, y: e.clientY };
            });

            renderer.domElement.addEventListener('wheel', (e) => {
                camera.position.z += e.deltaY * 0.002;
                camera.position.z = Math.max(0.5, Math.min(5, camera.position.z));
            });

            window.addEventListener('resize', () => {
                camera.aspect = window.innerWidth / window.innerHeight;
                camera.updateProjectionMatrix();
                renderer.setSize(window.innerWidth, window.innerHeight);
            });

            animate();
        }

        function animate() {
            requestAnimationFrame(animate);
            renderer.render(scene, camera);
        }

        loadVolume();
    </script>
</body>
</html>"""

        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.send_header("Content-Length", len(html))
        self.end_headers()
        self.wfile.write(html.encode())


class ThreadedHTTPServer(socketserver.ThreadingMixIn, HTTPServer):
    allow_reuse_address = True
    daemon_threads = True


# =============================================================================
# Main
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Projection Explorer")
    parser.add_argument("--session", required=True, help="Session ID")
    parser.add_argument("--storage", default="D:/Gently", help="Storage path")
    parser.add_argument("--port", type=int, default=8766, help="HTTP port")
    parser.add_argument("--list", action="store_true", help="List embryos and exit")
    parser.add_argument("--embryo", help="Start with specific embryo")
    parser.add_argument("--export", action="store_true", help="Export comparison images")
    parser.add_argument("--output", help="Export output directory")
    parser.add_argument("--no-browser", action="store_true", help="Don't open browser")

    args = parser.parse_args()

    ensure_dependencies()

    # Find session
    session_path = Path(args.storage) / "images" / args.session
    if not session_path.exists():
        print(f"ERROR: Session not found: {session_path}")
        sys.exit(1)

    print(f"Session: {session_path}")

    # Initialize session manager
    sm = SessionManager(session_path)

    if not sm.embryo_list:
        print("ERROR: No embryos found in session")
        sys.exit(1)

    print(f"Found {len(sm.embryo_list)} embryo(s): {', '.join(sm.embryo_list)}")
    for eid in sm.embryo_list:
        print(f"  {eid}: {len(sm.embryo_volumes[eid])} timepoints")

    if args.list:
        sys.exit(0)

    if args.embryo:
        sm.switch_embryo(args.embryo)

    if args.export:
        # Export mode: save comparison images
        output_dir = Path(args.output) if args.output else Path("projection_comparison")
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nExporting to {output_dir}...")

        for idx in range(min(10, sm.total_timepoints)):  # First 10 timepoints
            sm.current_idx = idx
            volume = sm.get_current_volume()
            if volume is None:
                continue

            for method_name, func in PROJECTION_METHODS.items():
                img, desc = func(volume)

                # Save image
                pil_img = PIL_Image.fromarray(img)
                if pil_img.mode != "RGB":
                    pil_img = pil_img.convert("RGB")

                filename = f"t{idx:04d}_{method_name}.jpg"
                pil_img.save(output_dir / filename, quality=90)

            print(f"  Exported timepoint {idx}")

        print(f"\nDone! Images saved to {output_dir}")
        sys.exit(0)

    # Start web server
    ExplorerHandler.session_manager = sm
    server = ThreadedHTTPServer(("127.0.0.1", args.port), ExplorerHandler)

    url = f"http://127.0.0.1:{args.port}"
    print(f"\nProjection Explorer running at: {url}")
    print("Press Ctrl+C to stop\n")

    if not args.no_browser:
        import webbrowser

        webbrowser.open(url)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...")
        server.shutdown()


if __name__ == "__main__":
    main()
