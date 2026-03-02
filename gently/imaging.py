"""
Standalone imaging utilities for Gently.

Single source of truth for image normalization, encoding, compression, and
3D volume projection. Used by perception, visualization, analysis, and
dataset modules.

Functions:
    normalize_to_uint8: Any-dtype image -> uint8 (percentile/minmax/simple)
    image_to_base64: uint8 array -> base64 string (JPEG or PNG)
    compress_image_for_api: 2D image -> base64 JPEG for Claude Vision API
    extract_view_a_and_max_project: 4D/3D volume -> 2D max projection
    generate_jpeg_projection: Volume -> JPEG file on disk
    compute_crop_bounds: 3D volume -> (y_min, y_max, x_min, x_max) crop bounds
    apply_crop_bounds: Crop a 3D volume to pre-computed bounds
    projection_three_view: 3D volume -> combined XY/YZ/XZ orthogonal view
    load_volume: TIFF file -> 3D numpy array (View A extracted)
    render_volume_view: 3D volume -> base64 depth-composited render
"""

import base64
import io
import logging
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


def normalize_to_uint8(
    image: np.ndarray,
    method: str = "percentile",
    p_low: float = 1.0,
    p_high: float = 99.0,
) -> np.ndarray:
    """Normalize any-dtype image to uint8.

    Parameters
    ----------
    image : np.ndarray
        Input image of any numeric dtype.
    method : str
        Normalization method:
        - "percentile": Clip to [p_low, p_high] percentiles, then scale.
          Best for microscopy where outlier hot pixels exist.
        - "minmax": Scale full dynamic range to 0-255.
        - "simple": Multiply by 255 (assumes input in [0, 1]).
    p_low : float
        Lower percentile for "percentile" method.
    p_high : float
        Upper percentile for "percentile" method.

    Returns
    -------
    np.ndarray
        uint8 image with values in [0, 255].
    """
    if image.dtype == np.uint8:
        return image

    img = image.astype(np.float32)

    if method == "simple":
        if img.max() <= 1.0:
            return (img * 255).astype(np.uint8)
        # Fall through to percentile if not 0-1 range
        method = "percentile"

    if method == "minmax":
        vmin, vmax = float(img.min()), float(img.max())
    else:  # percentile (default)
        vmin = float(np.percentile(img, p_low))
        vmax = float(np.percentile(img, p_high))

    if vmax > vmin:
        img = np.clip((img - vmin) / (vmax - vmin), 0, 1)
    else:
        img = np.zeros_like(img, dtype=np.float32)

    return (img * 255).astype(np.uint8)


def image_to_base64(
    image: np.ndarray,
    format: str = "JPEG",
    quality: int = 85,
    max_dimension: int = 0,
    ensure_rgb: bool = False,
) -> str:
    """Convert a uint8 numpy array to a base64-encoded image string.

    Call ``normalize_to_uint8`` first if the image is not already uint8.

    Parameters
    ----------
    image : np.ndarray
        uint8 image array (2D grayscale or 3D RGB/RGBA).
    format : str
        Output format: "JPEG" or "PNG".
    quality : int
        JPEG quality (ignored for PNG).
    max_dimension : int
        If > 0, resize so longest edge <= this value.
    ensure_rgb : bool
        Convert grayscale to RGB (needed for some APIs).

    Returns
    -------
    str
        Base64-encoded image string.
    """
    if not PIL_AVAILABLE:
        raise ImportError("Pillow is required: pip install Pillow")

    pil_image = Image.fromarray(image)

    if ensure_rgb and pil_image.mode not in ("RGB", "RGBA"):
        pil_image = pil_image.convert("RGB")

    if max_dimension > 0:
        w, h = pil_image.size
        if max(w, h) > max_dimension:
            scale = max_dimension / max(w, h)
            pil_image = pil_image.resize(
                (int(w * scale), int(h * scale)),
                Image.Resampling.LANCZOS,
            )

    buffer = io.BytesIO()
    save_kwargs = {"format": format, "optimize": True}
    if format.upper() == "JPEG":
        save_kwargs["quality"] = quality
    pil_image.save(buffer, **save_kwargs)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def extract_view_a_and_max_project(volume: np.ndarray) -> np.ndarray:
    """
    Extract View A from a diSPIM volume and create max-Z projection.

    Handles arbitrary dimensionality:
    - 1D: reshape to 2D
    - 2D: pass through
    - 3D (Z, Y, X): max along Z
    - 4D (Views, Z, Y, X): select View A (index 0), then max along Z

    Parameters
    ----------
    volume : np.ndarray
        Volume data of any dimensionality.

    Returns
    -------
    np.ndarray
        2D max projection (Y, X).
    """
    if volume.ndim == 1:
        side = int(np.sqrt(len(volume)))
        if side * side == len(volume):
            return volume.reshape(side, side)
        return volume.reshape(1, -1)

    if volume.ndim == 2:
        return volume

    # Squeeze singleton dims (e.g. (1, 1, 2048) -> (2048,))
    original_shape = volume.shape
    volume = np.squeeze(volume)
    if volume.shape != original_shape:
        logger.debug(f"Squeezed volume from {original_shape} to {volume.shape}")

    if volume.ndim <= 2:
        if volume.ndim == 1:
            return volume.reshape(1, -1)
        return volume

    # 4D (Views, Z, Y, X) -> select View A
    if volume.ndim == 4:
        volume = volume[0]

    # Max projection along Z
    max_proj = np.max(volume, axis=0)

    if max_proj.ndim == 1:
        max_proj = max_proj.reshape(1, -1)
    elif max_proj.ndim > 2:
        max_proj = np.squeeze(max_proj)
        if max_proj.ndim > 2:
            max_proj = np.max(max_proj, axis=0)

    return max_proj


def compress_image_for_api(
    image: np.ndarray,
    max_dimension: int = 800,
    quality: int = 85,
) -> tuple:
    """
    Compress a 2D image to base64-encoded JPEG for Claude Vision API.

    Parameters
    ----------
    image : np.ndarray
        2D image (Y, X) or (Y, X, C).
    max_dimension : int
        Maximum width or height in pixels.
    quality : int
        JPEG quality (1-100).

    Returns
    -------
    tuple of (str, float)
        (base64_string, size_in_kb)
    """
    if not PIL_AVAILABLE:
        raise ImportError("Pillow is required: pip install Pillow")

    # Reduce to 2D
    while image.ndim > 2:
        squeezed = False
        for axis in range(image.ndim):
            if image.shape[axis] == 1:
                image = image.squeeze(axis=axis)
                squeezed = True
                break
        if not squeezed:
            image = np.max(image, axis=0)

    if image.ndim == 1:
        image = image.reshape(1, -1)

    # Pad degenerate dimensions
    if image.ndim == 2:
        if image.shape[0] == 1:
            image = np.repeat(image, 10, axis=0)
        if image.shape[1] == 1:
            image = np.repeat(image, 10, axis=1)

    img = normalize_to_uint8(image, method="percentile", p_low=1, p_high=99.5)
    b64 = image_to_base64(img, format="JPEG", quality=quality,
                          max_dimension=max_dimension)
    size_kb = len(base64.b64decode(b64)) / 1024
    return b64, size_kb


def generate_jpeg_projection(
    volume: np.ndarray,
    output_path: Path,
    max_dimension: int = 1024,
    quality: int = 90,
) -> Optional[Path]:
    """
    Generate a JPEG max-projection from a volume and write to disk.

    Parameters
    ----------
    volume : np.ndarray
        3D or 4D volume.
    output_path : Path
        Where to write the JPEG file.
    max_dimension : int
        Max width/height in pixels.
    quality : int
        JPEG quality.

    Returns
    -------
    Path or None
        Path to the written JPEG, or None on failure.
    """
    if not PIL_AVAILABLE:
        logger.warning("Pillow not available, skipping projection generation")
        return None

    try:
        max_proj = extract_view_a_and_max_project(volume)
        normalized = normalize_to_uint8(max_proj, method="percentile",
                                        p_low=1, p_high=99.5)

        pil_image = Image.fromarray(normalized)

        # Resize if too large
        w, h = pil_image.size
        if max(w, h) > max_dimension:
            scale = max_dimension / max(w, h)
            pil_image = pil_image.resize(
                (int(w * scale), int(h * scale)),
                Image.Resampling.LANCZOS,
            )

        output_path.parent.mkdir(parents=True, exist_ok=True)
        pil_image.save(str(output_path), format="JPEG", quality=quality, optimize=True)
        return output_path

    except Exception as e:
        logger.error(f"Failed to generate projection: {e}")
        return None


# =============================================================================
# 3D Volume Projection Utilities
# =============================================================================
# Moved from gently/agent/perception/projection.py to fix layer violation
# (visualization/ was importing from agent/).

# Optional tifffile import (only needed for load_volume)
try:
    import tifffile as _tifffile
except ImportError:
    _tifffile = None


def load_volume(path: Path) -> np.ndarray:
    """Load a volume from TIFF file, extract View A.

    Parameters
    ----------
    path : Path
        Path to a TIFF file.

    Returns
    -------
    np.ndarray
        3D volume (Z, Y, X).
    """
    if _tifffile is None:
        raise ImportError("tifffile is required for load_volume")
    vol = _tifffile.imread(str(path))
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
    """Compute crop bounds for 3D volume using center-of-mass of bright pixels.

    Parameters
    ----------
    volume : np.ndarray
        3D volume (Z, Y, X).
    padding : int
        Extra padding around the detected region.
    sigma_mult : float
        Number of standard deviations to include.

    Returns
    -------
    tuple of (int, int, int, int)
        (y_min, y_max, x_min, x_max) bounds for cropping.
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
    """Apply pre-computed crop bounds to a volume.

    Parameters
    ----------
    volume : np.ndarray
        3D volume (Z, Y, X).
    bounds : tuple of (int, int, int, int)
        (y_min, y_max, x_min, x_max) from :func:`compute_crop_bounds`.

    Returns
    -------
    np.ndarray
        Cropped volume.
    """
    y_min, y_max, x_min, x_max = bounds
    return volume[
        :,
        max(0, y_min) : min(volume.shape[1], y_max),
        max(0, x_min) : min(volume.shape[2], x_max),
    ]


def projection_three_view(volume: np.ndarray) -> Tuple[np.ndarray, str]:
    """Generate three orthogonal views layout from a 3D volume.

    Layout::

        +----------+---+----------+
        |   XY     |   |   YZ     |  (TOP ROW)
        |  (top)   |   |  (side)  |
        +----------+---+----------+
        |         XZ               |  (BOTTOM ROW)
        |       (front)            |
        +--------------------------+

    Parameters
    ----------
    volume : np.ndarray
        3D volume with shape (Z, Y, X).

    Returns
    -------
    combined : np.ndarray
        Combined three-view image.
    description : str
        Description of the layout.
    """
    if not PIL_AVAILABLE:
        raise ImportError("Pillow is required for projection_three_view")

    if volume.ndim != 3:
        return normalize_to_uint8(volume), "2D input"

    z_depth, height, width = volume.shape

    xy_proj = normalize_to_uint8(np.max(volume, axis=0))
    xz_proj = normalize_to_uint8(np.max(volume, axis=1))
    yz_proj = normalize_to_uint8(np.max(volume, axis=2))

    xy_h, xy_w = xy_proj.shape
    z_display_h = max(xy_h // 3, int(z_depth * 3))

    pil_xz = Image.fromarray(xz_proj)
    pil_xz = pil_xz.resize((xy_w, z_display_h), Image.Resampling.LANCZOS)
    xz_scaled = np.array(pil_xz)

    yz_rotated = yz_proj.T
    z_display_w = z_display_h
    pil_yz = Image.fromarray(yz_rotated)
    pil_yz = pil_yz.resize((z_display_w, xy_h), Image.Resampling.LANCZOS)
    yz_scaled = np.array(pil_yz)

    sep = 3
    v_sep = np.ones((xy_h, sep), dtype=np.uint8) * 128
    top_row = np.concatenate([xy_proj, v_sep, yz_scaled], axis=1)
    total_width = top_row.shape[1]

    if xz_scaled.shape[1] < total_width:
        pad = np.zeros(
            (xz_scaled.shape[0], total_width - xz_scaled.shape[1]), dtype=np.uint8
        )
        bottom_row = np.concatenate([xz_scaled, pad], axis=1)
    else:
        bottom_row = xz_scaled[:, :total_width]

    h_sep = np.ones((sep, total_width), dtype=np.uint8) * 128
    combined = np.concatenate([top_row, h_sep, bottom_row], axis=0)
    return combined, "Three-view: [XY|YZ] top, [XZ] bottom"


def render_volume_view(
    volume: np.ndarray,
    rotation_x: float = 0,
    rotation_y: float = 0,
    threshold: float = 0.2,
) -> str:
    """Render a 3D volume from a specific viewing angle using alpha compositing.

    Parameters
    ----------
    volume : np.ndarray
        3D volume (Z, Y, X) or 4D (Views, Z, Y, X).
    rotation_x : float
        Rotation around X axis in degrees (-90 to 90).
    rotation_y : float
        Rotation around Y axis in degrees (-180 to 180).
    threshold : float
        Intensity threshold for transparency (0-1).

    Returns
    -------
    str
        Base64-encoded JPEG image.
    """
    from scipy import ndimage

    if volume.ndim == 4:
        volume = volume[0]

    vol = volume.astype(np.float32)
    p1, p99 = np.percentile(vol, [1, 99])
    vol = np.clip((vol - p1) / (p99 - p1 + 1e-8), 0, 1)

    if rotation_y != 0:
        vol = ndimage.rotate(vol, rotation_y, axes=(0, 2), reshape=False, order=1)
    if rotation_x != 0:
        vol = ndimage.rotate(vol, rotation_x, axes=(0, 1), reshape=False, order=1)

    z_depth = vol.shape[0]
    result = np.zeros(vol.shape[1:], dtype=np.float32)
    accumulated_alpha = np.zeros_like(result)

    for z in range(z_depth):
        slice_val = vol[z]
        alpha = np.clip((slice_val - threshold) / (1 - threshold + 1e-8), 0, 1) * 0.3
        result += slice_val * alpha * (1 - accumulated_alpha)
        accumulated_alpha += alpha * (1 - accumulated_alpha)

    if result.max() > 0:
        result = (result / result.max() * 255).astype(np.uint8)
    else:
        result = result.astype(np.uint8)

    return image_to_base64(result, format="JPEG", quality=85, max_dimension=800)
