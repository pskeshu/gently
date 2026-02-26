"""
Standalone imaging utilities for Gently.

Single source of truth for image normalization, encoding, and compression.
Used by perception, visualization, analysis, and dataset modules.

Functions:
    normalize_to_uint8: Any-dtype image -> uint8 (percentile/minmax/simple)
    image_to_base64: uint8 array -> base64 string (JPEG or PNG)
    compress_image_for_api: 2D image -> base64 JPEG for Claude Vision API
    extract_view_a_and_max_project: 4D/3D volume -> 2D max projection
    generate_jpeg_projection: Volume -> JPEG file on disk
"""

import base64
import io
import logging
from pathlib import Path
from typing import Optional

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
