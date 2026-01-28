"""
Standalone imaging utilities for Gently.

Extracted from ImageManager so they can be used by GentlyStore and any
other component without depending on the ImageManager class.

Functions:
    extract_view_a_and_max_project: 4D/3D volume -> 2D max projection
    compress_image_for_api: 2D image -> base64 JPEG for Claude Vision API
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

    # Normalize to uint8
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            img_min = np.percentile(image, 1)
            img_max = np.percentile(image, 99.5)
            if img_max > img_min:
                image = np.clip(image, img_min, img_max)
                image = ((image - img_min) / (img_max - img_min) * 255).astype(np.uint8)
            else:
                image = np.zeros_like(image, dtype=np.uint8)

    pil_image = Image.fromarray(image)

    # Resize if too large
    width, height = pil_image.size
    if max(width, height) > max_dimension:
        scale = max_dimension / max(width, height)
        new_size = (int(width * scale), int(height * scale))
        pil_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)

    buffer = io.BytesIO()
    pil_image.save(buffer, format="JPEG", quality=quality, optimize=True)
    jpeg_bytes = buffer.getvalue()

    b64_string = base64.b64encode(jpeg_bytes).decode("utf-8")
    size_kb = len(jpeg_bytes) / 1024

    return b64_string, size_kb


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

        # Normalize to uint8
        if max_proj.dtype != np.uint8:
            if max_proj.max() <= 1.0 and max_proj.min() >= 0.0:
                normalized = (max_proj * 255).astype(np.uint8)
            else:
                p_lo = np.percentile(max_proj, 1)
                p_hi = np.percentile(max_proj, 99.5)
                if p_hi > p_lo:
                    clipped = np.clip(max_proj, p_lo, p_hi)
                    normalized = ((clipped - p_lo) / (p_hi - p_lo) * 255).astype(np.uint8)
                else:
                    normalized = np.zeros_like(max_proj, dtype=np.uint8)
        else:
            normalized = max_proj

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
