"""
Volume Helpers for the Visualization Server
=============================================

Shared volume loading, processing, and UID parsing utilities.
Consolidates duplicated code from route handlers.
"""

import logging
import re
from io import BytesIO
from pathlib import Path
from typing import Optional

import numpy as np

from gently.imaging import normalize_to_uint8, image_to_base64

logger = logging.getLogger(__name__)

# Compiled pattern for volume UID format: volume_{embryo_id}_t{timepoint}
VOLUME_UID_PATTERN = re.compile(r"volume_(.+)_t(\d+)$")


def parse_volume_uid(uid: str) -> Optional[tuple]:
    """Parse a volume UID into (embryo_id, timepoint) or return None."""
    if not uid.startswith("volume_"):
        return None
    match = VOLUME_UID_PATTERN.match(uid)
    if match:
        embryo_id, timepoint_str = match.groups()
        return embryo_id, int(timepoint_str)
    return None


def load_volume_from_disk(volume_path: str) -> np.ndarray:
    """Load and preprocess a volume from a TIFF file on disk.

    Handles dual-view format (diSPIM) and auto-crops to embryo region.

    Returns:
        Cropped 3D numpy array (Z, H, W)
    """
    import tifffile
    from gently.agent.perception.projection import (
        compute_crop_bounds,
        apply_crop_bounds,
    )

    path = Path(volume_path)
    if not path.exists():
        raise FileNotFoundError(f"Volume file not found: {volume_path}")

    vol = tifffile.imread(str(path))
    vol = np.squeeze(vol)

    # Handle dual-view format (diSPIM)
    if vol.ndim == 3:
        z_depth, height, width = vol.shape
        if width > height * 2:
            vol = vol[:, :, :width // 2]

    # Auto-crop to embryo region
    bounds = compute_crop_bounds(vol)
    vol = apply_crop_bounds(vol, bounds)

    return vol


def array_to_png_bytes(img_array: np.ndarray) -> bytes:
    """Convert a numpy array to PNG bytes."""
    from PIL import Image

    img_array = normalize_to_uint8(img_array, method="simple")
    img = Image.fromarray(img_array)
    buf = BytesIO()
    img.save(buf, format='PNG')
    return buf.getvalue()


def image_to_base64_png(img_array: np.ndarray) -> str:
    """Convert numpy array to base64-encoded PNG string."""
    img = normalize_to_uint8(img_array, method="simple")
    return image_to_base64(img, format="PNG")
