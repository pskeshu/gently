"""
Image management for copilot analysis
Handles storage, retrieval, and compression for Claude Vision API
"""

import numpy as np
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
import tifffile
import base64
import io
from PIL import Image

from .state import ImageRecord, EmbryoState


def extract_view_a_and_max_project(volume: np.ndarray) -> np.ndarray:
    """
    Extract View A and create max projection

    Parameters
    ----------
    volume : np.ndarray
        3D volume (Z, Y, X)

    Returns
    -------
    np.ndarray
        2D max projection (Y, X)
    """
    # For diSPIM, View A is typically the full volume
    # Create max projection along Z axis
    max_proj = np.max(volume, axis=0)
    return max_proj


def compress_image_for_api(image: np.ndarray, max_dimension: int = 800,
                           quality: int = 85) -> tuple[str, float]:
    """
    Compress image for Claude Vision API

    Parameters
    ----------
    image : np.ndarray
        2D image (Y, X) or (Y, X, C)
    max_dimension : int
        Maximum width or height in pixels
    quality : int
        JPEG quality (1-100)

    Returns
    -------
    b64_string : str
        Base64-encoded JPEG
    size_kb : float
        Size in kilobytes
    """
    # Normalize to 0-255 uint8
    if image.dtype != np.uint8:
        # Handle different bit depths
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            # Normalize to range
            img_min, img_max = image.min(), image.max()
            if img_max > img_min:
                image = ((image - img_min) / (img_max - img_min) * 255).astype(np.uint8)
            else:
                image = np.zeros_like(image, dtype=np.uint8)

    # Convert to PIL Image
    pil_image = Image.fromarray(image)

    # Resize if too large
    width, height = pil_image.size
    if max(width, height) > max_dimension:
        scale = max_dimension / max(width, height)
        new_size = (int(width * scale), int(height * scale))
        pil_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)

    # Compress to JPEG
    buffer = io.BytesIO()
    pil_image.save(buffer, format='JPEG', quality=quality, optimize=True)
    jpeg_bytes = buffer.getvalue()

    # Base64 encode
    b64_string = base64.b64encode(jpeg_bytes).decode('utf-8')
    size_kb = len(jpeg_bytes) / 1024

    return b64_string, size_kb


class ImageManager:
    """Manages image storage and retrieval for copilot analysis"""

    def __init__(self, storage_path: Path, history_length: int = 10):
        """
        Parameters
        ----------
        storage_path : Path
            Directory to store volume TIFFs
        history_length : int
            Number of recent images to keep per embryo for temporal context
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.history_length = history_length

    def store_volume(self, embryo_state: EmbryoState, timepoint: int,
                     volume: np.ndarray) -> ImageRecord:
        """
        Store volume and generate max projection for analysis

        Parameters
        ----------
        embryo_state : EmbryoState
            Embryo being imaged
        timepoint : int
            Timepoint number
        volume : np.ndarray
            3D volume (Z, Y, X)

        Returns
        -------
        ImageRecord
            Record of stored image
        """
        embryo_id = embryo_state.id

        # Save full volume to disk
        volume_filename = f"{embryo_id}_t{timepoint:04d}.tif"
        volume_path = self.storage_path / volume_filename
        tifffile.imwrite(str(volume_path), volume, compression='zlib')

        # Generate max projection for Claude Vision
        max_proj = extract_view_a_and_max_project(volume)
        b64_image, size_kb = compress_image_for_api(max_proj)

        # Create record
        record = ImageRecord(
            embryo_id=embryo_id,
            timepoint=timepoint,
            timestamp=datetime.now(),
            volume_path=str(volume_path),
            max_projection_b64=b64_image,
            size_kb=size_kb
        )

        # Add to embryo's recent images (sliding window)
        embryo_state.recent_images.append(record)

        # Keep only last N
        if len(embryo_state.recent_images) > self.history_length:
            embryo_state.recent_images.pop(0)

        # Update embryo state
        embryo_state.last_imaged = datetime.now()
        embryo_state.timepoints_acquired = timepoint + 1

        return record

    def get_recent_context(self, embryo_state: EmbryoState,
                           num_images: int = 5) -> List[Dict]:
        """
        Get recent images for Claude Vision temporal context

        Parameters
        ----------
        embryo_state : EmbryoState
            Embryo to get images for
        num_images : int
            Number of recent images to include

        Returns
        -------
        List[Dict]
            List of image content blocks for Claude API
        """
        recent = embryo_state.recent_images[-num_images:]

        return [
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": img.max_projection_b64
                }
            }
            for img in recent
        ]

    def get_latest_image(self, embryo_state: EmbryoState) -> Optional[ImageRecord]:
        """Get most recent image for embryo"""
        if embryo_state.recent_images:
            return embryo_state.recent_images[-1]
        return None

    def load_volume(self, image_record: ImageRecord) -> np.ndarray:
        """Load full volume from disk"""
        return tifffile.imread(image_record.volume_path)
