"""
Image Processing Utilities for Real-time Hatching Detection
Converts 3D diSPIM volumes to images suitable for Claude Vision API
"""
import numpy as np
from PIL import Image
import io
import base64
from typing import Tuple
from pathlib import Path


def extract_view_a_and_max_project(volume: np.ndarray) -> np.ndarray:
    """
    Extract View A (left half) from twin-view diSPIM data and compute max projection

    Parameters
    ----------
    volume : np.ndarray
        3D volume stack with shape (Z, Y, X) where X contains both views side-by-side

    Returns
    -------
    np.ndarray
        2D max projection of View A with shape (Y, X/2)
    """
    if len(volume.shape) != 3:
        raise ValueError(f"Expected 3D volume, got shape {volume.shape}")

    z_slices, height, width = volume.shape

    # Extract left half (View A)
    mid_x = width // 2
    view_a = volume[:, :, :mid_x]

    # Take max projection along Z axis
    max_proj = np.max(view_a, axis=0)

    return max_proj


def normalize_image(array: np.ndarray) -> np.ndarray:
    """
    Normalize array to 0-255 uint8 range

    Parameters
    ----------
    array : np.ndarray
        Input array

    Returns
    -------
    np.ndarray
        Normalized uint8 array
    """
    if array.max() > array.min():
        normalized = ((array - array.min()) / (array.max() - array.min()) * 255).astype(np.uint8)
    else:
        normalized = np.zeros_like(array, dtype=np.uint8)

    return normalized


def compress_image_for_api(array: np.ndarray, max_size: int = 800,
                           quality: int = 85) -> Tuple[str, int]:
    """
    Compress image for Claude API with resizing and JPEG compression

    Parameters
    ----------
    array : np.ndarray
        Input image array
    max_size : int
        Maximum width/height (maintains aspect ratio)
    quality : int
        JPEG quality (1-100)

    Returns
    -------
    tuple
        (base64_string, compressed_size_bytes)
    """
    # Normalize to uint8
    img_normalized = normalize_image(array)

    # Convert to PIL Image
    img = Image.fromarray(img_normalized)

    # Resize if needed (maintain aspect ratio)
    if max(img.size) > max_size:
        img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

    # Convert to RGB if needed (for JPEG)
    if img.mode != 'RGB':
        img = img.convert('RGB')

    # Compress to JPEG in memory
    buffer = io.BytesIO()
    img.save(buffer, format='JPEG', quality=quality, optimize=True)
    buffer.seek(0)

    # Encode to base64
    img_bytes = buffer.read()
    b64_string = base64.standard_b64encode(img_bytes).decode("utf-8")

    return b64_string, len(img_bytes)


def process_volume_for_claude(volume: np.ndarray, max_size: int = 800,
                              quality: int = 85) -> Tuple[str, int, np.ndarray]:
    """
    Complete pipeline: volume → View A max projection → compressed base64 for Claude

    Parameters
    ----------
    volume : np.ndarray
        3D diSPIM volume (Z, Y, X) with twin views
    max_size : int
        Maximum image dimension for Claude
    quality : int
        JPEG compression quality

    Returns
    -------
    tuple
        (base64_string, size_bytes, max_projection_array)
    """
    # Extract View A and compute max projection
    max_proj = extract_view_a_and_max_project(volume)

    # Compress for API
    b64_string, size_bytes = compress_image_for_api(max_proj, max_size, quality)

    return b64_string, size_bytes, max_proj


def save_processed_image(max_proj: np.ndarray, output_path: Path):
    """
    Save processed max projection as PNG

    Parameters
    ----------
    max_proj : np.ndarray
        Max projection image
    output_path : Path
        Output file path
    """
    # Normalize and save
    img_normalized = normalize_image(max_proj)
    img = Image.fromarray(img_normalized)
    img.save(output_path)


class ImageHistory:
    """Maintains sliding window of recent images for temporal context"""

    def __init__(self, window_size: int = 10):
        """
        Parameters
        ----------
        window_size : int
            Number of recent images to keep
        """
        self.window_size = window_size
        self.history = {}  # embryo_id -> list of (timepoint, b64_image, size)

    def add_image(self, embryo_id: str, timepoint: int, b64_image: str, size: int):
        """Add image to history for embryo"""
        if embryo_id not in self.history:
            self.history[embryo_id] = []

        self.history[embryo_id].append({
            'timepoint': timepoint,
            'b64_image': b64_image,
            'size': size
        })

        # Keep only recent images
        if len(self.history[embryo_id]) > self.window_size:
            self.history[embryo_id] = self.history[embryo_id][-self.window_size:]

    def get_recent_images(self, embryo_id: str, num_images: int = None) -> list:
        """
        Get recent images for embryo

        Parameters
        ----------
        embryo_id : str
            Embryo identifier
        num_images : int, optional
            Number of recent images to return (None = all available)

        Returns
        -------
        list
            List of image dicts with timepoint, b64_image, size
        """
        if embryo_id not in self.history:
            return []

        images = self.history[embryo_id]

        if num_images is None:
            return images
        else:
            return images[-num_images:]

    def get_image_count(self, embryo_id: str) -> int:
        """Get number of images stored for embryo"""
        return len(self.history.get(embryo_id, []))

    def clear_embryo(self, embryo_id: str):
        """Clear history for specific embryo"""
        if embryo_id in self.history:
            del self.history[embryo_id]


# Example usage
if __name__ == "__main__":
    # Test with synthetic data
    print("Testing image processing utilities...")

    # Create synthetic volume (Z=50, Y=512, X=2048 for twin views)
    test_volume = np.random.randint(0, 4096, size=(50, 512, 2048), dtype=np.uint16)

    print(f"Input volume shape: {test_volume.shape}")

    # Process
    b64_image, size, max_proj = process_volume_for_claude(test_volume)

    print(f"Max projection shape: {max_proj.shape}")
    print(f"Compressed size: {size / 1024:.1f} KB")
    print(f"Base64 length: {len(b64_image)}")

    # Test image history
    history = ImageHistory(window_size=10)
    history.add_image("embryo_001", 0, b64_image, size)
    history.add_image("embryo_001", 1, b64_image, size)

    recent = history.get_recent_images("embryo_001")
    print(f"\nImage history: {len(recent)} images stored")

    print("\n✓ All tests passed!")
