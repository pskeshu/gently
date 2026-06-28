"""
Offline Testset for Perception Benchmarks.

Loads session data and pairs with ground truth for sequential testing.
"""

import base64
import io
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .ground_truth import GroundTruth

# Lazy imports
tifffile: Any = None
PIL_Image: Any = None


def _ensure_dependencies():
    """Ensure required dependencies are available."""
    global tifffile, PIL_Image

    if tifffile is None:
        import tifffile as _tifffile

        tifffile = _tifffile

    if PIL_Image is None:
        from PIL import Image as _Image

        PIL_Image = _Image


@dataclass
class TestCase:
    """A single test case for perception benchmark."""

    embryo_id: str
    timepoint: int
    image_b64: str  # Combined view (for backward compatibility)
    top_image_b64: str | None  # TOP view only
    side_image_b64: str | None  # SIDE view only
    volume: np.ndarray | None
    ground_truth_stage: str | None
    acquired_at: datetime | None = None


def _discover_volumes(
    session_dir: Path, embryo_id: str | None = None
) -> dict[str, list[tuple[datetime, Path]]]:
    """Discover volume files (with parsed acquisition timestamps) in a session directory."""
    if not session_dir.exists():
        return {}

    tif_files = (
        list(session_dir.glob("*.tif"))
        + list(session_dir.glob("*.tiff"))
        + list(session_dir.glob("**/*.tif"))
        + list(session_dir.glob("**/*.tiff"))
    )
    # Deduplicate (flat + recursive may overlap)
    tif_files = list({f.resolve(): f for f in tif_files}.values())
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
        result[eid] = volumes

    return result


def _load_volume(path: Path) -> np.ndarray:
    """Load a volume from TIFF file."""
    from gently.core.imaging import load_volume

    return load_volume(path)


def _normalize_image(img: np.ndarray, p_low: float = 1, p_high: float = 99) -> np.ndarray:
    """Normalize image to 0-255 uint8."""
    img = img.astype(np.float32)
    vmin = np.percentile(img, p_low)
    vmax = np.percentile(img, p_high)
    if vmax > vmin:
        img = np.clip((img - vmin) / (vmax - vmin), 0, 1)
    else:
        img = np.zeros_like(img)
    return (img * 255).astype(np.uint8)


def _create_three_view_image(volume: np.ndarray, max_dim: int = 1500) -> str:
    """Create three-view orthogonal projection from volume, return base64.

    Uses the shared projection utility to generate:
    - XY (top-left): Looking down - best for shape, curvature, folding
    - YZ (top-right): Looking from side - best for depth, body height
    - XZ (bottom): Looking from front - best for symmetry, coiling

    Parameters
    ----------
    volume : np.ndarray
        3D volume array (Z, Y, X)
    max_dim : int
        Maximum dimension in pixels (default 1500 to stay under API limits)
    """
    _ensure_dependencies()

    from gently.core.imaging import (
        apply_crop_bounds,
        compute_crop_bounds,
        projection_three_view,
    )

    # Auto-crop to embryo region
    bounds = compute_crop_bounds(volume)
    cropped = apply_crop_bounds(volume, bounds)

    # Generate three-view projection
    three_view_img, _ = projection_three_view(cropped)

    # Convert to PIL for final processing
    pil_img = PIL_Image.fromarray(three_view_img)

    # Resize if too large (API limit is 8000px, use smaller for safety/performance)
    if max(pil_img.size) > max_dim:
        scale = max_dim / max(pil_img.size)
        new_size = (int(pil_img.size[0] * scale), int(pil_img.size[1] * scale))
        pil_img = pil_img.resize(new_size, PIL_Image.Resampling.LANCZOS)

    # Convert to JPEG base64
    buffer = io.BytesIO()
    pil_img.save(buffer, format="JPEG", quality=90)

    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _create_separate_view_images(volume: np.ndarray, max_dim: int = 1000) -> tuple[str, str]:
    """Create separate TOP and SIDE view images from volume, return base64 tuple.

    Parameters
    ----------
    volume : np.ndarray
        3D volume array (Z, Y, X)
    max_dim : int
        Maximum dimension in pixels

    Returns
    -------
    Tuple[str, str]
        (top_image_b64, side_image_b64)
    """
    _ensure_dependencies()

    z_depth, height, width = volume.shape

    # TOP: max along Z (axis 0) -> shape (Y, X) - looking down
    top_proj = np.max(volume, axis=0)

    # SIDE: max along Y (axis 1) -> shape (Z, X) - looking from front
    side_proj = np.max(volume, axis=1)

    # Normalize
    top_norm = _normalize_image(top_proj)
    side_norm = _normalize_image(side_proj)

    # Scale side view to make Z dimension more visible
    target_width = top_norm.shape[1]
    side_new_h = max(height // 3, int(z_depth * 3))

    side_pil = PIL_Image.fromarray(side_norm)
    side_scaled = side_pil.resize((target_width, side_new_h), PIL_Image.Resampling.LANCZOS)

    # Create PIL images
    top_pil = PIL_Image.fromarray(top_norm)

    # Resize if too large
    if max(top_pil.size) > max_dim:
        scale = max_dim / max(top_pil.size)
        new_size = (int(top_pil.size[0] * scale), int(top_pil.size[1] * scale))
        top_pil = top_pil.resize(new_size, PIL_Image.Resampling.LANCZOS)

    if max(side_scaled.size) > max_dim:
        scale = max_dim / max(side_scaled.size)
        new_size = (int(side_scaled.size[0] * scale), int(side_scaled.size[1] * scale))
        side_scaled = side_scaled.resize(new_size, PIL_Image.Resampling.LANCZOS)

    # Convert to JPEG base64
    top_buffer = io.BytesIO()
    top_pil.save(top_buffer, format="JPEG", quality=90)
    top_b64 = base64.b64encode(top_buffer.getvalue()).decode("utf-8")

    side_buffer = io.BytesIO()
    side_scaled.save(side_buffer, format="JPEG", quality=90)
    side_b64 = base64.b64encode(side_buffer.getvalue()).decode("utf-8")

    return top_b64, side_b64


class OfflineTestset:
    """
    Offline testset for perception benchmarks.

    Loads session data from disk and pairs with ground truth labels.
    Supports sequential iteration through embryos to simulate real-time acquisition.
    """

    def __init__(
        self,
        session_path: Path,
        ground_truth: GroundTruth,
        load_volumes: bool = True,
    ):
        """
        Parameters
        ----------
        session_path : Path
            Path to session directory containing TIF volumes
        ground_truth : GroundTruth
            Ground truth labels for this session
        load_volumes : bool
            Whether to load full 3D volumes (for view_embryo tool testing)
        """
        self.session_path = Path(session_path)
        self.ground_truth = ground_truth
        self.load_volumes = load_volumes

        # Discover available volumes
        self._embryo_volumes = _discover_volumes(self.session_path)

    @property
    def embryo_ids(self) -> list[str]:
        """Get list of embryo IDs with both volumes and ground truth."""
        gt_embryos = set(self.ground_truth.embryo_ids)
        vol_embryos = set(self._embryo_volumes.keys())
        return sorted(gt_embryos & vol_embryos)

    def get_timepoint_count(self, embryo_id: str) -> int:
        """Get number of timepoints for an embryo."""
        return len(self._embryo_volumes.get(embryo_id, []))

    def iter_embryo(
        self,
        embryo_id: str,
        start_timepoint: int = 0,
        end_timepoint: int | None = None,
    ) -> Iterator[TestCase]:
        """
        Iterate through timepoints for an embryo sequentially.

        Yields TestCase objects in temporal order, simulating real-time acquisition.

        Parameters
        ----------
        embryo_id : str
            Embryo ID to iterate
        start_timepoint : int
            First timepoint to include
        end_timepoint : int, optional
            Last timepoint to include (exclusive)

        Yields
        ------
        TestCase
            Test case with image, volume, and ground truth
        """
        if embryo_id not in self._embryo_volumes:
            return

        volumes = self._embryo_volumes[embryo_id]

        if end_timepoint is None:
            end_timepoint = len(volumes)

        for timepoint in range(start_timepoint, min(end_timepoint, len(volumes))):
            acquired_at, vol_path = volumes[timepoint]

            # Load volume
            volume = _load_volume(vol_path) if self.load_volumes else None

            # Create images
            if volume is not None:
                image_b64 = _create_three_view_image(volume)
                top_b64, side_b64 = _create_separate_view_images(volume)
            else:
                # Load just for image if not loading full volumes
                temp_vol = _load_volume(vol_path)
                image_b64 = _create_three_view_image(temp_vol)
                top_b64, side_b64 = _create_separate_view_images(temp_vol)
                del temp_vol

            # Get ground truth
            gt_stage = self.ground_truth.get_stage_at(embryo_id, timepoint)

            yield TestCase(
                embryo_id=embryo_id,
                timepoint=timepoint,
                image_b64=image_b64,
                top_image_b64=top_b64,
                side_image_b64=side_b64,
                volume=volume,
                ground_truth_stage=gt_stage,
                acquired_at=acquired_at,
            )

    def iter_all(self) -> Iterator[tuple[str, Iterator[TestCase]]]:
        """
        Iterate through all embryos in the testset.

        Yields (embryo_id, test_case_iterator) pairs.
        """
        for embryo_id in self.embryo_ids:
            yield embryo_id, self.iter_embryo(embryo_id)
