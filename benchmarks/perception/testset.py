"""
Offline Testset for Perception Benchmarks.

Loads session data and pairs with ground truth for sequential testing.
"""

import base64
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Tuple, Dict

import numpy as np

from .ground_truth import GroundTruth

# Lazy imports
tifffile = None
PIL_Image = None


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
    image_b64: str
    volume: Optional[np.ndarray]
    ground_truth_stage: Optional[str]


def _discover_volumes(session_dir: Path, embryo_id: Optional[str] = None) -> Dict[str, List[Path]]:
    """Discover volume files in a session directory."""
    from datetime import datetime

    if not session_dir.exists():
        return {}

    tif_files = list(session_dir.glob("*.tif")) + list(session_dir.glob("*.tiff"))
    embryo_volumes = {}

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


def _load_volume(path: Path) -> np.ndarray:
    """Load a volume from TIFF file."""
    _ensure_dependencies()

    vol = tifffile.imread(str(path))
    vol = np.squeeze(vol)

    if vol.ndim == 3:
        z_depth, height, width = vol.shape
        # Extract View A (left half) if dual-view format
        if width > height * 2:
            vol = vol[:, :, :width // 2]

    return vol


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


def _create_dual_view_image(volume: np.ndarray) -> str:
    """Create dual-view (top + side) image from volume, return base64."""
    _ensure_dependencies()

    # Top view (Z max projection)
    top_proj = np.max(volume, axis=0)

    # Side view (Y max projection)
    side_proj = np.max(volume, axis=1)

    # Normalize
    top_norm = _normalize_image(top_proj)
    side_norm = _normalize_image(side_proj)

    # Combine side-by-side
    h_top, w_top = top_norm.shape
    h_side, w_side = side_norm.shape

    # Resize side view to match top view height
    side_pil = PIL_Image.fromarray(side_norm)
    new_width = int(w_side * h_top / h_side)
    side_resized = side_pil.resize((new_width, h_top), PIL_Image.Resampling.LANCZOS)
    side_arr = np.array(side_resized)

    # Combine
    combined = np.concatenate([top_norm, side_arr], axis=1)

    # Convert to JPEG base64
    pil_img = PIL_Image.fromarray(combined)
    buffer = io.BytesIO()
    pil_img.save(buffer, format="JPEG", quality=90)

    return base64.b64encode(buffer.getvalue()).decode("utf-8")


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
    def embryo_ids(self) -> List[str]:
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
        end_timepoint: Optional[int] = None,
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
            vol_path = volumes[timepoint]

            # Load volume
            volume = _load_volume(vol_path) if self.load_volumes else None

            # Create dual-view image
            if volume is not None:
                image_b64 = _create_dual_view_image(volume)
            else:
                # Load just for image if not loading full volumes
                temp_vol = _load_volume(vol_path)
                image_b64 = _create_dual_view_image(temp_vol)
                del temp_vol

            # Get ground truth
            gt_stage = self.ground_truth.get_stage_at(embryo_id, timepoint)

            yield TestCase(
                embryo_id=embryo_id,
                timepoint=timepoint,
                image_b64=image_b64,
                volume=volume,
                ground_truth_stage=gt_stage,
            )

    def iter_all(self) -> Iterator[Tuple[str, Iterator[TestCase]]]:
        """
        Iterate through all embryos in the testset.

        Yields (embryo_id, test_case_iterator) pairs.
        """
        for embryo_id in self.embryo_ids:
            yield embryo_id, self.iter_embryo(embryo_id)
