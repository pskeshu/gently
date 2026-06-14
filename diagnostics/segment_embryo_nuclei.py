"""
Cellpose nuclei segmentation script for embryo data.
Run with venv_cv: venv_cv/Scripts/python segment_embryo_nuclei.py
"""

from pathlib import Path

import napari
import numpy as np
import tifffile
from cellpose import models


def load_volume(tiff_path: Path) -> np.ndarray:
    """Load a volume from a TIFF file."""
    print(f"Loading volume from {tiff_path}")
    volume = tifffile.imread(tiff_path)
    print(f"  Shape: {volume.shape}, dtype: {volume.dtype}")
    return volume


def segment_nuclei_3d(volume: np.ndarray, diameter: float = 30.0) -> np.ndarray:
    """
    Segment nuclei using Cellpose with 2D-per-slice + stitching (fast 3D).
    """
    # Handle 4D volumes (C, Z, Y, X) - take first channel
    if volume.ndim == 4:
        volume = volume[0]
        print(f"  Extracted first channel, new shape: {volume.shape}")

    # Normalize to 0-255 for cellpose
    vol_norm = volume.astype(np.float32)
    vol_norm = (vol_norm - vol_norm.min()) / (vol_norm.max() - vol_norm.min() + 1e-8)
    vol_norm = (vol_norm * 255).astype(np.uint8)

    print(f"Segmenting nuclei with Cellpose stitch mode (diameter={diameter})")
    print(f"  Running 2D on {volume.shape[0]} slices, then stitching...")

    model = models.CellposeModel(gpu=True)
    masks, flows, styles = model.eval(
        vol_norm,
        diameter=diameter,
        do_3D=False,  # 2D per slice (fast!)
        z_axis=0,
        stitch_threshold=0.5,  # stitch 2D masks into 3D
        batch_size=64,  # larger batch for speed
    )

    n_nuclei = len(np.unique(masks)) - 1
    print(f"  Found {n_nuclei} nuclei")

    return masks


def main():
    # Specific embryo file
    tiff_path = Path("D:/Gently/images/a7dad590/embryo_1_t0001.tif")

    if not tiff_path.exists():
        print(f"File not found: {tiff_path}")
        return

    print(f"\nProcessing: {tiff_path.name}")

    # Load the volume
    volume = load_volume(tiff_path)

    # Handle 4D volumes
    if volume.ndim == 4:
        volume = volume[0]
        print(f"  Extracted first channel, new shape: {volume.shape}")

    # Crop to View A only (left half of image)
    mid_x = volume.shape[-1] // 2
    volume = volume[..., :mid_x]
    print(f"  Cropped to View A (left half): {volume.shape}")

    # Downsample by factor of 2
    from scipy.ndimage import zoom

    volume = zoom(volume, (1, 0.5, 0.5), order=1)
    print(f"  Downsampled 2x: {volume.shape}")

    # Segment nuclei in 3D (diameter halved due to downsampling)
    masks = segment_nuclei_3d(volume, diameter=15.0)

    # Visualize in Napari
    print("Opening Napari viewer...")
    viewer = napari.Viewer()
    viewer.add_image(volume, name="Volume", colormap="gray")
    viewer.add_labels(masks, name="Nuclei segmentation")
    napari.run()


if __name__ == "__main__":
    main()
