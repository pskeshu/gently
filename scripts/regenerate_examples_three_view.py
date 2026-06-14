"""
Regenerate stage reference images in three-view format.

For each stage with stored volumes:
1. Load representative volume from gently/examples/stages/{stage}/volumes/
2. Apply auto-crop to embryo region
3. Generate three-view projection
4. Save as three_view.jpg

Usage:
    python scripts/regenerate_examples_three_view.py
"""

# Add parent to path for imports
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

from gently.core.imaging import (
    apply_crop_bounds,
    compute_crop_bounds,
    projection_three_view,
)

STAGES_PATH = Path(__file__).parent.parent / "gently" / "examples" / "stages"
STAGES = ["early", "bean", "comma", "1.5fold", "2fold", "pretzel"]


def regenerate_stage(stage: str) -> bool:
    """
    Regenerate three-view image for a single stage.

    Returns True if successful, False otherwise.
    """
    stage_dir = STAGES_PATH / stage
    volumes_dir = stage_dir / "volumes"

    if not volumes_dir.exists():
        print(f"  No volumes directory for {stage}")
        return False

    npz_files = sorted(volumes_dir.glob("T*.npz"))
    if not npz_files:
        print(f"  No volume files found for {stage}")
        return False

    # Use middle timepoint for best representation
    middle_idx = len(npz_files) // 2
    vol_path = npz_files[middle_idx]

    print(f"  Loading {vol_path.name}...")

    # Load volume
    data = np.load(vol_path)
    volume = data["volume"]

    print(f"  Volume shape: {volume.shape}")

    # Handle 4D volumes (Views, Z, Y, X) - take first view
    if volume.ndim == 4:
        volume = volume[0]

    # Handle dual-view format in X dimension
    if volume.ndim == 3:
        z_depth, height, width = volume.shape
        if width > height * 2:
            volume = volume[:, :, : width // 2]
            print(f"  Extracted View A: {volume.shape}")

    # Auto-crop to embryo region
    bounds = compute_crop_bounds(volume)
    volume = apply_crop_bounds(volume, bounds)
    print(f"  Cropped shape: {volume.shape}")

    # Generate three-view projection
    three_view_img, description = projection_three_view(volume)
    print(f"  Generated: {description}")

    # Save as JPEG
    pil_img = Image.fromarray(three_view_img)
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")

    output_path = stage_dir / "three_view.jpg"
    pil_img.save(output_path, quality=90)
    print(f"  Saved: {output_path}")

    return True


def main():
    print("Regenerating stage examples in three-view format")
    print("=" * 50)

    success_count = 0
    for stage in STAGES:
        print(f"\n[{stage}]")
        if regenerate_stage(stage):
            success_count += 1

    print("\n" + "=" * 50)
    print(f"Complete: {success_count}/{len(STAGES)} stages regenerated")


if __name__ == "__main__":
    main()
