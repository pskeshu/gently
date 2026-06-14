#!/usr/bin/env python
"""
Generate compact stage example montages from embryo 2 of session 59799c78.

Creates one montage per stage showing progression through that stage.
Format: columns = timepoints, rows = TOP/SIDE views (tightly cropped)
"""

import sys
from pathlib import Path

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.projection_explorer import (
    discover_volumes,
    ensure_dependencies,
    load_volume,
    normalize_image,
)

# Ground truth for embryo 2
EMBRYO_2_TRANSITIONS = {
    "early": 0,
    "bean": 33,
    "comma": 39,
    "1.5fold": 45,
    "2fold": 60,
    "pretzel": 80,
}

# Timepoints to sample for each stage (showing progression within stage)
# Pick timepoints that span the stage duration
STAGE_TIMEPOINTS = {
    "early": [5, 12, 20, 28],  # T0-32: 4 timepoints
    "bean": [34, 36, 38],  # T33-38: 3 timepoints (short stage)
    "comma": [40, 42, 44],  # T39-44: 3 timepoints (short stage)
    "1.5fold": [47, 52, 57],  # T45-59: 3 timepoints
    "2fold": [62, 68, 75],  # T60-79: 3 timepoints
    "pretzel": [82, 88, 95],  # T80+: 3 timepoints
}


def crop_to_content(img, padding=10):
    """Crop image tightly to embryo content with padding.

    Uses a robust approach: find the bounding box where intensity
    is significantly above the background level.
    """
    if img.ndim == 3:
        gray = np.mean(img, axis=2).astype(np.float32)
    else:
        gray = img.astype(np.float32)

    # Estimate background as the median of edge pixels
    edge_pixels = np.concatenate(
        [
            gray[0, :],  # top row
            gray[-1, :],  # bottom row
            gray[:, 0],  # left column
            gray[:, -1],  # right column
        ]
    )
    background = np.median(edge_pixels)

    # Threshold: significantly above background (background + 25% of dynamic range)
    dynamic_range = gray.max() - background
    if dynamic_range < 10:
        return img  # Low contrast image

    threshold = background + dynamic_range * 0.15

    # Find content region
    content_mask = gray > threshold

    # Use morphological operations to clean up noise
    from scipy import ndimage as ndi

    # Close small gaps, then open to remove small noise
    content_mask = ndi.binary_closing(content_mask, iterations=2)
    content_mask = ndi.binary_opening(content_mask, iterations=2)

    rows = np.any(content_mask, axis=1)
    cols = np.any(content_mask, axis=0)

    if not np.any(rows) or not np.any(cols):
        return img

    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]

    # Add padding
    y_min = max(0, y_min - padding)
    y_max = min(img.shape[0], y_max + padding + 1)
    x_min = max(0, x_min - padding)
    x_max = min(img.shape[1], x_max + padding + 1)

    cropped = img[y_min:y_max, x_min:x_max]
    return cropped


def create_stage_montage(volumes_list, timepoints, stage_name):
    """
    Create a compact montage for a stage.

    Format:
    [T1 TOP] [T2 TOP] [T3 TOP] ...
    [T1 SIDE] [T2 SIDE] [T3 SIDE] ...

    With tight cropping and labels.
    """
    from PIL import Image, ImageDraw, ImageFont

    top_views = []
    side_views = []
    valid_timepoints = []

    for tp in timepoints:
        if tp >= len(volumes_list):
            print(f"    Skipping T{tp} (out of range)")
            continue

        vol_path = volumes_list[tp]
        volume = load_volume(vol_path)
        if volume is None:
            print(f"    Failed to load T{tp}")
            continue

        # TOP view: max projection along Z
        top = normalize_image(np.max(volume, axis=0))
        top = crop_to_content(top, padding=3)

        # SIDE view: max projection along Y
        side = normalize_image(np.max(volume, axis=1))
        side = crop_to_content(side, padding=3)

        top_views.append(top)
        side_views.append(side)
        valid_timepoints.append(tp)
        print(f"    T{tp}: TOP {top.shape}, SIDE {side.shape}")

    if not top_views:
        return None

    # Scale SIDE views to match TOP width and have visible height
    # Same approach as dual_view in projection_explorer
    from PIL import Image as PILImage

    scaled_sides = []
    for _i, (top, side) in enumerate(zip(top_views, side_views, strict=False)):
        target_width = top.shape[1]  # Match TOP width
        # Scale height to be visible (3x original Z depth)
        target_height = max(top.shape[0] // 3, side.shape[0] * 3)

        pil_img = PILImage.fromarray(side)
        pil_img = pil_img.resize((target_width, target_height), PILImage.Resampling.LANCZOS)
        scaled_sides.append(np.array(pil_img))
    side_views = scaled_sides

    # Find max dimensions for uniform sizing
    max_top_h = max(t.shape[0] for t in top_views)
    max_top_w = max(t.shape[1] for t in top_views)
    max_side_h = max(s.shape[0] for s in side_views)
    max_side_w = max(s.shape[1] for s in side_views)

    # Use same width for both rows
    cell_w = max(max_top_w, max_side_w)

    # Pad and center each view
    def pad_center(img, target_h, target_w):
        result = np.zeros((target_h, target_w), dtype=np.uint8)
        h, w = img.shape[:2]
        y_off = (target_h - h) // 2
        x_off = (target_w - w) // 2
        if img.ndim == 3:
            img = img[:, :, 0] if img.shape[2] > 1 else img.squeeze()
        result[y_off : y_off + h, x_off : x_off + w] = img
        return result

    padded_tops = [pad_center(t, max_top_h, cell_w) for t in top_views]
    padded_sides = [pad_center(s, max_side_h, cell_w) for s in side_views]

    # Create separator
    sep_w = 2
    len(padded_tops)

    # Build rows with separators
    def build_row(views):
        row_parts = []
        for i, v in enumerate(views):
            if i > 0:
                sep = np.ones((v.shape[0], sep_w), dtype=np.uint8) * 40
                row_parts.append(sep)
            row_parts.append(v)
        return np.hstack(row_parts)

    top_row = build_row(padded_tops)
    side_row = build_row(padded_sides)

    # Horizontal separator between TOP and SIDE
    h_sep = np.ones((sep_w, top_row.shape[1]), dtype=np.uint8) * 40

    # Combine
    montage = np.vstack([top_row, h_sep, side_row])

    # Convert to PIL for adding labels
    img = Image.fromarray(montage)
    draw = ImageDraw.Draw(img)

    # Try to get a font, fall back to default
    try:
        font = ImageFont.truetype("arial.ttf", 12)
        small_font = ImageFont.truetype("arial.ttf", 10)
    except Exception:
        font = ImageFont.load_default()
        small_font = font

    # Add timepoint labels at bottom
    label_h = 16
    labeled_img = Image.new("L", (img.width, img.height + label_h), 0)
    labeled_img.paste(img, (0, 0))
    draw = ImageDraw.Draw(labeled_img)

    # Draw timepoint labels
    x_pos = 0
    for i, tp in enumerate(valid_timepoints):
        if i > 0:
            x_pos += sep_w
        cell_center = x_pos + cell_w // 2
        label = f"T{tp}"
        # Get text size
        bbox = draw.textbbox((0, 0), label, font=small_font)
        text_w = bbox[2] - bbox[0]
        draw.text(
            (cell_center - text_w // 2, img.height + 2),
            label,
            fill=200,
            font=small_font,
        )
        x_pos += cell_w

    # Add row labels on left
    final_w = labeled_img.width + 35
    final_img = Image.new("L", (final_w, labeled_img.height), 0)
    final_img.paste(labeled_img, (35, 0))
    draw = ImageDraw.Draw(final_img)

    # TOP label
    top_center_y = max_top_h // 2
    draw.text((2, top_center_y - 6), "TOP", fill=180, font=small_font)

    # SIDE label
    side_center_y = max_top_h + sep_w + max_side_h // 2
    draw.text((2, side_center_y - 6), "SIDE", fill=180, font=small_font)

    return np.array(final_img)


def main():
    ensure_dependencies()

    # Find session
    session_id = "59799c78"
    session_paths = [
        Path("D:/Gently/images") / session_id,
        Path("Z:/embryo_data") / session_id,
        Path("D:/embryo_data") / session_id,
        Path.home() / "embryo_data" / session_id,
    ]

    session_path = None
    for p in session_paths:
        if p.exists():
            session_path = p
            break

    if not session_path:
        print(f"ERROR: Session {session_id} not found")
        print(f"Tried: {session_paths}")
        return

    print(f"Found session: {session_path}")

    # Discover volumes
    embryo_volumes = discover_volumes(session_path)

    if "embryo_2" not in embryo_volumes:
        print("ERROR: embryo_2 not found in session")
        print(f"Available: {list(embryo_volumes.keys())}")
        return

    volumes_list = embryo_volumes["embryo_2"]
    print(f"Found {len(volumes_list)} timepoints for embryo_2")

    # Output directory
    output_dir = Path("gently/examples/stages")
    output_dir.mkdir(parents=True, exist_ok=True)

    import json

    from PIL import Image

    # Generate montage and save volumes for each stage
    for stage, timepoints in STAGE_TIMEPOINTS.items():
        print(f"\n=== {stage.upper()} ===")

        stage_dir = output_dir / stage
        stage_dir.mkdir(exist_ok=True)

        montage = create_stage_montage(volumes_list, timepoints, stage)

        if montage is not None:
            output_path = stage_dir / "progression.jpg"
            Image.fromarray(montage).save(output_path, quality=95)
            print(f"  Saved: {output_path} ({montage.shape[1]}x{montage.shape[0]})")
        else:
            print("  ERROR: Failed to create montage")

        # Save volumes for all timepoints in this stage
        volumes_dir = stage_dir / "volumes"
        volumes_dir.mkdir(exist_ok=True)

        saved_volumes = []
        for tp in timepoints:
            if tp >= len(volumes_list):
                continue
            vol_path = volumes_list[tp]
            volume = load_volume(vol_path)
            if volume is not None:
                # Save as compressed numpy with timepoint in filename
                volume_out = volumes_dir / f"T{tp:03d}.npz"
                np.savez_compressed(volume_out, volume=volume)
                saved_volumes.append(
                    {
                        "timepoint": tp,
                        "filename": f"T{tp:03d}.npz",
                        "shape": list(volume.shape),
                    }
                )
                print(f"  Saved volume: T{tp} ({volume.shape})")

        # Save metadata
        metadata = {
            "stage": stage,
            "timepoints": timepoints,
            "volumes": saved_volumes,
            "source_session": session_id,
            "source_embryo": "embryo_2",
        }
        with open(stage_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        print(f"  Saved metadata.json ({len(saved_volumes)} volumes)")

    print("\n=== Done ===")
    print(f"Examples saved to: {output_dir}")


if __name__ == "__main__":
    main()
