#!/usr/bin/env python
"""
Extract stage examples from timelapse data for perception system training.

This script:
1. Loads timelapse volumes from D:/Gently/images/{session}/ or local paths
2. Creates dual-view projections (TOP | SIDE) matching perception engine format
3. Optionally uses existing hatching_detection_log.json for auto-labeling
4. Saves examples to gently/examples/stages/{stage}/

Usage:
    # Using session ID (from viz server) - data in D:/Gently/images/{session}/
    python scripts/extract_stage_examples.py --session 3fb70aca --embryo embryo_1

    # Using session ID with custom storage path
    python scripts/extract_stage_examples.py \
        --session exp_20251219_185542_f7df153a --storage D:/Gently --embryo embryo_1

    # Using direct source path
    python scripts/extract_stage_examples.py \
        --source multi_embryo_volumes/20251122_141240 --embryo embryo_001

    # Extract specific timepoint range
    python scripts/extract_stage_examples.py \
        --session 3fb70aca --embryo embryo_1 --start 50 --end 100
"""

import argparse
import base64
import io
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from gently.harness.perception.stages import STAGES

# Lazy imports
tifffile: Any = None
PIL_Image: Any = None


def ensure_dependencies():
    """Ensure required dependencies are available."""
    global tifffile, PIL_Image

    try:
        import tifffile as _tifffile

        tifffile = _tifffile
    except ImportError:
        print("ERROR: tifffile is required. Install with: pip install tifffile")
        sys.exit(1)

    try:
        from PIL import Image as _Image

        PIL_Image = _Image
    except ImportError:
        print("ERROR: Pillow is required. Install with: pip install Pillow")
        sys.exit(1)


def discover_volumes(session_dir: Path, embryo_id: str | None = None) -> dict[str, list[Path]]:
    """
    Discover volume files in a session directory.

    Returns dict mapping embryo_id -> list of volume paths (sorted by time).
    """
    if not session_dir.exists():
        return {}

    tif_files = list(session_dir.glob("*.tif")) + list(session_dir.glob("*.tiff"))

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
        result[eid] = [v[1] for v in volumes]

    return result


def normalize_image(img: np.ndarray) -> np.ndarray:
    """Normalize image to 8-bit using percentile scaling."""
    img = img.astype(np.float32)
    p_low = np.percentile(img, 1)
    p_high = np.percentile(img, 99.5)
    img = np.clip(img, p_low, p_high)
    if p_high > p_low:
        img = (img - p_low) / (p_high - p_low) * 255
    return img.astype(np.uint8)


def create_dual_view_projection(volume: np.ndarray) -> np.ndarray:
    """
    Create dual-view projection (TOP | SIDE) from 3D volume.

    This matches the gently perception system approach:
    - TOP: max projection along Z axis
    - SIDE: max projection along Y axis, rotated

    Parameters
    ----------
    volume : np.ndarray
        3D or 4D volume

    Returns
    -------
    np.ndarray
        Combined dual-view image (Y, X) with TOP on left, SIDE on right
    """
    # Handle 4D volumes: extract View A (first view)
    if volume.ndim == 4:
        view_a = volume[0]  # Extract View A -> (Z, Y, X)
    else:
        view_a = np.squeeze(volume)

    # Handle 2D input
    if view_a.ndim == 2:
        return normalize_image(view_a)

    # Handle 3D volumes
    if view_a.ndim == 3:
        z_depth, height, width = view_a.shape

        # Check if width contains dual-view data (views side-by-side in X)
        if width > height * 2:
            # Extract View A (left half)
            view_a = view_a[:, :, : width // 2]

        # TOP projection: max along Z axis (looking down at embryo)
        top_proj = np.max(view_a, axis=0)  # Shape: (Y, X)

        # SIDE projection: max along Y axis (looking from side)
        side_proj = np.max(view_a, axis=1)  # Shape: (Z, X)

        top_norm = normalize_image(top_proj)
        side_norm = normalize_image(side_proj)

        # Rotate side view 90 clockwise so Z becomes horizontal
        side_rotated = np.rot90(side_norm, k=-1)

        # Scale side view to match top view height
        target_height = top_norm.shape[0]
        new_width = max(150, int(side_rotated.shape[1] * target_height / side_rotated.shape[0]))
        side_pil = PIL_Image.fromarray(side_rotated).resize(
            (new_width, target_height), PIL_Image.Resampling.LANCZOS
        )
        side_scaled = np.array(side_pil)

        # Concatenate horizontally: TOP | separator | SIDE
        sep_width = 4
        separator = np.ones((target_height, sep_width), dtype=np.uint8) * 128
        combined = np.concatenate([top_norm, separator, side_scaled], axis=1)

        return combined

    # Fallback
    return normalize_image(view_a)


def image_to_jpeg_b64(image: np.ndarray, max_dim: int = 800, quality: int = 85) -> str:
    """Convert image to base64-encoded JPEG."""
    pil_img = PIL_Image.fromarray(image)

    # Convert to RGB if grayscale
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")

    # Resize if too large
    if max(pil_img.size) > max_dim:
        ratio = max_dim / max(pil_img.size)
        new_size = (int(pil_img.size[0] * ratio), int(pil_img.size[1] * ratio))
        pil_img = pil_img.resize(new_size, PIL_Image.Resampling.LANCZOS)

    # Convert to JPEG
    buffer = io.BytesIO()
    pil_img.save(buffer, format="JPEG", quality=quality)

    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def save_example(image: np.ndarray, stage: str, output_dir: Path) -> Path:
    """Save an example image to the appropriate stage folder."""
    stage_dir = output_dir / stage
    stage_dir.mkdir(parents=True, exist_ok=True)

    # Find next example number
    existing = list(stage_dir.glob("example_*.jpg"))
    next_num = len(existing) + 1
    filename = f"example_{next_num:03d}.jpg"
    filepath = stage_dir / filename

    # Save image
    pil_img = PIL_Image.fromarray(image)
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")

    # Resize to standard size (max 800px)
    max_dim = 800
    if max(pil_img.size) > max_dim:
        ratio = max_dim / max(pil_img.size)
        new_size = (int(pil_img.size[0] * ratio), int(pil_img.size[1] * ratio))
        pil_img = pil_img.resize(new_size, PIL_Image.Resampling.LANCZOS)

    pil_img.save(filepath, format="JPEG", quality=85)

    return filepath


def load_hatching_log(session_dir: Path) -> dict | None:
    """Load hatching_detection_log.json if it exists."""
    log_path = session_dir / "hatching_detection_log.json"
    if log_path.exists():
        with open(log_path) as f:
            return json.load(f)
    return None


def get_stage_from_log(log: dict, embryo_id: str, timepoint: int) -> str | None:
    """
    Extract stage from hatching detection log for a specific timepoint.

    The log structure varies, so we try to parse what's available.
    """
    if not log:
        return None

    # Try to find embryo data
    embryo_data = log.get(embryo_id, {})
    if not embryo_data:
        # Try alternate key formats
        for key in log.keys():
            if embryo_id in key:
                embryo_data = log[key]
                break

    # Look for detection history
    history = embryo_data.get("detection_history", [])
    for entry in history:
        if entry.get("timepoint") == timepoint:
            return entry.get("stage")

    return None


def save_preview_image(
    image: np.ndarray,
    timepoint: int,
    preview_dir: Path,
    suggested_stage: str | None = None,
) -> Path:
    """Save image to preview folder for manual review."""
    preview_dir.mkdir(parents=True, exist_ok=True)

    stage_suffix = f"_{suggested_stage}" if suggested_stage else ""
    filename = f"t{timepoint:04d}{stage_suffix}.jpg"
    filepath = preview_dir / filename

    pil_img = PIL_Image.fromarray(image)
    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")
    pil_img.save(filepath, format="JPEG", quality=90)

    return filepath


def interactive_classify(
    image: np.ndarray, timepoint: int, suggested_stage: str | None = None
) -> str | None:
    """
    Display image and let user classify interactively.

    Returns the selected stage or None to skip.
    """
    try:
        import cv2

        # Check if GUI is available
        display = image.copy()
        if display.ndim == 2:
            display = cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)

        # Add info overlay
        info = f"Timepoint: {timepoint}"
        if suggested_stage:
            info += f" | Suggested: {suggested_stage}"
        cv2.putText(display, info, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        # Add stage key legend
        legend = (
            "Keys: (e)arly (b)ean (c)omma (1).5fold (2)fold (3)fold"
            " (h)atching (d)one(hatched) (s)kip (q)uit"
        )
        cv2.putText(
            display,
            legend,
            (10, display.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (200, 200, 200),
            1,
        )

        cv2.imshow("Classify Stage", display)

        key = cv2.waitKey(0) & 0xFF

        key_map = {
            ord("e"): "early",
            ord("b"): "bean",
            ord("c"): "comma",
            ord("1"): "1.5fold",
            ord("2"): "2fold",
            ord("3"): "3fold",
            ord("h"): "hatching",
            ord("d"): "hatched",
            ord("s"): None,  # Skip
            ord("q"): "QUIT",
        }

        return key_map.get(key, None)

    except Exception:
        # GUI not available, return None to trigger preview mode
        return "NO_GUI"


def main():
    parser = argparse.ArgumentParser(description="Extract stage examples from timelapse data")

    # Source specification (either --session or --source)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--session",
        type=str,
        help="Session ID (e.g., 3fb70aca or exp_20251219_185542_f7df153a)",
    )
    source_group.add_argument(
        "--source",
        type=str,
        help="Direct source path (multi_embryo_volumes/SESSION or lightsheet_captures/batch)",
    )

    parser.add_argument(
        "--storage",
        type=str,
        default="D:/Gently",
        help="Storage path for session data (default: D:/Gently)",
    )
    parser.add_argument(
        "--embryo",
        type=str,
        default=None,
        help="Specific embryo ID to process (default: all)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory (default: gently/examples/stages)",
    )
    parser.add_argument(
        "--use-log",
        action="store_true",
        help="Use hatching_detection_log.json for auto-labeling",
    )
    parser.add_argument("--start", type=int, default=0, help="Start timepoint")
    parser.add_argument("--end", type=int, default=None, help="End timepoint")
    parser.add_argument(
        "--step", type=int, default=10, help="Sample every N timepoints (default: 10)"
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="Auto-save without interactive review (requires --use-log)",
    )
    parser.add_argument(
        "--preview",
        action="store_true",
        help="Save images to preview folder instead of interactive classification",
    )
    parser.add_argument(
        "--list-embryos",
        action="store_true",
        help="Just list available embryos and exit",
    )

    args = parser.parse_args()

    ensure_dependencies()

    # Resolve source directory
    if args.session:
        # Use session ID with storage path
        storage_path = Path(args.storage)
        source_dir = storage_path / "images" / args.session
        print(f"Using session: {args.session}")
        print(f"Storage path: {storage_path}")
    else:
        source_dir = Path(args.source)

    if not source_dir.exists():
        print(f"ERROR: Source directory not found: {source_dir}")
        sys.exit(1)

    # Set default output
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = Path(__file__).parent.parent / "gently" / "examples" / "stages"

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")

    # Discover volumes
    embryo_volumes = discover_volumes(source_dir, args.embryo)

    if not embryo_volumes:
        print(f"No volumes found in {source_dir}")
        sys.exit(1)

    print(f"\nFound {len(embryo_volumes)} embryo(s):")
    for eid, volumes in embryo_volumes.items():
        print(f"  {eid}: {len(volumes)} timepoints")

    if args.list_embryos:
        sys.exit(0)

    # Load hatching log if available
    hatching_log = None
    if args.use_log:
        hatching_log = load_hatching_log(source_dir)
        if hatching_log:
            print("\nLoaded hatching detection log")
        else:
            print("\nWARNING: No hatching_detection_log.json found")

    # Process each embryo
    for embryo_id, volume_paths in embryo_volumes.items():
        print(f"\n{'=' * 60}")
        print(f"Processing {embryo_id}: {len(volume_paths)} timepoints")
        print(f"{'=' * 60}")

        # Filter timepoints
        end_tp = args.end if args.end else len(volume_paths)
        selected = []
        for i, path in enumerate(volume_paths):
            if i >= args.start and i < end_tp and (i - args.start) % args.step == 0:
                selected.append((i, path))

        print(f"Selected {len(selected)} timepoints (step={args.step})")

        saved_count = {stage: 0 for stage in STAGES}

        for timepoint, vol_path in selected:
            # Load volume
            try:
                volume = tifffile.imread(str(vol_path))
            except Exception as e:
                print(f"  ERROR loading {vol_path.name}: {e}")
                continue

            # Create dual-view projection
            projection = create_dual_view_projection(volume)

            # Get suggested stage from log
            suggested_stage = None
            if hatching_log:
                suggested_stage = get_stage_from_log(hatching_log, embryo_id, timepoint)

            # Classify
            if args.auto and suggested_stage:
                # Auto-save with suggested stage
                stage = suggested_stage
            else:
                # Interactive classification
                stage = interactive_classify(projection, timepoint, suggested_stage)

            if stage == "QUIT":
                print("\nQuitting...")
                break

            if stage and stage in STAGES:
                filepath = save_example(projection, stage, output_dir)
                saved_count[stage] += 1
                print(f"  T{timepoint:03d}: Saved as {stage} -> {filepath.name}")
            elif stage:
                print(f"  T{timepoint:03d}: Skipped (invalid stage: {stage})")
            else:
                print(f"  T{timepoint:03d}: Skipped")

        # Summary
        print(f"\nSummary for {embryo_id}:")
        for stage, count in saved_count.items():
            if count > 0:
                print(f"  {stage}: {count} examples")

    # Final cleanup
    try:
        import cv2

        cv2.destroyAllWindows()
    except Exception:
        pass

    print("\nDone!")


if __name__ == "__main__":
    main()
