"""
Auto-Label Example Images for Perception System

Samples images from a timelapse, uses Claude to auto-label stages,
and populates the examples folder with minimal user effort.

Usage:
    python scripts/auto_label_examples.py D:/path/to/timelapse --embryo 1

Then review with:
    python scripts/auto_label_examples.py --review
"""

import argparse
import asyncio
import base64
import io
import json
import sys
from pathlib import Path
from typing import Any, cast

import anthropic
import numpy as np
import tifffile
from anthropic.types import TextBlock
from PIL import Image

# Add parent to path for gently imports
sys.path.insert(0, str(Path(__file__).parent.parent))


STAGES = ["early", "comma", "pretzel", "3fold", "hatching", "hatched"]

# Staging prompt for Claude
STAGE_PROMPT = """You are classifying C. elegans embryo developmental stages from diSPIM
microscopy max projections.

The stages in order are:
1. EARLY - Round/oval shape, no visible folding, may see cell divisions
2. COMMA - Elongated comma shape, clear head/tail orientation beginning
3. PRETZEL - Folded 1.5-2x, pretzel-like loops visible
4. 3FOLD - Tightly folded 3x, worm shape clearly visible inside egg
5. HATCHING - Eggshell breach visible, worm partially emerging
6. HATCHED - Worm fully outside egg, may see deflated eggshell

Look at this image and classify the stage. Be concise.

Respond in JSON format:
{"stage": "comma", "confidence": 0.85, "notes": "clear comma shape visible"}
"""


def load_and_project(tif_path: Path) -> np.ndarray:
    """Load volume and create max intensity projection."""
    vol = tifffile.imread(tif_path)
    if vol.ndim == 3:
        proj = np.max(vol, axis=0)
    else:
        proj = vol
    return proj


def to_jpeg_b64(img: np.ndarray, quality: int = 85) -> str:
    """Convert numpy array to base64 JPEG."""
    # Normalize to 8-bit
    if img.dtype != np.uint8:
        img = ((img - img.min()) / (img.max() - img.min() + 1e-8) * 255).astype(np.uint8)

    pil_img = Image.fromarray(img)
    buffer = io.BytesIO()
    pil_img.save(buffer, format="JPEG", quality=quality)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def sample_timelapse(
    timelapse_dir: Path,
    embryo_num: int = 1,
    num_samples: int = 40,
) -> list[tuple[int, Path]]:
    """
    Sample images spread across timelapse to cover all stages.

    Returns list of (timepoint, filepath) tuples.
    """
    pattern = f"embryo_{embryo_num:03d}_*.tif"
    files = sorted(timelapse_dir.glob(pattern))

    if not files:
        # Try alternate pattern
        pattern = f"embryo_00{embryo_num}_*.tif"
        files = sorted(timelapse_dir.glob(pattern))

    if not files:
        raise ValueError(f"No files found matching embryo {embryo_num} in {timelapse_dir}")

    print(f"Found {len(files)} files for embryo {embryo_num}")

    # Extract timepoints
    def get_timepoint(f: Path) -> int:
        # Pattern: embryo_001_embryo001_t0042_...
        name = f.stem
        for part in name.split("_"):
            if part.startswith("t") and part[1:].isdigit():
                return int(part[1:])
        return 0

    files_with_tp = [(get_timepoint(f), f) for f in files]
    files_with_tp.sort(key=lambda x: x[0])

    # Sample evenly across timelapse
    total = len(files_with_tp)
    if total <= num_samples:
        return files_with_tp

    step = total / num_samples
    indices = [int(i * step) for i in range(num_samples)]
    return [files_with_tp[i] for i in indices]


async def classify_image(
    client: anthropic.Anthropic,
    image_b64: str,
    timepoint: int,
) -> dict:
    """Use Claude to classify a single image."""
    response = await asyncio.to_thread(
        client.messages.create,  # type: ignore[arg-type]  # overloaded SDK method not resolvable by to_thread
        model="claude-haiku-4-5-20251001",  # Fast and cheap for bulk labeling
        max_tokens=200,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": STAGE_PROMPT},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": image_b64,
                        },
                    },
                ],
            }
        ],
    )

    # Parse response
    text = cast(TextBlock, response.content[0]).text
    try:
        # Find JSON in response
        import re

        json_match = re.search(r"\{[^{}]+\}", text)
        if json_match:
            result = json.loads(json_match.group())
            result["timepoint"] = timepoint
            return result
    except Exception:
        pass

    return {
        "stage": "unknown",
        "confidence": 0.0,
        "notes": text[:100],
        "timepoint": timepoint,
    }


async def auto_label_batch(
    timelapse_dir: Path,
    embryo_num: int,
    num_samples: int = 40,
    output_file: Path | None = None,
) -> dict[str, list[dict]]:
    """
    Auto-label a batch of images from a timelapse.

    Returns dict mapping stage -> list of {timepoint, confidence, notes, filepath}
    """
    client = anthropic.Anthropic()

    # Sample images
    print(f"\nSampling {num_samples} images from timelapse...")
    samples = sample_timelapse(timelapse_dir, embryo_num, num_samples)

    # Process images
    results_by_stage: dict[str, list[dict]] = {stage: [] for stage in STAGES}
    results_by_stage["unknown"] = []

    print(f"Classifying {len(samples)} images with Claude...")

    for i, (timepoint, filepath) in enumerate(samples):
        # Progress
        if (i + 1) % 10 == 0:
            print(f"  Processed {i + 1}/{len(samples)}")

        try:
            # Load and convert
            proj = load_and_project(filepath)
            b64 = to_jpeg_b64(proj)

            # Classify
            result = await classify_image(client, b64, timepoint)
            result["filepath"] = str(filepath)
            result["image_b64"] = b64  # Keep for saving

            stage = result.get("stage", "unknown").lower()
            if stage not in results_by_stage:
                stage = "unknown"

            results_by_stage[stage].append(result)

        except Exception as e:
            print(f"  Error processing {filepath.name}: {e}")

    # Print summary
    print("\n" + "=" * 50)
    print("AUTO-LABELING COMPLETE")
    print("=" * 50)
    for stage in STAGES + ["unknown"]:
        count = len(results_by_stage[stage])
        if count > 0:
            avg_conf = sum(r["confidence"] for r in results_by_stage[stage]) / count
            timepoints = [r["timepoint"] for r in results_by_stage[stage]]
            print(
                f"  {stage:10s}: {count:3d} images (avg confidence: {avg_conf:.0%},"
                f" T={min(timepoints)}-{max(timepoints)})"
            )

    # Save results for review
    output_file = output_file or Path("labeled_results.json")
    save_data: dict[str, Any] = {
        stage: [{k: v for k, v in r.items() if k != "image_b64"} for r in results]
        for stage, results in results_by_stage.items()
    }
    save_data["_metadata"] = {
        "timelapse_dir": str(timelapse_dir),
        "embryo_num": embryo_num,
        "num_samples": len(samples),
    }

    with open(output_file, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nResults saved to {output_file}")

    return results_by_stage


def populate_examples(
    results_by_stage: dict[str, list[dict]],
    examples_dir: Path,
    max_per_stage: int = 3,
    min_confidence: float = 0.7,
):
    """
    Copy high-confidence examples to the examples folder.
    """
    examples_dir = Path(examples_dir)
    stages_dir = examples_dir / "stages"

    print("\n" + "=" * 50)
    print("POPULATING EXAMPLES FOLDER")
    print("=" * 50)

    for stage in STAGES:
        results = results_by_stage.get(stage, [])

        # Filter by confidence and sort
        good_results = [r for r in results if r.get("confidence", 0) >= min_confidence]
        good_results.sort(key=lambda x: x.get("confidence", 0), reverse=True)

        # Take top N
        to_save = good_results[:max_per_stage]

        if not to_save:
            print(f"  {stage}: No high-confidence examples found")
            continue

        # Create stage directory
        stage_dir = stages_dir / stage
        stage_dir.mkdir(parents=True, exist_ok=True)

        # Save examples
        for i, result in enumerate(to_save):
            # Reload and save if we have filepath
            filepath = Path(result["filepath"])
            if filepath.exists():
                proj = load_and_project(filepath)
                b64 = to_jpeg_b64(proj, quality=90)

                # Save as JPEG
                img_bytes = base64.b64decode(b64)
                out_path = stage_dir / f"example_{i + 1:03d}.jpg"
                with open(out_path, "wb") as f:
                    f.write(img_bytes)

                print(
                    f"  {stage}: Saved {out_path.name}"
                    f" (T={result['timepoint']}, conf={result['confidence']:.0%})"
                )

    print("\nDone! Examples populated.")


async def interactive_review(results_file: Path, examples_dir: Path):
    """
    Interactive review mode - show images and let user confirm/reject.
    """
    with open(results_file) as f:
        data = json.load(f)

    metadata = data.pop("_metadata", {})

    print("\n" + "=" * 50)
    print("INTERACTIVE REVIEW")
    print("=" * 50)
    print(f"Source: {metadata.get('timelapse_dir', 'unknown')}")
    print(f"Embryo: {metadata.get('embryo_num', '?')}")
    print()

    print("Summary of auto-labeled images:")
    for stage in STAGES + ["unknown"]:
        if stage in data and data[stage]:
            print(f"  {stage}: {len(data[stage])} images")

    print()
    print("Options:")
    print("  [a] Accept all high-confidence (>70%) examples")
    print("  [s] Show examples for a specific stage")
    print("  [p] Populate examples folder with current labels")
    print("  [q] Quit")

    while True:
        choice = input("\nChoice: ").strip().lower()

        if choice == "q":
            break
        elif choice == "a":
            # Reload with image data and populate
            print("\nReloading images and populating examples...")
            # We need to reload since we didn't save b64 data
            timelapse_dir = Path(metadata["timelapse_dir"])
            results = await auto_label_batch(
                timelapse_dir,
                metadata["embryo_num"],
                num_samples=40,
            )
            populate_examples(results, examples_dir)
            break
        elif choice == "p":
            # Just populate from saved file - need to reload images
            print("Reloading images...")
            results_with_images: dict = {}
            for stage, items in data.items():
                if stage.startswith("_"):
                    continue
                results_with_images[stage] = []
                for item in items:
                    if "filepath" in item and Path(item["filepath"]).exists():
                        item_copy = item.copy()
                        proj = load_and_project(Path(item["filepath"]))
                        item_copy["image_b64"] = to_jpeg_b64(proj)
                        results_with_images[stage].append(item_copy)
            populate_examples(results_with_images, examples_dir)
            break
        elif choice == "s":
            stage = (
                input("Which stage? (early/comma/pretzel/3fold/hatching/hatched): ").strip().lower()
            )
            if stage in data:
                print(f"\n{stage} examples:")
                for item in data[stage]:
                    print(
                        f"  T={item['timepoint']:04d}: {item.get('notes', 'no notes')}"
                        f" (conf={item.get('confidence', 0):.0%})"
                    )
            else:
                print(f"No examples for {stage}")


def main():
    parser = argparse.ArgumentParser(description="Auto-label example images for perception system")
    parser.add_argument("timelapse_dir", nargs="?", help="Path to timelapse directory")
    parser.add_argument("--embryo", "-e", type=int, default=1, help="Embryo number to sample")
    parser.add_argument("--samples", "-n", type=int, default=40, help="Number of images to sample")
    parser.add_argument("--review", "-r", action="store_true", help="Review existing labels")
    parser.add_argument(
        "--output", "-o", default="labeled_results.json", help="Output file for labels"
    )
    parser.add_argument("--examples-dir", default="gently/examples", help="Examples directory")
    parser.add_argument(
        "--auto-populate",
        "-a",
        action="store_true",
        help="Auto-populate without review",
    )

    args = parser.parse_args()

    examples_dir = Path(args.examples_dir)
    output_file = Path(args.output)

    if args.review:
        if not output_file.exists():
            print(f"No results file found at {output_file}")
            print("Run without --review first to generate labels.")
            return
        asyncio.run(interactive_review(output_file, examples_dir))
    elif args.timelapse_dir:
        timelapse_dir = Path(args.timelapse_dir)
        if not timelapse_dir.exists():
            print(f"Directory not found: {timelapse_dir}")
            return

        results = asyncio.run(
            auto_label_batch(
                timelapse_dir,
                args.embryo,
                args.samples,
                output_file,
            )
        )

        if args.auto_populate:
            populate_examples(results, examples_dir)
        else:
            print("\nRun with --review to interactively review and populate examples.")
            print("Or run with --auto-populate to automatically save high-confidence examples.")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
