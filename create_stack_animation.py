#!/usr/bin/env python3
"""
Create Animated GIF from TIFF Stack

Takes a multi-page TIFF stack and creates an animated GIF that:
1. Plays through all Z-slices
2. Shows the max intensity projection at the end

Usage:
    python create_stack_animation.py <input_tiff> [output_gif]
"""

import sys
import numpy as np
import tifffile
from PIL import Image
from pathlib import Path


def create_animated_gif(input_path, output_path=None, fps=10, loop_max_proj=3):
    """
    Create animated GIF from TIFF stack with max projection finale.

    Args:
        input_path: Path to input TIFF stack
        output_path: Path for output GIF (optional)
        fps: Frames per second for animation
        loop_max_proj: Number of times to repeat max projection at end
    """
    input_path = Path(input_path)

    if output_path is None:
        output_path = Path("animated_video") / f"{input_path.stem}_animation.gif"
    else:
        output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print("Creating Animated GIF from TIFF Stack")
    print(f"{'='*70}")
    print(f"Input:  {input_path}")
    print(f"Output: {output_path}")
    print(f"FPS:    {fps}")

    # Load TIFF stack
    print(f"\nLoading TIFF stack...")
    stack = tifffile.imread(str(input_path))
    print(f"  Stack shape: {stack.shape}")
    print(f"  Data type: {stack.dtype}")

    num_slices = stack.shape[0]

    # Normalize to 8-bit
    print(f"\nNormalizing to 8-bit...")
    stack_norm = stack.astype(np.float32)
    vmin, vmax = np.percentile(stack_norm, [0.5, 99.5])
    print(f"  Intensity range: {vmin:.0f} - {vmax:.0f}")
    stack_norm = np.clip((stack_norm - vmin) / (vmax - vmin) * 255, 0, 255).astype(np.uint8)

    # Create max projection
    print(f"\nCreating max intensity projection...")
    max_proj = np.max(stack_norm, axis=0)

    # Create PIL images
    print(f"\nCreating animation frames...")
    frames = []

    # Add all slices
    for i in range(num_slices):
        img = Image.fromarray(stack_norm[i])

        # Add text overlay showing slice number
        from PIL import ImageDraw, ImageFont
        draw = ImageDraw.Draw(img)

        try:
            font = ImageFont.truetype("arial.ttf", 24)
        except:
            font = ImageFont.load_default()

        text = f"Slice {i+1}/{num_slices}"

        # Draw text with background
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        padding = 10
        draw.rectangle([(padding, padding),
                       (padding + text_width + 20, padding + text_height + 10)],
                      fill=0)  # Black background
        draw.text((padding + 10, padding + 5), text, fill=255, font=font)  # White text

        frames.append(img)

    # Add max projection at the end (repeat for emphasis)
    print(f"\nAdding max projection finale ({loop_max_proj} frames)...")
    max_proj_img = Image.fromarray(max_proj)

    draw = ImageDraw.Draw(max_proj_img)
    text = f"MAX PROJECTION ({num_slices} slices)"
    bbox = draw.textbbox((0, 0), text, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]

    draw.rectangle([(padding, padding),
                   (padding + text_width + 20, padding + text_height + 10)],
                  fill=0)  # Black background
    draw.text((padding + 10, padding + 5), text, fill=255, font=font)  # White text (yellow won't show in grayscale)

    for _ in range(loop_max_proj):
        frames.append(max_proj_img.copy())

    # Save as GIF
    print(f"\nSaving GIF...")
    duration_ms = int(1000 / fps)

    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0  # Loop forever
    )

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    total_frames = len(frames)
    duration_s = total_frames * duration_ms / 1000

    print(f"\n{'='*70}")
    print("COMPLETE")
    print(f"{'='*70}")
    print(f"Output: {output_path}")
    print(f"  Size: {file_size_mb:.1f} MB")
    print(f"  Frames: {total_frames} ({num_slices} slices + {loop_max_proj} max proj)")
    print(f"  Duration: {duration_s:.1f}s @ {fps} FPS")
    print(f"  Loop: Forever")
    print()


def main():
    if len(sys.argv) < 2:
        print("Usage: python create_stack_animation.py <input_tiff> [output_gif]")
        print("\nExample:")
        print("  python create_stack_animation.py scan_images/my_stack.tif")
        print("  python create_stack_animation.py scan_images/my_stack.tif animated_video/output.gif")
        return 1

    input_path = sys.argv[1]
    output_path = sys.argv[2] if len(sys.argv) > 2 else None

    if not Path(input_path).exists():
        print(f"Error: Input file not found: {input_path}")
        return 1

    create_animated_gif(input_path, output_path, fps=10, loop_max_proj=3)

    return 0


if __name__ == "__main__":
    sys.exit(main())
