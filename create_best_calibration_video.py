#!/usr/bin/env python3
"""
Create Best Calibration Video - Manually Curated

Uses the best quality images from different calibration sessions
to create the most representative workflow video for presentations.

Selection criteria:
- Phase 0: Nov 6/11 (best embryo visibility and centering)
- Phase 1: Nov 11 (good edge detection sequence)
- Phase 2-3: Oct 28 (only complete sweeps available)
"""

import cv2
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import tifffile


# Manually curated image selection
PHASE_IMAGES = {
    'phase0_centering': [
        ('calibration_images_embryo/embryo_centering_check_0.00um_20251106_110526.png',
         "Embryo centered at Piezo=0, Galvo=0", 0.0)
    ],

    'phase1a_top_edge': [
        ('calibration_images_embryo/top_edge_detection_0.00um_20251111_133824.png', "Start at center", 0.00),
        ('calibration_images_embryo/top_edge_detection_-0.05um_20251111_133829.png', "Sweeping up", -0.05),
        ('calibration_images_embryo/top_edge_detection_-0.10um_20251111_133835.png', "Finding top edge", -0.10),
    ],

    'phase1b_bottom_edge': [
        ('calibration_images_embryo/bottom_edge_detection_0.00um_20251111_133839.png', "Start at center - embryo visible", 0.00),
        ('calibration_images_embryo/bottom_edge_detection_0.05um_20251111_133844.png', "Sweeping down - still visible", 0.05),
        ('calibration_images_embryo/bottom_edge_detection_0.10um_20251111_133848.png', "Embryo getting fainter", 0.10),
        ('calibration_images_embryo/bottom_edge_detection_0.15um_20251111_133852.png', "Embryo fading...", 0.15),
        ('calibration_images_embryo/bottom_edge_detection_0.20um_20251111_133857.png', "Embryo very faint", 0.20),
        ('calibration_images_embryo/bottom_edge_detection_0.25um_20251111_133901.png', "EMPTY - Claude says 'no embryo' = EDGE FOUND", 0.25),
    ],

    'phase2_top_sweep': [
        ('calibration_images_embryo/top_sweep_-16.02um_20251028_150939.png', "Far from focus", -16.02),
        ('calibration_images_embryo/top_sweep_-12.02um_20251028_150940.png', "Approaching", -12.02),
        ('calibration_images_embryo/top_sweep_-8.02um_20251028_150941.png', "Getting closer", -8.02),
        ('calibration_images_embryo/top_sweep_-4.02um_20251028_150942.png', "Near focus", -4.02),
        ('calibration_images_embryo/top_sweep_-2.02um_20251028_150943.png', "Very close", -2.02),
        ('calibration_images_embryo/top_sweep_-0.02um_20251028_150943.png', "Best focus (FFT peak)", -0.02),
        ('calibration_images_embryo/top_sweep_1.98um_20251028_150944.png', "Past focus", 1.98),
        ('calibration_images_embryo/top_sweep_5.98um_20251028_150945.png', "Moving away", 5.98),
        ('calibration_images_embryo/top_sweep_9.98um_20251028_150946.png', "Far from focus", 9.98),
    ],

    'phase3_bottom_sweep': [
        ('calibration_images_embryo/bottom_sweep_-4.08um_20251028_150955.png', "Far from focus (blurry)", -4.08),
        ('calibration_images_embryo/bottom_sweep_-0.08um_20251028_150956.png', "Still out of focus", -0.08),
        ('calibration_images_embryo/bottom_sweep_1.92um_20251028_150957.png', "Approaching focus", 1.92),
        ('calibration_images_embryo/bottom_sweep_3.92um_20251028_150957.png', "Getting sharper", 3.92),
        ('calibration_images_embryo/bottom_sweep_5.92um_20251028_150958.png', "Good focus", 5.92),
        ('calibration_images_embryo/bottom_sweep_7.92um_20251028_150958.png', "Better focus", 7.92),
        ('calibration_images_embryo/bottom_sweep_9.92um_20251028_150959.png', "Excellent focus", 9.92),
        ('calibration_images_embryo/bottom_sweep_11.92um_20251028_150959.png', "BEST FOCUS - cells visible!", 11.92),
    ],
}

# Test volume file
TEST_VOLUME_PATH = Path("embryo_volume_test.tif")

# Phase configuration
PHASE_CONFIG = [
    ('phase0_centering', "PHASE 0: Initial Centering Check",
     "Claude AI vision: 'Is embryo visible and centered?'",
     (100, 200, 255), 30),  # Blue, hold longer for centering

    ('phase1a_top_edge', "PHASE 1.5a: Top Edge Detection",
     "Claude AI: Detect when embryo disappears (top boundary)",
     (255, 200, 100), 15),  # Orange

    ('phase1b_bottom_edge', "PHASE 1.5b: Bottom Edge Detection",
     "Claude AI: Detect when embryo disappears (bottom boundary)",
     (255, 150, 100), 15),  # Light orange

    ('phase2_top_sweep', "PHASE 2: Top Interior Calibration",
     "FFT bandpass focus scoring + Claude AI validation",
     (150, 255, 150), 10),  # Green

    ('phase3_bottom_sweep', "PHASE 3: Bottom Interior Calibration",
     "FFT bandpass focus scoring + Claude AI validation",
     (150, 255, 150), 10),  # Green

    ('phase4_test_volume', "PHASE 4: Test Volume Acquisition",
     "Calibrated system in action - hardware-triggered 3D scan",
     (255, 100, 255), 2),  # Magenta, fast playback
]


def add_text_overlay(img, title, subtitle, position_info, annotation, phase_color):
    """Add comprehensive text overlay to image."""
    # Convert to RGB if needed
    if img.mode != 'RGB':
        img = img.convert('RGB')

    img_overlay = img.copy()
    draw = ImageDraw.Draw(img_overlay)
    width, height = img.size

    # Load fonts
    try:
        title_font = ImageFont.truetype("arial.ttf", 40)
        subtitle_font = ImageFont.truetype("arial.ttf", 28)
        info_font = ImageFont.truetype("arial.ttf", 24)
        annot_font = ImageFont.truetype("arial.ttf", 22)
        method_font = ImageFont.truetype("arial.ttf", 18)
    except:
        title_font = ImageFont.load_default()
        subtitle_font = ImageFont.load_default()
        info_font = ImageFont.load_default()
        annot_font = ImageFont.load_default()
        method_font = ImageFont.load_default()

    # Phase banner
    banner_height = 80
    draw.rectangle([(0, 0), (width, banner_height)], fill=phase_color)
    title_bbox = draw.textbbox((0, 0), title, font=title_font)
    title_width = title_bbox[2] - title_bbox[0]
    draw.text(((width - title_width) // 2, 15), title, fill=(0, 0, 0), font=title_font)

    # Subtitle
    if subtitle:
        subtitle_bbox = draw.textbbox((0, 0), subtitle, font=subtitle_font)
        subtitle_width = subtitle_bbox[2] - subtitle_bbox[0]
        draw.rectangle([(0, banner_height), (width, banner_height + 60)], fill=(0, 0, 0))
        draw.text(((width - subtitle_width) // 2, banner_height + 15),
                 subtitle, fill=(255, 255, 255), font=subtitle_font)

    # Annotation (what's happening)
    if annotation:
        annot_bbox = draw.textbbox((0, 0), annotation, font=annot_font)
        annot_width = annot_bbox[2] - annot_bbox[0]
        y_pos = banner_height + 70 if subtitle else banner_height + 10
        draw.rectangle([(0, y_pos), (width, y_pos + 45)], fill=(50, 50, 50))
        draw.text(((width - annot_width) // 2, y_pos + 10),
                 annotation, fill=(255, 255, 100), font=annot_font)

    # Position info at bottom
    if position_info:
        info_bbox = draw.textbbox((0, 0), position_info, font=info_font)
        info_width = info_bbox[2] - info_bbox[0]
        draw.rectangle([(0, height - 70), (width, height)], fill=(0, 0, 0))
        draw.text(((width - info_width) // 2, height - 55),
                 position_info, fill=(255, 255, 255), font=info_font)

        # Add method note (empty for now, phase-specific info in subtitle)
        method_note = "Calibration workflow uses: Claude Vision API + FFT bandpass focus scoring"
        method_bbox = draw.textbbox((0, 0), method_note, font=method_font)
        method_width = method_bbox[2] - method_bbox[0]
        draw.text(((width - method_width) // 2, height - 25),
                 method_note, fill=(150, 150, 150), font=method_font)

    return img_overlay


def create_test_volume_frames(title, subtitle, color, hold_frames, target_size=(1024, 512)):
    """
    Create frames from test volume TIFF stack.

    Returns list of annotated frames ready for video.
    """
    print(f"    Loading TIFF stack from {TEST_VOLUME_PATH}...")

    # Load volume
    volume = tifffile.imread(str(TEST_VOLUME_PATH))
    print(f"    Volume shape: {volume.shape}")

    num_slices = volume.shape[0]

    # Normalize volume for display
    volume_norm = volume.astype(np.float32)
    vmin, vmax = np.percentile(volume_norm, [1, 99.5])
    volume_norm = np.clip((volume_norm - vmin) / (vmax - vmin) * 255, 0, 255).astype(np.uint8)

    print(f"    Creating {num_slices} slice frames (forward + backward pass)...")

    frames = []

    # Determine which camera view to use (left or right) based on first slice
    # This matches how calibration images select the best view
    first_slice = volume_norm[0]
    h, w = first_slice.shape
    mid_x = w // 2
    left_intensity = np.mean(first_slice[:, :mid_x])
    right_intensity = np.mean(first_slice[:, mid_x:])

    if left_intensity >= right_intensity:
        use_left_view = True
        crop_x = 0
        print(f"    Using LEFT camera view (intensity: {left_intensity:.1f} vs {right_intensity:.1f})")
    else:
        use_left_view = False
        crop_x = mid_x
        print(f"    Using RIGHT camera view (intensity: {left_intensity:.1f} vs {right_intensity:.1f})")

    # Forward pass
    for slice_idx in range(num_slices):
        img_array = volume_norm[slice_idx]

        # Crop to match calibration images (left or right camera view)
        # Volume is (512, 2048), calibration is (512, 1024)
        height, width = img_array.shape
        crop_width = target_size[0]  # 1024
        img_array_cropped = img_array[:, crop_x:crop_x + crop_width]

        # Convert to PIL for overlay
        img_pil = Image.fromarray(img_array_cropped)

        # Create position info
        position_info = f"Slice {slice_idx + 1}/{num_slices} (Z = {slice_idx * 0.5:.1f} µm)"
        annotation = f"3D volume pass-through (forward)"

        # Add overlay
        img_annotated = add_text_overlay(img_pil, title, subtitle, position_info, annotation, color)

        # Convert to OpenCV format
        frame = np.array(img_annotated)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        # Add multiple times
        for _ in range(hold_frames):
            frames.append(frame)

    # Backward pass for smooth loop
    for slice_idx in range(num_slices - 2, 0, -1):  # Skip first and last to avoid duplicate
        img_array = volume_norm[slice_idx]

        # Crop to same camera view
        height, width = img_array.shape
        crop_width = target_size[0]
        img_array_cropped = img_array[:, crop_x:crop_x + crop_width]

        img_pil = Image.fromarray(img_array_cropped)

        position_info = f"Slice {slice_idx + 1}/{num_slices} (Z = {slice_idx * 0.5:.1f} µm)"
        annotation = f"3D volume pass-through (backward)"

        img_annotated = add_text_overlay(img_pil, title, subtitle, position_info, annotation, color)
        frame = np.array(img_annotated)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        for _ in range(hold_frames):
            frames.append(frame)

    print(f"    Created {len(frames)} frames from volume")

    return frames


def create_best_quality_video(output_path, fps=10, include_test_volume=True):
    """Create video from manually curated best images."""
    print(f"\n{'='*70}")
    print("Creating Best Quality Calibration Video")
    print(f"{'='*70}")
    print(f"Using manually curated images from Nov 6/11 + Oct 28")
    if include_test_volume:
        print(f"Including test volume acquisition (Phase 4)")
    print(f"Output: {output_path}")
    print(f"FPS: {fps}\n")

    all_frames = []

    for phase_key, title, subtitle, color, hold_frames in PHASE_CONFIG:
        # Special handling for test volume
        if phase_key == 'phase4_test_volume':
            if not include_test_volume or not TEST_VOLUME_PATH.exists():
                print(f"  ! Skipping {phase_key} (file not found: {TEST_VOLUME_PATH})")
                continue

            print(f"\n  [{phase_key}] Loading 3D volume stack...")
            volume_frames = create_test_volume_frames(title, subtitle, color, hold_frames)
            all_frames.extend(volume_frames)
            continue
        if phase_key not in PHASE_IMAGES:
            print(f"  ! Skipping {phase_key} (not in selection)")
            continue

        phase_imgs = PHASE_IMAGES[phase_key]
        print(f"  [{phase_key}] {len(phase_imgs)} curated images")

        for img_path, annotation, position in phase_imgs:
            # Load image
            img = Image.open(img_path)

            # Create position string
            if 'edge' in phase_key:
                position_info = f"Galvo Y: {position:+.3f}°"
            elif 'sweep' in phase_key:
                position_info = f"Piezo: {position:.2f} µm"
            else:
                position_info = ""

            # Add overlay
            img_annotated = add_text_overlay(img, title, subtitle, position_info,
                                            annotation, color)

            # Convert to OpenCV format
            frame = np.array(img_annotated)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Add multiple times
            for _ in range(hold_frames):
                all_frames.append(frame)

    if len(all_frames) == 0:
        print("\n  X No frames created!")
        return False

    # Get dimensions
    height, width = all_frames[0].shape[:2]

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    if not video_writer.isOpened():
        print("\n  X Failed to open video writer!")
        return False

    # Write frames
    print(f"\n  Writing {len(all_frames)} frames...")
    for frame in tqdm(all_frames, desc="  Progress"):
        video_writer.write(frame)

    video_writer.release()

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    duration_s = len(all_frames) / fps

    print(f"\n  Video created successfully!")
    print(f"    Size: {file_size_mb:.1f} MB")
    print(f"    Duration: {duration_s:.1f} seconds")
    print(f"    Total frames: {len(all_frames)}")

    return True


def main():
    output_dir = Path("calibration_videos")
    output_dir.mkdir(exist_ok=True, parents=True)

    output_path = output_dir / "calibration_workflow_BEST_QUALITY.mp4"

    success = create_best_quality_video(output_path, fps=10)

    if success:
        print(f"\n{'='*70}")
        print("COMPLETE")
        print(f"{'='*70}")
        print(f"Best quality video: {output_path}")
        print(f"\nThis video uses:")
        print(f"  - Nov 6/11 centering (excellent embryo visibility)")
        print(f"  - Nov 11 edge detection (clear boundaries)")
        print(f"  - Oct 28 focus sweeps (only complete sweeps available)")
        print()
        return 0
    else:
        print("\n X Video creation failed!")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())
