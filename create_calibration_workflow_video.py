#!/usr/bin/env python3
"""
Create Calibration Workflow Video

Reconstructs the embryo calibration workflow from saved images,
adding annotations for each phase to create a presentation-ready video.

Phases illustrated:
1. Phase 0: Initial embryo centering check
2. Phase 1.5a: Top edge detection (sweeping to find top boundary)
3. Phase 1.5b: Bottom edge detection (sweeping to find bottom boundary)
4. Phase 2: Top calibration (focus sweep at interior position)
5. Phase 3: Bottom calibration (focus sweep at interior position)
6. Summary: Show final calibration results

Usage:
    python create_calibration_workflow_video.py
    python create_calibration_workflow_video.py --session 20251028_150914
"""

import argparse
import re
from pathlib import Path
from collections import defaultdict
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm


# Configuration
IMAGE_DIR = Path("calibration_images_embryo")
OUTPUT_DIR = Path("calibration_videos")
DEFAULT_FPS = 10


def extract_session_timestamp(filename):
    """
    Extract session timestamp from filename, grouped by 10-minute window.

    This groups images from the same calibration run together, since
    images are typically captured over ~5-10 minutes.
    """
    match = re.search(r'(\d{8}_\d{4})', filename.name)  # Match YYYYMMDD_HHMM
    if match:
        # Round down to nearest 10-minute window
        date_time = match.group(1)
        date_part = date_time[:9]  # YYYYMMDD_
        hour_minute = date_time[9:13]  # HHMM

        hour = int(hour_minute[:2])
        minute = int(hour_minute[2:4])

        # Round to 10-minute window
        minute_window = (minute // 10) * 10

        return f"{date_part}{hour:02d}{minute_window:02d}"
    return None


def extract_position(filename):
    """Extract position in microns from filename."""
    match = re.search(r'([-]?\d+\.\d+)um', filename.name)
    if match:
        return float(match.group(1))
    return 0.0


def group_images_by_session():
    """
    Group all calibration images by session timestamp.

    Returns:
        dict: {session_timestamp: {phase_name: [(position, filepath), ...]}}
    """
    sessions = defaultdict(lambda: defaultdict(list))

    for img_file in IMAGE_DIR.glob("*.png"):
        if img_file.name == "top_sweep_montage.png":
            continue

        session_ts = extract_session_timestamp(img_file)
        if not session_ts:
            continue

        position = extract_position(img_file)

        # Categorize by phase
        if "embryo_centering_check" in img_file.name:
            phase = "phase0_centering"
        elif "top_edge_detection" in img_file.name:
            phase = "phase1a_top_edge"
        elif "bottom_edge_detection" in img_file.name:
            phase = "phase1b_bottom_edge"
        elif "top_sweep" in img_file.name:
            phase = "phase2_top_sweep"
        elif "bottom_sweep" in img_file.name:
            phase = "phase3_bottom_sweep"
        elif "top_assessment" in img_file.name:
            phase = "phase1c_top_assessment"
        elif "bottom_assessment" in img_file.name:
            phase = "phase1d_bottom_assessment"
        else:
            phase = "unknown"

        sessions[session_ts][phase].append((position, img_file))

    # Sort images within each phase by position
    for session_data in sessions.values():
        for phase in session_data:
            session_data[phase].sort(key=lambda x: x[0])

    return sessions


def find_complete_sessions(sessions):
    """Find sessions that have all phases."""
    complete_sessions = []

    for session_ts, phases in sessions.items():
        # Check if session has edge detection and at least one sweep
        has_edge = ("phase1a_top_edge" in phases and "phase1b_bottom_edge" in phases)
        has_sweep = ("phase2_top_sweep" in phases or "phase3_bottom_sweep" in phases)

        if has_edge:
            score = 0
            if "phase0_centering" in phases:
                score += 1
            if "phase2_top_sweep" in phases:
                score += 2
            if "phase3_bottom_sweep" in phases:
                score += 2

            complete_sessions.append((session_ts, score, phases))

    # Sort by completeness score
    complete_sessions.sort(key=lambda x: x[1], reverse=True)

    return complete_sessions


def add_text_overlay(img, title, subtitle="", position_info="", phase_color=(0, 255, 255)):
    """
    Add text overlay to image with phase information.

    Args:
        img: PIL Image
        title: Main title (phase name)
        subtitle: Subtitle text
        position_info: Position information (e.g., "Piezo: 5.2 µm, Galvo: +0.15°")
        phase_color: RGB color for phase title banner

    Returns:
        PIL Image with overlay
    """
    # Convert to RGB if grayscale
    if img.mode != 'RGB':
        img = img.convert('RGB')

    # Create copy
    img_overlay = img.copy()
    draw = ImageDraw.Draw(img_overlay)

    width, height = img.size

    # Try to load a good font, fall back to default
    try:
        title_font = ImageFont.truetype("arial.ttf", 40)
        subtitle_font = ImageFont.truetype("arial.ttf", 28)
        info_font = ImageFont.truetype("arial.ttf", 24)
    except:
        title_font = ImageFont.load_default()
        subtitle_font = ImageFont.load_default()
        info_font = ImageFont.load_default()

    # Draw phase title banner at top
    banner_height = 80
    draw.rectangle([(0, 0), (width, banner_height)], fill=phase_color)

    # Draw title text (centered)
    title_bbox = draw.textbbox((0, 0), title, font=title_font)
    title_width = title_bbox[2] - title_bbox[0]
    title_x = (width - title_width) // 2
    draw.text((title_x, 15), title, fill=(0, 0, 0), font=title_font)

    # Draw subtitle if provided
    if subtitle:
        subtitle_bbox = draw.textbbox((0, 0), subtitle, font=subtitle_font)
        subtitle_width = subtitle_bbox[2] - subtitle_bbox[0]
        subtitle_x = (width - subtitle_width) // 2
        draw.rectangle([(0, banner_height), (width, banner_height + 60)],
                      fill=(0, 0, 0, 200))
        draw.text((subtitle_x, banner_height + 15), subtitle, fill=(255, 255, 255),
                 font=subtitle_font)

    # Draw position info at bottom
    if position_info:
        info_bbox = draw.textbbox((0, 0), position_info, font=info_font)
        info_width = info_bbox[2] - info_bbox[0]
        info_x = (width - info_width) // 2
        draw.rectangle([(0, height - 50), (width, height)], fill=(0, 0, 0, 200))
        draw.text((info_x, height - 40), position_info, fill=(255, 255, 255),
                 font=info_font)

    return img_overlay


def create_workflow_video(session_ts, phases, output_path, fps=10):
    """
    Create annotated workflow video from session images.

    Args:
        session_ts: Session timestamp
        phases: Dict of phase data
        output_path: Output video file path
        fps: Frames per second
    """
    print(f"\n{'='*70}")
    print(f"Creating Calibration Workflow Video")
    print(f"{'='*70}")
    print(f"Session: {session_ts}")
    print(f"Output: {output_path}")
    print(f"FPS: {fps}")

    # Phase configuration: (phase_key, title, subtitle, color, hold_frames)
    phase_config = [
        ("phase0_centering", "PHASE 0: Initial Centering Check",
         "Verify embryo is visible at center (Piezo=0, Galvo=0)",
         (100, 200, 255), 20),

        ("phase1a_top_edge", "PHASE 1.5a: Top Edge Detection",
         "Sweeping galvo to find top boundary of embryo",
         (255, 200, 100), 2),

        ("phase1b_bottom_edge", "PHASE 1.5b: Bottom Edge Detection",
         "Sweeping galvo to find bottom boundary of embryo",
         (255, 150, 100), 2),

        ("phase2_top_sweep", "PHASE 2: Top Calibration (Focus Sweep)",
         "Finding optimal piezo position at top interior location",
         (150, 255, 150), 3),

        ("phase3_bottom_sweep", "PHASE 3: Bottom Calibration (Focus Sweep)",
         "Finding optimal piezo position at bottom interior location",
         (150, 255, 150), 3),
    ]

    # Collect all frames
    all_frames = []

    for phase_key, title, subtitle, color, hold_frames in phase_config:
        if phase_key not in phases:
            print(f"  ! Skipping {phase_key} (no images)")
            continue

        phase_images = phases[phase_key]
        print(f"\n  [{phase_key}] {len(phase_images)} images")

        for position, img_path in phase_images:
            # Load image
            img = Image.open(img_path)

            # Add overlay with position info
            if "edge" in phase_key:
                position_info = f"Galvo: {position:+.3f}°"
            elif "sweep" in phase_key:
                position_info = f"Piezo: {position:.2f} µm"
            else:
                position_info = ""

            img_annotated = add_text_overlay(img, title, subtitle, position_info, color)

            # Convert to numpy array for OpenCV
            frame = np.array(img_annotated)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Add frame multiple times based on hold_frames
            for _ in range(hold_frames):
                all_frames.append(frame)

    if len(all_frames) == 0:
        print(f"\n  X No frames to write!")
        return

    # Get frame dimensions from first frame
    height, width = all_frames[0].shape[:2]

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    if not video_writer.isOpened():
        print(f"\n  X Failed to open video writer!")
        return

    # Write all frames
    print(f"\n  Writing {len(all_frames)} frames to video...")
    for frame in tqdm(all_frames, desc="  Progress"):
        video_writer.write(frame)

    video_writer.release()

    file_size_mb = output_path.stat().st_size / (1024 * 1024)
    duration_s = len(all_frames) / fps

    print(f"\n  Video created successfully!")
    print(f"    Size: {file_size_mb:.1f} MB")
    print(f"    Duration: {duration_s:.1f} seconds")
    print(f"    Total frames: {len(all_frames)}")


def main():
    parser = argparse.ArgumentParser(
        description="Create annotated calibration workflow video from saved images",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--session', type=str, default=None,
                       help='Session timestamp to use (YYYYMMDD_HHMMSS), or auto-select most complete')
    parser.add_argument('--fps', type=int, default=DEFAULT_FPS,
                       help='Frames per second for output video')
    parser.add_argument('--output', type=str, default=None,
                       help='Output video filename (default: calibration_workflow_SESSION.mp4)')
    parser.add_argument('--list', action='store_true',
                       help='List available calibration sessions and exit')

    args = parser.parse_args()

    # Check image directory exists
    if not IMAGE_DIR.exists():
        print(f"X Error: Image directory not found: {IMAGE_DIR}")
        print(f"  Make sure calibration images have been saved.")
        return 1

    # Group images by session
    print(f"Scanning calibration images in: {IMAGE_DIR}")
    sessions = group_images_by_session()
    print(f"Found {len(sessions)} calibration sessions")

    # Find complete sessions
    complete_sessions = find_complete_sessions(sessions)

    if args.list:
        print(f"\n{'='*70}")
        print("Available Calibration Sessions")
        print(f"{'='*70}\n")

        for i, (session_ts, score, phases) in enumerate(complete_sessions, 1):
            print(f"{i}. Session: {session_ts} (completeness: {score}/5)")
            for phase_key in sorted(phases.keys()):
                num_images = len(phases[phase_key])
                print(f"   - {phase_key}: {num_images} images")
            print()

        return 0

    if len(complete_sessions) == 0:
        print(f"\nX No complete calibration sessions found!")
        print(f"  Total sessions found: {len(sessions)}")
        print(f"  Debugging: showing first 3 sessions...")
        for i, (ts, phases) in enumerate(list(sessions.items())[:3]):
            print(f"    Session {ts}:")
            for phase, images in phases.items():
                print(f"      {phase}: {len(images)} images")
        print(f"  Use --list to see all sessions")
        return 1

    # Select session
    if args.session:
        # Find requested session
        selected_session = None
        for session_ts, score, phases in complete_sessions:
            if session_ts == args.session:
                selected_session = (session_ts, score, phases)
                break

        if selected_session is None:
            print(f"\nX Session {args.session} not found!")
            print(f"  Use --list to see available sessions")
            return 1
    else:
        # Auto-select most complete session
        selected_session = complete_sessions[0]
        print(f"\nAuto-selected most complete session: {selected_session[0]} (score: {selected_session[1]}/5)")

    session_ts, score, phases = selected_session

    # Create output directory
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

    # Generate output filename
    if args.output:
        output_path = OUTPUT_DIR / args.output
    else:
        output_path = OUTPUT_DIR / f"calibration_workflow_{session_ts}.mp4"

    # Create video
    create_workflow_video(session_ts, phases, output_path, fps=args.fps)

    print(f"\n{'='*70}")
    print("COMPLETE")
    print(f"{'='*70}")
    print(f"Video saved to: {output_path}")
    print(f"\nYou can now use this video in your presentation to illustrate")
    print(f"the multi-phase embryo calibration workflow!")
    print()

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
