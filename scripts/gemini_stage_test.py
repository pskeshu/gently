#!/usr/bin/env python
"""
Test Gemini video understanding for C. elegans developmental stage classification.

This script:
1. Creates a timelapse video from embryo images
2. Uploads to Gemini and requests developmental stage analysis
3. Parses the response and creates an annotated video

Usage:
    python scripts/gemini_stage_test.py --session 3fb70aca --embryo embryo_1
    python scripts/gemini_stage_test.py --session 3fb70aca --embryo embryo_1 \
        --model gemini-3-pro-preview
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

# Lazy imports for optional dependencies
cv2: Any = None
tifffile: Any = None
genai: Any = None


def ensure_dependencies():
    """Ensure required dependencies are available."""
    global cv2, tifffile, genai

    try:
        import cv2 as _cv2

        cv2 = _cv2
    except ImportError:
        print("ERROR: opencv-python is required. Install with: pip install opencv-python")
        sys.exit(1)

    try:
        import tifffile as _tifffile

        tifffile = _tifffile
    except ImportError:
        print("ERROR: tifffile is required. Install with: pip install tifffile")
        sys.exit(1)

    try:
        from google import genai as _genai

        genai = _genai
    except ImportError:
        print("ERROR: google-genai is required. Install with: pip install google-genai")
        sys.exit(1)


# ============================================================================
# Video Creation (adapted from video_maker.py)
# ============================================================================


def discover_volumes(session_dir: Path, embryo_id: str | None = None) -> dict:
    """Discover volume files in a session directory."""
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


def make_max_projection(volume: np.ndarray) -> np.ndarray:
    """Create max projection from 3D volume (simple Z-projection)."""
    if volume.ndim == 2:
        return volume
    elif volume.ndim == 3:
        return np.max(volume, axis=0)
    else:
        volume = np.squeeze(volume)
        if volume.ndim == 3:
            return np.max(volume, axis=0)
        return volume


def create_dual_view_projection(volume: np.ndarray) -> np.ndarray:
    """
    Create dual-view projection (TOP | SIDE) from 3D volume.

    This matches the gently perception system approach:
    - Extract View A from diSPIM data
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
    from PIL import Image

    def normalize(img):
        img = img.astype(np.float32)
        p_low = np.percentile(img, 1)
        p_high = np.percentile(img, 99.5)
        img = np.clip(img, p_low, p_high)
        if p_high > p_low:
            img = (img - p_low) / (p_high - p_low) * 255
        return img.astype(np.uint8)

    # Handle 4D volumes: extract View A (first view)
    if volume.ndim == 4:
        view_a = volume[0]  # Extract View A -> (Z, Y, X)
    else:
        view_a = np.squeeze(volume)

    # Handle 2D input
    if view_a.ndim == 2:
        return normalize(view_a)

    # Handle 3D volumes
    if view_a.ndim == 3:
        z_depth, height, width = view_a.shape

        # Check if width contains dual-view data (views side-by-side in X)
        # diSPIM format has width roughly 4x height when dual-view
        if width > height * 2:
            # Extract View A (left half)
            view_a = view_a[:, :, : width // 2]

        # TOP projection: max along Z axis (looking down at embryo)
        top_proj = np.max(view_a, axis=0)  # Shape: (Y, X)

        # SIDE projection: max along Y axis (looking from side)
        side_proj = np.max(view_a, axis=1)  # Shape: (Z, X)

        top_norm = normalize(top_proj)
        side_norm = normalize(side_proj)

        # Rotate side view 90° clockwise so Z becomes horizontal
        # (Z, X) -> (X, Z) after rotation
        side_rotated = np.rot90(side_norm, k=-1)

        # Scale side view to match top view height
        target_height = top_norm.shape[0]
        # Make side view at least 150px wide for visibility
        new_width = max(150, int(side_rotated.shape[1] * target_height / side_rotated.shape[0]))
        side_pil = Image.fromarray(side_rotated).resize(
            (new_width, target_height), Image.Resampling.LANCZOS
        )
        side_scaled = np.array(side_pil)

        # Concatenate horizontally: TOP | separator | SIDE
        sep_width = 4
        separator = np.ones((target_height, sep_width), dtype=np.uint8) * 128
        combined = np.concatenate([top_norm, separator, side_scaled], axis=1)

        return combined

    # Fallback
    return normalize(view_a)


def normalize_for_video(
    image: np.ndarray, percentile_low: float = 1, percentile_high: float = 99.5
) -> np.ndarray:
    """Normalize image to 8-bit for video encoding."""
    p_low = np.percentile(image, percentile_low)
    p_high = np.percentile(image, percentile_high)

    image = np.clip(image, p_low, p_high)

    if p_high > p_low:
        image = ((image - p_low) / (p_high - p_low) * 255).astype(np.uint8)
    else:
        image = np.zeros_like(image, dtype=np.uint8)

    return image


def add_text_overlay(
    image: np.ndarray,
    text: str,
    position: str = "top-left",
    font_scale: float = 0.6,
    color=(255, 255, 255),
) -> np.ndarray:
    """Add text overlay to image."""
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 1

    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    margin = 10
    if position == "top-left":
        x, y = margin, text_height + margin
    elif position == "top-right":
        x, y = image.shape[1] - text_width - margin, text_height + margin
    elif position == "bottom-left":
        x, y = margin, image.shape[0] - margin
    else:  # bottom-right
        x, y = image.shape[1] - text_width - margin, image.shape[0] - margin

    # Draw background rectangle
    cv2.rectangle(
        image,
        (x - 2, y - text_height - 2),
        (x + text_width + 2, y + baseline + 2),
        (0, 0, 0),
        -1,
    )

    cv2.putText(image, text, (x, y), font, font_scale, color, thickness)

    return image


def create_embryo_video(
    session_id: str,
    embryo_id: str,
    storage_path: Path,
    fps: int = 10,
    output_path: Path | None = None,
) -> dict:
    """
    Create timelapse video from embryo images.

    Returns dict with video info or error.
    """
    session_dir = storage_path / "images" / session_id

    if not session_dir.exists():
        return {"error": f"Session directory not found: {session_dir}"}

    # Discover volumes
    volumes = discover_volumes(session_dir, embryo_id)

    if not volumes or embryo_id not in volumes:
        return {"error": f"No volumes found for {embryo_id} in {session_dir}"}

    volume_paths = volumes[embryo_id]
    print(f"Found {len(volume_paths)} volumes for {embryo_id}")

    # Set output path
    if output_path is None:
        output_dir = storage_path / "videos" / session_id
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{embryo_id}_timelapse.mp4"

    # Create video
    writer = None
    frame_count = 0
    first_shape: Any = None

    try:
        for i, vol_path in enumerate(volume_paths):
            if (i + 1) % 10 == 0 or i == 0:
                print(f"  Processing frame {i + 1}/{len(volume_paths)}...")

            try:
                volume = tifffile.imread(str(vol_path))
                # Create dual-view projection (TOP | SIDE) like gently perception
                frame = create_dual_view_projection(volume)

                # Add timestamp
                parts = vol_path.stem.split("_")
                if len(parts) >= 4:
                    ts_str = f"{parts[2]}_{parts[3]}"
                    try:
                        ts = datetime.strptime(ts_str, "%Y%m%d_%H%M%S")
                        timestamp = ts.strftime("%H:%M:%S")
                    except ValueError:
                        timestamp = f"t={i}"
                else:
                    timestamp = f"t={i}"

                frame = add_text_overlay(frame, f"{embryo_id} | {timestamp}", "top-left")

                # Add view labels (TOP on left side, SIDE on right side)
                # The frame is already BGR from add_text_overlay
                height, width = frame.shape[:2]
                # Approximate where the separator is (TOP view takes most of the width)
                # TOP label near center-left, SIDE label near right
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(frame, "TOP", (10, height - 10), font, 0.5, (200, 200, 200), 1)
                # SIDE label - estimate separator position based on aspect ratio
                side_start = int(width * 0.75)  # Approximate
                cv2.putText(
                    frame,
                    "SIDE",
                    (side_start, height - 10),
                    font,
                    0.5,
                    (200, 200, 200),
                    1,
                )

                # Initialize writer
                if writer is None:
                    first_shape = frame.shape
                    height, width = frame.shape[:2]

                    codecs = [("mp4v", ".mp4"), ("avc1", ".mp4"), ("XVID", ".avi")]

                    for codec, ext in codecs:
                        fourcc = cv2.VideoWriter_fourcc(*codec)
                        test_path = output_path.with_suffix(ext)
                        writer = cv2.VideoWriter(
                            str(test_path), fourcc, fps, (width, height), isColor=True
                        )
                        if writer.isOpened():
                            output_path = test_path
                            print(f"  Using codec: {codec}")
                            break
                        writer.release()
                        writer = None

                    if writer is None:
                        return {"error": "Could not initialize video writer"}

                if frame.shape != first_shape:
                    frame = cv2.resize(frame, (first_shape[1], first_shape[0]))

                writer.write(frame)
                frame_count += 1

            except Exception as e:
                print(f"  Warning: Failed to process {vol_path.name}: {e}")
                continue

        if writer:
            writer.release()

        if frame_count == 0:
            return {"error": "No frames could be processed"}

        duration = frame_count / fps

        return {
            "success": True,
            "output_path": str(output_path),
            "frame_count": frame_count,
            "duration_seconds": duration,
            "fps": fps,
        }

    except Exception as e:
        if writer:
            writer.release()
        return {"error": str(e)}


# ============================================================================
# Gemini API Integration
# ============================================================================


def build_stage_classification_prompt(frame_count: int, fps: int, duration_seconds: float) -> str:
    """Build the classification prompt with video-specific information."""
    return f"""You are analyzing a time-lapse microscopy video of a C. elegans embryo developing.

VIDEO INFORMATION:
- This video has {frame_count} frames at {fps} FPS
- Total video duration: {duration_seconds:.1f} seconds (playback time)
- Each frame represents ~3 minutes of real developmental time
- The video compresses ~5 hours of development into {duration_seconds:.1f} seconds

IMAGE LAYOUT:
Each frame shows TWO VIEWS side-by-side:
- LEFT (TOP view): Looking down at the embryo from above (max projection along Z-axis)
- RIGHT (SIDE view): Looking at the embryo from the side (max projection along Y-axis)
Both views together give 3D morphological information.

DEVELOPMENTAL STAGES (in chronological order):

1. EARLY (gastrulation):
   - Oval/elliptical shape, relatively uniform cellular mass
   - Grainy texture showing many individual cells (~100+ cell stage)
   - No elongation or asymmetry
   - Side view shows compact, rounded blob

2. BEAN:
   - Slightly more elongated than early stage
   - Beginning asymmetry - one end slightly narrower
   - Subtle kidney/bean shape forming
   - Still mostly a compact cellular mass

3. COMMA:
   - Clear elongation - distinctly longer shape
   - Pronounced bend/curve forming C-shape
   - Head and tail regions becoming distinguishable
   - Body axis clearly established

4. 1.5FOLD:
   - Embryo has elongated significantly
   - Body folded back on itself ~1.5 times its width
   - Worm shape becoming apparent
   - Can see the fold/turn in the body

5. 2FOLD:
   - Compact "pretzel" shape
   - Embryo fully folded back on itself (~2x)
   - Very compact within eggshell
   - Clear worm morphology

6. 3FOLD:
   - Tightly coiled within eggshell
   - Three body segments visible in the coil
   - Maximum compaction before movement begins
   - May see occasional twitching

7. HATCHING:
   - Embryo actively moving/thrashing
   - Breach in eggshell visible
   - Worm partially emerging
   - Active escape behavior

8. HATCHED:
   - Worm fully outside eggshell
   - Free-moving larva visible
   - Empty or nearly empty eggshell remains

TASK:
Watch the ENTIRE video and identify each developmental stage. For each stage you observe:

1. Report the FRAME NUMBER (0 to {frame_count - 1}) when that stage BEGINS
2. Identify the stage name (use lowercase: early, bean, comma, 1.5fold, 2fold, 3fold,
   hatching, hatched)
3. Describe the specific morphological features you observe that indicate this stage
4. Provide your confidence level (HIGH, MEDIUM, or LOW)

Return your analysis as JSON with this exact structure:
{{
    "embryo_analysis": [
        {{
            "frame": 0,
            "stage": "early",
            "features": "Description of what you see",
            "confidence": "HIGH"
        }},
        {{
            "frame": 25,
            "stage": "bean",
            "features": "Description of what you see",
            "confidence": "HIGH"
        }}
    ],
    "overall_development": "Brief summary of the complete developmental progression observed",
    "final_stage": "The stage at the end of the video",
    "video_quality": "Assessment of image quality for classification (good/moderate/poor)",
    "notes": "Any additional observations or concerns"
}}

IMPORTANT: Use FRAME NUMBERS (0 to {frame_count - 1}), not timestamps!
Be precise with frame numbers. Watch the ENTIRE video to capture all stage transitions."""


def analyze_with_gemini(
    video_path: str | Path,
    model: str = "gemini-3-pro-preview",
    api_key: str | None = None,
    frame_count: int = 100,
    fps: int = 10,
    duration_seconds: float = 10.0,
) -> dict:
    """
    Upload video to Gemini and analyze for developmental stages.

    Returns dict with analysis results or error.
    """
    # Check multiple possible environment variable names
    if not api_key:
        api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        api_key = os.environ.get("gemini_API_Key")
    if not api_key:
        api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        return {
            "error": "No API key found. Set GEMINI_API_KEY environment variable or pass --api-key"
        }

    print(f"\nInitializing Gemini client with model: {model}")

    try:
        client = genai.Client(api_key=api_key)
    except Exception as e:
        return {"error": f"Failed to initialize Gemini client: {e}"}

    # Upload video
    print(f"Uploading video: {video_path}")
    video_path = Path(video_path)

    if not video_path.exists():
        return {"error": f"Video file not found: {video_path}"}

    file_size_mb = video_path.stat().st_size / (1024 * 1024)
    print(f"  File size: {file_size_mb:.1f} MB")

    try:
        video_file = client.files.upload(file=str(video_path))
        print(f"  Upload complete. File name: {video_file.name}")
    except Exception as e:
        return {"error": f"Failed to upload video: {e}"}

    # Wait for processing
    print("Waiting for video processing...")
    max_wait = 300  # 5 minutes max
    wait_time = 0

    while video_file.state.name == "PROCESSING" and wait_time < max_wait:
        time.sleep(5)
        wait_time += 5
        video_file = client.files.get(name=video_file.name)
        print(f"  Processing... ({wait_time}s)")

    if video_file.state.name == "PROCESSING":
        return {"error": "Video processing timed out"}

    if video_file.state.name != "ACTIVE":
        return {"error": f"Video processing failed. State: {video_file.state.name}"}

    print("  Video ready for analysis")

    # Analyze with Gemini
    print(f"\nSending to Gemini {model} for analysis...")
    start_time = time.time()

    try:
        response = client.models.generate_content(
            model=model,
            contents=[
                video_file,
                build_stage_classification_prompt(frame_count, fps, duration_seconds),
            ],
        )

        elapsed = time.time() - start_time
        print(f"  Response received in {elapsed:.1f}s")

        return {
            "success": True,
            "response_text": response.text,
            "model": model,
            "elapsed_seconds": elapsed,
        }

    except Exception as e:
        return {"error": f"Gemini API error: {e}"}


def parse_stage_response(response_text: str) -> dict:
    """
    Parse Gemini's response to extract stage annotations.

    Returns dict with parsed data or error.
    """
    # Try to extract JSON from response
    # Handle cases where response has markdown code blocks
    json_match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", response_text)

    if json_match:
        json_str = json_match.group(1)
    else:
        # Try to find raw JSON
        json_match = re.search(r"\{[\s\S]*\}", response_text)
        if json_match:
            json_str = json_match.group(0)
        else:
            return {
                "error": "Could not find JSON in response",
                "raw_response": response_text,
            }

    try:
        data = json.loads(json_str)
        return {"success": True, "data": data}
    except json.JSONDecodeError as e:
        return {"error": f"Failed to parse JSON: {e}", "raw_response": response_text}


def timestamp_to_seconds(timestamp: str) -> float:
    """Convert MM:SS or HH:MM:SS timestamp to seconds."""
    parts = timestamp.split(":")
    if len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    elif len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    return 0.0


# ============================================================================
# Annotated Video Creation
# ============================================================================


def create_annotated_video(
    source_video: str, stage_data: dict, output_path: str | Path, fps: int = 10
) -> dict:
    """
    Create annotated video with stage labels overlaid.

    Args:
        source_video: Path to original video
        stage_data: Parsed stage analysis data
        output_path: Path for annotated output video
        fps: Frames per second of source video

    Returns dict with result info or error.
    """
    if "embryo_analysis" not in stage_data:
        return {"error": "No embryo_analysis in stage data"}

    # Build stage timeline
    stages = stage_data["embryo_analysis"]
    stage_timeline = []

    for i, stage_info in enumerate(stages):
        # Support both frame numbers and timestamps for backwards compatibility
        if "frame" in stage_info:
            start_frame = int(stage_info["frame"])
        elif "timestamp" in stage_info:
            start_seconds = timestamp_to_seconds(stage_info["timestamp"])
            start_frame = int(start_seconds * fps)
        else:
            start_frame = 0

        # End frame is start of next stage or end of video
        if i + 1 < len(stages):
            next_stage = stages[i + 1]
            if "frame" in next_stage:
                end_frame: float = int(next_stage["frame"])
            elif "timestamp" in next_stage:
                end_seconds = timestamp_to_seconds(next_stage["timestamp"])
                end_frame = int(end_seconds * fps)
            else:
                end_frame = float("inf")
        else:
            end_frame = float("inf")

        stage_timeline.append(
            {
                "start_frame": start_frame,
                "end_frame": end_frame,
                "stage": stage_info["stage"],
                "confidence": stage_info.get("confidence", "?"),
                "features": stage_info.get("features", ""),
            }
        )

    print("\nCreating annotated video...")
    print(f"  Source: {source_video}")
    print(f"  Output: {output_path}")
    print(f"  Stages: {len(stage_timeline)}")

    # Debug: print stage timeline
    print("  Stage timeline:")
    for st in stage_timeline:
        print(f"    Frame {st['start_frame']}-{st['end_frame']}: {st['stage']}")

    # Open source video
    cap = cv2.VideoCapture(source_video)
    if not cap.isOpened():
        return {"error": f"Could not open source video: {source_video}"}

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    source_fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"  Resolution: {width}x{height}")
    print(f"  Total frames: {total_frames}")
    print(f"  FPS: {source_fps}")

    # Create output video writer
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, source_fps, (width, height), isColor=True)

    if not writer.isOpened():
        cap.release()
        return {"error": "Could not create output video writer"}

    # Process frames
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Find current stage
        current_stage = "unknown"
        current_confidence = "?"

        for stage_info in stage_timeline:
            if stage_info["start_frame"] <= frame_idx < stage_info["end_frame"]:
                current_stage = stage_info["stage"]
                current_confidence = stage_info["confidence"]
                break

        # Add stage overlay at top-right
        stage_text = f"Stage: {current_stage.upper()} ({current_confidence})"

        # Color based on confidence
        if current_confidence == "HIGH":
            color = (0, 255, 0)  # Green
        elif current_confidence == "MEDIUM":
            color = (0, 255, 255)  # Yellow
        else:
            color = (0, 165, 255)  # Orange

        frame = add_text_overlay(frame, stage_text, "top-right", font_scale=0.7, color=color)

        writer.write(frame)
        frame_idx += 1

        if frame_idx % 20 == 0:
            print(f"  Processed {frame_idx}/{total_frames} frames...")

    cap.release()
    writer.release()

    print(f"  Annotated video saved: {output_path}")

    return {
        "success": True,
        "output_path": str(output_path),
        "frame_count": frame_idx,
        "stages_annotated": len(stage_timeline),
    }


# ============================================================================
# Main
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Test Gemini video understanding for C. elegans developmental stage classification"
        )
    )
    parser.add_argument("--session", default="3fb70aca", help="Session ID")
    parser.add_argument("--embryo", default="embryo_1", help="Embryo ID")
    parser.add_argument(
        "--model",
        default="gemini-2.5-flash",
        choices=[
            "gemini-2.5-pro",
            "gemini-2.5-flash",
            "gemini-3-pro-preview",
            "gemini-3-flash-preview",
        ],
        help="Gemini model to use",
    )
    parser.add_argument("--storage", default="D:/Gently", help="Storage path")
    parser.add_argument("--fps", type=int, default=10, help="Video frames per second")
    parser.add_argument(
        "--skip-video-creation",
        action="store_true",
        help="Skip video creation if video already exists",
    )
    parser.add_argument(
        "--video-only",
        action="store_true",
        help="Only create video, skip Gemini analysis",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="Gemini API key (or set GEMINI_API_KEY env var)",
    )

    args = parser.parse_args()

    # Ensure dependencies
    ensure_dependencies()

    storage_path = Path(args.storage)

    print("=" * 60)
    print("Gemini Developmental Stage Classification Test")
    print("=" * 60)
    print(f"Session: {args.session}")
    print(f"Embryo: {args.embryo}")
    print(f"Model: {args.model}")
    print(f"Storage: {storage_path}")
    print("=" * 60)

    # Step 1: Create video
    video_dir = storage_path / "videos" / args.session
    video_path = video_dir / f"{args.embryo}_timelapse.mp4"

    # Video info for Gemini prompt
    video_frame_count = 100  # default
    video_duration = 10.0  # default

    if args.skip_video_creation and video_path.exists():
        print(f"\nUsing existing video: {video_path}")
        # Get video info from file
        cap = cv2.VideoCapture(str(video_path))
        video_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_duration = video_frame_count / args.fps
        cap.release()
        print(f"  Frames: {video_frame_count}, Duration: {video_duration:.1f}s")
    else:
        print("\n[Step 1] Creating timelapse video...")
        video_result = create_embryo_video(args.session, args.embryo, storage_path, args.fps)

        if "error" in video_result:
            print(f"ERROR: {video_result['error']}")
            return 1

        video_path = Path(video_result["output_path"])
        video_frame_count = video_result["frame_count"]
        video_duration = video_result["duration_seconds"]
        print("\nVideo created successfully!")
        print(f"  Path: {video_path}")
        print(f"  Frames: {video_frame_count}")
        print(f"  Duration: {video_duration:.1f}s")

    if args.video_only:
        print("\n--video-only flag set, skipping Gemini analysis")
        return 0

    # Step 2: Analyze with Gemini
    print("\n[Step 2] Analyzing with Gemini...")
    result = analyze_with_gemini(
        str(video_path),
        args.model,
        args.api_key,
        frame_count=video_frame_count,
        fps=args.fps,
        duration_seconds=video_duration,
    )

    if "error" in result:
        print(f"ERROR: {result['error']}")
        return 1

    print(f"\n{'=' * 60}")
    print("GEMINI RESPONSE")
    print("=" * 60)
    print(result["response_text"])
    print("=" * 60)

    # Save raw response
    response_file = video_dir / f"{args.embryo}_gemini_response.json"
    with open(response_file, "w") as f:
        json.dump(
            {
                "model": args.model,
                "session": args.session,
                "embryo": args.embryo,
                "response": result["response_text"],
                "elapsed_seconds": result["elapsed_seconds"],
                "timestamp": datetime.now().isoformat(),
            },
            f,
            indent=2,
        )
    print(f"\nResponse saved to: {response_file}")

    # Step 3: Parse response
    print("\n[Step 3] Parsing response...")
    parsed = parse_stage_response(result["response_text"])

    if "error" in parsed:
        print(f"WARNING: {parsed['error']}")
        print("Raw response saved but cannot create annotated video")
        return 0

    stage_data = parsed["data"]

    # Print summary
    print("\nDetected Stages:")
    for stage in stage_data.get("embryo_analysis", []):
        # Support both frame numbers and timestamps
        if "frame" in stage:
            loc = f"Frame {stage['frame']}"
        else:
            loc = stage.get("timestamp", "?")
        print(f"  {loc} - {stage['stage']} ({stage.get('confidence', '?')})")

    print(f"\nFinal stage: {stage_data.get('final_stage', 'unknown')}")
    print(f"Overall: {stage_data.get('overall_development', 'N/A')}")

    # Save parsed data
    parsed_file = video_dir / f"{args.embryo}_stages_parsed.json"
    with open(parsed_file, "w") as f:
        json.dump(stage_data, f, indent=2)
    print(f"\nParsed stages saved to: {parsed_file}")

    # Step 4: Create annotated video
    print("\n[Step 4] Creating annotated video...")
    annotated_path = video_dir / f"{args.embryo}_timelapse_gemini_annotated.mp4"

    result = create_annotated_video(str(video_path), stage_data, str(annotated_path), args.fps)

    if "error" in result:
        print(f"ERROR: {result['error']}")
        return 1

    print(f"\n{'=' * 60}")
    print("COMPLETE!")
    print("=" * 60)
    print(f"Original video: {video_path}")
    print(f"Annotated video: {annotated_path}")
    print(f"Response JSON: {response_file}")
    print(f"Parsed stages: {parsed_file}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
