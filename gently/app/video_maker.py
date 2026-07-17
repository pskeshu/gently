"""
Video generation from timelapse volumes

Creates MP4 videos from max projections of timelapse volumes.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def discover_volumes(session_dir: Path, embryo_id: str | None = None) -> dict[str, list[Path]]:
    """
    Discover volume files in a session directory.

    Parameters
    ----------
    session_dir : Path
        Path to session's image directory (e.g., D:/Gently/images/2e8c5aa9/)
    embryo_id : str, optional
        Filter to specific embryo. If None, returns all embryos.

    Returns
    -------
    dict
        Mapping of embryo_id -> sorted list of volume paths
    """
    if not session_dir.exists():
        return {}

    # Find all tif files
    tif_files = list(session_dir.glob("*.tif")) + list(session_dir.glob("*.tiff"))

    # Group by embryo ID (filename format: embryo_1_20251210_095317.tif)
    embryo_volumes: dict[str, list[tuple[datetime, Path]]] = {}

    for f in tif_files:
        parts = f.stem.split("_")
        if len(parts) >= 3:
            # Extract embryo ID (e.g., "embryo_1")
            eid = f"{parts[0]}_{parts[1]}"

            # Extract timestamp (e.g., "20251210_095317")
            try:
                timestamp_str = f"{parts[2]}_{parts[3]}" if len(parts) >= 4 else parts[2]
                timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
            except (ValueError, IndexError):
                timestamp = datetime.fromtimestamp(f.stat().st_mtime)

            if embryo_id is None or eid == embryo_id:
                if eid not in embryo_volumes:
                    embryo_volumes[eid] = []
                embryo_volumes[eid].append((timestamp, f))

    # Sort by timestamp and return just paths
    result = {}
    for eid, volumes in embryo_volumes.items():
        volumes.sort(key=lambda x: x[0])
        result[eid] = [v[1] for v in volumes]

    return result


def make_max_projection(volume: np.ndarray) -> np.ndarray:
    """Create max projection from 3D volume."""
    if volume.ndim == 2:
        return volume
    elif volume.ndim == 3:
        return np.max(volume, axis=0)
    else:
        # Handle 4D+ by squeezing first
        volume = np.squeeze(volume)
        if volume.ndim == 3:
            return np.max(volume, axis=0)
        return volume


def normalize_for_video(
    image: np.ndarray, percentile_low: float = 1, percentile_high: float = 99.5
) -> np.ndarray:
    """Normalize image to 8-bit for video encoding."""
    from gently.core.imaging import normalize_to_uint8

    return normalize_to_uint8(
        image, method="percentile", p_low=percentile_low, p_high=percentile_high
    )


def add_timestamp_overlay(
    image: np.ndarray, timestamp: str, position: str = "top-left"
) -> np.ndarray:
    """Add timestamp text overlay to image."""
    import cv2

    # Convert to BGR for cv2 if grayscale
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Font settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 1
    color = (255, 255, 255)  # White

    # Get text size
    (text_width, text_height), baseline = cv2.getTextSize(timestamp, font, font_scale, thickness)

    # Position
    margin = 10
    if position == "top-left":
        x, y = margin, text_height + margin
    elif position == "top-right":
        x, y = image.shape[1] - text_width - margin, text_height + margin
    elif position == "bottom-left":
        x, y = margin, image.shape[0] - margin
    else:  # bottom-right
        x, y = image.shape[1] - text_width - margin, image.shape[0] - margin

    # Draw background rectangle for readability
    cv2.rectangle(
        image,
        (x - 2, y - text_height - 2),
        (x + text_width + 2, y + baseline + 2),
        (0, 0, 0),
        -1,
    )

    # Draw text
    cv2.putText(image, timestamp, (x, y), font, font_scale, color, thickness)

    return image


def create_timelapse_video(
    volume_paths: list[Path],
    output_path: Path,
    fps: int = 10,
    add_timestamps: bool = True,
    embryo_id: str | None = None,
    progress_callback=None,
) -> dict:
    """
    Create MP4 video from list of volume files.

    Parameters
    ----------
    volume_paths : list of Path
        Sorted list of volume TIFF files
    output_path : Path
        Output MP4 file path
    fps : int
        Frames per second
    add_timestamps : bool
        Whether to overlay timestamps on frames
    embryo_id : str
        Embryo ID for labeling
    progress_callback : callable
        Called with (current, total) for progress updates

    Returns
    -------
    dict
        Result info including output path, frame count, duration
    """
    import cv2
    import tifffile

    if not volume_paths:
        return {"error": "No volume files provided"}

    # Ensure output directory exists
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Initialize video writer
    writer = None
    frame_count = 0
    first_shape = None

    try:
        for i, vol_path in enumerate(volume_paths):
            if progress_callback:
                progress_callback(i, len(volume_paths))

            try:
                # Load volume
                volume = tifffile.imread(str(vol_path))

                # Create max projection
                projection = make_max_projection(volume)

                # Normalize to 8-bit
                frame = normalize_for_video(projection)

                # Add timestamp if requested
                if add_timestamps:
                    # Simple timepoint label: T=1, T=2, ...
                    timepoint = i + 1  # 1-indexed
                    timestamp = f"T={timepoint}"

                    if embryo_id:
                        timestamp = f"{embryo_id} | {timestamp}"

                    frame = add_timestamp_overlay(frame, timestamp)
                elif frame.ndim == 2:
                    # Convert grayscale to BGR for OpenCV video writer
                    frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

                # Ensure frame is BGR (OpenCV expects BGR)
                if frame.ndim == 3 and frame.shape[2] == 3:
                    # Check if it's RGB and convert to BGR
                    # add_timestamp_overlay returns BGR, so this should be fine
                    pass

                # Initialize writer with first frame dimensions
                if writer is None:
                    first_shape = frame.shape
                    height, width = frame.shape[:2]

                    # Try different codecs in order of preference
                    codecs = [
                        ("mp4v", ".mp4"),
                        ("avc1", ".mp4"),
                        ("XVID", ".avi"),
                        ("MJPG", ".avi"),
                    ]

                    for codec, ext in codecs:
                        fourcc = cv2.VideoWriter_fourcc(*codec)  # type: ignore[attr-defined]  # missing from cv2 stubs
                        test_path = output_path.with_suffix(ext)
                        writer = cv2.VideoWriter(
                            str(test_path), fourcc, fps, (width, height), isColor=True
                        )
                        if writer.isOpened():
                            output_path = test_path
                            logger.info(f"Using codec {codec} for video output")
                            break
                        writer.release()
                        writer = None

                    if writer is None:
                        return {"error": "Could not initialize video writer with any codec"}

                # Ensure consistent frame size
                if first_shape is not None and frame.shape != first_shape:
                    frame = cv2.resize(frame, (first_shape[1], first_shape[0]))

                writer.write(frame)
                frame_count += 1

            except Exception as e:
                logger.warning(f"Failed to process {vol_path.name}: {e}")
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
            "resolution": f"{first_shape[1]}x{first_shape[0]}" if first_shape else "unknown",
        }

    except Exception as e:
        if writer:
            try:
                writer.release()
            except Exception:
                pass
        return {"error": str(e)}


def make_session_videos(
    storage_path: Path,
    session_id: str,
    output_dir: Path | None = None,
    embryo_ids: list[str] | None = None,
    fps: int = 10,
    progress_callback=None,
) -> dict[str, Any]:
    """
    Create videos for all embryos in a session.

    Parameters
    ----------
    storage_path : Path
        Base storage path (e.g., D:/Gently)
    session_id : str
        Session ID
    output_dir : Path, optional
        Output directory. Defaults to storage_path/videos/session_id/
    embryo_ids : list of str, optional
        Specific embryos to process. None = all.
    fps : int
        Frames per second
    progress_callback : callable
        Called with (embryo_id, current, total) for progress

    Returns
    -------
    dict
        Mapping of embryo_id -> result dict
    """
    session_images_dir = storage_path / "images" / session_id

    if not session_images_dir.exists():
        return {"error": f"No images found for session {session_id}"}

    # Set output directory
    if output_dir is None:
        output_dir = storage_path / "videos" / session_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover volumes
    all_volumes = discover_volumes(session_images_dir)

    if not all_volumes:
        return {"error": "No volume files found"}

    # Filter to requested embryos
    if embryo_ids:
        all_volumes = {k: v for k, v in all_volumes.items() if k in embryo_ids}

    results = {}

    for embryo_id, volumes in all_volumes.items():
        logger.info(f"Creating video for {embryo_id} ({len(volumes)} frames)")

        output_path = output_dir / f"{embryo_id}_timelapse.mp4"

        def embryo_progress(current, total, _eid=embryo_id):
            if progress_callback:
                progress_callback(_eid, current, total)

        result = create_timelapse_video(
            volume_paths=volumes,
            output_path=output_path,
            fps=fps,
            add_timestamps=True,
            embryo_id=embryo_id,
            progress_callback=embryo_progress,
        )

        results[embryo_id] = result

    return results
