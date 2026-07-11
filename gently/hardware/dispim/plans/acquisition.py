"""
Core Bluesky plans for DiSPIM hardware control.

These plans run on the DEVICE LAYER SERVER (no Claude API access).
They are "dumb" — they follow fixed sequences without AI guidance.

For AI-guided calibration, see calibration_plans.py (runs on agent side).
For multi-embryo orchestration, see multi_embryo_plans.py.
For agent planning context (campaigns, plan items), see context/_plans.py.

Plan hierarchy:
    plans.py                  → Device-side Bluesky plans (focus, acquire, calibrate)
    calibration_plans.py      → Agent-side Vision-guided calibration
    multi_embryo_plans.py     → Loops calibration_plans over multiple embryos
    context/_plans.py         → Agent memory (PlanItem CRUD, unrelated to hardware)

Organization:
- Utility Plans: Move, read, capture (simple device operations)
- Focus Analysis: FFT bandpass scoring, embryo ROI detection
- Calibration Plans: Piezo-galvo calibration, focus sweeps
- Embryo Detection: Interactive marking, automated centering
- Volume Acquisition: Single volumes, time-lapse series
- Multi-Embryo Workflows: Full end-to-end acquisition

All plans use standard Bluesky plan stubs:
    - bps.mv(device, value)      # Move and wait
    - bps.trigger_and_read([dev]) # Acquire data
    - bps.open_run() / bps.close_run() # Data collection boundaries
"""

import logging
import time
from collections.abc import Generator
from typing import Any

import bluesky.plan_stubs as bps
import bluesky.plans as bp
import bluesky.preprocessors as bpp
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


# =======================
# FOCUS ANALYSIS UTILITIES
# =======================


def compute_fft_bandpass_score(
    image: np.ndarray,
    lower_cutoff: float = 0.025,
    upper_cutoff: float = 0.14,
    roi: tuple[int, int, int, int] | None = None,
) -> float:
    """
    Compute FFT bandpass focus score (ASI diSPIM OughtaFocus algorithm).

    Analyzes the power spectrum of spatial frequencies. Well-focused images
    have more high-frequency content (sharp edges) than defocused images.

    The default frequency band (2.5% - 14% of maximum) was empirically
    determined by Bill Mohler (UConn) for light sheet microscopy.

    Parameters
    ----------
    image : np.ndarray
        2D grayscale image
    lower_cutoff : float
        Lower frequency cutoff as fraction of max (default: 0.025 = 2.5%)
    upper_cutoff : float
        Upper frequency cutoff as fraction of max (default: 0.14 = 14%)
    roi : Tuple[int, int, int, int], optional
        ROI as (y_min, y_max, x_min, x_max) to crop before analysis

    Returns
    -------
    float
        Mean power in frequency band (higher = better focus)
    """
    # Apply ROI cropping if provided
    if roi is not None:
        y_min, y_max, x_min, x_max = roi
        image = image[y_min:y_max, x_min:x_max]

    # Ensure float for FFT
    img_float = image.astype(np.float64)

    # Compute 2D FFT and power spectrum
    fft = np.fft.fft2(img_float)
    fft_shifted = np.fft.fftshift(fft)  # Move DC to center
    power_spectrum = np.abs(fft_shifted) ** 2

    # Create frequency grid for bandpass mask
    h, w = image.shape
    cy, cx = h // 2, w // 2  # Center coordinates

    # Create distance map from center
    y, x = np.ogrid[:h, :w]
    distance_from_center = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

    # Maximum frequency (corner distance)
    max_freq = np.sqrt(cx**2 + cy**2)

    # Normalized distance (0 at DC, 1 at corners)
    normalized_distance = distance_from_center / max_freq

    # Create bandpass mask
    bandpass_mask = (normalized_distance >= lower_cutoff) & (normalized_distance <= upper_cutoff)

    # Compute mean power in band
    masked_power = power_spectrum * bandpass_mask

    if np.sum(bandpass_mask) > 0:
        mean_power = np.sum(masked_power) / np.sum(bandpass_mask)
    else:
        mean_power = 0.0

    return mean_power


def detect_embryo_roi(
    image: np.ndarray, margin_fraction: float = 0.1, min_threshold_ratio: float = 1.15
) -> tuple[int, int, int, int]:
    """
    Detect embryo region and return bounding box ROI for focus analysis.

    Uses adaptive thresholding to find embryo boundary and creates a bounding
    box with specified margin. This ROI excludes empty background for more
    accurate focus scoring.

    Parameters
    ----------
    image : np.ndarray
        2D grayscale image (single camera view)
    margin_fraction : float
        Fractional margin around embryo (default: 0.1 = 10%)
    min_threshold_ratio : float
        Minimum brightness ratio embryo:background (default: 1.15)

    Returns
    -------
    Tuple[int, int, int, int]
        ROI as (y_min, y_max, x_min, x_max) in pixels
    """
    h, w = image.shape

    # Calculate background from edge regions
    edge_margin = min(50, h // 10, w // 10)
    edge_pixels = np.concatenate(
        [
            image[:edge_margin, :].flatten(),
            image[-edge_margin:, :].flatten(),
            image[:, :edge_margin].flatten(),
            image[:, -edge_margin:].flatten(),
        ]
    )

    background_level = np.median(edge_pixels)
    threshold = background_level * min_threshold_ratio

    # Create binary mask
    mask = image > threshold

    # Find bounding box of masked region
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)

    if not np.any(rows) or not np.any(cols):
        # No embryo detected, return full image
        return (0, h, 0, w)

    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]

    # Add margin
    y_margin = int((y_max - y_min) * margin_fraction)
    x_margin = int((x_max - x_min) * margin_fraction)

    y_min = max(0, y_min - y_margin)
    y_max = min(h, y_max + y_margin)
    x_min = max(0, x_min - x_margin)
    x_max = min(w, x_max + x_margin)

    return (y_min, y_max, x_min, x_max)


def select_best_camera_view(image: np.ndarray) -> np.ndarray:
    """
    Select the better view from dual-view diSPIM image.

    DiSPIM captures two views side-by-side. Returns the brighter view
    (better signal) by splitting at midpoint and comparing mean intensity.

    Note
    ----
    This function is for hardware-stitched images from a single camera device.
    If using DiSPIMDualCamera, it already returns both views separately as
    'image_a' and 'image_b', so you can choose which to use directly.

    This is useful when:
    - Using DiSPIMCamera (not DiSPIMDualCamera) during calibration
    - Hardware returns stitched images
    - You need to analyze just one view (e.g., for focus scoring)

    Parameters
    ----------
    image : np.ndarray
        Full diSPIM image (both views side-by-side)

    Returns
    -------
    np.ndarray
        Single view (left or right half) with higher intensity
    """
    h, w = image.shape
    mid_x = w // 2

    left_view = image[:, :mid_x]
    right_view = image[:, mid_x:]

    left_intensity = np.mean(left_view)
    right_intensity = np.mean(right_view)

    if left_intensity >= right_intensity:
        return left_view
    else:
        return right_view


# ================================
# STAGE POSITIONING PLAN STUBS
# ================================


def get_stage_position_plan(xy_stage):
    """
    Read XY stage position within a plan.

    This is a plan stub that yields Bluesky messages to read the stage position.
    Use this instead of direct device access within plans to maintain proper
    message flow and enable databroker integration.

    Parameters
    ----------
    xy_stage : DiSPIMXYStage
        XY stage device

    Yields
    ------
    Bluesky messages
        Messages to read stage position

    Returns
    -------
    np.ndarray
        Current stage position as [x, y] in micrometers

    Examples
    --------
    >>> current_pos = yield from get_stage_position_plan(xy_stage)
    >>> print(f"Stage at: {current_pos}")
    """
    # bps.rd returns the device's single reading value directly (here the
    # [x, y] position array), not a {name: {value}} dict.
    return (yield from bps.rd(xy_stage))


def move_to_pixel_plan(xy_stage, bottom_camera, pixel_x: float, pixel_y: float):
    """
    Move stage to center on pixel coordinates from bottom camera image.

    This plan encapsulates the complete workflow of converting pixel coordinates
    to stage coordinates (with X-axis inversion) and moving the stage to center
    the feature. Uses the coordinate utilities from gently.coordinates.

    Parameters
    ----------
    xy_stage : DiSPIMXYStage
        XY stage device
    bottom_camera : DiSPIMBottomCamera
        Bottom camera device (provides effective_pixel_size)
    pixel_x : float
        Target pixel X coordinate
    pixel_y : float
        Target pixel Y coordinate

    Yields
    ------
    Bluesky messages
        Messages to read position and move stage

    Returns
    -------
    np.ndarray
        Final stage position as [x, y] in micrometers

    Examples
    --------
    >>> # User clicked at pixel (1024, 1027)
    >>> final_pos = yield from move_to_pixel_plan(
    ...     xy_stage, bottom_camera, 1024, 1027
    ... )
    """
    from gently.core.coordinates import pixel_displacement_to_stage_movement

    # Get current position
    current_pos = yield from get_stage_position_plan(xy_stage)

    # Get image dimensions from camera (assuming square or known dimensions)
    # For now, assume 2048x2048 based on typical PCO camera
    image_center_x = 2048 / 2.0
    image_center_y = 2048 / 2.0

    # Calculate pixel offset from image center
    pixel_offset_x = pixel_x - image_center_x
    pixel_offset_y = pixel_y - image_center_y

    # Convert pixel displacement to stage movement
    dx, dy = pixel_displacement_to_stage_movement(
        pixel_offset_x, pixel_offset_y, bottom_camera.effective_pixel_size
    )

    # Calculate target position
    target_pos = current_pos + np.array([dx, dy])

    # Move to target
    yield from bps.mov(xy_stage, target_pos)

    return target_pos


def center_on_feature_plan(xy_stage, bottom_camera, image: np.ndarray, feature_detector_func):
    """
    Complete workflow: detect feature in image and center stage on it.

    This plan combines feature detection with stage positioning. The feature
    detector function should return pixel coordinates of the detected feature.

    Parameters
    ----------
    xy_stage : DiSPIMXYStage
        XY stage device
    bottom_camera : DiSPIMBottomCamera
        Bottom camera device
    image : np.ndarray
        Image containing feature to detect
    feature_detector_func : callable
        Function that takes image and returns (x, y) pixel coordinates

    Yields
    ------
    Bluesky messages
        Messages to move stage

    Returns
    -------
    Dict
        Dictionary with 'feature_pos' (pixel coords) and 'stage_pos' (final position)

    Examples
    --------
    >>> def detect_brightest_spot(image):
    ...     y, x = np.unravel_index(np.argmax(image), image.shape)
    ...     return (x, y)
    >>>
    >>> result = yield from center_on_feature_plan(
    ...     xy_stage, bottom_camera, image, detect_brightest_spot
    ... )
    """
    # Detect feature in image
    feature_x, feature_y = feature_detector_func(image)

    # Move to center the feature
    final_pos = yield from move_to_pixel_plan(xy_stage, bottom_camera, feature_x, feature_y)

    return {"feature_pos": (feature_x, feature_y), "stage_pos": final_pos}


# =======================
# CALIBRATION PLANS
# =======================


def focus_sweep_plan(
    lightsheet_snap,
    galvo_positions: list[float],
    roi_detection: bool = True,
    metadata: dict | None = None,
):
    """
    Perform focus sweep by moving galvo Y-offset and analyzing image quality.

    This plan:
    1. Iterates through galvo Y positions (light sheet vertical positions)
    2. Captures image at each position using light sheet snap device
    3. Analyzes focus quality using FFT bandpass scoring
    4. Records images and scores

    Parameters
    ----------
    lightsheet_snap : DiSPIMLightSheetSnap
        Light sheet snapshot device (scanner + camera)
    galvo_positions : List[float]
        List of galvo Y-axis offsets in degrees to sweep through
    roi_detection : bool
        Whether to detect embryo ROI for focus analysis (default: True)
    metadata : Dict, optional
        Additional metadata to include in run

    Yields
    ------
    msg : Msg
        Bluesky plan messages

    Returns
    -------
    Dict
        Results dictionary with positions, images, scores, and best focus info
    """
    # Prepare storage for results
    results: dict[str, Any] = {
        "galvo_positions": galvo_positions,
        "images": [],
        "focus_scores": [],
        "rois": [],
        "timestamp": time.time(),
    }

    # Open data collection run
    _md = {"plan_name": "focus_sweep"}
    if metadata:
        _md.update(metadata)

    @bpp.run_decorator(md=_md)
    def inner():
        logger.info("=" * 70)
        logger.info("FOCUS SWEEP: %d positions", len(galvo_positions))
        logger.info(
            "Galvo Y range: [%.4f, %.4f] deg",
            min(galvo_positions),
            max(galvo_positions),
        )
        logger.info("ROI detection: %s", "enabled" if roi_detection else "disabled")
        logger.info("=" * 70)

        for idx, galvo_y in enumerate(galvo_positions):
            logger.info("[%d/%d] Galvo Y = %.4f deg", idx + 1, len(galvo_positions), galvo_y)

            # Set galvo Y position
            lightsheet_snap.set_y_position(galvo_y)

            # Capture image
            yield from bps.trigger_and_read([lightsheet_snap])

            # Get captured image from device
            image_data = lightsheet_snap.read()[lightsheet_snap.camera.name]["value"]

            # Select best view if dual-camera
            if image_data.shape[1] > image_data.shape[0] * 2:  # Heuristic for stitched image
                image = select_best_camera_view(image_data)
            else:
                image = image_data

            # Detect embryo ROI if enabled
            if roi_detection:
                roi = detect_embryo_roi(image)
            else:
                roi = None

            # Compute focus score
            focus_score = compute_fft_bandpass_score(image, roi=roi)

            # Store results
            results["images"].append(image)
            results["focus_scores"].append(focus_score)
            results["rois"].append(roi)

            logger.debug("Score: %.2e", focus_score)

    yield from inner()

    # Analyze results
    scores = np.array(results["focus_scores"])
    best_idx = np.argmax(scores)
    results["best_position"] = galvo_positions[best_idx]
    results["best_score"] = scores[best_idx]
    results["best_image"] = results["images"][best_idx]

    logger.info("BEST FOCUS: Position %d/%d", best_idx + 1, len(galvo_positions))
    logger.info("Galvo Y = %.4f deg", results["best_position"])
    logger.info("Score = %.2e", results["best_score"])

    return results


def calibrate_piezo_galvo_plan(
    lightsheet_snap,
    piezo_positions: list[float],
    initial_galvo_position: float = 0.0,
    search_range_deg: float = 0.02,
    n_sweep_points: int = 21,
    metadata: dict | None = None,
):
    """
    Calibrate piezo-galvo synchronization using 2-point linear fit.

    This plan performs the core piezo-galvo calibration:
    1. For each piezo position, perform focus sweep to find best galvo Y
    2. Build (piezo_position, galvo_position) pairs
    3. Fit linear relationship: galvo_y = slope * piezo_z + offset

    Parameters
    ----------
    lightsheet_snap : DiSPIMLightSheetSnap
        Light sheet snapshot device
    piezo_positions : List[float]
        Two or more piezo Z positions to calibrate (micrometers)
    initial_galvo_position : float
        Starting galvo Y position for first sweep (degrees)
    search_range_deg : float
        Search range around initial position (default: ±0.02 deg)
    n_sweep_points : int
        Number of positions in each focus sweep (default: 21)
    metadata : Dict, optional
        Additional metadata

    Yields
    ------
    msg : Msg
        Bluesky plan messages

    Returns
    -------
    Dict
        Calibration results with slope, offset, and fit quality
    """
    results = {
        "piezo_positions": piezo_positions,
        "galvo_positions": [],
        "focus_scores": [],
        "sweep_results": [],
        "timestamp": time.time(),
    }

    _md = {"plan_name": "calibrate_piezo_galvo"}
    if metadata:
        _md.update(metadata)

    @bpp.run_decorator(md=_md)
    def inner():
        logger.info("=" * 70)
        logger.info("PIEZO-GALVO CALIBRATION")
        logger.info("Piezo positions: %d", len(piezo_positions))
        logger.info("Initial galvo: %.4f deg", initial_galvo_position)
        logger.info("Search range: +/-%.4f deg", search_range_deg)
        logger.info("=" * 70)

        current_galvo_guess = initial_galvo_position

        for piezo_idx, piezo_z in enumerate(piezo_positions):
            logger.info(
                "[PIEZO %d/%d] Z = %.2f um",
                piezo_idx + 1,
                len(piezo_positions),
                piezo_z,
            )

            # Generate galvo sweep positions around current guess
            galvo_sweep = np.linspace(
                current_galvo_guess - search_range_deg,
                current_galvo_guess + search_range_deg,
                n_sweep_points,
            )

            # Perform focus sweep at this piezo position
            sweep_results = yield from focus_sweep_plan(
                lightsheet_snap,
                galvo_sweep.tolist(),
                roi_detection=True,
                metadata={"piezo_position": piezo_z},
            )

            # Store results
            best_galvo = sweep_results["best_position"]
            best_score = sweep_results["best_score"]

            results["galvo_positions"].append(best_galvo)
            results["focus_scores"].append(best_score)
            results["sweep_results"].append(sweep_results)

            # Update guess for next iteration (assume roughly linear)
            current_galvo_guess = best_galvo

            logger.info("Best galvo Y: %.6f deg (score: %.2e)", best_galvo, best_score)

    yield from inner()

    # Compute linear fit: galvo_y = slope * piezo_z + offset
    piezo_array = np.array(piezo_positions)
    galvo_array = np.array(results["galvo_positions"])

    # Linear regression
    coeffs = np.polyfit(piezo_array, galvo_array, deg=1)
    slope = coeffs[0]
    offset = coeffs[1]

    # Predicted values and residuals
    galvo_predicted = slope * piezo_array + offset
    residuals = galvo_array - galvo_predicted
    rmse = np.sqrt(np.mean(residuals**2))

    # Store calibration parameters
    results["calibration"] = {
        "slope": slope,
        "offset": offset,
        "rmse": rmse,
        "equation": f"galvo_y = {slope:.6e} * piezo_z + {offset:.6f}",
    }

    logger.info("=" * 70)
    logger.info("CALIBRATION RESULTS")
    logger.info("Slope: %.6e deg/um", slope)
    logger.info("Offset: %.6f deg", offset)
    logger.info("RMSE: %.6e deg", rmse)
    logger.info("Equation: galvo_y = %.6e * piezo_z + %.6f", slope, offset)
    logger.info("=" * 70)

    return results


# =======================
# EMBRYO DETECTION PLANS
# =======================


def mark_embryo_interactive_plan(
    bottom_camera, xy_stage, embryo_number: int, metadata: dict | None = None
):
    """
    Interactive plan for user to mark embryo position and center it.

    This plan:
    1. Captures initial image from bottom camera
    2. Displays image with matplotlib interface for user to click embryo
    3. Moves XY stage to center the marked embryo
    4. Captures confirmation image

    Parameters
    ----------
    bottom_camera : DiSPIMBottomCamera
        Bottom camera device with LED control and pixel calibration.
        LED is automatically managed (on during capture, off after).
    xy_stage : DiSPIMXYStage
        XY stage device with coordinate conversion
    embryo_number : int
        Embryo number for display/tracking
    metadata : Dict, optional
        Additional metadata

    Yields
    ------
    msg : Msg
        Bluesky plan messages

    Returns
    -------
    Dict
        Results with embryo position, stage position, and images
    """
    _md = {"plan_name": "mark_embryo_interactive", "embryo_number": embryo_number}
    if metadata:
        _md.update(metadata)

    @bpp.run_decorator(md=_md)
    def inner():
        logger.info("=" * 70)
        logger.info("MARKING EMBRYO #%d", embryo_number)
        logger.info("=" * 70)

        # Capture initial image
        logger.info("Capturing initial image...")
        yield from bps.trigger_and_read([bottom_camera])
        initial_image = bottom_camera.read()[bottom_camera.name]["value"]

        # Get current stage position
        initial_stage_pos = xy_stage.read()[xy_stage.name]["value"]
        logger.info("Initial stage: (%.2f, %.2f) um", initial_stage_pos[0], initial_stage_pos[1])

        # Display interactive marking interface
        logger.info("INTERACTIVE MARKING - Please click on embryo #%d", embryo_number)

        from matplotlib.widgets import Button

        # Create interactive figure
        fig, ax = plt.subplots(figsize=(12, 10))
        img_norm = (initial_image - initial_image.min()) / (
            initial_image.max() - initial_image.min()
        )
        ax.imshow(img_norm, cmap="gray")

        # Draw center crosshair
        h, w = initial_image.shape
        ax.axvline(w / 2, color="red", linestyle="--", linewidth=2, label="Center")
        ax.axhline(h / 2, color="red", linestyle="--", linewidth=2)

        ax.set_title(f"Click on Embryo #{embryo_number}", fontsize=14, fontweight="bold")

        # Storage for click
        embryo_position = [None, None]

        def onclick(event):
            if event.inaxes == ax and event.button == 1:  # Left click
                embryo_position[0] = event.xdata
                embryo_position[1] = event.ydata
                # Draw marker
                ax.plot(
                    event.xdata,
                    event.ydata,
                    "o",
                    color="lime",
                    markersize=15,
                    markeredgewidth=3,
                    markeredgecolor="white",
                )
                fig.canvas.draw()
                logger.info("Marked at pixel (%.0f, %.0f)", event.xdata, event.ydata)

        def on_done(event):
            plt.close(fig)

        # Connect handlers
        fig.canvas.mpl_connect("button_press_event", onclick)

        # Add Done button
        ax_done = plt.axes([0.81, 0.05, 0.1, 0.04])
        btn_done = Button(ax_done, "Done")
        btn_done.on_clicked(on_done)

        plt.show()

        if embryo_position[0] is None:
            logger.warning("No embryo marked!")
            return {"success": False}

        # Center the embryo (using plan stub for proper Bluesky message flow)
        logger.info("Moving stage to center embryo...")
        yield from move_to_pixel_plan(
            xy_stage, bottom_camera, embryo_position[0], embryo_position[1]
        )

        time.sleep(0.5)

        # Capture confirmation image
        logger.info("Capturing confirmation image...")
        yield from bps.trigger_and_read([bottom_camera])
        centered_image = bottom_camera.read()[bottom_camera.name]["value"]

        final_stage_pos = xy_stage.read()[xy_stage.name]["value"]
        logger.info("Final stage: (%.2f, %.2f) um", final_stage_pos[0], final_stage_pos[1])

        results["success"] = True
        results["embryo_number"] = embryo_number
        results["pixel_position"] = tuple(embryo_position)
        results["initial_stage_position"] = initial_stage_pos
        results["final_stage_position"] = final_stage_pos
        results["initial_image"] = initial_image
        results["centered_image"] = centered_image
        results["timestamp"] = time.time()

        logger.info("Embryo #%d centered!", embryo_number)

    results: dict[str, Any] = {}
    yield from inner()
    return results


# =======================
# VOLUME ACQUISITION PLANS
# =======================


def acquire_single_volume_plan(
    volume_scanner,
    num_slices: int = 100,
    exposure_ms: float = 5.0,
    galvo_amplitude: float = 0.5,
    galvo_center: float = 0.0,
    piezo_amplitude: float = 25.0,
    piezo_center: float = 50.0,
    laser_config: str = "488 and 561",
    laser_power_488_pct: float | None = None,
    laser_power_561_pct: float | None = None,
    laser_power_405_pct: float | None = None,
    laser_power_637_pct: float | None = None,
    timing_params: dict | None = None,
    metadata: dict | None = None,
):
    """
    Acquire a single hardware-triggered 3D volume.

    Uses the DiSPIMVolumeScanner compound device to orchestrate synchronized
    camera+scanner+piezo+laser acquisition.

    Parameters
    ----------
    volume_scanner : DiSPIMVolumeScanner
        Volume scanner compound device (must have laser_control configured)
    num_slices : int
        Number of Z slices (default: 100)
    exposure_ms : float
        Camera exposure in milliseconds (default: 5.0)
    galvo_amplitude : float
        Galvo Y-axis amplitude in degrees (default: 0.5)
    galvo_center : float
        Galvo Y-axis center offset in degrees (default: 0.0)
    piezo_amplitude : float
        Piezo amplitude in micrometers (default: 25.0)
    piezo_center : float
        Piezo center offset in micrometers (default: 50.0)
    laser_config : str
        Laser configuration name (default: "488 and 561").
        Common options: "488 and 561", "488 only", "561 only"
    timing_params : Dict, optional
        Custom SPIM timing parameters
    metadata : Dict, optional
        Additional metadata

    Yields
    ------
    msg : Msg
        Bluesky plan messages

    Returns
    -------
    Dict
        Results with volume data
    """
    _md = {
        "plan_name": "acquire_single_volume",
        "num_slices": num_slices,
        "exposure_ms": exposure_ms,
        "galvo_amplitude": galvo_amplitude,
        "galvo_center": galvo_center,
        "piezo_amplitude": piezo_amplitude,
        "piezo_center": piezo_center,
        "laser_config": laser_config,
        "laser_power_488_pct": laser_power_488_pct,
        "laser_power_561_pct": laser_power_561_pct,
        "laser_power_405_pct": laser_power_405_pct,
        "laser_power_637_pct": laser_power_637_pct,
    }
    if metadata:
        _md.update(metadata)

    results = {}

    @bpp.run_decorator(md=_md)
    def inner():
        logger.info("=" * 70)
        logger.info("VOLUME ACQUISITION")
        logger.info("Slices: %d, Exposure: %.1f ms", num_slices, exposure_ms)
        logger.info("Galvo: %.3f deg amplitude, %.4f deg center", galvo_amplitude, galvo_center)
        logger.info("Piezo: %.2f um amplitude, %.2f um center", piezo_amplitude, piezo_center)
        logger.info("Lasers: %s", laser_config)
        power_log = []
        if laser_power_488_pct is not None:
            power_log.append(f"488@{laser_power_488_pct}%")
        if laser_power_561_pct is not None:
            power_log.append(f"561@{laser_power_561_pct}%")
        if laser_power_405_pct is not None:
            power_log.append(f"405@{laser_power_405_pct}%")
        if laser_power_637_pct is not None:
            power_log.append(f"637@{laser_power_637_pct}%")
        if power_log:
            logger.info("Laser power: %s", ", ".join(power_log))
        logger.info("=" * 70)

        # Configure volume scanner
        logger.info("Configuring hardware...")
        volume_scanner.configure(
            num_slices=num_slices,
            exposure_ms=exposure_ms,
            galvo_amplitude=galvo_amplitude,
            galvo_center=galvo_center,
            piezo_amplitude=piezo_amplitude,
            piezo_center=piezo_center,
            timing_params=timing_params,
            laser_config=laser_config,
            laser_power_488_pct=laser_power_488_pct,
            laser_power_561_pct=laser_power_561_pct,
            laser_power_405_pct=laser_power_405_pct,
            laser_power_637_pct=laser_power_637_pct,
        )

        # Acquire volume
        logger.info("Triggering acquisition...")
        start_time = time.time()
        yield from bps.trigger_and_read([volume_scanner])
        elapsed = time.time() - start_time

        # Get volume data
        volume_data = volume_scanner.read()[volume_scanner.name]["value"]

        logger.info("Volume acquired! Shape: %s, Time: %.2f s", volume_data.shape, elapsed)

        results["volume"] = volume_data
        results["shape"] = volume_data.shape
        results["acquisition_time"] = elapsed
        results["timestamp"] = time.time()

    yield from inner()
    return results


def burst_plan(
    volume_scanner,
    frames: int = 60,
    mode: str = "1hz",
    num_slices: int = 1,
    exposure_ms: float = 5.0,
    galvo_amplitude: float = 0.5,
    galvo_center: float = 0.0,
    piezo_amplitude: float = 25.0,
    piezo_center: float = 50.0,
    laser_config: str = "488 only",
    laser_power_488_pct: float | None = None,
    laser_power_561_pct: float | None = None,
    laser_power_405_pct: float | None = None,
    laser_power_637_pct: float | None = None,
    timing_params: dict | None = None,
    metadata: dict | None = None,
):
    """
    Acquire a burst of N volumes back-to-back as a single Bluesky run.

    Configures the volume scanner once, then yields ``trigger_and_read`` N
    times. The whole burst runs inside one ``run_decorator`` so the device
    layer holds ``pause_state_updates()`` for the entire duration — no
    inter-frame gap for the position pollers to race into.

    Each ``trigger_and_read`` produces one event document; ``collect_docs``
    in the device layer turns the volume into a file reference, so the
    HTTP response stays small (60 paths, not 60 arrays).

    Parameters
    ----------
    volume_scanner : DiSPIMVolumeScanner
        Volume scanner compound device.
    frames : int
        Number of volumes to capture (default 60).
    mode : {"1hz", "asap"}
        ``"1hz"`` paces each frame to a 1 s tick; ``"asap"`` chains
        acquisitions as fast as the hardware delivers.
    num_slices : int
        Z-slices per volume. Default 1 (snap-style — fastest path to ~1 Hz).
    exposure_ms : float
        Camera exposure per slice.
    galvo_amplitude, galvo_center, piezo_amplitude, piezo_center : float
        Scan geometry — same meaning as in ``acquire_single_volume_plan``.
    laser_config : str
        Laser preset, e.g. ``"488 only"``.
    laser_power_488_pct, laser_power_561_pct, laser_power_405_pct,
    laser_power_637_pct : float, optional
        Per-line laser power %. Hard-limited at the device layer.
    timing_params : Dict, optional
        Custom SPIM timing parameters.
    metadata : Dict, optional
        Extra metadata merged into the start document.

    Yields
    ------
    msg : Msg
        Bluesky plan messages.

    Returns
    -------
    Dict
        ``{'frames_captured', 'frames_requested', 'mode', 'duration_s',
           'sustained_hz'}``. Volumes are not in the return value — they
        ride out in the run's event documents.
    """
    if mode not in ("1hz", "asap"):
        mode = "1hz"
    target_dt = 1.0 if mode == "1hz" else 0.0

    _md = {
        "plan_name": "burst",
        "frames": frames,
        "mode": mode,
        "num_slices": num_slices,
        "exposure_ms": exposure_ms,
        "laser_config": laser_config,
        "laser_power_488_pct": laser_power_488_pct,
        "laser_power_561_pct": laser_power_561_pct,
        "laser_power_405_pct": laser_power_405_pct,
        "laser_power_637_pct": laser_power_637_pct,
    }
    if metadata:
        _md.update(metadata)

    results = {
        "frames_captured": 0,
        "frames_requested": frames,
        "mode": mode,
        "duration_s": 0.0,
        "sustained_hz": 0.0,
    }

    @bpp.run_decorator(md=_md)
    def inner():
        logger.info("=" * 70)
        logger.info("BURST ACQUISITION")
        logger.info("Frames: %d, Mode: %s, Slices/frame: %d", frames, mode, num_slices)
        power_log = []
        if laser_power_488_pct is not None:
            power_log.append(f"488@{laser_power_488_pct}%")
        if laser_power_561_pct is not None:
            power_log.append(f"561@{laser_power_561_pct}%")
        if laser_power_405_pct is not None:
            power_log.append(f"405@{laser_power_405_pct}%")
        if laser_power_637_pct is not None:
            power_log.append(f"637@{laser_power_637_pct}%")
        if power_log:
            logger.info("Laser power: %s", ", ".join(power_log))
        logger.info("=" * 70)

        # Configure once — keeps laser/scanner/piezo settings stable for the
        # whole burst and saves the per-frame configure overhead.
        volume_scanner.configure(
            num_slices=num_slices,
            exposure_ms=exposure_ms,
            galvo_amplitude=galvo_amplitude,
            galvo_center=galvo_center,
            piezo_amplitude=piezo_amplitude,
            piezo_center=piezo_center,
            timing_params=timing_params,
            laser_config=laser_config,
            laser_power_488_pct=laser_power_488_pct,
            laser_power_561_pct=laser_power_561_pct,
            laser_power_405_pct=laser_power_405_pct,
            laser_power_637_pct=laser_power_637_pct,
        )

        burst_start = time.time()
        for i in range(frames):
            tick_start = time.time()
            # One event per frame, each carrying its own volume file_ref.
            yield from bps.trigger_and_read([volume_scanner])
            results["frames_captured"] = i + 1

            # Pacing — only between frames, not after the last.
            if i < frames - 1 and target_dt > 0:
                elapsed = time.time() - tick_start
                wait = target_dt - elapsed
                if wait > 0:
                    yield from bps.sleep(wait)

        duration = time.time() - burst_start
        results["duration_s"] = duration
        results["sustained_hz"] = results["frames_captured"] / duration if duration > 0 else 0.0

        logger.info(
            "BURST COMPLETE — %d/%d frames in %.2fs (%.2f Hz)",
            results["frames_captured"],
            frames,
            duration,
            results["sustained_hz"],
        )

    yield from inner()
    return results


def timelapse_volume_plan(
    volume_scanner, num_timepoints: int, interval_seconds: float, **volume_kwargs
):
    """
    Acquire time-lapse series of 3D volumes.

    Parameters
    ----------
    volume_scanner : DiSPIMVolumeScanner
        Volume scanner device
    num_timepoints : int
        Number of time points to acquire
    interval_seconds : float
        Time interval between acquisitions (seconds)
    **volume_kwargs
        Additional arguments passed to acquire_single_volume_plan

    Yields
    ------
    msg : Msg
        Bluesky plan messages

    Returns
    -------
    Dict
        Results with all volumes and timestamps
    """
    results = {
        "volumes": [],
        "timestamps": [],
        "num_timepoints": num_timepoints,
        "interval_seconds": interval_seconds,
    }

    _md = {
        "plan_name": "timelapse_volume",
        "num_timepoints": num_timepoints,
        "interval_seconds": interval_seconds,
    }

    @bpp.run_decorator(md=_md)
    def inner():
        logger.info("=" * 70)
        logger.info("TIME-LAPSE VOLUME ACQUISITION")
        logger.info("Timepoints: %d", num_timepoints)
        logger.info("Interval: %d s", interval_seconds)
        logger.info("=" * 70)

        for tp in range(num_timepoints):
            logger.info("[TIMEPOINT %d/%d]", tp + 1, num_timepoints)

            # Acquire volume
            vol_results = yield from acquire_single_volume_plan(
                volume_scanner, metadata={"timepoint": tp}, **volume_kwargs
            )

            results["volumes"].append(vol_results["volume"])
            results["timestamps"].append(vol_results["timestamp"])

            # Wait for next timepoint (except after last one)
            if tp < num_timepoints - 1:
                logger.info("Waiting %d s until next timepoint...", interval_seconds)
                yield from bps.sleep(interval_seconds)

        logger.info("TIME-LAPSE COMPLETE - Total volumes: %d", len(results["volumes"]))

    yield from inner()
    return results


# =======================
# MULTI-EMBRYO WORKFLOWS
# =======================


def multi_embryo_calibration_workflow(
    bottom_camera,
    xy_stage,
    lightsheet_snap,
    num_embryos: int,
    calibration_params: dict | None = None,
):
    """
    Full multi-embryo calibration workflow.

    For each embryo:
    1. Mark embryo interactively
    2. Center embryo with stage
    3. Perform piezo-galvo calibration
    4. Save calibration to database

    Parameters
    ----------
    bottom_camera : DiSPIMBottomCamera
        Bottom camera for embryo detection (must have LED control configured).
        LED automatically managed during imaging.
    xy_stage : DiSPIMXYStage
        XY stage for positioning
    lightsheet_snap : DiSPIMLightSheetSnap
        Light sheet device for calibration
    num_embryos : int
        Number of embryos to calibrate
    calibration_params : Dict, optional
        Parameters for piezo-galvo calibration

    Yields
    ------
    msg : Msg
        Bluesky plan messages

    Returns
    -------
    Dict
        Results with all embryo calibrations
    """
    # Default calibration parameters
    if calibration_params is None:
        calibration_params = {
            "piezo_positions": [40.0, 60.0],  # Two-point calibration
            "search_range_deg": 0.02,
            "n_sweep_points": 21,
        }

    results = {"embryos": [], "num_embryos": num_embryos, "timestamp": time.time()}

    _md = {"plan_name": "multi_embryo_calibration_workflow", "num_embryos": num_embryos}

    @bpp.run_decorator(md=_md)
    def inner():
        logger.info("#" * 70)
        logger.info("MULTI-EMBRYO CALIBRATION WORKFLOW")
        logger.info("Number of embryos: %d", num_embryos)
        logger.info("#" * 70)

        for emb_num in range(1, num_embryos + 1):
            logger.info("#" * 70)
            logger.info("EMBRYO %d/%d", emb_num, num_embryos)
            logger.info("#" * 70)

            # Mark and center embryo
            marking_results = yield from mark_embryo_interactive_plan(
                bottom_camera, xy_stage, embryo_number=emb_num
            )

            if not marking_results.get("success", False):
                logger.warning("Skipping embryo %d", emb_num)
                continue

            # Perform calibration
            calib_results = yield from calibrate_piezo_galvo_plan(
                lightsheet_snap,
                metadata={"embryo_number": emb_num},
                **calibration_params,
            )

            # Store results
            embryo_data = {
                "embryo_number": emb_num,
                "marking": marking_results,
                "calibration": calib_results,
                "timestamp": time.time(),
            }

            results["embryos"].append(embryo_data)

            logger.info("Embryo %d calibration complete!", emb_num)

        logger.info(
            "WORKFLOW COMPLETE - Calibrated %d/%d embryos",
            len(results["embryos"]),
            num_embryos,
        )

    yield from inner()
    return results


# =============================================================================
# Utility Plans (simple device operations for HTTP API)
# =============================================================================


def move_stage_plan(xy_stage, x: float, y: float) -> Generator[Any, Any, dict]:
    """Move XY stage to specified position."""
    yield from bps.mv(xy_stage, [x, y])
    return {"x": x, "y": y, "success": True}


def read_stage_plan(xy_stage) -> Generator[Any, Any, None]:
    """Read current XY stage position."""
    yield from bp.count([xy_stage], num=1)


def read_piezo_plan(piezo) -> Generator[Any, Any, None]:
    """Read current piezo position."""
    yield from bp.count([piezo], num=1)


def capture_bottom_image_plan(bottom_camera, led=None) -> Generator[Any, Any, None]:
    """Capture a single image from the bottom camera."""
    if led is not None:
        try:
            yield from bps.mv(led, "Open")
        except Exception:
            pass
    yield from bp.count([bottom_camera], num=1)
    if led is not None:
        try:
            yield from bps.mv(led, "Closed")
        except Exception:
            pass


def capture_lightsheet_image_plan(
    lightsheet_snap,
    scanner,
    piezo,
    laser_control,
    piezo_position: float = 50.0,
    galvo_position: float = 0.0,
    laser_config: str = "488 and 561",
) -> Generator[Any, Any, None]:
    """Capture a single lightsheet image at specified piezo/galvo positions."""
    yield from bps.mv(piezo, piezo_position)
    yield from bps.mv(scanner.sa_offset_y, galvo_position)
    lightsheet_snap.configure(y_position_deg=galvo_position)
    yield from bps.mv(laser_control, laser_config)
    try:
        yield from bp.count([lightsheet_snap], num=1)
    finally:
        yield from bps.mv(laser_control, "ALL OFF")


def move_piezo_plan(piezo, position: float) -> Generator[Any, Any, dict]:
    """Move piezo to specified position."""
    yield from bps.mv(piezo, position)
    return {"position": position, "success": True}


def move_scanner_plan(scanner, offset_y: float) -> Generator[Any, Any, dict]:
    """Move scanner galvo to specified offset."""
    yield from bps.mv(scanner.sa_offset_y, offset_y)
    return {"offset_y": offset_y, "success": True}


def set_laser_plan(laser_control, state: str = "ON") -> Generator[Any, Any, dict]:
    """Set laser state."""
    yield from bps.mv(laser_control.state, state)
    return {"state": state, "success": True}


def set_led_plan(led, state: str = "Open") -> Generator[Any, Any, dict]:
    """Set LED state."""
    yield from bps.mv(led, state)
    return {"state": state, "success": True}


def set_light_source_power_plan(
    light_source,
    wavelength: int,
    pct: float,
) -> Generator[Any, Any, dict]:
    """Set per-line laser power %.

    Hard-limited at the device layer by ``light_source.POWER_LIMITS_PCT``.
    Out-of-range values raise ``ValueError`` from inside the device setter
    — the plan will then fail loudly through the queue server.
    """
    light_source.set_power_pct(wavelength, pct)
    yield from bps.null()  # plan must be a generator
    return {
        "wavelength": wavelength,
        "pct": pct,
        "success": True,
    }


def get_light_source_power_plan(
    light_source,
    wavelength: int,
) -> Generator[Any, Any, dict]:
    """Read the current per-line laser power %."""
    value = light_source.get_power_pct(wavelength)
    yield from bps.null()
    return {
        "wavelength": wavelength,
        "pct": value,
        "success": True,
    }


# ============================================================================
# SPIM HEAD FOCUS
# Bring the SPIM head down onto an XY-positioned embryo, lock focus, then
# register the two objective views on top of each other via the XY stage.
# ============================================================================


def spim_head_focus_descent_plan(
    fdrive,
    camera,
    led=None,
    *,
    traverse_to_um: float = 5000.0,
    coarse_step_um: float = 1000.0,
    coarse_stop_um: float = 500.0,
    fine_top_um: float = 150.0,
    fine_bottom_um: float = 35.0,
    fine_step_um: float = 5.0,
    focus_algorithm: str = "volath",
    coarse_settle_s: float = 2.0,
    fine_settle_s: float = 0.5,
    led_state: str = "Open",
    detect_roi: bool = True,
) -> Generator[Any, Any, dict]:
    """
    Bring the SPIM head down onto an (already XY-positioned) embryo and lock focus.

    The head parks fully raised (~25000 um, "Load Sample") and the embryo only
    comes into focus near the bottom of travel (~50 um), so almost the whole
    descent is empty space -- and full-travel F moves are slow. Rather than scan
    the entire 25000->50 range, this descends in three phases:

      1. Traverse -- one fast move down to ``traverse_to_um`` (default 5000). No imaging.
      2. Coarse   -- step down by ``coarse_step_um`` (1000) to ``coarse_stop_um``
                     (~500), settling at each step. No focus decision yet.
      3. Fine     -- bounded sweep-and-fit from ``fine_top_um`` (150) down to
                     ``fine_bottom_um`` (35, >= the 30 um hard floor) in
                     ``fine_step_um`` (5) steps: snap ``camera`` + score each
                     frame, fit the focus curve (gently.analysis.focus), and
                     move to the best-focus position.

    ``led`` (transmitted illumination) is opened for the descent and ALWAYS
    closed afterwards (finalize), mirroring the bottom camera's LED discipline.

    Every F move is clamped to ``fdrive.limits`` (30-25000 um); the 30 um floor
    is the hard collision stop and ``fine_bottom_um`` must sit above it.

    Parameters
    ----------
    fdrive : DiSPIMFDrive
        SPIM-head F-drive positioner (``bps.mv(fdrive, z)``).
    camera : DiSPIMCamera
        SPIM objective camera; ``camera.snap()`` returns the frame to score.
    led : DiSPIMLED, optional
        Transmitted-light source. Opened during the descent, closed after.
    focus_algorithm : str
        Focus metric: 'volath' (default), 'gradient', 'variance', 'fft_bandpass'.

    Returns
    -------
    dict
        success, best_position_um, best_score, r_squared, start_position_um,
        and fine_curve (list of (z_um, score) tuples).
    """
    from gently.analysis.focus import (
        FocusAnalysisConfig,
        FocusDataPoint,
        analyze_focus_sweep,
        score_single_image,
    )

    lo, hi = fdrive.limits
    config = FocusAnalysisConfig(algorithm=focus_algorithm)
    out: dict[str, Any] = {}

    def _clamp(z: float) -> float:
        return max(lo, min(hi, float(z)))

    def _grab_and_score():
        frame = camera.snap()
        # A single SPIM snap may be a side-by-side dual view; score the brighter
        # objective half (focus is per-objective). Heuristic: width >> height.
        if frame.ndim == 2 and frame.shape[1] >= frame.shape[0] * 1.5:
            view = select_best_camera_view(frame)
        else:
            view = frame
        score, roi = score_single_image(view, config, detect_roi=detect_roi)
        return view, score, roi

    def inner():
        # bps.rd returns the device's single reading value directly (a float
        # here) in this bluesky version -- not a {name: {value}} dict.
        start_z = float((yield from bps.rd(fdrive)))
        logger.info("[SPIM head focus] start %.1f um, floor %.0f um", start_z, lo)

        if led is not None:
            yield from bps.mv(led, led_state)

        # Phase 1 -- fast traverse (skip if already below the traverse height)
        z = start_z
        if z > traverse_to_um:
            z = _clamp(traverse_to_um)
            logger.info("[SPIM head focus] traverse -> %.0f um (slow move)", z)
            yield from bps.mv(fdrive, z)
            yield from bps.sleep(coarse_settle_s)

        # Phase 2 -- coarse stepped descent to ~coarse_stop_um
        while z > coarse_stop_um + 1e-6:
            z = _clamp(max(coarse_stop_um, z - coarse_step_um))
            logger.info("[SPIM head focus] coarse -> %.0f um", z)
            yield from bps.mv(fdrive, z)
            yield from bps.sleep(coarse_settle_s)

        # Phase 3 -- fine bounded sweep-and-fit through the focus window
        top, bottom = _clamp(fine_top_um), _clamp(fine_bottom_um)
        n = max(3, int(round(abs(top - bottom) / max(fine_step_um, 1e-6))) + 1)
        positions = [_clamp(p) for p in np.linspace(top, bottom, n)]  # descending
        logger.info("[SPIM head focus] fine scan %.0f->%.0f um in %d steps", top, bottom, n)

        sweep = []
        for zp in positions:
            yield from bps.mv(fdrive, zp)
            yield from bps.sleep(fine_settle_s)
            view, score, roi = _grab_and_score()
            sweep.append(FocusDataPoint(position=zp, score=score, image=view, roi=roi))
            logger.debug("[SPIM head focus] z=%.1f score=%.3g", zp, score)

        result = analyze_focus_sweep(sweep, config)
        if result.success:
            best = _clamp(result.best_position)
        else:
            best = _clamp(max(sweep, key=lambda d: d.score).position)

        yield from bps.mv(fdrive, best)
        yield from bps.sleep(fine_settle_s)

        out.update(
            success=bool(result.success),
            best_position_um=float(best),
            best_score=float(result.best_score),
            r_squared=float(result.r_squared),
            start_position_um=start_z,
            fine_curve=[(float(d.position), float(d.score)) for d in sweep],
        )
        logger.info(
            "[SPIM head focus] locked %.1f um (score %.3g, R2=%.3f, fit=%s)",
            best,
            result.best_score,
            result.r_squared,
            result.success,
        )

    def cleanup():
        if led is not None:
            yield from bps.mv(led, "Closed")

    yield from bpp.finalize_wrapper(inner(), cleanup())
    return out


def register_views_xy_plan(
    camera,
    xy_stage,
    *,
    view_pixel_size_um: float,
    tolerance_px: float = 20.0,
    max_iters: int = 4,
    settle_s: float = 0.5,
    led=None,
    led_state: str = "Open",
    reference_view: int = 0,
    x_sign: float = -1.0,
    y_sign: float = 1.0,
) -> Generator[Any, Any, dict]:
    """
    Register the two SPIM objective views on top of each other via the XY stage.

    Snaps ``camera`` (a side-by-side dual view: left = view A, right = view B),
    detects the embryo centroid in each half, and moves the XY stage to bring
    the embryo to the centre of the ``reference_view`` (0 = A/left, 1 = B/right).
    The residual offset in the *other* view is recorded each iteration as the
    registration error. Iterates up to ``max_iters`` or until the reference
    view is within ``tolerance_px``.

    The view->stage mapping is the rig-specific part and likely needs tuning:
      - ``view_pixel_size_um`` is the effective um/pixel of the OBJECTIVE view
        (NOT the bottom camera) -- required.
      - ``x_sign`` / ``y_sign`` flip the stage axes to match the camera
        orientation. Defaults follow the bottom-camera convention (X inverted);
        confirm them on the rig from the reported per-iteration residuals (if
        the offset grows instead of shrinking, flip the offending sign).

    XY moves are bounded by the stage's own hardware limits, so a mis-tuned
    sign fails loudly (limit error) rather than driving anywhere unsafe.

    Returns
    -------
    dict
        converged (bool), iterations (per-iter offsets per view, in px),
        final_stage_um, and (if nothing was found) error.
    """
    from gently.core.coordinates import pixel_displacement_to_stage_movement

    out: dict[str, Any] = {"converged": False, "iterations": []}

    def _views(frame):
        if frame.ndim == 2 and frame.shape[1] >= frame.shape[0] * 1.5:
            mid = frame.shape[1] // 2
            return [frame[:, :mid], frame[:, mid:]]
        return [frame]

    def _centroid_offset(view):
        # (dx, dy) of the embryo centroid from the view centre, in pixels;
        # None if no embryo (detect_embryo_roi falls back to the whole frame).
        y0, y1, x0, x1 = detect_embryo_roi(view)
        h, w = view.shape
        if (y1 - y0) >= h and (x1 - x0) >= w:
            return None
        cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
        return (cx - w / 2.0, cy - h / 2.0)

    def inner():
        if led is not None:
            yield from bps.mv(led, led_state)

        for _ in range(max_iters):
            frame = camera.snap()
            views = _views(frame)
            offsets = [_centroid_offset(v) for v in views]
            out["iterations"].append(
                {"offsets_px": [None if o is None else (float(o[0]), float(o[1])) for o in offsets]}
            )

            ref = offsets[reference_view] if reference_view < len(offsets) else None
            if ref is None:
                seen = [o for o in offsets if o is not None]
                if not seen:
                    out["error"] = "no embryo detected in any view"
                    break
                ref = seen[0]

            if max(abs(ref[0]), abs(ref[1])) <= tolerance_px:
                out["converged"] = True
                break

            dx_um, dy_um = pixel_displacement_to_stage_movement(ref[0], ref[1], view_pixel_size_um)
            cur = yield from bps.rd(xy_stage)  # -> [x, y] um (single reading value)
            target = [cur[0] + x_sign * dx_um, cur[1] + y_sign * dy_um]
            yield from bps.mov(xy_stage, target)
            yield from bps.sleep(settle_s)

        final = yield from bps.rd(xy_stage)
        out["final_stage_um"] = [float(final[0]), float(final[1])]

    def cleanup():
        if led is not None:
            yield from bps.mv(led, "Closed")

    yield from bpp.finalize_wrapper(inner(), cleanup())
    return out


def spim_head_focus_and_align_plan(
    fdrive,
    camera,
    xy_stage,
    led=None,
    *,
    view_pixel_size_um: float | None = None,
    align: bool = True,
    focus_kwargs: dict | None = None,
    align_kwargs: dict | None = None,
) -> Generator[Any, Any, dict]:
    """
    Full SPIM-head focus workflow for one XY-positioned embryo.

    Descends and locks focus (``spim_head_focus_descent_plan``), then registers
    the two objective views via the XY stage (``register_views_xy_plan``).

    ``align=True`` requires ``view_pixel_size_um`` (objective-view um/pixel).
    """
    out: dict[str, Any] = {}
    out["focus"] = yield from spim_head_focus_descent_plan(
        fdrive, camera, led=led, **(focus_kwargs or {})
    )
    if align:
        if view_pixel_size_um is None:
            raise ValueError(
                "align=True requires view_pixel_size_um (objective-view um/pixel calibration)"
            )
        out["align"] = yield from register_views_xy_plan(
            camera,
            xy_stage,
            view_pixel_size_um=view_pixel_size_um,
            led=led,
            **(align_kwargs or {}),
        )
    return out
