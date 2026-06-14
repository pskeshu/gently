#!/usr/bin/env python3
"""
Multi-Embryo Calibration Plans for Bluesky
==========================================

Orchestration plans for calibrating multiple embryos in a single session.
Integrates napari-based interactive marking with automated per-embryo calibration.

Plans included:
- multi_embryo_calibration_session_plan: Top-level orchestrator
- center_and_verify_embryo_plan: Stage movement and verification
- calibrate_single_embryo_in_session_plan: Per-embryo calibration wrapper

All plans use databroker as primary storage with JSON export at completion.
"""

import logging
from datetime import datetime
from pathlib import Path

import bluesky.plan_stubs as bps
import numpy as np

# Import existing calibration infrastructure
from gently.core.database import (
    add_embryo_to_database,
    save_multi_embryo_database,
)
from gently.ui.web.embryo_marker import mark_embryos_web

from .calibration import calibrate_embryo_piezo_galvo

logger = logging.getLogger(__name__)

# ============================================================================
# PLAN: CENTER AND VERIFY EMBRYO
# ============================================================================


def center_and_verify_embryo_plan(
    bottom_camera,
    xy_stage,
    embryo_data: dict,
    save_verification: bool = True,
    image_dir: Path | None = None,
):
    """
    Center XY stage on marked embryo position and capture verification image.

    This plan:
    1. Reads current stage position
    2. Calculates movement from pixel offset
    3. Moves stage to center embryo
    4. Captures verification image
    5. Records before/after positions in metadata

    Parameters
    ----------
    bottom_camera : DiSPIMBottomCamera
        Bottom camera device with LED control
    xy_stage : DiSPIMXYStage
        XY stage device
    embryo_data : dict
        Embryo data with 'pixel_position', 'initial_stage_position', 'embryo_id'
    save_verification : bool, optional
        Save verification image to disk (default: True)
    image_dir : Path, optional
        Directory for saving images (default: None, uses current dir)

    Yields
    ------
    Bluesky messages for run control and data collection

    Returns
    -------
    dict
        Updated embryo data with 'centered_stage_position_um' and verification info
    """
    embryo_id = embryo_data["embryo_id"]
    pixel_x, pixel_y = embryo_data["pixel_position"]
    initial_stage_x, initial_stage_y = embryo_data["initial_stage_position"]

    logger.info("[Centering %s]", embryo_id)
    logger.info("Pixel position: (%.1f, %.1f)", pixel_x, pixel_y)

    # Read current stage position
    current_pos = xy_stage.get_position()
    logger.info("Current stage: (%.2f, %.2f) um", current_pos[0], current_pos[1])

    # Calculate pixel displacement needed to center the embryo
    # Note: Must use (CENTER - EMBRYO), not (EMBRYO - CENTER)!
    # This matches the original multi_embryo_calibration.py logic
    image_height, image_width = 2048, 2048  # Bottom camera dimensions
    image_center_x = image_width / 2.0
    image_center_y = image_height / 2.0

    pixel_displacement_x = image_center_x - pixel_x  # CENTER - EMBRYO
    pixel_displacement_y = image_center_y - pixel_y

    # Convert to stage movement with X-axis inversion
    # Original formula: dx_stage = -pixel_displacement_x * pixel_size (X inverted)
    dx = -pixel_displacement_x * bottom_camera.effective_pixel_size
    dy = pixel_displacement_y * bottom_camera.effective_pixel_size

    target_pos = current_pos + np.array([dx, dy])

    # Debug output matching original code format
    logger.debug("Image center: (%.0f, %.0f) pixels", image_center_x, image_center_y)
    logger.debug(
        "Pixel displacement: (%+.1f, %+.1f) pixels",
        pixel_displacement_x,
        pixel_displacement_y,
    )
    logger.debug("Stage movement: (%+.2f, %+.2f) um", dx, dy)
    logger.info("Target stage: (%.2f, %.2f) um", target_pos[0], target_pos[1])

    # Check if target position is within stage limits
    x_min, x_max = xy_stage._x_limits
    y_min, y_max = xy_stage._y_limits

    if not (x_min <= target_pos[0] <= x_max):
        logger.error(
            "Target X position %.2f outside limits (%s, %s)",
            target_pos[0],
            x_min,
            x_max,
        )
        logger.error("Skipping %s - marked position unreachable", embryo_id)
        embryo_data["error"] = (
            f"Stage X out of bounds: {target_pos[0]:.2f} not in ({x_min}, {x_max})"
        )
        return embryo_data

    if not (y_min <= target_pos[1] <= y_max):
        logger.error(
            "Target Y position %.2f outside limits (%s, %s)",
            target_pos[1],
            y_min,
            y_max,
        )
        logger.error("Skipping %s - marked position unreachable", embryo_id)
        embryo_data["error"] = (
            f"Stage Y out of bounds: {target_pos[1]:.2f} not in ({y_min}, {y_max})"
        )
        return embryo_data

    # Move stage
    yield from bps.mov(xy_stage, target_pos)

    # Wait for settling
    yield from bps.sleep(0.5)

    # Verify position
    final_pos = xy_stage.get_position()
    logger.info("Final stage: (%.2f, %.2f) um", final_pos[0], final_pos[1])

    # Capture verification image
    logger.info("Capturing verification image...")
    yield from bps.trigger_and_read([bottom_camera], name="verification")

    # Save verification image if requested
    if save_verification and image_dir is not None:
        image_dir = Path(image_dir)
        image_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = image_dir / f"{embryo_id}_AFTER_centering_{timestamp}.png"

        # Get image from device
        verification_image = bottom_camera._last_image

        if verification_image is not None:
            import tifffile

            tifffile.imwrite(save_path, verification_image)
            logger.info("Saved: %s", save_path.name)

    # Update embryo data with centered position
    embryo_data["centered_stage_x"] = float(final_pos[0])
    embryo_data["centered_stage_y"] = float(final_pos[1])
    embryo_data["centering_timestamp"] = datetime.now().isoformat()

    logger.info("%s centered!", embryo_id)

    return embryo_data


# ============================================================================
# PLAN: CALIBRATE SINGLE EMBRYO IN SESSION
# ============================================================================


def calibrate_single_embryo_in_session_plan(
    embryo_data: dict,
    embryo_detector,
    laser_control,
    image_dir: Path | None = None,
    calibration_params: dict | None = None,
):
    """
    Calibrate single embryo and store results in databroker.

    Wrapper around calibrate_embryo_piezo_galvo() that adds session-level
    metadata and updates embryo data with calibration results.

    Parameters
    ----------
    embryo_data : dict
        Embryo data with positions and identifiers
    embryo_detector : DiSPIMEmbryoDetector
        Composite device for calibration (camera, galvo, piezo, Claude)
    laser_control : DiSPIMLaserControl
        Laser control device
    image_dir : Path, optional
        Directory for saving calibration images
    calibration_params : dict, optional
        Custom calibration parameters to override defaults

    Yields
    ------
    Bluesky messages for run control and data collection

    Returns
    -------
    dict
        Updated embryo data with calibration results
    """
    embryo_id = embryo_data["embryo_id"]
    embryo_number = embryo_data["embryo_number"]

    logger.info("=" * 70)
    logger.info("CALIBRATING %s", embryo_id.upper())
    logger.info("=" * 70)

    # Prepare metadata for this embryo run
    {
        "embryo_id": embryo_id,
        "embryo_number": embryo_number,
        "pixel_x": embryo_data["pixel_position"][0],
        "pixel_y": embryo_data["pixel_position"][1],
        "initial_stage_x": embryo_data["initial_stage_position"][0],
        "initial_stage_y": embryo_data["initial_stage_position"][1],
        "centered_stage_x": embryo_data.get("centered_stage_x", 0.0),
        "centered_stage_y": embryo_data.get("centered_stage_y", 0.0),
        "marking_timestamp": embryo_data.get("marking_timestamp", ""),
        "centering_timestamp": embryo_data.get("centering_timestamp", ""),
    }

    # Merge custom calibration parameters
    calib_params = calibration_params or {}

    # Run calibration plan
    # Note: embryo_detector parameter in calibrate_embryo_piezo_galvo is the Claude client
    # Metadata will be captured in the plan's own run, not passed as parameter
    calibration_result = yield from calibrate_embryo_piezo_galvo(
        camera=embryo_detector.camera,
        galvo=embryo_detector.galvo,
        piezo=embryo_detector.piezo,
        focus_scorer=embryo_detector.focus_scorer,
        embryo_detector=embryo_detector.claude_client,
        core=embryo_detector.core,
        image_dir=image_dir,
        **calib_params,
    )

    # Extract calibration data from result
    if calibration_result is not None:
        embryo_data["calibration"] = calibration_result
        logger.info("%s calibration complete!", embryo_id)
    else:
        logger.error("%s calibration failed!", embryo_id)
        embryo_data["calibration"] = None

    return embryo_data


# ============================================================================
# PLAN: MULTI-EMBRYO CALIBRATION SESSION
# ============================================================================


def multi_embryo_calibration_session_plan(
    bottom_camera,
    xy_stage,
    embryo_detector,
    laser_control,
    output_database_path: Path,
    image_dir: Path | None = None,
    auto_mark: bool = False,
    pre_marked_embryos: list[dict] | None = None,
    calibration_params: dict | None = None,
    **kwargs,
):
    """
    Complete multi-embryo calibration workflow with Bluesky architecture.

    This is the top-level plan that orchestrates the entire multi-embryo
    calibration session:
    1. Capture bottom camera overview
    2. Interactive embryo marking with napari (or use pre-marked positions)
    3. For each embryo:
       a. Center stage on embryo
       b. Verify centering with bottom camera
       c. Run complete piezo-galvo calibration
       d. Store results in databroker
    4. Export all results to JSON database file

    Parameters
    ----------
    bottom_camera : DiSPIMBottomCamera
        Bottom camera for overview and verification
    xy_stage : DiSPIMXYStage
        XY stage for positioning
    embryo_detector : DiSPIMEmbryoDetector
        Composite device for calibration (camera, galvo, piezo, Claude, focus_scorer)
    laser_control : DiSPIMLaserControl
        Laser control for SPIM imaging
    output_database_path : Path
        Path for output JSON database file
    image_dir : Path, optional
        Directory for saving images (default: None, creates temp directory)
    auto_mark : bool, optional
        Auto-detect embryos instead of interactive marking (not yet implemented)
    pre_marked_embryos : list of dict, optional
        Pre-marked embryo positions (skips interactive marking)
    calibration_params : dict, optional
        Custom calibration parameters passed to each embryo calibration

    Yields
    ------
    Bluesky messages for run control and data collection

    Returns
    -------
    Path
        Path to exported JSON database file

    Examples
    --------
    >>> from bluesky import RunEngine
    >>> from databroker import Broker
    >>> from gently.devices import DiSPIMBottomCamera, DiSPIMXYStage, ...
    >>> from gently.multi_embryo_plans import multi_embryo_calibration_session_plan
    >>>
    >>> RE = RunEngine({})
    >>> db = Broker.named('temp')
    >>> RE.subscribe(db.insert)
    >>>
    >>> # Create devices
    >>> bottom_camera = DiSPIMBottomCamera(...)
    >>> xy_stage = DiSPIMXYStage(...)
    >>> embryo_detector = DiSPIMEmbryoDetector(...)
    >>> laser_control = DiSPIMLaserControl(...)
    >>>
    >>> # Run calibration
    >>> uids = RE(multi_embryo_calibration_session_plan(
    ...     bottom_camera, xy_stage, embryo_detector, laser_control,
    ...     output_database_path=Path("multi_embryo_database.json")
    ... ))
    >>>
    >>> # Database exported to JSON file
    """
    # Setup image directory
    if image_dir is None:
        image_dir = Path(f"calibration_session_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    image_dir = Path(image_dir)
    image_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("MULTI-EMBRYO CALIBRATION SESSION - BLUESKY ARCHITECTURE")
    logger.info("=" * 70)

    # Prepare session metadata
    session_metadata = {
        "plan_name": "multi_embryo_calibration_session",
        "output_database_path": str(output_database_path),
        "image_dir": str(image_dir),
        "session_start": datetime.now().isoformat(),
        "embryo_runs": [],  # Will be populated with UIDs
    }

    def inner_session_plan():
        """Inner plan with session metadata."""

        # ====================================================================
        # PHASE 1: SETUP (OVERVIEW + MARKING DONE OUTSIDE IF PRE-MARKED)
        # ====================================================================
        logger.info("[PHASE 1] Embryo positions...")

        # Use pre-marked embryos if provided (marking done outside plan to avoid Qt threading)
        if pre_marked_embryos is not None:
            logger.info(
                "Using pre-marked embryo positions (%d embryos)",
                len(pre_marked_embryos),
            )
            marked_embryos = pre_marked_embryos
        elif auto_mark:
            raise NotImplementedError(
                "Auto-detection not yet implemented. Use pre-marked positions."
            )
        else:
            # This path requires capturing overview and interactive marking inside plan
            # WARNING: This may cause Qt threading issues with napari!
            logger.warning("Interactive marking inside plan may cause Qt threading errors!")
            logger.warning("Recommended: Mark embryos outside plan and pass as pre_marked_embryos")

            # Get initial stage position
            initial_stage_pos = xy_stage.get_position()
            logger.info(
                "Initial stage position: (%.2f, %.2f) um",
                initial_stage_pos[0],
                initial_stage_pos[1],
            )

            # Capture overview image
            logger.info("Capturing bottom camera overview...")
            yield from bps.trigger_and_read([bottom_camera], name="overview")

            # Get overview image from device
            overview_image = bottom_camera._last_image

            if overview_image is None:
                raise RuntimeError("Failed to capture overview image!")

            # Save overview image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            overview_path = image_dir / f"initial_view_{timestamp}.png"
            import tifffile

            tifffile.imwrite(overview_path, overview_image)
            logger.info("Saved overview: %s", overview_path.name)

            logger.info("Launching interactive embryo marking...")

            # Prefer web-based marking (no Qt dependency, works remotely)
            viz_server = kwargs.get("viz_server")
            if viz_server is not None:
                import asyncio

                logger.info("Using web-based marking via visualization server")
                marked_embryos = asyncio.get_event_loop().run_until_complete(
                    mark_embryos_web(
                        viz_server=viz_server,
                        image=overview_image,
                        initial_stage_position=tuple(initial_stage_pos),
                        pixel_size_um=bottom_camera.effective_pixel_size,
                        save_image_path=image_dir / f"all_embryos_marked_{timestamp}.png",
                    )
                )
            else:
                # napari has been retired; the web map view is the only
                # interactive marking surface. Start the viz server before
                # invoking this plan.
                logger.error(
                    "No viz_server available. The web map view is the only "
                    "interactive marking surface (napari has been retired). "
                    "Start the visualization server before calling this plan."
                )
                raise RuntimeError("Interactive marking requires the web visualization server.")

        if len(marked_embryos) == 0:
            logger.error("No embryos marked! Aborting session.")
            # Return empty path - can't just return None in a generator
            return Path("no_database_created.json")

        logger.info("Total embryos to calibrate: %d", len(marked_embryos))

        # ====================================================================
        # PHASE 2: CALIBRATE EACH EMBRYO
        # ====================================================================
        logger.info("[PHASE 2] Calibrating each embryo...")

        calibrated_embryos = []

        for i, embryo_data in enumerate(marked_embryos, 1):
            embryo_id = embryo_data["embryo_id"]

            logger.info("-" * 70)
            logger.info("EMBRYO %d/%d: %s", i, len(marked_embryos), embryo_id)
            logger.info("-" * 70)

            # Center stage on embryo
            embryo_data = yield from center_and_verify_embryo_plan(
                bottom_camera=bottom_camera,
                xy_stage=xy_stage,
                embryo_data=embryo_data,
                save_verification=True,
                image_dir=image_dir,
            )

            # Check if centering failed (position out of bounds)
            if "error" in embryo_data:
                logger.warning("Skipping calibration for %s due to centering error", embryo_id)
                calibrated_embryos.append(embryo_data)
                continue

            # Run calibration for this embryo
            embryo_data = yield from calibrate_single_embryo_in_session_plan(
                embryo_data=embryo_data,
                embryo_detector=embryo_detector,
                laser_control=laser_control,
                image_dir=image_dir / embryo_id,
                calibration_params=calibration_params,
            )

            calibrated_embryos.append(embryo_data)

            # TODO: Get run UID from this embryo's calibration
            # For now, just track embryo IDs
            # embryo_run_uids.append(embryo_run_uid)

        # ====================================================================
        # PHASE 3: EXPORT TO JSON DATABASE
        # ====================================================================
        logger.info("[PHASE 3] Exporting results to JSON database...")

        # Build database structure
        database = {
            "created": session_metadata["session_start"],
            "embryos": {},
            "last_updated": datetime.now().isoformat(),
        }

        for embryo_data in calibrated_embryos:
            embryo_id = embryo_data["embryo_id"]
            database = add_embryo_to_database(database, embryo_id, embryo_data)

        # Save JSON database
        output_database_path_resolved = Path(output_database_path)
        save_multi_embryo_database(database, output_database_path_resolved)

        logger.info("Exported database: %s", output_database_path_resolved)
        logger.info("Total embryos: %d", len(calibrated_embryos))
        logger.info(
            "Successful calibrations: %d",
            sum(1 for e in calibrated_embryos if e.get("calibration") is not None),
        )

        # ====================================================================
        # SESSION COMPLETE
        # ====================================================================
        logger.info("=" * 70)
        logger.info("MULTI-EMBRYO CALIBRATION SESSION COMPLETE")
        logger.info("=" * 70)
        logger.info("Database: %s", output_database_path_resolved)
        logger.info("Images: %s/", image_dir)
        logger.info("Total embryos: %d", len(calibrated_embryos))

        return output_database_path_resolved

    # Run the session plan WITHOUT run_wrapper to avoid nested run conflicts
    # Each embryo calibration creates its own runs internally
    # Session metadata is stored in the database, not as a Bluesky run
    result = yield from inner_session_plan()
    return result
