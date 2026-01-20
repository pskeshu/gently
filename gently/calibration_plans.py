"""
Bluesky Plans for diSPIM Calibration
=====================================

Complete calibration workflow plans for embryo-based piezo-galvo calibration.

Plans included:
- verify_embryo_centered: Check if embryo is visible at center
- detect_embryo_edge: Find where embryo disappears (top or bottom)
- calibrate_focus_at_position: Perform focus sweep at a galvo position
- calibrate_embryo_piezo_galvo: Full calibration workflow orchestration

All plans are device-agnostic and work with standard Bluesky RunEngine.
"""

import time
import rpyc
import bluesky.plan_stubs as bps
import numpy as np
from pathlib import Path
from datetime import datetime


# ============================================================================
# CLAUDE VISION PROMPTS
# ============================================================================

EMBRYO_CENTERING_PROMPT = """You are an expert microscopist examining a diSPIM light sheet microscopy image of a biological embryo sample.

This image shows ONE camera view from the diSPIM system. You should look for an embryo structure somewhere in the field of view. The embryo will appear as a brighter structure against a dark background, but the signal may be MODERATE (not necessarily super bright).

IMPORTANT CONTEXT:
- This is a REAL microscopy image with typical noise and artifacts
- Room lighting may cause some background glow (this is normal - ignore it)
- Embryos appear as irregularly-shaped bright regions, NOT perfectly uniform
- The embryo does NOT need to be perfectly centered, just reasonably visible in the frame
- Signal levels are moderate - don't expect extremely bright fluorescence

YOUR TASK:
Determine if an embryo structure is visible in this image that we can use for calibration.

WHAT TO LOOK FOR (be forgiving):
✓ EMBRYO VISIBLE:
  - ANY distinct structure brighter than the background (doesn't need to be super bright)
  - Some defined boundary or edge (even if irregular or somewhat soft)
  - Structure appears biological (rounded, irregular, or oblong shape)
  - Structure is not cut off at the edge of the frame
  - You can distinguish the embryo from background even if contrast is moderate

✗ NO USABLE EMBRYO:
  - Absolutely no structure visible, only uniform background
  - Frame is completely empty or saturated
  - No contrast at all between any structures and background
  - Any visible structure is completely cut off at frame edge

RESPOND FORMAT:
Line 1: "yes" if ANY embryo structure is visible that we can work with, "no" only if truly absent
Line 2: Brief description of what you see (1-2 sentences)

Example response:
yes
An irregularly-shaped embryo structure is visible in the left-center region with moderate brightness and defined boundaries against the dark background."""

EMBRYO_EDGE_PROMPT = """You are an expert microscopist specializing in diSPIM light sheet microscopy of embryos.

This image shows ONE camera view of an embryo captured with light sheet illumination. We are trying to determine if the embryo is still visible at this Z position.

CONTEXT:
- We are sweeping through Z positions to find where the embryo appears/disappears
- This may be at the edge of the embryo where it starts to fade out
- We need to detect even faint/sparse embryo signal

YOUR TASK:
Determine if there is ANY embryo structure visible in this image, even if faint or sparse.

WHAT COUNTS AS VISIBLE:
✓ YES (embryo visible):
  - Any distinct embryo structure, even if faint
  - Partial embryo at edge (sparse but present)
  - Moderate contrast showing biological structure
  - Even if only a small portion is visible

✗ NO (embryo not visible):
  - Completely empty/uniform background
  - Only noise and artifacts, no structure
  - Embryo has completely disappeared

RESPOND FORMAT:
Line 1: "yes" if embryo is visible (even faintly), "no" if completely absent
Line 2: Brief description

Example:
yes
Faint embryo structure visible in center, appears to be at edge of sample."""


# ============================================================================
# PLAN: VERIFY EMBRYO CENTERED
# ============================================================================

def verify_embryo_centered(embryo_detector, image_dir=None):
    """
    Verify that embryo is centered and visible.

    This is Phase 1 of the calibration workflow. Checks that the embryo
    is visible at the center position (galvo=0°, piezo=0µm) before
    proceeding with edge detection and calibration.

    Parameters
    ----------
    embryo_detector : DiSPIMEmbryoDetector
        Composite device with camera, galvo, piezo, and Claude
    image_dir : Path, optional
        Directory to save calibration images (default: None, uses temp)

    Yields
    ------
    Bluesky messages for run control and data collection

    Returns
    -------
    bool
        True if embryo is centered and visible, False otherwise

    Examples
    --------
    >>> from bluesky import RunEngine
    >>> RE = RunEngine({})
    >>> centered = RE(verify_embryo_centered(detector))
    >>> print(f"Embryo centered: {centered}")
    """
    print(f"\n{'='*70}")
    print("PHASE 1: CENTERING VERIFICATION")
    print(f"{'='*70}\n")

    # Metadata for this phase
    metadata = {
        'plan_name': 'verify_embryo_centered',
        'phase': 'centering',
        'timestamp': datetime.now().isoformat()
    }

    # Start run
    uid = yield from bps.open_run(md=metadata)

    # Prepare image path
    if image_dir is not None:
        image_path = Path(image_dir) / f"centering_check_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    else:
        image_path = None

    # Check embryo at center position
    print("  Checking embryo at center (galvo=0.0°, piezo=0.0µm)...")

    result = embryo_detector.check_embryo_at_position(
        galvo_deg=0.0,
        piezo_um=0.0,
        prompt=EMBRYO_CENTERING_PROMPT,
        save_image_path=image_path
    )

    # Log result
    yield from bps.create()
    yield from bps.read(embryo_detector)
    yield from bps.save()

    # Close run
    yield from bps.close_run()

    # Report result
    if result['embryo_visible']:
        print(f"  ✓ Embryo VISIBLE at center")
        print(f"    Claude: {result['description']}")
        print(f"    Confidence: {result['confidence']:.1%}")
        if image_path:
            print(f"    Image: {image_path}")
        return True
    else:
        print(f"  ✗ Embryo NOT VISIBLE at center")
        print(f"    Claude: {result['description']}")
        print(f"  ⚠ Please adjust sample position and try again")
        return False


# ============================================================================
# PLAN: DETECT EMBRYO EDGE
# ============================================================================

def detect_embryo_edge(embryo_detector, direction='top',
                       start_deg=0.0, end_deg=0.5, step_deg=0.05,
                       tolerance_deg=0.20, piezo_um=0.0,
                       image_dir=None):
    """
    Detect embryo edge by sweeping until embryo disappears.

    This is Phase 1.5 of calibration. Sweeps from center outward (top or bottom)
    to find where embryo first disappears, then adds tolerance margin for
    volume scan range.

    Strategy:
    - Start at center (where embryo is confirmed visible)
    - Step outward until Claude reports embryo not visible
    - Add tolerance margin beyond edge
    - Return edge position and scan boundary

    Parameters
    ----------
    embryo_detector : DiSPIMEmbryoDetector
        Composite device for embryo detection
    direction : str
        'top' (sweep negative) or 'bottom' (sweep positive)
    start_deg : float
        Starting galvo position in degrees (default: 0.0, center)
    end_deg : float
        Ending galvo position in degrees (default: 0.5)
    step_deg : float
        Step size in degrees (default: 0.05, ~5µm)
    tolerance_deg : float
        Tolerance margin beyond edge (default: 0.20, ~20µm)
    piezo_um : float
        Piezo position to use during sweep (default: 0.0)
    image_dir : Path, optional
        Directory to save sweep images

    Yields
    ------
    Bluesky messages

    Returns
    -------
    dict
        Edge detection result with keys:
        - 'edge_deg': Position where embryo disappears
        - 'with_tolerance_deg': Edge position + tolerance
        - 'num_steps': Number of positions tested
        - 'all_positions': List of tested positions
        - 'all_visible': List of visibility results

    Examples
    --------
    >>> # Detect top edge (sweep upward from center)
    >>> top = RE(detect_embryo_edge(detector, direction='top',
    ...                              start_deg=0.0, end_deg=-0.5))
    >>> print(f"Top edge: {top['edge_deg']:.3f}°")
    >>> print(f"Scan boundary: {top['with_tolerance_deg']:.3f}°")
    """
    print(f"\n{'='*70}")
    print(f"PHASE 1.5: DETECT {direction.upper()} EDGE")
    print(f"{'='*70}\n")

    # Determine sweep direction
    if direction == 'top':
        # Sweep upward (negative direction)
        step = -abs(step_deg)
        tolerance_sign = -1
    else:
        # Sweep downward (positive direction)
        step = abs(step_deg)
        tolerance_sign = +1

    # Generate positions
    num_steps = int(abs(end_deg - start_deg) / abs(step)) + 1
    positions = [start_deg + i * step for i in range(num_steps)]

    print(f"  Sweep strategy: Start at {start_deg:+.3f}°, step {step:+.3f}°")
    print(f"  Testing {num_steps} positions from {start_deg:+.3f}° to {end_deg:+.3f}°")
    print(f"  Looking for position where embryo disappears...\n")

    # Metadata
    metadata = {
        'plan_name': 'detect_embryo_edge',
        'phase': f'edge_detection_{direction}',
        'direction': direction,
        'start_deg': start_deg,
        'end_deg': end_deg,
        'step_deg': step,
        'tolerance_deg': tolerance_deg,
        'piezo_um': piezo_um,
        'timestamp': datetime.now().isoformat()
    }

    uid = yield from bps.open_run(md=metadata)

    # Sweep through positions
    all_results = []
    edge_found = False
    edge_position = None

    for i, pos in enumerate(positions):
        # Prepare image path
        if image_dir is not None:
            image_path = Path(image_dir) / f"edge_{direction}_pos{pos:+.3f}deg_{datetime.now().strftime('%H%M%S')}.png"
        else:
            image_path = None

        # Check embryo at this position
        print(f"  [{i+1}/{num_steps}] Position {pos:+.3f}°...", end=" ")

        result = embryo_detector.check_embryo_at_position(
            galvo_deg=pos,
            piezo_um=piezo_um,
            prompt=EMBRYO_EDGE_PROMPT,
            save_image_path=image_path
        )

        all_results.append(result)

        # Log to databroker
        yield from bps.create()
        yield from bps.read(embryo_detector)
        yield from bps.save()

        # Check if embryo disappeared
        if result['embryo_visible']:
            print(f"✓ visible")
        else:
            print(f"✗ NOT visible - EDGE FOUND!")
            edge_found = True
            edge_position = pos
            break

    if not edge_found:
        print(f"\n  ⚠ WARNING: Embryo still visible at end of sweep!")
        print(f"  Using last position as edge: {positions[-1]:+.3f}°")
        edge_position = positions[-1]

    # Add tolerance
    edge_with_tolerance = edge_position + (tolerance_sign * tolerance_deg)

    print(f"\n  {'─'*68}")
    print(f"  EDGE DETECTION COMPLETE")
    print(f"  {'─'*68}")
    print(f"    Detected edge: {edge_position:+.3f}° (embryo disappears here)")
    print(f"    Tolerance margin: {tolerance_deg:.3f}° (~{tolerance_deg*100:.0f} µm)")
    print(f"    Scan boundary: {edge_with_tolerance:+.3f}° (edge + tolerance)")
    print(f"    Positions tested: {len(all_results)}")

    # Prepare result
    edge_result = {
        'edge_deg': edge_position,
        'with_tolerance_deg': edge_with_tolerance,
        'num_steps': len(all_results),
        'all_positions': [r['galvo_deg'] for r in all_results],
        'all_visible': [r['embryo_visible'] for r in all_results]
    }

    yield from bps.close_run()

    return edge_result


# ============================================================================
# PLAN: CALIBRATE FOCUS AT POSITION
# ============================================================================

def calibrate_focus_at_position(camera, galvo, piezo, focus_scorer, core,
                                 galvo_deg, piezo_center_um,
                                 sweep_range_um=20.0, sweep_step_um=2.0,
                                 min_r_squared=0.75, image_dir=None,
                                 phase_name="FOCUS CALIBRATION"):
    """
    Perform focus sweep at a galvo position to find optimal piezo position.

    This is Phase 2/3 of calibration. Sweeps piezo through Z positions at a
    fixed galvo angle, scores each image with FFT bandpass, fits Gaussian curve,
    and finds optimal focus position.

    Workflow:
    1. Move galvo to specified position
    2. Sweep piezo from center-range to center+range
    3. At each position:
       - Snap image
       - Calculate FFT bandpass focus score
       - Store (position, score)
    4. Fit Gaussian curve to scores
    5. Validate R² >= threshold
    6. Return optimal position

    Parameters
    ----------
    camera : device
        Camera device for image capture
    galvo : device
        Galvo Y-axis device
    piezo : device
        Piezo Z-stage device
    focus_scorer : DiSPIMFocusScorer
        Focus scoring device with FFT bandpass
    core : pymmcore.CMMCore
        Micro-Manager core
    galvo_deg : float
        Galvo Y angle for this calibration point
    piezo_center_um : float
        Center of piezo sweep range
    sweep_range_um : float
        ± range around center (default: 20µm)
    sweep_step_um : float
        Step size (default: 2µm)
    min_r_squared : float
        Minimum R² for acceptable fit (default: 0.75)
    image_dir : Path, optional
        Directory to save focus sweep images
    phase_name : str
        Phase name for logging (default: "FOCUS CALIBRATION")

    Yields
    ------
    Bluesky messages

    Returns
    -------
    dict
        Focus calibration result with keys:
        - 'success': bool
        - 'optimal_position_um': float
        - 'r_squared': float
        - 'all_positions': list
        - 'all_scores': list
        - 'galvo_deg': float

    Examples
    --------
    >>> result = RE(calibrate_focus_at_position(
    ...     camera, galvo, piezo, scorer, core,
    ...     galvo_deg=-0.2,
    ...     piezo_center_um=-20.0
    ... ))
    >>> print(f"Best focus: {result['optimal_position_um']:.2f} µm")
    >>> print(f"R²: {result['r_squared']:.3f}")
    """
    print(f"\n{'='*70}")
    print(f"{phase_name}")
    print(f"{'='*70}\n")
    print(f"  Galvo position: {galvo_deg:+.3f}°")
    print(f"  Piezo sweep: {piezo_center_um:.1f} ± {sweep_range_um:.1f} µm")
    print(f"  Step size: {sweep_step_um:.2f} µm")

    # Generate sweep positions
    piezo_start = piezo_center_um - sweep_range_um
    piezo_end = piezo_center_um + sweep_range_um
    num_steps = int(2 * sweep_range_um / sweep_step_um) + 1
    positions = np.linspace(piezo_start, piezo_end, num_steps)

    print(f"  Total positions: {num_steps}\n")

    # Metadata
    metadata = {
        'plan_name': 'calibrate_focus_at_position',
        'phase': phase_name,
        'galvo_deg': galvo_deg,
        'piezo_center_um': piezo_center_um,
        'sweep_range_um': sweep_range_um,
        'sweep_step_um': sweep_step_um,
        'timestamp': datetime.now().isoformat()
    }

    uid = yield from bps.open_run(md=metadata)

    # Move galvo to position
    galvo.setPosition(float(galvo_deg))
    core.waitForDevice(galvo.device_name)
    time.sleep(0.2)

    # Perform focus sweep
    all_positions = []
    all_scores = []
    all_images = []
    embryo_roi = None

    for i, pos in enumerate(positions):
        # Move piezo
        piezo.setPosition(float(pos))
        core.waitForDevice(piezo.device_name)
        time.sleep(0.1)

        # Snap image
        core.snapImage()
        img = core.getImage()

        # Transfer RPyC netref to local numpy array using rpyc.classic.obtain()
        # This is the only way to properly transfer numpy arrays across RPyC boundary
        img = rpyc.classic.obtain(img)

        # DiSPIM captures two views side-by-side (Path A and Path B)
        # Select the brighter view for focus scoring
        h, w = img.shape
        mid_x = w // 2

        # Split into left and right halves
        left_view = img[:, :mid_x]
        right_view = img[:, mid_x:]

        # Calculate mean intensity for each view
        left_intensity = np.mean(left_view)
        right_intensity = np.mean(right_view)

        # Select the brighter view (better signal for FFT analysis)
        if left_intensity >= right_intensity:
            img = left_view
            view_name = "left"
        else:
            img = right_view
            view_name = "right"

        if pos == positions[0]:  # Only print once at start
            print(f"    Using {view_name} camera view (L:{left_intensity:.1f} vs R:{right_intensity:.1f})")

        # Store image for ROI detection
        all_images.append(img)

        # Detect embryo ROI from center image (most likely to have good visibility)
        if i == len(positions) // 2 and embryo_roi is None:
            print(f"\n    [ROI DETECTION] Detecting embryo region from center image...")
            embryo_roi = focus_scorer.detect_embryo_roi(img)
            y_min, y_max, x_min, x_max = embryo_roi
            roi_height = y_max - y_min
            roi_width = x_max - x_min
            roi_percent = (roi_width * roi_height) / (img.shape[0] * img.shape[1]) * 100
            print(f"    ✓ Embryo ROI: [{y_min}:{y_max}, {x_min}:{x_max}] "
                  f"({roi_width}x{roi_height} px, {roi_percent:.1f}% of frame)")

        # Convert to 8-bit with auto-scaling for better visibility
        # This matches the working calibrate_embryo_piezo_galvo.py behavior
        if img.dtype != np.uint8:
            if img.max() > 255:
                img_8bit = (img / img.max() * 255).astype(np.uint8)
            else:
                img_8bit = img.astype(np.uint8)
        else:
            img_8bit = img

        # Save image if requested
        if image_dir is not None:
            from PIL import Image
            image_path = Path(image_dir) / f"focus_pos{pos:.1f}um_{datetime.now().strftime('%H%M%S')}.png"
            image_path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(img_8bit).save(image_path)

        # Calculate focus score (use ROI if detected)
        score = focus_scorer.score_image(img, roi=embryo_roi)

        all_positions.append(pos)
        all_scores.append(score)

        # Log to databroker
        yield from bps.create()
        yield from bps.read(focus_scorer)
        yield from bps.save()

        print(f"  [{i+1}/{num_steps}] z={pos:+6.1f} µm → score={score:.2e}")

    # Fit Gaussian curve
    print(f"\n  Fitting Gaussian curve to focus scores...")
    fit_result = focus_scorer.fit_focus_curve(all_positions, all_scores)

    if fit_result['success']:
        print(f"  ✓ Fit successful!")
        print(f"    Best focus: {fit_result['best_position']:.2f} µm")
        print(f"    R²: {fit_result['r_squared']:.3f} ({fit_result['fit_quality']})")
        print(f"    Peak in center: {fit_result['peak_in_center']}")

        result = {
            'success': True,
            'optimal_position_um': fit_result['best_position'],
            'r_squared': fit_result['r_squared'],
            'all_positions': all_positions,
            'all_scores': all_scores,
            'galvo_deg': galvo_deg,
            'fit_params': fit_result['params']
        }
    else:
        print(f"  ✗ Fit failed: {fit_result.get('error_message', 'Unknown error')}")
        print(f"    R²: {fit_result['r_squared']:.3f} (threshold: {min_r_squared:.3f})")
        print(f"  ⚠ Using maximum score position as fallback")

        max_idx = np.argmax(all_scores)
        fallback_position = all_positions[max_idx]

        result = {
            'success': False,
            'optimal_position_um': fallback_position,
            'r_squared': fit_result['r_squared'],
            'all_positions': all_positions,
            'all_scores': all_scores,
            'galvo_deg': galvo_deg,
            'error_message': fit_result.get('error_message', 'Poor fit quality')
        }

    yield from bps.close_run()

    return result


# ============================================================================
# PLAN: FULL CALIBRATION ORCHESTRATION
# ============================================================================

def calibrate_embryo_piezo_galvo(
    camera, galvo, piezo, focus_scorer, embryo_detector, core,
    calibration_inset_fraction=0.4,
    edge_detection_step_deg=0.05,
    edge_tolerance_deg=0.20,
    sweep_range_um=20.0,
    sweep_step_um=2.0,
    min_r_squared=0.75,
    heuristic_slope=100.0,
    heuristic_offset=0.0,
    image_dir=None,
    save_path=None
):
    """
    Complete embryo-based piezo-galvo calibration workflow.

    Orchestrates all calibration phases:
    1. Verify embryo centered (Phase 1)
    2. Detect top edge (Phase 1.5)
    3. Detect bottom edge (Phase 1.5)
    4. Calculate interior calibration positions (inset from edges)
    5. Calibrate focus at TOP interior position (Phase 2)
    6. Calibrate focus at BOTTOM interior position (Phase 3)
    7. Calculate 2-point linear calibration (Phase 4)
    8. Save calibration file

    This replaces the standalone calibrate_embryo_piezo_galvo.py script
    with a proper Bluesky plan that captures all data in databroker.

    Parameters
    ----------
    camera : device
        Camera device for image capture
    galvo : device
        Galvo Y-axis device
    piezo : device
        Piezo Z-stage device
    focus_scorer : DiSPIMFocusScorer
        Focus scoring device with FFT bandpass
    embryo_detector : DiSPIMEmbryoDetector
        Composite device for embryo detection with Claude
    core : pymmcore.CMMCore
        Micro-Manager core instance
    calibration_inset_fraction : float
        Fraction to move inward from edges for calibration (default: 0.4)
    edge_detection_step_deg : float
        Step size for edge detection sweep (default: 0.05)
    edge_tolerance_deg : float
        Tolerance margin beyond edges (default: 0.20)
    sweep_range_um : float
        ± range for focus sweeps (default: 20.0 µm)
    sweep_step_um : float
        Step size for focus sweeps (default: 2.0 µm)
    min_r_squared : float
        Minimum R² for acceptable Gaussian fit (default: 0.75)
    heuristic_slope : float
        Initial slope estimate for piezo positioning (default: 100.0 µm/deg)
    heuristic_offset : float
        Initial offset estimate (default: 0.0 µm)
    image_dir : Path, optional
        Directory to save all calibration images
    save_path : Path, optional
        Path to save calibration JSON file

    Yields
    ------
    Bluesky messages

    Returns
    -------
    dict
        Calibration result (also stored in device for databroker access)

    Examples
    --------
    >>> from pathlib import Path
    >>> from gently.config import PiezoGalvoCalibration
    >>>
    >>> # Run full calibration
    >>> result = RE(calibrate_embryo_piezo_galvo(
    ...     camera, galvo, piezo, scorer, detector, core,
    ...     image_dir=Path("calibration_images"),
    ...     save_path=Path("piezo_galvo_calibration_embryo.json")
    ... ))
    >>>
    >>> # Access calibration
    >>> calib = PiezoGalvoCalibration.from_file("piezo_galvo_calibration_embryo.json")
    >>> print(f"Slope: {calib.slope_um_per_deg:.1f} µm/deg")
    """
    from gently.config import PiezoGalvoCalibration

    print(f"\n{'='*70}")
    print("EMBRYO-BASED PIEZO-GALVO CALIBRATION")
    print("Automated with Claude Vision + FFT Bandpass Focus")
    print(f"{'='*70}\n")

    # Overall metadata
    # Note: This orchestration plan does NOT create its own run.
    # Each sub-plan (verify_embryo_centered, detect_embryo_edge, calibrate_focus_at_position)
    # creates its own run in databroker. This allows each phase to be queried independently
    # and avoids nested run issues.

    try:
        # ====================================================================
        # PHASE 1: CENTERING VERIFICATION
        # ====================================================================
        print(f"[1/8] Phase 1: Centering Verification")
        print(f"{'─'*70}")

        centered = yield from verify_embryo_centered(embryo_detector, image_dir)

        if not centered:
            print(f"\n✗ CALIBRATION ABORTED: Embryo not centered")
            print(f"Please adjust sample position and try again.")
            return {'success': False, 'error': 'Embryo not centered'}

        # ====================================================================
        # PHASE 1.5: EDGE DETECTION
        # ====================================================================
        print(f"\n[2/8] Phase 1.5: Edge Detection")
        print(f"{'─'*70}")
        print(f"Detecting embryo extent to optimize scan range...\n")

        # Detect TOP edge (sweep upward from center)
        print(f"Detecting TOP edge (sweeping upward from center)...")
        top_edge_result = yield from detect_embryo_edge(
            embryo_detector,
            direction='top',
            start_deg=0.0,
            end_deg=-0.5,
            step_deg=edge_detection_step_deg,
            tolerance_deg=edge_tolerance_deg,
            piezo_um=0.0,
            image_dir=image_dir
        )

        edge_top_deg = top_edge_result['edge_deg']
        scan_top_deg = top_edge_result['with_tolerance_deg']

        # Detect BOTTOM edge (sweep downward from center)
        print(f"\nDetecting BOTTOM edge (sweeping downward from center)...")
        bottom_edge_result = yield from detect_embryo_edge(
            embryo_detector,
            direction='bottom',
            start_deg=0.0,
            end_deg=0.5,
            step_deg=edge_detection_step_deg,
            tolerance_deg=edge_tolerance_deg,
            piezo_um=0.0,
            image_dir=image_dir
        )

        edge_bottom_deg = bottom_edge_result['edge_deg']
        scan_bottom_deg = bottom_edge_result['with_tolerance_deg']

        detected_range = scan_bottom_deg - scan_top_deg

        print(f"\n{'─'*70}")
        print(f"EDGE DETECTION SUMMARY")
        print(f"{'─'*70}")
        print(f"  Detected edges:")
        print(f"    TOP: {edge_top_deg:+.3f}° (embryo starts to disappear)")
        print(f"    BOTTOM: {edge_bottom_deg:+.3f}° (embryo starts to disappear)")
        print(f"  Scan boundaries (with tolerance):")
        print(f"    TOP: {scan_top_deg:+.3f}° (includes {edge_tolerance_deg:.3f}° margin)")
        print(f"    BOTTOM: {scan_bottom_deg:+.3f}° (includes {edge_tolerance_deg:.3f}° margin)")
        print(f"  Total scan range: {detected_range:.3f}° (~{detected_range*100:.1f} µm)")

        # ====================================================================
        # CALCULATE INTERIOR CALIBRATION POSITIONS
        # ====================================================================
        print(f"\n[3/8] Calculate Interior Calibration Positions")
        print(f"{'─'*70}")
        print(f"Strategy: Calibrate at INTERIOR positions (not at edges)")
        print(f"Reason: Edges have sparse embryo → poor focus detection")
        print(f"        Interior has sharp morphology → accurate focus\n")

        # Move inward from scan boundaries (not edges!)
        scan_range = scan_bottom_deg - scan_top_deg
        inset_amount = scan_range * calibration_inset_fraction

        calib_top_deg = scan_top_deg + inset_amount
        calib_bottom_deg = scan_bottom_deg - inset_amount
        calib_range = calib_bottom_deg - calib_top_deg

        print(f"  Inset fraction: {calibration_inset_fraction*100:.0f}%")
        print(f"  Inset distance: {inset_amount:.3f}° (~{inset_amount*100:.1f} µm)")
        print(f"\n  TOP calibration position:")
        print(f"    Scan boundary: {scan_top_deg:+.3f}° → Calibrate at: {calib_top_deg:+.3f}°")
        print(f"  BOTTOM calibration position:")
        print(f"    Scan boundary: {scan_bottom_deg:+.3f}° → Calibrate at: {calib_bottom_deg:+.3f}°")
        print(f"\n  Calibration range: {calib_range:.3f}° (interior positions)")
        print(f"  Volume scan range: {detected_range:.3f}° (full detected extent)")

        # Estimate piezo positions using heuristic
        piezo_top_heuristic = calib_top_deg * heuristic_slope + heuristic_offset
        piezo_bottom_heuristic = calib_bottom_deg * heuristic_slope + heuristic_offset

        print(f"\n  Heuristic piezo positions (for focus sweep centers):")
        print(f"    TOP: {piezo_top_heuristic:.1f} µm (from heuristic calibration)")
        print(f"    BOTTOM: {piezo_bottom_heuristic:.1f} µm (from heuristic calibration)")

        # ====================================================================
        # PHASE 2: TOP CALIBRATION
        # ====================================================================
        print(f"\n[4/8] Phase 2: TOP Interior Focus Calibration")
        print(f"{'─'*70}")

        top_result = yield from calibrate_focus_at_position(
            camera, galvo, piezo, focus_scorer, core,
            galvo_deg=calib_top_deg,
            piezo_center_um=piezo_top_heuristic,
            sweep_range_um=sweep_range_um,
            sweep_step_um=sweep_step_um,
            min_r_squared=min_r_squared,
            image_dir=image_dir,
            phase_name="PHASE 2: TOP CALIBRATION"
        )

        if not top_result['success']:
            print(f"\n⚠ WARNING: TOP calibration fit quality poor (R²={top_result['r_squared']:.3f})")
            print(f"Using best score position as fallback")

        piezo_top_final = top_result['optimal_position_um']
        print(f"\n✓ TOP optimal position: {piezo_top_final:.2f} µm")

        # ====================================================================
        # PHASE 3: BOTTOM CALIBRATION
        # ====================================================================
        print(f"\n[5/8] Phase 3: BOTTOM Interior Focus Calibration")
        print(f"{'─'*70}")

        bottom_result = yield from calibrate_focus_at_position(
            camera, galvo, piezo, focus_scorer, core,
            galvo_deg=calib_bottom_deg,
            piezo_center_um=piezo_bottom_heuristic,
            sweep_range_um=sweep_range_um,
            sweep_step_um=sweep_step_um,
            min_r_squared=min_r_squared,
            image_dir=image_dir,
            phase_name="PHASE 3: BOTTOM CALIBRATION"
        )

        if not bottom_result['success']:
            print(f"\n⚠ WARNING: BOTTOM calibration fit quality poor (R²={bottom_result['r_squared']:.3f})")
            print(f"Using best score position as fallback")

        piezo_bottom_final = bottom_result['optimal_position_um']
        print(f"\n✓ BOTTOM optimal position: {piezo_bottom_final:.2f} µm")

        # ====================================================================
        # PHASE 4: CALCULATE CALIBRATION
        # ====================================================================
        print(f"\n[6/8] Phase 4: Calculate 2-Point Linear Calibration")
        print(f"{'─'*70}")

        # Calculate slope and offset from 2 points
        delta_galvo = calib_bottom_deg - calib_top_deg
        delta_piezo = piezo_bottom_final - piezo_top_final

        if abs(delta_galvo) < 0.001:
            raise ValueError("Galvo positions too close for calibration")

        slope = delta_piezo / delta_galvo
        offset = piezo_top_final - slope * calib_top_deg

        print(f"  Calibration points:")
        print(f"    TOP:    galvo={calib_top_deg:+.3f}°, piezo={piezo_top_final:+.2f} µm")
        print(f"    BOTTOM: galvo={calib_bottom_deg:+.3f}°, piezo={piezo_bottom_final:+.2f} µm")
        print(f"\n  Linear fit: piezo(µm) = {slope:.2f} × galvo(deg) + {offset:.2f}")
        print(f"    Slope: {slope:.2f} µm/deg")
        print(f"    Offset: {offset:.2f} µm")

        # Calculate full range positions (for volume scanning)
        piezo_top_scan = scan_top_deg * slope + offset
        piezo_bottom_scan = scan_bottom_deg * slope + offset

        print(f"\n  Volume scan range (applying calibration to scan boundaries):")
        print(f"    TOP:    galvo={scan_top_deg:+.3f}°, piezo={piezo_top_scan:+.2f} µm")
        print(f"    BOTTOM: galvo={scan_bottom_deg:+.3f}°, piezo={piezo_bottom_scan:+.2f} µm")
        print(f"    Total: {abs(piezo_bottom_scan - piezo_top_scan):.1f} µm")

        # ====================================================================
        # CREATE CALIBRATION OBJECT
        # ====================================================================
        print(f"\n[7/8] Create Calibration Object")
        print(f"{'─'*70}")

        calibration = PiezoGalvoCalibration(
            slope_um_per_deg=slope,
            offset_um=offset,
            galvo_top_deg=scan_top_deg,
            galvo_bottom_deg=scan_bottom_deg,
            piezo_top_um=piezo_top_scan,
            piezo_bottom_um=piezo_bottom_scan,
            edge_top_deg=edge_top_deg,
            edge_bottom_deg=edge_bottom_deg,
            sample_type='embryo',
            timestamp=datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        )

        print(f"  ✓ Calibration object created")
        print(f"  {calibration}")

        # ====================================================================
        # SAVE CALIBRATION
        # ====================================================================
        if save_path is not None:
            print(f"\n[8/8] Save Calibration")
            print(f"{'─'*70}")

            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)

            calibration.to_file(save_path)

            print(f"  ✓ Saved to: {save_path.absolute()}")
            print(f"  Format: JSON with complete metadata")

        # ====================================================================
        # SUCCESS SUMMARY
        # ====================================================================
        print(f"\n{'='*70}")
        print("✓ CALIBRATION SUCCESSFUL")
        print(f"{'='*70}")
        print(f"\nResults:")
        print(f"  Calibration: piezo(µm) = {slope:.2f} × galvo(deg) + {offset:.2f}")
        print(f"  Scan range: {scan_top_deg:+.3f}° to {scan_bottom_deg:+.3f}°")
        print(f"  Piezo range: {piezo_top_scan:+.2f} to {piezo_bottom_scan:+.2f} µm")
        print(f"  Quality:")
        print(f"    TOP fit R²: {top_result['r_squared']:.3f}")
        print(f"    BOTTOM fit R²: {bottom_result['r_squared']:.3f}")

        if save_path:
            print(f"\nCalibration saved to: {save_path}")
            print(f"\nNext step:")
            print(f"  python test_embryo_bluesky_plan.py --tolerance 1.5")

        # Store result for databroker access
        result = {
            'success': True,
            'calibration': calibration,
            'slope_um_per_deg': slope,
            'offset_um': offset,
            'top_r_squared': top_result['r_squared'],
            'bottom_r_squared': bottom_result['r_squared'],
            'scan_top_deg': scan_top_deg,
            'scan_bottom_deg': scan_bottom_deg,
            'edge_top_deg': edge_top_deg,
            'edge_bottom_deg': edge_bottom_deg
        }

    except Exception as e:
        print(f"\n{'='*70}")
        print("✗ CALIBRATION FAILED")
        print(f"{'='*70}")
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

        result = {
            'success': False,
            'error': str(e)
        }

    return result


# ============================================================================
# EXPORTS
# ============================================================================

__all__ = [
    'EMBRYO_CENTERING_PROMPT',
    'EMBRYO_EDGE_PROMPT',
    'verify_embryo_centered',
    'detect_embryo_edge',
    'calibrate_focus_at_position',
    'calibrate_embryo_piezo_galvo'
]
