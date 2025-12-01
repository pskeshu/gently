"""
Hardware Control Tools

Tools for controlling microscope hardware including stage movement,
LED control, calibration, and image acquisition.
"""

from typing import Dict, List
import json
import asyncio

from ..tool_registry import tool, ToolCategory, ToolExample
from ..tool_helpers import require_copilot, get_embryo_or_error
from gently.coordinates import stage_to_pixel_position, get_um_per_pixel


@tool(
    name="move_to_embryo",
    description="""Move the XY stage to a specific embryo's stored position. The embryo must have been detected and have a valid stage_position.
Use when user says "go to embryo X", "move to embryo X", or before imaging a specific embryo.
This only moves XY - piezo/galvo are controlled separately during acquisition. Movement takes ~0.5 seconds.""",
    category=ToolCategory.MOVEMENT,
    requires_microscope=True,
    examples=[
        ToolExample("Go to embryo 1", {"embryo_id": "embryo_1"}),
        ToolExample("Move to embryo 3", {"embryo_id": "embryo_3"}),
    ],
)
async def move_to_embryo(embryo_id: str, context: Dict) -> str:
    """Move stage to embryo position"""
    copilot = context.get('copilot')
    client = context.get('client')

    if not copilot:
        return "Error: No copilot context"

    embryo, err = get_embryo_or_error(copilot, embryo_id)
    if err:
        return err

    if not embryo.stage_position:
        return f"Embryo '{embryo_id}' has no stored position. Run calibration first."

    try:
        x = embryo.stage_position.get('x', 0)
        y = embryo.stage_position.get('y', 0)
        await client.move_to_position(x, y)

        return f"Moved to {embryo_id}\nPosition: ({x:.2f}, {y:.2f}) um"

    except Exception as e:
        import traceback
        return f"Error moving to embryo: {str(e)}\n{traceback.format_exc()}"


@tool(
    name="get_stage_position",
    description="""Get the current XY stage position in micrometers. Returns the real-time position from the hardware.
Use when user asks "where is the stage?", "current position?", or when you need to know the microscope's current location.
This reads from hardware - different from embryo stored positions which are in the experiment data.""",
    category=ToolCategory.HARDWARE,
    requires_microscope=True,
    examples=[
        ToolExample("Where is the stage?", {}),
        ToolExample("Current XY position?", {}),
    ],
)
async def get_stage_position(context: Dict) -> str:
    """Get current stage position"""
    client = context.get('client')

    if not client:
        return "Error: No microscope client connected"

    try:
        pos = await client.get_stage_position()
        return f"Current stage position: X={pos[0]:.1f} µm, Y={pos[1]:.1f} µm"

    except Exception as e:
        return f"Error reading stage position: {str(e)}"


@tool(
    name="move_stage",
    description="""Move the XY stage to specific coordinates in micrometers.
Use when user wants to move to arbitrary coordinates (e.g., "move to x=1000, y=500", "move stage to 1200, -600").
For moving to a specific embryo, use move_to_embryo instead.""",
    category=ToolCategory.HARDWARE,
    requires_microscope=True,
    examples=[
        ToolExample("Move to x=1000, y=500", {"x": 1000, "y": 500}),
        ToolExample("Move stage to coordinates 1200, -600", {"x": 1200, "y": -600}),
    ],
)
async def move_stage(
    x: float,
    y: float,
    context: Dict = None
) -> str:
    """Move stage to arbitrary XY coordinates"""
    client = context.get('client')

    if not client:
        return "Error: No microscope client connected"

    try:
        await client.move_to_position(x=x, y=y)
        pos = await client.get_stage_position()
        return f"Moved to X={pos[0]:.1f} µm, Y={pos[1]:.1f} µm"

    except Exception as e:
        return f"Error moving stage: {str(e)}"


@tool(
    name="set_led",
    description="Set the LED illumination state",
    category=ToolCategory.HARDWARE,
    requires_microscope=True,
)
async def set_led(state: str, context: Dict) -> str:
    """Set LED state"""
    client = context.get('client')

    try:
        result = await client.set_led(state)
        if result.get('success'):
            return f"LED set to '{state}'"
        else:
            return f"Error setting LED: {result.get('error', 'Unknown error')}"
    except Exception as e:
        return f"Error setting LED: {str(e)}"


@tool(
    name="get_led_status",
    description="Get the current LED illumination status",
    category=ToolCategory.HARDWARE,
    requires_microscope=True,
)
async def get_led_status(context: Dict) -> str:
    """Get LED status"""
    client = context.get('client')

    try:
        result = await client.get_led_status()
        if result.get('success'):
            current = result.get('current_state', 'unknown')
            available = result.get('available_configs', [])
            group = result.get('group_name', 'unknown')

            return (f"LED Status:\n"
                    f"  Current state: {current}\n"
                    f"  ConfigGroup: {group}\n"
                    f"  Available configs: {available}")
        else:
            return f"Error getting LED status: {result.get('error', 'Unknown error')}"
    except Exception as e:
        return f"Error getting LED status: {str(e)}"


@tool(
    name="calibrate_embryo",
    description="""Run full piezo-galvo calibration for a specific embryo using Claude vision.
This performs:
1. Move to embryo XY position
2. Use Claude vision to detect embryo Z extent (top/bottom edges)
3. Two-stage focus sweeps at interior positions (coarse ±20µm then fine ±5µm)
4. FFT bandpass scoring with Gaussian fit (R² ≥ 0.75 threshold)
5. 2-point linear fit to establish piezo = slope*galvo + offset
6. Store calibration including volume acquisition parameters

Use after detection to prepare an embryo for volume acquisition. Takes ~3-5 minutes per embryo.""",
    category=ToolCategory.CALIBRATION,
    requires_microscope=True,
    examples=[
        ToolExample("Calibrate embryo 1", {"embryo_id": "embryo_1"}),
        ToolExample("Skip edge detection", {"embryo_id": "embryo_2", "skip_edge_detection": True}),
    ],
)
async def calibrate_embryo(
    embryo_id: str,
    skip_edge_detection: bool = False,
    galvo_top: float = None,
    galvo_bottom: float = None,
    edge_step: float = 0.05,
    edge_max_range: float = 0.5,
    inset_fraction: float = 0.4,
    context: Dict = None
) -> str:
    """Run piezo-galvo calibration with Claude vision edge detection"""
    import numpy as np
    import tempfile
    from pathlib import Path
    from PIL import Image
    from gently.analysis.core import calculate_focus_score, fit_focus_curve, FitFunction
    from gently.claude_client import AsyncClaudeClient

    copilot = context.get('copilot')
    client = context.get('client')

    if not copilot:
        return "Error: No copilot context"

    embryo, err = get_embryo_or_error(copilot, embryo_id)
    if err:
        return err

    # Helper to select best camera view from dual-view diSPIM image
    def select_best_view(image: np.ndarray) -> np.ndarray:
        """Select brighter half from dual-view image (View A or B)"""
        if image.ndim != 2:
            return image
        h, w = image.shape
        if w < 100:  # Already single view or too small
            return image
        mid_x = w // 2
        left_view = image[:, :mid_x]
        right_view = image[:, mid_x:]
        # Select view with higher mean intensity (better signal)
        if np.mean(left_view) >= np.mean(right_view):
            return left_view
        return right_view

    # Heuristic calibration for edge detection sweeps
    # Use previous calibration if available, otherwise default 100 µm/deg
    if embryo.calibration and embryo.calibration.get('slope_um_per_deg'):
        HEURISTIC_SLOPE = embryo.calibration['slope_um_per_deg']
        HEURISTIC_OFFSET = embryo.calibration.get('offset_um', 0.0)
        print(f"  Using previous calibration as heuristic: {HEURISTIC_SLOPE:.1f} µm/deg, offset {HEURISTIC_OFFSET:.1f} µm")
    else:
        HEURISTIC_SLOPE = 100.0  # Default empirical value
        HEURISTIC_OFFSET = 0.0
        print(f"  Using default heuristic: {HEURISTIC_SLOPE:.1f} µm/deg")

    # Track total exposures during calibration
    total_exposures = 0

    try:
        # First move to embryo position
        pos = embryo.stage_position
        if pos and pos.get('x') is not None and pos.get('y') is not None:
            print(f"  Moving to {embryo_id} position...")
            await client.move_to_position(pos['x'], pos['y'])

        # Initialize Claude client for vision
        claude_vision = AsyncClaudeClient()

        # Helper to save image and check embryo presence
        async def check_embryo_at_position(galvo_pos: float) -> bool:
            """Capture image and ask Claude if embryo is visible"""
            nonlocal total_exposures
            piezo_pos = HEURISTIC_SLOPE * galvo_pos + HEURISTIC_OFFSET  # Track light sheet
            result = await client.capture_lightsheet_image(
                piezo_position=float(piezo_pos),
                galvo_position=float(galvo_pos)
            )
            if result.get('success'):
                total_exposures += 1
            if not result.get('success') or result.get('image') is None:
                return False

            img = result['image']
            # Select best view from dual-view image
            img = select_best_view(img)
            # Save to temp file for Claude
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                temp_path = Path(f.name)
                # Normalize and save
                img_norm = ((img - img.min()) / (img.max() - img.min() + 1e-10) * 255).astype(np.uint8)
                Image.fromarray(img_norm).save(temp_path)

            try:
                visible, description = await claude_vision.detect_embryo_presence(temp_path)
                print(f"    galvo={galvo_pos:+.3f}°: {'VISIBLE' if visible else 'EMPTY'} - {description[:50]}...")
                return visible
            finally:
                temp_path.unlink(missing_ok=True)

        # === PHASE 1: EDGE DETECTION (unless skipped) ===
        if skip_edge_detection:
            # Use provided galvo positions or defaults
            detected_top = galvo_top if galvo_top is not None else -0.15
            detected_bottom = galvo_bottom if galvo_bottom is not None else 0.15
            print(f"\n  Skipping edge detection, using galvo range: {detected_top:.3f}° to {detected_bottom:.3f}°")
        else:
            print(f"\n  Phase 1: Detecting embryo Z extent with Claude vision...")

            # Detect TOP edge (sweep from center toward negative)
            print(f"\n  Detecting TOP edge (sweeping galvo toward negative)...")
            detected_top = 0.0
            for galvo in np.arange(0.0, -edge_max_range - edge_step/2, -edge_step):
                visible = await check_embryo_at_position(galvo)
                if visible:
                    detected_top = galvo
                else:
                    print(f"    → Embryo disappeared at galvo={galvo:.3f}°")
                    break

            # Detect BOTTOM edge (sweep from center toward positive)
            print(f"\n  Detecting BOTTOM edge (sweeping galvo toward positive)...")
            detected_bottom = 0.0
            for galvo in np.arange(0.0, edge_max_range + edge_step/2, edge_step):
                visible = await check_embryo_at_position(galvo)
                if visible:
                    detected_bottom = galvo
                else:
                    print(f"    → Embryo disappeared at galvo={galvo:.3f}°")
                    break

            print(f"\n  Detected embryo extent:")
            print(f"    Top edge: galvo={detected_top:.3f}°")
            print(f"    Bottom edge: galvo={detected_bottom:.3f}°")
            print(f"    Range: {detected_bottom - detected_top:.3f}° (~{(detected_bottom - detected_top) * 100:.0f}µm)")

        # === PHASE 2: CALCULATE INTERIOR CALIBRATION POSITIONS ===
        # Don't calibrate at edges where embryo is sparse - move inward
        galvo_range = detected_bottom - detected_top
        calib_top = detected_top + galvo_range * inset_fraction
        calib_bottom = detected_bottom - galvo_range * inset_fraction

        print(f"\n  Calibration positions (interior, {inset_fraction*100:.0f}% inset from edges):")
        print(f"    Top calibration: galvo={calib_top:.3f}°")
        print(f"    Bottom calibration: galvo={calib_bottom:.3f}°")

        # === PHASE 3: TWO-STAGE FOCUS SWEEPS AT CALIBRATION POSITIONS ===
        # Parameters matching calibrate_embryo_piezo_galvo.py
        COARSE_RANGE = 20.0   # ±20µm
        COARSE_STEP = 2.0     # 2µm steps
        FINE_RANGE = 5.0      # ±5µm around coarse best
        FINE_STEP = 0.5       # 0.5µm steps
        MIN_R_SQUARED = 0.75  # Quality threshold
        EDGE_EXCLUSION = 0.20 # Reject if peak in outer 20%

        print(f"\n  Phase 2: Two-stage focus sweeps at calibration positions...")
        results = {}

        for galvo_name, galvo_pos in [("top", calib_top), ("bottom", calib_bottom)]:
            print(f"\n  === {galvo_name.upper()} FOCUS SWEEP at galvo={galvo_pos:.3f}° ===")

            # Expected piezo from heuristic
            expected_piezo = galvo_pos * HEURISTIC_SLOPE + HEURISTIC_OFFSET

            # --- STAGE 1: COARSE SWEEP ---
            print(f"  Stage 1: Coarse sweep ±{COARSE_RANGE}µm, {COARSE_STEP}µm steps...")
            coarse_positions = np.arange(
                expected_piezo - COARSE_RANGE,
                expected_piezo + COARSE_RANGE + COARSE_STEP,
                COARSE_STEP
            )

            coarse_scores = []
            coarse_valid = []

            for piezo in coarse_positions:
                result = await client.capture_lightsheet_image(
                    piezo_position=float(piezo),
                    galvo_position=float(galvo_pos)
                )
                if result.get('success'):
                    total_exposures += 1
                if result.get('success') and result.get('image') is not None:
                    img = select_best_view(result['image'])
                    score = calculate_focus_score(img, algorithm='fft_bandpass')
                    coarse_scores.append(score)
                    coarse_valid.append(piezo)

            if len(coarse_scores) < 4:
                return f"Error: Not enough coarse images at galvo={galvo_pos}"

            coarse_scores = np.array(coarse_scores)
            coarse_valid = np.array(coarse_valid)

            # Find coarse best (Gaussian fit or max)
            try:
                _, _, params, coarse_r2 = fit_focus_curve(
                    coarse_valid, coarse_scores, FitFunction.GAUSSIAN.value
                )
                if coarse_r2 >= 0.5:
                    coarse_best = float(params[1])
                    coarse_best = max(min(coarse_best, coarse_valid.max()), coarse_valid.min())
                else:
                    coarse_best = float(coarse_valid[np.argmax(coarse_scores)])
            except Exception:
                coarse_best = float(coarse_valid[np.argmax(coarse_scores)])
                coarse_r2 = 0.0

            print(f"    Coarse best: {coarse_best:.1f}µm (R²={coarse_r2:.3f})")

            # --- STAGE 2: FINE SWEEP ---
            print(f"  Stage 2: Fine sweep ±{FINE_RANGE}µm around {coarse_best:.1f}µm, {FINE_STEP}µm steps...")
            fine_positions = np.arange(
                coarse_best - FINE_RANGE,
                coarse_best + FINE_RANGE + FINE_STEP,
                FINE_STEP
            )

            fine_scores = []
            fine_valid = []

            for piezo in fine_positions:
                result = await client.capture_lightsheet_image(
                    piezo_position=float(piezo),
                    galvo_position=float(galvo_pos)
                )
                if result.get('success'):
                    total_exposures += 1
                if result.get('success') and result.get('image') is not None:
                    img = select_best_view(result['image'])
                    score = calculate_focus_score(img, algorithm='fft_bandpass')
                    fine_scores.append(score)
                    fine_valid.append(piezo)
                    print(f"    piezo={piezo:.1f}: score={score:.2e}")

            if len(fine_scores) < 4:
                return f"Error: Not enough fine images at galvo={galvo_pos}"

            fine_scores = np.array(fine_scores)
            fine_valid = np.array(fine_valid)

            # Final Gaussian fit with quality checks
            try:
                _, _, params, r_squared = fit_focus_curve(
                    fine_valid, fine_scores, FitFunction.GAUSSIAN.value
                )
                best_piezo = float(params[1])

                # Check if peak is in center region (not edge)
                sweep_min, sweep_max = fine_valid.min(), fine_valid.max()
                sweep_range = sweep_max - sweep_min
                edge_margin = sweep_range * EDGE_EXCLUSION
                peak_in_center = (best_piezo >= sweep_min + edge_margin) and (best_piezo <= sweep_max - edge_margin)

                if r_squared >= MIN_R_SQUARED and peak_in_center:
                    # Good fit, use Gaussian peak
                    best_piezo = max(min(best_piezo, sweep_max), sweep_min)
                    fit_quality = "good"
                elif r_squared >= 0.5:
                    # Moderate fit, use with caution
                    best_piezo = max(min(best_piezo, sweep_max), sweep_min)
                    fit_quality = "moderate"
                else:
                    # Poor fit, fall back to max score
                    best_piezo = float(fine_valid[np.argmax(fine_scores)])
                    r_squared = 0.0
                    fit_quality = "fallback"
            except Exception:
                best_piezo = float(fine_valid[np.argmax(fine_scores)])
                r_squared = 0.0
                fit_quality = "failed"

            results[galvo_name] = {
                'galvo': galvo_pos,
                'piezo': best_piezo,
                'max_score': float(fine_scores.max()),
                'r_squared': r_squared,
            }
            print(f"    → Best focus: piezo={best_piezo:.2f}µm (R²={r_squared:.3f}, {fit_quality})")

        # === PHASE 4: CALCULATE 2-POINT LINEAR CALIBRATION ===
        g_top = results['top']['galvo']
        p_top = results['top']['piezo']
        g_bottom = results['bottom']['galvo']
        p_bottom = results['bottom']['piezo']

        slope = (p_bottom - p_top) / (g_bottom - g_top)
        offset = p_top - slope * g_top

        # Calculate volume acquisition parameters from embryo extent
        # Use detected edges (not calibration positions) for full coverage
        galvo_center = (detected_top + detected_bottom) / 2
        galvo_amplitude = (detected_bottom - detected_top) / 2

        # Calculate piezo range using the linear relationship
        piezo_at_top = slope * detected_top + offset
        piezo_at_bottom = slope * detected_bottom + offset
        piezo_center = (piezo_at_top + piezo_at_bottom) / 2
        piezo_amplitude = abs(piezo_at_bottom - piezo_at_top) / 2

        # Store calibration
        embryo.calibration = {
            'slope_um_per_deg': slope,
            'offset_um': offset,
            'galvo_top_deg': detected_top,
            'galvo_bottom_deg': detected_bottom,
            'galvo_calib_top_deg': g_top,
            'galvo_calib_bottom_deg': g_bottom,
            'piezo_top_um': p_top,
            'piezo_bottom_um': p_bottom,
            # Volume acquisition parameters
            'galvo_center': galvo_center,
            'galvo_amplitude': galvo_amplitude,
            'piezo_center': piezo_center,
            'piezo_amplitude': piezo_amplitude,
            'r_squared_top': results['top']['r_squared'],
            'r_squared_bottom': results['bottom']['r_squared'],
        }

        # Add to focus history
        for name in ['top', 'bottom']:
            embryo.add_focus_datapoint(
                galvo=results[name]['galvo'],
                piezo=results[name]['piezo'],
                score=results[name]['max_score'],
                r_squared=results[name]['r_squared'],
                method='calibration',
                algorithm='fft_bandpass',
            )

        # Record total light exposure from calibration (50ms default exposure)
        if total_exposures > 0:
            embryo.record_exposure(exposure_ms=50.0, num_frames=total_exposures)

        copilot._mark_significant_action("calibration")

        return (
            f"✓ Calibrated {embryo_id}\n"
            f"  Embryo extent: galvo {detected_top:.3f}° to {detected_bottom:.3f}° "
            f"(~{(detected_bottom - detected_top) * 100:.0f}µm)\n"
            f"  Slope: {slope:.2f} µm/deg\n"
            f"  Offset: {offset:.2f} µm (piezo at galvo=0)\n"
            f"  Top: galvo={g_top:.3f}° → piezo={p_top:.1f}µm\n"
            f"  Bottom: galvo={g_bottom:.3f}° → piezo={p_bottom:.1f}µm\n"
            f"  Volume params: galvo={galvo_center:.3f}°±{galvo_amplitude:.3f}°, "
            f"piezo={piezo_center:.1f}µm±{piezo_amplitude:.1f}µm"
        )

    except Exception as e:
        import traceback
        return f"Error calibrating embryo: {str(e)}\n{traceback.format_exc()}"


@tool(
    name="calibrate_all_embryos",
    description="""Run piezo-galvo calibration for all detected embryos sequentially.
Uses Claude vision to detect embryo Z extent for each embryo, then runs focus sweeps.
Use after detecting multiple embryos.""",
    category=ToolCategory.CALIBRATION,
    requires_microscope=True,
    examples=[
        ToolExample("Calibrate all embryos", {}),
        ToolExample("Quick calibration without edge detection", {"skip_edge_detection": True}),
    ],
)
async def calibrate_all_embryos(
    embryo_ids: List[str] = None,
    skip_edge_detection: bool = False,
    context: Dict = None
) -> str:
    """Calibrate all embryos sequentially with Claude vision"""
    copilot = context.get('copilot')

    if not copilot:
        return "Error: No copilot context"

    # Get embryos to calibrate
    if embryo_ids:
        ids_to_calibrate = embryo_ids
    else:
        ids_to_calibrate = list(copilot.experiment.embryos.keys())

    if not ids_to_calibrate:
        return "No embryos to calibrate. Detect embryos first."

    results = []
    for i, eid in enumerate(ids_to_calibrate, 1):
        print(f"\n{'='*60}")
        print(f"Calibrating {eid} ({i}/{len(ids_to_calibrate)})")
        print(f"{'='*60}")

        result = await calibrate_embryo(
            embryo_id=eid,
            skip_edge_detection=skip_edge_detection,
            context=context
        )
        # Get first two lines of result
        lines = result.split('\n')
        summary = lines[0] if len(lines) == 1 else f"{lines[0]} {lines[1]}"
        results.append(f"{eid}: {summary}")

    return f"Calibration complete for {len(ids_to_calibrate)} embryo(s):\n" + "\n".join(results)


@tool(
    name="acquire_volume",
    description="""Acquire a single 3D lightsheet volume for a specific embryo. Moves to embryo position and uses its calibration data.
Use when user wants a full 3D stack of an embryo (e.g., "acquire volume of embryo 1", "take a 3D image").
Embryo must be calibrated first. Default 50 slices at 10ms exposure takes ~2.5 seconds. Turns laser on during acquisition.""",
    category=ToolCategory.HARDWARE,
    requires_microscope=True,
    examples=[
        ToolExample("Acquire volume of embryo 1", {"embryo_id": "embryo_1"}),
        ToolExample("Take a 3D image of embryo 2 with 80 slices", {"embryo_id": "embryo_2", "num_slices": 80}),
    ],
)
async def acquire_volume(
    embryo_id: str,
    num_slices: int = 50,
    exposure_ms: float = 10.0,
    context: Dict = None
) -> str:
    """Acquire single volume - moves to embryo first, uses calibration"""
    copilot = context.get('copilot')
    client = context.get('client')

    if not copilot:
        return "Error: No copilot context"

    embryo, err = get_embryo_or_error(copilot, embryo_id)
    if err:
        return err

    try:
        # Move to embryo position first
        pos = embryo.stage_position
        if pos and pos.get('x') is not None and pos.get('y') is not None:
            await client.move_to_position(pos['x'], pos['y'])

        # Get calibration parameters (use defaults if not calibrated)
        cal = embryo.calibration or {}
        galvo_amplitude = cal.get('galvo_amplitude', 0.5)
        galvo_center = cal.get('galvo_center', 0.0)
        piezo_amplitude = cal.get('piezo_amplitude', 25.0)
        piezo_center = cal.get('piezo_center', 50.0)

        result = await client.acquire_volume(
            num_slices=num_slices,
            exposure_ms=exposure_ms,
            galvo_amplitude=galvo_amplitude,
            galvo_center=galvo_center,
            piezo_amplitude=piezo_amplitude,
            piezo_center=piezo_center
        )

        if result.get('success'):
            # Update embryo state
            embryo.timepoints_acquired += 1
            from datetime import datetime
            embryo.last_imaged = datetime.now()
            # Record light exposure (num_slices frames at exposure_ms each)
            embryo.record_exposure(exposure_ms=exposure_ms, num_frames=num_slices)
            return f"Acquired volume for {embryo.id}\nShape: {result.get('shape', 'unknown')}"
        else:
            return f"Acquisition failed: {result.get('error', 'Unknown error')}"

    except Exception as e:
        return f"Error acquiring volume: {str(e)}"


@tool(
    name="view_image",
    description="""Capture and display the current bottom camera widefield image. Shows what's visible at the current stage position.
Use when user says "show me the view", "take a picture", "what does it look like?", or to check sample positioning.
This is the widefield/brightfield camera looking up at the sample - good for seeing embryo outlines and overall positioning.
Image is automatically saved to camera_captures/ folder.""",
    category=ToolCategory.HARDWARE,
    requires_microscope=True,
    examples=[
        ToolExample("Show me the current view", {}),
        ToolExample("What does the sample look like?", {}),
    ],
)
async def view_image(
    title: str = "Bottom Camera Image",
    exposure_ms: float = None,
    show: bool = True,
    show_embryos: bool = True,
    context: Dict = None
) -> str:
    """Capture and display bottom camera image with embryo annotations"""
    client = context.get('client')
    copilot = context.get('copilot')

    try:
        image = await client.capture_bottom_image(exposure_ms=exposure_ms)

        if image is None or image.shape == (100, 100):
            return "Failed to capture image from bottom camera"

        # Get current stage position for coordinate conversion
        stage_pos = await client.get_stage_position()

        # Prepare embryo annotations if requested
        embryo_annotations = []
        if show_embryos and copilot and copilot.experiment.embryos:
            um_per_pixel = get_um_per_pixel()  # Uses centralized defaults from coordinates.py
            image_center_x = image.shape[1] / 2
            image_center_y = image.shape[0] / 2

            for embryo_id, embryo in copilot.experiment.embryos.items():
                if embryo.stage_position:
                    # Convert stage position to pixel position using centralized function
                    emb_x = embryo.stage_position.get('x', 0)
                    emb_y = embryo.stage_position.get('y', 0)
                    pixel_x, pixel_y = stage_to_pixel_position(
                        stage_x=emb_x,
                        stage_y=emb_y,
                        current_stage_x=stage_pos[0],
                        current_stage_y=stage_pos[1],
                        image_center_x=image_center_x,
                        image_center_y=image_center_y,
                        um_per_pixel=um_per_pixel
                    )

                    embryo_annotations.append({
                        'embryo_id': embryo_id,
                        'pixel_x': pixel_x,
                        'pixel_y': pixel_y,
                        'label': embryo.user_label or embryo_id
                    })

        if show:
            from datetime import datetime
            from pathlib import Path
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = f"camera_captures/bottom_camera_{timestamp}.jpg"
            Path("camera_captures").mkdir(exist_ok=True)

            view_result = await client.view_image(
                image=image,
                title=title,
                save_path=save_path,
                show=True,
                embryo_annotations=embryo_annotations if embryo_annotations else None
            )
            num_visible = len([a for a in embryo_annotations
                              if 0 <= a['pixel_x'] < image.shape[1] and 0 <= a['pixel_y'] < image.shape[0]])
            annotation_msg = f"\nShowing {num_visible} embryo(s) in view" if embryo_annotations else ""
            return f"Captured bottom camera image ({image.shape[0]}x{image.shape[1]})\nSaved to: {save_path}{annotation_msg}"
        else:
            return f"Captured bottom camera image ({image.shape[0]}x{image.shape[1]})"

    except Exception as e:
        return f"Error capturing image: {str(e)}"


@tool(
    name="capture_lightsheet",
    description="""Capture a single 2D lightsheet fluorescence image at specified piezo/galvo position. Uses 50ms exposure by default.
Use when user says "take a lightsheet image", "lightsheet snap", or wants to see fluorescence at a specific Z position.
This is a COMPLETE action - do NOT follow up with acquire_volume unless user explicitly asks for a 3D volume.

IMPORTANT: Always pass embryo_id when capturing for an embryo. This ensures the image is captured at the correct
focus position from the embryo's focus_history (set by fine_focus). Without embryo_id, focus may be incorrect.

The piezo position is determined by priority:
1. Explicit piezo_position parameter (if provided)
2. Embryo's focus_history for the given galvo_position (if embryo_id provided and has focus data)
3. Hardware query fallback (unreliable, may return 0)

If embryo has no focus data for the requested galvo_position, consider running fine_focus first.""",
    category=ToolCategory.HARDWARE,
    requires_microscope=True,
    examples=[
        ToolExample("Take a lightsheet image of embryo 1", {"embryo_id": "embryo_1"}),
        ToolExample("Lightsheet snap at specific piezo", {"embryo_id": "embryo_1", "piezo_position": 50.0}),
        ToolExample("Capture at different galvo", {"embryo_id": "embryo_1", "galvo_position": 0.5}),
    ],
)
async def capture_lightsheet(
    piezo_position: float = None,
    galvo_position: float = 0.0,
    embryo_id: str = None,
    show: bool = True,
    context: Dict = None
) -> str:
    """Capture and optionally display a single lightsheet image"""
    client = context.get('client')
    copilot = context.get('copilot')

    try:
        embryo = None
        # If embryo_id specified, get the embryo and move to its position
        if embryo_id and copilot:
            embryo = copilot.experiment.get_embryo_by_any_name(embryo_id)
            if embryo and embryo.stage_position:
                # Move stage to embryo's position
                await client.move_to_position(
                    x=embryo.stage_position['x'],
                    y=embryo.stage_position['y']
                )

        # Determine piezo position for best focus
        # Priority: explicit param > embryo focus history > hardware query
        focus_source = None
        if piezo_position is None:
            # Check embryo's focus history first (from fine_focus)
            if embryo and embryo.focus_history:
                    # Try interpolation if we have 2+ points
                    fit = embryo.get_piezo_galvo_fit()
                    if fit is not None:
                        slope, intercept = fit
                        piezo_position = slope * galvo_position + intercept
                        focus_source = "interpolated"
                    else:
                        # Single point or exact match
                        piezo_position = embryo.get_focus_at_galvo(galvo_position)
                        if piezo_position is not None:
                            focus_source = "focus_history"

            # Fall back to hardware query (unreliable)
            if piezo_position is None:
                piezo_position = await client.get_piezo_position()
                focus_source = "hardware_query"

        result = await client.capture_lightsheet_image(
            piezo_position=piezo_position,
            galvo_position=galvo_position
        )

        if result.get('success'):
            image = result.get('image')
            run_uid = result.get('run_uid', 'unknown')

            # Update embryo's last_imaged and exposure tracking if specified
            if embryo:
                from datetime import datetime
                embryo.last_imaged = datetime.now()
                # Default lightsheet exposure is 50ms
                embryo.record_exposure(exposure_ms=50.0, num_frames=1)

            # Build focus info string
            focus_info = ""
            if focus_source == "interpolated":
                focus_info = " (focus: interpolated from calibration)"
            elif focus_source == "focus_history":
                focus_info = " (focus: from fine_focus)"
            elif focus_source == "hardware_query":
                focus_info = " (focus: hardware query - may be inaccurate)"

            if image is not None and show:
                # Display the image
                from datetime import datetime
                from pathlib import Path
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = f"lightsheet_captures/lightsheet_{timestamp}.jpg"
                Path("lightsheet_captures").mkdir(exist_ok=True)

                view_result = await client.view_image(
                    image=image,
                    title=f"Lightsheet: piezo={piezo_position:.2f}um, galvo={galvo_position}V",
                    save_path=save_path,
                    show=True
                )
                return f"✓ Captured lightsheet at piezo={piezo_position:.2f}μm, galvo={galvo_position}V{focus_info}\nSaved to: {save_path}"
            elif image is None:
                return f"✓ Lightsheet captured at piezo={piezo_position:.2f}μm, galvo={galvo_position}V{focus_info} (image not displayed)\nRun UID: {run_uid}"
            else:
                return f"✓ Captured lightsheet at piezo={piezo_position:.2f}μm, galvo={galvo_position}V{focus_info}"
        else:
            return f"Failed: {result.get('error', 'Unknown error')}"

    except Exception as e:
        return f"Error capturing lightsheet: {str(e)}"


@tool(
    name="batch_lightsheet",
    description="""Capture lightsheet images from ALL embryos and display them together in a single napari viewer.
Use when user says "lightsheet all embryos", "capture all embryos", "show me all embryos in lightsheet".
Moves to each embryo, captures a lightsheet image, then opens napari with all images as separate layers.
Much more efficient than capturing one at a time.""",
    category=ToolCategory.HARDWARE,
    requires_microscope=True,
    examples=[
        ToolExample("Take lightsheet of all embryos", {}),
        ToolExample("Capture all embryos", {}),
    ],
)
async def batch_lightsheet(
    galvo_position: float = 0.0,
    context: Dict = None
) -> str:
    """Capture lightsheet images from all embryos and show in single napari viewer"""
    copilot = context.get('copilot')
    client = context.get('client')

    if not copilot or not client:
        return "Error: Copilot or microscope not available"

    if not copilot.experiment.embryos:
        return "No embryos in experiment. Run detect_embryos first."

    # Collect images from all embryos
    images = []
    embryo_ids = []
    errors = []

    active_embryos = [
        (eid, emb) for eid, emb in copilot.experiment.embryos.items()
        if not emb.should_skip
    ]

    if not active_embryos:
        return "No active embryos to capture. All embryos are marked as skipped."

    print(f"  Capturing lightsheet from {len(active_embryos)} embryos...")

    for embryo_id, embryo in active_embryos:
        try:
            # Move to embryo position
            if embryo.stage_position:
                x = embryo.stage_position.get('x', 0)
                y = embryo.stage_position.get('y', 0)
                print(f"  Moving to {embryo_id} at ({x:.1f}, {y:.1f})...")
                await client.move_to_position(x, y)
                # Wait for stage to settle
                await asyncio.sleep(0.5)

            # Get piezo and galvo positions from calibration or defaults
            piezo_position = 4.0  # default for galvo=0
            embryo_galvo = galvo_position  # use parameter as default

            if embryo.calibration:
                # Get piezo center
                if embryo.calibration.get('piezo_center'):
                    piezo_position = embryo.calibration['piezo_center']
                elif embryo.calibration.get('focus_position'):
                    piezo_position = embryo.calibration['focus_position']

                # Get galvo center (critical for light sheet alignment)
                if embryo.calibration.get('galvo_center'):
                    embryo_galvo = embryo.calibration['galvo_center']

            # Capture lightsheet
            print(f"  Capturing {embryo_id} at piezo={piezo_position:.1f}μm, galvo={embryo_galvo:.2f}...")
            result = await client.capture_lightsheet_image(
                piezo_position=piezo_position,
                galvo_position=embryo_galvo
            )

            if result.get('success') and result.get('image') is not None:
                images.append(result['image'])
                embryo_ids.append(embryo_id)
                # Track light exposure (default 50ms)
                embryo.record_exposure(exposure_ms=50.0, num_frames=1)
            else:
                errors.append(f"{embryo_id}: {result.get('error', 'no image')}")

        except Exception as e:
            errors.append(f"{embryo_id}: {str(e)}")

    if not images:
        return f"Failed to capture any images. Errors: {'; '.join(errors)}"

    # Save images
    from datetime import datetime
    from pathlib import Path
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = Path("lightsheet_captures") / f"batch_{timestamp}"
    save_dir.mkdir(parents=True, exist_ok=True)

    for i, (img, eid) in enumerate(zip(images, embryo_ids)):
        import tifffile
        save_path = save_dir / f"{eid}.tiff"
        tifffile.imwrite(str(save_path), img)

    print(f"  Saved {len(images)} images to {save_dir}")

    # Open single napari viewer with all images as a stack
    import napari
    import numpy as np
    print(f"  Opening napari with {len(images)} embryo images as stack...")

    # Stack images into a single array for slider navigation
    image_stack = np.stack(images, axis=0)

    viewer = napari.Viewer(title=f"Batch Lightsheet - {len(images)} embryos")

    # Add as single stack with slider (grayscale)
    viewer.add_image(
        image_stack,
        name='Embryos',
        colormap='gray',
    )

    # Print embryo ID mapping for reference
    print("  Slider index → Embryo ID:")
    for i, eid in enumerate(embryo_ids):
        print(f"    {i}: {eid}")

    napari.run()

    # Summary
    summary = f"✓ Captured {len(images)} embryos: {', '.join(embryo_ids)}"
    if errors:
        summary += f"\n⚠ Errors: {'; '.join(errors)}"
    summary += f"\nSaved to: {save_dir}"

    return summary


@tool(
    name="view_volume",
    description="""Open a volume in napari for 3D visualization.
Can open a volume by file path OR by embryo ID (opens latest volume or specific timepoint).
Use when user says "open volume", "view volume", "show volume in napari", or "look at the 3D data".""",
    category=ToolCategory.ANALYSIS,
    requires_microscope=False,
    examples=[
        ToolExample("Open latest volume for embryo 2", {"embryo_id": "embryo_2"}),
        ToolExample("Open specific timepoint", {"embryo_id": "embryo_2", "timepoint": 5}),
        ToolExample("Open volume file", {"file_path": "D:/Gently/volumes/embryo_1_t0001.tif"}),
    ],
)
async def view_volume(
    embryo_id: str = None,
    timepoint: int = None,
    file_path: str = None,
    context: Dict = None
) -> str:
    """Open a volume in napari for visualization"""
    import napari
    import tifffile
    import numpy as np
    from pathlib import Path

    copilot, err = require_copilot(context)
    if err:
        return err

    volume = None
    volume_path = None
    title = "Volume Viewer"

    # Determine which volume to open
    if file_path:
        # Open from file path
        volume_path = Path(file_path)
        if not volume_path.exists():
            return f"Error: File not found: {file_path}"
        title = f"Volume: {volume_path.name}"

    elif embryo_id:
        # Get volume for embryo - check both recent_images and disk
        storage_dir = copilot.image_manager.storage_path

        if timepoint is not None:
            # Try to find specific timepoint - first check disk directly
            volume_path = storage_dir / f"{embryo_id}_t{timepoint:04d}.tif"
            if volume_path.exists():
                title = f"{embryo_id} - t{timepoint:04d}"
            else:
                # Check recent_images as fallback
                embryo, err = get_embryo_or_error(copilot, embryo_id)
                if err:
                    return err
                if embryo.recent_images:
                    matching = [img for img in embryo.recent_images if img.timepoint == timepoint]
                    if matching:
                        volume_path = Path(matching[0].volume_path)
                        title = f"{embryo_id} - t{timepoint:04d}"

                if not volume_path.exists():
                    # List available timepoints from disk
                    available = []
                    for f in storage_dir.glob(f"{embryo_id}_t*.tif"):
                        import re
                        match = re.search(r'_t(\d+)\.tif$', f.name)
                        if match:
                            available.append(int(match.group(1)))
                    available.sort()
                    return f"Timepoint {timepoint} not found for {embryo_id}. Available: {available}"
        else:
            # Find latest volume from disk
            import re
            volume_files = list(storage_dir.glob(f"{embryo_id}_t*.tif"))
            if not volume_files:
                return f"No volumes found for {embryo_id} in {storage_dir}"

            # Find highest timepoint
            latest_tp = -1
            for f in volume_files:
                match = re.search(r'_t(\d+)\.tif$', f.name)
                if match:
                    tp = int(match.group(1))
                    if tp > latest_tp:
                        latest_tp = tp
                        volume_path = f

            title = f"{embryo_id} - t{latest_tp:04d}"

    else:
        return "Error: Specify either embryo_id or file_path"

    # Load the volume
    try:
        volume = tifffile.imread(str(volume_path))
        print(f"  Loaded volume: {volume.shape}, dtype={volume.dtype}")
    except Exception as e:
        return f"Error loading volume: {e}"

    # Open in napari
    print(f"  Opening napari viewer...")
    viewer = napari.Viewer(title=title)

    # Add volume with appropriate settings
    viewer.add_image(
        volume,
        name='Volume',
        colormap='gray',
        rendering='mip',  # Maximum intensity projection for 3D
    )

    # Add scale bar info
    viewer.scale_bar.visible = True
    viewer.scale_bar.unit = "um"

    napari.run()

    return f"✓ Opened volume in napari: {volume_path.name} (shape: {volume.shape})"


@tool(
    name="list_volumes",
    description="""List available volumes for an embryo or all embryos.
Shows volume files with timepoints and file sizes. Scans the storage directory for all volumes (not just recent ones in memory).
Use to see what data is available before viewing.""",
    category=ToolCategory.ANALYSIS,
    requires_microscope=False,
    examples=[
        ToolExample("List volumes for embryo 2", {"embryo_id": "embryo_2"}),
        ToolExample("List all volumes", {}),
    ],
)
async def list_volumes(
    embryo_id: str = None,
    context: Dict = None
) -> str:
    """List available volumes"""
    from pathlib import Path
    import re

    copilot, err = require_copilot(context)
    if err:
        return err

    # Get storage directory
    storage_dir = copilot.image_manager.storage_path
    lines = []

    # Pattern to match volume files: embryo_id_tXXXX.tif
    volume_pattern = re.compile(r'^(.+)_t(\d+)\.tif$')

    # Scan storage directory for all volume files
    all_volumes = {}  # embryo_id -> list of (timepoint, path)
    if storage_dir.exists():
        for f in storage_dir.glob("*.tif"):
            match = volume_pattern.match(f.name)
            if match:
                eid = match.group(1)
                tp = int(match.group(2))
                if eid not in all_volumes:
                    all_volumes[eid] = []
                all_volumes[eid].append((tp, f))

    # Sort by timepoint
    for eid in all_volumes:
        all_volumes[eid].sort(key=lambda x: x[0])

    if embryo_id:
        # List volumes for specific embryo
        if embryo_id not in all_volumes:
            return f"No volume files found for {embryo_id} in {storage_dir}"

        volumes = all_volumes[embryo_id]
        lines.append(f"Volumes for {embryo_id}: {len(volumes)} file(s)")
        lines.append(f"Storage: {storage_dir}")
        lines.append("")

        for tp, path in volumes:
            size_mb = path.stat().st_size / (1024 * 1024)
            lines.append(f"  t{tp:04d}: {path.name} ({size_mb:.1f} MB)")

    else:
        # List volumes for all embryos
        if not all_volumes:
            return f"No volume files found in {storage_dir}"

        total_files = sum(len(v) for v in all_volumes.values())
        lines.append(f"Available volumes: {total_files} file(s) across {len(all_volumes)} embryo(s)")
        lines.append(f"Storage: {storage_dir}")

        for eid in sorted(all_volumes.keys()):
            volumes = all_volumes[eid]
            timepoints = [tp for tp, _ in volumes]
            tp_range = f"t{min(timepoints):04d}-t{max(timepoints):04d}" if len(timepoints) > 1 else f"t{timepoints[0]:04d}"
            total_size = sum(p.stat().st_size for _, p in volumes) / (1024 * 1024)
            lines.append(f"\n{eid}: {len(volumes)} volume(s) [{tp_range}] ({total_size:.1f} MB total)")

            # Show last few timepoints
            for tp, path in volumes[-3:]:
                size_mb = path.stat().st_size / (1024 * 1024)
                lines.append(f"    t{tp:04d}: {size_mb:.1f} MB")
            if len(volumes) > 3:
                lines.append(f"    ... and {len(volumes) - 3} more")

    return "\n".join(lines)
