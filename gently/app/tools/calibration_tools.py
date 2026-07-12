"""
Calibration Tools

Tools for microscope piezo-galvo calibration including adaptive focus sweeps,
binary edge search, and full/fast calibration routines.
"""

import logging
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)

import numpy as np  # noqa: E402

from gently.analysis.core import AdaptiveSweepState, FitFunction, fit_focus_curve  # noqa: E402
from gently.harness.state import CalibrationPrior  # noqa: E402
from gently.harness.tools.helpers import ctx_get, get_embryo_or_error  # noqa: E402
from gently.harness.tools.registry import ToolCategory, ToolExample, tool  # noqa: E402
from gently.ui.web.plots import (  # noqa: E402
    generate_calibration_summary_plot,
    generate_focus_curve_plot,
)

from .hardware_common import (  # noqa: E402
    select_best_view,
    select_view_and_crop_roi,
)


async def _adaptive_focus_sweep(
    client,
    agent,
    embryo_id: str,
    galvo_name: str,
    galvo_pos: float,
    expected_piezo: float,
    session_prior: CalibrationPrior,
    select_best_view,
    calculate_focus_score,
) -> tuple[dict, int]:
    """
    Adaptive focus sweep with early stopping.

    Uses sparse-then-dense approach:
    1. Sparse survey over a wide window (±15µm, 5µm steps) to find approximate
       peak with early stopping. The window is intentionally wide so a stale
       heuristic or a previously-unseen sample region still catches the true
       peak instead of fitting noise on the shoulder.
    2. Dense refinement (0.5µm steps) around the sparse peak with early stopping

    Parameters
    ----------
    client : QueueServerClient
        Microscope client for image capture
    agent : MicroscopyAgent
        Agent for viz server access
    embryo_id : str
        Embryo identifier for viz metadata
    galvo_name : str
        'top' or 'bottom' for logging
    galvo_pos : float
        Galvo position (degrees)
    expected_piezo : float
        Expected piezo position from prior/heuristic
    session_prior : CalibrationPrior
        Session-level calibration prior
    select_best_view : callable
        Function to select best view from dual-view image
    calculate_focus_score : callable
        Function to calculate focus score

    Returns
    -------
    tuple
        (result_dict, total_exposures)
        result_dict has 'galvo', 'piezo', 'max_score', 'r_squared', 'fit_params'
    """
    total_exposures = 0

    # Adaptive parameters based on prior confidence
    SPARSE_STEP = 5.0  # µm - larger steps for survey
    DENSE_STEP = 0.5  # µm - fine steps for refinement
    DENSE_RANGE = 3.0  # µm - narrow window around peak
    MIN_R_SQUARED = 0.75

    # Sparse window is wide (±15 µm) so the sweep still catches the true peak
    # when the heuristic piezo is off by up to ~10 µm — e.g. after moving to a
    # new sample region or when the session prior is stale. Prior versions used
    # ±5 µm, which silently fit noise when the heuristic missed.
    sparse_range = 15.0  # µm sweep range

    logger.info("=== ADAPTIVE %s FOCUS SWEEP at galvo=%.3f deg ===", galvo_name.upper(), galvo_pos)
    logger.info(
        "Using range: +/-%.1f um (prior: %d calibrations)",
        sparse_range,
        session_prior.num_calibrations,
    )

    # --- PHASE 1: SPARSE SURVEY ---
    logger.info("Phase 1: Sparse survey +/-%.1f um, %.1f um steps...", sparse_range, SPARSE_STEP)

    sparse_state = AdaptiveSweepState()
    sparse_positions = np.arange(
        expected_piezo - sparse_range, expected_piezo + sparse_range + SPARSE_STEP, SPARSE_STEP
    )

    for piezo in sparse_positions:
        result = await client.capture_lightsheet_image(
            piezo_position=float(piezo), galvo_position=float(galvo_pos)
        )
        if result.get("success"):
            total_exposures += 1
        if result.get("success") and result.get("image") is not None:
            img = select_best_view(result["image"])
            score = calculate_focus_score(img, algorithm="fft_bandpass")

            # Add to adaptive state and check for early stopping
            decision = sparse_state.add_point(float(piezo), float(score))

            # Push to viz server
            if agent.viz_server:
                agent.push_viz(
                    array=img,
                    uid=f"focus_sparse_{embryo_id}_{galvo_name}_{piezo:.1f}",
                    data_type="focus_sweep",
                    metadata={
                        "embryo_id": embryo_id,
                        "sweep": "sparse",
                        "galvo_name": galvo_name,
                        "galvo": float(galvo_pos),
                        "piezo": float(piezo),
                        "score": float(score),
                        "peak_detected": sparse_state.peak_detected,
                    },
                )

            if decision["should_stop"]:
                logger.info(
                    "Early stop: %s (confidence: %.2f)", decision["reason"], decision["confidence"]
                )
                break

    sparse_best, sparse_r2 = sparse_state.get_best_position()
    logger.info(
        "Sparse best: %.1f um (R2=%.3f, %d points)",
        sparse_best,
        sparse_r2,
        len(sparse_state.positions),
    )

    # --- PHASE 2: DENSE REFINEMENT ---
    logger.info(
        "Phase 2: Dense refinement +/-%.1f um around %.1f um, %.1f um steps...",
        DENSE_RANGE,
        sparse_best,
        DENSE_STEP,
    )

    dense_state = AdaptiveSweepState()
    dense_positions = np.arange(
        sparse_best - DENSE_RANGE, sparse_best + DENSE_RANGE + DENSE_STEP, DENSE_STEP
    )

    for piezo in dense_positions:
        result = await client.capture_lightsheet_image(
            piezo_position=float(piezo), galvo_position=float(galvo_pos)
        )
        if result.get("success"):
            total_exposures += 1
        if result.get("success") and result.get("image") is not None:
            img = select_best_view(result["image"])
            score = calculate_focus_score(img, algorithm="fft_bandpass")

            # Add to adaptive state and check for early stopping
            decision = dense_state.add_point(float(piezo), float(score))

            logger.debug("piezo=%.1f: score=%.2e", piezo, score)

            # Push to viz server
            if agent.viz_server:
                agent.push_viz(
                    array=img,
                    uid=f"focus_dense_{embryo_id}_{galvo_name}_{piezo:.1f}",
                    data_type="focus_sweep",
                    metadata={
                        "embryo_id": embryo_id,
                        "sweep": "dense",
                        "galvo_name": galvo_name,
                        "galvo": float(galvo_pos),
                        "piezo": float(piezo),
                        "score": float(score),
                        "r_squared": dense_state.current_r_squared,
                    },
                )

            if decision["should_stop"]:
                logger.info(
                    "Early stop: %s (confidence: %.2f)", decision["reason"], decision["confidence"]
                )
                break

    # Get final result
    best_piezo, r_squared = dense_state.get_best_position()

    # Validate result
    if r_squared < MIN_R_SQUARED and len(dense_state.positions) >= 5:
        # Try fitting again with all data
        try:
            positions = np.array(dense_state.positions)
            scores = np.array(dense_state.scores)
            _, _, params, r_squared = fit_focus_curve(positions, scores, FitFunction.GAUSSIAN.value)
            if r_squared >= 0.5:
                best_piezo = float(params[1])
                best_piezo = max(min(best_piezo, positions.max()), positions.min())
        except Exception:
            pass

    # Determine fit quality
    if r_squared >= MIN_R_SQUARED:
        fit_quality = "good"
    elif r_squared >= 0.5:
        fit_quality = "moderate"
    else:
        fit_quality = "fallback"

    logger.info("Best focus: piezo=%.2f um (R2=%.3f, %s)", best_piezo, r_squared, fit_quality)
    logger.info(
        "Total exposures: %d (sparse: %d, dense: %d)",
        total_exposures,
        len(sparse_state.positions),
        len(dense_state.positions),
    )

    # Build result dict
    result_dict: dict[str, Any] = {
        "galvo": galvo_pos,
        "piezo": best_piezo,
        "max_score": float(max(dense_state.scores)) if dense_state.scores else 0.0,
        "r_squared": r_squared,
        "fit_params": None,  # Will be set by focus curve plot generation
    }

    # Push focus curve plot
    if agent.viz_server and len(dense_state.positions) >= 4:
        try:
            positions = np.array(dense_state.positions)
            scores = np.array(dense_state.scores)
            try:
                _, _, fit_params, _ = fit_focus_curve(positions, scores, FitFunction.GAUSSIAN.value)
                result_dict["fit_params"] = fit_params
            except Exception:
                fit_params = None

            plot_img = generate_focus_curve_plot(
                positions=positions,
                scores=scores,
                best_position=best_piezo,
                fit_params=fit_params,
                r_squared=r_squared,
                title=f"{embryo_id} - {galvo_name.upper()} Focus Curve (Adaptive)",
            )
            agent.push_viz(
                array=plot_img,
                uid=f"focus_curve_{embryo_id}_{galvo_name}",
                data_type="focus_plot",
                metadata={
                    "embryo_id": embryo_id,
                    "galvo_name": galvo_name,
                    "galvo": float(galvo_pos),
                    "best_piezo": best_piezo,
                    "r_squared": r_squared,
                    "adaptive": True,
                    "exposures": total_exposures,
                },
            )
        except Exception as plot_err:
            logger.warning("Failed to generate focus plot: %s", plot_err)

    return result_dict, total_exposures


async def _fine_focus_sweep(
    client,
    agent,
    embryo_id: str,
    galvo_name: str,
    galvo_pos: float,
    expected_piezo: float,
    select_best_view,
    calculate_focus_score,
) -> tuple[dict, int]:
    """
    Fine-only focus sweep - assumes heuristic is close.

    Used when Claude has identified feature-rich positions during edge detection.
    Since heuristic piezo position is already good, we only do a fine sweep.

    Parameters
    ----------
    client : QueueServerClient
        Microscope client for image capture
    agent : MicroscopyAgent
        Agent for viz server access
    embryo_id : str
        Embryo identifier for viz metadata
    galvo_name : str
        'top' or 'bottom' for logging
    galvo_pos : float
        Galvo position (degrees)
    expected_piezo : float
        Expected piezo position from heuristic
    select_best_view : callable
        Function to select best view from dual-view image and crop to ROI
    calculate_focus_score : callable
        Function to calculate focus score

    Returns
    -------
    tuple
        (result_dict, total_exposures)
        result_dict has 'galvo', 'piezo', 'max_score', 'r_squared', 'fit_params'
    """
    FINE_RANGE = 5.0  # ±5µm around expected
    FINE_STEP = 0.5  # 0.5µm steps
    MIN_R_SQUARED = 0.75
    total_exposures = 0

    logger.info("=== FINE-ONLY %s FOCUS SWEEP at galvo=%.3f deg ===", galvo_name.upper(), galvo_pos)
    logger.info(
        "Sweeping +/-%.1f um around heuristic (%.1f um), %.1f um steps...",
        FINE_RANGE,
        expected_piezo,
        FINE_STEP,
    )

    positions = np.arange(
        expected_piezo - FINE_RANGE, expected_piezo + FINE_RANGE + FINE_STEP, FINE_STEP
    )

    piezo_scores = []
    for piezo in positions:
        result = await client.capture_lightsheet_image(
            piezo_position=float(piezo), galvo_position=float(galvo_pos)
        )
        if result.get("success"):
            total_exposures += 1
        if result.get("success") and result.get("image") is not None:
            img = select_best_view(result["image"])
            score = calculate_focus_score(img, algorithm="fft_bandpass")
            piezo_scores.append((float(piezo), float(score)))

            logger.debug("piezo=%.1f: score=%.2e", piezo, score)

            # Push to viz server
            if agent.viz_server:
                agent.push_viz(
                    array=img,
                    uid=f"focus_fine_{embryo_id}_{galvo_name}_{piezo:.1f}",
                    data_type="focus_sweep",
                    metadata={
                        "embryo_id": embryo_id,
                        "sweep": "fine_only",
                        "galvo_name": galvo_name,
                        "galvo": float(galvo_pos),
                        "piezo": float(piezo),
                        "score": float(score),
                    },
                )

    # Fit Gaussian to find peak
    if len(piezo_scores) >= 4:
        positions_arr = np.array([p for p, _ in piezo_scores])
        scores_arr = np.array([s for _, s in piezo_scores])

        try:
            _, _, fit_params, r_squared = fit_focus_curve(
                positions_arr, scores_arr, FitFunction.GAUSSIAN.value
            )
            # Gaussian fit: params = [amplitude, center, sigma, offset]
            best_piezo = float(fit_params[1])
            # Clamp to measured range
            best_piezo = max(min(best_piezo, positions_arr.max()), positions_arr.min())
        except Exception as fit_err:
            logger.warning("Gaussian fit failed (%s), using max score position", fit_err)
            best_idx = np.argmax(scores_arr)
            best_piezo = float(positions_arr[best_idx])
            r_squared = 0.0
            fit_params = None
    else:
        # Not enough points for fit
        if piezo_scores:
            best_piezo = max(piezo_scores, key=lambda x: x[1])[0]
        else:
            best_piezo = expected_piezo
        r_squared = 0.0
        fit_params = None

    # Determine fit quality
    if r_squared >= MIN_R_SQUARED:
        fit_quality = "good"
    elif r_squared >= 0.5:
        fit_quality = "moderate"
    else:
        fit_quality = "fallback"

    max_score = max(s for _, s in piezo_scores) if piezo_scores else 0.0
    logger.info("Best focus: piezo=%.2f um (R2=%.3f, %s)", best_piezo, r_squared, fit_quality)
    logger.info("Total exposures: %d", total_exposures)

    # Build result dict
    result_dict = {
        "galvo": galvo_pos,
        "piezo": best_piezo,
        "max_score": max_score,
        "r_squared": r_squared,
        "fit_params": fit_params,
    }

    # Push focus curve plot
    if agent.viz_server and len(piezo_scores) >= 4:
        try:
            positions_arr = np.array([p for p, _ in piezo_scores])
            scores_arr = np.array([s for _, s in piezo_scores])

            plot_img = generate_focus_curve_plot(
                positions=positions_arr,
                scores=scores_arr,
                best_position=best_piezo,
                fit_params=fit_params,
                r_squared=r_squared,
                title=f"{embryo_id} - {galvo_name.upper()} Focus Curve (Fine-Only)",
            )
            agent.push_viz(
                array=plot_img,
                uid=f"focus_curve_{embryo_id}_{galvo_name}",
                data_type="focus_plot",
                metadata={
                    "embryo_id": embryo_id,
                    "galvo_name": galvo_name,
                    "galvo": float(galvo_pos),
                    "best_piezo": best_piezo,
                    "r_squared": r_squared,
                    "fine_only": True,
                    "exposures": total_exposures,
                },
            )
        except Exception as plot_err:
            logger.warning("Failed to generate focus plot: %s", plot_err)

    return result_dict, total_exposures


# ============================================================================
# FAST CALIBRATION ALGORITHM (Vision-Guided)
# ============================================================================


async def hybrid_focus_selection(
    images: list[np.ndarray],
    offsets: list[float],
    claude_vision,
    agent,
    embryo_id: str,
    fft_confidence_threshold: float = 0.85,
) -> tuple[int, str, float | None]:
    """
    Two-stage focus selection: FFT first, Vision if ambiguous.

    Stage 1: FFT bandpass scoring (instant, ~10ms)
    Stage 2: Vision API only if FFT is ambiguous (>85% score similarity)

    Parameters
    ----------
    images : List[np.ndarray]
        Focus images at different offsets
    offsets : List[float]
        Piezo offsets in µm for each image
    claude_vision : AsyncClaudeClient
        Vision API client
    agent : MicroscopyAgent
        For viz server access
    embryo_id : str
        Embryo identifier for logging
    fft_confidence_threshold : float
        If second-best score is below this ratio of best, use FFT

    Returns
    -------
    tuple
        (best_idx, method, confidence_ratio)
        method is 'fft' or 'vision'
    """
    from gently.analysis.core import calculate_focus_score

    # Stage 1: FFT scoring (instant)
    scores = [calculate_focus_score(img, "fft_bandpass") for img in images]
    max_score = max(scores)
    best_idx = scores.index(max_score)

    # Check FFT confidence
    score_ratios = [s / max_score if max_score > 0 else 0 for s in scores]
    sorted_ratios = sorted(score_ratios, reverse=True)
    second_best_ratio = sorted_ratios[1] if len(sorted_ratios) > 1 else 0

    confidence_ratio = 1.0 / second_best_ratio if second_best_ratio > 0 else float("inf")

    if second_best_ratio < fft_confidence_threshold:
        # FFT is confident - best is >15% better than second
        logger.info("FFT confident: position %d (ratio %.2f)", best_idx, confidence_ratio)
        return best_idx, "fft", confidence_ratio

    # Stage 2: FFT ambiguous - ask Vision
    logger.info("FFT ambiguous (ratio %.2f), consulting Vision...", confidence_ratio)

    import tempfile
    from pathlib import Path

    from PIL import Image

    from gently.analysis.core import create_focus_montage

    # Create montage
    labels = [chr(ord("A") + i) for i in range(len(images))]
    montage = create_focus_montage(images, labels=labels, offsets=offsets)

    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        montage_path = Path(f.name)
        Image.fromarray(montage).save(montage_path)

    try:
        # Call Vision API
        vision_idx, vision_label, reasoning = await claude_vision.select_best_focus(
            montage_path, offsets, labels
        )
        logger.info("Vision selected: %s (%s)", vision_label, reasoning)

        # Push montage to viz server
        if agent.viz_server:
            agent.push_viz(
                array=montage,
                uid=f"focus_montage_{embryo_id}",
                data_type="focus_montage",
                metadata={
                    "embryo_id": embryo_id,
                    "offsets": offsets,
                    "fft_scores": scores,
                    "vision_pick": vision_label,
                    "reasoning": reasoning,
                },
            )

        return vision_idx, "vision", None

    finally:
        # Clean up temp file
        try:
            montage_path.unlink()
        except Exception:
            pass


async def binary_edge_search(
    client,
    claude_vision,
    direction: str,
    agent,
    embryo_id: str,
    piezo_heuristic: float,
    max_range: float = 0.25,
    num_iterations: int = 4,
) -> tuple[float, int]:
    """
    Binary search for embryo edge in 4 steps.

    Instead of linear sweep (10-20 exposures), uses binary search
    to find embryo edge in 4-5 exposures.

    Parameters
    ----------
    client : QueueServerClient
        Microscope client
    claude_vision : AsyncClaudeClient
        Vision API client
    direction : str
        'top' (negative galvo) or 'bottom' (positive galvo)
    agent : MicroscopyAgent
        For viz server and state
    embryo_id : str
        Embryo identifier
    piezo_heuristic : float
        Expected piezo position at galvo=0
    max_range : float
        Maximum galvo range to search
    num_iterations : int
        Number of binary search steps

    Returns
    -------
    tuple
        (edge_galvo, num_exposures)
    """
    import tempfile
    from pathlib import Path

    from PIL import Image

    sign = -1 if direction == "top" else 1
    low, high = 0.0, max_range * sign
    last_visible = 0.0
    exposures = 0

    logger.info("Binary edge search (%s): range 0 to %.3f deg", direction, high)

    for i in range(num_iterations):
        mid = (low + high) / 2

        # Calculate piezo position using heuristic slope
        HEURISTIC_SLOPE = 100.0  # µm/degree
        piezo = piezo_heuristic + HEURISTIC_SLOPE * mid

        # Capture image
        result = await client.capture_lightsheet_image(
            piezo_position=float(piezo), galvo_position=float(mid)
        )
        exposures += 1

        if not result.get("success") or result.get("image") is None:
            # Assume not visible on failure
            high = mid if sign > 0 else low
            low = mid if sign < 0 else high
            continue

        img = select_best_view(result["image"])

        # Check visibility with Vision
        img_norm = ((img - img.min()) / (img.max() - img.min() + 1e-8) * 255).astype(np.uint8)

        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            temp_path = Path(f.name)
            Image.fromarray(img_norm).save(temp_path)

        try:
            visible, feature_score, desc = await claude_vision.detect_embryo_presence(temp_path)
        finally:
            try:
                temp_path.unlink()
            except Exception:
                pass

        logger.debug(
            "iter %d: galvo=%.3f deg, visible=%s, features=%s", i + 1, mid, visible, feature_score
        )

        if visible:
            last_visible = mid
            # Embryo visible, search further out
            if sign > 0:
                low = mid
            else:
                high = mid
        else:
            # Gone too far, search back
            if sign > 0:
                high = mid
            else:
                low = mid

    return last_visible, exposures


async def fast_calibrate_embryo(
    embryo_id: str, context: dict, z_buffer_um: float = 25.0
) -> tuple[bool, str, int]:
    """
    Vision-guided fast calibration using session slope.

    For first embryo: Full bootstrap calibration to establish slope
    For subsequent: Only find offset using hybrid FFT+Vision focus selection

    Reduces exposures from 60-80 to 8-15 per embryo.

    Parameters
    ----------
    embryo_id : str
        Embryo to calibrate
    context : Dict
        Tool context with agent and client
    z_buffer_um : float
        Z padding above/below embryo for volume acquisition

    Returns
    -------
    tuple
        (success, message, total_exposures)
    """
    from gently.hardware.dispim.claude_client import AsyncClaudeClient

    agent = ctx_get(context, "agent")
    client = ctx_get(context, "client")

    if not agent:
        return False, "Error: No agent context", 0

    if not client or not getattr(client, "is_connected", False):
        return False, "Error: Not connected to microscope server", 0

    embryo, err = get_embryo_or_error(agent, embryo_id)
    if err:
        return False, err, 0

    # Get session prior
    session_prior = agent.calibration_prior

    # Initialize Claude Vision client
    claude_vision = AsyncClaudeClient()

    total_exposures = 0
    HEURISTIC_SLOPE = 100.0  # Default µm/degree

    logger.info("=== FAST CALIBRATION: %s ===", embryo_id)

    # Check if this is bootstrap (first embryo) or fast mode
    is_bootstrap = not session_prior.is_ready_for_fast_calibration()

    if is_bootstrap:
        logger.info("Mode: BOOTSTRAP (establishing session slope)")
    else:
        logger.info("Mode: FAST (using session slope %.2f um/deg)", session_prior.slope_um_per_deg)

    # --- STEP 1: Binary Edge Detection ---
    logger.info("Step 1: Binary edge detection...")

    # Start with heuristic piezo at galvo=0
    piezo_heuristic = session_prior.offset_um if session_prior.num_calibrations > 0 else 0.0

    galvo_top, exp_top = await binary_edge_search(
        client, claude_vision, "top", agent, embryo_id, piezo_heuristic
    )
    total_exposures += exp_top

    galvo_bottom, exp_bottom = await binary_edge_search(
        client, claude_vision, "bottom", agent, embryo_id, piezo_heuristic
    )
    total_exposures += exp_bottom

    logger.info("Edges detected: top=%.3f deg, bottom=%.3f deg", galvo_top, galvo_bottom)

    # Validate edges
    if galvo_top >= galvo_bottom:
        return (
            False,
            f"Invalid edges: top {galvo_top:.3f}° >= bottom {galvo_bottom:.3f}°",
            total_exposures,
        )

    galvo_center = (galvo_top + galvo_bottom) / 2
    galvo_extent = galvo_bottom - galvo_top

    # --- STEP 2: Focus at Center Position ---
    logger.info("Step 2: Focus at center (galvo=%.3f deg)...", galvo_center)

    # Use session slope or heuristic
    slope = (
        session_prior.slope_um_per_deg
        if session_prior.is_ready_for_fast_calibration()
        else HEURISTIC_SLOPE
    )
    piezo_expected = slope * galvo_center + session_prior.offset_um

    # Capture 3-point focus grid (±2µm)
    focus_offsets = [-2.0, 0.0, 2.0]
    focus_images = []

    for offset in focus_offsets:
        result = await client.capture_lightsheet_image(
            piezo_position=float(piezo_expected + offset), galvo_position=float(galvo_center)
        )
        total_exposures += 1

        if result.get("success") and result.get("image") is not None:
            img = select_best_view(result["image"])
            focus_images.append(img)
        else:
            focus_images.append(np.zeros((100, 100), dtype=np.uint8))

    # --- STEP 3: Hybrid Focus Selection ---
    logger.info("Step 3: Hybrid focus selection...")

    best_idx, method, confidence = await hybrid_focus_selection(
        focus_images, focus_offsets, claude_vision, agent, embryo_id
    )

    embryo_offset = focus_offsets[best_idx]
    piezo_center = piezo_expected + embryo_offset

    logger.info("Selected: offset %+.1f um via %s", embryo_offset, method)

    # --- STEP 4: Refinement if Edge Picked ---
    if best_idx in [0, len(focus_offsets) - 1]:
        logger.info("Step 4: Refining (edge position picked)...")

        # Extend search in direction of edge
        if best_idx == 0:
            extend_offsets = [-3.0, -4.0]
        else:
            extend_offsets = [3.0, 4.0]

        for ext_offset in extend_offsets:
            result = await client.capture_lightsheet_image(
                piezo_position=float(piezo_expected + ext_offset),
                galvo_position=float(galvo_center),
            )
            total_exposures += 1

            if result.get("success") and result.get("image") is not None:
                img = select_best_view(result["image"])
                focus_images.append(img)
                focus_offsets.append(ext_offset)

        # Re-run hybrid selection
        best_idx, method, confidence = await hybrid_focus_selection(
            focus_images, focus_offsets, claude_vision, agent, embryo_id
        )
        embryo_offset = focus_offsets[best_idx]
        piezo_center = piezo_expected + embryo_offset
        logger.info("Refined: offset %+.1f um via %s", embryo_offset, method)
    else:
        logger.info("Step 4: Skipped (center position picked)")

    # --- STEP 5: Bootstrap - Establish Session Slope ---
    if is_bootstrap:
        logger.info("Step 5: Bootstrap - calibrating second position for slope...")

        # Pick second position 30% away from center
        galvo_second = galvo_center + 0.3 * galvo_extent
        piezo_expected_second = HEURISTIC_SLOPE * galvo_second + session_prior.offset_um

        # Capture 3-point focus grid at second position
        focus_images_2 = []
        for offset in [-2.0, 0.0, 2.0]:
            result = await client.capture_lightsheet_image(
                piezo_position=float(piezo_expected_second + offset),
                galvo_position=float(galvo_second),
            )
            total_exposures += 1

            if result.get("success") and result.get("image") is not None:
                img = select_best_view(result["image"])
                focus_images_2.append(img)
            else:
                focus_images_2.append(np.zeros((100, 100), dtype=np.uint8))

        best_idx_2, _, _ = await hybrid_focus_selection(
            focus_images_2, [-2.0, 0.0, 2.0], claude_vision, agent, embryo_id
        )
        piezo_second = piezo_expected_second + [-2.0, 0.0, 2.0][best_idx_2]

        # Calculate slope from two points
        calibrated_slope = (piezo_second - piezo_center) / (galvo_second - galvo_center)
        calibrated_offset = piezo_center - calibrated_slope * galvo_center

        # Lock session slope
        session_prior.lock_session_slope(calibrated_slope, 0.85, embryo_id)
        logger.info(
            "Session slope locked: %.2f um/deg (bootstrap embryo: %s)", calibrated_slope, embryo_id
        )
    else:
        # Fast mode - use session slope, calculate offset
        calibrated_slope = session_prior.slope_um_per_deg
        calibrated_offset = piezo_center - calibrated_slope * galvo_center

    # --- STEP 6: Store Calibration ---
    embryo.calibration = {
        "slope_um_per_deg": calibrated_slope,
        "offset_um": calibrated_offset,
        "galvo_top_deg": galvo_top,
        "galvo_bottom_deg": galvo_bottom,
        "galvo_calib_top_deg": galvo_center,
        "galvo_calib_bottom_deg": galvo_center + 0.3 * galvo_extent
        if is_bootstrap
        else galvo_center,
        "piezo_calib_top_um": piezo_center,
        "piezo_calib_bottom_um": piezo_center
        if not is_bootstrap
        else piezo_center + calibrated_slope * 0.3 * galvo_extent,
        "r_squared": 0.85,  # Assumed for Vision-based selection
        "method": "fast_vision_guided",
        "bootstrap": is_bootstrap,
        "timestamp": datetime.now().isoformat(),
    }

    # Calculate volume parameters
    extent_um = galvo_extent * calibrated_slope
    total_range_um = extent_um + 2 * z_buffer_um
    recommended_slices = max(30, min(150, int(total_range_um / 0.5)))  # 0.5µm per slice

    embryo.calibration["volume_params"] = {
        "piezo_start_um": calibrated_offset + calibrated_slope * galvo_top - z_buffer_um,
        "piezo_end_um": calibrated_offset + calibrated_slope * galvo_bottom + z_buffer_um,
        "total_range_um": total_range_um,
        "recommended_slices": recommended_slices,
    }

    agent._save_state()

    msg = f"""\u2713 Fast calibration complete for {embryo_id}
  Mode: {"BOOTSTRAP" if is_bootstrap else "FAST"}
  Slope: {calibrated_slope:.2f} \u00b5m/\u00b0
  Offset: {calibrated_offset:.2f} \u00b5m
  Edges: {galvo_top:.3f}\u00b0 to {galvo_bottom:.3f}\u00b0 ({galvo_extent:.3f}\u00b0 extent)
  Total exposures: {total_exposures}
  Recommended slices: {recommended_slices}"""

    logger.info("%s", msg)
    return True, msg, total_exposures


@tool(
    name="calibrate_embryo",
    description="""Run full piezo-galvo calibration for a specific embryo using Claude vision.
This performs:
1. Move to embryo XY position
2. Use Claude vision to detect embryo Z extent (top/bottom edges) by linear sweep
3. Place calibration positions inside the detected edges by inset_fraction of
   the galvo range (default 0.4, matching the v0.4.0 calibration plan)
4. Run adaptive focus sweeps (\u00b115\u00b5m sparse survey, \u00b13\u00b5m dense refinement)
   at each calibration position
5. FFT bandpass scoring with Gaussian fit (R\u00b2 \u2265 0.75 threshold)
6. 2-point linear fit to establish piezo = slope*galvo + offset
7. Store calibration including volume acquisition parameters

Use after detection to prepare an embryo for volume acquisition. Takes ~2-4 minutes per embryo.

The z_buffer_um parameter controls how much empty space is captured above and below the embryo.
Default is 25\u00b5m. Increase for more context (useful for segmentation), decrease for faster
acquisition.""",
    category=ToolCategory.CALIBRATION,
    requires_microscope=True,
    examples=[
        ToolExample("Calibrate embryo 1", {"embryo_id": "embryo_1"}),
        ToolExample("Skip edge detection", {"embryo_id": "embryo_2", "skip_edge_detection": True}),
        ToolExample("More Z padding", {"embryo_id": "embryo_1", "z_buffer_um": 25.0}),
    ],
)
async def calibrate_embryo(
    embryo_id: str,
    skip_edge_detection: bool = False,
    galvo_top: float | None = None,
    galvo_bottom: float | None = None,
    edge_step: float = 0.05,
    edge_max_range: float = 0.5,
    edge_tolerance_deg: float = 0.20,
    inset_fraction: float = 0.4,
    z_buffer_um: float = 25.0,
    use_v04_plan: bool = False,
    context: dict | None = None,
) -> str:
    """Run piezo-galvo calibration with Claude vision edge detection.

    Calibration path: this function mirrors the v0.4.0 `calibrate_embryo_piezo_galvo`
    Bluesky plan (in `gently/hardware/dispim/plans/calibration.py`). After edge
    detection, the detected top/bottom edges are extended outward by
    `edge_tolerance_deg` to define the scan boundaries, and calibration positions
    are inset inward from those scan boundaries by `scan_range * inset_fraction`:

        scan_top    = detected_top    - edge_tolerance_deg
        scan_bottom = detected_bottom + edge_tolerance_deg
        scan_range  = scan_bottom - scan_top
        calib_top   = scan_top + scan_range * inset_fraction
        calib_bottom= scan_bottom - scan_range * inset_fraction

    With the defaults (tolerance=0.20, inset_fraction=0.4), the two calibration
    positions sit 0.2 * scan_range apart - i.e. 40% of the detected galvo range
    when the embryo is small, 20% of the tolerance-extended range overall. That's
    wide enough to give the 2-point slope fit real signal-to-noise. An earlier
    port of this logic used the raw detected range (not tolerance-extended), which
    squeezed the two calibration points to only 20% of the detected range apart
    and produced noise-amplified slopes on small embryos - see commit history.

    An even earlier design picked calibration positions from Claude Vision "feature
    richness" scores; that was removed because Claude sometimes ranked the
    outermost visible edge frame highest, landing calib_top at or beyond the real
    embryo signal.

    The `use_v04_plan` kwarg is an escape hatch reserved for the case where this
    surgical fix is found to be insufficient on hardware. It's intentionally
    unwired (raises NotImplementedError) because delegating to the actual Bluesky
    plan requires a RunEngine and device objects that live on the device layer,
    not on the agent side. If you need it, wire it through the queue server
    plan-submission API. Until then, the surgical path IS the v0.4.0 path.
    """
    import tempfile
    from pathlib import Path

    import numpy as np
    from PIL import Image

    from gently.analysis.core import calculate_focus_score
    from gently.hardware.dispim.claude_client import AsyncClaudeClient

    agent = ctx_get(context, "agent")
    client = ctx_get(context, "client")

    if not agent:
        return "Error: No agent context"

    if not client or not getattr(client, "is_connected", False):
        return f"Error: Not connected to microscope server. Cannot calibrate {embryo_id}."

    if use_v04_plan:
        # Escape hatch reserved for a future hardware-regression follow-up. It is
        # intentionally unwired (delegating to the real Bluesky plan needs a
        # RunEngine + device objects that live on the device layer). Return a
        # clear message instead of raising, so a model that sets this flag gets a
        # graceful answer rather than a hard NotImplementedError — the default
        # surgical path already mirrors v0.4.0 behavior.
        return (
            "use_v04_plan is not available: the default calibration path already "
            "replicates the v0.4.0 plan (edge detection + inset + wide adaptive "
            "sweep). Re-run calibrate_embryo without use_v04_plan."
        )
    logger.info("calibration path: surgical (v0.4.0-equivalent inset + adaptive sweep)")

    embryo, err = get_embryo_or_error(agent, embryo_id)
    if err:
        return err

    # Get session-level calibration prior for cross-embryo learning
    session_prior = agent.experiment.calibration_prior

    # Heuristic calibration for edge detection sweeps
    # Priority: 1) Session prior (cross-embryo learning), 2) Embryo's own calibration, 3) Default
    if session_prior.num_calibrations > 0 and session_prior.r_squared_mean >= 0.75:
        HEURISTIC_SLOPE = session_prior.slope_um_per_deg
        HEURISTIC_OFFSET = session_prior.offset_um
        logger.info(
            "Using session prior: %.1f um/deg, offset %.1f um", HEURISTIC_SLOPE, HEURISTIC_OFFSET
        )
        logger.info(
            "Prior from %d embryo(s), R2=%.3f",
            session_prior.num_calibrations,
            session_prior.r_squared_mean,
        )
    elif embryo.calibration and embryo.calibration.get("slope_um_per_deg"):
        HEURISTIC_SLOPE = embryo.calibration["slope_um_per_deg"]
        HEURISTIC_OFFSET = embryo.calibration.get("offset_um", 0.0)
        logger.info(
            "Using previous embryo calibration: %.1f um/deg, offset %.1f um",
            HEURISTIC_SLOPE,
            HEURISTIC_OFFSET,
        )
    else:
        HEURISTIC_SLOPE = 100.0  # Default empirical value
        HEURISTIC_OFFSET = 0.0
        logger.info("Using default heuristic: %.1f um/deg", HEURISTIC_SLOPE)

    # Track total exposures during calibration
    total_exposures = 0

    try:
        # First move to embryo position
        pos = embryo.stage_position
        if pos and pos.get("x") is not None and pos.get("y") is not None:
            logger.info("Moving to %s position...", embryo_id)
            await client.move_to_position(pos["x"], pos["y"])

        # Initialize Claude client for vision
        claude_vision = AsyncClaudeClient()

        # Track feature richness during edge detection for optimal focus position selection
        # Claude rates each slice 1-10 for how good it would be for focus calibration
        edge_detection_data = []  # List of {galvo, piezo, visible, feature_score}

        # Helper to save image and check embryo presence
        async def check_embryo_at_position(galvo_pos: float) -> bool:
            """Capture image, check embryo presence, and get feature richness score from Claude"""
            nonlocal total_exposures
            piezo_pos = HEURISTIC_SLOPE * galvo_pos + HEURISTIC_OFFSET  # Track light sheet
            result = await client.capture_lightsheet_image(
                piezo_position=float(piezo_pos), galvo_position=float(galvo_pos)
            )
            if result.get("success"):
                total_exposures += 1
            if not result.get("success") or result.get("image") is None:
                return False

            img = result["image"]
            # Select best view from dual-view image
            img_view = select_best_view(img)

            # Save to temp file for Claude
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                temp_path = Path(f.name)
                # Normalize and save
                img_norm = (
                    (img_view - img_view.min()) / (img_view.max() - img_view.min() + 1e-10) * 255
                ).astype(np.uint8)
                Image.fromarray(img_norm).save(temp_path)

            try:
                # Claude now returns (visible, feature_score, description)
                visible, feature_score, description = await claude_vision.detect_embryo_presence(
                    temp_path
                )
                logger.debug(
                    "galvo=%+.3f deg: %s (features=%s/10) - %s...",
                    galvo_pos,
                    "VISIBLE" if visible else "EMPTY",
                    feature_score,
                    description[:40],
                )

                # Record for optimal focus position selection
                edge_detection_data.append(
                    {
                        "galvo": float(galvo_pos),
                        "piezo": float(piezo_pos),
                        "visible": visible,
                        "feature_score": feature_score,
                    }
                )

                # Push edge detection image to viz server
                if agent.viz_server:
                    agent.push_viz(
                        array=img_norm,
                        uid=f"edge_detect_{embryo_id}_{galvo_pos:.3f}",
                        data_type="edge_detection",
                        metadata={
                            "embryo_id": embryo_id,
                            "galvo": float(galvo_pos),
                            "piezo": float(piezo_pos),
                            "visible": visible,
                            "feature_score": feature_score,
                        },
                    )

                return visible
            finally:
                temp_path.unlink(missing_ok=True)

        # === PHASE 1: EDGE DETECTION (unless skipped) ===
        if skip_edge_detection:
            # Use provided galvo positions or defaults
            detected_top = galvo_top if galvo_top is not None else -0.15
            detected_bottom = galvo_bottom if galvo_bottom is not None else 0.15
            logger.info(
                "Skipping edge detection, using galvo range: %.3f deg to %.3f deg",
                detected_top,
                detected_bottom,
            )
        else:
            logger.info("Phase 1: Detecting embryo Z extent with Claude vision...")

            # Detect TOP edge (sweep from center toward negative)
            logger.info("Detecting TOP edge (sweeping galvo toward negative)...")
            detected_top = 0.0
            for galvo in np.arange(0.0, -edge_max_range - edge_step / 2, -edge_step):
                visible = await check_embryo_at_position(galvo)
                if visible:
                    detected_top = galvo
                else:
                    logger.info("Embryo disappeared at galvo=%.3f deg", galvo)
                    break

            # Detect BOTTOM edge (sweep from center toward positive)
            logger.info("Detecting BOTTOM edge (sweeping galvo toward positive)...")
            detected_bottom = 0.0
            for galvo in np.arange(0.0, edge_max_range + edge_step / 2, edge_step):
                visible = await check_embryo_at_position(galvo)
                if visible:
                    detected_bottom = galvo
                else:
                    logger.info("Embryo disappeared at galvo=%.3f deg", galvo)
                    break

            logger.info(
                "Detected embryo extent: top=%.3f deg, bottom=%.3f deg, range=%.3f deg (~%.0f um)",
                detected_top,
                detected_bottom,
                detected_bottom - detected_top,
                (detected_bottom - detected_top) * 100,
            )

        # === PHASE 2: COMPUTE CALIBRATION POSITIONS (v0.4.0 inset formula) ===
        # Extend the detected edges outward by `edge_tolerance_deg` to get the
        # scan boundaries, then inset inward from those boundaries by
        # `inset_fraction` of the scan range. This matches the v0.4.0 Bluesky
        # plan `calibrate_embryo_piezo_galvo` (plans/calibration.py:745-782)
        # exactly. Using the tolerance-extended range (instead of the raw
        # detected range) gives calibration positions enough separation to
        # make the 2-point slope fit robust on small embryos.
        scan_top = detected_top - edge_tolerance_deg
        scan_bottom = detected_bottom + edge_tolerance_deg
        scan_range = scan_bottom - scan_top
        inset_amount = scan_range * inset_fraction
        calib_top = scan_top + inset_amount
        calib_bottom = scan_bottom - inset_amount
        # Store on the embryo so the viz server's "scan top"/"scan bot" dashed
        # lines match what the calibration actually scanned.
        detected_bottom - detected_top
        logger.info(
            "Scan boundaries (edges +/- %.3f tolerance): %.3f to %.3f deg (range %.3f)",
            edge_tolerance_deg,
            scan_top,
            scan_bottom,
            scan_range,
        )
        logger.info(
            "Calibration positions (%.0f%% inset from scan boundary):"
            " top=%.3f deg, bottom=%.3f deg (%.3f deg apart)",
            inset_fraction * 100,
            calib_top,
            calib_bottom,
            calib_bottom - calib_top,
        )

        # === PHASE 3: ADAPTIVE FOCUS SWEEPS AT CALIBRATION POSITIONS ===
        logger.info("Phase 3: Adaptive focus sweeps at calibration positions...")

        results = {}

        for galvo_name, galvo_pos in [("top", calib_top), ("bottom", calib_bottom)]:
            # Expected piezo from heuristic - used only to center the sparse
            # survey window. The window is ±15µm (see _adaptive_focus_sweep),
            # so a stale heuristic off by <10µm still catches the true peak.
            expected_piezo = galvo_pos * HEURISTIC_SLOPE + HEURISTIC_OFFSET

            result_dict, sweep_exposures = await _adaptive_focus_sweep(
                client=client,
                agent=agent,
                embryo_id=embryo_id,
                galvo_name=galvo_name,
                galvo_pos=galvo_pos,
                expected_piezo=expected_piezo,
                session_prior=session_prior,
                select_best_view=select_view_and_crop_roi,  # Use ROI cropping for focus
                calculate_focus_score=calculate_focus_score,
            )

            results[galvo_name] = result_dict
            total_exposures += sweep_exposures

            # Check for sweep failure
            if result_dict["r_squared"] < 0.5:
                logger.warning(
                    "Low confidence for %s (R2=%.3f)", galvo_name, result_dict["r_squared"]
                )

        # === PHASE 4: CALCULATE 2-POINT LINEAR CALIBRATION ===
        g_top = results["top"]["galvo"]
        p_top = results["top"]["piezo"]
        g_bottom = results["bottom"]["galvo"]
        p_bottom = results["bottom"]["piezo"]

        slope = (p_bottom - p_top) / (g_bottom - g_top)
        offset = p_top - slope * g_top

        # Calculate volume acquisition parameters from embryo extent
        # Use detected edges (not calibration positions) for full coverage
        # Add buffer zone above and below embryo (configurable via z_buffer_um)
        # Conversion: 0.01° ≈ 1µm, so z_buffer_um / 100 gives degrees
        volume_buffer_deg = z_buffer_um / 100.0

        galvo_center = (detected_top + detected_bottom) / 2
        galvo_amplitude = (detected_bottom - detected_top) / 2 + volume_buffer_deg

        # Calculate piezo range using the linear relationship
        # Include buffer in the range calculation
        galvo_top_with_buffer = detected_top - volume_buffer_deg
        galvo_bottom_with_buffer = detected_bottom + volume_buffer_deg
        piezo_at_top = slope * galvo_top_with_buffer + offset
        piezo_at_bottom = slope * galvo_bottom_with_buffer + offset
        piezo_center = (piezo_at_top + piezo_at_bottom) / 2
        piezo_amplitude = abs(piezo_at_bottom - piezo_at_top) / 2

        # Store calibration
        embryo.calibration = {
            "slope_um_per_deg": slope,
            "offset_um": offset,
            "galvo_top_deg": detected_top,
            "galvo_bottom_deg": detected_bottom,
            "galvo_calib_top_deg": g_top,
            "galvo_calib_bottom_deg": g_bottom,
            "piezo_top_um": p_top,
            "piezo_bottom_um": p_bottom,
            # Volume acquisition parameters
            "galvo_center": galvo_center,
            "galvo_amplitude": galvo_amplitude,
            "piezo_center": piezo_center,
            "piezo_amplitude": piezo_amplitude,
            "z_buffer_um": z_buffer_um,
            "r_squared_top": results["top"]["r_squared"],
            "r_squared_bottom": results["bottom"]["r_squared"],
        }

        # Update session prior for cross-embryo learning
        avg_r_squared = (results["top"]["r_squared"] + results["bottom"]["r_squared"]) / 2
        extent_deg = detected_bottom - detected_top
        session_prior.update_from_calibration(
            slope=slope,
            offset=offset,
            r_squared=avg_r_squared,
            extent_deg=extent_deg,
        )
        logger.info(
            "Updated session prior (now %d embryo(s), avg R2=%.3f)",
            session_prior.num_calibrations,
            session_prior.r_squared_mean,
        )

        # Add to focus history
        for name in ["top", "bottom"]:
            embryo.add_focus_datapoint(
                galvo=results[name]["galvo"],
                piezo=results[name]["piezo"],
                score=results[name]["max_score"],
                r_squared=results[name]["r_squared"],
                method="calibration",
                algorithm="fft_bandpass",
            )

        # Record total light exposure from calibration (50ms default exposure)
        if total_exposures > 0:
            embryo.record_exposure(exposure_ms=50.0, num_frames=total_exposures)

        # Push calibration summary plot to viz server
        if agent.viz_server:
            try:
                summary_plot = generate_calibration_summary_plot(
                    embryo_id=embryo_id,
                    galvo_top=g_top,
                    galvo_bottom=g_bottom,
                    piezo_top=p_top,
                    piezo_bottom=p_bottom,
                    slope=slope,
                    offset=offset,
                    r_squared_top=results["top"]["r_squared"],
                    r_squared_bottom=results["bottom"]["r_squared"],
                )
                agent.push_viz(
                    array=summary_plot,
                    uid=f"calibration_summary_{embryo_id}",
                    data_type="calibration_summary",
                    metadata={
                        "embryo_id": embryo_id,
                        "slope": slope,
                        "offset": offset,
                        "galvo_top": g_top,
                        "galvo_bottom": g_bottom,
                        "piezo_top": p_top,
                        "piezo_bottom": p_bottom,
                        "r_squared_top": results["top"]["r_squared"],
                        "r_squared_bottom": results["bottom"]["r_squared"],
                    },
                )
            except Exception as plot_err:
                logger.warning("Failed to generate calibration summary plot: %s", plot_err)

        agent._mark_significant_action("calibration")

        return (
            f"\u2713 Calibrated {embryo_id}\n"
            f"  Embryo extent: galvo {detected_top:.3f}\u00b0 to {detected_bottom:.3f}\u00b0 "
            f"(~{(detected_bottom - detected_top) * 100:.0f}\u00b5m)\n"
            f"  Slope: {slope:.2f} \u00b5m/deg\n"
            f"  Offset: {offset:.2f} \u00b5m (piezo at galvo=0)\n"
            f"  Top: galvo={g_top:.3f}\u00b0 \u2192 piezo={p_top:.1f}\u00b5m\n"
            f"  Bottom: galvo={g_bottom:.3f}\u00b0 \u2192 piezo={p_bottom:.1f}\u00b5m\n"
            f"  Volume params: galvo={galvo_center:.3f}\u00b0\u00b1{galvo_amplitude:.3f}\u00b0, "
            f"piezo={piezo_center:.1f}\u00b5m\u00b1{piezo_amplitude:.1f}\u00b5m\n"
            f"  Z buffer: \u00b1{z_buffer_um:.0f}\u00b5m above/below embryo"
        )

    except Exception as e:
        import traceback

        return f"Error calibrating embryo: {str(e)}\n{traceback.format_exc()}"


@tool(
    name="calibrate_all_embryos",
    description="""Run piezo-galvo calibration for all detected embryos sequentially.
Uses Claude vision to detect embryo Z extent for each embryo, then runs focus sweeps.
Use after detecting multiple embryos.

The z_buffer_um parameter controls how much empty space is captured above and below each embryo.
Default is 15\u00b5m.""",
    category=ToolCategory.CALIBRATION,
    requires_microscope=True,
    examples=[
        ToolExample("Calibrate all embryos", {}),
        ToolExample("Quick calibration without edge detection", {"skip_edge_detection": True}),
        ToolExample("More Z padding for all", {"z_buffer_um": 25.0}),
    ],
)
async def calibrate_all_embryos(
    embryo_ids: list[str] | None = None,
    skip_edge_detection: bool = False,
    z_buffer_um: float = 25.0,
    context: dict | None = None,
) -> str:
    """Calibrate all embryos sequentially with Claude vision"""
    agent = ctx_get(context, "agent")

    if not agent:
        return "Error: No agent context"

    # Get embryos to calibrate
    if embryo_ids:
        ids_to_calibrate = embryo_ids
    else:
        ids_to_calibrate = list(agent.experiment.embryos.keys())

    if not ids_to_calibrate:
        return "No embryos to calibrate. Detect embryos first."

    results = []
    for i, eid in enumerate(ids_to_calibrate, 1):
        logger.info("=" * 60)
        logger.info("Calibrating %s (%d/%d)", eid, i, len(ids_to_calibrate))
        logger.info("=" * 60)

        result = await calibrate_embryo(
            embryo_id=eid,
            skip_edge_detection=skip_edge_detection,
            z_buffer_um=z_buffer_um,
            context=context,
        )
        # Get first two lines of result
        lines = result.split("\n")
        summary = lines[0] if len(lines) == 1 else f"{lines[0]} {lines[1]}"
        results.append(f"{eid}: {summary}")

    return f"Calibration complete for {len(ids_to_calibrate)} embryo(s):\n" + "\n".join(results)


def _calibration_quality_score(cal: dict) -> float:
    """Composite fit-quality score for a calibration dict.

    Uses ``min(r_squared_top, r_squared_bottom)`` — the WORSE of the two
    fits matters more than the average, because acquisitions span both
    galvo extremes and the weaker end dominates focus quality. Falls
    back to 0 when R² fields are missing.
    """
    if not cal:
        return 0.0
    top = cal.get("r_squared_top")
    bot = cal.get("r_squared_bottom")
    if top is None and bot is None:
        # Last-resort: average R² field if present (older formats).
        return float(cal.get("r_squared", 0.0) or 0.0)
    vals = [v for v in (top, bot) if v is not None]
    return float(min(vals)) if vals else 0.0


def _format_quality(cal: dict) -> str:
    """Human-readable summary of a calibration's fit quality."""
    if not cal:
        return "no fit"
    top = cal.get("r_squared_top")
    bot = cal.get("r_squared_bottom")
    if top is not None and bot is not None:
        return f"R²={top:.2f}/{bot:.2f} (min={min(top, bot):.2f})"
    if top is not None:
        return f"R²(top)={top:.2f}"
    if bot is not None:
        return f"R²(bot)={bot:.2f}"
    avg = cal.get("r_squared")
    if avg is not None:
        return f"R²={avg:.2f}"
    return "no R² recorded"


@tool(
    name="apply_calibration_to_embryos",
    description="""Copy one embryo's calibration onto one or more target embryos. Useful when
one embryo has a strong piezo-galvo fit and others can borrow it as-is.

**Quality metric is R²**, NOT galvo extent. The right "source" is the embryo with the
highest ``min(r_squared_top, r_squared_bottom)`` — both ends of the galvo sweep need a clean
Gaussian fit for the calibration to hold up at the volume edges. Wider extent just means a
bigger embryo; it does not imply better calibration.

**Auto-pick by quality**: pass ``source_embryo_id="auto"`` to let the tool pick the
calibration with the highest min-R² across all currently-calibrated embryos. The response
includes the per-embryo R² ranking so the agent can narrate the choice.

Pass ``target_embryo_ids=None`` (or omit it) to apply to ALL other embryos in the
experiment that are not skipped.

Caveats — calibration is position-dependent: piezo-galvo slope drifts across the XY field,
and embryos may sit at slightly different Z depths. The agent should warn the user about this
when applying broadly. Best practice is still to calibrate each embryo individually; this
tool is for "good enough" propagation or when individual calibration would burn too much
light dose.""",
    category=ToolCategory.HARDWARE,
    requires_microscope=False,
    examples=[
        ToolExample(
            "Auto-pick the best calibration and apply it",
            {"source_embryo_id": "auto"},
        ),
        ToolExample(
            "Apply embryo_3's calibration to embryos 1 and 2",
            {"source_embryo_id": "embryo_3", "target_embryo_ids": ["embryo_1", "embryo_2"]},
        ),
    ],
)
def apply_calibration_to_embryos(
    source_embryo_id: str,
    target_embryo_ids: list[str] | None = None,
    overwrite_existing: bool = True,
    context: dict | None = None,
) -> str:
    """Broadcast one embryo's calibration to others.

    ``source_embryo_id="auto"`` picks the embryo with the highest
    ``min(r_squared_top, r_squared_bottom)`` calibration. The response
    includes the full per-embryo R² ranking so the agent has the data
    to narrate the choice (and the user can audit).
    """
    from gently.harness.tools.helpers import require_agent

    agent, err = require_agent(context)
    if err:
        return err

    # Auto-pick by quality.
    ranking_lines = []
    if source_embryo_id == "auto" or source_embryo_id == "best":
        ranked = []
        for eid, emb in agent.experiment.embryos.items():
            if emb.should_skip or not emb.calibration:
                continue
            ranked.append((eid, _calibration_quality_score(emb.calibration), emb.calibration))
        if not ranked:
            return (
                "No calibrated embryos available for auto-pick. "
                "Run calibrate_embryo / calibrate_all_embryos first."
            )
        ranked.sort(key=lambda t: t[1], reverse=True)
        source_embryo_id = ranked[0][0]
        ranking_lines.append("Ranking by min(R²_top, R²_bot):")
        for eid, _score, cal in ranked:
            mark = " ← chosen" if eid == source_embryo_id else ""
            ranking_lines.append(f"  {eid}: {_format_quality(cal)}{mark}")

    if source_embryo_id not in agent.experiment.embryos:
        return f"Source embryo '{source_embryo_id}' not found."
    source = agent.experiment.embryos[source_embryo_id]
    if not source.calibration:
        return f"{source_embryo_id} has no calibration data to copy. Run calibrate_embryo first."

    if target_embryo_ids is None:
        target_embryo_ids = [
            eid
            for eid, e in agent.experiment.embryos.items()
            if eid != source_embryo_id and not e.should_skip
        ]

    if not target_embryo_ids:
        return "No target embryos. Nothing to do."

    applied, skipped = [], []
    for tid in target_embryo_ids:
        if tid not in agent.experiment.embryos:
            skipped.append((tid, "not found"))
            continue
        tgt = agent.experiment.embryos[tid]
        if tgt.calibration and not overwrite_existing:
            skipped.append((tid, "already calibrated (overwrite_existing=False)"))
            continue
        # Deep-ish copy so subsequent mutations don't alias the source.
        import copy

        tgt.calibration = copy.deepcopy(source.calibration)
        applied.append(tid)

    lines = []
    if ranking_lines:
        lines.extend(ranking_lines)
        lines.append("")
    lines.append(
        f"Applied {source_embryo_id}'s calibration "
        f"({_format_quality(source.calibration)}) to {len(applied)} embryo(s):"
    )
    lines.append("  " + (", ".join(applied) if applied else "(none)"))
    if skipped:
        lines.append(f"Skipped: {', '.join(f'{tid} ({reason})' for tid, reason in skipped)}")
    lines.append(
        "Note: calibration is position-dependent — piezo-galvo slope can drift "
        "across the XY field. Verify with a quick acquisition on each target "
        "before committing to a full timelapse."
    )
    return "\n".join(lines)
