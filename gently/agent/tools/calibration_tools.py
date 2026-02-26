"""
Calibration Tools

Tools for microscope piezo-galvo calibration including adaptive focus sweeps,
binary edge search, and full/fast calibration routines.
"""

from typing import Dict, List, Optional, Tuple
from datetime import datetime
import json
import asyncio

import numpy as np

from ..tool_registry import tool, ToolCategory, ToolExample
from ..tool_helpers import get_embryo_or_error
from ..state import CalibrationPrior
from gently.analysis.core import AdaptiveSweepState, FitFunction, fit_focus_curve
from gently.visualization.plots import (
    generate_focus_curve_plot,
    generate_calibration_summary_plot,
    generate_edge_detection_plot,
)
from .hardware_common import select_best_view, crop_to_embryo_roi, select_view_and_crop_roi


async def _adaptive_focus_sweep(
    client,
    copilot,
    embryo_id: str,
    galvo_name: str,
    galvo_pos: float,
    expected_piezo: float,
    session_prior: CalibrationPrior,
    select_best_view,
    calculate_focus_score,
) -> Tuple[Dict, int]:
    """
    Adaptive focus sweep with early stopping.

    Uses sparse-then-dense approach:
    1. Sparse survey (5µm steps) to find approximate peak with early stopping
    2. Dense refinement (0.5µm steps) around peak with early stopping

    Parameters
    ----------
    client : QueueServerClient
        Microscope client for image capture
    copilot : MicroscopyCopilot
        Copilot for viz server access
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
    DENSE_STEP = 0.5   # µm - fine steps for refinement
    DENSE_RANGE = 3.0  # µm - narrow window around peak
    MIN_R_SQUARED = 0.75

    # Get adaptive range from prior
    sparse_range = session_prior.get_reduced_sweep_range(base_range_um=5.0)

    print(f"\n  === ADAPTIVE {galvo_name.upper()} FOCUS SWEEP at galvo={galvo_pos:.3f}° ===")
    print(f"  Using adaptive range: ±{sparse_range:.1f}µm (prior: {session_prior.num_calibrations} calibrations)")

    # --- PHASE 1: SPARSE SURVEY ---
    print(f"  Phase 1: Sparse survey ±{sparse_range:.1f}µm, {SPARSE_STEP}µm steps...")

    sparse_state = AdaptiveSweepState()
    sparse_positions = np.arange(
        expected_piezo - sparse_range,
        expected_piezo + sparse_range + SPARSE_STEP,
        SPARSE_STEP
    )

    for piezo in sparse_positions:
        result = await client.capture_lightsheet_image(
            piezo_position=float(piezo),
            galvo_position=float(galvo_pos)
        )
        if result.get('success'):
            total_exposures += 1
        if result.get('success') and result.get('image') is not None:
            img = select_best_view(result['image'])
            score = calculate_focus_score(img, algorithm='fft_bandpass')

            # Add to adaptive state and check for early stopping
            decision = sparse_state.add_point(float(piezo), float(score))

            # Push to viz server
            if copilot.viz_server:
                copilot.push_viz(
                    array=img,
                    uid=f"focus_sparse_{embryo_id}_{galvo_name}_{piezo:.1f}",
                    data_type="focus_sweep",
                    metadata={
                        'embryo_id': embryo_id,
                        'sweep': 'sparse',
                        'galvo_name': galvo_name,
                        'galvo': float(galvo_pos),
                        'piezo': float(piezo),
                        'score': float(score),
                        'peak_detected': sparse_state.peak_detected,
                    }
                )

            if decision['should_stop']:
                print(f"    Early stop: {decision['reason']} (confidence: {decision['confidence']:.2f})")
                break

    sparse_best, sparse_r2 = sparse_state.get_best_position()
    print(f"    Sparse best: {sparse_best:.1f}µm (R²={sparse_r2:.3f}, {len(sparse_state.positions)} points)")

    # --- PHASE 2: DENSE REFINEMENT ---
    print(f"  Phase 2: Dense refinement ±{DENSE_RANGE}µm around {sparse_best:.1f}µm, {DENSE_STEP}µm steps...")

    dense_state = AdaptiveSweepState()
    dense_positions = np.arange(
        sparse_best - DENSE_RANGE,
        sparse_best + DENSE_RANGE + DENSE_STEP,
        DENSE_STEP
    )

    for piezo in dense_positions:
        result = await client.capture_lightsheet_image(
            piezo_position=float(piezo),
            galvo_position=float(galvo_pos)
        )
        if result.get('success'):
            total_exposures += 1
        if result.get('success') and result.get('image') is not None:
            img = select_best_view(result['image'])
            score = calculate_focus_score(img, algorithm='fft_bandpass')

            # Add to adaptive state and check for early stopping
            decision = dense_state.add_point(float(piezo), float(score))

            print(f"    piezo={piezo:.1f}: score={score:.2e}")

            # Push to viz server
            if copilot.viz_server:
                copilot.push_viz(
                    array=img,
                    uid=f"focus_dense_{embryo_id}_{galvo_name}_{piezo:.1f}",
                    data_type="focus_sweep",
                    metadata={
                        'embryo_id': embryo_id,
                        'sweep': 'dense',
                        'galvo_name': galvo_name,
                        'galvo': float(galvo_pos),
                        'piezo': float(piezo),
                        'score': float(score),
                        'r_squared': dense_state.current_r_squared,
                    }
                )

            if decision['should_stop']:
                print(f"    Early stop: {decision['reason']} (confidence: {decision['confidence']:.2f})")
                break

    # Get final result
    best_piezo, r_squared = dense_state.get_best_position()

    # Validate result
    if r_squared < MIN_R_SQUARED and len(dense_state.positions) >= 5:
        # Try fitting again with all data
        try:
            positions = np.array(dense_state.positions)
            scores = np.array(dense_state.scores)
            _, _, params, r_squared = fit_focus_curve(
                positions, scores, FitFunction.GAUSSIAN.value
            )
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

    print(f"    → Best focus: piezo={best_piezo:.2f}µm (R²={r_squared:.3f}, {fit_quality})")
    print(f"    Total exposures: {total_exposures} (sparse: {len(sparse_state.positions)}, dense: {len(dense_state.positions)})")

    # Build result dict
    result_dict = {
        'galvo': galvo_pos,
        'piezo': best_piezo,
        'max_score': float(max(dense_state.scores)) if dense_state.scores else 0.0,
        'r_squared': r_squared,
        'fit_params': None,  # Will be set by focus curve plot generation
    }

    # Push focus curve plot
    if copilot.viz_server and len(dense_state.positions) >= 4:
        try:
            positions = np.array(dense_state.positions)
            scores = np.array(dense_state.scores)
            try:
                _, _, fit_params, _ = fit_focus_curve(positions, scores, FitFunction.GAUSSIAN.value)
                result_dict['fit_params'] = fit_params
            except Exception:
                fit_params = None

            plot_img = generate_focus_curve_plot(
                positions=positions,
                scores=scores,
                best_position=best_piezo,
                fit_params=fit_params,
                r_squared=r_squared,
                title=f'{embryo_id} - {galvo_name.upper()} Focus Curve (Adaptive)',
            )
            copilot.push_viz(
                array=plot_img,
                uid=f"focus_curve_{embryo_id}_{galvo_name}",
                data_type="focus_plot",
                metadata={
                    'embryo_id': embryo_id,
                    'galvo_name': galvo_name,
                    'galvo': float(galvo_pos),
                    'best_piezo': best_piezo,
                    'r_squared': r_squared,
                    'adaptive': True,
                    'exposures': total_exposures,
                }
            )
        except Exception as plot_err:
            print(f"    Warning: Failed to generate focus plot: {plot_err}")

    return result_dict, total_exposures


async def _fine_focus_sweep(
    client,
    copilot,
    embryo_id: str,
    galvo_name: str,
    galvo_pos: float,
    expected_piezo: float,
    select_best_view,
    calculate_focus_score,
) -> Tuple[Dict, int]:
    """
    Fine-only focus sweep - assumes heuristic is close.

    Used when Claude has identified feature-rich positions during edge detection.
    Since heuristic piezo position is already good, we only do a fine sweep.

    Parameters
    ----------
    client : QueueServerClient
        Microscope client for image capture
    copilot : MicroscopyCopilot
        Copilot for viz server access
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
    FINE_RANGE = 5.0   # ±5µm around expected
    FINE_STEP = 0.5    # 0.5µm steps
    MIN_R_SQUARED = 0.75
    total_exposures = 0

    print(f"\n  === FINE-ONLY {galvo_name.upper()} FOCUS SWEEP at galvo={galvo_pos:.3f}° ===")
    print(f"  Sweeping ±{FINE_RANGE}µm around heuristic ({expected_piezo:.1f}µm), {FINE_STEP}µm steps...")

    positions = np.arange(
        expected_piezo - FINE_RANGE,
        expected_piezo + FINE_RANGE + FINE_STEP,
        FINE_STEP
    )

    piezo_scores = []
    for piezo in positions:
        result = await client.capture_lightsheet_image(
            piezo_position=float(piezo),
            galvo_position=float(galvo_pos)
        )
        if result.get('success'):
            total_exposures += 1
        if result.get('success') and result.get('image') is not None:
            img = select_best_view(result['image'])
            score = calculate_focus_score(img, algorithm='fft_bandpass')
            piezo_scores.append((float(piezo), float(score)))

            print(f"    piezo={piezo:.1f}: score={score:.2e}")

            # Push to viz server
            if copilot.viz_server:
                copilot.push_viz(
                    array=img,
                    uid=f"focus_fine_{embryo_id}_{galvo_name}_{piezo:.1f}",
                    data_type="focus_sweep",
                    metadata={
                        'embryo_id': embryo_id,
                        'sweep': 'fine_only',
                        'galvo_name': galvo_name,
                        'galvo': float(galvo_pos),
                        'piezo': float(piezo),
                        'score': float(score),
                    }
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
            print(f"    Warning: Gaussian fit failed ({fit_err}), using max score position")
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
    print(f"    → Best focus: piezo={best_piezo:.2f}µm (R²={r_squared:.3f}, {fit_quality})")
    print(f"    Total exposures: {total_exposures}")

    # Build result dict
    result_dict = {
        'galvo': galvo_pos,
        'piezo': best_piezo,
        'max_score': max_score,
        'r_squared': r_squared,
        'fit_params': fit_params,
    }

    # Push focus curve plot
    if copilot.viz_server and len(piezo_scores) >= 4:
        try:
            positions_arr = np.array([p for p, _ in piezo_scores])
            scores_arr = np.array([s for _, s in piezo_scores])

            plot_img = generate_focus_curve_plot(
                positions=positions_arr,
                scores=scores_arr,
                best_position=best_piezo,
                fit_params=fit_params,
                r_squared=r_squared,
                title=f'{embryo_id} - {galvo_name.upper()} Focus Curve (Fine-Only)',
            )
            copilot.push_viz(
                array=plot_img,
                uid=f"focus_curve_{embryo_id}_{galvo_name}",
                data_type="focus_plot",
                metadata={
                    'embryo_id': embryo_id,
                    'galvo_name': galvo_name,
                    'galvo': float(galvo_pos),
                    'best_piezo': best_piezo,
                    'r_squared': r_squared,
                    'fine_only': True,
                    'exposures': total_exposures,
                }
            )
        except Exception as plot_err:
            print(f"    Warning: Failed to generate focus plot: {plot_err}")

    return result_dict, total_exposures


# ============================================================================
# FAST CALIBRATION ALGORITHM (Vision-Guided)
# ============================================================================

async def hybrid_focus_selection(
    images: List[np.ndarray],
    offsets: List[float],
    claude_vision,
    copilot,
    embryo_id: str,
    fft_confidence_threshold: float = 0.85
) -> Tuple[int, str, float]:
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
    copilot : MicroscopyCopilot
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
    scores = [calculate_focus_score(img, 'fft_bandpass') for img in images]
    max_score = max(scores)
    best_idx = scores.index(max_score)

    # Check FFT confidence
    score_ratios = [s / max_score if max_score > 0 else 0 for s in scores]
    sorted_ratios = sorted(score_ratios, reverse=True)
    second_best_ratio = sorted_ratios[1] if len(sorted_ratios) > 1 else 0

    confidence_ratio = 1.0 / second_best_ratio if second_best_ratio > 0 else float('inf')

    if second_best_ratio < fft_confidence_threshold:
        # FFT is confident - best is >15% better than second
        print(f"    FFT confident: position {best_idx} (ratio {confidence_ratio:.2f})")
        return best_idx, 'fft', confidence_ratio

    # Stage 2: FFT ambiguous - ask Vision
    print(f"    FFT ambiguous (ratio {confidence_ratio:.2f}), consulting Vision...")

    import tempfile
    from pathlib import Path
    from PIL import Image
    from gently.analysis.core import create_focus_montage

    # Create montage
    labels = [chr(ord('A') + i) for i in range(len(images))]
    montage = create_focus_montage(images, labels=labels, offsets=offsets)

    # Save to temp file
    with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
        montage_path = Path(f.name)
        Image.fromarray(montage).save(montage_path)

    try:
        # Call Vision API
        vision_idx, vision_label, reasoning = await claude_vision.select_best_focus(
            montage_path, offsets, labels
        )
        print(f"    Vision selected: {vision_label} ({reasoning})")

        # Push montage to viz server
        if copilot.viz_server:
            copilot.push_viz(
                array=montage,
                uid=f"focus_montage_{embryo_id}",
                data_type="focus_montage",
                metadata={
                    'embryo_id': embryo_id,
                    'offsets': offsets,
                    'fft_scores': scores,
                    'vision_pick': vision_label,
                    'reasoning': reasoning,
                }
            )

        return vision_idx, 'vision', None

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
    copilot,
    embryo_id: str,
    piezo_heuristic: float,
    max_range: float = 0.25,
    num_iterations: int = 4
) -> Tuple[float, int]:
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
    copilot : MicroscopyCopilot
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

    sign = -1 if direction == 'top' else 1
    low, high = 0.0, max_range * sign
    last_visible = 0.0
    exposures = 0

    print(f"    Binary edge search ({direction}): range 0 to {high:.3f}°")

    for i in range(num_iterations):
        mid = (low + high) / 2

        # Calculate piezo position using heuristic slope
        HEURISTIC_SLOPE = 100.0  # µm/degree
        piezo = piezo_heuristic + HEURISTIC_SLOPE * mid

        # Capture image
        result = await client.capture_lightsheet_image(
            piezo_position=float(piezo),
            galvo_position=float(mid)
        )
        exposures += 1

        if not result.get('success') or result.get('image') is None:
            # Assume not visible on failure
            high = mid if sign > 0 else low
            low = mid if sign < 0 else high
            continue

        img = select_best_view(result['image'])

        # Check visibility with Vision
        img_norm = ((img - img.min()) / (img.max() - img.min() + 1e-8) * 255).astype(np.uint8)

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            temp_path = Path(f.name)
            Image.fromarray(img_norm).save(temp_path)

        try:
            visible, feature_score, desc = await claude_vision.detect_embryo_presence(temp_path)
        finally:
            try:
                temp_path.unlink()
            except Exception:
                pass

        print(f"      iter {i+1}: galvo={mid:.3f}°, visible={visible}, features={feature_score}")

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
    embryo_id: str,
    context: Dict,
    z_buffer_um: float = 25.0
) -> Tuple[bool, str, int]:
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
        Tool context with copilot and client
    z_buffer_um : float
        Z padding above/below embryo for volume acquisition

    Returns
    -------
    tuple
        (success, message, total_exposures)
    """
    from gently.claude_client import AsyncClaudeClient

    copilot = context.get('copilot')
    client = context.get('client')

    if not copilot:
        return False, "Error: No copilot context", 0

    if not client or not getattr(client, 'is_connected', False):
        return False, f"Error: Not connected to microscope server", 0

    embryo, err = get_embryo_or_error(copilot, embryo_id)
    if err:
        return False, err, 0

    # Get session prior
    session_prior = copilot.calibration_prior

    # Initialize Claude Vision client
    claude_vision = AsyncClaudeClient()

    total_exposures = 0
    HEURISTIC_SLOPE = 100.0  # Default µm/degree

    print(f"\n=== FAST CALIBRATION: {embryo_id} ===")

    # Check if this is bootstrap (first embryo) or fast mode
    is_bootstrap = not session_prior.is_ready_for_fast_calibration()

    if is_bootstrap:
        print(f"  Mode: BOOTSTRAP (establishing session slope)")
    else:
        print(f"  Mode: FAST (using session slope {session_prior.slope_um_per_deg:.2f} µm/°)")

    # --- STEP 1: Binary Edge Detection ---
    print(f"\n  Step 1: Binary edge detection...")

    # Start with heuristic piezo at galvo=0
    piezo_heuristic = session_prior.offset_um if session_prior.num_calibrations > 0 else 0.0

    galvo_top, exp_top = await binary_edge_search(
        client, claude_vision, 'top', copilot, embryo_id, piezo_heuristic
    )
    total_exposures += exp_top

    galvo_bottom, exp_bottom = await binary_edge_search(
        client, claude_vision, 'bottom', copilot, embryo_id, piezo_heuristic
    )
    total_exposures += exp_bottom

    print(f"  Edges detected: top={galvo_top:.3f}°, bottom={galvo_bottom:.3f}°")

    # Validate edges
    if galvo_top >= galvo_bottom:
        return False, f"Invalid edges: top {galvo_top:.3f}° >= bottom {galvo_bottom:.3f}°", total_exposures

    galvo_center = (galvo_top + galvo_bottom) / 2
    galvo_extent = galvo_bottom - galvo_top

    # --- STEP 2: Focus at Center Position ---
    print(f"\n  Step 2: Focus at center (galvo={galvo_center:.3f}°)...")

    # Use session slope or heuristic
    slope = session_prior.slope_um_per_deg if session_prior.is_ready_for_fast_calibration() else HEURISTIC_SLOPE
    piezo_expected = slope * galvo_center + session_prior.offset_um

    # Capture 3-point focus grid (±2µm)
    focus_offsets = [-2.0, 0.0, 2.0]
    focus_images = []

    for offset in focus_offsets:
        result = await client.capture_lightsheet_image(
            piezo_position=float(piezo_expected + offset),
            galvo_position=float(galvo_center)
        )
        total_exposures += 1

        if result.get('success') and result.get('image') is not None:
            img = select_best_view(result['image'])
            focus_images.append(img)
        else:
            focus_images.append(np.zeros((100, 100), dtype=np.uint8))

    # --- STEP 3: Hybrid Focus Selection ---
    print(f"\n  Step 3: Hybrid focus selection...")

    best_idx, method, confidence = await hybrid_focus_selection(
        focus_images, focus_offsets, claude_vision, copilot, embryo_id
    )

    embryo_offset = focus_offsets[best_idx]
    piezo_center = piezo_expected + embryo_offset

    print(f"  Selected: offset {embryo_offset:+.1f}µm via {method}")

    # --- STEP 4: Refinement if Edge Picked ---
    if best_idx in [0, len(focus_offsets) - 1]:
        print(f"\n  Step 4: Refining (edge position picked)...")

        # Extend search in direction of edge
        if best_idx == 0:
            extend_offsets = [-3.0, -4.0]
        else:
            extend_offsets = [3.0, 4.0]

        for ext_offset in extend_offsets:
            result = await client.capture_lightsheet_image(
                piezo_position=float(piezo_expected + ext_offset),
                galvo_position=float(galvo_center)
            )
            total_exposures += 1

            if result.get('success') and result.get('image') is not None:
                img = select_best_view(result['image'])
                focus_images.append(img)
                focus_offsets.append(ext_offset)

        # Re-run hybrid selection
        best_idx, method, confidence = await hybrid_focus_selection(
            focus_images, focus_offsets, claude_vision, copilot, embryo_id
        )
        embryo_offset = focus_offsets[best_idx]
        piezo_center = piezo_expected + embryo_offset
        print(f"  Refined: offset {embryo_offset:+.1f}µm via {method}")
    else:
        print(f"\n  Step 4: Skipped (center position picked)")

    # --- STEP 5: Bootstrap - Establish Session Slope ---
    if is_bootstrap:
        print(f"\n  Step 5: Bootstrap - calibrating second position for slope...")

        # Pick second position 30% away from center
        galvo_second = galvo_center + 0.3 * galvo_extent
        piezo_expected_second = HEURISTIC_SLOPE * galvo_second + session_prior.offset_um

        # Capture 3-point focus grid at second position
        focus_images_2 = []
        for offset in [-2.0, 0.0, 2.0]:
            result = await client.capture_lightsheet_image(
                piezo_position=float(piezo_expected_second + offset),
                galvo_position=float(galvo_second)
            )
            total_exposures += 1

            if result.get('success') and result.get('image') is not None:
                img = select_best_view(result['image'])
                focus_images_2.append(img)
            else:
                focus_images_2.append(np.zeros((100, 100), dtype=np.uint8))

        best_idx_2, _, _ = await hybrid_focus_selection(
            focus_images_2, [-2.0, 0.0, 2.0], claude_vision, copilot, embryo_id
        )
        piezo_second = piezo_expected_second + [-2.0, 0.0, 2.0][best_idx_2]

        # Calculate slope from two points
        calibrated_slope = (piezo_second - piezo_center) / (galvo_second - galvo_center)
        calibrated_offset = piezo_center - calibrated_slope * galvo_center

        # Lock session slope
        session_prior.lock_session_slope(calibrated_slope, 0.85, embryo_id)
        print(f"  Session slope locked: {calibrated_slope:.2f} µm/° (bootstrap embryo: {embryo_id})")
    else:
        # Fast mode - use session slope, calculate offset
        calibrated_slope = session_prior.slope_um_per_deg
        calibrated_offset = piezo_center - calibrated_slope * galvo_center

    # --- STEP 6: Store Calibration ---
    embryo.calibration = {
        'slope_um_per_deg': calibrated_slope,
        'offset_um': calibrated_offset,
        'galvo_top_deg': galvo_top,
        'galvo_bottom_deg': galvo_bottom,
        'galvo_calib_top_deg': galvo_center,
        'galvo_calib_bottom_deg': galvo_center + 0.3 * galvo_extent if is_bootstrap else galvo_center,
        'piezo_calib_top_um': piezo_center,
        'piezo_calib_bottom_um': piezo_center if not is_bootstrap else piezo_center + calibrated_slope * 0.3 * galvo_extent,
        'r_squared': 0.85,  # Assumed for Vision-based selection
        'method': 'fast_vision_guided',
        'bootstrap': is_bootstrap,
        'timestamp': datetime.now().isoformat(),
    }

    # Calculate volume parameters
    extent_um = galvo_extent * calibrated_slope
    total_range_um = extent_um + 2 * z_buffer_um
    recommended_slices = max(30, min(150, int(total_range_um / 0.5)))  # 0.5µm per slice

    embryo.calibration['volume_params'] = {
        'piezo_start_um': calibrated_offset + calibrated_slope * galvo_top - z_buffer_um,
        'piezo_end_um': calibrated_offset + calibrated_slope * galvo_bottom + z_buffer_um,
        'total_range_um': total_range_um,
        'recommended_slices': recommended_slices,
    }

    copilot._save_state()

    msg = f"""\u2713 Fast calibration complete for {embryo_id}
  Mode: {'BOOTSTRAP' if is_bootstrap else 'FAST'}
  Slope: {calibrated_slope:.2f} \u00b5m/\u00b0
  Offset: {calibrated_offset:.2f} \u00b5m
  Edges: {galvo_top:.3f}\u00b0 to {galvo_bottom:.3f}\u00b0 ({galvo_extent:.3f}\u00b0 extent)
  Total exposures: {total_exposures}
  Recommended slices: {recommended_slices}"""

    print(f"\n{msg}")
    return True, msg, total_exposures


@tool(
    name="calibrate_embryo",
    description="""Run full piezo-galvo calibration for a specific embryo using Claude vision.
This performs:
1. Move to embryo XY position
2. Use Claude vision to detect embryo Z extent (top/bottom edges) AND rate feature richness
3. Select two feature-rich positions (score \u22656/10) that are \u226530% of embryo range apart
4. Run focus sweeps at selected positions:
   - Fine-only (\u00b15\u00b5m) if feature-rich positions found (faster, ~20 exposures per position)
   - Adaptive coarse+fine if fallback positions used (~30-40 exposures per position)
5. FFT bandpass scoring with Gaussian fit (R\u00b2 \u2265 0.75 threshold)
6. 2-point linear fit to establish piezo = slope*galvo + offset
7. Store calibration including volume acquisition parameters

Use after detection to prepare an embryo for volume acquisition. Takes ~2-4 minutes per embryo.

The z_buffer_um parameter controls how much empty space is captured above and below the embryo.
Default is 15\u00b5m. Increase for more context (useful for segmentation), decrease for faster acquisition.""",
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
    galvo_top: float = None,
    galvo_bottom: float = None,
    edge_step: float = 0.05,
    edge_max_range: float = 0.5,
    inset_fraction: float = 0.4,
    z_buffer_um: float = 25.0,
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

    if not client or not getattr(client, 'is_connected', False):
        return f"Error: Not connected to microscope server. Cannot calibrate {embryo_id}."

    embryo, err = get_embryo_or_error(copilot, embryo_id)
    if err:
        return err

    # Get session-level calibration prior for cross-embryo learning
    session_prior = copilot.experiment.calibration_prior

    # Heuristic calibration for edge detection sweeps
    # Priority: 1) Session prior (cross-embryo learning), 2) Embryo's own calibration, 3) Default
    if session_prior.num_calibrations > 0 and session_prior.r_squared_mean >= 0.75:
        HEURISTIC_SLOPE = session_prior.slope_um_per_deg
        HEURISTIC_OFFSET = session_prior.offset_um
        print(f"  Using session prior: {HEURISTIC_SLOPE:.1f} \u00b5m/deg, offset {HEURISTIC_OFFSET:.1f} \u00b5m")
        print(f"    Prior from {session_prior.num_calibrations} embryo(s), R\u00b2={session_prior.r_squared_mean:.3f}")
    elif embryo.calibration and embryo.calibration.get('slope_um_per_deg'):
        HEURISTIC_SLOPE = embryo.calibration['slope_um_per_deg']
        HEURISTIC_OFFSET = embryo.calibration.get('offset_um', 0.0)
        print(f"  Using previous embryo calibration: {HEURISTIC_SLOPE:.1f} \u00b5m/deg, offset {HEURISTIC_OFFSET:.1f} \u00b5m")
    else:
        HEURISTIC_SLOPE = 100.0  # Default empirical value
        HEURISTIC_OFFSET = 0.0
        print(f"  Using default heuristic: {HEURISTIC_SLOPE:.1f} \u00b5m/deg")

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

        # Track feature richness during edge detection for optimal focus position selection
        # Claude rates each slice 1-10 for how good it would be for focus calibration
        edge_detection_data = []  # List of {galvo, piezo, visible, feature_score}

        # Helper to save image and check embryo presence
        async def check_embryo_at_position(galvo_pos: float) -> bool:
            """Capture image, check embryo presence, and get feature richness score from Claude"""
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
            img_view = select_best_view(img)

            # Save to temp file for Claude
            with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
                temp_path = Path(f.name)
                # Normalize and save
                img_norm = ((img_view - img_view.min()) / (img_view.max() - img_view.min() + 1e-10) * 255).astype(np.uint8)
                Image.fromarray(img_norm).save(temp_path)

            try:
                # Claude now returns (visible, feature_score, description)
                visible, feature_score, description = await claude_vision.detect_embryo_presence(temp_path)
                print(f"    galvo={galvo_pos:+.3f}\u00b0: {'VISIBLE' if visible else 'EMPTY'} (features={feature_score}/10) - {description[:40]}...")

                # Record for optimal focus position selection
                edge_detection_data.append({
                    'galvo': float(galvo_pos),
                    'piezo': float(piezo_pos),
                    'visible': visible,
                    'feature_score': feature_score,
                })

                # Push edge detection image to viz server
                if copilot.viz_server:
                    copilot.push_viz(
                        array=img_norm,
                        uid=f"edge_detect_{embryo_id}_{galvo_pos:.3f}",
                        data_type="edge_detection",
                        metadata={
                            'embryo_id': embryo_id,
                            'galvo': float(galvo_pos),
                            'piezo': float(piezo_pos),
                            'visible': visible,
                            'feature_score': feature_score,
                        }
                    )

                return visible
            finally:
                temp_path.unlink(missing_ok=True)

        # === PHASE 1: EDGE DETECTION (unless skipped) ===
        if skip_edge_detection:
            # Use provided galvo positions or defaults
            detected_top = galvo_top if galvo_top is not None else -0.15
            detected_bottom = galvo_bottom if galvo_bottom is not None else 0.15
            print(f"\n  Skipping edge detection, using galvo range: {detected_top:.3f}\u00b0 to {detected_bottom:.3f}\u00b0")
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
                    print(f"    \u2192 Embryo disappeared at galvo={galvo:.3f}\u00b0")
                    break

            # Detect BOTTOM edge (sweep from center toward positive)
            print(f"\n  Detecting BOTTOM edge (sweeping galvo toward positive)...")
            detected_bottom = 0.0
            for galvo in np.arange(0.0, edge_max_range + edge_step/2, edge_step):
                visible = await check_embryo_at_position(galvo)
                if visible:
                    detected_bottom = galvo
                else:
                    print(f"    \u2192 Embryo disappeared at galvo={galvo:.3f}\u00b0")
                    break

            print(f"\n  Detected embryo extent:")
            print(f"    Top edge: galvo={detected_top:.3f}\u00b0")
            print(f"    Bottom edge: galvo={detected_bottom:.3f}\u00b0")
            print(f"    Range: {detected_bottom - detected_top:.3f}\u00b0 (~{(detected_bottom - detected_top) * 100:.0f}\u00b5m)")

        # === PHASE 2: SELECT OPTIMAL FOCUS POSITIONS ===
        # Use Claude's feature richness scores to find best positions for calibration
        # Need two positions that are ≥30% of embryo range apart for good 2-point slope
        galvo_range = detected_bottom - detected_top
        min_separation = galvo_range * 0.3  # Minimum 30% separation for good slope

        use_fine_only = False  # Will be set to True if we found good feature-rich positions

        if edge_detection_data and not skip_edge_detection:
            # Filter to visible positions with good features (score >= 6)
            good_positions = [p for p in edge_detection_data if p['visible'] and p['feature_score'] >= 6]

            if len(good_positions) >= 2:
                # Sort by feature score (best first)
                good_positions.sort(key=lambda x: x['feature_score'], reverse=True)

                # Best overall position
                calib_pos_1 = good_positions[0]

                # Find second position that's far enough from first (≥30% apart)
                calib_pos_2 = None
                for pos in good_positions[1:]:
                    if abs(pos['galvo'] - calib_pos_1['galvo']) >= min_separation:
                        calib_pos_2 = pos
                        break

                if calib_pos_2 is None and len(good_positions) > 1:
                    # Fallback: use second best regardless of distance
                    calib_pos_2 = good_positions[1]

                if calib_pos_2 is not None:
                    # Assign to top/bottom based on galvo position
                    if calib_pos_1['galvo'] < calib_pos_2['galvo']:
                        calib_top, calib_bottom = calib_pos_1['galvo'], calib_pos_2['galvo']
                        score_top, score_bottom = calib_pos_1['feature_score'], calib_pos_2['feature_score']
                    else:
                        calib_top, calib_bottom = calib_pos_2['galvo'], calib_pos_1['galvo']
                        score_top, score_bottom = calib_pos_2['feature_score'], calib_pos_1['feature_score']

                    actual_separation = abs(calib_bottom - calib_top) / galvo_range * 100
                    use_fine_only = True  # Feature-rich positions, use fine-only sweep

                    print(f"\n  Optimal focus positions (selected by Claude feature richness):")
                    print(f"    Position 1: galvo={calib_top:.3f}\u00b0 (features={score_top}/10)")
                    print(f"    Position 2: galvo={calib_bottom:.3f}\u00b0 (features={score_bottom}/10)")
                    print(f"    Separation: {actual_separation:.0f}% of embryo range")
                    print(f"    \u2192 Using FINE-ONLY focus sweeps (heuristic is good enough)")
                else:
                    # Only one good position found, fall back to inset
                    calib_top = detected_top + galvo_range * inset_fraction
                    calib_bottom = detected_bottom - galvo_range * inset_fraction
                    print(f"\n  Calibration positions (fallback - only 1 feature-rich position found):")
                    print(f"    Top calibration: galvo={calib_top:.3f}\u00b0")
                    print(f"    Bottom calibration: galvo={calib_bottom:.3f}\u00b0")
            else:
                # Not enough good positions (score >= 6), try positions with any visibility
                visible_positions = [p for p in edge_detection_data if p['visible']]
                if len(visible_positions) >= 2:
                    # Sort by feature score and pick best two that are apart
                    visible_positions.sort(key=lambda x: x['feature_score'], reverse=True)
                    calib_pos_1 = visible_positions[0]
                    calib_pos_2 = None
                    for pos in visible_positions[1:]:
                        if abs(pos['galvo'] - calib_pos_1['galvo']) >= min_separation:
                            calib_pos_2 = pos
                            break
                    if calib_pos_2 is None and len(visible_positions) > 1:
                        calib_pos_2 = visible_positions[1]

                    if calib_pos_2 is not None:
                        if calib_pos_1['galvo'] < calib_pos_2['galvo']:
                            calib_top, calib_bottom = calib_pos_1['galvo'], calib_pos_2['galvo']
                        else:
                            calib_top, calib_bottom = calib_pos_2['galvo'], calib_pos_1['galvo']
                        print(f"\n  Calibration positions (moderate features, using adaptive sweep):")
                        print(f"    Top: galvo={calib_top:.3f}\u00b0 (features={calib_pos_1['feature_score']}/10)")
                        print(f"    Bottom: galvo={calib_bottom:.3f}\u00b0 (features={calib_pos_2['feature_score']}/10)")
                    else:
                        calib_top = detected_top + galvo_range * inset_fraction
                        calib_bottom = detected_bottom - galvo_range * inset_fraction
                else:
                    calib_top = detected_top + galvo_range * inset_fraction
                    calib_bottom = detected_bottom - galvo_range * inset_fraction
                    print(f"\n  Calibration positions (fallback to {inset_fraction*100:.0f}% inset):")
                    print(f"    Top calibration: galvo={calib_top:.3f}\u00b0")
                    print(f"    Bottom calibration: galvo={calib_bottom:.3f}\u00b0")
        else:
            # No edge detection data, use traditional inset method
            calib_top = detected_top + galvo_range * inset_fraction
            calib_bottom = detected_bottom - galvo_range * inset_fraction
            print(f"\n  Calibration positions (interior, {inset_fraction*100:.0f}% inset from edges):")
            print(f"    Top calibration: galvo={calib_top:.3f}\u00b0")
            print(f"    Bottom calibration: galvo={calib_bottom:.3f}\u00b0")

        # === PHASE 3: FOCUS SWEEPS AT CALIBRATION POSITIONS ===
        # Use fine-only if we found feature-rich positions, otherwise adaptive sweep

        if use_fine_only:
            print(f"\n  Phase 2: Fine-only focus sweeps at feature-rich positions...")
        else:
            print(f"\n  Phase 2: Adaptive focus sweeps at calibration positions...")

        results = {}

        for galvo_name, galvo_pos in [("top", calib_top), ("bottom", calib_bottom)]:
            # Expected piezo from heuristic
            expected_piezo = galvo_pos * HEURISTIC_SLOPE + HEURISTIC_OFFSET

            if use_fine_only:
                # Fine-only sweep - heuristic is good enough at feature-rich positions
                result_dict, sweep_exposures = await _fine_focus_sweep(
                    client=client,
                    copilot=copilot,
                    embryo_id=embryo_id,
                    galvo_name=galvo_name,
                    galvo_pos=galvo_pos,
                    expected_piezo=expected_piezo,
                    select_best_view=select_view_and_crop_roi,  # Use ROI cropping for focus
                    calculate_focus_score=calculate_focus_score,
                )
            else:
                # Adaptive sweep with early stopping for lower-confidence positions
                result_dict, sweep_exposures = await _adaptive_focus_sweep(
                    client=client,
                    copilot=copilot,
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
            if result_dict['r_squared'] < 0.5:
                print(f"  Warning: Low confidence for {galvo_name} (R\u00b2={result_dict['r_squared']:.3f})")

        # === PHASE 4: CALCULATE 2-POINT LINEAR CALIBRATION ===
        g_top = results['top']['galvo']
        p_top = results['top']['piezo']
        g_bottom = results['bottom']['galvo']
        p_bottom = results['bottom']['piezo']

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
            'z_buffer_um': z_buffer_um,
            'r_squared_top': results['top']['r_squared'],
            'r_squared_bottom': results['bottom']['r_squared'],
        }

        # Update session prior for cross-embryo learning
        avg_r_squared = (results['top']['r_squared'] + results['bottom']['r_squared']) / 2
        extent_deg = detected_bottom - detected_top
        session_prior.update_from_calibration(
            slope=slope,
            offset=offset,
            r_squared=avg_r_squared,
            extent_deg=extent_deg,
        )
        print(f"\n  Updated session prior (now {session_prior.num_calibrations} embryo(s), avg R\u00b2={session_prior.r_squared_mean:.3f})")

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

        # Push calibration summary plot to viz server
        if copilot.viz_server:
            try:
                summary_plot = generate_calibration_summary_plot(
                    embryo_id=embryo_id,
                    galvo_top=g_top,
                    galvo_bottom=g_bottom,
                    piezo_top=p_top,
                    piezo_bottom=p_bottom,
                    slope=slope,
                    offset=offset,
                    r_squared_top=results['top']['r_squared'],
                    r_squared_bottom=results['bottom']['r_squared'],
                )
                copilot.push_viz(
                    array=summary_plot,
                    uid=f"calibration_summary_{embryo_id}",
                    data_type="calibration_summary",
                    metadata={
                        'embryo_id': embryo_id,
                        'slope': slope,
                        'offset': offset,
                        'galvo_top': g_top,
                        'galvo_bottom': g_bottom,
                        'piezo_top': p_top,
                        'piezo_bottom': p_bottom,
                        'r_squared_top': results['top']['r_squared'],
                        'r_squared_bottom': results['bottom']['r_squared'],
                    }
                )
            except Exception as plot_err:
                print(f"  Warning: Failed to generate calibration summary plot: {plot_err}")

        copilot._mark_significant_action("calibration")

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
    embryo_ids: List[str] = None,
    skip_edge_detection: bool = False,
    z_buffer_um: float = 25.0,
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
            z_buffer_um=z_buffer_um,
            context=context
        )
        # Get first two lines of result
        lines = result.split('\n')
        summary = lines[0] if len(lines) == 1 else f"{lines[0]} {lines[1]}"
        results.append(f"{eid}: {summary}")

    return f"Calibration complete for {len(ids_to_calibrate)} embryo(s):\n" + "\n".join(results)
