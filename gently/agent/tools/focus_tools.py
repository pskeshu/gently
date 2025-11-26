"""
Focus Tools

Tools for microscope focus operations including fine focus adjustment
using FFT bandpass or gradient algorithms.
"""

from typing import Dict, List, Optional
import numpy as np

from ..tool_registry import tool, ToolCategory, ToolExample
from ..tool_helpers import get_embryo_or_error

# Import focus analysis functions from core
from gently.analysis.core import (
    calculate_focus_score,
    fit_focus_curve,
    FocusAlgorithm,
    FocusAnalysisConfig,
    FitFunction,
)


@tool(
    name="fine_focus",
    description="""Perform fine focus adjustment by scanning piezo positions and finding optimal focus using image analysis.
Sweeps the piezo through a range of positions, captures lightsheet images at each position, calculates focus scores
using FFT bandpass or gradient algorithm, fits a Gaussian curve, and optionally moves to the best focus position.

Use when user says "focus", "fine focus", "adjust focus", "find best focus", or after moving to an embryo position.
Default sweep is ±10μm with 1μm steps (21 positions). Algorithm options: 'fft_bandpass' (default, best for lightsheet) or 'gradient'.

Returns the optimal piezo position and fit quality (R²). Higher R² indicates more reliable focus detection.""",
    category=ToolCategory.CALIBRATION,
    requires_microscope=True,
    examples=[
        ToolExample("Focus at current position", {}),
        ToolExample("Fine focus with gradient algorithm", {"algorithm": "gradient"}),
        ToolExample("Focus sweep ±5um with 0.5um steps", {"range_um": 5.0, "step_um": 0.5}),
    ],
)
async def fine_focus(
    range_um: float = 10.0,
    step_um: float = 1.0,
    center_um: Optional[float] = None,
    algorithm: str = "fft_bandpass",
    move_to_best: bool = True,
    galvo_position: float = 0.0,
    context: Dict = None
) -> str:
    """
    Perform fine focus sweep to find optimal piezo position.

    Parameters
    ----------
    range_um : float
        Half-range of sweep in micrometers (±range_um around center)
    step_um : float
        Step size in micrometers
    center_um : float, optional
        Center position for sweep. If None, uses current piezo position (0.0)
    algorithm : str
        Focus algorithm: 'fft_bandpass' (default) or 'gradient'
    move_to_best : bool
        Whether to move piezo to best focus position after sweep
    galvo_position : float
        Galvo position to use during sweep (default: 0.0)
    context : dict
        Execution context with client
    """
    client = context.get('client')

    if not client:
        return "Error: No microscope client connected"

    # Validate algorithm
    valid_algorithms = ['fft_bandpass', 'gradient', 'volath', 'variance']
    if algorithm not in valid_algorithms:
        return f"Error: Unknown algorithm '{algorithm}'. Valid options: {valid_algorithms}"

    # Determine center position
    if center_um is None:
        center_um = 0.0  # Default to center of piezo range

    # Generate sweep positions
    num_steps = int(2 * range_um / step_um) + 1
    positions = np.linspace(center_um - range_um, center_um + range_um, num_steps)

    print(f"  Fine focus sweep: {center_um:.1f} ± {range_um:.1f} μm, step {step_um:.1f} μm")
    print(f"  Algorithm: {algorithm}")
    print(f"  Positions: {len(positions)}")

    try:
        # Capture images at each position
        images = []
        captured_positions = []

        for i, pos in enumerate(positions):
            print(f"  [{i+1}/{len(positions)}] Capturing at piezo={pos:.1f}μm...", end='\r')

            result = await client.capture_lightsheet_image(
                piezo_position=float(pos),
                galvo_position=float(galvo_position)
            )

            if result.get('success') and result.get('image') is not None:
                images.append(result['image'])
                captured_positions.append(pos)
            else:
                print(f"\n  Warning: Failed to capture at position {pos:.1f}μm")

        print()  # New line after progress

        if len(images) < 3:
            return f"Error: Only captured {len(images)} images, need at least 3 for focus analysis"

        # Calculate focus scores
        print(f"  Calculating focus scores...")
        scores = []
        config = FocusAnalysisConfig(algorithm=algorithm)

        for i, img in enumerate(images):
            score = calculate_focus_score(img, algorithm=algorithm, config=config)
            scores.append(score)

        scores = np.array(scores)
        captured_positions = np.array(captured_positions)

        # Find best position
        max_idx = np.argmax(scores)
        best_measured_position = captured_positions[max_idx]
        best_measured_score = scores[max_idx]

        # Fit Gaussian curve for sub-step precision
        print(f"  Fitting Gaussian curve...")
        try:
            fitted_positions, fitted_scores, fit_params, r_squared = fit_focus_curve(
                captured_positions, scores, FitFunction.GAUSSIAN.value
            )

            # Extract optimal position from fit
            if r_squared >= 0.5:  # Reasonable fit
                # Gaussian params: [amplitude, mu, sigma, offset]
                best_position = float(fit_params[1])  # mu is the peak position
                fit_quality = "good" if r_squared >= 0.75 else "moderate"

                # Check if fitted peak is within sweep range
                if best_position < captured_positions.min() or best_position > captured_positions.max():
                    print(f"  Warning: Fitted peak ({best_position:.2f}μm) outside sweep range, using max score position")
                    best_position = best_measured_position
                    fit_quality = "fallback"
            else:
                best_position = best_measured_position
                fit_quality = "poor"
                print(f"  Warning: Poor fit (R²={r_squared:.3f}), using max score position")

        except Exception as e:
            print(f"  Warning: Curve fitting failed ({e}), using max score position")
            best_position = best_measured_position
            r_squared = 0.0
            fit_quality = "failed"

        # Move to best position if requested
        if move_to_best:
            print(f"  Moving to optimal position: {best_position:.2f}μm")
            await client.capture_lightsheet_image(
                piezo_position=float(best_position),
                galvo_position=float(galvo_position)
            )

        # Build result message
        result_lines = [
            f"✓ Fine focus complete",
            f"  Optimal position: {best_position:.2f} μm",
            f"  Fit quality: {fit_quality} (R²={r_squared:.3f})",
            f"  Algorithm: {algorithm}",
            f"  Sweep: {captured_positions.min():.1f} to {captured_positions.max():.1f} μm ({len(captured_positions)} positions)",
        ]

        if move_to_best:
            result_lines.append(f"  Moved to: {best_position:.2f} μm")

        # Add score statistics
        score_range = scores.max() - scores.min()
        score_cv = np.std(scores) / np.mean(scores) if np.mean(scores) > 0 else 0
        result_lines.append(f"  Score variation: {score_cv:.1%} (higher is better for focus detection)")

        return "\n".join(result_lines)

    except Exception as e:
        import traceback
        return f"Error during fine focus: {str(e)}\n{traceback.format_exc()}"


@tool(
    name="get_focus_score",
    description="""Calculate focus score for the current lightsheet image without moving the piezo.
Captures a single lightsheet image and returns its focus quality score using the specified algorithm.
Use to check focus quality at current position or compare different positions manually.
If piezo_position is not specified, uses CURRENT position (preserves focus after fine_focus).
Algorithm options: 'fft_bandpass' (default), 'gradient', 'volath', 'variance'.""",
    category=ToolCategory.ANALYSIS,
    requires_microscope=True,
    examples=[
        ToolExample("Check focus quality", {}),
        ToolExample("Get gradient focus score", {"algorithm": "gradient"}),
    ],
)
async def get_focus_score(
    piezo_position: float = None,
    galvo_position: float = 0.0,
    algorithm: str = "fft_bandpass",
    context: Dict = None
) -> str:
    """
    Get focus score for current or specified position.

    Parameters
    ----------
    piezo_position : float, optional
        Piezo position to capture at. If None, uses current position.
    galvo_position : float
        Galvo position to use
    algorithm : str
        Focus algorithm: 'fft_bandpass', 'gradient', 'volath', or 'variance'
    context : dict
        Execution context
    """
    client = context.get('client')

    if not client:
        return "Error: No microscope client connected"

    valid_algorithms = ['fft_bandpass', 'gradient', 'volath', 'variance']
    if algorithm not in valid_algorithms:
        return f"Error: Unknown algorithm '{algorithm}'. Valid options: {valid_algorithms}"

    try:
        # If no piezo position specified, use current position
        if piezo_position is None:
            piezo_position = await client.get_piezo_position()
            print(f"  Using current piezo position: {piezo_position:.1f}μm")

        # Capture image
        result = await client.capture_lightsheet_image(
            piezo_position=float(piezo_position),
            galvo_position=float(galvo_position)
        )

        if not result.get('success') or result.get('image') is None:
            return f"Error: Failed to capture image at piezo={piezo_position}μm"

        image = result['image']

        # Calculate focus score
        config = FocusAnalysisConfig(algorithm=algorithm)
        score = calculate_focus_score(image, algorithm=algorithm, config=config)

        return (
            f"Focus score at piezo={piezo_position:.1f}μm, galvo={galvo_position:.1f}V\n"
            f"  Algorithm: {algorithm}\n"
            f"  Score: {score:.2e}\n"
            f"  Image shape: {image.shape}"
        )

    except Exception as e:
        return f"Error getting focus score: {str(e)}"
