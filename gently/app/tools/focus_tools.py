"""
Focus Tools

Tools for microscope focus operations including fine focus adjustment
using FFT bandpass or gradient algorithms.

Focus measurements are logged to each embryo's focus_history, building
a sample-aware focus map over time. This enables:
- Tracking the piezo-galvo relationship per embryo
- Detecting focus drift during long timelapses
- Predicting optimal focus for future acquisitions
"""

from datetime import datetime
from typing import Any

import numpy as np

# Import focus analysis functions from core
from gently.analysis.core import (
    FitFunction,
    FocusAnalysisConfig,
    calculate_focus_score,
    fit_focus_curve,
)
from gently.harness.tools.helpers import ctx_get
from gently.harness.tools.registry import ToolCategory, ToolExample, tool


@tool(
    name="fine_focus",
    description="""Perform fine focus adjustment by scanning piezo positions and finding
optimal focus using image analysis.
Sweeps the piezo through a range of positions, captures lightsheet images at each position,
calculates focus scores using FFT bandpass or gradient algorithm, fits a Gaussian curve,
and optionally moves to the best focus position.

Use when user says "focus", "fine focus", "adjust focus", "find best focus", or after
moving to an embryo position. Default sweep is ±3μm around 4μm with 1μm steps (7
positions). Algorithm options: 'fft_bandpass' (default, best for lightsheet) or 'gradient'.

If embryo_id is provided, logs the focus measurement to the embryo's focus_history for
drift tracking and future reference. Returns the optimal piezo position and fit quality (R²).
Higher R² indicates more reliable focus detection.""",
    category=ToolCategory.CALIBRATION,
    requires_microscope=True,
    examples=[
        ToolExample("Focus at current position", {}),
        ToolExample("Fine focus with gradient algorithm", {"algorithm": "gradient"}),
        ToolExample("Focus sweep ±5um with 0.5um steps", {"range_um": 5.0, "step_um": 0.5}),
        ToolExample("Focus for specific embryo", {"embryo_id": "embryo_2"}),
    ],
)
async def fine_focus(
    range_um: float = 3.0,
    step_um: float = 1.0,
    center_um: float | None = 4.0,
    algorithm: str = "fft_bandpass",
    move_to_best: bool = True,
    galvo_position: float = 0.0,
    embryo_id: str | None = None,
    context: dict | None = None,
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
    embryo_id : str, optional
        Embryo to associate this focus measurement with. If provided, logs to embryo's
        focus_history.
    context : dict
        Execution context with client and agent
    """
    client = ctx_get(context, "client")
    agent = ctx_get(context, "agent")

    if not client:
        return "Error: No microscope client connected"

    # Validate algorithm
    valid_algorithms = ["fft_bandpass", "gradient", "volath", "variance"]
    if algorithm not in valid_algorithms:
        return f"Error: Unknown algorithm '{algorithm}'. Valid options: {valid_algorithms}"

    # Determine center position (default is 4.0 for galvo=0)
    if center_um is None:
        center_um = 4.0

    # Generate sweep positions
    num_steps = int(2 * range_um / step_um) + 1
    positions = np.linspace(center_um - range_um, center_um + range_um, num_steps)

    try:
        # Capture images at each position
        images = []
        captured_positions: Any = []

        for _i, pos in enumerate(positions):
            result = await client.capture_lightsheet_image(
                piezo_position=float(pos), galvo_position=float(galvo_position)
            )

            if result.get("success") and result.get("image") is not None:
                images.append(result["image"])
                captured_positions.append(pos)

        if len(images) < 3:
            return f"Error: Only captured {len(images)} images, need at least 3 for focus analysis"

        # Calculate focus scores
        scores: Any = []
        config = FocusAnalysisConfig(algorithm=algorithm)

        for _i, img in enumerate(images):
            score = calculate_focus_score(img, algorithm=algorithm, config=config)
            scores.append(score)

        scores = np.array(scores)
        captured_positions = np.array(captured_positions)

        # Find best position
        max_idx = np.argmax(scores)
        best_measured_position = captured_positions[max_idx]
        best_measured_score = scores[max_idx]

        # Fit Gaussian curve for sub-step precision
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
                if (
                    best_position < captured_positions.min()
                    or best_position > captured_positions.max()
                ):
                    best_position = best_measured_position
                    fit_quality = "fallback (peak outside range)"
            else:
                best_position = best_measured_position
                fit_quality = "poor"

        except Exception:
            best_position = best_measured_position
            r_squared = 0.0
            fit_quality = "failed"

        # Move to best position if requested
        if move_to_best:
            await client.capture_lightsheet_image(
                piezo_position=float(best_position),
                galvo_position=float(galvo_position),
            )

        # Log focus datapoint to embryo's focus_history if embryo_id provided
        logged_to_embryo = False
        if embryo_id and agent:
            embryo = agent.experiment.get_embryo_by_any_name(embryo_id)
            if embryo:
                embryo.add_focus_datapoint(
                    galvo=galvo_position,
                    piezo=best_position,
                    score=float(best_measured_score),
                    r_squared=float(r_squared),
                    method="fine_focus",
                    algorithm=algorithm,
                )
                logged_to_embryo = True
                # Track light exposure: num_positions + 1 if moved to best
                num_exposures = len(captured_positions) + (1 if move_to_best else 0)
                embryo.record_exposure(exposure_ms=50.0, num_frames=num_exposures)

        # Build result message
        result_lines = [
            "✓ Fine focus complete",
            f"  Optimal position: {best_position:.2f} μm",
            f"  Fit quality: {fit_quality} (R²={r_squared:.3f})",
            f"  Algorithm: {algorithm}",
            f"  Sweep: {captured_positions.min():.1f} to {captured_positions.max():.1f} μm"
            f" ({len(captured_positions)} positions)",
        ]

        if move_to_best:
            result_lines.append(f"  Moved to: {best_position:.2f} μm")

        if logged_to_embryo:
            result_lines.append(f"  Logged to: {embryo_id} focus history")

        # Add score statistics
        scores.max() - scores.min()
        score_cv = np.std(scores) / np.mean(scores) if np.mean(scores) > 0 else 0
        result_lines.append(
            f"  Score variation: {score_cv:.1%} (higher is better for focus detection)"
        )

        return "\n".join(result_lines)

    except Exception as e:
        import traceback

        return f"Error during fine focus: {str(e)}\n{traceback.format_exc()}"


@tool(
    name="get_focus_score",
    description="""Calculate focus score for the current lightsheet image without moving the piezo.
Captures a single lightsheet image and returns its focus quality score using the specified
algorithm.
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
    piezo_position: float | None = None,
    galvo_position: float = 0.0,
    algorithm: str = "fft_bandpass",
    context: dict | None = None,
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
    client = ctx_get(context, "client")

    if not client:
        return "Error: No microscope client connected"

    valid_algorithms = ["fft_bandpass", "gradient", "volath", "variance"]
    if algorithm not in valid_algorithms:
        return f"Error: Unknown algorithm '{algorithm}'. Valid options: {valid_algorithms}"

    try:
        # If no piezo position specified, use current position
        if piezo_position is None:
            piezo_position = await client.get_piezo_position()

        # Capture image
        result = await client.capture_lightsheet_image(
            piezo_position=float(piezo_position), galvo_position=float(galvo_position)
        )

        if not result.get("success") or result.get("image") is None:
            return f"Error: Failed to capture image at piezo={piezo_position}μm"

        image = result["image"]

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


@tool(
    name="get_focus_history",
    description="""Get the focus history for an embryo showing all piezo-galvo measurements
over time.
Shows drift rate, piezo-galvo fit, and individual measurements. Use to understand how focus
has changed during a timelapse and whether recalibration is needed.""",
    category=ToolCategory.ANALYSIS,
    requires_microscope=False,
    examples=[
        ToolExample("Check focus history for embryo 2", {"embryo_id": "embryo_2"}),
    ],
)
async def get_focus_history(embryo_id: str, context: dict | None = None) -> str:
    """
    Get focus measurement history for an embryo.

    Parameters
    ----------
    embryo_id : str
        Embryo to query focus history for
    context : dict
        Execution context with agent
    """
    agent = ctx_get(context, "agent")

    if not agent:
        return "Error: No agent context available"

    embryo = agent.experiment.get_embryo_by_any_name(embryo_id)
    if not embryo:
        return f"Error: Embryo '{embryo_id}' not found"

    if not embryo.focus_history:
        return f"No focus measurements recorded for {embryo_id}"

    # Get summary
    summary = embryo.get_focus_summary()

    # Show recent measurements
    lines = [f"Focus history for {embryo_id}:", ""]
    lines.append(summary)
    lines.append("")
    lines.append("Recent measurements:")

    for fp in embryo.focus_history[-10:]:  # Last 10
        age_mins = (datetime.now() - fp.timestamp).total_seconds() / 60
        lines.append(
            f"  {fp.timestamp.strftime('%H:%M:%S')} ({age_mins:.0f}m ago): "
            f"galvo={fp.galvo:.2f}, piezo={fp.piezo:.2f}µm, "
            f"R²={fp.r_squared:.3f} [{fp.method}]"
        )

    # Check if refocus needed
    if embryo.needs_refocus(max_age_minutes=60):
        lines.append("")
        lines.append("⚠ Focus data is stale (>60 min). Consider running fine_focus.")

    return "\n".join(lines)
