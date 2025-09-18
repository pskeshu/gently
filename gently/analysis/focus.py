"""
Gently DiSPIM Focus Analysis
============================

Focus-specific analysis functions separated from general analysis module.
Provides clean, functional interfaces for focus analysis without plan dependencies.

Key principle: Pure functions that take data → return results.
No device dependencies, no Bluesky plan integration here.
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass

# Import from parent analysis module
from ..analysis import (
    calculate_focus_score, analyze_focus_stack, fit_focus_curve,
    find_curve_maximum, FocusAnalysisConfig, FocusResult
)
from ..detection import get_embryo_focus_roi


@dataclass
class FocusDataPoint:
    """Single focus measurement data point"""
    position: float
    score: float
    image: np.ndarray
    roi: Optional[Tuple[int, int, int, int]] = None


@dataclass
class FocusSweepResult:
    """Result of a focus sweep with analysis"""
    success: bool
    best_position: float
    best_score: float
    all_data: List[FocusDataPoint]
    r_squared: float = 0.0
    error_message: Optional[str] = None


def score_single_image(image: np.ndarray,
                      config: FocusAnalysisConfig,
                      detect_roi: bool = True) -> Tuple[float, Optional[Tuple[int, int, int, int]]]:
    """
    Score a single image for focus quality

    Pure function: image + config → score + ROI

    Parameters
    ----------
    image : np.ndarray
        Input image to analyze
    config : FocusAnalysisConfig
        Focus analysis configuration
    detect_roi : bool
        Whether to detect embryo ROI automatically

    Returns
    -------
    Tuple[float, Optional[Tuple]]
        (focus_score, roi) where roi is (x, y, w, h) or None
    """
    roi = None
    if detect_roi and image.size > 1000:
        roi = get_embryo_focus_roi(image)

    score = calculate_focus_score(image, config.algorithm, roi=roi, config=config)
    return score, roi


def find_best_focus_position(positions: List[float],
                           scores: List[float],
                           images: List[np.ndarray],
                           config: FocusAnalysisConfig) -> float:
    """
    Find the best focus position from sweep data

    Pure function with clean fallback logic. No complex if/else chains.

    Parameters
    ----------
    positions : List[float]
        List of motor positions
    scores : List[float]
        List of focus scores at each position
    images : List[np.ndarray]
        List of images at each position
    config : FocusAnalysisConfig
        Analysis configuration

    Returns
    -------
    float
        Best focus position
    """
    if len(positions) != len(scores) != len(images):
        raise ValueError("Positions, scores, and images must have same length")

    if len(positions) < 3:
        # Not enough data for curve fitting, return highest score position
        return positions[np.argmax(scores)]

    try:
        # Try curve fitting analysis
        result = analyze_focus_stack(positions, images, config)

        if result.success and result.r_squared >= config.minimum_r_squared:
            # Curve fitting successful
            return result.best_position
        else:
            # Curve fitting failed or poor fit, use highest score
            return positions[np.argmax(scores)]

    except Exception:
        # Analysis failed, fall back to highest score
        return positions[np.argmax(scores)]


def analyze_focus_sweep(sweep_data: List[FocusDataPoint],
                       config: FocusAnalysisConfig) -> FocusSweepResult:
    """
    Analyze a complete focus sweep

    Pure function: sweep_data → analysis result

    Parameters
    ----------
    sweep_data : List[FocusDataPoint]
        List of focus data points from sweep
    config : FocusAnalysisConfig
        Analysis configuration

    Returns
    -------
    FocusSweepResult
        Complete analysis results
    """
    if len(sweep_data) < 3:
        return FocusSweepResult(
            success=False,
            best_position=0.0,
            best_score=0.0,
            all_data=sweep_data,
            error_message="Insufficient data points for analysis"
        )

    try:
        positions = [d.position for d in sweep_data]
        scores = [d.score for d in sweep_data]
        images = [d.image for d in sweep_data]

        # Find best position
        best_position = find_best_focus_position(positions, scores, images, config)

        # Get score at best position (interpolate if needed)
        best_idx = np.argmin(np.abs(np.array(positions) - best_position))
        best_score = scores[best_idx]

        # Calculate R-squared if possible
        r_squared = 0.0
        try:
            result = analyze_focus_stack(positions, images, config)
            if result.success:
                r_squared = result.r_squared
        except Exception:
            pass

        return FocusSweepResult(
            success=True,
            best_position=best_position,
            best_score=best_score,
            all_data=sweep_data,
            r_squared=r_squared
        )

    except Exception as e:
        return FocusSweepResult(
            success=False,
            best_position=0.0,
            best_score=0.0,
            all_data=sweep_data,
            error_message=str(e)
        )


def create_focus_positions(center: float,
                          range_um: float,
                          num_steps: int,
                          limits: Tuple[float, float]) -> List[float]:
    """
    Create focus sweep positions with limit checking

    Pure function: center + range + limits → valid positions

    Parameters
    ----------
    center : float
        Center position for sweep
    range_um : float
        Total range in micrometers (±range/2 around center)
    num_steps : int
        Number of steps in sweep
    limits : Tuple[float, float]
        Motor limits (min, max)

    Returns
    -------
    List[float]
        List of valid positions within limits
    """
    # Generate positions
    positions = np.linspace(
        center - range_um/2,
        center + range_um/2,
        num_steps
    )

    # Filter to limits
    min_pos, max_pos = limits
    valid_positions = [p for p in positions if min_pos <= p <= max_pos]

    return valid_positions


def print_focus_summary(result: FocusSweepResult, scan_type: str = "focus") -> None:
    """
    Print a clean summary of focus results

    Parameters
    ----------
    result : FocusSweepResult
        Focus analysis result
    scan_type : str
        Type of scan ("coarse", "fine", etc.)
    """
    if result.success:
        print(f"{scan_type.capitalize()} analysis: "
              f"best position {result.best_position:.2f} μm "
              f"(score: {result.best_score:.1f}, R²: {result.r_squared:.3f})")
    else:
        print(f"{scan_type.capitalize()} analysis failed: {result.error_message}")


# Convenience functions for common operations
def quick_focus_score(image: np.ndarray, algorithm: str = 'gradient') -> float:
    """Quick focus scoring with default parameters"""
    config = FocusAnalysisConfig(algorithm=algorithm)
    score, _ = score_single_image(image, config, detect_roi=True)
    return score


def is_good_focus_curve(scores: List[float], threshold: float = 0.1) -> bool:
    """
    Check if focus curve has sufficient variation for analysis

    Parameters
    ----------
    scores : List[float]
        Focus scores from sweep
    threshold : float
        Minimum coefficient of variation required

    Returns
    -------
    bool
        True if curve has good variation
    """
    if len(scores) < 3:
        return False

    std_dev = np.std(scores)
    mean_score = np.mean(scores)

    if mean_score == 0:
        return False

    coefficient_of_variation = std_dev / mean_score
    return coefficient_of_variation >= threshold