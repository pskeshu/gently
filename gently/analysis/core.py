"""
Gently DiSPIM Analysis Core
===========================

Essential analysis functions for DiSPIM autofocus workflows.
Contains only the core functions needed for the current test_embryo_focus.py workflow.

Pure functions that work with any image data from any detector device.
Designed for easy AI integration and testing.
"""

import logging
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
import numpy as np
from scipy import optimize, stats
from scipy.ndimage import gaussian_filter, sobel
import warnings


class FocusAlgorithm(Enum):
    """Focus scoring algorithms available"""
    VOLATH = "volath"
    GRADIENT = "gradient"
    VARIANCE = "variance"


class FitFunction(Enum):
    """Curve fitting functions available"""
    GAUSSIAN = "gaussian"
    PARABOLIC = "parabolic"
    NONE = "none"


@dataclass
class FocusAnalysisConfig:
    """Configuration for focus analysis operations"""
    algorithm: str = FocusAlgorithm.VOLATH.value
    fit_function: str = FitFunction.GAUSSIAN.value
    minimum_r_squared: float = 0.75
    gaussian_sigma: float = 1.0  # For gradient-based methods
    edge_crop: int = 10  # Pixels to crop from image edges
    outlier_threshold: float = 3.0  # Standard deviations for outlier detection


@dataclass
class FocusResult:
    """Result of focus analysis"""
    success: bool
    best_position: float
    best_score: float
    r_squared: float
    fit_params: Optional[np.ndarray] = None
    all_positions: Optional[np.ndarray] = None
    all_scores: Optional[np.ndarray] = None
    error_message: Optional[str] = None


def calculate_focus_score(image: np.ndarray, algorithm: str = FocusAlgorithm.VOLATH.value,
                         roi: Optional[Tuple[int, int, int, int]] = None,
                         config: Optional[FocusAnalysisConfig] = None) -> float:
    """
    Calculate focus score for an image using specified algorithm

    Pure function: image data in → focus score out
    Device-agnostic and AI-friendly.

    Parameters
    ----------
    image : np.ndarray
        Input image array (2D grayscale or 3D with last dimension as channels)
    algorithm : str
        Focus scoring algorithm to use ('volath', 'gradient', 'variance')
    roi : Tuple[int, int, int, int], optional
        Region of interest as (x, y, width, height)
    config : FocusAnalysisConfig, optional
        Analysis configuration parameters

    Returns
    -------
    float
        Focus score (higher = better focus)
    """
    if config is None:
        config = FocusAnalysisConfig()

    # Ensure 2D grayscale image
    if image.ndim == 3:
        image = np.mean(image, axis=2)
    elif image.ndim != 2:
        raise ValueError(f"Image must be 2D or 3D, got {image.ndim}D")

    # Apply ROI if specified
    if roi is not None:
        x, y, w, h = roi
        image = image[y:y+h, x:x+w]

    # Crop edges to avoid boundary effects
    if config.edge_crop > 0:
        crop = config.edge_crop
        if image.shape[0] > 2*crop and image.shape[1] > 2*crop:
            image = image[crop:-crop, crop:-crop]

    # Convert to float for calculations
    image = image.astype(np.float64)

    # Calculate focus score based on algorithm
    try:
        if algorithm == FocusAlgorithm.VOLATH.value:
            return _volath_focus_score(image)
        elif algorithm == FocusAlgorithm.GRADIENT.value:
            return _gradient_focus_score(image, config.gaussian_sigma)
        elif algorithm == FocusAlgorithm.VARIANCE.value:
            return _variance_focus_score(image)
        else:
            raise ValueError(f"Unknown focus algorithm: {algorithm}")

    except Exception as e:
        logging.getLogger(__name__).error(f"Focus score calculation failed: {e}")
        return 0.0


def _volath_focus_score(image: np.ndarray) -> float:
    """Volath focus measure - autocorrelation based"""
    try:
        # Compute mean
        mean_val = np.mean(image)

        # Volath F4 measure: sum of (I(i,j) * I(i,j+1)) - mean^2
        shifted = np.roll(image, 1, axis=1)
        product_sum = np.sum(image * shifted)

        return product_sum - (mean_val ** 2) * image.size

    except Exception as e:
        logging.getLogger(__name__).error(f"Volath focus score failed: {e}")
        return 0.0


def _gradient_focus_score(image: np.ndarray, sigma: float = 1.0) -> float:
    """Gradient-based focus measure - good for embryo edges"""
    try:
        # Apply Gaussian smoothing to reduce noise
        if sigma > 0:
            image = gaussian_filter(image, sigma=sigma)

        # Calculate gradients using Sobel operators
        grad_x = sobel(image, axis=1)
        grad_y = sobel(image, axis=0)

        # Gradient magnitude
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)

        # Sum of gradient magnitudes
        return np.sum(gradient_magnitude)

    except Exception as e:
        logging.getLogger(__name__).error(f"Gradient focus score failed: {e}")
        return 0.0


def _variance_focus_score(image: np.ndarray) -> float:
    """Simple variance-based focus measure"""
    try:
        return np.var(image)
    except Exception as e:
        logging.getLogger(__name__).error(f"Variance focus score failed: {e}")
        return 0.0


def analyze_focus_stack(positions: List[float], images: List[np.ndarray],
                       config: FocusAnalysisConfig) -> FocusResult:
    """
    Analyze a complete focus stack to find best focus position

    Pure function: positions + images → focus result
    AI-friendly interface with structured result.

    Parameters
    ----------
    positions : List[float]
        Motor positions corresponding to each image
    images : List[np.ndarray]
        List of images at each position
    config : FocusAnalysisConfig
        Analysis configuration

    Returns
    -------
    FocusResult
        Complete analysis result with best position, scores, and fit quality
    """
    try:
        if len(positions) != len(images):
            return FocusResult(
                success=False, best_position=0.0, best_score=0.0, r_squared=0.0,
                error_message="Positions and images length mismatch"
            )

        if len(positions) < 3:
            return FocusResult(
                success=False, best_position=0.0, best_score=0.0, r_squared=0.0,
                error_message="Need at least 3 data points for analysis"
            )

        # Calculate focus scores for all images
        scores = []
        for image in images:
            score = calculate_focus_score(image, config.algorithm, config=config)
            scores.append(score)

        positions = np.array(positions)
        scores = np.array(scores)

        # Try curve fitting to find optimal position
        try:
            fitted_positions, fitted_scores, fit_params, r_squared = fit_focus_curve(
                positions, scores, config.fit_function
            )

            if r_squared >= config.minimum_r_squared:
                # Use curve fit result
                best_idx = np.argmax(fitted_scores)
                best_position = fitted_positions[best_idx]
                best_score = fitted_scores[best_idx]
            else:
                # Fallback to highest measured score
                best_idx = np.argmax(scores)
                best_position = positions[best_idx]
                best_score = scores[best_idx]

        except Exception as fit_error:
            # Fallback to highest measured score
            best_idx = np.argmax(scores)
            best_position = positions[best_idx]
            best_score = scores[best_idx]
            r_squared = 0.0
            fit_params = None

        return FocusResult(
            success=True,
            best_position=float(best_position),
            best_score=float(best_score),
            r_squared=float(r_squared),
            fit_params=fit_params,
            all_positions=positions,
            all_scores=scores
        )

    except Exception as e:
        return FocusResult(
            success=False, best_position=0.0, best_score=0.0, r_squared=0.0,
            error_message=str(e)
        )


def fit_focus_curve(positions: np.ndarray, scores: np.ndarray,
                   fit_function: str = FitFunction.GAUSSIAN.value) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Fit a curve to focus score data

    Parameters
    ----------
    positions : np.ndarray
        Motor positions
    scores : np.ndarray
        Focus scores at each position
    fit_function : str
        Type of curve to fit ('gaussian' or 'parabolic')

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, float]
        (fitted_positions, fitted_scores, fit_parameters, r_squared)
    """
    if len(positions) < 3:
        raise ValueError("Need at least 3 points for curve fitting")

    # Create high-resolution position array for smooth curve
    pos_range = np.max(positions) - np.min(positions)
    pos_center = (np.max(positions) + np.min(positions)) / 2
    fitted_positions = np.linspace(pos_center - pos_range*0.6, pos_center + pos_range*0.6, 100)

    try:
        if fit_function == FitFunction.GAUSSIAN.value:
            fit_params, r_squared = _fit_gaussian(positions, scores)
        elif fit_function == FitFunction.PARABOLIC.value:
            fit_params, r_squared = _fit_parabolic(positions, scores)
        else:
            raise ValueError(f"Unknown fit function: {fit_function}")

        # Generate fitted curve
        if fit_function == FitFunction.GAUSSIAN.value:
            a, mu, sigma, c = fit_params
            fitted_scores = a * np.exp(-((fitted_positions - mu) ** 2) / (2 * sigma ** 2)) + c
        else:  # parabolic
            a, b, c = fit_params
            fitted_scores = a * fitted_positions**2 + b * fitted_positions + c

        return fitted_positions, fitted_scores, fit_params, r_squared

    except Exception as e:
        logging.getLogger(__name__).error(f"Curve fitting failed: {e}")
        raise


def _fit_gaussian(positions: np.ndarray, scores: np.ndarray) -> Tuple[np.ndarray, float]:
    """Fit Gaussian curve to focus data"""
    def gaussian(x, a, mu, sigma, c):
        return a * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) + c

    # Initial parameter estimates
    a_init = np.max(scores) - np.min(scores)
    mu_init = positions[np.argmax(scores)]
    sigma_init = (np.max(positions) - np.min(positions)) / 4
    c_init = np.min(scores)

    p0 = [a_init, mu_init, sigma_init, c_init]

    # Fit with bounds to ensure physical parameters
    bounds = (
        [0, np.min(positions), 0.1, 0],  # Lower bounds
        [np.inf, np.max(positions), np.inf, np.inf]  # Upper bounds
    )

    popt, pcov = optimize.curve_fit(gaussian, positions, scores, p0=p0, bounds=bounds, maxfev=1000)

    # Calculate R-squared
    fitted_scores = gaussian(positions, *popt)
    ss_res = np.sum((scores - fitted_scores) ** 2)
    ss_tot = np.sum((scores - np.mean(scores)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    return popt, r_squared


def _fit_parabolic(positions: np.ndarray, scores: np.ndarray) -> Tuple[np.ndarray, float]:
    """Fit parabolic curve to focus data"""
    # Fit quadratic polynomial: y = ax^2 + bx + c
    coeffs = np.polyfit(positions, scores, 2)

    # Calculate R-squared
    fitted_scores = np.polyval(coeffs, positions)
    ss_res = np.sum((scores - fitted_scores) ** 2)
    ss_tot = np.sum((scores - np.mean(scores)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    return coeffs, r_squared