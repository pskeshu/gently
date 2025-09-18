"""
Gently DiSPIM Analysis Algorithms
=================================

Optional and advanced analysis algorithms for DiSPIM microscopy.
These are additional focus algorithms and advanced curve fitting functions
that are not essential for the basic autofocus workflow.

Separated from core.py to keep essential functions simple and focused.
"""

import logging
from typing import Tuple, Optional
import numpy as np
from scipy import optimize
from scipy.ndimage import convolve


def _laplacian_focus_score(image: np.ndarray) -> float:
    """
    Laplacian-based focus measure

    Uses the Laplacian operator to detect edges and texture.
    Alternative algorithm to the core volath/gradient methods.
    """
    if image.size == 0:
        return 0.0

    try:
        # Laplacian kernel
        laplacian_kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])

        # Apply convolution
        laplacian = convolve(image, laplacian_kernel)

        # Return variance of Laplacian
        return np.var(laplacian)

    except Exception as e:
        logging.getLogger(__name__).error(f"Laplacian focus score failed: {e}")
        return 0.0


def _brenner_focus_score(image: np.ndarray) -> float:
    """
    Brenner focus measure

    Based on squared differences between adjacent pixels.
    Alternative algorithm to the core methods.
    """
    if image.size == 0:
        return 0.0

    try:
        # Squared differences in horizontal direction
        h_diff = (image[2:, :] - image[:-2, :])**2

        # Squared differences in vertical direction
        v_diff = (image[:, 2:] - image[:, :-2])**2

        return np.sum(h_diff) + np.sum(v_diff)

    except Exception as e:
        logging.getLogger(__name__).error(f"Brenner focus score failed: {e}")
        return 0.0


def _remove_outliers(positions: np.ndarray, scores: np.ndarray,
                    threshold: float = 3.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove outliers from focus data using z-score method

    Parameters
    ----------
    positions : np.ndarray
        Position data
    scores : np.ndarray
        Focus score data
    threshold : float
        Z-score threshold for outlier detection

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        (filtered_positions, filtered_scores)
    """
    if len(positions) != len(scores):
        raise ValueError("Positions and scores must have same length")

    if len(positions) < 3:
        return positions, scores

    try:
        # Calculate z-scores for focus scores
        mean_score = np.mean(scores)
        std_score = np.std(scores)

        if std_score == 0:
            return positions, scores

        z_scores = np.abs((scores - mean_score) / std_score)

        # Keep points within threshold
        mask = z_scores <= threshold

        if np.sum(mask) < 3:
            # Too many outliers removed, return original data
            return positions, scores

        return positions[mask], scores[mask]

    except Exception as e:
        logging.getLogger(__name__).error(f"Outlier removal failed: {e}")
        return positions, scores


def fit_advanced_curves(positions: np.ndarray, scores: np.ndarray,
                       method: str = 'spline') -> Tuple[np.ndarray, float]:
    """
    Advanced curve fitting methods beyond basic gaussian/parabolic

    Parameters
    ----------
    positions : np.ndarray
        Position data
    scores : np.ndarray
        Focus score data
    method : str
        Advanced fitting method ('spline', 'polynomial', etc.)

    Returns
    -------
    Tuple[np.ndarray, float]
        (fit_parameters, r_squared)
    """
    # Placeholder for future advanced curve fitting
    # Could include spline fitting, higher-order polynomials, etc.
    raise NotImplementedError("Advanced curve fitting not implemented yet")