"""
Gently DiSPIM Analysis Core
===========================

Essential analysis functions for DiSPIM autofocus workflows.
Contains only the core functions needed for the current test_embryo_focus.py workflow.

Pure functions that work with any image data from any detector device.
Designed for easy AI integration and testing.
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
from scipy import optimize
from scipy.ndimage import gaussian_filter, sobel

from ..exceptions import FocusFitError


class FocusAlgorithm(Enum):
    """Focus scoring algorithms available"""

    VOLATH = "volath"
    GRADIENT = "gradient"
    VARIANCE = "variance"
    FFT_BANDPASS = "fft_bandpass"  # ASI diSPIM OughtaFocus algorithm


# FFT Bandpass parameters (from ASI diSPIM OughtaFocus implementation)
# These define the spatial frequency band analyzed for focus quality
FFT_LOWER_CUTOFF = 0.025  # 2.5% of max frequency - filters DC and low spatial frequencies
FFT_UPPER_CUTOFF = 0.14  # 14% of max frequency - filters high-frequency noise


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
    fit_params: np.ndarray | None = None
    all_positions: np.ndarray | None = None
    all_scores: np.ndarray | None = None
    error_message: str | None = None


class AdaptiveSweepState:
    """
    Real-time state during an adaptive focus sweep.

    Tracks running statistics for early stopping decisions. Used by the
    adaptive calibration algorithm to determine when to stop sweeping
    (either because the peak has been found or confidence is high enough).

    Early stopping criteria:
    - Score drops below 70% of max with 3+ consecutive declines
    - R² >= 0.90 with stable peak position across fits
    - Sufficient points past peak for robust fitting
    """

    # Early stopping thresholds
    DECLINE_THRESHOLD: float = 0.70  # Stop if score drops below 70% of max
    MIN_DECLINE_COUNT: int = 3  # Require N consecutive declines
    MIN_POINTS_FOR_FIT: int = 5  # Minimum points before attempting fit
    MIN_POINTS_PAST_PEAK: int = 3  # Need N points past peak for robust fit
    STABILITY_THRESHOLD_UM: float = 0.5  # Peak position stability threshold
    HIGH_CONFIDENCE_R2: float = 0.90  # R² threshold for early exit

    def __init__(self):
        self.positions: list[float] = []
        self.scores: list[float] = []

        # Peak detection state
        self.running_max_score: float = 0.0
        self.running_max_position: float = 0.0
        self.running_max_idx: int = 0
        self.decline_count: int = 0
        self.peak_detected: bool = False

        # Confidence metrics updated after each point
        self.current_r_squared: float = 0.0
        self.fit_stable: bool = False
        self.last_fit_position: float = 0.0
        self.fit_history: list[dict[str, float]] = []

    def add_point(self, position: float, score: float) -> dict[str, Any]:
        """
        Add new measurement and compute early stopping decision.

        Parameters
        ----------
        position : float
            Piezo position (µm)
        score : float
            Focus score at this position

        Returns
        -------
        dict
            - should_stop: bool - whether to stop sweeping
            - reason: str - reason for stopping (if applicable)
            - confidence: float - confidence in current fit (0-1)
        """
        self.positions.append(position)
        self.scores.append(score)

        result: dict[str, Any] = {
            "should_stop": False,
            "reason": None,
            "confidence": 0.0,
        }

        # Update running max
        if score > self.running_max_score:
            self.running_max_score = score
            self.running_max_position = position
            self.running_max_idx = len(self.positions) - 1
            self.decline_count = 0
        else:
            # Check for decline from peak
            if score < self.running_max_score * self.DECLINE_THRESHOLD:
                self.decline_count += 1

        # Early stopping check: peak detection via decline
        points_past_peak = len(self.positions) - self.running_max_idx - 1
        if self.decline_count >= self.MIN_DECLINE_COUNT:
            self.peak_detected = True
            result["confidence"] = 0.7

            # Continue a few more points past detected peak for robust fitting
            if points_past_peak >= self.MIN_POINTS_PAST_PEAK:
                result["should_stop"] = True
                result["reason"] = "peak_passed"

        # Confidence-based early exit (if we have enough points)
        if len(self.positions) >= self.MIN_POINTS_FOR_FIT:
            fit_result = self._attempt_fit()
            if fit_result:
                self.current_r_squared = fit_result["r_squared"]
                new_position = fit_result["peak_position"]

                # Track fit history for stability check
                self.fit_history.append(
                    {
                        "position": new_position,
                        "r_squared": fit_result["r_squared"],
                    }
                )

                # Check stability (position change across recent fits)
                if len(self.fit_history) >= 3:
                    recent_positions = [f["position"] for f in self.fit_history[-3:]]
                    position_range = max(recent_positions) - min(recent_positions)
                    self.fit_stable = position_range < self.STABILITY_THRESHOLD_UM

                self.last_fit_position = new_position

                # High confidence early exit
                if (
                    self.current_r_squared >= self.HIGH_CONFIDENCE_R2
                    and self.fit_stable
                    and len(self.positions) >= 7
                ):
                    result["should_stop"] = True
                    result["reason"] = "high_confidence_fit"
                    result["confidence"] = self.current_r_squared

        return result

    def _attempt_fit(self) -> dict[str, Any] | None:
        """
        Attempt Gaussian fit on current data.

        Returns
        -------
        dict or None
            Fit results with 'peak_position', 'r_squared', 'params'
        """
        if len(self.positions) < self.MIN_POINTS_FOR_FIT:
            return None

        try:
            positions = np.array(self.positions)
            scores = np.array(self.scores)

            # Use existing fit_focus_curve
            _, _, params, r_squared = fit_focus_curve(positions, scores, FitFunction.GAUSSIAN.value)

            return {
                "peak_position": float(params[1]),  # mu
                "r_squared": r_squared,
                "params": params,
            }
        except Exception:
            return None

    def get_best_position(self) -> tuple[float, float]:
        """
        Get best focus position from current data.

        Returns
        -------
        tuple
            (best_position, r_squared)
        """
        if not self.positions:
            return (0.0, 0.0)

        # Try fit first
        fit_result = self._attempt_fit()
        if fit_result and fit_result["r_squared"] >= 0.5:
            return (fit_result["peak_position"], fit_result["r_squared"])

        # Fall back to max score position
        return (self.running_max_position, 0.0)

    def reset(self):
        """Reset state for a new sweep."""
        self.positions = []
        self.scores = []
        self.running_max_score = 0.0
        self.running_max_position = 0.0
        self.running_max_idx = 0
        self.decline_count = 0
        self.peak_detected = False
        self.current_r_squared = 0.0
        self.fit_stable = False
        self.last_fit_position = 0.0
        self.fit_history = []


def calculate_focus_score(
    image: np.ndarray,
    algorithm: str = FocusAlgorithm.VOLATH.value,
    roi: tuple[int, int, int, int] | None = None,
    config: FocusAnalysisConfig | None = None,
) -> float:
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
        image = image[y : y + h, x : x + w]

    # Crop edges to avoid boundary effects
    if config.edge_crop > 0:
        crop = config.edge_crop
        if image.shape[0] > 2 * crop and image.shape[1] > 2 * crop:
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
        elif algorithm == FocusAlgorithm.FFT_BANDPASS.value:
            return _fft_bandpass_focus_score(image)
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

        return product_sum - (mean_val**2) * image.size

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


def _fft_bandpass_focus_score(
    image: np.ndarray,
    lower_cutoff: float = FFT_LOWER_CUTOFF,
    upper_cutoff: float = FFT_UPPER_CUTOFF,
) -> float:
    """
    FFT bandpass focus measure (ASI diSPIM OughtaFocus algorithm).

    This algorithm analyzes the power spectrum of spatial frequencies in the image.
    Well-focused images have more high-frequency content (sharp edges) than defocused
    images, which appear blurred and lack high-frequency components.

    Algorithm:
    1. Compute 2D FFT power spectrum of the image
    2. Create bandpass mask to keep only frequencies in [lower, upper] cutoff range
    3. Calculate mean power within that frequency band

    The default frequency band (2.5% - 14% of maximum) was empirically determined
    by Bill Mohler (UConn) to work well for light sheet microscopy.

    Parameters
    ----------
    image : np.ndarray
        2D grayscale image (already preprocessed by calculate_focus_score)
    lower_cutoff : float
        Lower frequency cutoff as fraction of max frequency (default: 0.025)
    upper_cutoff : float
        Upper frequency cutoff as fraction of max frequency (default: 0.14)

    Returns
    -------
    float
        Mean power in the specified frequency band (higher = better focus)
    """
    try:
        # Compute 2D FFT
        fft = np.fft.fft2(image)
        fft_shifted = np.fft.fftshift(fft)  # Move DC component to center

        # Compute power spectrum (magnitude squared)
        power_spectrum = np.abs(fft_shifted) ** 2

        # Create frequency grid for bandpass mask
        h, w = image.shape
        cy, cx = h // 2, w // 2  # Center coordinates

        # Create distance map from center (DC component)
        y, x = np.ogrid[:h, :w]
        distance_from_center = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)

        # Maximum frequency (corner distance)
        max_freq = np.sqrt(cx**2 + cy**2)

        # Normalized distance (0 at DC, 1 at corners)
        normalized_distance = distance_from_center / max_freq

        # Create bandpass mask: keep frequencies in [lower_cutoff, upper_cutoff]
        bandpass_mask = (normalized_distance >= lower_cutoff) & (
            normalized_distance <= upper_cutoff
        )

        # Apply mask and compute mean power in band
        masked_power = power_spectrum * bandpass_mask

        if np.sum(bandpass_mask) > 0:
            mean_power = np.sum(masked_power) / np.sum(bandpass_mask)
        else:
            mean_power = 0.0

        return mean_power

    except Exception as e:
        logging.getLogger(__name__).error(f"FFT bandpass focus score failed: {e}")
        return 0.0


def analyze_focus_stack(
    positions: list[float], images: list[np.ndarray], config: FocusAnalysisConfig
) -> FocusResult:
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
                success=False,
                best_position=0.0,
                best_score=0.0,
                r_squared=0.0,
                error_message="Positions and images length mismatch",
            )

        if len(positions) < 3:
            return FocusResult(
                success=False,
                best_position=0.0,
                best_score=0.0,
                r_squared=0.0,
                error_message="Need at least 3 data points for analysis",
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

        except (FocusFitError, Exception):
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
            all_scores=scores,
        )

    except Exception as e:
        return FocusResult(
            success=False,
            best_position=0.0,
            best_score=0.0,
            r_squared=0.0,
            error_message=str(e),
        )


def fit_focus_curve(
    positions: np.ndarray,
    scores: np.ndarray,
    fit_function: str = FitFunction.GAUSSIAN.value,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
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
    fitted_positions = np.linspace(pos_center - pos_range * 0.6, pos_center + pos_range * 0.6, 100)

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
            fitted_scores = a * np.exp(-((fitted_positions - mu) ** 2) / (2 * sigma**2)) + c
        else:  # parabolic
            a, b, c = fit_params
            fitted_scores = a * fitted_positions**2 + b * fitted_positions + c

        return fitted_positions, fitted_scores, fit_params, r_squared

    except (optimize.OptimizeWarning, RuntimeError, ValueError) as e:
        raise FocusFitError(f"Curve fitting failed: {e}") from e
    except Exception as e:
        logging.getLogger(__name__).error(f"Curve fitting failed: {e}")
        raise


def _fit_gaussian(positions: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, float]:
    """Fit Gaussian curve to focus data"""

    def gaussian(x, a, mu, sigma, c):
        return a * np.exp(-((x - mu) ** 2) / (2 * sigma**2)) + c

    # Initial parameter estimates
    a_init = np.max(scores) - np.min(scores)
    mu_init = positions[np.argmax(scores)]
    sigma_init = (np.max(positions) - np.min(positions)) / 4
    c_init = np.min(scores)

    p0 = [a_init, mu_init, sigma_init, c_init]

    # Fit with bounds to ensure physical parameters
    bounds = (
        [0, np.min(positions), 0.1, 0],  # Lower bounds
        [np.inf, np.max(positions), np.inf, np.inf],  # Upper bounds
    )

    popt, pcov = optimize.curve_fit(gaussian, positions, scores, p0=p0, bounds=bounds, maxfev=1000)

    # Calculate R-squared
    fitted_scores = gaussian(positions, *popt)
    ss_res = np.sum((scores - fitted_scores) ** 2)
    ss_tot = np.sum((scores - np.mean(scores)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    return popt, r_squared


def _fit_parabolic(positions: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, float]:
    """Fit parabolic curve to focus data"""
    # Fit quadratic polynomial: y = ax^2 + bx + c
    coeffs = np.polyfit(positions, scores, 2)

    # Calculate R-squared
    fitted_scores = np.polyval(coeffs, positions)
    ss_res = np.sum((scores - fitted_scores) ** 2)
    ss_tot = np.sum((scores - np.mean(scores)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    return coeffs, r_squared


def create_focus_montage(
    images: list[np.ndarray],
    labels: list[str] | None = None,
    offsets: list[float] | None = None,
    normalize: bool = True,
    gap: int = 4,
) -> np.ndarray:
    """
    Create a labeled side-by-side montage of focus images for Vision comparison.

    Used by the hybrid focus selection algorithm to present multiple focus
    positions to Claude Vision for visual assessment.

    Parameters
    ----------
    images : List[np.ndarray]
        List of images to combine (should be same dimensions)
    labels : List[str], optional
        Labels for each image (e.g., ['A', 'B', 'C']). Defaults to A, B, C, ...
    offsets : List[float], optional
        Piezo offsets in µm for each image. If provided, adds offset annotation.
    normalize : bool, default True
        Whether to normalize each image to 0-255 range
    gap : int, default 4
        Pixel gap between images (filled with white)

    Returns
    -------
    np.ndarray
        Combined montage image (uint8, shape: H x (W*N + gap*(N-1)))

    Example
    -------
    >>> images = [img_minus2, img_center, img_plus2]
    >>> offsets = [-2.0, 0.0, 2.0]
    >>> montage = create_focus_montage(images, offsets=offsets)
    >>> # Send montage to Vision API for focus comparison
    """
    if not images:
        raise ValueError("Need at least one image")

    # Default labels: A, B, C, ...
    if labels is None:
        labels = [chr(ord("A") + i) for i in range(len(images))]

    # Ensure all images are 2D and same size
    processed = []
    for img in images:
        # Handle 3D images (take center slice or max projection)
        if img.ndim == 3:
            if img.shape[2] <= 4:  # Likely channels dimension
                img = img[:, :, 0]  # Take first channel
            else:  # Likely Z-stack
                img = np.max(img, axis=2)  # Max projection

        # Normalize to 0-255 if requested
        if normalize:
            img_min, img_max = img.min(), img.max()
            if img_max > img_min:
                img = ((img - img_min) / (img_max - img_min) * 255).astype(np.uint8)
            else:
                img = np.zeros_like(img, dtype=np.uint8)
        else:
            img = img.astype(np.uint8)

        processed.append(img)

    # Get dimensions (use first image as reference)
    h, w = processed[0].shape[:2]

    # Create montage canvas
    n = len(processed)
    total_width = w * n + gap * (n - 1)
    montage = (
        np.ones((h + 30, total_width), dtype=np.uint8) * 255
    )  # White background, extra space for labels

    # Place images with gaps
    for i, (img, label) in enumerate(zip(processed, labels, strict=False)):
        x_start = i * (w + gap)

        # Resize if needed to match reference dimensions
        if img.shape != (h, w):
            from scipy.ndimage import zoom

            zoom_factors = (h / img.shape[0], w / img.shape[1])
            img = zoom(img, zoom_factors, order=1).astype(np.uint8)

        montage[:h, x_start : x_start + w] = img

        # Add label text (simple pixel drawing for letter)
        # Position label at top-left of each image
        _draw_label(montage, label, x_start + 5, h + 5)

        # Add offset annotation if provided
        if offsets is not None and i < len(offsets):
            offset_text = f"{offsets[i]:+.1f}um"
            _draw_label(montage, offset_text, x_start + 5, h + 18, small=True)

    return montage


def _draw_label(image: np.ndarray, text: str, x: int, y: int, small: bool = False):
    """
    Draw simple text label on image using basic pixel patterns.

    This is a minimal implementation that doesn't require PIL/OpenCV.
    For production, consider using PIL.ImageDraw or cv2.putText.
    """
    # Simple 5x7 pixel font patterns for common characters
    # Each character is a list of (row, col) offsets that should be black
    scale = 1 if small else 2

    # Simplified - just draw a rectangle with the label area darker
    # In production, use proper text rendering
    char_width = 6 * scale
    text_width = len(text) * char_width

    # Draw dark background box for label
    y_end = min(y + 10 * scale, image.shape[0])
    x_end = min(x + text_width + 4, image.shape[1])
    if y < image.shape[0] and x < image.shape[1]:
        image[y:y_end, x:x_end] = 40  # Dark gray background

    # Note: For proper text rendering, the caller should use PIL or OpenCV
    # This placeholder ensures the montage structure works
