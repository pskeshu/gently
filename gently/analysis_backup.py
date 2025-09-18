"""
Gently DiSPIM Analysis
=====================

Device-agnostic analysis utilities for DiSPIM microscopy.
Focus scoring, curve fitting, and image analysis functions that work with
any image data from any detector device.

Ported from the Java DiSPIM plugin autofocus algorithms but structured
as pure Python functions that integrate with Bluesky data structures.
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
    LAPLACIAN = "laplacian"
    BRENNER = "brenner"


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


# =============================================================================
# FOCUS SCORING ALGORITHMS
# =============================================================================

def calculate_focus_score(image: np.ndarray, algorithm: str = FocusAlgorithm.VOLATH.value,
                         roi: Optional[Tuple[int, int, int, int]] = None,
                         config: Optional[FocusAnalysisConfig] = None) -> float:
    """
    Calculate focus score for an image using specified algorithm
    
    Device-agnostic function that works with any image array from any detector.
    
    Parameters
    ----------
    image : np.ndarray
        Input image array (2D grayscale or 3D with last dimension as channels)
    algorithm : str
        Focus scoring algorithm to use
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
        # Convert to grayscale if color
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
        elif algorithm == FocusAlgorithm.LAPLACIAN.value:
            return _laplacian_focus_score(image)
        elif algorithm == FocusAlgorithm.BRENNER.value:
            return _brenner_focus_score(image)
        else:
            raise ValueError(f"Unknown focus algorithm: {algorithm}")
            
    except Exception as e:
        logging.getLogger(__name__).error(f"Focus score calculation failed: {e}")
        return 0.0


def _volath_focus_score(image: np.ndarray) -> float:
    """
    Volath focus measure - autocorrelation based
    
    Based on the Volath4 algorithm from the Java DiSPIM plugin.
    Measures autocorrelation between adjacent pixels.
    """
    if image.size == 0:
        return 0.0
    
    # Volath4: sum of (I(x,y) * I(x+1,y)) - mean^2
    mean_val = np.mean(image)
    
    # Horizontal autocorrelation
    h_corr = np.sum(image[:-1, :] * image[1:, :])
    
    # Vertical autocorrelation  
    v_corr = np.sum(image[:, :-1] * image[:, 1:])
    
    # Total pixels used in calculation
    total_pixels = image[:-1, :].size + image[:, :-1].size
    
    if total_pixels == 0:
        return 0.0
    
    # Volath4 formula
    focus_measure = (h_corr + v_corr) / total_pixels - mean_val**2
    
    return max(0.0, focus_measure)


def _gradient_focus_score(image: np.ndarray, sigma: float = 1.0) -> float:
    """
    Gradient-based focus measure using Sobel operators
    
    Measures the magnitude of intensity gradients in the image.
    Higher gradients indicate sharper edges and better focus.
    """
    if image.size == 0:
        return 0.0
    
    # Apply slight Gaussian blur to reduce noise
    if sigma > 0:
        image = gaussian_filter(image, sigma)
    
    # Calculate gradients using Sobel operators
    grad_x = sobel(image, axis=1)
    grad_y = sobel(image, axis=0)
    
    # Gradient magnitude
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    
    # Return mean gradient magnitude
    return np.mean(gradient_magnitude)


def _variance_focus_score(image: np.ndarray) -> float:
    """
    Variance-based focus measure
    
    Simple measure based on pixel intensity variance.
    Well-focused images have higher variance.
    """
    if image.size == 0:
        return 0.0
    
    return np.var(image)


def _laplacian_focus_score(image: np.ndarray) -> float:
    """
    Laplacian-based focus measure
    
    Uses the Laplacian operator to detect edges and texture.
    """
    if image.size == 0:
        return 0.0
    
    # Laplacian kernel
    laplacian_kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    
    # Apply convolution
    from scipy.ndimage import convolve
    laplacian = convolve(image, laplacian_kernel)
    
    # Return variance of Laplacian
    return np.var(laplacian)


def _brenner_focus_score(image: np.ndarray) -> float:
    """
    Brenner focus measure
    
    Based on squared differences between adjacent pixels.
    """
    if image.size == 0:
        return 0.0
    
    # Squared differences in horizontal direction
    h_diff = (image[2:, :] - image[:-2, :])**2
    
    # Squared differences in vertical direction  
    v_diff = (image[:, 2:] - image[:, :-2])**2
    
    return np.sum(h_diff) + np.sum(v_diff)


# =============================================================================
# CURVE FITTING FUNCTIONS
# =============================================================================

def fit_focus_curve(positions: np.ndarray, scores: np.ndarray, 
                   fit_function: str = FitFunction.GAUSSIAN.value,
                   config: Optional[FocusAnalysisConfig] = None) -> Tuple[np.ndarray, float]:
    """
    Fit curve to focus score data
    
    Device-agnostic curve fitting that works with any position/score data.
    
    Parameters
    ----------
    positions : np.ndarray
        Array of positions where focus was measured
    scores : np.ndarray  
        Array of focus scores at each position
    fit_function : str
        Curve fitting function to use
    config : FocusAnalysisConfig, optional
        Analysis configuration
        
    Returns
    -------
    Tuple[np.ndarray, float]
        (fit_parameters, r_squared)
    """
    if config is None:
        config = FocusAnalysisConfig()
    
    if len(positions) != len(scores):
        raise ValueError("Positions and scores must have same length")
    
    if len(positions) < 3:
        raise ValueError("Need at least 3 points for curve fitting")
    
    # Remove outliers
    positions_clean, scores_clean = _remove_outliers(positions, scores, config.outlier_threshold)
    
    if len(positions_clean) < 3:
        logging.getLogger(__name__).warning("Too few points after outlier removal, using original data")
        positions_clean, scores_clean = positions, scores
    
    try:
        if fit_function == FitFunction.GAUSSIAN.value:
            return _fit_gaussian(positions_clean, scores_clean)
        elif fit_function == FitFunction.PARABOLIC.value:
            return _fit_parabola(positions_clean, scores_clean)
        elif fit_function == FitFunction.NONE.value:
            # No fitting - return empty params
            return np.array([]), 1.0
        else:
            raise ValueError(f"Unknown fit function: {fit_function}")
            
    except Exception as e:
        logging.getLogger(__name__).error(f"Curve fitting failed: {e}")
        return np.array([]), 0.0


def _remove_outliers(positions: np.ndarray, scores: np.ndarray, 
                    threshold: float = 3.0) -> Tuple[np.ndarray, np.ndarray]:
    """Remove statistical outliers from data"""
    if len(scores) < 4:  # Need at least 4 points to detect outliers
        return positions, scores
    
    z_scores = np.abs(stats.zscore(scores))
    mask = z_scores < threshold
    
    return positions[mask], scores[mask]


def _fit_gaussian(positions: np.ndarray, scores: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Fit Gaussian curve to focus data
    
    Gaussian: y = a * exp(-(x - mu)^2 / (2 * sigma^2)) + c
    Parameters: [a, mu, sigma, c]
    """
    def gaussian(x, a, mu, sigma, c):
        return a * np.exp(-(x - mu)**2 / (2 * sigma**2)) + c
    
    # Initial parameter guesses
    max_score = np.max(scores)
    min_score = np.min(scores)
    peak_pos = positions[np.argmax(scores)]
    pos_range = np.max(positions) - np.min(positions)
    
    p0 = [
        max_score - min_score,  # amplitude
        peak_pos,               # center (mu)
        pos_range / 6,          # width (sigma) 
        min_score               # offset (c)
    ]
    
    try:
        # Fit with bounds to ensure reasonable parameters
        bounds = (
            [0, np.min(positions), 0, 0],  # lower bounds
            [np.inf, np.max(positions), pos_range, np.inf]  # upper bounds
        )
        
        popt, _ = optimize.curve_fit(gaussian, positions, scores, p0=p0, bounds=bounds)
        
        # Calculate R-squared
        y_pred = gaussian(positions, *popt)
        r_squared = _calculate_r_squared(scores, y_pred)
        
        return popt, r_squared
        
    except Exception as e:
        logging.getLogger(__name__).warning(f"Gaussian fit failed: {e}")
        return np.array([]), 0.0


def _fit_parabola(positions: np.ndarray, scores: np.ndarray) -> Tuple[np.ndarray, float]:
    """
    Fit parabolic curve to focus data
    
    Parabola: y = a * x^2 + b * x + c
    Parameters: [a, b, c]
    """
    try:
        # Use numpy polyfit for robust parabolic fitting
        coeffs = np.polyfit(positions, scores, 2)  # 2nd order polynomial
        
        # Calculate R-squared
        y_pred = np.polyval(coeffs, positions)
        r_squared = _calculate_r_squared(scores, y_pred)
        
        return coeffs, r_squared
        
    except Exception as e:
        logging.getLogger(__name__).warning(f"Parabolic fit failed: {e}")
        return np.array([]), 0.0


def _calculate_r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate coefficient of determination (R²)"""
    try:
        ss_res = np.sum((y_true - y_pred)**2)
        ss_tot = np.sum((y_true - np.mean(y_true))**2)
        
        if ss_tot == 0:
            return 1.0 if ss_res == 0 else 0.0
        
        r_squared = 1 - (ss_res / ss_tot)
        return max(0.0, r_squared)  # Ensure non-negative
        
    except Exception:
        return 0.0


# =============================================================================
# FOCUS ANALYSIS FUNCTIONS
# =============================================================================

def find_curve_maximum(positions: np.ndarray, scores: np.ndarray, 
                      fit_params: np.ndarray, fit_function: str) -> float:
    """
    Find the position of maximum focus from fitted curve
    
    Parameters
    ----------
    positions : np.ndarray
        Original position data
    scores : np.ndarray
        Original score data
    fit_params : np.ndarray
        Fitted curve parameters
    fit_function : str
        Type of fitted curve
        
    Returns
    -------
    float
        Position of maximum focus
    """
    if len(fit_params) == 0 or fit_function == FitFunction.NONE.value:
        # No fit - return position with highest score
        return positions[np.argmax(scores)]
    
    try:
        if fit_function == FitFunction.GAUSSIAN.value:
            # Gaussian: maximum is at mu (parameter 1)
            return fit_params[1]
            
        elif fit_function == FitFunction.PARABOLIC.value:
            # Parabola: y = ax² + bx + c, maximum at x = -b/(2a)
            a, b, c = fit_params
            if a >= 0:
                # Parabola opens upward - no maximum, return peak of data
                return positions[np.argmax(scores)]
            return -b / (2 * a)
            
        else:
            return positions[np.argmax(scores)]
            
    except Exception as e:
        logging.getLogger(__name__).warning(f"Failed to find curve maximum: {e}")
        return positions[np.argmax(scores)]


def validate_autofocus_result(best_position: float, positions: np.ndarray,
                             scores: np.ndarray, r_squared: float,
                             config: FocusAnalysisConfig) -> Tuple[bool, str]:
    """
    Validate autofocus result based on quality criteria
    
    Parameters
    ----------
    best_position : float
        Found best focus position
    positions : np.ndarray
        All tested positions
    scores : np.ndarray
        All focus scores
    r_squared : float
        Goodness of fit
    config : FocusAnalysisConfig
        Analysis configuration with validation criteria
        
    Returns
    -------
    Tuple[bool, str]
        (is_valid, reason_if_invalid)
    """
    # Check R² threshold
    if r_squared < config.minimum_r_squared:
        return False, f"R² ({r_squared:.3f}) below threshold ({config.minimum_r_squared})"
    
    # Check if best position is within central 80% of scan range
    pos_min, pos_max = np.min(positions), np.max(positions)
    pos_range = pos_max - pos_min
    
    if pos_range > 0:
        center_start = pos_min + 0.1 * pos_range
        center_end = pos_max - 0.1 * pos_range
        
        if not (center_start <= best_position <= center_end):
            return False, f"Best position ({best_position:.2f}) outside central 80% of range"
    
    # Check for sufficient signal variation
    score_std = np.std(scores)
    score_mean = np.mean(scores)
    if score_mean > 0:
        cv = score_std / score_mean
        if cv < 0.05:  # Less than 5% coefficient of variation
            return False, "Insufficient focus signal variation (flat response)"
    
    return True, "Validation passed"


def analyze_focus_stack(positions: List[float], images: List[np.ndarray],
                       config: Optional[FocusAnalysisConfig] = None) -> FocusResult:
    """
    Complete focus stack analysis - scores images, fits curve, finds best position
    
    Device-agnostic analysis that works with any list of images from any detector.
    This is the main entry point for autofocus analysis.
    
    Parameters
    ----------
    positions : List[float]
        List of positions where images were acquired
    images : List[np.ndarray]
        List of image arrays corresponding to each position
    config : FocusAnalysisConfig, optional
        Analysis configuration
        
    Returns
    -------
    FocusResult
        Complete analysis results
    """
    if config is None:
        config = FocusAnalysisConfig()
    
    if len(positions) != len(images):
        return FocusResult(
            success=False,
            best_position=0.0,
            best_score=0.0,
            r_squared=0.0,
            error_message="Positions and images length mismatch"
        )
    
    if len(images) < 3:
        return FocusResult(
            success=False,
            best_position=0.0,
            best_score=0.0,
            r_squared=0.0,
            error_message="Need at least 3 images for analysis"
        )
    
    try:
        # Calculate focus scores for all images
        scores = []
        for img in images:
            score = calculate_focus_score(img, config.algorithm, config=config)
            scores.append(score)
        
        positions_arr = np.array(positions)
        scores_arr = np.array(scores)
        
        # Fit curve to focus data
        fit_params, r_squared = fit_focus_curve(positions_arr, scores_arr, 
                                              config.fit_function, config)
        
        # Find best focus position
        best_position = find_curve_maximum(positions_arr, scores_arr, 
                                         fit_params, config.fit_function)
        
        # Get best score (from data, not fit)
        best_idx = np.argmin(np.abs(positions_arr - best_position))
        best_score = scores_arr[best_idx]
        
        # Validate result
        is_valid, reason = validate_autofocus_result(best_position, positions_arr, 
                                                   scores_arr, r_squared, config)
        
        return FocusResult(
            success=is_valid,
            best_position=best_position,
            best_score=best_score,
            r_squared=r_squared,
            fit_params=fit_params,
            all_positions=positions_arr,
            all_scores=scores_arr,
            error_message=None if is_valid else reason
        )
        
    except Exception as e:
        logging.getLogger(__name__).error(f"Focus analysis failed: {e}")
        return FocusResult(
            success=False,
            best_position=0.0,
            best_score=0.0,
            r_squared=0.0,
            error_message=str(e)
        )


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def plot_focus_curve(result: FocusResult, title: str = "Focus Curve") -> None:
    """
    Plot focus curve results for visualization
    
    Requires matplotlib for plotting.
    """
    try:
        import matplotlib.pyplot as plt
        
        if result.all_positions is None or result.all_scores is None:
            print("No data to plot")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot data points
        ax.scatter(result.all_positions, result.all_scores, 
                  color='blue', alpha=0.7, s=50, label='Data')
        
        # Plot fitted curve if available
        if result.fit_params is not None and len(result.fit_params) > 0:
            x_smooth = np.linspace(result.all_positions.min(), 
                                 result.all_positions.max(), 100)
            
            # This would need the fit function type to plot correctly
            # For now, just connect the dots
            ax.plot(result.all_positions, result.all_scores, 
                   color='red', alpha=0.5, label='Trend')
        
        # Mark best position
        ax.axvline(result.best_position, color='green', linestyle='--', 
                  label=f'Best Position: {result.best_position:.2f}')
        
        ax.set_xlabel('Position')
        ax.set_ylabel('Focus Score')
        ax.set_title(f'{title} (R² = {result.r_squared:.3f})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
    except ImportError:
        print("matplotlib not available for plotting")
    except Exception as e:
        print(f"Plotting failed: {e}")


def get_algorithm_info() -> Dict[str, str]:
    """Get information about available focus algorithms"""
    return {
        FocusAlgorithm.VOLATH.value: "Autocorrelation-based measure, good for textured samples",
        FocusAlgorithm.GRADIENT.value: "Edge-based measure using Sobel operators",
        FocusAlgorithm.VARIANCE.value: "Simple intensity variance measure",
        FocusAlgorithm.LAPLACIAN.value: "Laplacian edge detection based measure",
        FocusAlgorithm.BRENNER.value: "Squared pixel differences measure"
    }


if __name__ == "__main__":
    # Example usage and testing
    logging.basicConfig(level=logging.INFO)
    
    print("Gently DiSPIM Analysis")
    print("=====================")
    print()
    print("Device-agnostic focus analysis utilities")
    print("Compatible with any image data from any detector")
    print()
    print("Available Focus Algorithms:")
    for name, description in get_algorithm_info().items():
        print(f"  {name}: {description}")
    print()
    print("Available Fit Functions:")
    for func in FitFunction:
        print(f"  {func.value}")
    print()
    print("Main Functions:")
    print("  calculate_focus_score(image, algorithm)")
    print("  fit_focus_curve(positions, scores, fit_function)")
    print("  analyze_focus_stack(positions, images, config)")
    print()
    
    # Test with synthetic data
    print("Testing with synthetic focus curve...")
    positions = np.linspace(-10, 10, 21)
    # Gaussian-like focus curve with noise
    true_peak = 2.0
    scores = 100 * np.exp(-(positions - true_peak)**2 / 8) + np.random.normal(0, 5, len(positions))
    
    config = FocusAnalysisConfig(
        algorithm=FocusAlgorithm.VARIANCE.value,
        fit_function=FitFunction.GAUSSIAN.value
    )
    
    # Simulate images (just noise for this test)
    images = [np.random.rand(100, 100) * score/100 for score in scores]
    
    result = analyze_focus_stack(positions.tolist(), images, config)
    
    print(f"Analysis Result:")
    print(f"  Success: {result.success}")
    print(f"  Best Position: {result.best_position:.2f}")
    print(f"  R²: {result.r_squared:.3f}")
    print(f"  True Peak: {true_peak:.2f}")
    print(f"  Error: {abs(result.best_position - true_peak):.2f}")
    
    if result.error_message:
        print(f"  Error Message: {result.error_message}")