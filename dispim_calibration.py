"""
DiSPIM Calibration Module

Python implementation of DiSPIM calibration procedures including:
- Two-point calibration for light sheet alignment
- Autofocus algorithms for precise focusing
- Movement correction and drift compensation
"""

import time
import logging
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import numpy as np
from scipy.optimize import curve_fit
from scipy.ndimage import laplace

from dispim_config import DiSPIMCore, DiSPIMConfig


class FocusScorer(Enum):
    """Focus scoring algorithms"""
    VARIANCE = "variance"
    LAPLACIAN = "laplacian"
    GRADIENT = "gradient"
    VOLATH = "volath"


@dataclass
class CalibrationPoint:
    """Single calibration point data"""
    piezo_position: float
    galvo_position: float
    focus_score: float
    timestamp: float


@dataclass
class CalibrationResult:
    """Results from two-point calibration"""
    slope: float
    offset: float
    r_squared: float
    points: List[CalibrationPoint]
    is_valid: bool


@dataclass
class AutofocusConfig:
    """Configuration for autofocus procedures"""
    num_images: int = 21
    step_size: float = 1.0  # μm
    scoring_algorithm: FocusScorer = FocusScorer.VOLATH
    min_r_squared: float = 0.8
    max_offset_change: float = 10.0  # μm
    movement_threshold: float = 0.1  # μm


class DiSPIMCalibration:
    """DiSPIM calibration and autofocus procedures"""
    
    def __init__(self, dispim_core: DiSPIMCore):
        self.core = dispim_core
        self.logger = logging.getLogger(__name__)
        self.autofocus_config = AutofocusConfig()
        self._last_calibration: Optional[CalibrationResult] = None
        self._focus_scorers = {
            FocusScorer.VARIANCE: self._variance_scorer,
            FocusScorer.LAPLACIAN: self._laplacian_scorer,
            FocusScorer.GRADIENT: self._gradient_scorer,
            FocusScorer.VOLATH: self._volath_scorer
        }
    
    def _variance_scorer(self, image: np.ndarray) -> float:
        """Variance-based focus scoring"""
        return np.var(image.astype(np.float64))
    
    def _laplacian_scorer(self, image: np.ndarray) -> float:
        """Laplacian variance focus scoring"""
        lap = laplace(image.astype(np.float64))
        return np.var(lap)
    
    def _gradient_scorer(self, image: np.ndarray) -> float:
        """Gradient magnitude focus scoring"""
        gy, gx = np.gradient(image.astype(np.float64))
        return np.mean(np.sqrt(gx**2 + gy**2))
    
    def _volath_scorer(self, image: np.ndarray) -> float:
        """VOLATH (Variance of Laplacian) focus scoring"""
        img = image.astype(np.float64)
        mean_val = np.mean(img)
        score = 0.0
        
        for i in range(img.shape[0] - 1):
            for j in range(img.shape[1]):
                score += (img[i,j] - mean_val) * (img[i+1,j] - mean_val)
        
        return score / (img.shape[0] * img.shape[1])
    
    def compute_focus_score(self, image: np.ndarray, 
                          scorer: FocusScorer = None) -> float:
        """Compute focus score for an image"""
        if scorer is None:
            scorer = self.autofocus_config.scoring_algorithm
        
        scorer_func = self._focus_scorers.get(scorer)
        if not scorer_func:
            raise ValueError(f"Unknown focus scorer: {scorer}")
        
        return scorer_func(image)
    
    def sweep_focus(self, device: str, center_pos: float, 
                   sweep_range: float, num_points: int = None,
                   step_size: float = None) -> Tuple[List[float], List[float]]:
        """Sweep device positions and collect focus scores"""
        if num_points is None:
            num_points = self.autofocus_config.num_images
        if step_size is None:
            step_size = self.autofocus_config.step_size
        
        # Calculate positions
        if num_points > 1:
            positions = np.linspace(center_pos - sweep_range/2, 
                                  center_pos + sweep_range/2, 
                                  num_points)
        else:
            positions = [center_pos]
        
        scores = []
        actual_positions = []
        
        self.logger.info(f"Starting focus sweep on {device}, "
                        f"{num_points} points, range {sweep_range}μm")
        
        for pos in positions:
            try:
                # Move device and wait for stability
                if device == self.core.config.piezo_imaging:
                    self.core.set_piezo_position(device, pos)
                elif device == self.core.config.galvo_device:
                    self.core.set_galvo_position(pos)
                else:
                    raise ValueError(f"Unsupported device for focus sweep: {device}")
                
                time.sleep(0.1)  # Brief settling time
                
                # Capture image and compute focus score
                image = self.core.snap_image()
                score = self.compute_focus_score(image)
                
                scores.append(score)
                actual_positions.append(pos)
                
                self.logger.debug(f"Position {pos:.2f}: score {score:.2f}")
                
            except Exception as e:
                self.logger.warning(f"Failed to measure at position {pos}: {e}")
                continue
        
        return actual_positions, scores
    
    def fit_focus_curve(self, positions: List[float], 
                       scores: List[float]) -> Tuple[float, float]:
        """Fit Gaussian curve to focus scores and find optimum"""
        if len(positions) < 3:
            raise ValueError("Need at least 3 points for curve fitting")
        
        positions = np.array(positions)
        scores = np.array(scores)
        
        # Gaussian function
        def gaussian(x, amp, mu, sigma):
            return amp * np.exp(-(x - mu)**2 / (2 * sigma**2))
        
        # Initial parameter estimates
        max_idx = np.argmax(scores)
        amp_guess = scores[max_idx]
        mu_guess = positions[max_idx]
        sigma_guess = (positions[-1] - positions[0]) / 4
        
        try:
            popt, pcov = curve_fit(gaussian, positions, scores, 
                                 p0=[amp_guess, mu_guess, sigma_guess],
                                 maxfev=1000)
            
            optimal_position = popt[1]  # mu parameter
            
            # Calculate R-squared
            ss_res = np.sum((scores - gaussian(positions, *popt)) ** 2)
            ss_tot = np.sum((scores - np.mean(scores)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            return optimal_position, r_squared
            
        except Exception as e:
            self.logger.warning(f"Curve fitting failed: {e}, using max score position")
            return positions[np.argmax(scores)], 0.0
    
    def run_autofocus(self, device: str, sweep_range: float = 20.0) -> float:
        """Run autofocus procedure on specified device"""
        current_pos = self.core.get_device_position(device)
        
        self.logger.info(f"Running autofocus on {device} around position {current_pos:.2f}")
        
        # Perform focus sweep
        positions, scores = self.sweep_focus(device, current_pos, sweep_range)
        
        if len(positions) < 3:
            raise RuntimeError("Insufficient focus measurement points")
        
        # Fit curve and find optimum
        optimal_pos, r_squared = self.fit_focus_curve(positions, scores)
        
        # Validate fit quality
        if r_squared < self.autofocus_config.min_r_squared:
            self.logger.warning(f"Poor autofocus fit quality: R² = {r_squared:.3f}")
        
        # Check movement threshold
        movement = abs(optimal_pos - current_pos)
        if movement > self.autofocus_config.max_offset_change:
            self.logger.warning(f"Large autofocus movement: {movement:.2f}μm")
        
        # Move to optimal position
        if device == self.core.config.piezo_imaging:
            self.core.set_piezo_position(device, optimal_pos)
        elif device == self.core.config.galvo_device:
            self.core.set_galvo_position(optimal_pos)
        
        self.logger.info(f"Autofocus complete: {current_pos:.2f} → {optimal_pos:.2f}μm "
                        f"(R² = {r_squared:.3f})")
        
        return optimal_pos
    
    def two_point_calibration(self, point1_piezo: float, point2_piezo: float,
                            auto_focus: bool = True) -> CalibrationResult:
        """Perform two-point calibration for light sheet alignment"""
        self.logger.info("Starting two-point calibration")
        
        points = []
        
        for i, piezo_pos in enumerate([point1_piezo, point2_piezo]):
            self.logger.info(f"Calibration point {i+1}: piezo = {piezo_pos}μm")
            
            # Move to piezo position
            self.core.set_piezo_position(self.core.config.piezo_imaging, piezo_pos)
            
            # Autofocus galvo if requested
            if auto_focus:
                galvo_pos = self.run_autofocus(self.core.config.galvo_device)
            else:
                galvo_pos = self.core.get_device_position(self.core.config.galvo_device)
            
            # Measure focus score at this position
            image = self.core.snap_image()
            focus_score = self.compute_focus_score(image)
            
            point = CalibrationPoint(
                piezo_position=piezo_pos,
                galvo_position=galvo_pos,
                focus_score=focus_score,
                timestamp=time.time()
            )
            points.append(point)
        
        # Calculate calibration slope and offset
        p1, p2 = points
        slope = (p2.galvo_position - p1.galvo_position) / (p2.piezo_position - p1.piezo_position)
        offset = p1.galvo_position - slope * p1.piezo_position
        
        # Calculate R-squared (simplified for two points)
        r_squared = 1.0  # Perfect fit for two points
        
        # Validate calibration
        is_valid = (abs(slope) < 10.0 and  # Reasonable slope
                   abs(offset) < 50.0)     # Reasonable offset
        
        result = CalibrationResult(
            slope=slope,
            offset=offset,
            r_squared=r_squared,
            points=points,
            is_valid=is_valid
        )
        
        # Update core calibration parameters
        if is_valid:
            self.core.config.calibration_slope = slope
            self.core.config.calibration_offset = offset
            self._last_calibration = result
            
            self.logger.info(f"Calibration complete: slope={slope:.4f}, offset={offset:.2f}")
        else:
            self.logger.error(f"Calibration failed validation: slope={slope:.4f}, offset={offset:.2f}")
        
        return result
    
    def update_calibration_offset(self) -> float:
        """Update calibration offset without changing slope"""
        if not self._last_calibration:
            raise RuntimeError("No previous calibration available")
        
        # Get current positions
        piezo_pos = self.core.get_device_position(self.core.config.piezo_imaging)
        
        # Run autofocus to find optimal galvo position
        optimal_galvo = self.run_autofocus(self.core.config.galvo_device)
        
        # Calculate new offset using existing slope
        slope = self._last_calibration.slope
        new_offset = optimal_galvo - slope * piezo_pos
        
        # Update calibration
        old_offset = self.core.config.calibration_offset
        self.core.config.calibration_offset = new_offset
        
        offset_change = abs(new_offset - old_offset)
        
        self.logger.info(f"Updated calibration offset: {old_offset:.2f} → {new_offset:.2f} "
                        f"(change: {offset_change:.2f})")
        
        return new_offset
    
    def validate_calibration(self, test_positions: List[float] = None) -> Dict[str, float]:
        """Validate current calibration by testing at multiple positions"""
        if test_positions is None:
            # Default test positions across range
            center = self.core.config.piezo_center
            test_positions = [center - 20, center, center + 20]
        
        validation_results = {
            'mean_error': 0.0,
            'max_error': 0.0,
            'rms_error': 0.0,
            'num_points': len(test_positions)
        }
        
        errors = []
        
        for piezo_pos in test_positions:
            # Move to test position
            self.core.synchronized_move(piezo_pos)
            
            # Measure actual optimal galvo position
            actual_galvo = self.run_autofocus(self.core.config.galvo_device)
            
            # Compare with predicted position
            predicted_galvo = self.core.compute_galvo_from_piezo(piezo_pos)
            error = abs(actual_galvo - predicted_galvo)
            errors.append(error)
            
            self.logger.debug(f"Position {piezo_pos}: predicted={predicted_galvo:.2f}, "
                            f"actual={actual_galvo:.2f}, error={error:.2f}")
        
        # Calculate error statistics
        validation_results['mean_error'] = np.mean(errors)
        validation_results['max_error'] = np.max(errors)
        validation_results['rms_error'] = np.sqrt(np.mean(np.array(errors)**2))
        
        self.logger.info(f"Calibration validation: mean error={validation_results['mean_error']:.2f}μm, "
                        f"max error={validation_results['max_error']:.2f}μm")
        
        return validation_results


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    from dispim_config import DiSPIMCore
    
    # Initialize DiSPIM and calibration
    dispim = DiSPIMCore()
    calibration = DiSPIMCalibration(dispim)
    
    print("DiSPIM Calibration module loaded successfully")
    print("Available focus scorers:", list(FocusScorer))