"""
Gently DiSPIM Plans
==================

Device-agnostic Bluesky plans for DiSPIM microscopy workflows.
Built using atomic plan stubs that compose into complex experimental procedures.

Autofocus serves as the "arrowhead" into the complete DiSPIM functionality,
including calibration, embryo detection, and multi-embryo acquisition workflows.

All plans are device-agnostic and use standard Bluesky plan stubs:
    - bps.mv(device, position) 
    - bps.trigger_and_read([detector])
    - bps.stage(device) / bps.unstage(device)
"""

import time
import logging
from typing import Dict, List, Optional, Tuple, Generator, Any, Union
from dataclasses import dataclass
import numpy as np

import bluesky.plans as bp
import bluesky.plan_stubs as bps
import bluesky.preprocessors as bpp
from bluesky import Msg
from bluesky.utils import short_uid


from .analysis.core import FocusAnalysisConfig
from .analysis.focus import (
    score_single_image, find_best_focus_position, analyze_focus_sweep,
    create_focus_positions, print_focus_summary, FocusDataPoint, FocusSweepResult
)


def focus_sweep_with_analysis(positioner, detector, positions: List[float],
                             config: FocusAnalysisConfig, callback=None,
                             metadata: Optional[Dict] = None) -> FocusSweepResult:
    """
    Atomic plan: Focus sweep with integrated analysis

    Pure device orchestration plan that:
    1. Moves through positions
    2. Captures images
    3. Scores focus
    4. Returns clean analysis result

    No complex logic, no detection algorithms - just device orchestration + analysis calls.

    Parameters
    ----------
    positioner : Ophyd positioner
        Device to move for focusing
    detector : Ophyd detector
        Camera device for image capture
    positions : List[float]
        List of positions to sweep through
    config : FocusAnalysisConfig
        Focus analysis configuration
    callback : callable, optional
        Callback for live plotting (scan_type, position, score, image, roi)
    metadata : Dict, optional
        Additional metadata for the scan

    Returns
    -------
    FocusSweepResult
        Analysis results from the sweep
    """
    if len(positions) < 3:
        raise ValueError(f"Need at least 3 positions for sweep, got {len(positions)}")

    md = {
        'plan_name': 'focus_sweep_with_analysis',
        'positioner': positioner.name,
        'detector': detector.name,
        'positions': positions,
        'config': config.__dict__
    }
    if metadata:
        md.update(metadata)

    # Data collection
    sweep_data = []

    @bpp.run_decorator(md=md)
    def inner():
        nonlocal sweep_data

        for i, pos in enumerate(positions):
            # Move to position
            yield from bps.mv(positioner, pos)
            actual_pos = yield from bps.rd(positioner)

            # Capture image
            yield from bps.trigger_and_read([detector, positioner],
                                          name=f'focus_point_{i:03d}')

            # Get image data (clean, simple way)
            image_data = yield from bps.rd(detector)
            image = image_data[detector.name]['value']

            # Score image (clean function call)
            score, roi = score_single_image(image, config, detect_roi=True)

            # Store data point
            sweep_data.append(FocusDataPoint(
                position=actual_pos,
                score=score,
                image=image,
                roi=roi
            ))

            print(f"Position {actual_pos:.2f} μm, focus score: {score:.2f}")

            # Callback for live plotting
            if callback:
                callback(metadata.get('scan_type', 'focus'), actual_pos, score, image, roi)

    # Execute the plan
    yield from inner()

    # Analyze collected data (pure function call)
    result = analyze_focus_sweep(sweep_data, config)
    return result


@dataclass
class AutofocusConfig:
    """Configuration for autofocus operations"""
    num_positions: int = 21
    step_size_um: float = 0.5
    algorithm: str = 'volath'  # 'volath', 'gradient', 'variance'
    fit_function: str = 'gaussian'  # 'gaussian', 'parabolic', 'none'
    minimum_r_squared: float = 0.75
    center_at_current: bool = True
    timeout_s: float = 60.0


@dataclass
class CalibrationConfig:
    """Configuration for two-point calibration"""
    point1_um: float = 25.0
    point2_um: float = 75.0
    autofocus_each_point: bool = True
    autofocus_config: Optional[AutofocusConfig] = None


# =============================================================================
# ATOMIC PLANS - Device-Agnostic Building Blocks
# =============================================================================

def focus_sweep(positioner, positions: List[float], detector, 
                metadata: Optional[Dict] = None) -> Generator[Msg, None, None]:
    """
    Device-agnostic focus sweep - works with ANY positioner and detector
    
    This is the fundamental atomic plan that underlies all autofocus operations.
    Can be used with piezo, galvo, xy_stage, focus_motor, etc.
    
    Parameters
    ----------
    positioner : Ophyd positioner device
        Any device that responds to bps.mv(positioner, position)
    positions : List[float]
        List of positions to sweep through
    detector : Ophyd detector device
        Any device that responds to bps.trigger_and_read([detector])
    metadata : Dict, optional
        Additional metadata for the scan
    """
    md = {
        'plan_name': 'focus_sweep',
        'positioner': positioner.name,
        'detector': detector.name,
        'positions': positions,
        'num_positions': len(positions)
    }
    if metadata:
        md.update(metadata)
    
    @bpp.run_decorator(md=md)
    def inner():
        for i, pos in enumerate(positions):
            # Move positioner
            yield from bps.mv(positioner, pos)
            
            # Acquire at this position
            yield from bps.trigger_and_read([detector, positioner], 
                                          name=f'focus_point_{i:03d}')
    
    yield from inner()



if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)