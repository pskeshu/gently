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



def test_lightsheet(lightsheet_snap,
                   sheet_width_deg: float = 2.0,
                   y_position_deg: float = 0.0,
                   metadata: Optional[Dict] = None) -> Generator[Msg, None, None]:
    """
    Test light sheet acquisition - simple plan for testing SPIM mode

    Configures light sheet, triggers acquisition, and returns image.
    Device-agnostic: works with any light sheet snap device.

    Parameters
    ----------
    lightsheet_snap : DiSPIMLightSheetSnap
        Light sheet snap device (scanner + camera)
    sheet_width_deg : float
        Light sheet width in degrees
    y_position_deg : float
        Y-axis position in degrees (Z-plane selection)
    metadata : Dict, optional
        Additional metadata

    Yields
    ------
    Msg
        Bluesky messages

    Example
    -------
    >>> from gently.devices import DiSPIMLightSheetSnap
    >>> ls_snap = DiSPIMLightSheetSnap("Scanner:AB:33", "HamCam1", core)
    >>> RE(test_lightsheet(ls_snap, sheet_width_deg=2.0, y_position_deg=0.0))
    """

    md = {
        'plan_name': 'test_lightsheet',
        'sheet_width_deg': sheet_width_deg,
        'y_position_deg': y_position_deg
    }
    if metadata:
        md.update(metadata)

    @bpp.run_decorator(md=md)
    def inner():
        # Configure light sheet
        lightsheet_snap.configure(
            sheet_width_deg=sheet_width_deg,
            y_position_deg=y_position_deg
        )

        print(f"Acquiring light sheet image: width={sheet_width_deg}°, Y={y_position_deg}°")

        # Trigger and read (standard Bluesky pattern!)
        yield from bps.trigger_and_read([lightsheet_snap], name='lightsheet_image')

        print("Light sheet image acquired")

    yield from inner()


# =============================================================================
# VOLUME ACQUISITION PLANS - Hardware-Triggered SPIM
# =============================================================================

def acquire_spim_volume(volume_scanner,
                       num_slices: int = 100,
                       exposure_ms: float = 5.0,
                       slice_step_um: float = 1.0,
                       metadata: Optional[Dict] = None) -> Generator[Msg, None, None]:
    """
    Acquire a hardware-triggered SPIM volume - atomic plan

    Single volume acquisition using Tiger controller hardware triggering.
    Device-agnostic: works with any DiSPIMVolumeScanner device.

    This plan encapsulates the complete SPIM volume acquisition workflow:
    - Camera configuration (PROGRESSIVE mode, EXTERNAL trigger)
    - SPIM timing calculation
    - Hardware-triggered acquisition
    - 3D volume retrieval

    Typical acquisition: 100 slices @ 59 fps (1.7 seconds total)

    Parameters
    ----------
    volume_scanner : DiSPIMVolumeScanner
        Volume scanner device (scanner + camera)
    num_slices : int
        Number of Z slices to acquire (default: 100)
    exposure_ms : float
        Camera exposure time in milliseconds (default: 5.0)
    slice_step_um : float
        Step size between slices in microns (default: 1.0)
    metadata : Dict, optional
        Additional metadata

    Yields
    ------
    Msg
        Bluesky messages

    Returns
    -------
    Volume data is stored in the Bluesky databroker with key 'volume_scanner'
    Access via: run.primary.read()['volume_scanner']['value']

    Example
    -------
    >>> from gently.devices import DiSPIMVolumeScanner
    >>> vol_scanner = DiSPIMVolumeScanner("Scanner:AB:33", "HamCam1", core)
    >>> RE(acquire_spim_volume(vol_scanner, num_slices=100, exposure_ms=5.0))
    """

    md = {
        'plan_name': 'acquire_spim_volume',
        'num_slices': num_slices,
        'exposure_ms': exposure_ms,
        'slice_step_um': slice_step_um,
        'expected_volume_shape': f'({num_slices}, 2304, 2304)',
    }
    if metadata:
        md.update(metadata)

    @bpp.run_decorator(md=md)
    def inner():
        # Configure volume scanner
        volume_scanner.configure(
            num_slices=num_slices,
            exposure_ms=exposure_ms,
            slice_step_um=slice_step_um
        )

        print(f"Acquiring SPIM volume: {num_slices} slices, {exposure_ms}ms exposure, {slice_step_um}μm step")

        # Trigger and read (standard Bluesky pattern!)
        yield from bps.trigger_and_read([volume_scanner], name='spim_volume')

        print(f"Volume acquired: {num_slices} slices")

    yield from inner()


def multi_position_volume(volume_scanner,
                         xy_stage,
                         positions: List[Tuple[float, float]],
                         num_slices: int = 100,
                         exposure_ms: float = 5.0,
                         slice_step_um: float = 1.0,
                         metadata: Optional[Dict] = None) -> Generator[Msg, None, None]:
    """
    Acquire SPIM volumes at multiple XY positions

    Device-agnostic plan for multi-position volume acquisition.
    Works with any XY stage and volume scanner combination.

    Parameters
    ----------
    volume_scanner : DiSPIMVolumeScanner
        Volume scanner device
    xy_stage : Ophyd XY stage device
        XY positioning stage (must have .x and .y attributes)
    positions : List[Tuple[float, float]]
        List of (x, y) positions in microns
    num_slices : int
        Number of Z slices per volume (default: 100)
    exposure_ms : float
        Camera exposure time in milliseconds (default: 5.0)
    slice_step_um : float
        Step size between slices in microns (default: 1.0)
    metadata : Dict, optional
        Additional metadata

    Yields
    ------
    Msg
        Bluesky messages

    Example
    -------
    >>> positions = [(0, 0), (100, 0), (0, 100), (100, 100)]
    >>> RE(multi_position_volume(vol_scanner, xy_stage, positions))
    """

    md = {
        'plan_name': 'multi_position_volume',
        'num_positions': len(positions),
        'positions': positions,
        'num_slices': num_slices,
        'exposure_ms': exposure_ms,
        'slice_step_um': slice_step_um,
    }
    if metadata:
        md.update(metadata)

    @bpp.run_decorator(md=md)
    def inner():
        # Configure volume scanner once
        volume_scanner.configure(
            num_slices=num_slices,
            exposure_ms=exposure_ms,
            slice_step_um=slice_step_um
        )

        print(f"Multi-position volume acquisition: {len(positions)} positions, {num_slices} slices each")

        for i, (x, y) in enumerate(positions):
            print(f"\nPosition {i+1}/{len(positions)}: ({x:.1f}, {y:.1f}) μm")

            # Move to position
            yield from bps.mv(xy_stage.x, x, xy_stage.y, y)

            # Acquire volume at this position
            yield from bps.trigger_and_read([volume_scanner, xy_stage],
                                          name=f'volume_pos_{i:03d}')

            print(f"  Volume {i+1} acquired")

        print(f"\nCompleted {len(positions)} volumes")

    yield from inner()


def volume_timelapse(volume_scanner,
                    num_timepoints: int,
                    interval_s: float,
                    num_slices: int = 100,
                    exposure_ms: float = 5.0,
                    slice_step_um: float = 1.0,
                    metadata: Optional[Dict] = None) -> Generator[Msg, None, None]:
    """
    Acquire time-lapse SPIM volumes

    Device-agnostic plan for time-lapse volume acquisition.
    Captures volumes at regular intervals.

    Parameters
    ----------
    volume_scanner : DiSPIMVolumeScanner
        Volume scanner device
    num_timepoints : int
        Number of time points to acquire
    interval_s : float
        Time interval between acquisitions in seconds
    num_slices : int
        Number of Z slices per volume (default: 100)
    exposure_ms : float
        Camera exposure time in milliseconds (default: 5.0)
    slice_step_um : float
        Step size between slices in microns (default: 1.0)
    metadata : Dict, optional
        Additional metadata

    Yields
    ------
    Msg
        Bluesky messages

    Example
    -------
    >>> # Acquire 10 volumes, one every 30 seconds
    >>> RE(volume_timelapse(vol_scanner, num_timepoints=10, interval_s=30.0))
    """

    md = {
        'plan_name': 'volume_timelapse',
        'num_timepoints': num_timepoints,
        'interval_s': interval_s,
        'num_slices': num_slices,
        'exposure_ms': exposure_ms,
        'slice_step_um': slice_step_um,
        'total_duration_s': num_timepoints * interval_s,
    }
    if metadata:
        md.update(metadata)

    @bpp.run_decorator(md=md)
    def inner():
        # Configure volume scanner once
        volume_scanner.configure(
            num_slices=num_slices,
            exposure_ms=exposure_ms,
            slice_step_um=slice_step_um
        )

        print(f"Time-lapse acquisition: {num_timepoints} volumes, {interval_s}s interval")
        print(f"Total duration: {num_timepoints * interval_s / 60:.1f} minutes")

        start_time = time.time()

        for t in range(num_timepoints):
            # Calculate when this acquisition should happen
            target_time = start_time + t * interval_s
            current_time = time.time()

            # Wait if we're ahead of schedule
            if current_time < target_time:
                wait_time = target_time - current_time
                print(f"\nTimepoint {t+1}/{num_timepoints}: waiting {wait_time:.1f}s...")
                time.sleep(wait_time)
            else:
                print(f"\nTimepoint {t+1}/{num_timepoints}:")

            # Acquire volume
            yield from bps.trigger_and_read([volume_scanner],
                                          name=f'volume_t_{t:03d}')

            elapsed = time.time() - start_time
            print(f"  Volume {t+1} acquired (elapsed: {elapsed:.1f}s)")

        print(f"\nCompleted {num_timepoints} time-lapse volumes")

    yield from inner()


def multi_position_volume_timelapse(volume_scanner,
                                   xy_stage,
                                   positions: List[Tuple[float, float]],
                                   num_timepoints: int,
                                   interval_s: float,
                                   num_slices: int = 100,
                                   exposure_ms: float = 5.0,
                                   slice_step_um: float = 1.0,
                                   metadata: Optional[Dict] = None) -> Generator[Msg, None, None]:
    """
    Acquire time-lapse SPIM volumes at multiple positions

    Device-agnostic plan combining multi-position and time-lapse acquisition.
    At each timepoint, visits all positions and acquires a volume.

    Parameters
    ----------
    volume_scanner : DiSPIMVolumeScanner
        Volume scanner device
    xy_stage : Ophyd XY stage device
        XY positioning stage (must have .x and .y attributes)
    positions : List[Tuple[float, float]]
        List of (x, y) positions in microns
    num_timepoints : int
        Number of time points to acquire
    interval_s : float
        Time interval between timepoint rounds in seconds
    num_slices : int
        Number of Z slices per volume (default: 100)
    exposure_ms : float
        Camera exposure time in milliseconds (default: 5.0)
    slice_step_um : float
        Step size between slices in microns (default: 1.0)
    metadata : Dict, optional
        Additional metadata

    Yields
    ------
    Msg
        Bluesky messages

    Example
    -------
    >>> positions = [(0, 0), (100, 0), (0, 100)]
    >>> # Acquire 3 positions every 60 seconds, 10 times
    >>> RE(multi_position_volume_timelapse(vol_scanner, xy_stage, positions,
    ...                                   num_timepoints=10, interval_s=60.0))
    """

    md = {
        'plan_name': 'multi_position_volume_timelapse',
        'num_positions': len(positions),
        'positions': positions,
        'num_timepoints': num_timepoints,
        'interval_s': interval_s,
        'num_slices': num_slices,
        'exposure_ms': exposure_ms,
        'slice_step_um': slice_step_um,
        'total_volumes': len(positions) * num_timepoints,
    }
    if metadata:
        md.update(metadata)

    @bpp.run_decorator(md=md)
    def inner():
        # Configure volume scanner once
        volume_scanner.configure(
            num_slices=num_slices,
            exposure_ms=exposure_ms,
            slice_step_um=slice_step_um
        )

        print(f"Multi-position time-lapse: {len(positions)} positions, {num_timepoints} timepoints")
        print(f"Total volumes: {len(positions) * num_timepoints}")

        start_time = time.time()

        for t in range(num_timepoints):
            # Calculate when this timepoint should start
            target_time = start_time + t * interval_s
            current_time = time.time()

            # Wait if we're ahead of schedule
            if current_time < target_time:
                wait_time = target_time - current_time
                print(f"\n{'='*60}")
                print(f"Timepoint {t+1}/{num_timepoints}: waiting {wait_time:.1f}s...")
                print(f"{'='*60}")
                time.sleep(wait_time)
            else:
                print(f"\n{'='*60}")
                print(f"Timepoint {t+1}/{num_timepoints}:")
                print(f"{'='*60}")

            # Visit all positions
            for i, (x, y) in enumerate(positions):
                print(f"  Position {i+1}/{len(positions)}: ({x:.1f}, {y:.1f}) μm")

                # Move to position
                yield from bps.mv(xy_stage.x, x, xy_stage.y, y)

                # Acquire volume
                yield from bps.trigger_and_read([volume_scanner, xy_stage],
                                              name=f'volume_t{t:03d}_p{i:03d}')

                print(f"    Volume acquired")

            elapsed = time.time() - start_time
            print(f"  Timepoint {t+1} complete (elapsed: {elapsed/60:.1f} min)")

        print(f"\nCompleted {len(positions) * num_timepoints} total volumes")

    yield from inner()


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)