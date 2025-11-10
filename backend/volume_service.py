"""
Volume acquisition service for multi-embryo SPIM imaging.

Extracted from run_multi_embryo_volumes.py and adapted for async operation.
"""

import sys
from pathlib import Path

# Add parent directory to Python path to import client.py
backend_dir = Path(__file__).parent
parent_dir = backend_dir.parent
sys.path.insert(0, str(parent_dir))

import time
import numpy as np
from datetime import datetime, timedelta
from client import get_mmc
import tifffile
import rpyc
from typing import Dict, List, Callable, Optional
import asyncio

# Device configuration
core = get_mmc()
CAMERA_NAME_SPIM = "HamCam1"
GALVO_DEVICE = "Scanner:AB:33"
PIEZO_DEVICE = "PiezoStage:P:34"
XY_STAGE_NAME = "XYStage:XY:31"


def move_to_embryo_position(stage_x: float, stage_y: float):
    """
    Move stage to embryo's calibrated position.

    Parameters
    ----------
    stage_x, stage_y : float
        Target position in micrometers
    """
    print(f"  Moving to embryo position: ({stage_x:.2f}, {stage_y:.2f}) µm")

    core.setXYStageDevice(XY_STAGE_NAME)
    core.setXYPosition(float(stage_x), float(stage_y))
    core.waitForDevice(XY_STAGE_NAME)
    time.sleep(0.5)


def configure_hardware_for_volume(calibration: Dict, num_slices: int):
    """
    Configure galvo, piezo, camera for hardware-triggered volume acquisition.

    Parameters
    ----------
    calibration : dict
        Calibration data with slope, offset, and edge positions
    num_slices : int
        Number of slices to acquire

    Returns
    -------
    dict
        Volume parameters (galvo_center, galvo_amplitude, piezo_center, piezo_amplitude)
    """
    print(f"  Configuring hardware for {num_slices} slices...")

    # Stop any existing sequence acquisition
    if core.isSequenceRunning():
        print(f"    Stopping previous sequence...")
        core.stopSequenceAcquisition()
        time.sleep(0.5)

    # Clear any pending images
    core.clearCircularBuffer()

    # Reset SPIM state machine
    try:
        core.setProperty(GALVO_DEVICE, "SPIMState", "Idle")
        time.sleep(0.2)
    except:
        pass

    # Extract calibration parameters
    slope = calibration['slope_um_per_deg']
    offset = calibration['offset_um']
    galvo_top = calibration.get('edge_top_deg', calibration['galvo_top_deg'])
    galvo_bottom = calibration.get('edge_bottom_deg', calibration['galvo_bottom_deg'])

    # Calculate galvo parameters
    galvo_center = (galvo_top + galvo_bottom) / 2.0
    galvo_range = galvo_bottom - galvo_top
    galvo_amplitude = galvo_range / 2.0

    # Calculate piezo parameters
    piezo_top = slope * galvo_top + offset
    piezo_bottom = slope * galvo_bottom + offset
    piezo_center = (piezo_top + piezo_bottom) / 2.0
    piezo_range = piezo_bottom - piezo_top
    piezo_amplitude = piezo_range / 2.0

    print(f"    Galvo: center={galvo_center:+.4f}°, amplitude=±{galvo_amplitude:.4f}° (range: {galvo_range:.4f}°)")
    print(f"    Piezo: center={piezo_center:.1f}µm, amplitude=±{piezo_amplitude:.1f}µm (range: {piezo_range:.1f}µm)")

    # System startup
    core.setConfig("System", "Startup")
    core.waitForConfig("System", "Startup")

    # Lasers on
    core.setConfig("Laser", "488 and 561")
    core.waitForConfig("Laser", "488 and 561")

    # Camera for hardware trigger
    core.setCameraDevice(CAMERA_NAME_SPIM)
    roi_x = 128
    roi_y = 896
    roi_width = 2048
    roi_height = 512
    core.setROI(CAMERA_NAME_SPIM, roi_x, roi_y, roi_width, roi_height)
    core.setProperty(CAMERA_NAME_SPIM, "TRIGGER SOURCE", "EXTERNAL")
    core.setProperty(CAMERA_NAME_SPIM, "SENSOR MODE", "PROGRESSIVE")
    core.setProperty(CAMERA_NAME_SPIM, "TRIGGER ACTIVE", "EDGE")
    core.setExposure(CAMERA_NAME_SPIM, 10.0)

    # Configure galvo for SPIM
    core.setProperty(GALVO_DEVICE, "SPIMState", "Idle")
    time.sleep(0.2)

    core.setProperty(GALVO_DEVICE, "LaserOutputMode", "shutter + side")
    core.setProperty(GALVO_DEVICE, "BeamEnabled", "No")

    # X-axis scanning (light sheet width)
    core.setProperty(GALVO_DEVICE, "SingleAxisXAmplitude(deg)", 8.0)
    core.setProperty(GALVO_DEVICE, "SingleAxisXOffset(deg)", 0.0005)
    core.setProperty(GALVO_DEVICE, "SingleAxisXPattern", "1 - Triangle")

    # Y-axis positioning (synchronized with piezo)
    core.setProperty(GALVO_DEVICE, "SingleAxisYAmplitude(deg)", float(galvo_amplitude))
    core.setProperty(GALVO_DEVICE, "SingleAxisYOffset(deg)", float(galvo_center))
    core.setProperty(GALVO_DEVICE, "SingleAxisYPattern", "1 - Triangle")

    # SPIM timing
    core.setProperty(GALVO_DEVICE, "SPIMDelayBeforeScan(ms)", 6.75)
    core.setProperty(GALVO_DEVICE, "SPIMNumScansPerSlice", 1)
    core.setProperty(GALVO_DEVICE, "SPIMScanDuration(ms)", 5.5)
    core.setProperty(GALVO_DEVICE, "SPIMDelayBeforeLaser(ms)", 8.0)
    core.setProperty(GALVO_DEVICE, "SPIMLaserDuration(ms)", 5.0)
    core.setProperty(GALVO_DEVICE, "SPIMDelayBeforeCamera(ms)", 8.0)
    core.setProperty(GALVO_DEVICE, "SPIMCameraDuration(ms)", 1.0)

    core.setProperty(GALVO_DEVICE, "SPIMNumSlices", num_slices)
    core.setProperty(GALVO_DEVICE, "SPIMNumSlicesPerPiezo", 1)
    core.setProperty(GALVO_DEVICE, "SPIMNumSides", 1)
    core.setProperty(GALVO_DEVICE, "SPIMFirstSide", "A")

    # Configure piezo for SPIM
    core.setFocusDevice(PIEZO_DEVICE)
    core.setProperty(PIEZO_DEVICE, "SingleAxisAmplitude(um)", float(piezo_amplitude))
    core.setProperty(PIEZO_DEVICE, "SingleAxisOffset(um)", float(piezo_center))
    core.setProperty(PIEZO_DEVICE, "SingleAxisPattern", "1 - Triangle")
    core.setProperty(PIEZO_DEVICE, "SPIMNumSlices", num_slices)
    core.setProperty(PIEZO_DEVICE, "SPIMState", "Armed")

    time.sleep(0.3)
    print(f"  ✓ Hardware configured for hardware-triggered acquisition")

    return {
        'galvo_center': galvo_center,
        'galvo_amplitude': galvo_amplitude,
        'piezo_center': piezo_center,
        'piezo_amplitude': piezo_amplitude
    }


def acquire_volume_stack(num_slices: int, progress_callback: Optional[Callable] = None) -> Optional[np.ndarray]:
    """
    Acquire volume stack using hardware-triggered SPIM acquisition.

    Parameters
    ----------
    num_slices : int
        Number of slices to acquire
    progress_callback : callable, optional
        Callback function(current_slice, total_slices) for progress updates

    Returns
    -------
    np.ndarray or None
        Volume stack (slices, height, width), or None if failed
    """
    print(f"  Acquiring {num_slices} slices with hardware triggering...")

    # Clear buffer and prepare sequence acquisition
    core.clearCircularBuffer()

    # Set buffer capacity
    buffer_capacity = core.getBufferTotalCapacity()
    if buffer_capacity < num_slices:
        core.setCircularBufferMemoryFootprint(512)
        time.sleep(0.1)
        buffer_capacity = core.getBufferTotalCapacity()
        print(f"    Buffer capacity: {buffer_capacity}")

    core.prepareSequenceAcquisition(CAMERA_NAME_SPIM)
    time.sleep(0.1)

    core.startSequenceAcquisition(CAMERA_NAME_SPIM, num_slices, 0, True)
    time.sleep(0.1)

    # Trigger SPIM state machine
    core.setProperty(GALVO_DEVICE, "SPIMState", "Running")
    print(f"  ✓ SPIM triggered")

    # Collect images
    images = []
    slice_period_ms = 50.0  # Approximate time per slice
    timeout_s = num_slices * slice_period_ms / 1000.0 * 2
    start_time = time.time()

    print(f"  Waiting for {num_slices} images...")

    while core.getRemainingImageCount() > 0 or core.isSequenceRunning():
        if core.getRemainingImageCount() > 0:
            img = core.popNextImage()

            try:
                img = rpyc.classic.obtain(img)
            except (ImportError, AttributeError):
                pass

            images.append(img)

            # Progress callback
            if progress_callback:
                progress_callback(len(images), num_slices)

            if len(images) % 10 == 0:
                print(f"    Received {len(images)}/{num_slices} images...")

        if time.time() - start_time > timeout_s:
            print(f"  ⚠ Timeout after {timeout_s:.1f}s!")
            break

        time.sleep(0.01)

    # Stop sequence
    if core.isSequenceRunning():
        core.stopSequenceAcquisition()

    print(f"  ✓ Acquired {len(images)} slices")

    if len(images) == 0:
        return None

    volume = np.array(images)
    return volume


def save_volume_tiff(volume: np.ndarray, output_dir: Path, embryo_id: str, embryo_number: int, timepoint: int) -> Path:
    """
    Save volume as TIFF file.

    Parameters
    ----------
    volume : np.ndarray
        Volume stack
    output_dir : Path
        Output directory
    embryo_id : str
        Embryo identifier
    embryo_number : int
        Embryo number
    timepoint : int
        Timepoint index

    Returns
    -------
    Path
        Path to saved TIFF file
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = output_dir / f"{embryo_id}_embryo{embryo_number:03d}_t{timepoint:04d}_{timestamp}.tif"

    tifffile.imwrite(filename, volume)
    print(f"  ✓ Saved: {filename.name}")

    return filename


def cleanup_hardware():
    """Turn off lasers and reset hardware state."""
    try:
        core.setConfig("Laser", "ALL OFF")
        print(f"  ✓ Lasers OFF")
    except Exception as e:
        print(f"  ⚠ Failed to turn off lasers: {e}")


async def run_volume_acquisition(
    embryos: List[Dict],
    num_slices: int,
    num_timepoints: int,
    interval_minutes: float,
    output_dir: Path,
    progress_callback: Optional[Callable] = None
) -> List[Dict]:
    """
    Run full volume acquisition workflow for multiple embryos with timelapse.

    Parameters
    ----------
    embryos : list of dict
        List of embryo data with calibration and position info
    num_slices : int
        Number of slices per volume
    num_timepoints : int
        Number of timepoints
    interval_minutes : float
        Interval between timepoints in minutes
    output_dir : Path
        Output directory for TIFF files
    progress_callback : callable, optional
        Async callback function for progress updates

    Returns
    -------
    list of dict
        List of acquisition results
    """
    results = []

    for timepoint in range(num_timepoints):
        timepoint_start = time.time()

        print(f"\n{'='*70}")
        print(f"TIMEPOINT {timepoint + 1}/{num_timepoints}")
        print(f"{'='*70}")

        # Acquire volume for each embryo
        for idx, embryo in enumerate(embryos):
            embryo_id = embryo['embryo_id']
            embryo_number = embryo['embryo_number']
            calibration = embryo['calibration']
            stage_x = embryo['stage_x_centered']
            stage_y = embryo['stage_y_centered']

            print(f"\n[Embryo {idx+1}/{len(embryos)}] {embryo_id} (Embryo #{embryo_number})")
            print(f"{'─'*70}")

            try:
                # Notify progress
                if progress_callback:
                    await progress_callback({
                        'type': 'embryo_progress',
                        'embryo_id': embryo['db_id'],
                        'embryo_number': embryo_number,
                        'total_embryos': len(embryos),
                        'timepoint': timepoint,
                        'stage': 'moving'
                    })

                # Move to embryo
                move_to_embryo_position(stage_x, stage_y)

                # Configure hardware
                if progress_callback:
                    await progress_callback({
                        'type': 'embryo_progress',
                        'embryo_id': embryo['db_id'],
                        'embryo_number': embryo_number,
                        'stage': 'configuring'
                    })

                configure_hardware_for_volume(calibration, num_slices)

                # Acquire volume
                if progress_callback:
                    await progress_callback({
                        'type': 'embryo_progress',
                        'embryo_id': embryo['db_id'],
                        'embryo_number': embryo_number,
                        'stage': 'acquiring'
                    })

                # Create progress callback for slices
                async def slice_progress(current, total):
                    if progress_callback:
                        await progress_callback({
                            'type': 'slice_progress',
                            'current_slice': current,
                            'total_slices': total,
                            'percentage': (current / total) * 100
                        })

                # Wrap sync progress callback
                def sync_slice_progress(current, total):
                    # Run async callback in event loop
                    try:
                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            asyncio.create_task(slice_progress(current, total))
                    except:
                        pass

                volume = acquire_volume_stack(num_slices, sync_slice_progress)

                if volume is None:
                    print(f"  ✗ Failed to acquire volume")
                    results.append({
                        'embryo_id': embryo['db_id'],
                        'embryo_number': embryo_number,
                        'timepoint': timepoint,
                        'success': False,
                        'error': 'Volume acquisition returned no data'
                    })
                    continue

                # Save volume
                if progress_callback:
                    await progress_callback({
                        'type': 'embryo_progress',
                        'embryo_id': embryo['db_id'],
                        'embryo_number': embryo_number,
                        'stage': 'saving'
                    })

                filename = save_volume_tiff(volume, output_dir, embryo_id, embryo_number, timepoint)

                results.append({
                    'embryo_id': embryo['db_id'],
                    'embryo_number': embryo_number,
                    'timepoint': timepoint,
                    'success': True,
                    'filename': str(filename),
                    'shape': volume.shape
                })

                print(f"  ✓ Complete: {volume.shape}")

            except Exception as e:
                print(f"  ✗ Error: {e}")
                import traceback
                traceback.print_exc()

                results.append({
                    'embryo_id': embryo['db_id'],
                    'embryo_number': embryo_number,
                    'timepoint': timepoint,
                    'success': False,
                    'error': str(e)
                })

        # Wait for next timepoint if needed
        if timepoint < num_timepoints - 1 and interval_minutes > 0:
            timepoint_duration = time.time() - timepoint_start
            wait_time = interval_minutes * 60 - timepoint_duration

            if wait_time > 0:
                next_time = datetime.now() + timedelta(seconds=wait_time)
                print(f"\n{'─'*70}")
                print(f"Waiting {wait_time/60:.1f} minutes until next timepoint...")
                print(f"Next timepoint at: {next_time.strftime('%H:%M:%S')}")
                print(f"{'─'*70}")

                if progress_callback:
                    await progress_callback({
                        'type': 'timepoint_complete',
                        'timepoint': timepoint,
                        'total_timepoints': num_timepoints,
                        'next_timepoint_at': next_time.isoformat()
                    })

                # Async sleep
                await asyncio.sleep(wait_time)
            else:
                print(f"\n{'─'*70}")
                print(f"⚠ Warning: Acquisition took {timepoint_duration/60:.1f} min (longer than {interval_minutes} min interval)")
                print(f"Proceeding immediately to next timepoint...")
                print(f"{'─'*70}")

    # Cleanup
    cleanup_hardware()

    return results
