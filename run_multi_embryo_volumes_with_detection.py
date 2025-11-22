#!/usr/bin/env python3
"""
Run Multi-Embryo Volume Acquisition with Real-time Hatching Detection
======================================================================

Integrates Claude Vision API for on-the-fly hatching detection.
Stops acquiring individual embryos when hatched, ends when all are hatched.

Usage:
    python run_multi_embryo_volumes_with_detection.py
"""

import time
import json
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
from client import get_mmc
import tifffile
import rpyc
from tqdm import tqdm

# Import hatching detection modules
from realtime_hatching_detector import RealtimeHatchingDetector
from image_processing_utils import process_volume_for_claude, ImageHistory, save_processed_image

# Device configuration
core = get_mmc()
XY_STAGE_NAME = "XYStage:XY:31"
CAMERA_NAME = "HamCam1"
GALVO_DEVICE = "Scanner:AB:33"
PIEZO_DEVICE = "PiezoStage:P:34"

# Database file
DATABASE_FILE = Path("multi_embryo_database.json")

# Output directory
OUTPUT_DIR = Path("multi_embryo_volumes")


# ========== CONFIGURATION ==========
HATCHING_DETECTION_CONFIG = {
    'enabled': True,                        # Enable real-time detection
    'min_timepoints_before_detection': 50,  # Don't check before this (embryos need ~100min to develop)
    'confidence_threshold': 'HIGH',         # Required confidence (HIGH/MEDIUM/LOW)
    'image_history_window': 10,             # Number of recent images to send to Claude
    'stop_when_all_hatched': True,          # End acquisition when all hatched
    'continue_after_hatching': 5,           # Continue for N timepoints after embryo hatches (for confirmation)
    'save_processed_images': True,          # Save max projections for debugging
    'detection_log_file': 'hatching_detection_log.json'  # Log file for detections
}


def load_database():
    """Load embryo database."""
    if not DATABASE_FILE.exists():
        raise FileNotFoundError(f"Database not found: {DATABASE_FILE}\nRun multi_embryo_calibration.py first!")

    with open(DATABASE_FILE, 'r') as f:
        return json.load(f)


def get_stage_position():
    """Get current XY stage position."""
    x = core.getXPosition(XY_STAGE_NAME)
    y = core.getYPosition(XY_STAGE_NAME)
    return (x, y)


def move_to_embryo(embryo_data):
    """Move stage to embryo's calibrated position."""
    target_x = embryo_data['stage_position_after_centering_um']['x']
    target_y = embryo_data['stage_position_after_centering_um']['y']

    print(f"  Moving to embryo position: ({target_x:.2f}, {target_y:.2f}) µm")

    core.setXYStageDevice(XY_STAGE_NAME)
    core.setXYPosition(float(target_x), float(target_y))
    core.waitForDevice(XY_STAGE_NAME)
    time.sleep(0.5)

    actual_pos = get_stage_position()
    print(f"  ✓ At position: ({actual_pos[0]:.2f}, {actual_pos[1]:.2f}) µm")


def configure_hardware_for_volume(calibration, num_slices):
    """Configure galvo, piezo, camera for hardware-triggered volume acquisition."""
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

    print(f"    Galvo: center={galvo_center:+.4f}°, amplitude=±{galvo_amplitude:.4f}°")
    print(f"    Piezo: center={piezo_center:.1f}µm, amplitude=±{piezo_amplitude:.1f}µm")

    # System startup
    core.setConfig("System", "Startup")
    core.waitForConfig("System", "Startup")

    # Lasers on
    core.setConfig("Laser", "488 and 561")
    core.waitForConfig("Laser", "488 and 561")

    # Camera for hardware trigger
    core.setCameraDevice(CAMERA_NAME)
    roi_x = 128
    roi_y = 896
    roi_width = 2048
    roi_height = 512
    core.setROI(CAMERA_NAME, roi_x, roi_y, roi_width, roi_height)
    core.setProperty(CAMERA_NAME, "TRIGGER SOURCE", "EXTERNAL")
    core.setProperty(CAMERA_NAME, "SENSOR MODE", "PROGRESSIVE")
    core.setProperty(CAMERA_NAME, "TRIGGER ACTIVE", "EDGE")
    core.setExposure(CAMERA_NAME, 10.0)

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
    print(f"  ✓ Hardware configured")

    return {
        'galvo_center': galvo_center,
        'galvo_amplitude': galvo_amplitude,
        'piezo_center': piezo_center,
        'piezo_amplitude': piezo_amplitude
    }


def acquire_volume_for_embryo(embryo_id, calibration, num_slices=50):
    """Acquire volume for one embryo using hardware-triggered SPIM acquisition."""
    print(f"  Acquiring {num_slices} slices...")

    # Clear buffer and prepare sequence acquisition
    core.clearCircularBuffer()

    # Set buffer capacity
    buffer_capacity = core.getBufferTotalCapacity()
    if buffer_capacity < num_slices:
        core.setCircularBufferMemoryFootprint(512)
        time.sleep(0.1)
        buffer_capacity = core.getBufferTotalCapacity()

    core.prepareSequenceAcquisition(CAMERA_NAME)
    time.sleep(0.1)

    core.startSequenceAcquisition(CAMERA_NAME, num_slices, 0, True)
    time.sleep(0.1)

    # Trigger SPIM state machine
    core.setProperty(GALVO_DEVICE, "SPIMState", "Running")

    # Collect images
    images = []
    slice_period_ms = 50.0
    timeout_s = num_slices * slice_period_ms / 1000.0 * 2
    start_time = time.time()

    while core.getRemainingImageCount() > 0 or core.isSequenceRunning():
        if core.getRemainingImageCount() > 0:
            img = core.popNextImage()

            try:
                img = rpyc.classic.obtain(img)
            except (ImportError, AttributeError):
                pass

            images.append(img)

        if time.time() - start_time > timeout_s:
            print(f"  ⚠ Timeout!")
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


def should_acquire_embryo(embryo_id, detector, timepoint, config):
    """
    Determine if embryo should be acquired

    Parameters
    ----------
    embryo_id : str
        Embryo identifier
    detector : RealtimeHatchingDetector
        Hatching detector instance
    timepoint : int
        Current timepoint
    config : dict
        Detection configuration

    Returns
    -------
    tuple
        (should_acquire: bool, reason: str)
    """
    if not config['enabled']:
        return True, "Detection disabled"

    if not detector.is_hatched(embryo_id):
        return True, "Not yet hatched"

    # Embryo has hatched - check if we should continue for confirmation
    hatching_timepoint = detector.get_hatching_timepoint(embryo_id)
    timepoints_since_hatching = timepoint - hatching_timepoint

    if timepoints_since_hatching < config['continue_after_hatching']:
        return True, f"Confirmation ({timepoints_since_hatching + 1}/{config['continue_after_hatching']})"

    return False, f"Hatched at t{hatching_timepoint:04d}"


def main():
    """Main multi-embryo volume acquisition with hatching detection."""
    print(f"{'='*70}")
    print("MULTI-EMBRYO ACQUISITION WITH REAL-TIME HATCHING DETECTION")
    print(f"{'='*70}")

    try:
        # Load database
        print(f"\n{'='*70}")
        print("LOADING DATABASE")
        print(f"{'='*70}")
        database = load_database()

        embryos = database.get('embryos', {})
        num_embryos = len(embryos)

        print(f"  Database: {DATABASE_FILE}")
        print(f"  Found {num_embryos} embryo(s)")

        if num_embryos == 0:
            print(f"\n  ⚠ No embryos in database!")
            return

        # List embryos
        print(f"\n  Embryos:")
        for emb_id, emb_data in embryos.items():
            emb_num = emb_data.get('embryo_number', '?')
            pos = emb_data['stage_position_after_centering_um']
            print(f"    {emb_id} (#{emb_num}): ({pos['x']:.1f}, {pos['y']:.1f}) µm")

        # Acquisition parameters
        print(f"\n{'='*70}")
        print("ACQUISITION PARAMETERS")
        print(f"{'='*70}")

        num_slices = int(input(f"  Number of slices per volume (default 50): ").strip() or "50")
        print(f"  ✓ Will acquire {num_slices} slices per embryo")

        # Timelapse parameters
        num_timepoints = int(input(f"  Number of timepoints (default 500 for ~16h at 2min intervals): ").strip() or "500")
        interval_minutes = float(input(f"  Interval between timepoints in minutes (default 2): ").strip() or "2")
        total_duration_hours = (num_timepoints - 1) * interval_minutes / 60.0
        print(f"  ✓ Timelapse: {num_timepoints} timepoints every {interval_minutes} min ({total_duration_hours:.1f} hours total)")

        # Hatching detection configuration
        print(f"\n{'='*70}")
        print("HATCHING DETECTION CONFIGURATION")
        print(f"{'='*70}")
        print(f"  Enabled: {HATCHING_DETECTION_CONFIG['enabled']}")
        print(f"  Min timepoints before detection: {HATCHING_DETECTION_CONFIG['min_timepoints_before_detection']} ({HATCHING_DETECTION_CONFIG['min_timepoints_before_detection'] * interval_minutes:.0f} min)")
        print(f"  Confidence threshold: {HATCHING_DETECTION_CONFIG['confidence_threshold']}")
        print(f"  Image history window: {HATCHING_DETECTION_CONFIG['image_history_window']} timepoints")
        print(f"  Stop when all hatched: {HATCHING_DETECTION_CONFIG['stop_when_all_hatched']}")
        print(f"  Continue after hatching: {HATCHING_DETECTION_CONFIG['continue_after_hatching']} timepoints")

        # Create output directory
        session_dir = OUTPUT_DIR / datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n  ✓ Output: {session_dir}")

        # Initialize hatching detection
        detector = None
        image_history = None
        if HATCHING_DETECTION_CONFIG['enabled']:
            print(f"\n  Initializing hatching detector...")
            detector = RealtimeHatchingDetector(
                min_timepoints_before_detection=HATCHING_DETECTION_CONFIG['min_timepoints_before_detection'],
                confidence_threshold=HATCHING_DETECTION_CONFIG['confidence_threshold']
            )
            image_history = ImageHistory(window_size=HATCHING_DETECTION_CONFIG['image_history_window'])
            print(f"  ✓ Hatching detector initialized")

        # Create processed images directory
        if HATCHING_DETECTION_CONFIG['save_processed_images']:
            processed_dir = session_dir / "processed_images"
            processed_dir.mkdir(exist_ok=True)

        # Timelapse loop
        all_results = []
        session_start_time = time.time()

        # Progress bar for timepoints
        timepoint_pbar = tqdm(
            total=num_timepoints,
            desc="Timepoints",
            unit="tp",
            position=0,
            colour='green'
        )

        for timepoint in range(num_timepoints):
            timepoint_start_time = time.time()
            elapsed_hours = (timepoint_start_time - session_start_time) / 3600.0

            # Update progress bar
            timepoint_pbar.set_description(f"Timepoint {timepoint+1}/{num_timepoints} (Elapsed: {elapsed_hours:.1f}h)")

            print(f"\n{'='*70}")
            print(f"TIMEPOINT {timepoint + 1}/{num_timepoints}")
            print(f"Elapsed: {elapsed_hours:.2f} hours")
            print(f"{'='*70}")

            # Check if all embryos hatched
            if detector and HATCHING_DETECTION_CONFIG['stop_when_all_hatched']:
                embryo_ids = list(embryos.keys())
                if detector.all_embryos_hatched(embryo_ids):
                    # Check if we've done enough confirmation timepoints
                    all_confirmed = True
                    for eid in embryo_ids:
                        hatch_tp = detector.get_hatching_timepoint(eid)
                        if timepoint - hatch_tp < HATCHING_DETECTION_CONFIG['continue_after_hatching']:
                            all_confirmed = False
                            break

                    if all_confirmed:
                        print(f"\n🎉 ALL EMBRYOS HATCHED AND CONFIRMED!")
                        print(f"Ending acquisition early at timepoint {timepoint}")
                        break

            # Acquire volume for each embryo
            timepoint_results = []

            # Count active embryos
            active_embryos = []
            for emb_id in embryos.keys():
                should_acq, reason = should_acquire_embryo(emb_id, detector, timepoint, HATCHING_DETECTION_CONFIG)
                if should_acq:
                    active_embryos.append(emb_id)

            print(f"Active embryos: {len(active_embryos)}/{num_embryos}")

            # Progress bar for embryos
            embryo_pbar = tqdm(
                total=num_embryos,
                desc=f"  Embryos (t{timepoint:04d})",
                unit="embryo",
                position=1,
                leave=False,
                colour='cyan'
            )

            for idx, (emb_id, emb_data) in enumerate(embryos.items(), 1):
                emb_num = emb_data.get('embryo_number', idx)

                # Check if should acquire
                should_acq, reason = should_acquire_embryo(emb_id, detector, timepoint, HATCHING_DETECTION_CONFIG)

                embryo_pbar.set_description(f"  Embryo {emb_num} (t{timepoint:04d})")

                if not should_acq:
                    print(f"\n[Embryo {idx}/{num_embryos}] {emb_id} - SKIPPED ({reason})")
                    timepoint_results.append({
                        'embryo_id': emb_id,
                        'embryo_number': emb_num,
                        'timepoint': timepoint,
                        'skipped': True,
                        'reason': reason
                    })
                    embryo_pbar.update(1)
                    continue

                print(f"\n[Embryo {idx}/{num_embryos}] {emb_id} (#{emb_num}) - {reason}")
                print(f"{'─'*70}")

                # Move to embryo
                move_to_embryo(emb_data)

                # Configure hardware
                calibration = emb_data['calibration']
                configure_hardware_for_volume(calibration, num_slices)

                # Acquire volume
                volume = acquire_volume_for_embryo(emb_id, calibration, num_slices)

                if volume is None:
                    print(f"  ✗ Failed to acquire volume")
                    timepoint_results.append({
                        'embryo_id': emb_id,
                        'timepoint': timepoint,
                        'success': False
                    })
                    embryo_pbar.update(1)
                    continue

                # Save volume
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = session_dir / f"{emb_id}_embryo{emb_num:03d}_t{timepoint:04d}_{timestamp}.tif"
                tifffile.imwrite(filename, volume)
                print(f"  ✓ Saved: {filename.name}")

                # Process for hatching detection
                if detector and detector.should_check_embryo(emb_id, timepoint):
                    print(f"  Processing for hatching detection...")
                    b64_image, size, max_proj = process_volume_for_claude(volume)

                    # Save processed image
                    if HATCHING_DETECTION_CONFIG['save_processed_images']:
                        proc_filename = processed_dir / f"{emb_id}_t{timepoint:04d}_maxproj.png"
                        save_processed_image(max_proj, proc_filename)

                    # Add to history
                    image_history.add_image(emb_id, timepoint, b64_image, size)

                    # Run detection
                    recent_images = image_history.get_recent_images(emb_id)
                    detection_result = detector.detect_hatching_single_image(emb_id, timepoint, recent_images)

                    # Update status
                    detector.update_hatching_status(emb_id, detection_result)

                timepoint_results.append({
                    'embryo_id': emb_id,
                    'embryo_number': emb_num,
                    'timepoint': timepoint,
                    'success': True,
                    'filename': str(filename),
                    'shape': volume.shape
                })

                embryo_pbar.update(1)

            embryo_pbar.close()
            all_results.extend(timepoint_results)

            # Update timepoint progress
            timepoint_pbar.update(1)

            # Wait for next timepoint
            if timepoint < num_timepoints - 1:
                timepoint_duration = time.time() - timepoint_start_time
                wait_time = interval_minutes * 60 - timepoint_duration

                if wait_time > 0:
                    next_timepoint_time = datetime.now() + timedelta(seconds=wait_time)
                    print(f"\n{'─'*70}")
                    print(f"Waiting {wait_time/60:.1f} minutes...")
                    print(f"Next timepoint at: {next_timepoint_time.strftime('%H:%M:%S')}")
                    print(f"{'─'*70}")

                    wait_pbar = tqdm(
                        total=int(wait_time),
                        desc="  Waiting",
                        unit="s",
                        position=1,
                        leave=False,
                        colour='yellow'
                    )
                    for _ in range(int(wait_time)):
                        time.sleep(1)
                        wait_pbar.update(1)
                    wait_pbar.close()

                    time.sleep(wait_time - int(wait_time))
                else:
                    print(f"\n⚠ Warning: Acquisition took {timepoint_duration/60:.1f} min (longer than {interval_minutes} min)")

        timepoint_pbar.close()

        # Cleanup
        print(f"\n{'='*70}")
        print("CLEANUP")
        print(f"{'='*70}")
        core.setConfig("Laser", "ALL OFF")
        print(f"  ✓ Lasers OFF")

        # Save detection log
        if detector:
            log_file = session_dir / HATCHING_DETECTION_CONFIG['detection_log_file']
            detector.save_detection_log(log_file)

        # Summary
        print(f"\n{'='*70}")
        print("ACQUISITION COMPLETE")
        print(f"{'='*70}")

        total_duration = time.time() - session_start_time
        successful = sum(1 for r in all_results if r.get('success'))

        print(f"\n  Session duration: {total_duration/3600:.2f} hours")
        print(f"  Successful acquisitions: {successful}")
        print(f"  Output directory: {session_dir}")

        if detector:
            summary = detector.get_summary()
            print(f"\n  Hatching Summary:")
            print(f"    Hatched embryos: {summary['hatched_count']}/{summary['total_embryos']}")
            for eid, status in summary['embryo_status'].items():
                if status.get('hatched'):
                    tp = status['timepoint']
                    conf = status['confidence']
                    print(f"      ✓ {eid}: t{tp:04d} ({tp*2}min, {conf})")

        # Save acquisition log
        log_file = session_dir / "acquisition_log.json"
        with open(log_file, 'w') as f:
            json.dump({
                'timestamp': datetime.now().isoformat(),
                'session_duration_hours': total_duration / 3600.0,
                'num_embryos': num_embryos,
                'num_slices': num_slices,
                'interval_minutes': interval_minutes,
                'detection_config': HATCHING_DETECTION_CONFIG,
                'hatching_summary': detector.get_summary() if detector else None,
                'results': all_results
            }, f, indent=2)

        print(f"\n  ✓ Log saved: {log_file}")
        print(f"\n{'='*70}\n")

    except KeyboardInterrupt:
        print(f"\n\nInterrupted by user\n")
        if detector:
            log_file = session_dir / "interrupted_detection_log.json"
            detector.save_detection_log(log_file)
    except Exception as e:
        print(f"\n{'='*70}")
        print("ERROR")
        print(f"{'='*70}")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
