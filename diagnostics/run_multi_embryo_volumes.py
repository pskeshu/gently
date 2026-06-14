#!/usr/bin/env python3
"""
Run Multi-Embryo Volume Acquisition
===================================

Loads embryo database and acquires volumes for all calibrated embryos.

Usage:
    python run_multi_embryo_volumes.py
"""

import json
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import rpyc
import tifffile
from client import get_mmc
from tqdm import tqdm

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


def load_database():
    """Load embryo database."""
    if not DATABASE_FILE.exists():
        raise FileNotFoundError(
            f"Database not found: {DATABASE_FILE}\nRun multi_embryo_calibration.py first!"
        )

    with open(DATABASE_FILE) as f:
        return json.load(f)


def get_stage_position():
    """Get current XY stage position."""
    x = core.getXPosition(XY_STAGE_NAME)
    y = core.getYPosition(XY_STAGE_NAME)
    return (x, y)


def move_to_embryo(embryo_data):
    """
    Move stage to embryo's calibrated position.

    Parameters
    ----------
    embryo_data : dict
        Embryo information from database
    """
    target_x = embryo_data["stage_position_after_centering_um"]["x"]
    target_y = embryo_data["stage_position_after_centering_um"]["y"]

    print(f"  Moving to embryo position: ({target_x:.2f}, {target_y:.2f}) µm")

    core.setXYStageDevice(XY_STAGE_NAME)
    core.setXYPosition(float(target_x), float(target_y))
    core.waitForDevice(XY_STAGE_NAME)
    time.sleep(0.5)

    actual_pos = get_stage_position()
    print(f"  ✓ At position: ({actual_pos[0]:.2f}, {actual_pos[1]:.2f}) µm")


def configure_hardware_for_volume(calibration, num_slices):
    """
    Configure galvo, piezo, camera for hardware-triggered volume acquisition.

    Uses the calibration data for this embryo to set up synchronized scanning.

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

    # Stop any existing sequence acquisition (from previous embryo or calibration)
    if core.isSequenceRunning():
        print("    Stopping previous sequence...")
        core.stopSequenceAcquisition()
        time.sleep(0.5)

    # Clear any pending images
    core.clearCircularBuffer()

    # Reset SPIM state machine
    try:
        core.setProperty(GALVO_DEVICE, "SPIMState", "Idle")
        time.sleep(0.2)
    except Exception:
        pass

    # Extract calibration parameters
    slope = calibration["slope_um_per_deg"]
    offset = calibration["offset_um"]
    galvo_top = calibration.get("edge_top_deg", calibration["galvo_top_deg"])
    galvo_bottom = calibration.get("edge_bottom_deg", calibration["galvo_bottom_deg"])

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

    print(
        f"    Galvo: center={galvo_center:+.4f}°, amplitude=±{galvo_amplitude:.4f}°"
        f" (range: {galvo_range:.4f}°)"
    )
    print(
        f"    Piezo: center={piezo_center:.1f}µm, amplitude=±{piezo_amplitude:.1f}µm"
        f" (range: {piezo_range:.1f}µm)"
    )

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
    print("  ✓ Hardware configured for hardware-triggered acquisition")

    return {
        "galvo_center": galvo_center,
        "galvo_amplitude": galvo_amplitude,
        "piezo_center": piezo_center,
        "piezo_amplitude": piezo_amplitude,
    }


def acquire_volume_for_embryo(embryo_id, calibration, num_slices=50):
    """
    Acquire volume for one embryo using hardware-triggered SPIM acquisition.

    Parameters
    ----------
    embryo_id : str
        Embryo identifier
    calibration : dict
        Calibration data for this embryo
    num_slices : int
        Number of slices to acquire

    Returns
    -------
    np.ndarray
        Volume stack (slices, height, width)
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

    core.prepareSequenceAcquisition(CAMERA_NAME)
    time.sleep(0.1)

    core.startSequenceAcquisition(CAMERA_NAME, num_slices, 0, True)
    time.sleep(0.1)

    # Trigger SPIM state machine
    core.setProperty(GALVO_DEVICE, "SPIMState", "Running")
    print("  ✓ SPIM triggered")

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


def save_volume(volume, embryo_id, embryo_number, output_dir):
    """Save volume as TIFF file."""
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = output_dir / f"{embryo_id}_embryo{embryo_number:03d}_{timestamp}.tif"

    tifffile.imwrite(filename, volume)
    print(f"  ✓ Saved: {filename}")

    return filename


def main():
    """Main multi-embryo volume acquisition workflow."""
    print(f"{'=' * 70}")
    print("MULTI-EMBRYO VOLUME ACQUISITION")
    print(f"{'=' * 70}")

    try:
        # Load database
        print(f"\n{'=' * 70}")
        print("LOADING DATABASE")
        print(f"{'=' * 70}")
        database = load_database()

        embryos = database.get("embryos", {})
        num_embryos = len(embryos)

        print(f"  Database: {DATABASE_FILE}")
        print(f"  Found {num_embryos} embryo(s)")

        if num_embryos == 0:
            print("\n  ⚠ No embryos in database!")
            print("  Run multi_embryo_calibration.py first.")
            return

        # List embryos
        print("\n  Embryos:")
        for emb_id, emb_data in embryos.items():
            emb_num = emb_data.get("embryo_number", "?")
            pos = emb_data["stage_position_after_centering_um"]
            print(f"    {emb_id} (#{emb_num}): ({pos['x']:.1f}, {pos['y']:.1f}) µm")

        # Acquisition parameters
        print(f"\n{'=' * 70}")
        print("ACQUISITION PARAMETERS")
        print(f"{'=' * 70}")

        num_slices = int(input("  Number of slices per volume (default 50): ").strip() or "50")
        print(f"  ✓ Will acquire {num_slices} slices per embryo")

        # Timelapse parameters
        num_timepoints = int(
            input("  Number of timepoints (default 1 for single acquisition): ").strip() or "1"
        )
        interval_minutes = 0
        if num_timepoints > 1:
            interval_minutes = float(
                input("  Interval between timepoints in minutes (e.g., 2): ").strip() or "2"
            )
            total_duration_hours = (num_timepoints - 1) * interval_minutes / 60.0
            print(
                f"  ✓ Timelapse: {num_timepoints} timepoints every {interval_minutes} min"
                f" ({total_duration_hours:.1f} hours total)"
            )
        else:
            print("  ✓ Single acquisition (no timelapse)")

        # Create output directory
        session_dir = OUTPUT_DIR / datetime.now().strftime("%Y%m%d_%H%M%S")
        session_dir.mkdir(parents=True, exist_ok=True)
        print(f"  ✓ Output: {session_dir}")

        # Timelapse loop
        all_results = []
        session_start_time = time.time()

        # Progress bar for timepoints
        timepoint_pbar = tqdm(
            total=num_timepoints,
            desc="Timepoints",
            unit="tp",
            position=0,
            colour="green",
            bar_format=(
                "{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt}"
                " [{elapsed}<{remaining}, {rate_fmt}]"
            ),
        )

        for timepoint in range(num_timepoints):
            timepoint_start_time = time.time()
            elapsed_hours = (timepoint_start_time - session_start_time) / 3600.0

            # Update timepoint progress bar
            timepoint_pbar.set_description(
                f"Timepoint {timepoint + 1}/{num_timepoints} (Elapsed: {elapsed_hours:.1f}h)"
            )

            print(f"\n{'=' * 70}")
            print(f"TIMEPOINT {timepoint + 1}/{num_timepoints}")
            if num_timepoints > 1:
                print(f"Elapsed: {elapsed_hours:.2f} hours")
            print(f"{'=' * 70}")

            # Acquire volume for each embryo
            timepoint_results = []

            # Progress bar for embryos within this timepoint
            embryo_pbar = tqdm(
                total=num_embryos,
                desc=f"  Embryos (t{timepoint:04d})",
                unit="embryo",
                position=1,
                leave=False,
                colour="cyan",
            )

            for idx, (emb_id, emb_data) in enumerate(embryos.items(), 1):
                emb_num = emb_data.get("embryo_number", idx)

                embryo_pbar.set_description(f"  Embryo {emb_num} (t{timepoint:04d})")

                print(f"\n[Embryo {idx}/{num_embryos}] {emb_id} (Embryo #{emb_num})")
                print(f"{'─' * 70}")

                # Move to embryo
                move_to_embryo(emb_data)

                # Configure hardware
                calibration = emb_data["calibration"]
                configure_hardware_for_volume(calibration, num_slices)

                # Acquire volume
                volume = acquire_volume_for_embryo(emb_id, calibration, num_slices)

                if volume is None:
                    print("  ✗ Failed to acquire volume")
                    timepoint_results.append(
                        {"embryo_id": emb_id, "timepoint": timepoint, "success": False}
                    )
                    embryo_pbar.update(1)
                    continue

                # Save volume with timepoint in filename
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = (
                    session_dir / f"{emb_id}_embryo{emb_num:03d}_t{timepoint:04d}_{timestamp}.tif"
                )
                tifffile.imwrite(filename, volume)
                print(f"  ✓ Saved: {filename.name}")

                timepoint_results.append(
                    {
                        "embryo_id": emb_id,
                        "embryo_number": emb_num,
                        "timepoint": timepoint,
                        "success": True,
                        "filename": str(filename),
                        "shape": volume.shape,
                    }
                )

                print(f"  ✓ Complete: {volume.shape}")
                embryo_pbar.update(1)

            embryo_pbar.close()
            all_results.extend(timepoint_results)

            # Update timepoint progress bar
            timepoint_pbar.update(1)

            # Wait for next timepoint
            if timepoint < num_timepoints - 1:
                timepoint_duration = time.time() - timepoint_start_time
                wait_time = interval_minutes * 60 - timepoint_duration

                if wait_time > 0:
                    next_timepoint_time = datetime.now() + timedelta(seconds=wait_time)
                    print(f"\n{'─' * 70}")
                    print(f"Waiting {wait_time / 60:.1f} minutes until next timepoint...")
                    print(f"Next timepoint at: {next_timepoint_time.strftime('%H:%M:%S')}")
                    print(f"{'─' * 70}")

                    # Progress bar for waiting
                    wait_pbar = tqdm(
                        total=int(wait_time),
                        desc="  Waiting",
                        unit="s",
                        position=1,
                        leave=False,
                        colour="yellow",
                    )
                    for _ in range(int(wait_time)):
                        time.sleep(1)
                        wait_pbar.update(1)
                    wait_pbar.close()

                    # Sleep remaining fractional seconds
                    time.sleep(wait_time - int(wait_time))
                else:
                    print(f"\n{'─' * 70}")
                    print(
                        f"⚠ Warning: Acquisition took {timepoint_duration / 60:.1f} min"
                        f" (longer than {interval_minutes} min interval)"
                    )
                    print("Proceeding immediately to next timepoint...")
                    print(f"{'─' * 70}")

        timepoint_pbar.close()

        # Cleanup
        print(f"\n{'=' * 70}")
        print("CLEANUP")
        print(f"{'=' * 70}")
        core.setConfig("Laser", "ALL OFF")
        print("  ✓ Lasers OFF")

        # Summary
        print(f"\n{'=' * 70}")
        print("ACQUISITION COMPLETE")
        print(f"{'=' * 70}")

        total_duration = time.time() - session_start_time
        successful = sum(1 for r in all_results if r["success"])
        total_acquisitions = num_embryos * num_timepoints

        print(f"\n  Session duration: {total_duration / 3600:.2f} hours")
        print(f"  Timepoints: {num_timepoints}")
        print(f"  Embryos per timepoint: {num_embryos}")
        print(f"  Successful acquisitions: {successful}/{total_acquisitions}")
        print(f"  Output directory: {session_dir}")

        print("\n  Results:")
        for result in all_results:
            if result["success"]:
                t = result.get("timepoint", 0)
                print(
                    f"    ✓ {result['embryo_id']} t{t:04d}: {result['shape']}"
                    f" → {Path(result['filename']).name}"
                )
            else:
                t = result.get("timepoint", 0)
                print(f"    ✗ {result['embryo_id']} t{t:04d}: Failed")

        # Save acquisition log
        log_file = session_dir / "acquisition_log.json"
        with open(log_file, "w") as f:
            json.dump(
                {
                    "timestamp": datetime.now().isoformat(),
                    "session_duration_hours": total_duration / 3600.0,
                    "num_embryos": num_embryos,
                    "num_slices": num_slices,
                    "num_timepoints": num_timepoints,
                    "interval_minutes": interval_minutes,
                    "total_acquisitions": total_acquisitions,
                    "successful_acquisitions": successful,
                    "results": all_results,
                },
                f,
                indent=2,
            )

        print(f"\n  ✓ Log saved: {log_file}")
        print(f"\n{'=' * 70}\n")

    except KeyboardInterrupt:
        print("\n\nInterrupted\n")
    except Exception as e:
        print(f"\n{'=' * 70}")
        print("ERROR")
        print(f"{'=' * 70}")
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
