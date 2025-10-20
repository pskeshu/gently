#!/usr/bin/env python3
"""
Hardware-Triggered Piezo-Galvo Synchronized Scan

This script sets up a hardware-triggered acquisition where the Tiger controller
orchestrates synchronized camera triggering, piezo Z-movement, and galvo positioning.

The Tiger controller's firmware handles all timing and synchronization:
- Generates camera trigger pulses
- Moves piezo through Z range
- Adjusts galvo Y offset to keep light sheet aligned
- All synchronized to internal clock

Usage:
    python hardware_triggered_scan.py
"""

import time
import json
from pathlib import Path
from client import get_mmc
import numpy as np

# Device configuration
core = get_mmc()
CAMERA_NAME = "HamCam1"
GALVO_DEVICE = "Scanner:AB:33"
PIEZO_DEVICE = "PiezoStage:P:34"

# Load calibration
CALIBRATION_FILE = Path("piezo_galvo_calibration.json")

# Scan parameters
# CRITICAL: Galvo Y offset gets reset to 0° during SPIM, so we must center piezo
# at the position that corresponds to galvo 0° according to calibration
# From calibration: piezo_position = slope * galvo_angle + offset
# At galvo 0°: piezo = 100.833 * 0 + 4.25 = 4.25 µm
PIEZO_CENTER_UM = 4.25        # Piezo center (µm) - matches galvo 0°
PIEZO_AMPLITUDE_UM = 32.3     # Piezo half-range (µm) - total scan = 64.6 µm

# Galvo parameters will be calculated from piezo using calibration formula:
# galvo_angle (°) = (piezo_position (µm) - offset) / slope

NUM_SLICES = 30               # Number of Z slices
SLICE_PERIOD_MS = 50.0        # Time per slice (ms) - camera exposure + overhead

# Camera exposure
CAMERA_EXPOSURE_MS = 10.0


def load_calibration():
    """Load piezo-galvo calibration from JSON file."""
    print(f"Loading calibration from: {CALIBRATION_FILE}")

    if not CALIBRATION_FILE.exists():
        raise FileNotFoundError(
            f"Calibration file not found: {CALIBRATION_FILE}\n"
            f"Please run calibrate_piezo_galvo.py first!"
        )

    with open(CALIBRATION_FILE, 'r') as f:
        cal = json.load(f)

    print(f"  Slope: {cal['slope_um_per_deg']:.3f} µm/°")
    print(f"  Offset: {cal['offset_um']:.3f} µm")

    return cal


def calculate_galvo_params(calibration):
    """Calculate galvo center and amplitude from piezo parameters using calibration.

    This follows the ASI diSPIM plugin formula (ControllerUtils.java lines 390-402):
    galvo_angle (°) = (piezo_position (µm) - offset) / slope
    """
    slope = calibration['slope_um_per_deg']
    offset = calibration['offset_um']

    # Calculate galvo parameters from piezo using calibration formula
    # galvo_center = (piezo_center - offset) / slope
    galvo_center = (PIEZO_CENTER_UM - offset) / slope

    # galvo_amplitude = piezo_amplitude / slope
    galvo_amplitude = PIEZO_AMPLITUDE_UM / slope

    print(f"\n  Calculated galvo parameters from piezo:")
    print(f"    Galvo center: {galvo_center:.4f}°")
    print(f"    Galvo amplitude: {galvo_amplitude:.4f}°")
    print(f"    Galvo range: {galvo_center - galvo_amplitude:.4f}° to {galvo_center + galvo_amplitude:.4f}°")

    return galvo_center, galvo_amplitude


def configure_camera_for_hardware_trigger():
    """Configure camera for external (hardware) triggering."""
    print(f"\nConfiguring camera for hardware trigger: {CAMERA_NAME}")

    core.setCameraDevice(CAMERA_NAME)

    # Set camera ROI
    roi_x = 128
    roi_y = 896
    roi_width = 2048
    roi_height = 512

    print(f"  Setting camera ROI: X={roi_x}, Y={roi_y}, W={roi_width}, H={roi_height}")
    core.setROI(CAMERA_NAME, roi_x, roi_y, roi_width, roi_height)

    # Configure for EXTERNAL trigger (hardware TTL)
    print(f"  Setting trigger source to EXTERNAL")
    core.setProperty(CAMERA_NAME, "TRIGGER SOURCE", "EXTERNAL")

    # CRITICAL: Must use PROGRESSIVE mode for hardware-triggered SPIM!
    # AREA mode causes sequence to stop immediately with external triggers
    print(f"  Setting sensor mode to PROGRESSIVE")
    core.setProperty(CAMERA_NAME, "SENSOR MODE", "PROGRESSIVE")
    core.setProperty(CAMERA_NAME, "TRIGGER ACTIVE", "EDGE")

    # Set exposure
    core.setExposure(CAMERA_NAME, CAMERA_EXPOSURE_MS)

    time.sleep(0.1)

    # Verify settings
    trigger_source = core.getProperty(CAMERA_NAME, "TRIGGER SOURCE")
    sensor_mode = core.getProperty(CAMERA_NAME, "SENSOR MODE")
    exposure = core.getExposure(CAMERA_NAME)

    print(f"  ✓ TRIGGER SOURCE: {trigger_source}")
    print(f"  ✓ SENSOR MODE: {sensor_mode}")
    print(f"  ✓ Exposure: {exposure} ms")


def configure_galvo_for_spim(galvo_center, galvo_amplitude):
    """Configure galvo for SPIM (hardware-triggered) mode."""
    print(f"\nConfiguring galvo for SPIM mode...")

    # Ensure SPIM is idle first
    core.setProperty(GALVO_DEVICE, "SPIMState", "Idle")
    time.sleep(0.2)

    # CRITICAL: Set laser output mode to enable TTL outputs
    core.setProperty(GALVO_DEVICE, "LaserOutputMode", "shutter + side")
    laser_mode = core.getProperty(GALVO_DEVICE, "LaserOutputMode")
    print(f"  LaserOutputMode: {laser_mode}")

    # CRITICAL: Disable BeamEnabled - SPIM state machine will control beam
    # (from ASI plugin ControllerUtils.java lines 104-115)
    core.setProperty(GALVO_DEVICE, "BeamEnabled", "No")
    print(f"  BeamEnabled: No (SPIM state machine will control)")

    # Configure X-axis for light sheet width (scanning)
    # Set amplitude/offset once - SPIM state machine will use these values internally
    # Larger amplitude = wider light sheet coverage
    # Typical range: 1.0° to 4.0°
    LIGHTSHEET_WIDTH_DEG = 4.0  # Sheet width
    LIGHTSHEET_OFFSET_DEG = 0.0  # Sheet center (was -0.5)

    core.setProperty(GALVO_DEVICE, "SingleAxisXAmplitude(deg)", LIGHTSHEET_WIDTH_DEG)
    core.setProperty(GALVO_DEVICE, "SingleAxisXOffset(deg)", LIGHTSHEET_OFFSET_DEG)
    core.setProperty(GALVO_DEVICE, "SingleAxisXPattern", "1 - Triangle")

    # CRITICAL: Disable SingleAxisXMode - SPIM state machine will control scanning
    # (from ASI plugin ControllerUtils.java lines 108-115)
    core.setProperty(GALVO_DEVICE, "SingleAxisXMode", "0 - Disabled")

    print(f"  Light sheet width: {LIGHTSHEET_WIDTH_DEG}° amplitude, offset: {LIGHTSHEET_OFFSET_DEG}°")
    print(f"  SingleAxisXMode: Disabled (SPIM state machine will control)")

    # Configure Y-axis for light sheet positioning (synchronized with piezo via calibration)
    print(f"  Setting galvo Y amplitude: {galvo_amplitude:.4f}°")
    print(f"  Setting galvo Y offset: {galvo_center:.4f}°")

    core.setProperty(GALVO_DEVICE, "SingleAxisYAmplitude(deg)", float(galvo_amplitude))
    core.setProperty(GALVO_DEVICE, "SingleAxisYOffset(deg)", float(galvo_center))
    core.setProperty(GALVO_DEVICE, "SingleAxisYPattern", "1 - Triangle")
    # NOTE: Don't enable SingleAxisYMode - let SPIM state machine control it
    # core.setProperty(GALVO_DEVICE, "SingleAxisYMode", "3 - Enabled with axes synced")

    # SPIM timing parameters
    print(f"  Setting SPIM timing parameters...")
    core.setProperty(GALVO_DEVICE, "SPIMDelayBeforeScan(ms)", 6.75)  # Delay before scan (from ASI plugin defaults)
    core.setProperty(GALVO_DEVICE, "SPIMScanDuration(ms)", float(SLICE_PERIOD_MS))
    core.setProperty(GALVO_DEVICE, "SPIMCameraDuration(ms)", float(CAMERA_EXPOSURE_MS))
    core.setProperty(GALVO_DEVICE, "SPIMDelayBeforeCamera(ms)", 0.5)  # Short delay before camera trigger
    core.setProperty(GALVO_DEVICE, "SPIMDelayBeforeLaser(ms)", 0.0)   # Laser on immediately
    core.setProperty(GALVO_DEVICE, "SPIMLaserDuration(ms)", float(CAMERA_EXPOSURE_MS + 1.0))

    # SPIM parameters for galvo
    print(f"  Setting SPIM scan parameters...")
    core.setProperty(GALVO_DEVICE, "SPIMNumSlices", NUM_SLICES)
    core.setProperty(GALVO_DEVICE, "SPIMNumSlicesPerPiezo", 1)  # 1 slice per piezo position
    core.setProperty(GALVO_DEVICE, "SPIMNumSides", 1)            # Single side acquisition
    core.setProperty(GALVO_DEVICE, "SPIMFirstSide", "A")         # Path A

    # Verify critical property
    camera_duration_check = float(core.getProperty(GALVO_DEVICE, 'SPIMCameraDuration(ms)'))
    if camera_duration_check <= 0:
        raise Exception("SPIMCameraDuration(ms) is 0 - triggers will NOT be generated!")

    print(f"  ✓ All timing properties configured correctly")


def configure_piezo_for_spim():
    """Configure piezo for SPIM (hardware-triggered) mode."""
    print(f"\nConfiguring piezo for SPIM mode...")

    core.setFocusDevice(PIEZO_DEVICE)

    print(f"  Piezo center: {PIEZO_CENTER_UM:.2f} µm")
    print(f"  Piezo amplitude: {PIEZO_AMPLITUDE_UM:.2f} µm")
    print(f"  Piezo range: {PIEZO_CENTER_UM - PIEZO_AMPLITUDE_UM:.2f} to {PIEZO_CENTER_UM + PIEZO_AMPLITUDE_UM:.2f} µm")

    # Set piezo single-axis amplitude and offset
    core.setProperty(PIEZO_DEVICE, "SingleAxisAmplitude(um)", float(PIEZO_AMPLITUDE_UM))
    core.setProperty(PIEZO_DEVICE, "SingleAxisOffset(um)", float(PIEZO_CENTER_UM))

    # CRITICAL: Set piezo to use TRIANGLE pattern to match galvo Y
    # This ensures they move in sync (both ramp up together, both ramp down together)
    core.setProperty(PIEZO_DEVICE, "SingleAxisPattern", "1 - Triangle")
    print(f"  Piezo pattern: Triangle (matches galvo Y)")

    # SPIM parameters for piezo
    core.setProperty(PIEZO_DEVICE, "SPIMNumSlices", NUM_SLICES)

    # CRITICAL: ARM the piezo (enables hardware trigger response)
    print(f"  Arming piezo for hardware trigger...")
    core.setProperty(PIEZO_DEVICE, "SPIMState", "Armed")

    time.sleep(0.3)

    spim_state = core.getProperty(PIEZO_DEVICE, "SPIMState")
    print(f"  ✓ Piezo SPIMState: {spim_state}")


def start_hardware_triggered_acquisition():
    """Start hardware-triggered acquisition by setting galvo to RUNNING."""
    print(f"\n" + "="*70)
    print("STARTING HARDWARE-TRIGGERED ACQUISITION")
    print("="*70)

    # Stop any running sequence
    if core.isSequenceRunning():
        print("  Stopping existing sequence...")
        core.stopSequenceAcquisition()
        time.sleep(0.5)

    # Clear and configure circular buffer
    print("  Configuring circular buffer...")
    core.clearCircularBuffer()

    # Check buffer capacity
    buffer_capacity = core.getBufferTotalCapacity()
    print(f"  Current buffer capacity: {buffer_capacity}")

    # Set sufficient buffer memory
    if buffer_capacity < NUM_SLICES:
        print(f"  Setting buffer memory footprint to 512 MB...")
        core.setCircularBufferMemoryFootprint(512)
        time.sleep(0.1)
        buffer_capacity = core.getBufferTotalCapacity()
        print(f"  New buffer capacity: {buffer_capacity}")

    # CRITICAL: Prepare sequence acquisition first (allocates camera buffer)
    print("  Preparing sequence acquisition...")
    core.prepareSequenceAcquisition(CAMERA_NAME)
    time.sleep(0.1)

    # Start sequence acquisition (camera enters WAITING state for external triggers)
    print(f"  Starting sequence acquisition...")
    core.startSequenceAcquisition(CAMERA_NAME, NUM_SLICES, 0, True)
    time.sleep(0.1)

    # Verify sequence started
    seq_running = core.isSequenceRunning(CAMERA_NAME)
    print(f"  Sequence running: {seq_running}")

    if not seq_running:
        raise Exception("Camera sequence failed to start!")

    # MASTER TRIGGER: Set galvo SPIM state to RUNNING
    # This tells the Tiger controller to start the synchronized acquisition
    print(f"\n  Sending master trigger (SPIMState = Running)...")
    core.setProperty(GALVO_DEVICE, "SPIMState", "Running")

    print(f"  ✓ Hardware acquisition started!")
    print(f"\n  Tiger controller is now:")
    print(f"    - Generating camera trigger pulses")
    print(f"    - Moving piezo through Z range")
    print(f"    - Adjusting galvo Y to track piezo")
    print(f"    - All synchronized to internal firmware clock")


def wait_for_acquisition():
    """Wait for hardware acquisition to complete and collect images."""
    print(f"\n  Waiting for {NUM_SLICES} images...")

    images = []
    timeout_s = NUM_SLICES * SLICE_PERIOD_MS / 1000.0 * 2  # 2x expected time
    start_time = time.time()

    while core.getRemainingImageCount() > 0 or core.isSequenceRunning():
        if core.getRemainingImageCount() > 0:
            # Get image from circular buffer
            img = core.popNextImage()

            # Handle remote core (rpyc)
            try:
                import rpyc
                img = rpyc.classic.obtain(img)
            except (ImportError, AttributeError):
                pass

            images.append(img)

            if len(images) % 5 == 0:
                print(f"    Received {len(images)}/{NUM_SLICES} images...")

        # Check timeout
        if time.time() - start_time > timeout_s:
            print(f"\n  WARNING: Timeout waiting for images!")
            break

        time.sleep(0.01)

    # Stop sequence acquisition
    if core.isSequenceRunning():
        core.stopSequenceAcquisition()

    print(f"\n  ✓ Acquisition complete! Received {len(images)} images")

    return np.array(images)


def save_tif_stack(volume, galvo_center, galvo_amplitude):
    """Save volume as a TIFF stack."""
    import tifffile
    from datetime import datetime

    output_dir = Path("scan_images")
    output_dir.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = output_dir / f"hardware_triggered_scan_{timestamp}.tif"

    print(f"\nSaving TIFF stack...")
    print(f"  Filename: {filename.name}")
    print(f"  Volume shape: {volume.shape} (Z, Y, X)")
    print(f"  Data type: {volume.dtype}")

    # Calculate positions for each slice
    piezo_positions = np.linspace(
        PIEZO_CENTER_UM - PIEZO_AMPLITUDE_UM,
        PIEZO_CENTER_UM + PIEZO_AMPLITUDE_UM,
        NUM_SLICES
    )

    tifffile.imwrite(
        filename,
        volume,
        metadata={
            'axes': 'ZYX',
            'piezo_center_um': float(PIEZO_CENTER_UM),
            'piezo_amplitude_um': float(PIEZO_AMPLITUDE_UM),
            'piezo_start_um': float(piezo_positions[0]),
            'piezo_end_um': float(piezo_positions[-1]),
            'galvo_center_deg': float(galvo_center),
            'galvo_amplitude_deg': float(galvo_amplitude),
            'num_slices': NUM_SLICES,
            'slice_period_ms': SLICE_PERIOD_MS,
            'hardware_triggered': True
        }
    )

    print(f"  ✓ Saved TIFF stack: {filename}")
    return filename


def cleanup():
    """Reset devices to safe state."""
    print("\n" + "="*70)
    print("CLEANUP")
    print("="*70)

    try:
        # Set SPIM states back to IDLE
        core.setProperty(GALVO_DEVICE, "SPIMState", "Idle")
        core.setProperty(PIEZO_DEVICE, "SPIMState", "Idle")
        print("  ✓ SPIM states set to Idle")
    except Exception as e:
        print(f"  Could not reset SPIM states: {e}")

    try:
        # Reset galvo Y to center
        core.setProperty(GALVO_DEVICE, "SingleAxisYOffset(deg)", 0.0)
        print("  ✓ Galvo Y reset to center")
    except Exception as e:
        print(f"  Could not reset galvo: {e}")

    try:
        # Set camera back to internal trigger for live mode
        core.setProperty(CAMERA_NAME, "TRIGGER SOURCE", "INTERNAL")
        print("  ✓ Camera trigger reset to INTERNAL")
    except Exception as e:
        print(f"  Could not reset camera trigger: {e}")

    try:
        # Lasers off
        core.setConfig("Laser", "ALL OFF")
        print("  ✓ Lasers OFF")
    except Exception as e:
        print(f"  Could not turn off lasers: {e}")


def main():
    """Main hardware-triggered acquisition workflow."""
    print("="*70)
    print("HARDWARE-TRIGGERED PIEZO-GALVO SYNCHRONIZED SCAN")
    print("="*70)

    try:
        # Load calibration
        print("\n[1/8] Loading calibration...")
        calibration = load_calibration()
        galvo_center, galvo_amplitude = calculate_galvo_params(calibration)

        print(f"\n  Piezo scan parameters:")
        print(f"    Center: {PIEZO_CENTER_UM:.2f} µm")
        print(f"    Amplitude: {PIEZO_AMPLITUDE_UM:.2f} µm")
        print(f"    Range: {PIEZO_CENTER_UM - PIEZO_AMPLITUDE_UM:.2f} to {PIEZO_CENTER_UM + PIEZO_AMPLITUDE_UM:.2f} µm")

        # System startup
        print("\n[2/8] Applying System Startup configuration...")
        core.setConfig("System", "Startup")
        core.waitForConfig("System", "Startup")
        print("  ✓ System configured")

        # Lasers on
        print("\n[3/8] Turning on lasers...")
        core.setConfig("Laser", "488 and 561")
        core.waitForConfig("Laser", "488 and 561")
        print("  ✓ Lasers ON")

        # Configure camera for hardware trigger
        print("\n[4/8] Configuring camera...")
        configure_camera_for_hardware_trigger()

        # Configure galvo for SPIM
        print("\n[5/8] Configuring galvo for SPIM...")
        configure_galvo_for_spim(galvo_center, galvo_amplitude)

        # Configure piezo for SPIM
        print("\n[6/8] Configuring piezo for SPIM...")
        configure_piezo_for_spim()

        # Start hardware-triggered acquisition
        print("\n[7/8] Starting hardware-triggered acquisition...")
        start_hardware_triggered_acquisition()

        # Wait and collect images
        volume = wait_for_acquisition()

        # Save TIFF stack
        print("\n[8/8] Saving results...")
        if len(volume) > 0:
            save_tif_stack(volume, galvo_center, galvo_amplitude)
            print("\n✓ Done!")
            print("\nHardware-triggered acquisition completed successfully.")
            print("All synchronization was handled by the Tiger controller firmware.")
        else:
            print("\n✗ No images captured!")
            print("Check that:")
            print("  - Camera is properly connected")
            print("  - Tiger controller cables are connected")
            print("  - SPIM states were properly armed")

    except Exception as e:
        print(f"\n{'='*70}")
        print("ERROR DURING ACQUISITION")
        print("="*70)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        cleanup()


if __name__ == "__main__":
    main()
