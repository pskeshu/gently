#!/usr/bin/env python3
"""
ASI diSPIM Hardware-Triggered Volume Acquisition - Reference Implementation

This script demonstrates the CORRECT sequence for hardware-triggered SPIM acquisition
based on analysis of the Micro-Manager ASI diSPIM plugin codebase.

KEY INSIGHT: The ASI Tiger controller contains a SPIM state machine that generates
synchronized TTL pulses to trigger the camera, galvo scanning, and laser firing.
You must configure the SPIM timing properties (not just SingleAxis properties) to
enable TTL pulse generation.

Hardware Architecture:
=====================
1. Camera receives TTL trigger pulse from Tiger controller
2. Tiger SPIM state machine orchestrates all timing
3. Per-slice sequence:
   - Y-axis galvo/piezo moves to slice position
   - Small settling delay
   - Camera trigger pulse HIGH (duration = SPIMCameraDuration)
   - Camera starts exposure on rising edge
   - X-axis galvo scans (light sheet)
   - Laser fires (synchronized with camera)
   - Camera trigger pulse LOW
   - Camera readout (during move to next slice)

Critical Configuration (COMPLETE LIST):
=======================================
1. Camera Properties (Hamamatsu):
   - TRIGGER SOURCE = "EXTERNAL" (wait for external TTL)
   - SENSOR MODE = "PROGRESSIVE" (NOT "AREA" - light sheet rolling shutter mode!)
   - TRIGGER ACTIVE = "EDGE" (trigger on rising edge - property name has SPACE!)

2. SPIM State Machine Timing:
   - SPIMNumSlices, SPIMNumSides
   - SPIMScanDuration, SPIMCameraDuration, SPIMLaserDuration
   - SPIMDelayBeforeScan, SPIMDelayBeforeCamera

3. Sequence Acquisition:
   - Must call prepareSequenceAcquisition() BEFORE startSequenceAcquisition()
   - Without this, you get "index was 0 count was 0" error

4. State Flow:
   - Idle -> Armed -> Running -> Idle

CRITICAL FIXES FOR "index was 0 count was 0" ERROR:
===================================================
This error occurs when the camera sequence buffer is not properly initialized.
Three things were missing:

1. SENSOR MODE must be "PROGRESSIVE" (not "AREA")
   - AREA mode = standard full-frame readout
   - PROGRESSIVE mode = light sheet rolling shutter synchronized with galvo scan
   - Using AREA mode prevents proper external triggering!

2. TRIGGER ACTIVE property must be set to "EDGE"
   - Property name has a SPACE: "TRIGGER ACTIVE" not "TriggerActive"
   - This explicitly configures edge triggering mode

3. prepareSequenceAcquisition() must be called first
   - This allocates the camera's internal sequence buffer
   - Without this, startSequenceAcquisition() fails silently
   - Based on pycromanager engine.py:629

Based on analysis of:
- mmCoreAndDevices/DeviceAdapters/ASITiger/ASIScanner.cpp
- plugins/ASIdiSPIM/src/main/java/org/micromanager/asidispim/utils/ControllerUtils.java
- plugins/ASIdiSPIM/src/main/java/org/micromanager/asidispim/data/Cameras.java (lines 228-260)
- pycromanager/pycromanager/acquisition/acq_eng_py/internal/engine.py (line 629)
"""

import time

import numpy as np
from client import get_mmc


def configure_camera_for_hardware_trigger(core, camera_name, exposure_ms):
    """
    Configure Hamamatsu camera for external edge triggering in light sheet mode.

    CRITICAL FINDINGS:
    - Must use SENSOR MODE = "PROGRESSIVE" (not "AREA") for light sheet acquisition
    - Must set TriggerActive = "EDGE" explicitly (property name is case-sensitive!)
    - Property order matters: set trigger properties BEFORE exposure

    Args:
        core: Micro-Manager core instance
        camera_name: Name of camera device
        exposure_ms: Desired exposure time in milliseconds

    Based on:
    - Cameras.java:228-260 (ASI diSPIM plugin)
    - Investigation of "index was 0 count was 0" error
    """
    print(f"Configuring {camera_name} for hardware triggering...")

    core.setCameraDevice(camera_name)

    # CRITICAL: Set camera properties in correct order
    # These MUST be set BEFORE exposure for proper initialization
    core.setProperty(camera_name, "TRIGGER SOURCE", "EXTERNAL")

    # CRITICAL FIX #1: Use PROGRESSIVE for light sheet mode
    # "AREA" = Standard sensor mode (read entire frame at once)
    # "PROGRESSIVE" = Light sheet mode (rolling shutter synchronized with scan)
    # Using AREA mode causes "index was 0 count was 0" error!
    core.setProperty(camera_name, "SENSOR MODE", "PROGRESSIVE")

    # CRITICAL FIX #2: Property name is "TRIGGER ACTIVE" (with SPACE, not camelCase!)
    # This property MUST be set for external triggering to work
    core.setProperty(camera_name, "TRIGGER ACTIVE", "EDGE")

    # Set exposure AFTER trigger properties
    core.setExposure(camera_name, exposure_ms)

    # Verify settings
    trigger_source = core.getProperty(camera_name, "TRIGGER SOURCE")
    sensor_mode = core.getProperty(camera_name, "SENSOR MODE")
    trigger_active = core.getProperty(camera_name, "TRIGGER ACTIVE")
    actual_exposure = core.getExposure(camera_name)

    print(f"  Exposure: {actual_exposure} ms")
    print(f"  TRIGGER SOURCE: {trigger_source}")
    print(f"  SENSOR MODE: {sensor_mode}")
    print(f"  TRIGGER ACTIVE: {trigger_active}")

    # Verify critical properties
    if trigger_source != "EXTERNAL":
        raise Exception(f"Failed to set TRIGGER SOURCE to EXTERNAL (got: {trigger_source})")
    if sensor_mode != "PROGRESSIVE":
        raise Exception(f"Failed to set SENSOR MODE to PROGRESSIVE (got: {sensor_mode})")
    if trigger_active != "EDGE":
        raise Exception(f"Failed to set TRIGGER ACTIVE to EDGE (got: {trigger_active})")


def configure_spim_scanner(
    core,
    scanner_name,
    num_slices,
    slice_step_um,
    scan_duration_ms,
    camera_duration_ms,
    laser_duration_ms,
):
    """
    Configure ASI Tiger scanner for SPIM state machine operation.

    This sets up both the galvo movement (SingleAxis properties) and the
    SPIM state machine timing (SPIM properties that control TTL pulse generation).

    Args:
        core: Micro-Manager core instance
        scanner_name: Name of scanner device (e.g., "Scanner:AB:33")
        num_slices: Number of slices in volume
        slice_step_um: Distance between slices in microns
        scan_duration_ms: Total time per slice (must be >= exposure + readout)
        camera_duration_ms: Duration of camera trigger TTL pulse HIGH
        laser_duration_ms: Duration of laser TTL pulse HIGH

    Based on ASIScanner.cpp and ControllerUtils.java:103-223
    """
    print(f"Configuring SPIM scanner {scanner_name}...")

    # Reset SPIM state machine
    core.setProperty(scanner_name, "SPIMState", "Idle")
    time.sleep(0.2)

    # -------------------------------------------------------------------------
    # Part 1: Configure SingleAxis properties (galvo movement)
    # -------------------------------------------------------------------------

    # X-axis: Fast light sheet scanning
    # Triangle pattern creates smooth scanning motion during exposure
    core.setProperty(scanner_name, "SingleAxisXAmplitude(deg)", 2.0)
    core.setProperty(scanner_name, "SingleAxisXOffset(deg)", 0.0)
    core.setProperty(scanner_name, "SingleAxisXPattern", "1 - Triangle")
    core.setProperty(scanner_name, "SingleAxisXMode", "3 - Enabled with axes synced")

    # Y-axis: Slow slice stepping
    # Amplitude depends on: num_slices, slice_step, and calibration
    # Typical calibration: 100 deg/mm (check your system's calibration)
    calibration_slope = 100.0  # deg/mm - ADJUST FOR YOUR SYSTEM
    total_scan_distance_um = (num_slices - 1) * slice_step_um
    y_amplitude = total_scan_distance_um / 1000.0 / calibration_slope  # Convert um to mm to deg

    core.setProperty(scanner_name, "SingleAxisYAmplitude(deg)", y_amplitude)
    core.setProperty(scanner_name, "SingleAxisYOffset(deg)", 0.0)
    core.setProperty(scanner_name, "SingleAxisYPattern", "1 - Triangle")
    core.setProperty(scanner_name, "SingleAxisYMode", "3 - Enabled with axes synced")

    print("  X-axis (light sheet): Amplitude=2.0°, Pattern=Triangle, Mode=Synced")
    print(f"  Y-axis (slice step): Amplitude={y_amplitude:.4f}°, Pattern=Triangle, Mode=Synced")
    print(f"    (Calculated for {num_slices} slices × {slice_step_um} μm steps)")

    # -------------------------------------------------------------------------
    # Part 2: Configure SPIM State Machine Timing (CRITICAL FOR TRIGGERING!)
    # -------------------------------------------------------------------------

    # These properties tell the Tiger controller's SPIM state machine:
    # - How many slices to acquire
    # - How long each slice takes
    # - When to send TTL trigger pulses
    # - How long to keep triggers HIGH

    core.setProperty(scanner_name, "SPIMNumSlices", num_slices)
    core.setProperty(scanner_name, "SPIMNumSides", 1)  # Single-view acquisition

    # Timing parameters (these control TTL pulse generation):
    core.setProperty(scanner_name, "SPIMScanDuration(ms)", scan_duration_ms)
    core.setProperty(scanner_name, "SPIMCameraDuration(ms)", camera_duration_ms)
    core.setProperty(scanner_name, "SPIMLaserDuration(ms)", laser_duration_ms)

    # Delays (for vibration settling and timing fine-tuning):
    core.setProperty(scanner_name, "SPIMDelayBeforeScan(ms)", 0.0)
    core.setProperty(scanner_name, "SPIMDelayBeforeCamera(ms)", 0.5)

    print("  SPIM State Machine:")
    print(f"    NumSlices: {num_slices}")
    print(f"    ScanDuration: {scan_duration_ms} ms (total time per slice)")
    print(f"    CameraDuration: {camera_duration_ms} ms (TTL trigger pulse width)")
    print(f"    LaserDuration: {laser_duration_ms} ms (laser on time)")

    # Verify critical timing relationships
    if camera_duration_ms > scan_duration_ms:
        raise Exception(
            f"CameraDuration ({camera_duration_ms}ms) must be <="
            f" ScanDuration ({scan_duration_ms}ms)"
        )

    if laser_duration_ms > camera_duration_ms:
        raise Exception(
            f"LaserDuration ({laser_duration_ms}ms) must be <="
            f" CameraDuration ({camera_duration_ms}ms)"
        )


def arm_spim_state_machine(core, scanner_name):
    """
    Arm the SPIM state machine.

    The Armed state prepares the controller to start generating TTL pulses.
    You must arm before setting to Running.

    Based on ControllerUtils.java:103-223 (prepareControllerForAcquisition)
    """
    print("Arming SPIM state machine...")
    core.setProperty(scanner_name, "SPIMState", "Armed")
    time.sleep(0.1)

    state = core.getProperty(scanner_name, "SPIMState")
    print(f"  SPIMState: {state}")

    if state != "Armed":
        raise Exception(f"Failed to arm SPIM state machine (state={state})")


def trigger_spim_acquisition(core, scanner_name):
    """
    Trigger the SPIM state machine to start generating TTL pulses.

    This transitions from Armed -> Running and begins the hardware-triggered
    acquisition sequence.

    Based on ControllerUtils.java:823-851 (triggerControllerStartAcquisition)
    """
    print("Triggering SPIM state machine to start...")
    core.setProperty(scanner_name, "SPIMState", "Running")

    state = core.getProperty(scanner_name, "SPIMState")
    print(f"  SPIMState: {state}")


def acquire_spim_volume(
    core, camera_name, scanner_name, num_slices, scan_duration_ms, timeout_extra_sec=5.0
):
    """
    Perform hardware-triggered SPIM volume acquisition.

    CRITICAL: Must call prepareSequenceAcquisition() BEFORE startSequenceAcquisition()
    to allocate the camera's internal sequence buffer. Failing to do this causes
    "index was 0 count was 0" error.

    This function:
    1. Prepares camera sequence buffer (allocates memory)
    2. Starts camera sequence acquisition (camera waits for triggers)
    3. Triggers SPIM state machine (generates TTL pulses)
    4. Waits for images to appear in circular buffer
    5. Retrieves and returns images as numpy array

    Args:
        core: Micro-Manager core instance
        camera_name: Name of camera device
        scanner_name: Name of scanner device
        num_slices: Number of slices to acquire
        scan_duration_ms: Duration per slice (for timeout calculation)
        timeout_extra_sec: Extra time to add to timeout

    Returns:
        numpy array of shape (num_slices, height, width) containing acquired images

    Based on:
    - AcquisitionPanel.java and ControllerUtils.java (ASI diSPIM plugin)
    - pycromanager engine.py:629 (prepareSequenceAcquisition requirement)
    """
    print(f"Starting hardware-triggered acquisition of {num_slices} slices...")

    # Step 1: CRITICAL - Prepare sequence acquisition first
    # This allocates the sequence buffer in the camera's internal memory
    # Without this, you get "index was 0 count was 0" error!
    print("  Preparing camera sequence buffer...")
    core.prepareSequenceAcquisition(camera_name)
    time.sleep(0.1)  # Small delay for camera to prepare

    # Step 2: Start camera sequence acquisition
    # Camera will wait for TTL trigger pulses from Tiger controller
    print("  Starting camera sequence acquisition...")
    core.startSequenceAcquisition(camera_name, num_slices, 0, True)
    time.sleep(0.1)

    print(f"    Sequence running: {core.isSequenceRunning(camera_name)}")
    print(f"    Buffer capacity: {core.getBufferTotalCapacity()}")
    print(f"    Images in buffer: {core.getRemainingImageCount()}")

    # Step 2: Trigger SPIM state machine
    # This starts the TTL pulse generation
    trigger_spim_acquisition(core, scanner_name)

    # Step 3: Wait for hardware-triggered images
    expected_time = num_slices * scan_duration_ms / 1000.0  # Convert to seconds
    timeout = expected_time * 2 + timeout_extra_sec

    print(f"  Waiting for images (timeout={timeout:.1f}s)...")
    print(f"    Expected acquisition time: {expected_time:.1f}s")

    start_time = time.time()
    last_print_time = start_time

    while core.getRemainingImageCount() < num_slices:
        elapsed = time.time() - start_time

        # Check timeout
        if elapsed > timeout:
            count = core.getRemainingImageCount()
            spim_state = core.getProperty(scanner_name, "SPIMState")
            raise Exception(
                f"Timeout waiting for images. Got {count}/{num_slices} after {elapsed:.1f}s. "
                f"SPIMState={spim_state}"
            )

        # Print status every 0.5s
        if (elapsed - (last_print_time - start_time)) >= 0.5:
            count = core.getRemainingImageCount()
            seq_running = core.isSequenceRunning(camera_name)
            spim_state = core.getProperty(scanner_name, "SPIMState")
            print(
                f"    t={elapsed:.1f}s: images={count}/{num_slices},"
                f" seq={seq_running}, SPIM={spim_state}"
            )
            last_print_time = time.time()

        time.sleep(0.01)

    # Step 4: Retrieve images
    count = core.getRemainingImageCount()
    elapsed = time.time() - start_time

    print(f"  SUCCESS! Acquired {count} images in {elapsed:.2f}s")

    print("  Retrieving images from buffer...")
    import rpyc

    images = []
    for i in range(count):
        img = core.popNextImage()
        img = rpyc.classic.obtain(img)  # Transfer from remote to local
        images.append(img)
        print(
            f"    Image {i + 1}/{count}: shape={img.shape}, dtype={img.dtype}, "
            f"range=[{img.min()}, {img.max()}], mean={img.mean():.1f}"
        )

    # Convert to 3D numpy array
    volume = np.array(images)
    return volume


def main():
    """
    Example usage of hardware-triggered SPIM acquisition.
    """
    # Connect to Micro-Manager
    core = get_mmc()
    camera_name = "HamCam1"
    scanner_name = "Scanner:AB:33"

    # Acquisition parameters
    num_slices = 5
    slice_step_um = 1.0
    exposure_ms = 150.0
    scan_duration_ms = 160.0  # Must be > exposure + readout time
    camera_duration_ms = 155.0  # TTL pulse width (should be ~= exposure)
    laser_duration_ms = 154.0  # Laser on time (slightly less than camera)

    print("=" * 80)
    print("ASI diSPIM HARDWARE-TRIGGERED VOLUME ACQUISITION")
    print("=" * 80)

    try:
        # Apply system configuration
        print("\nApplying System Startup configuration...")
        core.setConfig("System", "Startup")
        core.waitForConfig("System", "Startup")

        # Configure circular buffer
        print("\nConfiguring circular buffer...")
        if core.isSequenceRunning():
            core.stopSequenceAcquisition()
            time.sleep(0.5)

        core.clearCircularBuffer()
        core.setCircularBufferMemoryFootprint(1200)  # MB
        print(f"  Buffer capacity: {core.getBufferTotalCapacity()} images")

        # Turn on lasers
        print("\nTurning on lasers...")
        core.setConfig("Laser", "488 and 561")
        core.waitForConfig("Laser", "488 and 561")
        print("  Lasers: 488 and 561 ON")

        # Configure camera for external triggering
        print()
        configure_camera_for_hardware_trigger(core, camera_name, exposure_ms)

        # Configure SPIM scanner
        print()
        configure_spim_scanner(
            core,
            scanner_name,
            num_slices,
            slice_step_um,
            scan_duration_ms,
            camera_duration_ms,
            laser_duration_ms,
        )

        # Arm SPIM state machine
        print()
        arm_spim_state_machine(core, scanner_name)

        # Acquire volume
        print()
        volume = acquire_spim_volume(core, camera_name, scanner_name, num_slices, scan_duration_ms)

        # Save volume
        print("\nSaving volume...")
        from PIL import Image as PILImage

        img_list = [PILImage.fromarray(img.astype(np.uint16)) for img in volume]
        img_list[0].save(
            "spim_hardware_triggered_volume.tif",
            save_all=True,
            append_images=img_list[1:],
        )
        print(f"  Saved {len(volume)}-slice volume to: spim_hardware_triggered_volume.tif")
        print(f"  Volume shape: {volume.shape} (slices, height, width)")

        # Display in napari (optional)
        try:
            import napari

            print("\nDisplaying in napari...")
            viewer = napari.Viewer()
            viewer.add_image(
                volume,
                name="SPIM Volume",
                colormap="gray",
                contrast_limits=[np.percentile(volume, 1), np.percentile(volume, 99)],
            )
            viewer.dims.axis_labels = ["Z", "Y", "X"]
            print("  Close napari window to continue...")
            napari.run()
        except ImportError:
            print("  (napari not available, skipping visualization)")

        print("\n" + "=" * 80)
        print("ACQUISITION COMPLETE!")
        print("=" * 80)

    except Exception as e:
        print("\n" + "=" * 80)
        print("ACQUISITION FAILED")
        print("=" * 80)
        print(f"Error: {e}")
        import traceback

        traceback.print_exc()

    finally:
        # Cleanup
        print("\nCleanup...")
        try:
            if core.isSequenceRunning(camera_name):
                core.stopSequenceAcquisition(camera_name)
                print("  Stopped camera sequence")
        except Exception:
            pass

        try:
            core.setProperty(scanner_name, "SPIMState", "Idle")
            print("  Reset SPIM to Idle")
        except Exception:
            pass

        try:
            # Reset camera to internal triggering for live mode
            core.setProperty(camera_name, "TRIGGER SOURCE", "INTERNAL")
            print("  Reset camera to internal triggering")
        except Exception:
            pass

        try:
            core.setConfig("Laser", "ALL OFF")
            print("  Lasers OFF")
        except Exception:
            pass


if __name__ == "__main__":
    main()
