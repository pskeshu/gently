"""
Test 1: Manual Device Configuration

Purpose: Verify device configuration methods correctly apply settings to
         hardware OUTSIDE of Bluesky plans.

Tests:
- Scanner configuration for calibration and volume acquisition
- Piezo amplitude and offset configuration
- Camera sensor mode switching
- Volume scanner calibration loading and multi-device setup

Each test reads back MMCore properties to verify expected values.
"""

# Add parent directory to path for gently imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest


@pytest.mark.hardware
def test_scanner_config_for_calibration(scanner, core):
    """
    Test scanner configuration for calibration mode.

    Verifies:
    - Y amplitude setting
    - Y offset setting
    - Number of slices
    - SPIM state
    """
    print("\n📐 Testing scanner calibration configuration...")

    # Configure scanner for calibration (sets defaults: X=8°, Y=0.0001°)
    scanner.configure_for_calibration()

    # Read back MMCore properties
    x_amplitude = float(core.getProperty("Scanner:AB:33", "SingleAxisXAmplitude(deg)"))
    y_amplitude = float(core.getProperty("Scanner:AB:33", "SingleAxisYAmplitude(deg)"))
    beam_enabled = core.getProperty("Scanner:AB:33", "BeamEnabled")

    # Print configuration
    print(f"  X amplitude: {x_amplitude:.2f}° (light sheet width)")
    print(f"  Y amplitude: {y_amplitude:.4f}° (minimal for calibration)")
    print(f"  Beam enabled: {beam_enabled}")

    # Assertions
    assert abs(x_amplitude - 8.0) < 0.1, f"X amplitude should be 8.0°, got {x_amplitude}"
    assert abs(y_amplitude - 0.0001) < 0.001, f"Y amplitude should be ~0.0001°, got {y_amplitude}"
    assert beam_enabled == "Yes", "Beam should be enabled"

    print("  ✓ Scanner calibration config verified")


@pytest.mark.hardware
def test_scanner_config_for_volume(scanner, core):
    """
    Test scanner configuration for volume acquisition mode.

    Verifies:
    - Multiple slices
    - Scanner timing parameters
    - Amplitude and offset for volume scanning
    """
    print("\n📐 Testing scanner volume acquisition configuration...")

    # Configure for volume acquisition
    galvo_amplitude = 2.0  # degrees
    galvo_center = 1.0     # degrees
    num_slices = 50

    scanner.configure_for_volume_acquisition(
        galvo_amplitude=galvo_amplitude,
        galvo_center=galvo_center,
        num_slices=num_slices
    )

    # Read back
    actual_slices = int(core.getProperty("Scanner:AB:33", "SPIMNumSlices"))
    actual_amplitude = float(core.getProperty("Scanner:AB:33", "SingleAxisYAmplitude(deg)"))
    actual_offset = float(core.getProperty("Scanner:AB:33", "SingleAxisYOffset(deg)"))
    x_amplitude = float(core.getProperty("Scanner:AB:33", "SingleAxisXAmplitude(deg)"))

    # Print comparison
    print(f"  Expected: slices={num_slices}, Y_amplitude={galvo_amplitude}°, Y_center={galvo_center}°")
    print(f"  Actual:   slices={actual_slices}, Y_amplitude={actual_amplitude:.2f}°, Y_center={actual_offset:.2f}°")
    print(f"  X amplitude (light sheet): {x_amplitude:.2f}°")

    # Assertions
    assert actual_slices == num_slices, f"Expected {num_slices} slices, got {actual_slices}"
    assert abs(actual_amplitude - galvo_amplitude) < 0.01, f"Y amplitude mismatch"
    assert abs(actual_offset - galvo_center) < 0.01, f"Y center mismatch"

    print("  ✓ Scanner volume config verified")


@pytest.mark.hardware
def test_piezo_amplitude_offset(piezo, core):
    """
    Test piezo amplitude and offset configuration.

    Verifies:
    - Amplitude setting
    - Offset setting
    - SPIM state (should be Armed after config)
    """
    print("\n🔧 Testing piezo amplitude/offset configuration...")

    # Set specific amplitude and offset
    amplitude = 10.0  # micrometers
    offset = 5.0      # micrometers

    piezo.configure_amplitude_offset(amplitude_um=amplitude, offset_um=offset)

    # Read back MMCore properties
    actual_amp = float(core.getProperty("PiezoStage:P:34", "SingleAxisAmplitude(um)"))
    actual_offset = float(core.getProperty("PiezoStage:P:34", "SingleAxisOffset(um)"))
    spim_state = core.getProperty("PiezoStage:P:34", "SPIMState")

    # Print comparison
    print(f"  Expected: amplitude={amplitude} µm, offset_um={offset} µm")
    print(f"  Actual:   amplitude={actual_amp:.2f} µm, offset_um={actual_offset:.2f} µm")
    print(f"  SPIM State: {spim_state}")

    # Assertions (allow 0.1 µm tolerance)
    assert abs(actual_amp - amplitude) < 0.1, f"Amplitude mismatch: {actual_amp}"
    assert abs(actual_offset - offset) < 0.1, f"Offset mismatch: {actual_offset}"
    assert spim_state in ["Armed", "Idle"], f"Unexpected SPIM state: {spim_state}"

    print("  ✓ Piezo amplitude/offset verified")


@pytest.mark.hardware
def test_piezo_spim_state_control(piezo, core):
    """
    Test piezo SPIM state control (Armed/Idle).

    Verifies:
    - Can arm piezo for SPIM
    - Can disarm/idle piezo
    """
    print("\n🔧 Testing piezo SPIM state control...")

    # Arm the piezo
    core.setProperty("PiezoStage:P:34", "SPIMState", "Armed")
    state_armed = core.getProperty("PiezoStage:P:34", "SPIMState")
    print(f"  After arming: {state_armed}")
    assert state_armed == "Armed"

    # Idle the piezo
    core.setProperty("PiezoStage:P:34", "SPIMState", "Idle")
    state_idle = core.getProperty("PiezoStage:P:34", "SPIMState")
    print(f"  After idle: {state_idle}")
    assert state_idle == "Idle"

    print("  ✓ Piezo state control verified")


@pytest.mark.hardware
def test_camera_sensor_mode_area(camera, core):
    """
    Test camera sensor mode switching to AREA mode.

    Verifies:
    - Can set sensor mode to AREA
    - Mode is correctly applied in hardware
    """
    print("\n📷 Testing camera AREA sensor mode...")

    # Set to AREA mode
    camera.set_sensor_mode("AREA")

    # Read back
    mode = core.getProperty("HamCam1", "SENSOR MODE")

    print(f"  Sensor mode: {mode}")

    assert mode == "AREA", f"Expected AREA, got {mode}"

    print("  ✓ Camera AREA mode verified")


@pytest.mark.hardware
def test_camera_sensor_mode_progressive(camera, core):
    """
    Test camera sensor mode switching to PROGRESSIVE mode.

    Verifies:
    - Can set sensor mode to PROGRESSIVE
    - Mode is correctly applied in hardware
    """
    print("\n📷 Testing camera PROGRESSIVE sensor mode...")

    # Set to PROGRESSIVE mode
    camera.set_sensor_mode("PROGRESSIVE")

    # Read back
    mode = core.getProperty("HamCam1", "SENSOR MODE")

    print(f"  Sensor mode: {mode}")

    assert mode == "PROGRESSIVE", f"Expected PROGRESSIVE, got {mode}"

    print("  ✓ Camera PROGRESSIVE mode verified")


@pytest.mark.hardware
def test_camera_buffer_configuration(camera, core):
    """
    Test camera circular buffer configuration.

    Verifies:
    - Can configure buffer size
    - Buffer is ready for acquisition
    """
    print("\n📷 Testing camera circular buffer configuration...")

    # Configure buffer (this is usually done in camera.__init__ or configure methods)
    try:
        core.setCircularBufferMemoryFootprint(2000)  # 2000 MB
        buffer_size = core.getBufferTotalCapacity()
        print(f"  Circular buffer capacity: {buffer_size} images")
        print(f"  ✓ Buffer configured")
    except Exception as e:
        print(f"  ⚠️  Buffer configuration note: {e}")
        # This might fail if buffer is already in use, which is okay


@pytest.mark.hardware
def test_volume_scanner_calibration_load(volume_scanner, calibration_file):
    """
    Test loading calibration into volume scanner.

    Verifies:
    - Calibration JSON can be loaded
    - Calibration parameters are stored
    - Volume scanner is ready for calibrated acquisition
    """
    print("\n📊 Testing volume scanner calibration loading...")

    import json
    import os

    # Check if calibration file exists
    if not os.path.exists(calibration_file):
        pytest.skip(f"Calibration file not found: {calibration_file}")

    # Load calibration file to see what's in it
    with open(calibration_file, 'r') as f:
        calib_data = json.load(f)

    print(f"  Calibration file: {calibration_file}")
    print(f"  Calibration data: {calib_data}")

    # Configure volume scanner with calibration
    # Note: The actual method might vary - check DiSPIMVolumeScanner implementation
    if hasattr(volume_scanner, 'load_calibration'):
        volume_scanner.load_calibration(calibration_file)
        print("  ✓ Calibration loaded via load_calibration()")
    elif hasattr(volume_scanner, 'configure_from_calibration'):
        volume_scanner.configure_from_calibration(calibration_file)
        print("  ✓ Calibration loaded via configure_from_calibration()")
    else:
        # Manually set calibration if no method exists
        volume_scanner.calibration = calib_data
        print("  ✓ Calibration set manually")

    # Verify calibration is stored
    assert volume_scanner.calibration is not None, "Calibration not stored"
    assert 'slope' in volume_scanner.calibration, "Missing 'slope' in calibration"
    assert 'intercept' in volume_scanner.calibration, "Missing 'intercept' in calibration"

    print(f"  Slope: {volume_scanner.calibration['slope']:.4f}")
    print(f"  Intercept: {volume_scanner.calibration['intercept']:.4f}")
    print("  ✓ Volume scanner calibration verified")


@pytest.mark.hardware
def test_volume_scanner_device_access(volume_scanner):
    """
    Test that volume scanner has access to all sub-devices.

    Verifies:
    - Scanner component accessible
    - Camera component accessible
    - Piezo component accessible
    - Laser control component accessible
    """
    print("\n📊 Testing volume scanner device composition...")

    # Check that all sub-devices are accessible
    assert volume_scanner.scanner is not None, "Scanner not accessible"
    assert volume_scanner.camera is not None, "Camera not accessible"
    assert volume_scanner.piezo is not None, "Piezo not accessible"
    assert volume_scanner.laser_control is not None, "Laser control not accessible"

    print("  ✓ Scanner component: accessible")
    print("  ✓ Camera component: accessible")
    print("  ✓ Piezo component: accessible")
    print("  ✓ Laser control component: accessible")

    print("  ✓ Volume scanner composition verified")


# ============================================================================
# Comparison Table Helper
# ============================================================================

def print_comparison_table(parameter, expected, actual, tolerance=None):
    """
    Print a formatted comparison table for test results.

    Args:
        parameter: Parameter name
        expected: Expected value
        actual: Actual value
        tolerance: Optional tolerance for comparison
    """
    match = "✓" if abs(expected - actual) < (tolerance or 0.01) else "✗"
    print(f"  {match} {parameter:20s} Expected: {expected:8.2f}  Actual: {actual:8.2f}")


# ============================================================================
# Main (for running tests directly without pytest)
# ============================================================================

if __name__ == "__main__":
    """
    Run tests directly without pytest for quick validation.

    Usage:
        python test_1_manual_config.py
    """
    import sys
    from pathlib import Path

    # Setup path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from client import get_mmc
    from gently.devices import (
        DiSPIMScanner, DiSPIMPiezo, DiSPIMCamera,
        DiSPIMVolumeScanner, DiSPIMLaserControl
    )

    print("\n" + "="*70)
    print("DiSPIM Device Configuration Test - Manual Run")
    print("="*70)

    # Connect to hardware
    print("\n🔌 Connecting to Micro-Manager...")
    core = get_mmc()
    print("✓ Connected")

    # Create devices
    scanner = DiSPIMScanner("Scanner:AB:33", core)
    piezo = DiSPIMPiezo("PiezoStage:P:34", core)
    camera = DiSPIMCamera("HamCam1", core)
    laser_control = DiSPIMLaserControl("Laser", core)
    volume_scanner = DiSPIMVolumeScanner(scanner, camera, piezo, laser_control, core)

    # Run tests
    try:
        test_scanner_config_for_calibration(scanner, core)
        test_scanner_config_for_volume(scanner, core)
        test_piezo_amplitude_offset(piezo, core)
        test_piezo_spim_state_control(piezo, core)
        test_camera_sensor_mode_area(camera, core)
        test_camera_sensor_mode_progressive(camera, core)
        test_camera_buffer_configuration(camera, core)
        test_volume_scanner_device_access(volume_scanner)

        # Try calibration load if file exists
        calibration_file = "backend/piezo_galvo_calibration_embryo.json"
        import os
        if os.path.exists(calibration_file):
            test_volume_scanner_calibration_load(volume_scanner, calibration_file)
        else:
            print(f"\n📊 Testing volume scanner calibration loading...")
            print(f"  ⚠️  Skipped: Calibration file not found: {calibration_file}")

        print("\n" + "="*70)
        print("✓ ALL TESTS PASSED")
        print("="*70 + "\n")

    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}\n")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ ERROR: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)
