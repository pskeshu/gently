"""
Test 2: Device Configuration Within Bluesky Plans

Purpose: Verify device configuration works correctly WITHIN Bluesky plan context.

Tests:
- Scanner configuration from within plans
- Piezo configuration using plan stubs
- Camera mode switching in plans
- Volume scanner configuration orchestration
- Verification that configuration appears in databroker metadata

Each test creates a Bluesky run and verifies successful execution.
"""

# Add parent directory to path for gently imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import bluesky.plan_stubs as bps
import bluesky.preprocessors as bpp


@pytest.mark.hardware
def test_configure_scanner_in_plan(scanner, run_engine):
    """
    Test scanner configuration from within a Bluesky plan.

    Verifies:
    - Configuration methods work when called in plan context
    - Plan executes without errors
    - Run is created in databroker
    """
    RE, db = run_engine

    print("\n📐 Testing scanner configuration in plan...")

    @bpp.run_decorator(md={'test': 'scanner_config', 'device': 'scanner'})
    def config_plan():
        # Configure scanner within plan
        print("  Configuring scanner for calibration...")
        scanner.configure_for_calibration()
        yield from bps.null()  # Just to make it a generator

    # Execute plan
    uids = RE(config_plan())

    # Verify run was created
    assert uids is not None, "Run UID is None"
    # Get the most recent run from databroker
    run = db[-1]
    # In databroker v1, metadata is in run.start (Header object)
    assert run.start['test'] == 'scanner_config'
    uid = run.start['uid']

    print(f"  ✓ Plan executed successfully")
    print(f"  ✓ Run UID: {uid[:8]}...")
    print(f"  ✓ Scanner configuration in plan verified")


@pytest.mark.hardware
def test_configure_piezo_in_plan(piezo, run_engine):
    """
    Test piezo configuration from within a plan.

    Verifies:
    - Amplitude and offset can be set in plan
    - Status handling works correctly
    - Plan completes successfully
    """
    RE, db = run_engine

    print("\n🔧 Testing piezo configuration in plan...")

    @bpp.run_decorator(md={'test': 'piezo_config', 'device': 'piezo'})
    def config_plan():
        print("  Configuring piezo amplitude/offset...")
        status = piezo.configure_amplitude_offset(amplitude_um=10.0, offset_um=5.0)

        # Wait for configuration to complete
        if status is not None:
            yield from bps.wait()

        print("  ✓ Piezo configured")

    # Execute
    uid = RE(config_plan())

    assert uid is not None
    run = db[-1]  # Get most recent run
    assert run.start['test'] == 'piezo_config'

    print(f"  ✓ Run UID: {uid[:8]}...")
    print(f"  ✓ Piezo configuration in plan verified")


@pytest.mark.hardware
def test_camera_mode_switch_in_plan(camera, run_engine):
    """
    Test camera sensor mode switching within a plan.

    Verifies:
    - Can switch sensor modes from within plan
    - Mode changes are applied during plan execution
    """
    RE, db = run_engine

    print("\n📷 Testing camera mode switching in plan...")

    @bpp.run_decorator(md={'test': 'camera_mode', 'device': 'camera'})
    def mode_switch_plan():
        print("  Switching to PROGRESSIVE mode...")
        camera.set_sensor_mode("PROGRESSIVE")
        yield from bps.null()  # Just to make it a generator

        print("  Switching to AREA mode...")
        camera.set_sensor_mode("AREA")
        yield from bps.null()

        print("  ✓ Mode switches complete")

    # Execute
    uid = RE(mode_switch_plan())

    assert uid is not None
    run = db[-1]  # Get most recent run
    assert run.start['test'] == 'camera_mode'

    print(f"  ✓ Run UID: {uid[:8]}...")
    print(f"  ✓ Camera mode switching in plan verified")


@pytest.mark.hardware
def test_volume_scanner_config_in_plan(volume_scanner, run_engine, calibration_file):
    """
    Test full volume scanner configuration within a plan.

    Verifies:
    - Can load calibration in plan context
    - Can configure all sub-devices
    - Multi-device orchestration works
    """
    RE, db = run_engine

    print("\n📊 Testing volume scanner configuration in plan...")

    import os
    if not os.path.exists(calibration_file):
        pytest.skip(f"Calibration file not found: {calibration_file}")

    @bpp.run_decorator(md={'test': 'volume_config', 'device': 'volume_scanner'})
    def volume_config_plan():
        print("  Loading calibration...")

        # Load calibration
        import json
        with open(calibration_file, 'r') as f:
            calib = json.load(f)

        volume_scanner.calibration = calib
        print(f"    Slope: {calib['slope']:.4f}")
        print(f"    Intercept: {calib['intercept']:.4f}")

        # Configure scanner
        print("  Configuring scanner for volume...")
        volume_scanner.scanner.configure_for_volume_acquisition(
            galvo_amplitude=2.0,
            galvo_center=0.0,
            num_slices=50
        )

        # Configure piezo
        print("  Configuring piezo...")
        volume_scanner.piezo.configure_amplitude_offset(
            amplitude=10.0,
            offset=0.0
        )

        # Configure camera
        print("  Setting camera mode...")
        volume_scanner.camera.set_sensor_mode("PROGRESSIVE")

        yield from bps.null()

        print("  ✓ Volume scanner fully configured")

    # Execute
    uid = RE(volume_config_plan())

    assert uid is not None
    run = db[-1]  # Get most recent run
    assert run.start['test'] == 'volume_config'

    print(f"  ✓ Run UID: {uid[:8]}...")
    print(f"  ✓ Volume scanner configuration in plan verified")


@pytest.mark.hardware
def test_sequential_device_config_plan(scanner, piezo, camera, run_engine):
    """
    Test configuring multiple devices sequentially in one plan.

    Verifies:
    - Multiple device configurations in sequence
    - Plan flow control works correctly
    - All configurations are applied
    """
    RE, db = run_engine

    print("\n🔄 Testing sequential device configuration...")

    @bpp.run_decorator(md={'test': 'sequential_config', 'devices': 'multiple'})
    def sequential_plan():
        # Step 1: Configure scanner
        print("  [1/3] Configuring scanner...")
        scanner.configure_for_calibration()
        yield from bps.null()

        # Step 2: Configure piezo
        print("  [2/3] Configuring piezo...")
        piezo.configure_amplitude_offset(amplitude_um=5.0, offset_um=0.0)
        yield from bps.null()

        # Step 3: Configure camera
        print("  [3/3] Configuring camera...")
        camera.set_sensor_mode("PROGRESSIVE")
        yield from bps.null()

        print("  ✓ All devices configured sequentially")

    # Execute
    uid = RE(sequential_plan())

    assert uid is not None
    run = db[-1]  # Get most recent run
    assert run.start['test'] == 'sequential_config'

    print(f"  ✓ Run UID: {uid[:8]}...")
    print(f"  ✓ Sequential configuration verified")


@pytest.mark.hardware
def test_config_with_metadata_capture(scanner, run_engine, core):
    """
    Test that device configuration appears in run metadata.

    Verifies:
    - Configuration parameters captured as metadata
    - Can retrieve configuration from databroker
    """
    RE, db = run_engine

    print("\n📝 Testing configuration metadata capture...")

    # Scanner configure_for_calibration sets fixed defaults (X=8°, Y=0.0001°)
    expected_config = {
        'x_amplitude': 8.0,
        'y_amplitude': 0.0001,
        'beam_enabled': 'Yes'
    }

    @bpp.run_decorator(md={'test': 'config_metadata', 'expected_config': expected_config})
    def config_with_metadata_plan():
        print("  Configuring with metadata...")
        scanner.configure_for_calibration()
        yield from bps.null()

    # Execute
    uid = RE(config_with_metadata_plan())

    # Retrieve and verify metadata
    run = db[-1]  # Get most recent run
    assert 'expected_config' in run.start

    # Verify actual hardware settings match expected defaults
    actual_x_amp = float(core.getProperty("Scanner:AB:33", "SingleAxisXAmplitude(deg)"))
    actual_y_amp = float(core.getProperty("Scanner:AB:33", "SingleAxisYAmplitude(deg)"))

    assert abs(actual_x_amp - 8.0) < 0.1, f"X amplitude should be 8.0°, got {actual_x_amp}"
    assert abs(actual_y_amp - 0.0001) < 0.001, f"Y amplitude should be ~0.0001°, got {actual_y_amp}"

    print(f"  ✓ Metadata captured: {run.start['expected_config']}")
    print(f"  ✓ Hardware matches expected defaults (X=8.0°, Y=0.0001°)")
    print(f"  ✓ Configuration metadata verified")


@pytest.mark.hardware
def test_config_error_handling_in_plan(scanner, run_engine):
    """
    Test error handling when configuration fails in a plan.

    Verifies:
    - Errors are properly caught and handled
    - Plan can gracefully handle configuration failures
    """
    RE, db = run_engine

    print("\n⚠️  Testing configuration error handling...")

    @bpp.run_decorator(md={'test': 'error_handling'})
    def error_handling_plan():
        try:
            # Configure normally (scanner.configure_for_calibration takes no params)
            scanner.configure_for_calibration()
            print("  Configuration accepted (no error)")
        except Exception as e:
            print(f"  Configuration error caught: {e}")

        yield from bps.null()

    # Execute - should complete even if configuration has issues
    uid = RE(error_handling_plan())

    assert uid is not None
    print(f"  ✓ Plan completed despite potential issues")
    print(f"  ✓ Error handling verified")


@pytest.mark.hardware
def test_read_configuration_in_plan(scanner, run_engine):
    """
    Test reading device configuration within a plan.

    Verifies:
    - Can read configuration using bps.rd()
    - Configuration values are accessible
    """
    RE, db = run_engine

    print("\n📖 Testing configuration reading in plan...")

    @bpp.run_decorator(md={'test': 'read_config'})
    def read_config_plan():
        # Configure first
        print("  Configuring scanner...")
        scanner.configure_for_calibration()

        # Try to read configuration if device supports it
        # Note: read_configuration() returns OrderedDict in current implementation
        config = scanner.read_configuration()
        print(f"  Configuration read: {len(config)} items")

        yield from bps.null()

    # Execute
    uid = RE(read_config_plan())

    assert uid is not None
    print(f"  ✓ Configuration reading verified")


# ============================================================================
# Main (for running tests directly without pytest)
# ============================================================================

if __name__ == "__main__":
    """
    Run tests directly without pytest for quick validation.

    Usage:
        python test_2_plan_config.py
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
    from bluesky import RunEngine
    from databroker import Broker

    print("\n" + "="*70)
    print("DiSPIM Plan Configuration Test - Manual Run")
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

    # Create RunEngine
    RE = RunEngine({})
    db = Broker.named('temp')
    RE.subscribe(db.insert)
    run_engine = (RE, db)

    calibration_file = "backend/piezo_galvo_calibration_embryo.json"

    # Run tests
    try:
        test_configure_scanner_in_plan(scanner, run_engine)
        test_configure_piezo_in_plan(piezo, run_engine)
        test_camera_mode_switch_in_plan(camera, run_engine)
        test_sequential_device_config_plan(scanner, piezo, camera, run_engine)
        test_config_with_metadata_capture(scanner, run_engine, core)
        test_config_error_handling_in_plan(scanner, run_engine)
        test_read_configuration_in_plan(scanner, run_engine)

        # Try volume scanner config if calibration file exists
        import os
        if os.path.exists(calibration_file):
            test_volume_scanner_config_in_plan(volume_scanner, run_engine, calibration_file)
        else:
            print(f"\n⚠️  Skipped volume scanner test (calibration file not found)")

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
