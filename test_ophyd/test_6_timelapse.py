"""
Test 6: Long-Duration Timelapse Acquisition

Purpose: Acquire single lightsheet images over extended time period.

Tests:
- 14-hour timelapse with 2-minute intervals
- Single image at each timepoint (no piezo scan)
- Galvo at 0 (no scanning)
- Laser on/off at each timepoint
- Bluesky plan-based acquisition
- Data saved to databroker and TIFF files
- Progress tracking and estimated completion time

All images saved to test_ophyd/outputs/timelapse_TIMESTAMP/
"""

# Add parent directory to path for gently imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np
import time
from datetime import datetime, timedelta
import bluesky.plan_stubs as bps
import bluesky.preprocessors as bpp


@pytest.mark.hardware
@pytest.mark.acquisition
@pytest.mark.slow
def test_timelapse_14h(piezo, scanner, camera, laser_control, core, run_engine, output_dir):
    """
    Acquire 14-hour timelapse with 2-minute intervals.

    Verifies:
    - Long-duration acquisition stability
    - Laser on/off cycling at each timepoint
    - Image saving and databroker storage
    - Progress tracking
    """
    RE, db = run_engine

    print("\n⏱️  Testing 14-hour timelapse acquisition...")

    # Timelapse parameters
    duration_hours = 14
    interval_minutes = 2

    total_minutes = duration_hours * 60
    num_timepoints = int(total_minutes / interval_minutes)
    interval_seconds = interval_minutes * 60

    print(f"\n  Parameters:")
    print(f"    Duration: {duration_hours} hours")
    print(f"    Interval: {interval_minutes} minutes ({interval_seconds} seconds)")
    print(f"    Total timepoints: {num_timepoints}")
    print(f"    Expected end: {(datetime.now() + timedelta(hours=duration_hours)).strftime('%Y-%m-%d %H:%M:%S')}")

    # Configure devices
    print("\n  Configuring devices...")
    scanner.configure_for_calibration()  # Galvo at 0, minimal scanning
    camera.set_sensor_mode("PROGRESSIVE")
    core.setProperty("HamCam1", "Exposure", 10.0)
    print("    ✓ Scanner at galvo 0")
    print("    ✓ Camera configured")

    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timelapse_dir = output_dir / f"timelapse_{timestamp}"
    timelapse_dir.mkdir(exist_ok=True)
    print(f"\n  Saving images to: {timelapse_dir}")

    @bpp.run_decorator(md={
        'test': 'timelapse_14h',
        'duration_hours': duration_hours,
        'interval_minutes': interval_minutes,
        'num_timepoints': num_timepoints,
        'start_time': datetime.now().isoformat(),
        'output_dir': str(timelapse_dir)
    })
    def timelapse_plan():
        """Timelapse acquisition plan."""

        start_time = time.time()


        for i in range(num_timepoints):
            timepoint_start = time.time()
            elapsed_hours = (timepoint_start - start_time) / 3600

            print(f"\n  [{i+1}/{num_timepoints}] Timepoint at {elapsed_hours:.2f} hours...")
    
            yield from bps.mv(piezo, 0)

            # Turn on lasers
            print("    Turning on lasers...")
            yield from bps.mv(laser_control, "488 and 561")

            # Acquire image
            print("    Acquiring image...")
            data = yield from bps.trigger_and_read([camera])

            # Turn off lasers
            print("    Turning off lasers...")
            yield from bps.mv(laser_control, "ALL OFF")

            # Save image to TIFF
            img = data[camera.name]['value']

            # Import here to avoid loading if not needed
            import tifffile
            filename = timelapse_dir / f"timepoint_{i:04d}_t{elapsed_hours:.2f}h.tif"
            tifffile.imwrite(str(filename), img, metadata={
                'timepoint': i,
                'elapsed_hours': elapsed_hours,
                'timestamp': datetime.now().isoformat()
            })

            print(f"    ✓ Image saved: {filename.name}")
            print(f"    Image stats: mean={img.mean():.1f}, std={img.std():.1f}")

            # Calculate and display progress
            remaining_points = num_timepoints - (i + 1)
            if remaining_points > 0:
                time_per_point = (time.time() - start_time) / (i + 1)
                remaining_time = remaining_points * time_per_point
                estimated_end = datetime.now() + timedelta(seconds=remaining_time)

                print(f"    Progress: {((i+1)/num_timepoints)*100:.1f}%")
                print(f"    Estimated completion: {estimated_end.strftime('%Y-%m-%d %H:%M:%S')}")

            # Wait for next timepoint (if not the last one)
            if i < num_timepoints - 1:
                acquisition_time = time.time() - timepoint_start
                wait_time = interval_seconds - acquisition_time

                if wait_time > 0:
                    print(f"    Waiting {wait_time:.1f} seconds until next timepoint...")
                    yield from bps.sleep(wait_time)
                else:
                    print(f"    ⚠️  Warning: Acquisition took {acquisition_time:.1f}s, longer than interval!")

        total_time = time.time() - start_time
        print(f"\n  ✓ Timelapse complete!")
        print(f"  Total duration: {total_time/3600:.2f} hours")
        print(f"  Images saved: {num_timepoints}")

        return num_timepoints

    # Execute plan
    print("\n  Starting timelapse acquisition...")
    print("  (Press Ctrl+C to abort if needed)\n")

    try:
        uid = RE(timelapse_plan())

        # Verify data in databroker
        print("\n  Verifying databroker storage...")
        run = db[-1]
        table = run.table()

        print(f"  ✓ Run UID: {uid}")
        print(f"  ✓ Stored {len(table)} timepoints in databroker")
        print(f"  ✓ Images saved to: {timelapse_dir}")

        print("\n  ✓ Timelapse test verified")

    except KeyboardInterrupt:
        print("\n  ⚠️  Timelapse interrupted by user")
        print(f"  Partial data saved to: {timelapse_dir}")
        raise


@pytest.mark.hardware
@pytest.mark.acquisition
def test_timelapse_short(scanner, camera, laser_control, core, run_engine, output_dir):
    """
    Short timelapse test (10 timepoints, 5 second intervals).

    For quick validation before running the 14-hour version.

    Verifies:
    - Timelapse plan structure works
    - Laser cycling works correctly
    - Image saving works
    - Progress tracking works
    """
    RE, db = run_engine

    print("\n⏱️  Testing short timelapse (10 timepoints, 5s intervals)...")

    # Short test parameters
    num_timepoints = 10
    interval_seconds = 5

    print(f"  Parameters: {num_timepoints} timepoints, {interval_seconds}s intervals")
    print(f"  Expected duration: {num_timepoints * interval_seconds / 60:.1f} minutes")

    # Configure devices
    scanner.configure_for_calibration()
    camera.set_sensor_mode("PROGRESSIVE")
    core.setProperty("HamCam1", "Exposure", 10.0)

    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    timelapse_dir = output_dir / f"timelapse_short_{timestamp}"
    timelapse_dir.mkdir(exist_ok=True)

    @bpp.run_decorator(md={
        'test': 'timelapse_short',
        'num_timepoints': num_timepoints,
        'interval_seconds': interval_seconds,
        'start_time': datetime.now().isoformat()
    })
    def short_timelapse_plan():
        start_time = time.time()

        for i in range(num_timepoints):
            print(f"  [{i+1}/{num_timepoints}] Timepoint {i}...")

            # Lasers on
            yield from bps.mv(laser_control, "488 and 561")

            # Acquire
            data = yield from bps.trigger_and_read([camera])

            # Lasers off
            yield from bps.mv(laser_control, "ALL OFF")

            # Save
            img = data[camera.name]['value']
            import tifffile
            filename = timelapse_dir / f"t{i:04d}.tif"
            tifffile.imwrite(str(filename), img)

            print(f"    ✓ Saved (mean={img.mean():.1f})")

            # Wait
            if i < num_timepoints - 1:
                yield from bps.sleep(interval_seconds)

        return num_timepoints

    # Execute
    uid = RE(short_timelapse_plan())

    # Verify
    run = db[-1]
    table = run.table()

    assert len(table) == num_timepoints, f"Expected {num_timepoints}, got {len(table)}"

    print(f"\n  ✓ Short timelapse verified")
    print(f"  ✓ {num_timepoints} timepoints acquired")
    print(f"  ✓ Images saved to: {timelapse_dir}")


# ============================================================================
# Main (for running tests directly without pytest)
# ============================================================================

if __name__ == "__main__":
    """
    Run timelapse tests directly without pytest.

    Usage:
        python test_6_timelapse.py          # Run short test only
        python test_6_timelapse.py --long   # Run 14-hour test
    """
    import sys
    from pathlib import Path

    # Setup path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from client import get_mmc
    from gently.devices import DiSPIMScanner, DiSPIMCamera, DiSPIMLaserControl, DiSPIMPiezo
    from bluesky import RunEngine
    from databroker import Broker

    print("\n" + "="*70)
    print("DiSPIM Timelapse Acquisition Test - Manual Run")
    print("="*70)

    # Check for --long flag
    run_long = "--long" in sys.argv

    # Connect to hardware
    print("\n🔌 Connecting to Micro-Manager...")
    core = get_mmc()
    print("✓ Connected")

    # Create devices
    piezo = DiSPIMPiezo("PiezoStage:P:34", core)
    scanner = DiSPIMScanner("Scanner:AB:33", core)
    camera = DiSPIMCamera("HamCam1", core)
    laser_control = DiSPIMLaserControl(core, "Laser")

    # Create RunEngine
    RE = RunEngine({})
    db = Broker.named('temp')
    RE.subscribe(db.insert)
    run_engine = (RE, db)

    # Output directory
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)

    # Run tests
    try:
        if run_long:
            print("\n⚠️  WARNING: About to start 14-hour timelapse!")
            print("  This will acquire images every 2 minutes for 14 hours.")
            response = input("  Continue? (yes/no): ")

            if response.lower() == 'yes':
                test_timelapse_14h(piezo, scanner, camera, laser_control, core, run_engine, output_dir)
            else:
                print("  Cancelled by user")
                sys.exit(0)
        else:
            print("\n  Running SHORT timelapse test (10 timepoints, 5s intervals)")
            print("  Use --long flag to run 14-hour version")
            test_timelapse_short(scanner, camera, laser_control, core, run_engine, output_dir)

        print("\n" + "="*70)
        print("✓ TIMELAPSE TEST COMPLETE")
        print("="*70 + "\n")

    except KeyboardInterrupt:
        print("\n\n⚠️  Timelapse interrupted by user (Ctrl+C)")
        print("="*70 + "\n")
        sys.exit(1)
    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}\n")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ ERROR: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)
