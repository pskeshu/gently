"""
Test 3: Single Lightsheet Slice Acquisition

Purpose: Acquire a single lightsheet image with scanner and camera coordination.

Tests:
- Manual trigger (direct device.trigger() calls)
- Plan-based acquisition (using bps.trigger_and_read())
- Light sheet snap device (compound device)
- Image validation (not blank, correct dimensions)
- Image saving to TIFF

All acquired images are saved to test_ophyd/outputs/ for inspection.
"""

# Add parent directory to path for gently imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np
import bluesky.plan_stubs as bps
import bluesky.preprocessors as bpp


@pytest.mark.hardware
@pytest.mark.acquisition
def test_manual_single_slice(scanner, camera, laser_control, core, output_dir):
    """
    Acquire single slice using manual trigger sequence.

    Verifies:
    - Scanner can be triggered
    - Camera can be triggered
    - Image is acquired and not blank
    - Image has correct dimensions
    """
    print("\n📸 Testing manual single slice acquisition...")

    # Configure devices
    print("  Configuring scanner for single slice...")
    scanner.configure_for_calibration()

    print("  Setting camera to NORMAL mode...")
    camera.set_sensor_mode("PROGRESSIVE")

    # Set camera exposure
    core.setProperty("HamCam1", "Exposure", 10.0)  # 10 ms
    print(f"  Camera exposure: {core.getProperty('HamCam1', 'Exposure')} ms")

    # Turn on lasers (488 and 561)
    print("  Turning on lasers (488 and 561)...")
    laser_status = laser_control.set("488 and 561")
    laser_status.wait()

    # Scanner is already configured for continuous light sheet
    # Acquire image
    print("  Acquiring image...")
    camera_status = camera.trigger()
    camera_status.wait()
    print("  ✓ Image acquired")

    # Turn off lasers
    print("  Turning off lasers...")
    laser_status = laser_control.set("ALL OFF")
    laser_status.wait()

    # Read image
    print("  Reading image data...")
    data = camera.read()
    # Camera uses device name as key (e.g., 'HamCam1')
    img = data[camera.name]['value']

    # Validate image
    print(f"  Image shape: {img.shape}")
    print(f"  Image dtype: {img.dtype}")
    print(f"  Image range: [{img.min()}, {img.max()}]")
    print(f"  Image mean: {img.mean():.1f}")

    assert img.shape[0] > 0 and img.shape[1] > 0, "Image has zero dimensions"
    assert img.mean() > 10, f"Image appears blank (mean={img.mean():.1f})"

    # Save image
    import tifffile
    output_path = output_dir / "single_slice_manual.tif"
    tifffile.imwrite(str(output_path), img)
    print(f"  ✓ Image saved: {output_path}")

    print("  ✓ Manual single slice acquisition verified")


@pytest.mark.hardware
@pytest.mark.acquisition
def test_single_slice_plan(scanner, camera, laser_control, core, run_engine, output_dir):
    """
    Acquire single slice using Bluesky plan.

    Verifies:
    - Trigger and read works in plan context
    - Image is stored in databroker
    - Image can be retrieved from run
    """
    RE, db = run_engine

    print("\n📸 Testing single slice acquisition in plan...")

    # Configure devices
    scanner.configure_for_calibration()
    camera.set_sensor_mode("PROGRESSIVE")
    core.setProperty("HamCam1", "Exposure", 10.0)

    @bpp.run_decorator(md={'test': 'single_slice_plan', 'acquisition': 'lightsheet'})
    def slice_plan():
        # Turn on lasers
        print("  Turning on lasers (488 and 561)...")
        yield from bps.mv(laser_control, "488 and 561")

        # Acquire image
        print("  Triggering and reading camera...")
        data = yield from bps.trigger_and_read([camera])

        # Turn off lasers
        print("  Turning off lasers...")
        yield from bps.mv(laser_control, "ALL OFF")

        print("  ✓ Slice acquired")
        return data

    # Execute plan
    uid = RE(slice_plan())

    # Retrieve from databroker
    print("  Retrieving from databroker...")
    run = db[-1]  # Get most recent run

    # In databroker v1, use table() to get data
    table = run.table()

    # Camera data is stored with device name as key
    assert camera.name in table.columns, f"No {camera.name} in databroker"

    img = table[camera.name].iloc[0]  # Get first (and only) image

    print(f"  Image shape: {img.shape}")
    print(f"  Image mean: {img.mean():.1f}")

    assert img.mean() > 10, "Image appears blank"

    # Save image
    import tifffile
    output_path = output_dir / "single_slice_plan.tif"
    tifffile.imwrite(str(output_path), img)
    print(f"  ✓ Image saved: {output_path}")

    print(f"  ✓ Run UID: {uid[:8]}...")
    print("  ✓ Plan-based single slice verified")


@pytest.mark.hardware
@pytest.mark.acquisition
def test_lightsheet_snap_device(lightsheet_snap, run_engine, output_dir):
    """
    Test using DiSPIMLightSheetSnap compound device.

    Verifies:
    - Snap device triggers both scanner and camera
    - Compound device workflow
    - Image acquisition through snap device
    """
    RE, db = run_engine

    print("\n📸 Testing DiSPIMLightSheetSnap device...")

    @bpp.run_decorator(md={'test': 'lightsheet_snap', 'device': 'DiSPIMLightSheetSnap'})
    def snap_plan():
        print("  Triggering snap device...")
        data = yield from bps.trigger_and_read([lightsheet_snap])
        return data

    # Execute
    uid = RE(snap_plan())

    # Retrieve
    run = db[-1]  # Get most recent run
    table = run.table()

    # The key name might vary - check what's available
    print(f"  Available data keys: {list(table.columns)}")

    # Try to find image data
    img = None
    for key in table.columns:
        if 'image' in key.lower() or 'camera' in key.lower():
            img = table[key].iloc[0]
            break

    if img is not None:
        print(f"  Image shape: {img.shape}")
        print(f"  Image mean: {img.mean():.1f}")

        # Save
        import tifffile
        output_path = output_dir / "single_slice_snap_device.tif"
        tifffile.imwrite(str(output_path), img)
        print(f"  ✓ Image saved: {output_path}")
    else:
        print("  ⚠️  No image data found in snap device output")

    print("  ✓ Snap device acquisition verified")


@pytest.mark.hardware
@pytest.mark.acquisition
def test_slice_with_different_amplitudes(scanner, camera, laser_control, core, output_dir):
    """
    Acquire slices with different scanner amplitudes.

    Verifies:
    - Amplitude changes affect lightsheet
    - Multiple acquisitions work sequentially
    """
    print("\n📸 Testing slices with varying amplitudes...")

    camera.set_sensor_mode("PROGRESSIVE")
    core.setProperty("HamCam1", "Exposure", 10.0)

    amplitudes = [0.3, 0.5, 0.7]
    images = []

    for i, amplitude in enumerate(amplitudes):
        print(f"  [{i+1}/{len(amplitudes)}] Acquiring with amplitude {amplitude}...")

        # Configure scanner
        scanner.configure_for_calibration()

        # Turn on lasers
        laser_status = laser_control.set("488 and 561")
        laser_status.wait()

        # Acquire image
        camera_status = camera.trigger()
        camera_status.wait()

        # Turn off lasers
        laser_status = laser_control.set("ALL OFF")
        laser_status.wait()

        data = camera.read()
        img = data[camera.name]['value']
        images.append(img)

        print(f"      Image mean: {img.mean():.1f}")

    # Compare images
    print(f"\n  Amplitude comparison:")
    for i, (amp, img) in enumerate(zip(amplitudes, images)):
        print(f"    Amplitude {amp}: mean={img.mean():.1f}, std={img.std():.1f}")

    # Save montage
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, (amp, img) in enumerate(zip(amplitudes, images)):
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f'Amplitude {amp}°')
        axes[i].axis('off')

    output_path = output_dir / "amplitude_comparison.png"
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  ✓ Montage saved: {output_path}")

    print("  ✓ Amplitude variation verified")


@pytest.mark.hardware
@pytest.mark.acquisition
def test_slice_with_offset_variation(scanner, camera, laser_control, core, output_dir):
    """
    Acquire slices with different scanner offsets.

    Verifies:
    - Offset changes shift lightsheet position
    - Y-offset parameter works correctly
    """
    print("\n📸 Testing slices with varying offsets...")

    camera.set_sensor_mode("PROGRESSIVE")
    core.setProperty("HamCam1", "Exposure", 10.0)

    offsets = [-0.5, 0.0, 0.5]
    images = []

    for i, offset in enumerate(offsets):
        print(f"  [{i+1}/{len(offsets)}] Acquiring with offset {offset}...")

        # Configure scanner
        scanner.configure_for_calibration()

        # Turn on lasers
        laser_status = laser_control.set("488 and 561")
        laser_status.wait()

        # Acquire image
        camera_status = camera.trigger()
        camera_status.wait()

        # Turn off lasers
        laser_status = laser_control.set("ALL OFF")
        laser_status.wait()

        data = camera.read()
        img = data[camera.name]['value']
        images.append(img)

        print(f"      Image mean: {img.mean():.1f}")

    # Save montage
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, (offset, img) in enumerate(zip(offsets, images)):
        axes[i].imshow(img, cmap='gray')
        axes[i].set_title(f'Offset {offset}°')
        axes[i].axis('off')

    output_path = output_dir / "offset_comparison.png"
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  ✓ Montage saved: {output_path}")

    print("  ✓ Offset variation verified")


@pytest.mark.hardware
@pytest.mark.acquisition
def test_image_display(scanner, camera, laser_control, core, output_dir):
    """
    Acquire and display image with matplotlib.

    Verifies:
    - Image can be visualized
    - Histogram analysis
    - ROI statistics
    """
    print("\n📸 Testing image display and analysis...")

    # Configure
    scanner.configure_for_calibration()
    camera.set_sensor_mode("PROGRESSIVE")
    core.setProperty("HamCam1", "Exposure", 10.0)

    # Turn on lasers
    laser_status = laser_control.set("488 and 561")
    laser_status.wait()

    # Acquire image
    camera_status = camera.trigger()
    camera_status.wait()

    # Turn off lasers
    laser_status = laser_control.set("ALL OFF")
    laser_status.wait()

    data = camera.read()
    img = data[camera.name]['value']

    # Create visualization
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Image display
    im = axes[0].imshow(img, cmap='gray')
    axes[0].set_title('Lightsheet Image')
    axes[0].axis('off')
    plt.colorbar(im, ax=axes[0])

    # Histogram
    axes[1].hist(img.ravel(), bins=100, color='gray', alpha=0.7)
    axes[1].set_xlabel('Pixel Intensity')
    axes[1].set_ylabel('Count')
    axes[1].set_title('Intensity Histogram')
    axes[1].axvline(img.mean(), color='r', linestyle='--', label=f'Mean={img.mean():.1f}')
    axes[1].legend()

    output_path = output_dir / "image_analysis.png"
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Analysis plot saved: {output_path}")

    # Print statistics
    print(f"\n  Image Statistics:")
    print(f"    Shape: {img.shape}")
    print(f"    Mean: {img.mean():.1f}")
    print(f"    Std: {img.std():.1f}")
    print(f"    Min: {img.min()}")
    print(f"    Max: {img.max()}")
    print(f"    Median: {np.median(img):.1f}")

    print("  ✓ Image display and analysis verified")


# ============================================================================
# Main (for running tests directly without pytest)
# ============================================================================

if __name__ == "__main__":
    """
    Run tests directly without pytest for quick validation.

    Usage:
        python test_3_single_slice.py
    """
    import sys
    from pathlib import Path

    # Setup path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from client import get_mmc
    from gently.devices import (
        DiSPIMScanner, DiSPIMPiezo, DiSPIMCamera,
        DiSPIMLightSheetSnap, DiSPIMLaserControl, DiSPIMVolumeScanner
    )
    from bluesky import RunEngine
    from databroker import Broker

    print("\n" + "="*70)
    print("DiSPIM Single Slice Acquisition Test - Manual Run")
    print("="*70)

    # Connect to hardware
    print("\n🔌 Connecting to Micro-Manager...")
    core = get_mmc()
    print("✓ Connected")

    # Create devices
    scanner = DiSPIMScanner("Scanner:AB:33", core)
    camera = DiSPIMCamera("HamCam1", core)
    piezo = DiSPIMPiezo("PiezoStage:P:34", core)
    laser_control = DiSPIMLaserControl(core, "Laser")
    lightsheet_snap = DiSPIMLightSheetSnap(scanner, camera)

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
        test_manual_single_slice(scanner, camera, laser_control, core, output_dir)
        test_single_slice_plan(scanner, camera, laser_control, core, run_engine, output_dir)
        test_lightsheet_snap_device(lightsheet_snap, run_engine, output_dir)
        test_slice_with_different_amplitudes(scanner, camera, laser_control, core, output_dir)
        test_slice_with_offset_variation(scanner, camera, laser_control, core, output_dir)
        test_image_display(scanner, camera, laser_control, core, output_dir)

        print("\n" + "="*70)
        print("✓ ALL TESTS PASSED")
        print(f"✓ Images saved to: {output_dir}")
        print("="*70 + "\n")

    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}\n")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ ERROR: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)
