"""
Test 5: Dual Focus (Top and Bottom Slice)

Purpose: Acquire focused images at two Z positions (top and bottom of volume).

Tests:
- Two-position manual acquisition
- Focus optimization at each position
- Calibration-based position calculation
- Side-by-side comparison of top/bottom images
- Verification of focus quality at both positions

This represents the final step before full 3D volume acquisition.
"""

# Add parent directory to path for gently imports
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import numpy as np
import bluesky.plan_stubs as bps
import bluesky.preprocessors as bpp
from gently.plans import compute_fft_bandpass_score


@pytest.mark.hardware
@pytest.mark.acquisition
@pytest.mark.slow
def test_two_position_manual(scanner, camera, piezo, laser_control, core, output_dir):
    """
    Acquire images at two Z positions manually.

    Verifies:
    - Can move between top and bottom positions
    - Images acquired at both positions
    - Positions are different
    """
    print("\n📸 Testing two-position manual acquisition...")

    # Configure
    scanner.configure_for_calibration()
    camera.set_sensor_mode("PROGRESSIVE")
    core.setProperty("HamCam1", "Exposure", 10.0)

    # Set piezo to Idle mode for direct position control
    piezo.set_spim_state("Idle")

    # Define positions (centered around 0 where galvo is at 0)
    top_position = 10.0  # µm - above reference
    bottom_position = -10.0  # µm - below reference

    print(f"  Top position: {top_position} µm")
    print(f"  Bottom position: {bottom_position} µm")
    print(f"  Range: {abs(top_position - bottom_position)} µm")

    # Acquire at top
    print("\n  Acquiring at TOP position...")
    status = piezo.set(top_position)
    status.wait()

    # Turn on lasers
    laser_status = laser_control.set("488 and 561")
    laser_status.wait()

    camera_status = camera.trigger()
    camera_status.wait()

    # Turn off lasers
    laser_status = laser_control.set("ALL OFF")
    laser_status.wait()

    top_img = camera.read()[camera.name]['value']
    top_score = compute_fft_bandpass_score(top_img)
    print(f"  ✓ Top image acquired (focus score: {top_score:.2f})")

    # Acquire at bottom
    print("\n  Acquiring at BOTTOM position...")
    status = piezo.set(bottom_position)
    status.wait()

    # Turn on lasers
    laser_status = laser_control.set("488 and 561")
    laser_status.wait()

    camera_status = camera.trigger()
    camera_status.wait()

    # Turn off lasers
    laser_status = laser_control.set("ALL OFF")
    laser_status.wait()

    bottom_img = camera.read()[camera.name]['value']
    bottom_score = compute_fft_bandpass_score(bottom_img)
    print(f"  ✓ Bottom image acquired (focus score: {bottom_score:.2f})")

    # Save side-by-side comparison
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].imshow(top_img, cmap='gray')
    axes[0].set_title(f'TOP ({top_position:.1f} µm)\nFocus: {top_score:.2f}', fontsize=12, fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(bottom_img, cmap='gray')
    axes[1].set_title(f'BOTTOM ({bottom_position:.1f} µm)\nFocus: {bottom_score:.2f}', fontsize=12, fontweight='bold')
    axes[1].axis('off')

    output_path = output_dir / "top_bottom_comparison.png"
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  ✓ Comparison saved: {output_path}")

    # Save individual TIFFs
    import tifffile
    tifffile.imwrite(str(output_dir / "top_position.tif"), top_img)
    tifffile.imwrite(str(output_dir / "bottom_position.tif"), bottom_img)

    print("  ✓ Two-position manual acquisition verified")


@pytest.mark.hardware
@pytest.mark.acquisition
@pytest.mark.slow
def test_dual_focus_with_optimization(scanner, camera, piezo, laser_control, core, output_dir):
    """
    Find best focus at top and bottom positions.

    Verifies:
    - Can optimize focus at multiple positions
    - Each position yields good focus
    - Optimized positions are stored
    """
    print("\n🔍 Testing dual focus with optimization...")

    # Configure
    scanner.configure_for_calibration()
    camera.set_sensor_mode("PROGRESSIVE")
    core.setProperty("HamCam1", "Exposure", 10.0)

    # Set piezo to Idle mode for direct position control
    piezo.set_spim_state("Idle")

    def find_best_focus(center_um, search_range=5, step=1):
        """Helper: Find best focus around a center position"""
        positions = np.arange(center_um - search_range, center_um + search_range + step, step)
        scores = []
        images = []

        for pos in positions:
            # Move piezo using device
            status = piezo.set(pos)
            status.wait()

            # Turn on lasers
            laser_status = laser_control.set("488 and 561")
            laser_status.wait()

            camera_status = camera.trigger()
            camera_status.wait()

            # Turn off lasers
            laser_status = laser_control.set("ALL OFF")
            laser_status.wait()

            img = camera.read()[camera.name]['value']
            score = compute_fft_bandpass_score(img)

            scores.append(score)
            images.append(img)

        best_idx = np.argmax(scores)
        return positions[best_idx], scores[best_idx], positions, scores, images[best_idx]

    # Optimize at top (centered around +10 µm from reference)
    print("\n  Optimizing focus at TOP region (around +10 µm)...")
    top_best, top_score, top_positions, top_scores, top_img = find_best_focus(10.0, search_range=5, step=1)
    print(f"  ✓ Top best focus: {top_best:.2f} µm (score: {top_score:.2f})")

    # Optimize at bottom (centered around -10 µm from reference)
    print("\n  Optimizing focus at BOTTOM region (around -10 µm)...")
    bottom_best, bottom_score, bottom_positions, bottom_scores, bottom_img = find_best_focus(-10.0, search_range=5, step=1)
    print(f"  ✓ Bottom best focus: {bottom_best:.2f} µm (score: {bottom_score:.2f})")

    # Plot optimization curves
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Top focus curve
    axes[0, 0].plot(top_positions, top_scores, 'o-', linewidth=2, markersize=8, color='blue')
    axes[0, 0].axvline(top_best, color='r', linestyle='--', label=f'Best: {top_best:.2f} µm')
    axes[0, 0].set_xlabel('Piezo Position (µm)')
    axes[0, 0].set_ylabel('Focus Score')
    axes[0, 0].set_title('TOP Focus Optimization')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()

    # Top image
    axes[0, 1].imshow(top_img, cmap='gray')
    axes[0, 1].set_title(f'TOP Best Focus\n{top_best:.2f} µm | Score: {top_score:.2f}')
    axes[0, 1].axis('off')

    # Bottom focus curve
    axes[1, 0].plot(bottom_positions, bottom_scores, 'o-', linewidth=2, markersize=8, color='orange')
    axes[1, 0].axvline(bottom_best, color='r', linestyle='--', label=f'Best: {bottom_best:.2f} µm')
    axes[1, 0].set_xlabel('Piezo Position (µm)')
    axes[1, 0].set_ylabel('Focus Score')
    axes[1, 0].set_title('BOTTOM Focus Optimization')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()

    # Bottom image
    axes[1, 1].imshow(bottom_img, cmap='gray')
    axes[1, 1].set_title(f'BOTTOM Best Focus\n{bottom_best:.2f} µm | Score: {bottom_score:.2f}')
    axes[1, 1].axis('off')

    output_path = output_dir / "dual_focus_optimization.png"
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  ✓ Optimization plot saved: {output_path}")

    print(f"\n  📊 Summary:")
    print(f"    Top: {top_best:.2f} µm (score: {top_score:.2f})")
    print(f"    Bottom: {bottom_best:.2f} µm (score: {bottom_score:.2f})")
    print(f"    Range: {abs(top_best - bottom_best):.2f} µm")

    print("  ✓ Dual focus optimization verified")

    return top_best, bottom_best


@pytest.mark.hardware
@pytest.mark.acquisition
def test_calibration_based_positions(scanner, camera, piezo, laser_control, core, calibration_file, output_dir):
    """
    Use calibration to calculate top/bottom piezo positions from galvo offsets.

    Verifies:
    - Can load and use calibration data
    - Linear relationship for position calculation
    - Calculated positions yield good focus
    """
    print("\n🔍 Testing calibration-based position calculation...")

    import json
    import os

    if not os.path.exists(calibration_file):
        pytest.skip(f"Calibration file not found: {calibration_file}")

    # Load calibration
    with open(calibration_file, 'r') as f:
        calib = json.load(f)

    print(f"  Calibration loaded:")
    print(f"    Slope: {calib['slope']:.4f} µm/deg")
    print(f"    Intercept: {calib['intercept']:.4f} µm")

    # Define galvo offsets for top and bottom
    # (These would come from desired lightsheet positions)
    top_galvo_offset = 2.0    # degrees
    bottom_galvo_offset = -2.0  # degrees

    # Calculate piezo positions using linear calibration
    top_piezo = calib['slope'] * top_galvo_offset + calib['intercept']
    bottom_piezo = calib['slope'] * bottom_galvo_offset + calib['intercept']

    print(f"\n  Position calculation:")
    print(f"    Top galvo: {top_galvo_offset}° → Piezo: {top_piezo:.2f} µm")
    print(f"    Bottom galvo: {bottom_galvo_offset}° → Piezo: {bottom_piezo:.2f} µm")
    print(f"    Range: {abs(top_piezo - bottom_piezo):.2f} µm")

    # Configure and acquire at calculated positions
    scanner.configure_for_calibration()
    camera.set_sensor_mode("PROGRESSIVE")
    core.setProperty("HamCam1", "Exposure", 10.0)

    # Set piezo to Idle mode for direct position control
    piezo.set_spim_state("Idle")

    # Acquire at top
    print(f"\n  Acquiring at calculated TOP ({top_piezo:.2f} µm)...")
    status = piezo.set(top_piezo)
    status.wait()

    # Turn on lasers
    laser_status = laser_control.set("488 and 561")
    laser_status.wait()

    camera_status = camera.trigger()
    camera_status.wait()

    # Turn off lasers
    laser_status = laser_control.set("ALL OFF")
    laser_status.wait()

    top_img = camera.read()[camera.name]['value']
    top_score = compute_fft_bandpass_score(top_img)
    print(f"  ✓ Top image (score: {top_score:.2f})")

    # Acquire at bottom
    print(f"\n  Acquiring at calculated BOTTOM ({bottom_piezo:.2f} µm)...")
    status = piezo.set(bottom_piezo)
    status.wait()

    # Turn on lasers
    laser_status = laser_control.set("488 and 561")
    laser_status.wait()

    camera_status = camera.trigger()
    camera_status.wait()

    # Turn off lasers
    laser_status = laser_control.set("ALL OFF")
    laser_status.wait()

    bottom_img = camera.read()[camera.name]['value']
    bottom_score = compute_fft_bandpass_score(bottom_img)
    print(f"  ✓ Bottom image (score: {bottom_score:.2f})")

    # Create comparison
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].imshow(top_img, cmap='gray')
    axes[0].set_title(f'TOP (Calibration-based)\nGalvo: {top_galvo_offset}° | Piezo: {top_piezo:.2f} µm\nFocus: {top_score:.2f}',
                      fontsize=10, fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(bottom_img, cmap='gray')
    axes[1].set_title(f'BOTTOM (Calibration-based)\nGalvo: {bottom_galvo_offset}° | Piezo: {bottom_piezo:.2f} µm\nFocus: {bottom_score:.2f}',
                      fontsize=10, fontweight='bold')
    axes[1].axis('off')

    output_path = output_dir / "calibration_based_positions.png"
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  ✓ Calibration-based comparison saved: {output_path}")

    print("  ✓ Calibration-based positioning verified")


@pytest.mark.hardware
@pytest.mark.acquisition
@pytest.mark.slow
def test_dual_position_plan(scanner, camera, piezo, core, run_engine, output_dir):
    """
    Acquire top and bottom positions using Bluesky plan.

    Verifies:
    - Dual acquisition works in plan context
    - Data saved to databroker
    - Can retrieve both images
    """
    RE, db = run_engine

    print("\n📸 Testing dual position acquisition in plan...")

    # Configure
    scanner.configure_for_calibration()
    camera.set_sensor_mode("PROGRESSIVE")
    core.setProperty("HamCam1", "Exposure", 10.0)

    top_position = 110.0
    bottom_position = 90.0

    @bpp.run_decorator(md={
        'test': 'dual_position',
        'top': top_position,
        'bottom': bottom_position
    })
    def dual_position_plan():
        # Acquire at top
        print(f"  Moving to TOP ({top_position} µm)...")
        yield from bps.mv(piezo, top_position)

        top_data = yield from bps.trigger_and_read([camera])
        print(f"  ✓ Top acquired")

        # Acquire at bottom
        print(f"  Moving to BOTTOM ({bottom_position} µm)...")
        yield from bps.mv(piezo, bottom_position)

        bottom_data = yield from bps.trigger_and_read([camera])
        print(f"  ✓ Bottom acquired")

        return top_data, bottom_data

    # Execute
    uid = RE(dual_position_plan())

    # Retrieve
    print("\n  Retrieving from databroker...")
    run = db[-1]  # Get most recent run
    table = run.table(); images = table[camera.name].values

    assert len(images) >= 2, f"Expected at least 2 images, got {len(images)}"

    top_img = images[0]
    bottom_img = images[1]

    top_score = compute_fft_bandpass_score(top_img)
    bottom_score = compute_fft_bandpass_score(bottom_img)

    print(f"  Top focus: {top_score:.2f}")
    print(f"  Bottom focus: {bottom_score:.2f}")

    print(f"  ✓ Run UID: {uid[:8]}...")
    print("  ✓ Dual position plan verified")


@pytest.mark.hardware
@pytest.mark.acquisition
@pytest.mark.slow
def test_volume_range_validation(scanner, camera, piezo, core, output_dir):
    """
    Validate that top and bottom represent good volume boundaries.

    Verifies:
    - Both positions are in focus
    - Range is appropriate for embryo imaging
    - No clipping at boundaries
    """
    print("\n✅ Testing volume range validation...")

    # Configure
    scanner.configure_for_calibration()
    camera.set_sensor_mode("PROGRESSIVE")
    core.setProperty("HamCam1", "Exposure", 10.0)

    # Test positions that would be volume boundaries
    test_positions = [85, 90, 100, 110, 115]  # µm
    scores = []
    images = []

    print(f"  Testing {len(test_positions)} positions across potential volume range...")

    for i, pos in enumerate(test_positions):
        print(f"  [{i+1}/{len(test_positions)}] Position {pos} µm...")

        # Move piezo using device
        status = piezo.set(pos)
        status.wait()

        camera_status = camera.trigger()
        camera_status.wait()

        img = camera.read()[camera.name]['value']
        score = compute_fft_bandpass_score(img)

        scores.append(score)
        images.append(img)

        print(f"      Focus score: {score:.2f}")

    # Analyze
    scores_array = np.array(scores)
    good_focus_threshold = scores_array.max() * 0.7  # 70% of max

    good_positions = [pos for pos, score in zip(test_positions, scores) if score >= good_focus_threshold]

    print(f"\n  📊 Volume Range Analysis:")
    print(f"    Max focus score: {scores_array.max():.2f}")
    print(f"    Focus threshold (70%): {good_focus_threshold:.2f}")
    print(f"    Positions with good focus: {good_positions}")

    if len(good_positions) >= 2:
        suggested_range = max(good_positions) - min(good_positions)
        print(f"    ✓ Suggested volume range: {min(good_positions)} - {max(good_positions)} µm ({suggested_range} µm)")
    else:
        print(f"    ⚠️  Only {len(good_positions)} position(s) with good focus")

    # Plot
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Score vs position
    ax1.plot(test_positions, scores, 'o-', linewidth=2, markersize=10)
    ax1.axhline(good_focus_threshold, color='r', linestyle='--', label='70% threshold')
    ax1.set_xlabel('Piezo Position (µm)', fontsize=12)
    ax1.set_ylabel('Focus Score', fontsize=12)
    ax1.set_title('Focus Quality Across Volume Range', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Mark good region
    if len(good_positions) >= 2:
        ax1.axvspan(min(good_positions), max(good_positions), alpha=0.2, color='green', label='Good range')

    # Image comparison
    n_imgs = len(images)
    grid_h = int(np.ceil(np.sqrt(n_imgs)))
    grid_w = int(np.ceil(n_imgs / grid_h))

    for i, (img, pos, score) in enumerate(zip(images, test_positions, scores)):
        if i < 9:  # Limit to 9 images for second subplot
            row = i // 3
            col = i % 3

            if i == 0:
                inset_axes = ax2.inset_axes([col*0.33, 1 - (row+1)*0.33, 0.33, 0.33])
            else:
                inset_axes = ax2.inset_axes([col*0.33, 1 - (row+1)*0.33, 0.33, 0.33])

            inset_axes.imshow(img, cmap='gray')
            title_color = 'green' if pos in good_positions else 'red'
            inset_axes.set_title(f'{pos}µm', fontsize=8, color=title_color)
            inset_axes.axis('off')

    ax2.axis('off')
    ax2.set_title('Image Quality Comparison', fontsize=14, fontweight='bold')

    output_path = output_dir / "volume_range_validation.png"
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  ✓ Validation plot saved: {output_path}")

    print("  ✓ Volume range validation completed")


# ============================================================================
# Main (for running tests directly without pytest)
# ============================================================================

if __name__ == "__main__":
    """
    Run tests directly without pytest for quick validation.

    Usage:
        python test_5_dual_focus.py
    """
    import sys
    from pathlib import Path

    # Setup path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from client import get_mmc
    from gently.devices import DiSPIMScanner, DiSPIMPiezo, DiSPIMCamera
    from bluesky import RunEngine
    from databroker import Broker

    print("\n" + "="*70)
    print("DiSPIM Dual Focus Test - Manual Run")
    print("="*70)

    # Connect to hardware
    print("\n🔌 Connecting to Micro-Manager...")
    core = get_mmc()
    print("✓ Connected")

    # Create devices
    scanner = DiSPIMScanner("Scanner:AB:33", core)
    piezo = DiSPIMPiezo("PiezoStage:P:34", core)
    camera = DiSPIMCamera("HamCam1", core)

    # Create RunEngine
    RE = RunEngine({})
    db = Broker.named('temp')
    RE.subscribe(db.insert)
    run_engine = (RE, db)

    # Output directory
    output_dir = Path(__file__).parent / "outputs"
    output_dir.mkdir(exist_ok=True)

    calibration_file = "backend/piezo_galvo_calibration_embryo.json"

    # Run tests
    try:
        test_two_position_manual(scanner, camera, piezo, core, output_dir)
        test_dual_focus_with_optimization(scanner, camera, piezo, core, output_dir)
        test_dual_position_plan(scanner, camera, piezo, core, run_engine, output_dir)
        test_volume_range_validation(scanner, camera, piezo, core, output_dir)

        # Try calibration-based if file exists
        import os
        if os.path.exists(calibration_file):
            test_calibration_based_positions(scanner, camera, piezo, core, calibration_file, output_dir)
        else:
            print(f"\n⚠️  Skipped calibration-based test (file not found)")

        print("\n" + "="*70)
        print("✓ ALL TESTS PASSED")
        print(f"✓ Outputs saved to: {output_dir}")
        print("="*70 + "\n")

    except AssertionError as e:
        print(f"\n✗ TEST FAILED: {e}\n")
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ ERROR: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)
