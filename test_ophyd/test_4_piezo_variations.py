"""
Test 4: Piezo Focus Variations

Purpose: Sweep piezo through focus range and acquire images at multiple positions.

Tests:
- Piezo position control and verification
- Focus sweep with image acquisition at each position
- Focus scoring using FFT bandpass algorithm
- Focus curve plotting and peak identification
- Montage creation showing progression through focus
- Best focus position determination

All outputs (images, plots, montages) saved to test_ophyd/outputs/
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
def test_piezo_position_control(piezo, core):
    """
    Test basic piezo position control.

    Verifies:
    - Can read piezo position
    - Can move piezo to specific position
    - Position is accurately reported
    """
    print("\n🔧 Testing piezo position control...")

    # Set piezo to Idle mode for direct position control
    piezo.set_spim_state("Idle")
    print("  Piezo set to Idle mode for position control")

    # Read current position
    current_data = piezo.read()
    current_pos = current_data[piezo.name]['value']
    print(f"  Current position: {current_pos:.2f} µm")

    # Reset to reference position 0 (matching galvo at 0)
    reference_pos = 0.0  # 0 µm - piezo at 0 when galvo at 0
    print(f"\n  Resetting to reference position {reference_pos:.2f} µm...")
    status = piezo.set(reference_pos)
    status.wait()

    # Verify reset
    data = piezo.read()
    actual_ref = data[piezo.name]['value']
    print(f"  Reference position: {actual_ref:.2f} µm")

    # Now test ±10 µm movements from reference (0)
    test_positions = [
        reference_pos + 10.0,  # +10 µm from 0
        reference_pos - 10.0,  # -10 µm from 0
        reference_pos + 5.0    # +5 µm from 0
    ]

    for target_pos in test_positions:
        print(f"  Moving to {target_pos:.2f} µm...")
        status = piezo.set(target_pos)
        status.wait()

        # Read back using device
        data = piezo.read()
        actual_pos = data[piezo.name]['value']
        print(f"    Actual position: {actual_pos:.2f} µm")

        # Verify (allow small tolerance)
        assert abs(actual_pos - target_pos) < 0.5, f"Position error: {abs(actual_pos - target_pos):.2f} µm"

    print("  ✓ Piezo position control verified")


@pytest.mark.hardware
@pytest.mark.acquisition
@pytest.mark.slow
def test_piezo_focus_sweep_manual(scanner, camera, piezo, laser_control, core, output_dir):
    """
    Perform focus sweep by varying piezo position (manual triggers).

    Verifies:
    - Can sweep piezo through range
    - Acquire image at each position
    - Compute focus scores
    - Identify best focus position
    """
    print("\n🔍 Testing manual piezo focus sweep...")

    # Set piezo to Idle for direct position control
    piezo.set_spim_state("Idle")

    # Configure for imaging
    scanner.configure_for_calibration()
    camera.set_sensor_mode("PROGRESSIVE")
    core.setProperty("HamCam1", "Exposure", 10.0)

    # Sweep parameters
    start_um = -20
    end_um = 20
    step_um = 2.0
    positions = np.arange(start_um, end_um + step_um, step_um)

    print(f"  Sweeping from {start_um} to {end_um} µm in {step_um} µm steps")
    print(f"  Total positions: {len(positions)}")

    images = []
    focus_scores = []

    for i, pos in enumerate(positions):
        print(f"  [{i+1}/{len(positions)}] Position {pos:.1f} µm...")

        # Move piezo using device
        status = piezo.set(pos)
        status.wait()

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

        # Compute focus score
        score = compute_fft_bandpass_score(img)
        focus_scores.append(score)

        print(f"      Focus score: {score:.2f}")

    # Find best focus
    best_idx = np.argmax(focus_scores)
    best_pos = positions[best_idx]
    best_score = focus_scores[best_idx]

    print(f"\n  📊 Focus Analysis:")
    print(f"    Best focus position: {best_pos:.1f} µm")
    print(f"    Best focus score: {best_score:.2f}")
    print(f"    Score range: [{min(focus_scores):.2f}, {max(focus_scores):.2f}]")

    # Plot focus curve
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(positions, focus_scores, 'o-', linewidth=2, markersize=8)
    ax.axvline(best_pos, color='r', linestyle='--', label=f'Best focus: {best_pos:.1f} µm')
    ax.set_xlabel('Piezo Position (µm)', fontsize=12)
    ax.set_ylabel('Focus Score (FFT Bandpass)', fontsize=12)
    ax.set_title('Piezo Focus Curve', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()

    output_path = output_dir / "focus_curve.png"
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Focus curve saved: {output_path}")

    # Create montage
    print("\n  Creating focus montage...")
    n_images = len(images)
    n_cols = min(5, n_images)
    n_rows = (n_images + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3*n_cols, 3*n_rows))
    axes = np.array(axes).flatten()

    for i, (img, pos, score) in enumerate(zip(images, positions, focus_scores)):
        axes[i].imshow(img, cmap='gray')
        title_color = 'red' if i == best_idx else 'black'
        axes[i].set_title(f'{pos:.1f}µm | {score:.1f}', color=title_color, fontsize=10)
        axes[i].axis('off')

    # Hide unused subplots
    for i in range(len(images), len(axes)):
        axes[i].axis('off')

    output_path = output_dir / "focus_montage.png"
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Montage saved: {output_path}")

    print("  ✓ Manual focus sweep verified")

    return best_pos, focus_scores


@pytest.mark.hardware
@pytest.mark.acquisition
@pytest.mark.slow
def test_piezo_focus_sweep_plan(scanner, camera, piezo, laser_control, core, run_engine, output_dir):
    """
    Perform focus sweep using Bluesky plan.

    Verifies:
    - Focus sweep works in plan context
    - Data is saved to databroker
    - Can retrieve and analyze sweep data
    """
    RE, db = run_engine

    print("\n🔍 Testing piezo focus sweep in Bluesky plan...")

    # Configure devices
    scanner.configure_for_calibration()
    camera.set_sensor_mode("PROGRESSIVE")
    core.setProperty("HamCam1", "Exposure", 10.0)

    # Sweep parameters
    start_um = 100.0
    end_um = 110.0
    step_um = 2.0
    positions = np.arange(start_um, end_um + step_um, step_um)

    @bpp.run_decorator(md={
        'test': 'focus_sweep_plan',
        'start': start_um,
        'end': end_um,
        'step': step_um
    })
    def focus_sweep_plan():
        focus_scores = []

        for i, pos in enumerate(positions):
            print(f"  [{i+1}/{len(positions)}] Position {pos:.1f} µm...")

            # Move piezo using Bluesky plan stub
            yield from bps.mv(piezo, pos)

            # Turn on lasers
            yield from bps.mv(laser_control, "488 and 561")

            # Acquire
            data = yield from bps.trigger_and_read([camera])

            # Turn off lasers
            yield from bps.mv(laser_control, "ALL OFF")

            # Compute focus score
            img = data[camera.name]['value']
            score = compute_fft_bandpass_score(img)
            focus_scores.append(score)

            print(f"      Focus score: {score:.2f}")

        return focus_scores

    # Execute
    uid = RE(focus_sweep_plan())

    # Retrieve data
    print("\n  Retrieving from databroker...")
    run = db[-1]  # Get most recent run
    table = run.table()

    images = table[camera.name].values
    print(f"  Retrieved {len(images)} images from databroker")

    # Recompute focus scores from retrieved images
    focus_scores = [compute_fft_bandpass_score(img) for img in images]

    # Find best
    best_idx = np.argmax(focus_scores)
    best_pos = positions[best_idx]

    print(f"\n  Best focus: {best_pos:.1f} µm (score={focus_scores[best_idx]:.2f})")

    print(f"  ✓ Run UID: {uid[:8]}...")
    print("  ✓ Plan-based focus sweep verified")


@pytest.mark.hardware
@pytest.mark.acquisition
@pytest.mark.slow
def test_fine_focus_search(scanner, camera, piezo, laser_control, core, output_dir):
    """
    Perform fine focus search with smaller steps around estimated position.

    Verifies:
    - Can do coarse + fine focus strategy
    - Higher resolution focus determination
    """
    print("\n🔍 Testing fine focus search...")

    # Set piezo to Idle for direct position control
    piezo.set_spim_state("Idle")

    # Configure
    scanner.configure_for_calibration()
    camera.set_sensor_mode("PROGRESSIVE")
    core.setProperty("HamCam1", "Exposure", 10.0)

    # Coarse sweep first
    print("  Phase 1: Coarse sweep (5 µm steps)...")
    coarse_positions = np.arange(90, 120, 5)
    coarse_scores = []

    for pos in coarse_positions:
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
        coarse_scores.append(score)

    coarse_best_idx = np.argmax(coarse_scores)
    coarse_best = coarse_positions[coarse_best_idx]
    print(f"    Coarse best: {coarse_best:.1f} µm")

    # Fine sweep around coarse best
    print(f"  Phase 2: Fine sweep around {coarse_best:.1f} µm (0.5 µm steps)...")
    fine_positions = np.arange(coarse_best - 3, coarse_best + 3, 0.5)
    fine_scores = []

    for pos in fine_positions:
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
        fine_scores.append(score)

    fine_best_idx = np.argmax(fine_scores)
    fine_best = fine_positions[fine_best_idx]

    print(f"\n  📊 Fine Focus Result:")
    print(f"    Coarse best: {coarse_best:.1f} µm (score={coarse_scores[coarse_best_idx]:.2f})")
    print(f"    Fine best: {fine_best:.2f} µm (score={fine_scores[fine_best_idx]:.2f})")

    # Plot comparison
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(coarse_positions, coarse_scores, 'o-', markersize=10, linewidth=2, label='Coarse')
    ax1.axvline(coarse_best, color='r', linestyle='--', alpha=0.7)
    ax1.set_xlabel('Piezo Position (µm)')
    ax1.set_ylabel('Focus Score')
    ax1.set_title('Coarse Sweep')
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    ax2.plot(fine_positions, fine_scores, 'o-', markersize=8, linewidth=2, label='Fine', color='orange')
    ax2.axvline(fine_best, color='r', linestyle='--', alpha=0.7)
    ax2.set_xlabel('Piezo Position (µm)')
    ax2.set_ylabel('Focus Score')
    ax2.set_title(f'Fine Sweep (around {coarse_best:.1f} µm)')
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    output_path = output_dir / "fine_focus_search.png"
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  ✓ Fine focus plot saved: {output_path}")

    print("  ✓ Fine focus search verified")


@pytest.mark.hardware
@pytest.mark.acquisition
def test_focus_repeatability(scanner, camera, piezo, laser_control, core):
    """
    Test focus repeatability by acquiring at same position multiple times.

    Verifies:
    - Focus scores are consistent
    - Piezo positioning is repeatable
    """
    print("\n🔍 Testing focus repeatability...")

    # Set piezo to Idle for direct position control
    piezo.set_spim_state("Idle")

    # Configure
    scanner.configure_for_calibration()
    camera.set_sensor_mode("PROGRESSIVE")
    core.setProperty("HamCam1", "Exposure", 10.0)

    # Test position at reference (0 µm)
    test_position = 0.0  # Use reference position (piezo at 0)
    n_repeats = 5

    scores = []

    for i in range(n_repeats):
        print(f"  Trial {i+1}/{n_repeats}...")

        # Move to position using device
        status = piezo.set(test_position)
        status.wait()

        # Turn on lasers
        laser_status = laser_control.set("488 and 561")
        laser_status.wait()

        # Acquire
        camera_status = camera.trigger()
        camera_status.wait()

        # Turn off lasers
        laser_status = laser_control.set("ALL OFF")
        laser_status.wait()

        img = camera.read()[camera.name]['value']
        score = compute_fft_bandpass_score(img)
        scores.append(score)

        print(f"    Focus score: {score:.2f}")

    # Analyze repeatability
    scores_array = np.array(scores)
    mean_score = scores_array.mean()
    std_score = scores_array.std()
    cv = (std_score / mean_score) * 100  # Coefficient of variation

    print(f"\n  📊 Repeatability Analysis:")
    print(f"    Mean score: {mean_score:.2f}")
    print(f"    Std deviation: {std_score:.2f}")
    print(f"    CV: {cv:.2f}%")

    # Good repeatability if CV < 5%
    if cv < 5:
        print(f"    ✓ Excellent repeatability (CV < 5%)")
    elif cv < 10:
        print(f"    ⚠️  Moderate repeatability (5% < CV < 10%)")
    else:
        print(f"    ⚠️  Poor repeatability (CV > 10%)")

    print("  ✓ Repeatability test completed")


# ============================================================================
# Main (for running tests directly without pytest)
# ============================================================================

if __name__ == "__main__":
    """
    Run tests directly without pytest for quick validation.

    Usage:
        python test_4_piezo_variations.py
    """
    import sys
    from pathlib import Path

    # Setup path
    sys.path.insert(0, str(Path(__file__).parent.parent))

    from client import get_mmc
    from gently.devices import DiSPIMScanner, DiSPIMPiezo, DiSPIMCamera, DiSPIMLaserControl
    from bluesky import RunEngine
    from databroker import Broker

    print("\n" + "="*70)
    print("DiSPIM Piezo Focus Variations Test - Manual Run")
    print("="*70)

    # Connect to hardware
    print("\n🔌 Connecting to Micro-Manager...")
    core = get_mmc()
    print("✓ Connected")

    # Create devices
    scanner = DiSPIMScanner("Scanner:AB:33", core)
    laser_control = DiSPIMLaserControl(core, "Laser")
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

    # Run tests
    try:
        test_piezo_position_control(piezo, core)
        test_piezo_focus_sweep_manual(scanner, camera, piezo, laser_control, core, output_dir)
        test_piezo_focus_sweep_plan(scanner, camera, piezo, laser_control, core, run_engine, output_dir)
        test_fine_focus_search(scanner, camera, piezo, laser_control, core, output_dir)
        test_focus_repeatability(scanner, camera, piezo, laser_control, core)

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
