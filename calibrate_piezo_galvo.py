#!/usr/bin/env python3
"""
Piezo-Galvo 2-Point Calibration for ASI diSPIM

This script performs the critical calibration that synchronizes the objective piezo
focus with the galvo Y-axis light sheet position. The calibration establishes the
linear relationship: piezo_position (µm) = slope × galvo_angle (°) + offset

WORKFLOW:
1. Move galvo Y to TOP position → light sheet deflects up
2. Automated focus sweep: move piezo, capture images, find best focus
3. Manual refinement: display best image, allow adjustment
4. Record piezo position for TOP

5. Move galvo Y to BOTTOM position → light sheet deflects down
6. Repeat automated sweep + manual refinement
7. Record piezo position for BOTTOM

8. Calculate calibration: slope and offset
9. Save to JSON config file for use by acquisition scripts

Based on ASI diSPIM plugin calibration procedure.
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import json
from pathlib import Path
from client import get_mmc

# Device configuration
core = get_mmc()
CAMERA_NAME = "HamCam1"
GALVO_DEVICE = "Scanner:AB:33"
PIEZO_DEVICE = "PiezoStage:P:34"

# Calibration parameters
GALVO_Y_TOP = 0.3        # degrees - top of scan range (reduced to stay in ROI)
GALVO_Y_BOTTOM = -0.3    # degrees - bottom of scan range (reduced to stay in ROI)
GALVO_Y_CENTER = 0.0     # degrees - center position

# Focus sweep parameters
PIEZO_SWEEP_START = 20.0  # µm - starting position for focus search
PIEZO_SWEEP_END = 80.0    # µm - ending position for focus search
PIEZO_SWEEP_STEP = 2.0    # µm - step size during coarse sweep
PIEZO_FINE_STEP = 0.5     # µm - step size for manual refinement

# Camera settings
CAMERA_EXPOSURE_MS = 10.0  # milliseconds

# Output configuration
CALIBRATION_FILE = Path("piezo_galvo_calibration.json")
SAVE_IMAGES = True
IMAGE_DIR = Path("calibration_images")


def variance_of_laplacian(image):
    """
    Calculate focus metric using Laplacian variance.
    Higher values = sharper/more in-focus.

    This is a standard focus quality metric that responds to edge sharpness.
    """
    # Convert to float for calculation
    img_float = image.astype(np.float64)

    # Apply Laplacian kernel
    laplacian = np.array([[0, 1, 0],
                         [1, -4, 1],
                         [0, 1, 0]], dtype=np.float64)

    # Convolve
    from scipy import ndimage
    filtered = ndimage.convolve(img_float, laplacian)

    # Return variance
    return filtered.var()


def gradient_magnitude(image):
    """
    Alternative focus metric using gradient magnitude.
    Higher values = sharper/more in-focus.
    """
    # Convert to float
    img_float = image.astype(np.float64)

    # Sobel gradients
    from scipy import ndimage
    gx = ndimage.sobel(img_float, axis=0)
    gy = ndimage.sobel(img_float, axis=1)

    # Magnitude
    magnitude = np.sqrt(gx**2 + gy**2)

    return magnitude.mean()


def configure_camera():
    """Configure camera for internal trigger mode (not hardware triggered)."""
    print(f"\nConfiguring camera: {CAMERA_NAME}")

    core.setCameraDevice(CAMERA_NAME)

    # Set camera ROI to match ASI diSPIM plugin custom ROI
    # This creates a horizontal strip optimized for light sheet imaging
    roi_x = 128
    roi_y = 896
    roi_width = 2048
    roi_height = 512

    print(f"  Setting camera ROI: X={roi_x}, Y={roi_y}, W={roi_width}, H={roi_height}")
    core.setROI(CAMERA_NAME, roi_x, roi_y, roi_width, roi_height)

    # Verify ROI was set
    import rpyc
    actual_roi = rpyc.classic.obtain(core.getROI(CAMERA_NAME))
    print(f"  Actual ROI: X={actual_roi[0]}, Y={actual_roi[1]}, W={actual_roi[2]}, H={actual_roi[3]}")

    # Configure trigger and exposure
    core.setProperty(CAMERA_NAME, "TRIGGER SOURCE", "INTERNAL")
    core.setProperty(CAMERA_NAME, "SENSOR MODE", "AREA")
    core.setExposure(CAMERA_NAME, CAMERA_EXPOSURE_MS)

    time.sleep(0.1)

    trigger_source = core.getProperty(CAMERA_NAME, "TRIGGER SOURCE")
    exposure = core.getExposure(CAMERA_NAME)

    print(f"  TRIGGER SOURCE: {trigger_source}")
    print(f"  Exposure: {exposure} ms")

    assert trigger_source == "INTERNAL", "Camera not in INTERNAL trigger mode!"


def capture_image():
    """Capture a single image from the camera."""
    core.snapImage()
    img = core.getImage()

    # Handle remote core (rpyc)
    try:
        import rpyc
        img = rpyc.classic.obtain(img)
    except (ImportError, AttributeError):
        pass

    return img


def configure_galvo_for_calibration():
    """
    Configure galvo X-axis for light sheet generation during calibration.
    The X-axis creates the light sheet, Y-axis positions it in Z.
    """
    print(f"\n  Configuring galvo for light sheet generation...")

    # Ensure beam is enabled for light sheet generation
    core.setProperty(GALVO_DEVICE, "BeamEnabled", "Yes")

    # Configure X-axis for light sheet width (scanning)
    core.setProperty(GALVO_DEVICE, "SingleAxisXAmplitude(deg)", 2.0)
    core.setProperty(GALVO_DEVICE, "SingleAxisXOffset(deg)", -0.5)  # Optimized offset
    core.setProperty(GALVO_DEVICE, "SingleAxisXPattern", "1 - Triangle")
    core.setProperty(GALVO_DEVICE, "SingleAxisXMode", "3 - Enabled with axes synced")

    # Configure Y-axis with minimal amplitude (will be adjusted per calibration point)
    core.setProperty(GALVO_DEVICE, "SingleAxisYAmplitude(deg)", 0.0001)  # Minimal
    core.setProperty(GALVO_DEVICE, "SingleAxisYOffset(deg)", 0.0)  # Will be set later
    core.setProperty(GALVO_DEVICE, "SingleAxisYPattern", "1 - Triangle")
    core.setProperty(GALVO_DEVICE, "SingleAxisYMode", "3 - Enabled with axes synced")

    time.sleep(0.3)
    print(f"  ✓ Galvo configured for light sheet (X scanning, Y positioning)")


def set_galvo_y_position(angle_deg):
    """
    Set galvo Y-axis offset to position the light sheet at a specific Z-plane.

    The galvo Y-axis controls the axial (Z) position of the light sheet by
    changing the deflection angle. We position it by setting the SingleAxisYOffset
    property directly, not using the setGalvoPosition() API.

    When Y-axis is in scanning mode (non-zero amplitude), the offset defines the
    CENTER of the scan. For calibration, we use minimal amplitude and just position
    the light sheet at fixed angles.
    """
    print(f"\n  Setting galvo Y offset to {angle_deg:.3f}°")

    # Directly set the Y-axis offset property
    # This moves the light sheet to the desired Z-plane
    core.setProperty(GALVO_DEVICE, "SingleAxisYOffset(deg)", angle_deg)

    time.sleep(0.3)  # Allow galvo to settle

    # Read back the offset to verify
    actual_offset = float(core.getProperty(GALVO_DEVICE, "SingleAxisYOffset(deg)"))
    print(f"  Galvo Y offset set to: {actual_offset:.3f}°")


def focus_sweep_automated(galvo_position_name, galvo_angle,
                          sweep_start, sweep_end, sweep_step):
    """
    Automated focus sweep: move piezo through range and find best focus.

    Returns:
        tuple: (best_piezo_position, best_image, focus_curve_data)
    """
    print(f"\n{'='*70}")
    print(f"AUTOMATED FOCUS SWEEP - {galvo_position_name.upper()}")
    print(f"{'='*70}")
    print(f"Galvo Y: {galvo_angle:.3f}°")
    print(f"Piezo sweep: {sweep_start} to {sweep_end} µm (step {sweep_step} µm)")

    # Set piezo as focus device so we can use setPosition()
    core.setFocusDevice(PIEZO_DEVICE)
    print(f"  Set focus device: {PIEZO_DEVICE}")

    # Set galvo position
    set_galvo_y_position(galvo_angle)

    # Generate sweep positions
    positions = np.arange(sweep_start, sweep_end + sweep_step/2, sweep_step)
    print(f"\nScanning {len(positions)} positions...")

    # Storage for results
    focus_scores_laplacian = []
    focus_scores_gradient = []
    images = []

    # Sweep through positions
    for i, pos in enumerate(positions):
        # Move piezo using setPosition (works since we set it as focus device)
        # Convert to Python float for rpyc compatibility
        core.setPosition(float(pos))
        core.waitForDevice(PIEZO_DEVICE)
        time.sleep(0.1)  # Allow settling

        # Capture image
        img = capture_image()
        images.append(img)

        # Calculate focus metrics
        score_laplacian = variance_of_laplacian(img)
        score_gradient = gradient_magnitude(img)

        focus_scores_laplacian.append(score_laplacian)
        focus_scores_gradient.append(score_gradient)

        # Progress update
        if i % 5 == 0 or i == len(positions) - 1:
            print(f"  [{i+1}/{len(positions)}] Piezo {pos:.1f} µm: "
                  f"Laplacian={score_laplacian:.1f}, Gradient={score_gradient:.1f}")

    # Find best focus (using Laplacian as primary metric)
    best_idx = np.argmax(focus_scores_laplacian)
    best_position = positions[best_idx]
    best_image = images[best_idx]
    best_score = focus_scores_laplacian[best_idx]

    print(f"\n  Best focus found:")
    print(f"    Piezo position: {best_position:.2f} µm")
    print(f"    Focus score: {best_score:.1f}")

    # Plot focus curve
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    ax1.plot(positions, focus_scores_laplacian, 'b.-', label='Laplacian Variance')
    ax1.axvline(best_position, color='r', linestyle='--', label=f'Best: {best_position:.2f} µm')
    ax1.set_xlabel('Piezo Position (µm)')
    ax1.set_ylabel('Focus Score (Laplacian)')
    ax1.set_title(f'Focus Curve - {galvo_position_name} (Galvo Y = {galvo_angle:.3f}°)')
    ax1.legend()
    ax1.grid(True)

    ax2.imshow(best_image, cmap='gray')
    ax2.set_title(f'Best Focus Image (Piezo = {best_position:.2f} µm)')
    ax2.axis('off')

    plt.tight_layout()
    plt.show(block=False)
    plt.pause(0.1)

    focus_data = {
        'positions': positions.tolist(),
        'scores_laplacian': focus_scores_laplacian,
        'scores_gradient': focus_scores_gradient,
        'best_index': int(best_idx),
        'best_position': float(best_position)
    }

    return best_position, best_image, focus_data


def manual_refinement(initial_position, galvo_position_name):
    """
    Allow user to manually refine the focus position using napari.

    Returns:
        float: Final confirmed piezo position
    """
    print(f"\n{'='*70}")
    print(f"MANUAL FOCUS - {galvo_position_name.upper()}")
    print(f"{'='*70}")
    print(f"Starting position: {initial_position:.2f} µm")
    print(f"\nControls:")
    print(f"  w/s: Move up/down by {PIEZO_FINE_STEP} µm")
    print(f"  W/S: Move up/down by {PIEZO_FINE_STEP * 5} µm")
    print(f"  r: Refresh image at current position")
    print(f"  Enter: Confirm position")
    print(f"  q: Quit without saving")

    current_position = float(initial_position)
    # Piezo is already set as focus device
    core.setPosition(current_position)
    core.waitForDevice(PIEZO_DEVICE)
    time.sleep(0.2)

    # Create napari viewer
    try:
        import napari
        viewer = napari.Viewer(title=f"Piezo-Galvo Calibration - {galvo_position_name.upper()}")

        def update_image():
            """Capture and display image with focus score"""
            img = capture_image()
            score = variance_of_laplacian(img)

            # Update or create image layer
            if len(viewer.layers) == 0:
                viewer.add_image(img, name='Live', colormap='gray',
                               contrast_limits=[np.percentile(img, 1), np.percentile(img, 99)])
            else:
                viewer.layers[0].data = img
                viewer.layers[0].contrast_limits = [np.percentile(img, 1), np.percentile(img, 99)]

            # Update window title with position and score
            viewer.title = f"{galvo_position_name.upper()} - Piezo: {current_position:.2f} µm, Focus: {score:.1f}"

            return score

        # Initial display
        print(f"\nCapturing initial image...")
        update_image()
        print(f"Napari viewer opened. Use w/s to adjust focus.\n")

        while True:
            # Get user input
            key = input(f"Position: {current_position:.2f} µm [w/s/W/S/r/Enter/q]: ").strip()

            if key == 'w':
                current_position += PIEZO_FINE_STEP
                print(f"  Moving UP to {current_position:.2f} µm...")
            elif key == 's':
                current_position -= PIEZO_FINE_STEP
                print(f"  Moving DOWN to {current_position:.2f} µm...")
            elif key == 'W':
                current_position += PIEZO_FINE_STEP * 5
                print(f"  Moving UP (large) to {current_position:.2f} µm...")
            elif key == 'S':
                current_position -= PIEZO_FINE_STEP * 5
                print(f"  Moving DOWN (large) to {current_position:.2f} µm...")
            elif key == 'r':
                print(f"  Refreshing image at {current_position:.2f} µm...")
                update_image()
                continue
            elif key == '' or key.lower() == 'enter':
                print(f"\n  ✓ Confirmed position: {current_position:.2f} µm")
                viewer.close()
                return current_position
            elif key == 'q':
                print("\n  Cancelled")
                viewer.close()
                return None
            else:
                print(f"  Unknown command: '{key}'. Use w/s/W/S/r/Enter/q")
                continue

            # Move piezo and update display
            core.setPosition(current_position)
            core.waitForDevice(PIEZO_DEVICE)
            time.sleep(0.2)  # Allow settling
            score = update_image()
            print(f"  → Focus score: {score:.1f}")

    except ImportError:
        print("ERROR: napari not available. Please install: pip install napari[all]")
        return None


def calculate_calibration(galvo_top, piezo_top, galvo_bottom, piezo_bottom):
    """
    Calculate 2-point linear calibration.

    Formula: piezo_position = slope × galvo_angle + offset

    Returns:
        dict: Calibration parameters
    """
    print(f"\n{'='*70}")
    print("CALCULATING CALIBRATION")
    print(f"{'='*70}")

    # Calculate slope (µm/°)
    slope = (piezo_top - piezo_bottom) / (galvo_top - galvo_bottom)

    # Calculate offset using center point
    galvo_center = (galvo_top + galvo_bottom) / 2
    piezo_center = (piezo_top + piezo_bottom) / 2
    offset = piezo_center - (slope * galvo_center)

    print(f"\nCalibration points:")
    print(f"  TOP:    Galvo Y = {galvo_top:+.3f}° → Piezo = {piezo_top:.2f} µm")
    print(f"  BOTTOM: Galvo Y = {galvo_bottom:+.3f}° → Piezo = {piezo_bottom:.2f} µm")
    print(f"\nCalibration formula:")
    print(f"  piezo_position (µm) = {slope:.3f} × galvo_angle (°) + {offset:.3f}")
    print(f"\nParameters:")
    print(f"  Slope:  {slope:.3f} µm/°")
    print(f"  Offset: {offset:.3f} µm")

    # Verify formula at calibration points
    piezo_top_check = slope * galvo_top + offset
    piezo_bottom_check = slope * galvo_bottom + offset
    print(f"\nVerification:")
    print(f"  TOP:    {piezo_top:.2f} µm (measured) vs {piezo_top_check:.2f} µm (formula)")
    print(f"  BOTTOM: {piezo_bottom:.2f} µm (measured) vs {piezo_bottom_check:.2f} µm (formula)")

    calibration = {
        'slope_um_per_deg': float(slope),
        'offset_um': float(offset),
        'galvo_top_deg': float(galvo_top),
        'galvo_bottom_deg': float(galvo_bottom),
        'piezo_top_um': float(piezo_top),
        'piezo_bottom_um': float(piezo_bottom),
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'device_piezo': PIEZO_DEVICE,
        'device_galvo': GALVO_DEVICE
    }

    return calibration


def save_calibration(calibration, filename=CALIBRATION_FILE):
    """Save calibration to JSON file."""
    with open(filename, 'w') as f:
        json.dump(calibration, f, indent=2)
    print(f"\n✓ Calibration saved to: {filename}")


def load_previous_calibration(filename=CALIBRATION_FILE):
    """Load previous calibration if exists."""
    if filename.exists():
        with open(filename, 'r') as f:
            cal = json.load(f)
        print(f"\nPrevious calibration found:")
        print(f"  Date: {cal['timestamp']}")
        print(f"  Slope: {cal['slope_um_per_deg']:.3f} µm/°")
        print(f"  Offset: {cal['offset_um']:.3f} µm")
        return cal
    return None


def cleanup():
    """Reset devices to safe state."""
    print(f"\n{'='*70}")
    print("CLEANUP")
    print(f"{'='*70}")

    try:
        # Reset galvo Y to center
        core.setProperty(GALVO_DEVICE, "SingleAxisYOffset(deg)", 0.0)
        print("  ✓ Galvo Y reset to center")
    except Exception as e:
        print(f"  Could not reset galvo: {e}")

    try:
        # Lasers off
        core.setConfig("Laser", "ALL OFF")
        print("  ✓ Lasers OFF")
    except Exception as e:
        print(f"  Could not turn off lasers: {e}")


def main():
    """Main calibration workflow."""
    print("="*70)
    print("PIEZO-GALVO 2-POINT CALIBRATION")
    print("="*70)

    try:
        # Check for previous calibration
        prev_cal = load_previous_calibration()
        if prev_cal:
            response = input("\nOverwrite previous calibration? [y/N]: ").strip().lower()
            if response != 'y':
                print("Calibration cancelled")
                return

        # System startup
        print("\n[1/7] Applying System Startup configuration...")
        core.setConfig("System", "Startup")
        core.waitForConfig("System", "Startup")
        print("  ✓ System configured")

        # Lasers on
        print("\n[2/7] Turning on lasers...")
        core.setConfig("Laser", "488 and 561")
        core.waitForConfig("Laser", "488 and 561")
        print("  ✓ Lasers ON")

        # Configure camera
        print("\n[3/7] Configuring camera...")
        configure_camera()

        # Configure galvo for light sheet generation
        print("\n[3.5/7] Configuring galvo for light sheet generation...")
        configure_galvo_for_calibration()

        # Create output directory
        if SAVE_IMAGES:
            IMAGE_DIR.mkdir(exist_ok=True)

        # ===== TOP POSITION =====
        print("\n[4/7] Calibrating TOP position...")

        # Set piezo as focus device
        core.setFocusDevice(PIEZO_DEVICE)
        print(f"  Set focus device: {PIEZO_DEVICE}")

        # Set galvo to TOP position
        set_galvo_y_position(GALVO_Y_TOP)

        # Manual focus only (skip automated sweep)
        print("\n  Starting with middle of sweep range...")
        initial_pos = (PIEZO_SWEEP_START + PIEZO_SWEEP_END) / 2.0
        piezo_top_final = manual_refinement(initial_pos, "top")

        if piezo_top_final is None:
            print("\nCalibration cancelled")
            return

        # Capture final image
        core.setPosition(piezo_top_final)
        core.waitForDevice(PIEZO_DEVICE)
        time.sleep(0.2)
        img_top = capture_image()

        if SAVE_IMAGES:
            img_file = IMAGE_DIR / "calibration_top.tif"
            Image.fromarray(img_top.astype(np.uint16)).save(img_file)
            print(f"  Saved: {img_file}")

        # ===== BOTTOM POSITION =====
        print("\n[5/7] Calibrating BOTTOM position...")

        # Set galvo to BOTTOM position
        set_galvo_y_position(GALVO_Y_BOTTOM)

        # Manual focus only (skip automated sweep)
        print("\n  Starting with middle of sweep range...")
        initial_pos = (PIEZO_SWEEP_START + PIEZO_SWEEP_END) / 2.0
        piezo_bottom_final = manual_refinement(initial_pos, "bottom")

        if piezo_bottom_final is None:
            print("\nCalibration cancelled")
            return

        # Capture final image
        core.setPosition(piezo_bottom_final)
        core.waitForDevice(PIEZO_DEVICE)
        time.sleep(0.2)
        img_bottom = capture_image()

        if SAVE_IMAGES:
            img_file = IMAGE_DIR / "calibration_bottom.tif"
            Image.fromarray(img_bottom.astype(np.uint16)).save(img_file)
            print(f"  Saved: {img_file}")

        # ===== CALCULATE CALIBRATION =====
        print("\n[6/7] Calculating calibration...")
        calibration = calculate_calibration(
            GALVO_Y_TOP, piezo_top_final,
            GALVO_Y_BOTTOM, piezo_bottom_final
        )

        # ===== SAVE CALIBRATION =====
        print("\n[7/7] Saving calibration...")
        save_calibration(calibration)

        # Summary
        print(f"\n{'='*70}")
        print("✓ CALIBRATION COMPLETE")
        print(f"{'='*70}")
        print(f"\nUse these values in your acquisition script:")
        print(f"  PIEZO_GALVO_SLOPE = {calibration['slope_um_per_deg']:.3f}  # µm/°")
        print(f"  PIEZO_GALVO_OFFSET = {calibration['offset_um']:.3f}  # µm")
        print(f"\nCalibration file: {CALIBRATION_FILE}")

    except Exception as e:
        print(f"\n{'='*70}")
        print("ERROR DURING CALIBRATION")
        print("="*70)
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

    finally:
        cleanup()
        print("\nClose all matplotlib windows to exit.")
        plt.show()  # Keep windows open


if __name__ == "__main__":
    main()
