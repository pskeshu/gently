#!/usr/bin/env python3
"""
Multi-Embryo Volume Acquisition Calibration
============================================

Interactive workflow to:
1. Manually mark embryos by clicking on bottom camera view
2. Move stage to center each embryo
3. Run full calibration (edge detection, focus sweeps, 2-point linear fit)
4. Store each embryo's calibration in JSON database

Usage:
    python multi_embryo_calibration.py
"""

import time
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from client import get_mmc
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
import rpyc

# Device configuration
core = get_mmc()
CAMERA_NAME = "Bottom PCO"
XY_STAGE_NAME = "XYStage:XY:31"
CAMERA_NAME_SPIM = "HamCam1"
GALVO_DEVICE = "Scanner:AB:33"
PIEZO_DEVICE = "PiezoStage:P:34"

# Hardware specifications
CAMERA_PIXEL_SIZE_UM = 6.5
OBJECTIVE_MAGNIFICATION = 10.0
EFFECTIVE_PIXEL_SIZE = CAMERA_PIXEL_SIZE_UM / OBJECTIVE_MAGNIFICATION

# Camera exposure (ms)
CAMERA_EXPOSURE_MS = 50.0

# Database file
DATABASE_FILE = Path("multi_embryo_database.json")


class InteractiveEmbryoMarker:
    """Interactive matplotlib tool to mark embryo position."""

    def __init__(self, image, embryo_number, title=""):
        self.image = image
        self.embryo_number = embryo_number
        self.title = title
        self.h, self.w = image.shape
        self.center_x = self.w / 2.0
        self.center_y = self.h / 2.0

        self.embryo_clicks = []
        self.click_markers = []

        # Create figure
        self.fig, self.ax = plt.subplots(figsize=(14, 11))
        plt.subplots_adjust(bottom=0.15)

        # Display image
        img_norm = (image - image.min()) / (image.max() - image.min())
        self.im = self.ax.imshow(img_norm, cmap='gray')

        # Draw center crosshair
        self.ax.axvline(self.center_x, color='red', linestyle='--', linewidth=2.5, label='Center', alpha=0.8)
        self.ax.axhline(self.center_y, color='red', linestyle='--', linewidth=2.5, alpha=0.8)

        # Draw grid
        v_guidelines = [self.w * i for i in [0.2, 0.4, 0.6, 0.8]]
        h_guidelines = [self.h * i for i in [0.2, 0.4, 0.6, 0.8]]
        for x in v_guidelines:
            self.ax.axvline(x, color='cyan', linestyle=':', alpha=0.3, linewidth=1)
        for y in h_guidelines:
            self.ax.axhline(y, color='cyan', linestyle=':', alpha=0.3, linewidth=1)

        self.ax.set_title(f"{title}\n\nCLICK on EMBRYO #{embryo_number} to mark",
                         fontsize=14, fontweight='bold', pad=15)
        self.ax.set_xlabel(f"X (pixels)", fontsize=11)
        self.ax.set_ylabel(f"Y (pixels)", fontsize=11)

        # Instructions
        instructions = (
            f"MARKING EMBRYO #{embryo_number}\n"
            "━━━━━━━━━━━━━━━━━━━\n"
            "1. CLICK on embryo center\n"
            "2. 'Undo' to remove last click\n"
            "3. 'Done' when ready\n"
            "\n"
            "RED lines = Image center"
        )
        self.ax.text(0.02, 0.98, instructions,
                    transform=self.ax.transAxes,
                    fontsize=11,
                    verticalalignment='top',
                    color='white',
                    bbox=dict(boxstyle='round', facecolor='black', alpha=0.85))

        # Large embryo number annotation in center-top
        self.ax.text(0.5, 0.95, f"EMBRYO #{embryo_number}",
                    transform=self.ax.transAxes,
                    fontsize=20,
                    fontweight='bold',
                    horizontalalignment='center',
                    verticalalignment='top',
                    color='yellow',
                    bbox=dict(boxstyle='round', facecolor='red', alpha=0.9, pad=0.8))

        # Status
        self.status_text = self.ax.text(0.98, 0.02, f"No embryo marked",
                                       transform=self.ax.transAxes,
                                       fontsize=11,
                                       horizontalalignment='right',
                                       verticalalignment='bottom',
                                       color='yellow',
                                       bbox=dict(boxstyle='round', facecolor='black', alpha=0.85))

        # Buttons
        self.ax_undo = plt.axes([0.3, 0.05, 0.15, 0.075])
        self.btn_undo = Button(self.ax_undo, 'Undo', color='orange', hovercolor='red')
        self.btn_undo.on_clicked(self.undo_click)

        self.ax_done = plt.axes([0.55, 0.05, 0.15, 0.075])
        self.btn_done = Button(self.ax_done, 'Done', color='lightgreen', hovercolor='green')
        self.btn_done.on_clicked(self.done)

        self.cid = self.fig.canvas.mpl_connect('button_press_event', self.on_click)

        self.selected_position = None
        self.finished = False

    def on_click(self, event):
        if event.inaxes != self.ax:
            return
        if self.fig.canvas.toolbar and self.fig.canvas.toolbar.mode != '':
            return

        x, y = event.xdata, event.ydata
        if x is None or y is None:
            return

        self.embryo_clicks.append((x, y))

        marker, = self.ax.plot(x, y, 'o', color='lime', markersize=15,
                              markeredgewidth=3, markeredgecolor='white')
        self.click_markers.append(marker)
        self.ax.plot([x], [y], '+', color='white', markersize=20, markeredgewidth=3)

        offset_x = x - self.center_x
        offset_y = y - self.center_y
        self.status_text.set_text(
            f"Embryo #{self.embryo_number} marked\n"
            f"Position: ({x:.0f}, {y:.0f})\n"
            f"Offset: ({offset_x:+.0f}, {offset_y:+.0f}) px"
        )
        self.fig.canvas.draw()

    def undo_click(self, event):
        if len(self.embryo_clicks) == 0:
            return
        self.embryo_clicks.pop()
        if len(self.click_markers) > 0:
            marker = self.click_markers.pop()
            marker.remove()
        if len(self.embryo_clicks) == 0:
            self.status_text.set_text(f"No embryo marked")
        else:
            x, y = self.embryo_clicks[-1]
            offset_x = x - self.center_x
            offset_y = y - self.center_y
            self.status_text.set_text(
                f"Embryo #{self.embryo_number} marked\n"
                f"Position: ({x:.0f}, {y:.0f})\n"
                f"Offset: ({offset_x:+.0f}, {offset_y:+.0f}) px"
            )
        self.fig.canvas.draw()

    def done(self, event):
        if len(self.embryo_clicks) == 0:
            print("\n  ⚠ No embryo marked! Click on embryo first.")
            return
        self.selected_position = self.embryo_clicks[-1]
        self.finished = True
        plt.close(self.fig)

    def get_result(self):
        return self.selected_position


def configure_bottom_camera():
    """Configure bottom camera."""
    print(f"  Configuring bottom camera: {CAMERA_NAME}")
    core.setCameraDevice(CAMERA_NAME)
    core.setExposure(CAMERA_NAME, CAMERA_EXPOSURE_MS)
    try:
        core.setProperty(CAMERA_NAME, "TRIGGER SOURCE", "INTERNAL")
    except:
        pass
    time.sleep(0.1)
    print(f"  ✓ Bottom camera ready")


def get_stage_position():
    """Get current XY stage position."""
    x = core.getXPosition(XY_STAGE_NAME)
    y = core.getYPosition(XY_STAGE_NAME)
    return (x, y)


def capture_bottom_camera_image():
    """Capture image from bottom camera."""
    core.snapImage()
    img = core.getImage()
    try:
        img = rpyc.classic.obtain(img)
    except (ImportError, AttributeError):
        pass
    return img


def save_image_with_marker(image, embryo_pos, filename, title=""):
    """Save image with marked embryo position and center lines."""
    fig, ax = plt.subplots(figsize=(12, 10))

    img_norm = (image - image.min()) / (image.max() - image.min())
    ax.imshow(img_norm, cmap='gray')

    h, w = image.shape
    center_x, center_y = w / 2.0, h / 2.0

    # Draw center
    ax.axvline(center_x, color='red', linestyle='--', linewidth=2, label='Center')
    ax.axhline(center_y, color='red', linestyle='--', linewidth=2)

    # Draw embryo marker if provided
    if embryo_pos:
        ex, ey = embryo_pos
        ax.plot(ex, ey, 'o', color='lime', markersize=15,
               markeredgewidth=3, markeredgecolor='white', label='Embryo')
        ax.plot([ex], [ey], '+', color='white', markersize=20, markeredgewidth=3)

    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel(f"X (pixels)", fontsize=11)
    ax.set_ylabel(f"Y (pixels)", fontsize=11)
    ax.legend()

    plt.tight_layout()
    fig.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close(fig)


def save_all_embryos_marked(image, marked_embryos, filename):
    """Save image with all marked embryos annotated."""
    fig, ax = plt.subplots(figsize=(14, 12))

    img_norm = (image - image.min()) / (image.max() - image.min())
    ax.imshow(img_norm, cmap='gray')

    h, w = image.shape
    center_x, center_y = w / 2.0, h / 2.0

    # Draw center
    ax.axvline(center_x, color='red', linestyle='--', linewidth=2, label='Center', alpha=0.5)
    ax.axhline(center_y, color='red', linestyle='--', linewidth=2, alpha=0.5)

    # Draw all marked embryos
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(marked_embryos), 1)))
    for idx, emb in enumerate(marked_embryos):
        ex, ey = emb['pixel_position']
        emb_num = emb['embryo_number']

        ax.plot(ex, ey, 'o', color=colors[idx], markersize=18,
               markeredgewidth=4, markeredgecolor='white')
        ax.plot([ex], [ey], '+', color='white', markersize=22, markeredgewidth=4)
        ax.text(ex + 60, ey, f"#{emb_num}",
               color=colors[idx], fontsize=14, fontweight='bold',
               bbox=dict(boxstyle='round', facecolor='black', alpha=0.9))

    ax.set_title(f"All Marked Embryos ({len(marked_embryos)} total)",
                fontsize=16, fontweight='bold')
    ax.set_xlabel("X (pixels)", fontsize=12)
    ax.set_ylabel("Y (pixels)", fontsize=12)

    plt.tight_layout()
    fig.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Saved: {filename}")


def move_stage_to_center_embryo(embryo_pixel_pos, current_stage_pos, image_shape):
    """
    Move stage to center the marked embryo.

    Parameters
    ----------
    embryo_pixel_pos : tuple
        (x, y) pixel position of embryo
    current_stage_pos : tuple
        Current stage position in µm
    image_shape : tuple
        (height, width) of the image

    Returns
    -------
    tuple
        New stage position after movement
    """
    h, w = image_shape
    center_x_pixel = w / 2.0
    center_y_pixel = h / 2.0

    ex, ey = embryo_pixel_pos

    # Calculate displacement needed to center
    pixel_displacement_x = center_x_pixel - ex
    pixel_displacement_y = center_y_pixel - ey

    # Convert to stage movement
    # X is INVERTED: stage +X → embryo moves LEFT in camera
    # So to move embryo RIGHT, stage must move LEFT
    dx_stage = -pixel_displacement_x * EFFECTIVE_PIXEL_SIZE  # X inverted
    dy_stage = pixel_displacement_y * EFFECTIVE_PIXEL_SIZE   # Y same (to be tested)

    # Calculate target
    stage_x_current, stage_y_current = current_stage_pos
    target_x = stage_x_current + dx_stage
    target_y = stage_y_current + dy_stage

    print(f"\n{'─'*70}")
    print(f"  CENTERING EMBRYO - DEBUG")
    print(f"{'─'*70}")
    print(f"  Image shape: {image_shape} (H x W)")
    print(f"  Image center: ({center_x_pixel:.0f}, {center_y_pixel:.0f})")
    print(f"  Embryo at pixel: ({ex:.0f}, {ey:.0f})")
    print(f"  Pixel displacement: ({pixel_displacement_x:+.0f}, {pixel_displacement_y:+.0f}) pixels")
    print(f"  Effective pixel size: {EFFECTIVE_PIXEL_SIZE:.4f} µm/pixel")
    print(f"  Stage movement: ({dx_stage:+.2f}, {dy_stage:+.2f}) µm")
    print(f"  Current stage: ({stage_x_current:.2f}, {stage_y_current:.2f}) µm")
    print(f"  Target position: ({target_x:.2f}, {target_y:.2f}) µm")

    # Move
    core.setXYStageDevice(XY_STAGE_NAME)
    core.setXYPosition(float(target_x), float(target_y))
    core.waitForDevice(XY_STAGE_NAME)
    time.sleep(0.5)

    actual_pos = get_stage_position()
    print(f"  ✓ Moved to: ({actual_pos[0]:.2f}, {actual_pos[1]:.2f}) µm")

    return actual_pos


def run_calibration_for_embryo(embryo_id):
    """
    Run full piezo/galvo calibration workflow for one embryo.

    This imports and runs the calibration from calibrate_embryo_piezo_galvo.py

    Returns
    -------
    dict
        Calibration data, or None if failed
    """
    print(f"\n{'='*70}")
    print(f"RUNNING CALIBRATION FOR EMBRYO #{embryo_id}")
    print(f"{'='*70}")
    print(f"\n  This will run the full calibration workflow:")
    print(f"    - Edge detection (top/bottom)")
    print(f"    - Interior position calculation")
    print(f"    - Focus sweeps (coarse + fine)")
    print(f"    - 2-point linear fit")

    try:
        # Import the calibration module and run it
        print(f"\n  Starting calibration...")
        import calibrate_embryo_piezo_galvo

        # Run the main calibration workflow
        calibrate_embryo_piezo_galvo.main()

        # Load the generated calibration file
        cal_file = Path("piezo_galvo_calibration_embryo.json")
        if not cal_file.exists():
            print(f"  ✗ Calibration file not found!")
            return None

        with open(cal_file, 'r') as f:
            calibration = json.load(f)

        print(f"\n  ✓ Calibration complete")
        print(f"    Slope: {calibration['slope_um_per_deg']:.3f} µm/°")
        print(f"    Offset: {calibration['offset_um']:.3f} µm")
        print(f"    Top: {calibration['galvo_top_deg']:.3f}°")
        print(f"    Bottom: {calibration['galvo_bottom_deg']:.3f}°")

        return calibration

    except Exception as e:
        print(f"\n  ✗ Calibration failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def load_database():
    """Load existing embryo database or create new one."""
    if DATABASE_FILE.exists():
        with open(DATABASE_FILE, 'r') as f:
            return json.load(f)
    else:
        return {
            'created': datetime.now().isoformat(),
            'embryos': {}
        }


def save_database(database):
    """Save embryo database to JSON."""
    database['last_updated'] = datetime.now().isoformat()
    with open(DATABASE_FILE, 'w') as f:
        json.dump(database, f, indent=2)
    print(f"\n  ✓ Database saved: {DATABASE_FILE}")


def main():
    """Main multi-embryo calibration workflow."""
    print(f"{'='*70}")
    print("MULTI-EMBRYO CALIBRATION WORKFLOW")
    print(f"{'='*70}")
    print(f"\nThis workflow allows you to:")
    print(f"  1. Mark multiple embryos manually")
    print(f"  2. Center each embryo")
    print(f"  3. Run full calibration (edge, focus, fit)")
    print(f"  4. Store in multi-embryo database")

    try:
        # Load existing database
        print(f"\n{'='*70}")
        print("LOADING DATABASE")
        print(f"{'='*70}")
        database = load_database()
        num_existing = len(database.get('embryos', {}))
        print(f"  Existing embryos in database: {num_existing}")

        if num_existing > 0:
            print(f"\n  📋 Current embryos:")
            for emb_id, emb_data in database['embryos'].items():
                emb_num = emb_data.get('embryo_number', '?')
                pos = emb_data.get('stage_position_after_centering_um', {})
                print(f"    {emb_id} (Embryo #{emb_num}): ({pos.get('x', '?'):.1f}, {pos.get('y', '?'):.1f}) µm")

            print(f"\n  ⚠ WARNING: New embryos will start from #{num_existing + 1}")
            response = input(f"  Continue adding embryos (c) or Start fresh (f)? [c/f]: ").strip().lower()

            if response == 'f':
                print(f"\n  Starting fresh - clearing existing database...")
                database = {
                    'created': datetime.now().isoformat(),
                    'embryos': {}
                }
                num_existing = 0
                print(f"  ✓ Database cleared. Will start from embryo #1")
            else:
                print(f"  ✓ Continuing - will add to existing embryos")

        # Configure bottom camera
        print(f"\n{'='*70}")
        print("CONFIGURING HARDWARE")
        print(f"{'='*70}")
        configure_bottom_camera()

        # PHASE 1: MARK ALL EMBRYOS
        print(f"\n{'='*70}")
        print("PHASE 1: MARK ALL EMBRYOS")
        print(f"{'='*70}")

        # Get initial position
        initial_stage_pos = get_stage_position()
        print(f"  Initial stage position: ({initial_stage_pos[0]:.2f}, {initial_stage_pos[1]:.2f}) µm")

        # Capture initial bottom camera image
        print(f"\n  Capturing bottom camera image...")
        img_initial = capture_bottom_camera_image()
        print(f"  ✓ Captured: {img_initial.shape}")

        # Save initial image
        timestamp_session = datetime.now().strftime("%Y%m%d_%H%M%S")
        img_initial_file = f"initial_view_{timestamp_session}.png"
        save_image_with_marker(img_initial, None, img_initial_file,
                              f"Initial View - Stage: ({initial_stage_pos[0]:.1f}, {initial_stage_pos[1]:.1f}) µm")
        print(f"  ✓ Saved initial image: {img_initial_file}")

        # Mark all embryos
        marked_embryos = []
        embryo_counter = num_existing

        while True:
            embryo_counter += 1

            print(f"\n{'─'*70}")
            print(f"MARKING EMBRYO #{embryo_counter}")
            print(f"{'─'*70}")
            print(f"  Total marked so far: {len(marked_embryos)}")

            # Interactive marking
            print(f"\n  Opening marking window for embryo #{embryo_counter}...")
            marker = InteractiveEmbryoMarker(
                img_initial,
                embryo_counter,
                f"Mark Embryo #{embryo_counter} - Stage: ({initial_stage_pos[0]:.1f}, {initial_stage_pos[1]:.1f}) µm"
            )
            plt.show()

            embryo_pos = marker.get_result()

            if embryo_pos is None:
                print(f"\n  ✗ No embryo marked.")
                embryo_counter -= 1
                break

            ex, ey = embryo_pos
            print(f"  ✓ Embryo #{embryo_counter} marked at ({ex:.0f}, {ey:.0f})")

            # Store marked embryo
            marked_embryos.append({
                'embryo_number': embryo_counter,
                'embryo_id': f"embryo_{embryo_counter:03d}",
                'pixel_position': (ex, ey),
                'initial_stage_position': initial_stage_pos
            })

            # Ask to continue marking
            response = input(f"\n  Mark another embryo? (y/n): ").strip().lower()
            if response != 'y':
                break

        if len(marked_embryos) == 0:
            print(f"\n  ✗ No embryos marked. Exiting.")
            return

        print(f"\n{'='*70}")
        print(f"MARKING COMPLETE - {len(marked_embryos)} embryo(s) marked")
        print(f"{'='*70}")
        for emb in marked_embryos:
            print(f"  Embryo #{emb['embryo_number']}: ({emb['pixel_position'][0]:.0f}, {emb['pixel_position'][1]:.0f})")

        # Save marked embryos visualization
        img_marked_file = f"all_embryos_marked_{timestamp_session}.png"
        save_all_embryos_marked(img_initial, marked_embryos, img_marked_file)
        print(f"\n  ✓ Saved marked embryos: {img_marked_file}")

        # PHASE 2: CALIBRATE ALL EMBRYOS
        print(f"\n{'='*70}")
        print("PHASE 2: CALIBRATE ALL EMBRYOS")
        print(f"{'='*70}")

        for idx, emb_info in enumerate(marked_embryos, 1):
            embryo_counter = emb_info['embryo_number']
            embryo_id = emb_info['embryo_id']
            embryo_pos = emb_info['pixel_position']
            ex, ey = embryo_pos

            print(f"\n{'='*70}")
            print(f"CALIBRATING EMBRYO #{embryo_counter} ({idx}/{len(marked_embryos)})")
            print(f"{'='*70}")
            print(f"  Pixel position: ({ex:.0f}, {ey:.0f})")

            # Get current position (may have changed from previous calibration)
            pos_before_centering = get_stage_position()
            print(f"  Current stage: ({pos_before_centering[0]:.2f}, {pos_before_centering[1]:.2f}) µm")

            # Move to center embryo from initial position
            print(f"\n  Moving stage to center embryo...")
            print(f"  Image dimensions: {img_initial.shape}")
            # Calculate movement relative to initial stage position
            pos_after_centering = move_stage_to_center_embryo(embryo_pos, emb_info['initial_stage_position'], img_initial.shape)

            # Verify centering with new bottom camera image
            print(f"\n  Verifying embryo is centered...")
            time.sleep(0.5)  # Let stage settle
            img_after_centering = capture_bottom_camera_image()
            print(f"  ✓ Captured verification image")

            # Display verification image with all embryos
            print(f"\n  📷 Showing verification - current embryo should be at RED center")
            fig_verify, ax_verify = plt.subplots(figsize=(14, 11))

            img_norm = (img_after_centering - img_after_centering.min()) / (img_after_centering.max() - img_after_centering.min())
            ax_verify.imshow(img_norm, cmap='gray')

            h, w = img_after_centering.shape
            center_x, center_y = w / 2.0, h / 2.0

            # Center crosshair
            ax_verify.axvline(center_x, color='red', linestyle='--', linewidth=2.5, label='Center', alpha=0.9)
            ax_verify.axhline(center_y, color='red', linestyle='--', linewidth=2.5, alpha=0.9)

            # Grid
            v_guidelines = [w * i for i in [0.2, 0.4, 0.6, 0.8]]
            h_guidelines = [h * i for i in [0.2, 0.4, 0.6, 0.8]]
            for x in v_guidelines:
                ax_verify.axvline(x, color='cyan', linestyle=':', alpha=0.3, linewidth=1)
            for y in h_guidelines:
                ax_verify.axhline(y, color='cyan', linestyle=':', alpha=0.3, linewidth=1)

            # Mark current embryo (at center)
            ax_verify.plot(center_x, center_y, 'o', color='lime', markersize=20,
                          markeredgewidth=4, markeredgecolor='white',
                          label=f'Embryo #{embryo_counter} (CURRENT)', zorder=10)
            ax_verify.text(center_x + 60, center_y, f"#{embryo_counter}\nCURRENT",
                          color='lime', fontsize=14, fontweight='bold',
                          bbox=dict(boxstyle='round', facecolor='black', alpha=0.8))

            # Show all previously marked embryos
            if len(database['embryos']) > 0:
                colors = plt.cm.tab10(np.linspace(0, 1, 10))
                for idx, (emb_id, emb_data) in enumerate(database['embryos'].items()):
                    # Calculate where this embryo appears in current view
                    # Previous embryo's stage position
                    prev_stage_x = emb_data['stage_position_after_centering_um']['x']
                    prev_stage_y = emb_data['stage_position_after_centering_um']['y']

                    # Current stage position
                    curr_stage_x, curr_stage_y = pos_after_centering

                    # Stage offset
                    dx_stage = prev_stage_x - curr_stage_x
                    dy_stage = prev_stage_y - curr_stage_y

                    # Convert to pixel offset (with inverted X)
                    dx_pixel = -dx_stage / EFFECTIVE_PIXEL_SIZE  # X inverted
                    dy_pixel = dy_stage / EFFECTIVE_PIXEL_SIZE

                    # Calculate position in current image
                    embryo_x = center_x + dx_pixel
                    embryo_y = center_y + dy_pixel

                    # Only show if in frame
                    if 0 <= embryo_x < w and 0 <= embryo_y < h:
                        emb_num = emb_data.get('embryo_number', '?')
                        ax_verify.plot(embryo_x, embryo_y, 'o', color=colors[idx % 10],
                                      markersize=15, markeredgewidth=3,
                                      markeredgecolor='white', alpha=0.7)
                        ax_verify.text(embryo_x + 50, embryo_y, f"#{emb_num}",
                                      color=colors[idx % 10], fontsize=12, fontweight='bold',
                                      bbox=dict(boxstyle='round', facecolor='black', alpha=0.7))

            ax_verify.set_title(f"VERIFICATION - Embryo #{embryo_counter} Centered\n"
                               f"Stage: ({pos_after_centering[0]:.1f}, {pos_after_centering[1]:.1f}) µm  |  "
                               f"Total embryos: {len(database['embryos']) + 1}",
                               fontsize=14, fontweight='bold')
            ax_verify.set_xlabel("X (pixels)", fontsize=11)
            ax_verify.set_ylabel("Y (pixels)", fontsize=11)
            ax_verify.legend(loc='upper right')

            plt.tight_layout()

            # Save AFTER image
            img_after_file = f"embryo{embryo_counter:03d}_AFTER_centering_{timestamp_session}.png"
            fig_verify.savefig(img_after_file, dpi=150, bbox_inches='tight')
            print(f"  ✓ Saved AFTER image: {img_after_file}")

            plt.show(block=True)

            # Confirm before calibration
            print(f"\n  Embryo #{embryo_counter} should now be centered.")
            print(f"  (Check that embryo is at the RED crosshair)")
            response = input(f"  Run full calibration? (y/n): ").strip().lower()

            if response != 'y':
                print(f"  ✗ Skipping calibration")
                embryo_counter -= 1
                continue

            # Run calibration
            calibration = run_calibration_for_embryo(embryo_counter)

            if calibration is None:
                print(f"  ✗ Calibration failed for embryo #{embryo_counter}")
                embryo_counter -= 1
                continue

            # Store in database
            print(f"\n  Storing embryo #{embryo_counter} in database...")
            database['embryos'][embryo_id] = {
                'embryo_number': embryo_counter,
                'marking_timestamp': datetime.now().isoformat(),
                'bottom_camera_position_pixel': {
                    'x': float(ex),
                    'y': float(ey)
                },
                'initial_stage_position_um': {
                    'x': float(emb_info['initial_stage_position'][0]),
                    'y': float(emb_info['initial_stage_position'][1])
                },
                'stage_position_after_centering_um': {
                    'x': float(pos_after_centering[0]),
                    'y': float(pos_after_centering[1])
                },
                'calibration': calibration
            }

            save_database(database)

            print(f"\n  ✓ Embryo #{embryo_counter} complete!")
            print(f"    Database now contains {len(database['embryos'])} embryo(s)")

        # Summary
        print(f"\n{'='*70}")
        print("CALIBRATION SESSION COMPLETE")
        print(f"{'='*70}")
        print(f"\n  Total embryos in database: {len(database['embryos'])}")
        print(f"  Database file: {DATABASE_FILE}")

        print(f"\n  Embryos:")
        for emb_id, emb_data in database['embryos'].items():
            print(f"    {emb_id}: Stage ({emb_data['stage_position_after_centering_um']['x']:.1f}, "
                  f"{emb_data['stage_position_after_centering_um']['y']:.1f}) µm")

        print(f"\n{'='*70}\n")

    except KeyboardInterrupt:
        print(f"\n\nInterrupted\n")
    except Exception as e:
        print(f"\n{'='*70}")
        print("ERROR")
        print(f"{'='*70}")
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
