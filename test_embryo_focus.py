#!/usr/bin/env python3
"""
Simple Embryo Focus Test with Bottom Camera and Napari Visualization
"""

import numpy as np
import napari
from bluesky import RunEngine
import bluesky.plan_stubs as bps
from bluesky.callbacks.best_effort import BestEffortCallback

from gently.devices import DiSPIMCamera, DiSPIMZstage
from gently.plans import focus_sweep_with_analysis
from gently.analysis.focus import create_focus_positions
from gently.visualization import create_live_focus_plotter, setup_napari_camera_feed
from gently.analysis.core import FocusAnalysisConfig
from client import get_mmc

# Access DiSPIM with micromanager
core = get_mmc()

# A working DiSPIM model
bottom_camera = DiSPIMCamera("Bottom PCO", core, name="bottom_camera")
focus_bottom_z = DiSPIMZstage("ZStage:Z:32", core, name="focus_bottom_z")  # or whichever controls focus

# Setup Bluesky RunEngine - which will run our plans
RE = RunEngine()

# Add BestEffortCallback for terminal output
bec = BestEffortCallback()
RE.subscribe(bec)

# Add Napari camera feed visualization for our individual camera(s) for some human feedback
viewer, camera_feed = setup_napari_camera_feed("DiSPIM Focus Test", 'bottom_camera')
RE.subscribe(camera_feed)

def simple_focus_sweep(detector, motor, start, stop, num_points):
    """Simple focus sweep plan for embryo detection"""
    # Create position list for scan
    positions = np.linspace(start, stop, num_points)

    # Define the start of a run in Bluesky
    uid = yield from bps.open_run()

    for pos in positions:
        # Move to position
        yield from bps.mv(motor, pos)

        # Acquire image and read motor position
        yield from bps.trigger_and_read([detector, motor])

        print(f"Focus position: {pos:.2f} μm")

    # Note: Removed return to starting position to avoid floating point precision errors
    yield from bps.close_run()


def embryo_focus_test():
    """Test embryo focusing with bottom camera"""
    print("Starting embryo focus test...")

    # Focus sweep parameters
    start_pos = 100    # starting focus position (within new 50-250 limits)
    end_pos = 200     # ending focus position
    num_steps = 20    # number of focus steps

    # Run the focus sweep
    plan = simple_focus_sweep(bottom_camera, focus_bottom_z, start_pos, end_pos, num_steps)
    RE(plan)

    print("Focus test completed!")

def embryo_autofocus():
    """Simple embryo autofocus test"""
    print("Starting embryo autofocus...")

    # Simple focus analysis setup
    config = FocusAnalysisConfig(algorithm='gradient', minimum_r_squared=0.3)
    focus_plotter = create_live_focus_plotter("Embryo Focus")

    # Focus sweep: 150μm center, ±40μm range, 15 steps
    positions = create_focus_positions(150.0, 80.0, 15, focus_bottom_z.limits)
    print(f"Sweeping {len(positions)} positions from {min(positions):.1f} to {max(positions):.1f} μm")

    try:
        # Run focus sweep with analysis
        result = yield from focus_sweep_with_analysis(
            focus_bottom_z, bottom_camera, positions, config,
            callback=focus_plotter, metadata={'scan_type': 'embryo_focus'}
        )

        # Show results and move to best focus
        if result.success:
            print(f"Best focus: {result.best_position:.2f} μm (score: {result.best_score:.1f})")
            yield from bps.mv(focus_bottom_z, result.best_position)
        else:
            print(f"Focus failed: {result.error_message}")

    finally:
        focus_plotter.close()

def run_embryo_autofocus():
    """Run embryo autofocus test"""
    RE(embryo_autofocus())

def cleanup_napari():
    """Clean up napari viewer to prevent thread warnings"""
    try:
        viewer.close()
    except:
        pass

if __name__ == "__main__":
    try:
        run_embryo_autofocus()
    finally:
        cleanup_napari()