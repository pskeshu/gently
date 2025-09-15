#!/usr/bin/env python3
"""
Simple Embryo Focus Test with Bottom Camera and Napari Visualization
"""

import pymmcore
import os
import numpy as np
from bluesky import RunEngine
import bluesky.plan_stubs as bps
from bluesky.callbacks.best_effort import BestEffortCallback

from gently.devices import DiSPIMCamera, DiSPIMZstage
from gently.visualization import NapariCallback
from client import get_mmc

core = get_mmc()

# Create devices
bottom_camera = DiSPIMCamera("Bottom PCO", core, name="bottom_camera")
focus_bottom_z = DiSPIMZstage("ZStage:Z:32", core, name="focus_bottom_z")  # or whichever controls focus

# Setup RunEngine with napari visualization
RE = RunEngine()

# Add BestEffortCallback for terminal output
bec = BestEffortCallback()
RE.subscribe(bec)

# Simple napari visualization that avoids threading issues
try:
    import napari
    viewer = napari.Viewer(title="DiSPIM Focus Test")
    
    def safe_napari_callback(name, doc):
        """Thread-safe napari callback with minimal operations"""
        print(f"DEBUG: Callback called with name='{name}'")
        if name == 'event':
            print(f"DEBUG: Processing event document")
            try:
                data = doc.get('data', {})
                print(f"DEBUG: Event data keys: {list(data.keys())}")
                if 'bottom_camera' in data:
                    image = data['bottom_camera']
                    focus_pos = data.get('focus_bottom_z', 0)
                    print(f"DEBUG: Got image shape {image.shape}, focus = {focus_pos}")
                    
                    # Update existing layer or create new one
                    layer_name = 'focus_images'
                    if layer_name in viewer.layers:
                        print(f"DEBUG: Updating existing layer")
                        viewer.layers[layer_name].data = image
                    else:
                        print(f"DEBUG: Creating new layer")
                        viewer.add_image(image, name=layer_name, colormap='gray')
                    
                    print(f"Napari updated: focus = {focus_pos:.1f} μm")
                else:
                    print(f"DEBUG: No bottom_camera in data")
            except Exception as e:
                print(f"DEBUG: Napari callback error: {e}")
                import traceback
                traceback.print_exc()
    
    RE.subscribe(safe_napari_callback)
    print("Napari viewer created successfully")
    
except ImportError:
    print("Napari not available - install with: pip install napari[all]")

def simple_focus_sweep(detector, motor, start, stop, num_points):
    """Simple focus sweep plan for embryo detection"""
    positions = np.linspace(start, stop, num_points)

    starting_pos = yield from bps.rd(motor)

    uid = yield from bps.open_run()

    for pos in positions:
        # Move to position
        yield from bps.mv(motor, pos)
        
        # Acquire image and read motor position
        yield from bps.trigger_and_read([detector, motor])
        
        print(f"Focus position: {pos:.2f} μm")

    yield from bps.mv(motor, starting_pos)
    yield from bps.close_run()


def embryo_focus_test():
    """Test embryo focusing with bottom camera"""
    print("Starting embryo focus test...")
    
    # Focus sweep parameters
    start_pos = 50    # starting focus position
    end_pos = 150    # ending focus position  
    num_steps = 2     # number of focus steps
    
    # Run the focus sweep
    plan = simple_focus_sweep(bottom_camera, focus_bottom_z, start_pos, end_pos, num_steps)
    RE(plan)
    
    print("Focus test completed!")

if __name__ == "__main__":
    embryo_focus_test()