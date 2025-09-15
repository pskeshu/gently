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

# Create napari viewer first, then just update image data
try:
    import napari
    import numpy as np
    
    # Create viewer and add initial empty image layer
    viewer = napari.Viewer(title="DiSPIM Focus Test")
    dummy_image = np.zeros((2048, 2048), dtype=np.uint16)  # Initial empty image
    image_layer = viewer.add_image(dummy_image, name='Live Image', colormap='gray')
    
    def simple_image_updater(name, doc):
        """Just update the image data directly"""
        if name == 'event':
            data = doc.get('data', {})
            if 'bottom_camera' in data:
                image = data['bottom_camera']
                focus_pos = data.get('focus_bottom_z', 0)
                
                # Simply update the data property of the existing layer
                image_layer.data = image
                print(f"Updated napari: focus = {focus_pos:.1f} μm")
    
    RE.subscribe(simple_image_updater)
    print("Napari viewer created with live image layer")
    
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