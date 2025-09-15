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

# Disable complex napari callback due to threading issues
# napari_callback = NapariCallback()
# RE.subscribe(napari_callback)

# Simple napari callback that works with threading
try:
    import napari
    viewer = napari.Viewer(title="DiSPIM Focus Test")
    
    def simple_napari_callback(name, doc):
        if name == 'event':
            data = doc.get('data', {})
            if 'bottom_camera' in data:
                image = data['bottom_camera']
                focus_pos = data.get('focus_bottom_z', 0)
                
                # Simple layer update without complex threading
                try:
                    if 'focus_images' in viewer.layers:
                        viewer.layers['focus_images'].data = image
                    else:
                        viewer.add_image(image, name='focus_images', colormap='gray')
                    print(f"Updated napari with focus position: {focus_pos:.1f}")
                except:
                    pass  # Ignore napari errors
    
    RE.subscribe(simple_napari_callback)
    print("Simple napari viewer created")
    
except ImportError:
    print("Napari not available")

def simple_focus_sweep(detector, motor, start, stop, num_points):
    """Simple focus sweep plan for embryo detection"""
    positions = np.linspace(start, stop, num_points)

    starting_pos = yield from bps.rd(motor)

    uid = yield from bps.open_run()

    for pos in positions:
        # Move to position
        yield from bps.mv(motor, pos)
        
        # Acquire image
        yield from bps.trigger_and_read([detector])
        
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