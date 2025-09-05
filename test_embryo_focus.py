#!/usr/bin/env python3
"""
Simple Embryo Focus Test with Bottom Camera and Napari Visualization
"""

import pymmcore
import os
import numpy as np
from bluesky import RunEngine
import bluesky.plan_stubs as bps

from gently.devices import DiSPIMCamera, DiSPIMPiezo
from gently.visualization import NapariCallback

# MM setup
mm_dir = "C:/Program Files/Micro-Manager-1.4"
config_file = "C:\\Users\\dispim\\Documents\\GitHub\\gently\\MMConfig_tracking_screening.cfg"

# Initialize MM
core = pymmcore.CMMCore()
core.enableStderrLog(True)
os.environ["PATH"] += os.pathsep + mm_dir
core.setDeviceAdapterSearchPaths([mm_dir])
core.loadSystemConfiguration(config_file)

# Create devices
bottom_camera = DiSPIMCamera("Bottom PCO", core, name="bottom_camera")
focus_piezo = DiSPIMPiezo("PiezoStage:P:34", core, name="focus_piezo")  # or whichever controls focus

# Setup RunEngine with napari visualization
RE = RunEngine()
napari_callback = NapariCallback()
RE.subscribe(napari_callback)

def simple_focus_sweep(detector, motor, start, stop, num_points):
    """Simple focus sweep plan for embryo detection"""
    positions = np.linspace(start, stop, num_points)
    
    for pos in positions:
        # Move to position
        yield from bps.mv(motor, pos)
        
        # Acquire image
        yield from bps.trigger_and_read([detector])
        
        print(f"Focus position: {pos:.2f} μm")

def embryo_focus_test():
    """Test embryo focusing with bottom camera"""
    print("Starting embryo focus test...")
    
    # Focus sweep parameters
    start_pos = 0.0    # starting focus position
    end_pos = 50.0     # ending focus position  
    num_steps = 20     # number of focus steps
    
    # Run the focus sweep
    plan = simple_focus_sweep(bottom_camera, focus_piezo, start_pos, end_pos, num_steps)
    RE(plan)
    
    print("Focus test completed!")

if __name__ == "__main__":
    embryo_focus_test()