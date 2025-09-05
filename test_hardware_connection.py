#!/usr/bin/env python3
"""
Simple DiSPIM Hardware Connection Test
"""

import pymmcore
import os
from gently.devices import (
    DiSPIMPiezo,
    DiSPIMGalvo, 
    DiSPIMCamera,
    DiSPIMXYStage,
    DiSPIMLaserControl
)

# MM paths
mm_dir = "C:/Program Files/Micro-Manager-1.4"
config_file = "C:\\Users\\dispim\\Documents\\GitHub\\gently\\MMConfig_tracking_screening.cfg"

# Initialize MM
core = pymmcore.CMMCore()
core.enableStderrLog(True)
os.environ["PATH"] += os.pathsep + mm_dir
core.setDeviceAdapterSearchPaths([mm_dir])
core.loadSystemConfiguration(config_file)

print(f"Loaded MM devices: {list(core.getLoadedDevices())}")

# Create DiSPIM devices
piezo_a = DiSPIMPiezo("PiezoStage:P:34", core, name="piezo_a")
piezo_b = DiSPIMPiezo("PiezoStage:Q:35", core, name="piezo_b")
galvo_a = DiSPIMGalvo("Scanner:AB:33", core, name="galvo_a")  
galvo_b = DiSPIMGalvo("Scanner:CD:33", core, name="galvo_b")
camera_a = DiSPIMCamera("HamCam1", core, name="camera_a")
camera_b = DiSPIMCamera("HamCam2", core, name="camera_b")
xy_stage = DiSPIMXYStage("XYStage:XY:31", core, name="xy_stage")
laser = DiSPIMLaserControl(core, name="laser")

devices = [piezo_a, piezo_b, galvo_a, galvo_b, camera_a, camera_b, xy_stage, laser]

# Test reading from all devices
for device in devices:
    try:
        data = device.read()
        desc = device.describe()
        print(f"✓ {device.name}: {len(data)} signals")
    except Exception as e:
        print(f"✗ {device.name}: {e}")

print("Done")