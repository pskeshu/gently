#!/usr/bin/env python3
"""
Test script for DiSPIM microscope control

This script tests basic DiSPIM functionality:
1. Load the Micro-Manager configuration
2. Initialize devices 
3. Read piezo positions
4. Snap an image print the numpy arrary
"""

import pymmcore
import numpy as np
import os
import os.path
import matplotlib.pyplot as plt

mm_dir = "C:/Program Files/Micro-Manager-1.4"
config_path = "C:\\Users\\dispim\\Documents\\GitHub\\gently\\MMConfig_tracking_screening.cfg"


mmc = pymmcore.CMMCore()
mmc.enableStderrLog(True)

os.environ["PATH"] += os.pathsep.join(["", mm_dir]) # adviseable on Windows
mmc.setDeviceAdapterSearchPaths([mm_dir])
mmc.loadSystemConfiguration(config_path)

mmc.setCameraDevice("Bottom PCO")
mmc.snapImage()
img = mmc.getImage()

plt.imshow(img)
plt.show()

def startup():
    # Apply startup configuration
    try:
        mmc.setConfig('System', 'Startup')
        print("Startup configuration applied")
    except Exception as e:
        print(f"Could not apply startup configuration: {e}")


def read_piezo():
    # Read piezo positions
    try:
        piezo_p_position = mmc.getPosition('PiezoStage:P:34')
        piezo_q_position = mmc.getPosition('PiezoStage:Q:35')
        print(f"Piezo P position: {piezo_p_position}")
        print(f"Piezo Q position: {piezo_q_position}")
    except Exception as e:
        print(f"Could not read piezo positions: {e}")

def image_stuff():
    # Snap an image and print numpy array info
    try:
        mmc.snapImage()
        img = mmc.getImage()
        print(f"Image captured - Shape: {img.shape}, Type: {img.dtype}")
        print(f"Image stats - Min: {np.min(img)}, Max: {np.max(img)}, Mean: {np.mean(img):.2f}")
    except Exception as e:
        print(f"Could not capture image: {e}")

def shutdown():
    # Apply shutdown configuration
    try:
        mmc.setConfig('System', 'Shutdown')
        print("Shutdown configuration applied")
    except Exception as e:
        print(f"Could not apply shutdown configuration: {e}")

print("Test complete")