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
import yaml
import numpy as np
import os

# Read configuration from config.yml
with open('config.yml', 'r') as f:
    config = yaml.safe_load(f)

# Initialize Micro-Manager core
mmc = pymmcore.CMMCore()

# Load configuration file
config_path = config['mmconfig']
if os.path.exists(config_path):
    print(f"Loading Micro-Manager configuration: {config_path}")
    mmc.loadSystemConfiguration(config_path)
    print("Configuration loaded successfully")
else:
    print(f"Configuration file not found: {config_path}")

# Apply startup configuration
try:
    mmc.setConfig('System', 'Startup')
    print("Startup configuration applied")
except Exception as e:
    print(f"Could not apply startup configuration: {e}")

# Read piezo positions
try:
    piezo_p_position = mmc.getPosition('PiezoStage:P:34')
    piezo_q_position = mmc.getPosition('PiezoStage:Q:35')
    print(f"Piezo P position: {piezo_p_position}")
    print(f"Piezo Q position: {piezo_q_position}")
except Exception as e:
    print(f"Could not read piezo positions: {e}")

# Snap an image and print numpy array info
try:
    mmc.snapImage()
    img = mmc.getImage()
    print(f"Image captured - Shape: {img.shape}, Type: {img.dtype}")
    print(f"Image stats - Min: {np.min(img)}, Max: {np.max(img)}, Mean: {np.mean(img):.2f}")
except Exception as e:
    print(f"Could not capture image: {e}")

# Apply shutdown configuration
try:
    mmc.setConfig('System', 'Shutdown')
    print("Shutdown configuration applied")
except Exception as e:
    print(f"Could not apply shutdown configuration: {e}")

print("Test complete")