#!/usr/bin/env python3
"""
Test Data Reader - Read through datastore images and print image + z-position data
"""

import os
import numpy as np
from databroker import Broker
from datetime import datetime
import napari
import time

def load_databroker(data_dir="dispim_data"):
    """Load databroker from the dispim_data directory"""
    db_config = {
        'name': 'dispim_db',
        'metadatastore': {
            'module': 'databroker.headersource.sqlite',
            'config': {'directory': data_dir}
        },
        'assets': {
            'module': 'databroker.assets.sqlite',
            'config': {'directory': data_dir}
        }
    }

    try:
        db = Broker.from_config(db_config)
        return db
    except Exception as e:
        print(f"Error loading databroker: {e}")
        return None

def read_from_databroker(db, viewer=None):
    """Read data from databroker and display images in napari"""
    print("=== Reading from Databroker ===")

    # Get all runs
    runs = list(db())
    print(f"Found {len(runs)} runs in datastore")

    for i, run in enumerate(runs):
        print(f"\nRun {i+1}:")
        print(f"  UID: {run.start['uid']}")
        print(f"  Time: {datetime.fromtimestamp(run.start['time'])}")
        print(f"  Plan: {run.start.get('plan_name', 'unknown')}")

        # Get data from this run
        data = run.table()

        # Collect all images and z-positions for this run
        images = []
        z_positions = []

        for j, (idx, row) in enumerate(data.iterrows()):
            if 'bottom_camera' in row and 'focus_bottom_z' in row:
                image = row['bottom_camera']
                z_pos = row['focus_bottom_z']

                images.append(image)
                z_positions.append(z_pos)

                print(f"  Point {j+1}: Z={z_pos:.2f} μm, Image shape={image.shape}")

        # Add images to napari as a stack
        if images and viewer is not None:
            image_stack = np.array(images)
            layer_name = f"Run_{i+1}_{run.start['uid'][:8]}"

            # Create metadata for the layer
            metadata = {
                'z_positions': z_positions,
                'run_uid': run.start['uid'],
                'timestamp': datetime.fromtimestamp(run.start['time']).isoformat()
            }

            viewer.add_image(image_stack, name=layer_name, metadata=metadata)
            print(f"  Added {len(images)} images to napari layer '{layer_name}'")

def read_from_files(data_dir="dispim_data", viewer=None):
    """Read data from individual files and display images in napari"""
    print("\n=== Reading from Files ===")

    # Find all data files
    files = [f for f in os.listdir(data_dir) if f.startswith('dispim_run_') and f.endswith('_images.npz')]

    if not files:
        print("No data files found")
        return

    files.sort()  # Sort by timestamp
    print(f"Found {len(files)} data files")

    for file_idx, filename in enumerate(files):
        print(f"\nReading {filename}:")
        base_name = filename.replace('_images.npz', '')

        # Read metadata
        metadata_file = os.path.join(data_dir, f"{base_name}_metadata.txt")
        run_uid = "unknown"
        timestamp = "unknown"
        if os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                for line in f:
                    if line.startswith('Run UID:'):
                        run_uid = line.split(':', 1)[1].strip()
                    elif line.startswith('Time:'):
                        timestamp = line.split(':', 1)[1].strip()
                print(f"  UID: {run_uid}")
                print(f"  Time: {timestamp}")

        # Read positions
        positions_file = os.path.join(data_dir, f"{base_name}_positions.csv")
        positions = []
        if os.path.exists(positions_file):
            positions = np.loadtxt(positions_file, delimiter=',', skiprows=1)
            print(f"  Found {len(positions)} z-positions")

        # Read images
        images_file = os.path.join(data_dir, filename)
        if os.path.exists(images_file):
            data = np.load(images_file)
            images = data['images']
            print(f"  Found {len(images)} images")

            # Add images to napari if viewer is provided
            if viewer is not None and len(images) > 0:
                image_stack = np.array(images)
                layer_name = f"File_{file_idx+1}_{run_uid[:8] if len(run_uid) > 8 else run_uid}"

                # Create metadata for the layer
                metadata = {
                    'z_positions': positions.tolist() if len(positions) > 0 else [],
                    'run_uid': run_uid,
                    'timestamp': timestamp,
                    'filename': filename
                }

                viewer.add_image(image_stack, name=layer_name, metadata=metadata)
                print(f"  Added {len(images)} images to napari layer '{layer_name}'")

            # Print summary info
            for i, image in enumerate(images):
                z_pos = positions[i] if i < len(positions) else 0.0
                print(f"    Image {i+1}: shape={image.shape}, z={z_pos:.2f} μm, mean={image.mean():.1f}")

def main():
    """Main function to read and display data"""
    data_dir = "dispim_data"

    if not os.path.exists(data_dir):
        print(f"Data directory '{data_dir}' not found!")
        return

    print(f"Reading data from: {os.path.abspath(data_dir)}")

    # Create napari viewer
    print("Starting napari viewer...")
    viewer = napari.Viewer(title="DiSPIM Data Viewer")

    # Try reading from databroker first
    db = load_databroker(data_dir)
    if db:
        read_from_databroker(db, viewer)
    else:
        print("Could not load databroker, skipping...")

    # Read from files as backup/alternative
    read_from_files(data_dir, viewer)

    print("\nData loaded in napari. Use the layer controls to:")
    print("  - Toggle layers on/off")
    print("  - Scroll through z-stack with slider")
    print("  - Check layer metadata for z-positions")
    print("  - Adjust contrast/brightness")

    # Start napari (blocks until viewer is closed)
    if viewer.layers:
        print("Press Ctrl+C or close napari window to exit")
        napari.run()
    else:
        print("No data found to display")
        viewer.close()

if __name__ == "__main__":
    main()