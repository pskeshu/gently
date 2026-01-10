#!/usr/bin/env python3
"""
Test hardware-triggered SPIM volume acquisition using Bluesky/Ophyd integration

This script tests the DiSPIMVolumeScanner device and acquire_spim_volume() plan
to verify that the Ophyd/Bluesky integration works correctly.

Uses an in-memory document collector to retrieve volume data (Bluesky workflow pattern).
For production use with persistent storage, connect RunEngine to databroker or tiled.

Note: The default msgpack-based databroker has buffer size limitations with large
numpy arrays (100-slice volumes ~500MB). For production, use tiled server or
configure databroker with larger buffer limits.

Compare with test_volume_acq.py (standalone implementation) to verify equivalent functionality.
"""

import numpy as np
from bluesky import RunEngine
from bluesky.callbacks import LiveTable
from client import get_mmc

# Import the new Ophyd device and Bluesky plan
from gently.devices import DiSPIMVolumeScanner
from gently.plans import acquire_spim_volume

print("="*70)
print("BLUESKY SPIM VOLUME ACQUISITION TEST")
print("="*70)

# Initialize Micro-Manager core
core = get_mmc()

# Create RunEngine
RE = RunEngine({})

# For large data volumes, we'll use a simple in-memory collector instead of databroker
# (databroker's msgpack serializer has issues with large numpy arrays)
print("\nSetting up data collection...")
runs = []  # Store runs in memory

def collect_run(name, doc):
    """Simple callback to collect run documents in memory"""
    if name == 'start':
        runs.append({'start': doc, 'events': [], 'stop': None})
    elif name == 'event':
        if runs:
            runs[-1]['events'].append(doc)
    elif name == 'stop':
        if runs:
            runs[-1]['stop'] = doc

RE.subscribe(collect_run)
print("  RunEngine subscribed to in-memory collector")

# Create volume scanner device
print("\nCreating DiSPIMVolumeScanner device...")
vol_scanner = DiSPIMVolumeScanner(
    scanner_device_name="Scanner:AB:33",
    camera_device_name="HamCam1",
    core=core,
    name='volume_scanner'
)
print(f"  Device created: {vol_scanner.name}")

# Acquisition parameters
NUM_SLICES = 100
EXPOSURE_MS = 5.0
SLICE_STEP_UM = 1.0

print(f"\nAcquisition parameters:")
print(f"  Slices: {NUM_SLICES}")
print(f"  Exposure: {EXPOSURE_MS} ms")
print(f"  Slice step: {SLICE_STEP_UM} μm")

# Setup LiveTable callback to monitor acquisition
print("\nSetting up Bluesky callbacks...")
livetable = LiveTable([vol_scanner.name])

# Subscribe callback to RunEngine
RE.subscribe(livetable)

print("\n" + "="*70)
print("STARTING BLUESKY ACQUISITION")
print("="*70)

try:
    # Run the acquisition plan
    print("\nExecuting acquire_spim_volume() plan...")

    # Execute plan and capture run UID
    uid = RE(acquire_spim_volume(
        vol_scanner,
        num_slices=NUM_SLICES,
        exposure_ms=EXPOSURE_MS,
        slice_step_um=SLICE_STEP_UM,
        metadata={'test': 'bluesky_integration', 'operator': 'test_script'}
    ))

    print("\n" + "="*70)
    print("ACQUISITION RESULTS")
    print("="*70)

    # Retrieve run from collected documents
    print(f"\nRetrieving run data...")
    print(f"  Run UID: {uid[0][:8]}...")

    if not runs:
        raise ValueError("No run data collected!")

    run = runs[-1]  # Get the most recent run

    # Extract volume data from event documents
    if not run['events']:
        raise ValueError("No event data in run!")

    event_data = run['events'][0]['data']  # First (and only) event
    volume = event_data[vol_scanner.name]

    print(f"✓ SUCCESS! Volume retrieved from run data")
    print(f"\nVolume properties:")
    print(f"  Shape: {volume.shape} (Z, Y, X)")
    print(f"  Dtype: {volume.dtype}")
    print(f"  Range: [{volume.min()}, {volume.max()}]")
    print(f"  Mean: {volume.mean():.1f}")
    print(f"  Std: {volume.std():.1f}")

    # Display run metadata
    print(f"\nRun metadata:")
    start_doc = run['start']
    stop_doc = run['stop']
    print(f"  Plan name: {start_doc['plan_name']}")
    print(f"  Num slices: {start_doc['num_slices']}")
    print(f"  Exposure: {start_doc['exposure_ms']} ms")
    print(f"  Slice step: {start_doc['slice_step_um']} μm")
    print(f"  Start time: {start_doc['time']}")
    print(f"  Stop time: {stop_doc['time']}")
    duration = stop_doc['time'] - start_doc['time']
    print(f"  Duration: {duration:.2f} seconds")

    # Performance metrics
    fps = NUM_SLICES / duration
    print(f"\nPerformance:")
    print(f"  Acquisition time: {duration:.2f} seconds")
    print(f"  Frame rate: {fps:.1f} fps")

 
    # Display volume
    print(f"\nDisplaying volume...")
    try:
        import napari
        viewer = napari.Viewer()
        viewer.add_image(
            volume,
            name='SPIM Volume (Bluesky)',
            colormap='gray',
            contrast_limits=[np.percentile(volume, 1), np.percentile(volume, 99)]
        )
        viewer.dims.axis_labels = ['Z', 'Y', 'X']

        # Add run metadata to viewer
        viewer.text_overlay.text = f"Run {uid[0][:8]}... | {NUM_SLICES} slices @ {fps:.1f} fps"
        viewer.text_overlay.visible = True

        print("  Displaying in napari (close window to continue)...")
        napari.run()
    except ImportError:
        print("  napari not available - skipping visualization")

    print("\n" + "="*70)
    print("✓ BLUESKY INTEGRATION TEST PASSED")
    print("="*70)
    print("\nThe DiSPIMVolumeScanner device and acquire_spim_volume() plan")
    print("work correctly with Bluesky RunEngine!")
    print(f"\nVolume data stored with UID: {uid[0]}")
    print(f"Total runs collected: {len(runs)}")

except Exception as e:
    print(f"\n{'='*70}")
    print("ERROR DURING BLUESKY ACQUISITION")
    print("="*70)
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

    print("\nDiagnostics:")
    print(f"  Device configured: {vol_scanner._configured}")
    if vol_scanner._configured:
        print(f"  Num slices: {vol_scanner._num_slices}")
        print(f"  Exposure: {vol_scanner._exposure_ms} ms")
        print(f"  Slice step: {vol_scanner._slice_step_um} μm")

finally:
    # Cleanup
    print("\n" + "="*70)
    print("CLEANUP")
    print("="*70)

    try:
        # Turn off lasers
        core.setConfig("Laser", "ALL OFF")
        print("  ✓ Lasers OFF")
    except Exception as e:
        print(f"  Could not turn off lasers: {e}")

    print()
