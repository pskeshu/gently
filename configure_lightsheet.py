"""
Test Light Sheet Acquisition
=============================

Standalone test script for DiSPIM light sheet imaging.
Tests hardware-synchronized light sheet generation with single image acquisition.

Usage:
    python test_lightsheet.py
"""

import numpy as np
import matplotlib.pyplot as plt
from bluesky import RunEngine
from bluesky.callbacks import LiveImage
import pymmcore
import rpyc
import time

from gently.devices import DiSPIMLightSheetSnap
from gently.plans import test_lightsheet


def connect_to_micromanager(host='localhost', port=18861):
    """
    Connect to Micro-Manager via RPyC

    Parameters
    ----------
    host : str
        Host address (default: 'localhost')
    port : int
        RPyC port (default: 18861)

    Returns
    -------
    CMMCore
        Micro-Manager core instance
    """
    print(f"Connecting to Micro-Manager at {host}:{port}...")
    conn = rpyc.classic.connect(host, port)
    core = conn.root.getCore()
    print("Connected!")
    return core


def initialize_lightsheet_device(core, scanner_name="Scanner:AB:33",
                                 camera_name="HamCam1"):
    """
    Initialize DiSPIMLightSheetSnap device

    Parameters
    ----------
    core : CMMCore
        Micro-Manager core instance
    scanner_name : str
        Scanner device name in MM config
    camera_name : str
        Camera device name in MM config

    Returns
    -------
    DiSPIMLightSheetSnap
        Light sheet snap device
    """
    print(f"Initializing light sheet device: {scanner_name} + {camera_name}")

    ls_snap = DiSPIMLightSheetSnap(
        scanner_device_name=scanner_name,
        camera_device_name=camera_name,
        core=core,
        name='lightsheet_test'
    )

    print("Light sheet device initialized")
    return ls_snap


def run_lightsheet_acquisition(ls_snap, sheet_width_deg=2.0, y_position_deg=0.0):
    """
    Run light sheet acquisition and capture image

    Parameters
    ----------
    ls_snap : DiSPIMLightSheetSnap
        Light sheet device
    sheet_width_deg : float
        Light sheet width in degrees
    y_position_deg : float
        Y-axis position (Z-plane selection)

    Returns
    -------
    np.ndarray
        Acquired image
    """
    # Storage for captured image
    captured_images = []

    def capture_image(name, doc):
        """Callback to capture image from event document"""
        if name == 'event':
            img = doc['data']['lightsheet_test']
            captured_images.append(img)
            print(f"Image captured: shape={img.shape}, dtype={img.dtype}")

    # Create RunEngine
    RE = RunEngine()
    RE.subscribe(capture_image)

    # Run acquisition
    print(f"\nRunning light sheet acquisition...")
    print(f"  Sheet width: {sheet_width_deg}°")
    print(f"  Y position: {y_position_deg}°")

    RE(test_lightsheet(
        ls_snap,
        sheet_width_deg=sheet_width_deg,
        y_position_deg=y_position_deg,
        metadata={'test_type': 'standalone_test'}
    ))

    if captured_images:
        return captured_images[-1]
    else:
        raise RuntimeError("No image was captured")


def analyze_image(image):
    """
    Analyze and display image statistics

    Parameters
    ----------
    image : np.ndarray
        Acquired image
    """
    print("\n" + "="*60)
    print("IMAGE ANALYSIS")
    print("="*60)
    print(f"Shape:      {image.shape}")
    print(f"Dtype:      {image.dtype}")
    print(f"Min value:  {image.min()}")
    print(f"Max value:  {image.max()}")
    print(f"Mean value: {image.mean():.1f}")
    print(f"Std dev:    {image.std():.1f}")
    print("="*60)


def display_image(image, title="Light Sheet Image"):
    """
    Display image with matplotlib

    Parameters
    ----------
    image : np.ndarray
        Image to display
    title : str
        Plot title
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    im = ax.imshow(image, cmap='gray', interpolation='nearest')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel('X (pixels)')
    ax.set_ylabel('Y (pixels)')

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Intensity (counts)', rotation=270, labelpad=20)

    # Add text with image info
    info_text = f"Shape: {image.shape}\nMin: {image.min()}\nMax: {image.max()}\nMean: {image.mean():.1f}"
    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round',
            facecolor='wheat', alpha=0.8), fontsize=10)

    plt.tight_layout()
    plt.show()


def save_image(image, filename='lightsheet_test.tif'):
    """
    Save image to file

    Parameters
    ----------
    image : np.ndarray
        Image to save
    filename : str
        Output filename
    """
    from PIL import Image

    # Convert to uint16 if needed
    if image.dtype != np.uint16:
        image = image.astype(np.uint16)

    img = Image.fromarray(image)
    img.save(filename)
    print(f"\nImage saved to: {filename}")


def main():
    """Main test function"""
    print("\n" + "="*60)
    print("DISPIM LIGHT SHEET ACQUISITION TEST")
    print("="*60 + "\n")

    try:
        # 1. Connect to Micro-Manager
        core = connect_to_micromanager()

        # 2. Initialize light sheet device
        ls_snap = initialize_lightsheet_device(core)

        # 3. Run acquisition
        image = run_lightsheet_acquisition(
            ls_snap,
            sheet_width_deg=2.0,
            y_position_deg=0.0
        )

        # 4. Analyze image
        analyze_image(image)

        # 5. Display image
        display_image(image)

        # 6. Save image (optional)
        save_image(image, 'lightsheet_test.tif')

        print("\n" + "="*60)
        print("TEST COMPLETED SUCCESSFULLY")
        print("="*60 + "\n")

    except Exception as e:
        print("\n" + "="*60)
        print("ERROR")
        print("="*60)
        print(f"{type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        print("="*60 + "\n")
        raise


if __name__ == "__main__":
    main()
