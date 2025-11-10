"""
Hardware control functions for microscope operations.

Wraps Micro-Manager core and existing calibration scripts.
"""

import sys
from pathlib import Path

# Add parent directory to Python path to import client.py
backend_dir = Path(__file__).parent
parent_dir = backend_dir.parent
sys.path.insert(0, str(parent_dir))

import time
import numpy as np
from client import get_mmc
import rpyc
import base64
from io import BytesIO
from PIL import Image
import json

# Device configuration
core = get_mmc()
CAMERA_NAME_BOTTOM = "Bottom PCO"
CAMERA_NAME_SPIM = "HamCam1"
XY_STAGE_NAME = "XYStage:XY:31"
GALVO_DEVICE = "Scanner:AB:33"
PIEZO_DEVICE = "PiezoStage:P:34"

# Camera specifications for bottom camera
CAMERA_PIXEL_SIZE_UM = 6.5
OBJECTIVE_MAGNIFICATION = 10.0
EFFECTIVE_PIXEL_SIZE = CAMERA_PIXEL_SIZE_UM / OBJECTIVE_MAGNIFICATION  # 0.65 µm/pixel


def encode_image_to_base64(img_array):
    """
    Convert numpy array to base64 PNG string.

    Parameters
    ----------
    img_array : np.ndarray
        Image array (grayscale or RGB)

    Returns
    -------
    str
        Base64 encoded PNG data URI
    """
    # Normalize to 8-bit if needed
    if img_array.dtype != np.uint8:
        img_min = img_array.min()
        img_max = img_array.max()
        if img_max > img_min:
            img_array = ((img_array - img_min) / (img_max - img_min) * 255).astype(np.uint8)
        else:
            img_array = np.zeros_like(img_array, dtype=np.uint8)

    # Convert to PIL Image
    if len(img_array.shape) == 2:  # Grayscale
        pil_img = Image.fromarray(img_array, mode='L')
    else:
        pil_img = Image.fromarray(img_array)

    # Encode as PNG to base64
    buffer = BytesIO()
    pil_img.save(buffer, format='PNG')
    img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')

    return f"data:image/png;base64,{img_str}"


def configure_bottom_camera(exposure_ms=50.0):
    """
    Configure bottom camera for imaging.

    Parameters
    ----------
    exposure_ms : float
        Exposure time in milliseconds

    Returns
    -------
    dict
        Camera configuration status
    """
    try:
        core.setCameraDevice(CAMERA_NAME_BOTTOM)
        core.setExposure(CAMERA_NAME_BOTTOM, exposure_ms)
        try:
            core.setProperty(CAMERA_NAME_BOTTOM, "TRIGGER SOURCE", "INTERNAL")
        except:
            pass
        time.sleep(0.1)

        return {
            "success": True,
            "camera": CAMERA_NAME_BOTTOM,
            "exposure_ms": exposure_ms
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


def capture_bottom_camera_image():
    """
    Capture image from bottom camera.

    Returns
    -------
    dict
        Image data and metadata
    """
    try:
        core.snapImage()
        img = core.getImage()

        # Handle rpyc proxy if needed
        try:
            img = rpyc.classic.obtain(img)
        except (ImportError, AttributeError):
            pass

        # Get current stage position
        stage_pos = get_stage_position()

        # Encode to base64
        img_base64 = encode_image_to_base64(img)

        return {
            "success": True,
            "image": img_base64,
            "image_array": img,  # Keep raw array for backend processing
            "shape": img.shape,
            "stage_position": stage_pos,
            "timestamp": time.time()
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


def get_stage_position():
    """
    Get current XY stage position.

    Returns
    -------
    dict
        Stage position in micrometers
    """
    try:
        x = core.getXPosition(XY_STAGE_NAME)
        y = core.getYPosition(XY_STAGE_NAME)
        return {"x": float(x), "y": float(y)}
    except Exception as e:
        return {"x": 0.0, "y": 0.0, "error": str(e)}


def move_stage_to_center_embryo(embryo_pixel_x, embryo_pixel_y, current_stage_x, current_stage_y, image_shape):
    """
    Calculate and execute stage movement to center an embryo.

    Parameters
    ----------
    embryo_pixel_x, embryo_pixel_y : float
        Embryo position in pixels
    current_stage_x, current_stage_y : float
        Current stage position in µm
    image_shape : tuple
        (height, width) of image

    Returns
    -------
    dict
        Result with new stage position
    """
    try:
        h, w = image_shape
        center_x_pixel = w / 2.0
        center_y_pixel = h / 2.0

        # Calculate pixel displacement
        pixel_displacement_x = center_x_pixel - embryo_pixel_x
        pixel_displacement_y = center_y_pixel - embryo_pixel_y

        # Convert to stage movement (X is inverted)
        dx_stage = -pixel_displacement_x * EFFECTIVE_PIXEL_SIZE
        dy_stage = pixel_displacement_y * EFFECTIVE_PIXEL_SIZE

        # Calculate target
        target_x = current_stage_x + dx_stage
        target_y = current_stage_y + dy_stage

        # Move stage
        core.setXYStageDevice(XY_STAGE_NAME)
        core.setXYPosition(float(target_x), float(target_y))
        core.waitForDevice(XY_STAGE_NAME)
        time.sleep(0.5)

        # Get actual position
        actual_pos = get_stage_position()

        return {
            "success": True,
            "target_position": {"x": target_x, "y": target_y},
            "actual_position": actual_pos,
            "displacement_um": {"x": dx_stage, "y": dy_stage}
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


def run_calibration_for_embryo(embryo_id):
    """
    Run full piezo/galvo calibration workflow.

    Calls the existing calibrate_embryo_piezo_galvo.py script.

    Parameters
    ----------
    embryo_id : str
        Embryo identifier

    Returns
    -------
    dict
        Calibration data or error
    """
    try:
        print(f"\n{'='*70}")
        print(f"RUNNING CALIBRATION FOR {embryo_id}")
        print(f"{'='*70}")

        # Import existing calibration module
        import calibrate_embryo_piezo_galvo

        # Run calibration
        print("  Starting calibration workflow...")
        calibrate_embryo_piezo_galvo.main()

        # Load generated calibration file
        cal_file = Path("piezo_galvo_calibration_embryo.json")
        if not cal_file.exists():
            return {
                "success": False,
                "error": "Calibration file not generated. Check hardware connections."
            }

        with open(cal_file, 'r') as f:
            calibration = json.load(f)

        # Validate calibration data
        required_fields = ['slope_um_per_deg', 'offset_um', 'galvo_top_deg', 'galvo_bottom_deg']
        for field in required_fields:
            if field not in calibration:
                return {
                    "success": False,
                    "error": f"Calibration missing required field: {field}"
                }

        print(f"  ✓ Calibration complete:")
        print(f"    Slope: {calibration['slope_um_per_deg']:.3f} µm/°")
        print(f"    Offset: {calibration['offset_um']:.2f} µm")
        print(f"{'='*70}\n")

        return {
            "success": True,
            "calibration": calibration
        }

    except ImportError as e:
        return {
            "success": False,
            "error": f"Failed to import calibration module: {str(e)}. Ensure calibrate_embryo_piezo_galvo.py is in the same directory."
        }
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"  ✗ Calibration failed:")
        print(error_trace)
        return {
            "success": False,
            "error": f"Calibration failed: {str(e)}"
        }


def move_stage_to_position(x, y):
    """
    Move stage to absolute position.

    Parameters
    ----------
    x, y : float
        Target position in micrometers

    Returns
    -------
    dict
        Result with actual position
    """
    try:
        core.setXYStageDevice(XY_STAGE_NAME)
        core.setXYPosition(float(x), float(y))
        core.waitForDevice(XY_STAGE_NAME)
        time.sleep(0.5)

        actual_pos = get_stage_position()

        return {
            "success": True,
            "target_position": {"x": x, "y": y},
            "actual_position": actual_pos
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


def get_hardware_status():
    """
    Get comprehensive hardware status.

    Returns
    -------
    dict
        Hardware status including stage position, etc.
    """
    try:
        stage_pos = get_stage_position()

        status = {
            "connected": True,
            "stage_position": stage_pos,
            "bottom_camera": CAMERA_NAME_BOTTOM,
            "spim_camera": CAMERA_NAME_SPIM,
            "timestamp": time.time()
        }

        return status
    except Exception as e:
        return {
            "connected": False,
            "error": str(e)
        }
