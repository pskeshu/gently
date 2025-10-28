#!/usr/bin/env python3
"""
Embryo-Based Piezo-Galvo 2-Point Calibration for ASI diSPIM
===========================================================

This script performs piezo-galvo calibration using embryo samples instead of beads.
It uses Claude's vision model to assess focus quality by examining embryo boundary sharpness.

WORKFLOW:
1. Initial centering check: Verify embryo is visible at center (piezo=0, galvo=0)
2. For TOP position (galvo=+0.3°):
   - Move galvo, assess current focus state
   - Perform automated focus sweep using Claude vision
   - Find optimal piezo position where embryo top boundary is sharpest
3. For BOTTOM position (galvo=-0.3°):
   - Move galvo, assess current focus state
   - Perform automated focus sweep using Claude vision
   - Find optimal piezo position where embryo bottom boundary is sharpest
4. Calculate 2-point linear calibration: piezo_µm = slope × galvo_deg + offset
5. Save to piezo_galvo_calibration_embryo.json
6. Test volume acquisition with new calibration

Claude is positioned as an expert microscopist who can visually assess the
sharpness of embryo boundaries in diSPIM light sheet microscopy.
"""

import time
import json
import base64
from pathlib import Path
from datetime import datetime
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import os
from scipy.optimize import curve_fit

# Anthropic API for Claude vision analysis
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False
    print("WARNING: anthropic library not available. Install with: pip install anthropic")

# Microscope control
from client import get_mmc

# ============================================================================
# CONFIGURATION
# ============================================================================

# Device configuration
CAMERA_NAME = "HamCam1"
GALVO_DEVICE = "Scanner:AB:33"
PIEZO_DEVICE = "PiezoStage:P:34"

# Galvo calibration positions
# NOTE: Galvo Y angles control light sheet Z position
# - Positive angle (+0.3°) = light sheet at BOTTOM (physically lower in sample)
# - Negative angle (-0.3°) = light sheet at TOP (physically higher in sample)
# - GALVO_Y_TOP and GALVO_Y_BOTTOM are now DETECTED DYNAMICALLY in Phase 1.5
GALVO_Y_CENTER = 0.0     # degrees - center position (fixed, used for initial centering)

# Initial heuristic positions (will be updated after edge detection in Phase 1.5)
# These are rough starting guesses - the actual values will be calculated adaptively
HEURISTIC_TOP_PIEZO = 30.0       # µm - initial guess (updated dynamically)
HEURISTIC_BOTTOM_PIEZO = -30.0   # µm - initial guess (updated dynamically)

# Heuristic piezo-galvo calibration for edge detection
# Strategy: Use previous calibration if available, otherwise fall back to empirical default
# Historical data shows slope is consistently ~99.5-100 µm/degree
HEURISTIC_CALIBRATION_SLOPE = 100.0  # µm/degree - fallback default
HEURISTIC_CALIBRATION_OFFSET = 0.0   # µm - assume centered

# Embryo edge detection parameters (Phase 1.5 - adaptive extent detection)
# Strategy: Start from CENTER (where embryo is confirmed) and sweep OUTWARD until it disappears
EDGE_DETECTION_START = 0.0           # degrees - start at center (known embryo position)
EDGE_DETECTION_END_TOP = -0.5        # degrees - sweep up (negative) to find top edge
EDGE_DETECTION_END_BOTTOM = 0.5      # degrees - sweep down (positive) to find bottom edge
EDGE_DETECTION_STEP = 0.05           # degrees - fine steps (~5µm resolution with typical calibration)
EMBRYO_EDGE_TOLERANCE_DEG = 0.20     # degrees - absolute margin beyond detected edge (~20µm for empty frames)

# Calibration position strategy
# CRITICAL: Calibration (focus sweeps) should happen at INTERIOR positions with good morphology,
# NOT at detected edges where embryo is sparse/fading. This ensures accurate focus detection.
CALIBRATION_INSET_FRACTION = 0.4     # fraction - 40% inward from each edge for calibration

# Focus sweep parameters
# Using coarse sweep since we're doing initial calibration
SWEEP_RANGE_UM = 20.0     # ±20µm around heuristic position
SWEEP_STEP_UM = 2.0       # 2µm steps for initial sweep

# Fine refinement parameters
REFINEMENT_RANGE_UM = 5.0    # ±5µm around best position
REFINEMENT_STEP_UM = 0.5     # 0.5µm steps for refinement

# FFT Bandpass focus parameters (ASI diSPIM plugin approach)
# These parameters define the spatial frequency band analyzed for focus quality
FFT_LOWER_CUTOFF = 0.025     # 2.5% of max frequency - filters out DC and low spatial frequencies
FFT_UPPER_CUTOFF = 0.14      # 14% of max frequency - filters out high-frequency noise
MINIMUM_R_SQUARED = 0.75     # Gaussian fit quality threshold (reject if R² < this)
FOCUS_EDGE_EXCLUSION = 0.20  # Reject focus if peak in outer 20% of sweep range

# Camera settings
# Using longer exposure during calibration to integrate multiple galvo scan cycles
CAMERA_EXPOSURE_MS = 50.0  # milliseconds

# Volume acquisition test parameters
VOLUME_NUM_SLICES = 100
VOLUME_EXPOSURE_MS = 10.0
VOLUME_SLICE_PERIOD_MS = 50.0

# Output configuration
CALIBRATION_FILE = Path("piezo_galvo_calibration_embryo.json")
IMAGE_DIR = Path("calibration_images_embryo")
TEST_VOLUME_FILE = Path("embryo_volume_test.tif")

# Global state
_core = None
_anthropic_client = None
_current_fig = None
_current_ax = None


# ============================================================================
# ANTHROPIC API HELPER FUNCTIONS
# ============================================================================

def initialize_anthropic_client():
    """Initialize Anthropic API client."""
    if not ANTHROPIC_AVAILABLE:
        raise RuntimeError("anthropic library not installed. Run: pip install anthropic")

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise RuntimeError("ANTHROPIC_API_KEY environment variable not set")

    return anthropic.Anthropic(api_key=api_key)


def encode_image_to_base64(image_path):
    """Encode image file to base64 string."""
    with open(image_path, "rb") as image_file:
        return base64.standard_b64encode(image_file.read()).decode("utf-8")


def ask_claude_with_image(client, prompt, image_path, max_tokens=1024):
    """
    Send image and prompt to Claude and get response.

    Args:
        client: Anthropic client
        prompt: Text prompt for Claude
        image_path: Path to image file
        max_tokens: Maximum tokens in response

    Returns:
        str: Claude's response text
    """
    # Encode image
    image_data = encode_image_to_base64(image_path)

    # Determine media type from extension
    ext = Path(image_path).suffix.lower()
    media_type_map = {
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.gif': 'image/gif',
        '.webp': 'image/webp'
    }
    media_type = media_type_map.get(ext, 'image/png')

    # Call API with latest Sonnet
    message = client.messages.create(
        model="claude-sonnet-4-5-20250929",
        max_tokens=max_tokens,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": media_type,
                            "data": image_data,
                        },
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ],
            }
        ],
    )

    return message.content[0].text


# ============================================================================
# CLAUDE EXPERT PROMPTS FOR EMBRYO IMAGING
# ============================================================================

EMBRYO_CENTERING_PROMPT = """You are an expert microscopist examining a diSPIM light sheet microscopy image of a biological embryo sample.

This image shows ONE camera view from the diSPIM system. You should look for an embryo structure somewhere in the field of view. The embryo will appear as a brighter structure against a dark background, but the signal may be MODERATE (not necessarily super bright).

IMPORTANT CONTEXT:
- This is a REAL microscopy image with typical noise and artifacts
- Room lighting may cause some background glow (this is normal - ignore it)
- Embryos appear as irregularly-shaped bright regions, NOT perfectly uniform
- The embryo does NOT need to be perfectly centered, just reasonably visible in the frame
- Signal levels are moderate - don't expect extremely bright fluorescence

YOUR TASK:
Determine if an embryo structure is visible in this image that we can use for calibration.

WHAT TO LOOK FOR (be forgiving):
✓ EMBRYO VISIBLE:
  - ANY distinct structure brighter than the background (doesn't need to be super bright)
  - Some defined boundary or edge (even if irregular or somewhat soft)
  - Structure appears biological (rounded, irregular, or oblong shape)
  - Structure is not cut off at the edge of the frame
  - You can distinguish the embryo from background even if contrast is moderate

✗ NO USABLE EMBRYO:
  - Absolutely no structure visible, only uniform background
  - Frame is completely empty or saturated
  - No contrast at all between any structures and background
  - Any visible structure is completely cut off at frame edge

RESPOND FORMAT:
Line 1: "yes" if ANY embryo structure is visible that we can work with, "no" only if truly absent
Line 2: Brief description of what you see (1-2 sentences)

Example response:
yes
An irregularly-shaped embryo structure is visible in the left-center region with moderate brightness and defined boundaries against the dark background."""

EMBRYO_FOCUS_STATE_PROMPT = """You are an expert microscopist specializing in diSPIM light sheet microscopy of embryos.

This image shows ONE camera view of an embryo captured with light sheet illumination. Focus quality determines how sharp the embryo boundary appears. This is a REAL microscopy image with typical noise and moderate signal levels.

Rate the focus quality on this scale (be realistic about typical microscopy image quality):

1 = VERY BLURRY: Embryo boundary is extremely diffuse, edges completely unclear and barely distinguishable from background
2 = BLURRY: Embryo boundary noticeably blurred, edges are soft and indistinct, hard to trace the outline
3 = MODERATE: Embryo boundary somewhat defined, edges are visible but not crisp, you can roughly trace the outline
4 = GOOD: Embryo boundary fairly sharp, edges clearly visible and distinguishable, outline is well-defined
5 = EXCELLENT: Embryo boundary very sharp and crisp, edges are well-defined with clear transitions

IMPORTANT:
- Focus on the SHARPNESS of the embryo boundary/outline, NOT overall brightness
- Be realistic - most images will be 2-4, not all perfect 5s
- Even moderate-quality embryo images (rating 3-4) are perfectly usable for calibration
- Look at the CLEAREST edge you can find on the embryo

Respond with ONLY a single number (1-5) and nothing else."""

EMBRYO_MONTAGE_ANALYSIS_PROMPT_TEMPLATE = """You are an expert microscopist specializing in diSPIM light sheet microscopy of embryos.

This montage shows {num_positions} images of an embryo captured at different piezo Z positions during a focus sweep. Each panel shows ONE camera view and is labeled with its position number and piezo value in micrometers. These are REAL microscopy images with typical noise and moderate signal levels.

The embryo appears as a moderately bright structure against a dark background. When best focused, the embryo BOUNDARY is SHARPEST and most well-defined. When out of focus, the boundary becomes softer and more diffuse.

FOCUS QUALITY INDICATORS:
✓ BETTER FOCUS (what to look for):
  - Embryo boundary/outline is relatively sharp and well-defined
  - Better contrast between embryo edge and background
  - You can trace the embryo outline more clearly
  - Edges have cleaner transitions (less blur halo)
  - Internal structures or texture may be more visible

✗ WORSE FOCUS:
  - Embryo boundary is softer and more diffuse
  - Lower contrast, edges blend more into background
  - Outline is harder to trace precisely
  - Wider blur halo around edges
  - Loss of internal detail

IMPORTANT:
- Look for RELATIVE sharpness - which position looks SHARPEST compared to the others
- The "best" image may not be perfect, but it should be noticeably sharper than neighbors
- Focus on the clearest edge you can find on the embryo
- Small differences matter - even subtle improvement in edge definition indicates better focus

YOUR TASK:
Examine ALL {num_positions} panels carefully. Identify which position number shows the SHARPEST/CLEAREST embryo boundary relative to the other positions.

Look for:
1. Sharpest embryo outline/edge (even if not perfect)
2. Best contrast at boundary compared to neighbors
3. Clearest definition of embryo shape
4. Tightest edges with minimal blur

RESPOND FORMAT:
Line 1: Position number only (e.g., "6")
Line 2: Brief reasoning comparing this position to neighbors (1-2 sentences)

Example response:
6
Position 6 shows notably sharper embryo boundaries compared to positions 5 and 7, with clearer edge definition and better contrast at the outline."""

EMBRYO_EDGE_DETECTION_PROMPT = """You are an expert microscopist examining a diSPIM light sheet microscopy image.

This image shows ONE camera view that MAY OR MAY NOT contain part of an embryo. The embryo appears as a brighter structure against a dark background.

IMPORTANT CONTEXT:
- This is a REAL microscopy image with typical noise
- Room lighting may cause some background glow (ignore this completely)
- The embryo may be partially out of focus (that's okay for edge detection)
- We are scanning through Z to find where the embryo starts/ends
- We only need to know: Is ANY part of the embryo visible in this frame?

YOUR TASK:
Determine if ANY portion of an embryo structure is present in this image.

WHAT TO LOOK FOR:
✓ EMBRYO PRESENT:
  - ANY brighter structure that stands out from background (even if faint)
  - Irregular biological shape (doesn't need to be perfectly defined)
  - Even a partial view or edge of the embryo counts as "present"
  - Structure has some spatial extent (not just a few isolated bright pixels)

✗ EMBRYO ABSENT:
  - Frame shows only dark background with uniform noise
  - No distinguishable structure or pattern
  - Only background glow from room lighting (uniform, not localized)
  - Completely empty field of view

BE FORGIVING: If you see ANY structure that could plausibly be part of an embryo, respond "yes".
Only respond "no" if the frame is truly empty or shows only background.

RESPOND FORMAT:
Line 1: "yes" if you see any embryo structure (even faint), "no" if frame is essentially empty
Line 2: Very brief description (1 sentence)

Example responses:
yes
A faint irregular embryo structure is visible in the upper portion of the frame.

no
The frame shows only uniform dark background with no distinguishable embryo structure.

yes
Part of an embryo boundary is visible on the left side of the frame."""


# ============================================================================
# MATPLOTLIB VISUALIZATION FUNCTIONS
# ============================================================================

def setup_matplotlib_figure():
    """Setup matplotlib for live display."""
    print("\n[MATPLOTLIB] Setting up interactive display...")
    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots(1, 1, figsize=(12, 8))
    print("  ✓ Matplotlib ready")
    return fig, ax


def update_matplotlib_image(fig, ax, image, title_info):
    """Update matplotlib figure with new image."""
    if fig is None or ax is None:
        return

    # Build title string with clear phase information
    phase = title_info.get('phase', '')
    stage = title_info.get('stage', 'Calibration')
    position = title_info.get('position', '')
    galvo = title_info.get('galvo', '')
    note = title_info.get('note', '')

    # Main title with phase prominently displayed
    if phase:
        main_title = f"{phase} │ {stage}"
    else:
        main_title = stage

    # Additional info
    info_parts = []
    if galvo:
        info_parts.append(f"Galvo: {galvo}°")
    if position:
        info_parts.append(f"Piezo: {position}µm")
    if note:
        info_parts.append(f"[{note}]")

    if info_parts:
        full_title = f"{main_title}\n{' │ '.join(info_parts)}"
    else:
        full_title = main_title

    # Clear and replot
    ax.clear()
    vmin, vmax = np.percentile(image, [1, 99])
    ax.imshow(image, cmap='gray', vmin=vmin, vmax=vmax)
    ax.set_title(full_title, fontsize=12, fontweight='bold', pad=15)
    ax.axis('off')

    # Update display
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(0.01)


def display_montage_matplotlib(fig, ax, montage_image, title, phase=None):
    """Display montage image in matplotlib."""
    if fig is None or ax is None:
        return

    # Add phase information to title if provided
    if phase:
        full_title = f"{phase} │ {title}"
    else:
        full_title = title

    ax.clear()
    ax.imshow(montage_image, cmap='gray')
    ax.set_title(full_title, fontsize=12, fontweight='bold', pad=15)
    ax.axis('off')

    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.pause(0.01)


# ============================================================================
# HARDWARE CONFIGURATION FUNCTIONS
# ============================================================================

def configure_camera(core, exposure_ms):
    """Configure camera for internal trigger mode with diSPIM ROI."""
    print(f"\n[CAMERA] Configuring {CAMERA_NAME}...")

    core.setCameraDevice(CAMERA_NAME)

    # Set camera ROI for light sheet imaging
    roi_x = 128
    roi_y = 896
    roi_width = 2048
    roi_height = 512

    print(f"  Setting ROI: X={roi_x}, Y={roi_y}, W={roi_width}, H={roi_height}")
    core.setROI(CAMERA_NAME, roi_x, roi_y, roi_width, roi_height)

    # Configure trigger and exposure
    core.setProperty(CAMERA_NAME, "TRIGGER SOURCE", "INTERNAL")
    core.setProperty(CAMERA_NAME, "SENSOR MODE", "AREA")
    core.setExposure(CAMERA_NAME, exposure_ms)

    time.sleep(0.1)

    trigger_source = core.getProperty(CAMERA_NAME, "TRIGGER SOURCE")
    exposure = core.getExposure(CAMERA_NAME)

    print(f"  TRIGGER SOURCE: {trigger_source}")
    print(f"  Exposure: {exposure} ms")
    print("  ✓ Camera configured")


def configure_galvo_for_calibration(core):
    """Configure galvo for light sheet generation during calibration."""
    print(f"\n[GALVO] Configuring {GALVO_DEVICE}...")

    # Enable beam for light sheet generation
    core.setProperty(GALVO_DEVICE, "BeamEnabled", "Yes")

    # Configure X-axis for light sheet width (scanning)
    # Using 8.0° amplitude to match hardware_triggered_scan
    core.setProperty(GALVO_DEVICE, "SingleAxisXAmplitude(deg)", 8.0)
    core.setProperty(GALVO_DEVICE, "SingleAxisXOffset(deg)", 0.0005)
    core.setProperty(GALVO_DEVICE, "SingleAxisXPattern", "1 - Triangle")
    core.setProperty(GALVO_DEVICE, "SingleAxisXMode", "3 - Enabled with axes synced")

    # Configure Y-axis with minimal amplitude (will adjust offset for positioning)
    core.setProperty(GALVO_DEVICE, "SingleAxisYAmplitude(deg)", 0.0001)
    core.setProperty(GALVO_DEVICE, "SingleAxisYOffset(deg)", 0.0)
    core.setProperty(GALVO_DEVICE, "SingleAxisYPattern", "1 - Triangle")
    core.setProperty(GALVO_DEVICE, "SingleAxisYMode", "3 - Enabled with axes synced")

    time.sleep(0.3)
    print("  ✓ Galvo configured for light sheet (X scanning, Y positioning)")


def set_galvo_y_position(core, angle_deg):
    """Set galvo Y-axis offset to position the light sheet."""
    print(f"  Setting galvo Y offset to {angle_deg:+.3f}°")
    core.setProperty(GALVO_DEVICE, "SingleAxisYOffset(deg)", float(angle_deg))
    time.sleep(0.3)

    actual_offset = float(core.getProperty(GALVO_DEVICE, "SingleAxisYOffset(deg)"))
    print(f"  Galvo Y offset: {actual_offset:+.3f}°")


def capture_image(core):
    """Capture a single image from the camera."""
    core.snapImage()
    img = core.getImage()

    # Handle remote core (rpyc)
    try:
        import rpyc
        img = rpyc.classic.obtain(img)
    except (ImportError, AttributeError):
        pass

    return img


def select_best_camera_view(image):
    """
    Select the best camera view from dual-view diSPIM image.

    DiSPIM captures two views side-by-side (Path A and Path B). This function:
    1. Splits the image into left and right halves
    2. Calculates mean intensity for each half
    3. Returns the brighter half (better signal)

    Args:
        image: Full diSPIM image (numpy array, typically 2048x512)

    Returns:
        Cropped image containing only the better camera view
    """
    h, w = image.shape
    mid_x = w // 2

    # Split into left and right halves
    left_view = image[:, :mid_x]
    right_view = image[:, mid_x:]

    # Calculate mean intensity for each view
    left_intensity = np.mean(left_view)
    right_intensity = np.mean(right_view)

    # Select the brighter view
    if left_intensity >= right_intensity:
        selected_view = left_view
        view_name = "left"
    else:
        selected_view = right_view
        view_name = "right"

    print(f"  Camera view selection: {view_name} (intensity: {left_intensity:.1f} vs {right_intensity:.1f})")

    return selected_view


def compute_fft_bandpass_score(image, lower_cutoff=None, upper_cutoff=None):
    """
    Compute FFT bandpass focus score (ASI diSPIM OughtaFocus implementation).

    This algorithm analyzes the power spectrum of spatial frequencies in the image.
    Well-focused images have more high-frequency content (sharp edges) than defocused
    images, which appear blurred and lack high-frequency components.

    Algorithm:
    1. Compute 2D FFT power spectrum of the image
    2. Create bandpass mask to keep only frequencies in [lower, upper] cutoff range
    3. Calculate mean power within that frequency band

    The default frequency band (2.5% - 14% of maximum) was empirically determined
    by Bill Mohler (UConn) to work well for light sheet microscopy.

    Args:
        image: 2D numpy array (grayscale image)
        lower_cutoff: Lower frequency cutoff as fraction of max frequency (default: FFT_LOWER_CUTOFF)
        upper_cutoff: Upper frequency cutoff as fraction of max frequency (default: FFT_UPPER_CUTOFF)

    Returns:
        float: Mean power in the specified frequency band (higher = better focus)
    """
    if lower_cutoff is None:
        lower_cutoff = FFT_LOWER_CUTOFF
    if upper_cutoff is None:
        upper_cutoff = FFT_UPPER_CUTOFF

    # Ensure image is float for FFT
    img_float = image.astype(np.float64)

    # Compute 2D FFT
    fft = np.fft.fft2(img_float)
    fft_shifted = np.fft.fftshift(fft)  # Move DC component to center

    # Compute power spectrum (magnitude squared)
    power_spectrum = np.abs(fft_shifted) ** 2

    # Create frequency grid for bandpass mask
    h, w = image.shape
    cy, cx = h // 2, w // 2  # Center coordinates

    # Create distance map from center (DC component)
    y, x = np.ogrid[:h, :w]
    distance_from_center = np.sqrt((x - cx)**2 + (y - cy)**2)

    # Maximum possible frequency (corner of image)
    max_freq = np.sqrt(cx**2 + cy**2)

    # Normalized distance (0 at DC, 1 at corners)
    normalized_distance = distance_from_center / max_freq

    # Create bandpass mask: keep frequencies in [lower_cutoff, upper_cutoff]
    bandpass_mask = (normalized_distance >= lower_cutoff) & (normalized_distance <= upper_cutoff)

    # Apply mask and compute mean power in band
    masked_power = power_spectrum * bandpass_mask

    if np.sum(bandpass_mask) > 0:
        mean_power = np.sum(masked_power) / np.sum(bandpass_mask)
    else:
        mean_power = 0.0

    return mean_power


def gaussian_function(x, norm, mean, sigma, offset):
    """
    Gaussian function for curve fitting: f(x) = norm * exp(-((x-mean)²/(2*sigma²))) + offset

    Args:
        x: Independent variable (z-positions in µm)
        norm: Peak height above offset
        mean: Center position (best focus position in µm)
        sigma: Standard deviation (width of focus curve)
        offset: Background/minimum value

    Returns:
        Gaussian curve values
    """
    return norm * np.exp(-((x - mean)**2) / (2 * sigma**2)) + offset


def fit_gaussian_curve(positions, scores):
    """
    Fit Gaussian curve to focus scores and calculate R² goodness of fit.

    This follows the ASI diSPIM plugin approach (GaussianWithOffsetCurveFitter.java):
    1. Find max and min scores to estimate initial parameters
    2. Fit Gaussian: f(z) = norm * exp(-((z-mean)²/(2*sigma²))) + offset
    3. Calculate R² to assess fit quality
    4. Validate that peak is in center 80% of sweep range

    Args:
        positions: Array of z-positions in µm
        scores: Array of focus scores at each position

    Returns:
        dict with keys:
            - 'success': bool, True if fit succeeded
            - 'best_position': float, z-position of best focus (mean of Gaussian)
            - 'r_squared': float, goodness of fit (0-1, higher is better)
            - 'params': dict, fitted parameters (norm, mean, sigma, offset)
            - 'peak_in_center': bool, True if peak in center 80% of range
            - 'error_message': str, error description if fit failed
    """
    positions = np.array(positions)
    scores = np.array(scores)

    if len(positions) < 4:
        return {
            'success': False,
            'error_message': 'Need at least 4 points for Gaussian fitting',
            'r_squared': 0.0
        }

    try:
        # Estimate initial parameters (following ASI plugin approach)
        max_score = np.max(scores)
        min_score = np.min(scores)
        max_idx = np.argmax(scores)

        # Initial guesses
        norm_guess = max_score - min_score  # Peak height above offset
        mean_guess = positions[max_idx]     # Position of maximum score
        offset_guess = min_score             # Background level

        # Estimate FWHM (full width at half maximum) and convert to sigma
        half_max = (max_score + min_score) / 2
        above_half = scores > half_max
        if np.sum(above_half) >= 2:
            fwhm_estimate = np.ptp(positions[above_half])  # Peak-to-peak
            sigma_guess = fwhm_estimate / (2 * np.sqrt(2 * np.log(2)))  # FWHM = 2.355 * sigma
        else:
            sigma_guess = (np.max(positions) - np.min(positions)) / 4

        # Bounds: prevent negative sigma, keep mean within sweep range
        bounds = (
            [0, np.min(positions), 0.1, 0],  # Lower bounds
            [np.inf, np.max(positions), np.inf, np.inf]  # Upper bounds
        )

        # Perform curve fit
        params, covariance = curve_fit(
            gaussian_function,
            positions,
            scores,
            p0=[norm_guess, mean_guess, sigma_guess, offset_guess],
            bounds=bounds,
            maxfev=2000
        )

        norm_fit, mean_fit, sigma_fit, offset_fit = params

        # Calculate R² (coefficient of determination)
        fitted_scores = gaussian_function(positions, *params)
        ss_res = np.sum((scores - fitted_scores)**2)  # Residual sum of squares
        ss_tot = np.sum((scores - np.mean(scores))**2)  # Total sum of squares

        if ss_tot > 0:
            r_squared = 1 - (ss_res / ss_tot)
        else:
            r_squared = 0.0

        # Check if peak is in center 80% of sweep range (reject edge artifacts)
        pos_range = np.max(positions) - np.min(positions)
        edge_margin = pos_range * FOCUS_EDGE_EXCLUSION  # 20% on each side
        pos_min_allowed = np.min(positions) + edge_margin
        pos_max_allowed = np.max(positions) - edge_margin
        peak_in_center = (mean_fit >= pos_min_allowed) and (mean_fit <= pos_max_allowed)

        return {
            'success': True,
            'best_position': float(mean_fit),
            'r_squared': float(r_squared),
            'params': {
                'norm': float(norm_fit),
                'mean': float(mean_fit),
                'sigma': float(sigma_fit),
                'offset': float(offset_fit)
            },
            'peak_in_center': peak_in_center,
            'positions': positions,
            'scores': scores,
            'fitted_scores': fitted_scores
        }

    except Exception as e:
        return {
            'success': False,
            'error_message': f'Gaussian fit failed: {str(e)}',
            'r_squared': 0.0
        }


def save_image(image, position, label):
    """Save image to disk with descriptive filename."""
    IMAGE_DIR.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f"{label}_{position:.2f}um_{timestamp}.png"
    filepath = IMAGE_DIR / filename

    # Convert to 8-bit for saving
    if image.dtype != np.uint8:
        if image.max() > 255:
            image_8bit = (image / image.max() * 255).astype(np.uint8)
        else:
            image_8bit = image.astype(np.uint8)
    else:
        image_8bit = image

    Image.fromarray(image_8bit).save(filepath)
    return filepath


# ============================================================================
# MONTAGE CREATION
# ============================================================================

def create_image_montage(images, positions, title, cols=4):
    """
    Create a labeled montage grid from a list of images.

    Args:
        images: List of numpy arrays
        positions: List of piezo positions (µm)
        title: Title for the montage
        cols: Number of columns in grid

    Returns:
        PIL Image object (montage)
    """
    n_images = len(images)
    rows = (n_images + cols - 1) // cols  # Ceiling division

    # Convert images to 8-bit PIL images
    pil_images = []
    for img in images:
        if img.dtype != np.uint8:
            if img.max() > 255:
                img_8bit = (img / img.max() * 255).astype(np.uint8)
            else:
                img_8bit = img.astype(np.uint8)
        else:
            img_8bit = img

        pil_img = Image.fromarray(img_8bit)

        # Downsample if too large (for Claude API 5MB limit)
        if pil_img.width > 1024:
            new_width = pil_img.width // 2
            new_height = pil_img.height // 2
            pil_img = pil_img.resize((new_width, new_height), Image.Resampling.LANCZOS)

        pil_images.append(pil_img)

    # Get dimensions
    img_width, img_height = pil_images[0].size
    label_height = 40  # Height for label text

    # Create montage canvas
    montage_width = cols * img_width
    montage_height = rows * (img_height + label_height)
    montage = Image.new('L', (montage_width, montage_height), color=0)

    # Try to load a font
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()

    draw = ImageDraw.Draw(montage)

    # Place images in grid
    for idx, (pil_img, pos) in enumerate(zip(pil_images, positions)):
        row = idx // cols
        col = idx % cols

        # Position in montage
        x = col * img_width
        y = row * (img_height + label_height)

        # Paste image
        montage.paste(pil_img, (x, y))

        # Add label
        label_text = f"Pos {idx+1}: {pos:.2f}µm"
        label_y = y + img_height + 5

        # Draw text with background for visibility
        draw.rectangle([x, label_y - 2, x + img_width, label_y + 32], fill=128)
        draw.text((x + 10, label_y), label_text, fill=255, font=font)

    return montage


# ============================================================================
# PHASE 1: INITIAL CENTERING VERIFICATION
# ============================================================================

def verify_embryo_centering(core, fig, ax):
    """
    Phase 1: Verify that embryo is visible and centered at piezo=0, galvo=0.

    Returns:
        bool: True if embryo is visible and centered, False otherwise
    """
    print(f"\n{'='*70}")
    print("PHASE 1: INITIAL CENTERING VERIFICATION")
    print(f"{'='*70}")
    print("Checking if embryo is visible at center position...")
    print("  Piezo: 0 µm")
    print("  Galvo Y: 0°")
    print("  Light sheet: ON (X-axis scanning)")

    # Set piezo to center
    core.setFocusDevice(PIEZO_DEVICE)
    core.setPosition(0.0)
    core.waitForDevice(PIEZO_DEVICE)
    time.sleep(0.3)

    # Set galvo Y to center
    set_galvo_y_position(core, GALVO_Y_CENTER)

    # Capture image
    print("\n  Capturing image at center position...")
    img_full = capture_image(core)

    # Select best camera view (crop to single view)
    img = select_best_camera_view(img_full)

    # Save cropped image
    img_path = save_image(img, 0.0, "embryo_centering_check")
    print(f"  Image saved: {img_path}")

    # Display cropped image
    update_matplotlib_image(fig, ax, img, {
        'phase': 'PHASE 1: CENTERING',
        'stage': 'Initial Check',
        'position': '0.00',
        'galvo': '0.000',
        'note': 'Embryo visible?'
    })

    # Ask Claude if embryo is visible
    print("\n  Sending image to Claude for analysis...")
    try:
        response = ask_claude_with_image(
            _anthropic_client,
            EMBRYO_CENTERING_PROMPT,
            img_path,
            max_tokens=256
        )

        print(f"\n  Claude's response:")
        print(f"  {response}")

        # Parse response - expect first line to be "yes" or "no"
        lines = response.strip().split('\n')
        answer_line = lines[0].strip().lower()

        embryo_visible = 'yes' in answer_line

        if embryo_visible:
            print(f"\n  ✓ SUCCESS: Embryo is visible and centered")
            return True
        else:
            print(f"\n  ✗ FAILURE: Embryo not visible or not centered")
            print(f"\n  Please adjust:")
            print(f"    1. Sample position (XY stage)")
            print(f"    2. Coarse focus (Z stage)")
            print(f"    3. Laser power / camera exposure")
            return False

    except Exception as e:
        print(f"  ERROR calling Claude API: {e}")
        print(f"\n  Asking user for manual confirmation...")

        # Fallback to manual confirmation
        response = input("\n  Is embryo visible and centered in the image? (y/n): ").strip().lower()
        return response == 'y'


# ============================================================================
# PHASE 2 & 3: AUTOMATED FOCUS CALIBRATION
# ============================================================================

def assess_focus_state_embryo(core, fig, ax, galvo_name, galvo_angle, piezo_position):
    """
    Assess current embryo focus state by asking Claude API to rate the focus quality.

    Returns:
        int: Focus rating (1-5)
    """
    print(f"\n[FOCUS ASSESSMENT] {galvo_name} - Piezo: {piezo_position:.2f} µm")

    # Capture image
    img_full = capture_image(core)

    # Select best camera view (crop to single view)
    img = select_best_camera_view(img_full)

    # Save cropped image
    img_path = save_image(img, piezo_position, f"{galvo_name.lower()}_assessment")

    # Update display
    update_matplotlib_image(fig, ax, img, {
        'phase': 'PHASE 2/3: CALIBRATION',
        'stage': f'{galvo_name} Focus Assessment',
        'position': f'{piezo_position:.2f}',
        'galvo': f'{galvo_angle:+.2f}',
        'note': 'Assessing focus'
    })

    print(f"  Sending to Claude for focus quality rating...")

    try:
        response = ask_claude_with_image(
            _anthropic_client,
            EMBRYO_FOCUS_STATE_PROMPT,
            img_path,
            max_tokens=50
        )

        # Parse rating
        response_clean = response.strip()
        print(f"  Claude's rating: {response_clean}")

        import re
        match = re.search(r'[1-5]', response_clean)
        if match:
            rating = int(match.group())
        else:
            print(f"  WARNING: Could not parse rating, defaulting to 3")
            rating = 3

    except Exception as e:
        print(f"  ERROR calling Claude API: {e}")
        rating = 3

    print(f"  Focus rating: {rating}/5")
    return rating


# ============================================================================
# PHASE 1.5: EMBRYO EDGE DETECTION (Adaptive Extent Detection)
# ============================================================================

def detect_embryo_presence_auto(image):
    """
    Automatic fallback for embryo presence detection (if Claude API fails).

    Compares center region signal to background edges to determine if embryo is present.

    Args:
        image: numpy array (cropped single view)

    Returns:
        bool: True if embryo appears to be present based on intensity threshold
    """
    h, w = image.shape
    edge_margin = 50

    # Get background level from edge regions
    edge_pixels = np.concatenate([
        image[:edge_margin, :].flatten(),
        image[-edge_margin:, :].flatten(),
        image[:, :edge_margin].flatten(),
        image[:, -edge_margin:].flatten()
    ])

    background_level = np.median(edge_pixels)

    # Check center region for signal
    center_region = image[h//4:3*h//4, w//4:3*w//4]
    mean_signal = np.mean(center_region)

    # Threshold: center should be at least 20% brighter than background
    threshold_ratio = 1.2
    is_present = mean_signal > (background_level * threshold_ratio)

    print(f"    Auto-detection: background={background_level:.1f}, center={mean_signal:.1f}, "
          f"ratio={mean_signal/background_level:.2f}, present={is_present}")

    return is_present


def detect_embryo_top_edge(core, fig, ax):
    """
    Phase 1.5a: Detect TOP edge of embryo (where embryo disappears going upward).

    Sweeps galvo from CENTER (0°) toward NEGATIVE (physically above) until embryo disappears.
    Uses Claude vision to detect embryo presence, with automatic fallback.

    Returns:
        dict: {
            'galvo_position': float,  # Detected edge position (degrees)
            'galvo_position_with_tolerance': float,  # With safety margin
            'piezo_position': float,  # Piezo position used during detection
            'detection_images': list  # All captured images
        }
    """
    print(f"\n{'='*70}")
    print("PHASE 1.5a: DETECTING EMBRYO TOP EDGE")
    print(f"{'='*70}")
    print(f"Sweeping galvo from {EDGE_DETECTION_START:.3f}° toward {EDGE_DETECTION_END_TOP:.3f}°")
    print(f"Step size: {EDGE_DETECTION_STEP:.3f}° (~{EDGE_DETECTION_STEP*100:.1f}µm with typical calibration)")
    print(f"Strategy: Start at CENTER (embryo confirmed) and sweep UP until it disappears")
    print(f"Heuristic: Moving piezo {HEURISTIC_CALIBRATION_SLOPE:.1f} µm/deg to track light sheet\n")

    # Set piezo as focus device
    core.setFocusDevice(PIEZO_DEVICE)

    # Generate sweep positions (from center toward negative - sweeping UP)
    # Use negative step to go from 0 toward -0.5
    positions = np.arange(EDGE_DETECTION_START, EDGE_DETECTION_END_TOP - EDGE_DETECTION_STEP/2,
                          -EDGE_DETECTION_STEP)

    print(f"Scanning {len(positions)} positions...")

    # Sweep through positions
    images = []
    last_visible_position = EDGE_DETECTION_START  # Default to center if embryo never disappears

    for i, galvo_pos in enumerate(positions):
        # Move galvo
        set_galvo_y_position(core, galvo_pos)

        # ALSO move piezo to track light sheet using heuristic calibration
        piezo_pos = HEURISTIC_CALIBRATION_SLOPE * galvo_pos + HEURISTIC_CALIBRATION_OFFSET
        core.setPosition(float(piezo_pos))
        core.waitForDevice(PIEZO_DEVICE)
        time.sleep(0.2)  # Allow piezo to settle

        # Capture image (embryo should stay in focus!)
        img_full = capture_image(core)
        img = select_best_camera_view(img_full)
        images.append(img)

        # Save image
        img_path = save_image(img, galvo_pos, f"top_edge_detection")

        # Update display
        update_matplotlib_image(fig, ax, img, {
            'phase': 'PHASE 1.5: EDGE DETECTION',
            'stage': 'Top Edge Sweep',
            'position': f'{piezo_pos:.1f}',
            'galvo': f'{galvo_pos:+.3f}',
            'note': f'{i+1}/{len(positions)}'
        })

        print(f"  [{i+1}/{len(positions)}] Galvo: {galvo_pos:+.3f}°, Piezo: {piezo_pos:+.1f}µm ... ", end='')

        # Ask Claude if embryo is visible
        try:
            response = ask_claude_with_image(
                _anthropic_client,
                EMBRYO_EDGE_DETECTION_PROMPT,
                img_path,
                max_tokens=128
            )

            # Parse response
            lines = response.strip().split('\n')
            answer_line = lines[0].strip().lower()
            embryo_present = 'yes' in answer_line

            print(f"Claude: {'YES (embryo visible)' if embryo_present else 'NO (empty)'}")

        except Exception as e:
            print(f"Claude API error, using auto-detection... ", end='')
            embryo_present = detect_embryo_presence_auto(img)

        # Track last position where embryo was visible
        if embryo_present:
            last_visible_position = galvo_pos
        else:
            # Embryo disappeared - we found the top edge!
            print(f"\n  ✓ EMBRYO DISAPPEARED at galvo = {galvo_pos:+.3f}°")
            print(f"  ✓ LAST VISIBLE POSITION was galvo = {last_visible_position:+.3f}°")
            break

    detected_position = last_visible_position

    # Add tolerance margin (absolute degrees beyond detected edge)
    position_with_tolerance = detected_position - EMBRYO_EDGE_TOLERANCE_DEG  # Move more negative (higher up)

    # Calculate corresponding piezo positions
    detected_piezo = HEURISTIC_CALIBRATION_SLOPE * detected_position + HEURISTIC_CALIBRATION_OFFSET
    piezo_with_tolerance = HEURISTIC_CALIBRATION_SLOPE * position_with_tolerance + HEURISTIC_CALIBRATION_OFFSET

    print(f"\n  Detected edge:")
    print(f"    Galvo: {detected_position:+.3f}° → Piezo: {detected_piezo:+.1f} µm")
    print(f"  With {EMBRYO_EDGE_TOLERANCE_DEG:.2f}° tolerance:")
    print(f"    Galvo: {position_with_tolerance:+.3f}° → Piezo: {piezo_with_tolerance:+.1f} µm")
    print(f"    (This ensures at least {abs(detected_position - position_with_tolerance)/EDGE_DETECTION_STEP:.0f} empty frames)")

    return {
        'galvo_position': detected_position,
        'galvo_position_with_tolerance': position_with_tolerance,
        'piezo_position': detected_piezo,
        'detection_images': images
    }


def detect_embryo_bottom_edge(core, fig, ax):
    """
    Phase 1.5b: Detect BOTTOM edge of embryo (where embryo disappears going downward).

    Sweeps galvo from CENTER (0°) toward POSITIVE (physically below) until embryo disappears.
    Uses Claude vision to detect embryo presence, with automatic fallback.

    Returns:
        dict: {
            'galvo_position': float,  # Detected edge position (degrees)
            'galvo_position_with_tolerance': float,  # With safety margin
            'piezo_position': float,  # Piezo position used during detection
            'detection_images': list  # All captured images
        }
    """
    print(f"\n{'='*70}")
    print("PHASE 1.5b: DETECTING EMBRYO BOTTOM EDGE")
    print(f"{'='*70}")
    print(f"Sweeping galvo from {EDGE_DETECTION_START:.3f}° toward {EDGE_DETECTION_END_BOTTOM:.3f}°")
    print(f"Step size: {EDGE_DETECTION_STEP:.3f}° (~{EDGE_DETECTION_STEP*100:.1f}µm with typical calibration)")
    print(f"Strategy: Start at CENTER (embryo confirmed) and sweep DOWN until it disappears")
    print(f"Heuristic: Moving piezo {HEURISTIC_CALIBRATION_SLOPE:.1f} µm/deg to track light sheet\n")

    # Set piezo as focus device
    core.setFocusDevice(PIEZO_DEVICE)

    # Generate sweep positions (from center toward positive - sweeping DOWN)
    positions = np.arange(EDGE_DETECTION_START, EDGE_DETECTION_END_BOTTOM + EDGE_DETECTION_STEP/2,
                          EDGE_DETECTION_STEP)

    print(f"Scanning {len(positions)} positions...")

    # Sweep through positions
    images = []
    last_visible_position = EDGE_DETECTION_START  # Default to center if embryo never disappears

    for i, galvo_pos in enumerate(positions):
        # Move galvo
        set_galvo_y_position(core, galvo_pos)

        # ALSO move piezo to track light sheet using heuristic calibration
        piezo_pos = HEURISTIC_CALIBRATION_SLOPE * galvo_pos + HEURISTIC_CALIBRATION_OFFSET
        core.setPosition(float(piezo_pos))
        core.waitForDevice(PIEZO_DEVICE)
        time.sleep(0.2)  # Allow piezo to settle

        # Capture image (embryo should stay in focus!)
        img_full = capture_image(core)
        img = select_best_camera_view(img_full)
        images.append(img)

        # Save image
        img_path = save_image(img, galvo_pos, f"bottom_edge_detection")

        # Update display
        update_matplotlib_image(fig, ax, img, {
            'phase': 'PHASE 1.5: EDGE DETECTION',
            'stage': 'Bottom Edge Sweep',
            'position': f'{piezo_pos:.1f}',
            'galvo': f'{galvo_pos:+.3f}',
            'note': f'{i+1}/{len(positions)}'
        })

        print(f"  [{i+1}/{len(positions)}] Galvo: {galvo_pos:+.3f}°, Piezo: {piezo_pos:+.1f}µm ... ", end='')

        # Ask Claude if embryo is visible
        try:
            response = ask_claude_with_image(
                _anthropic_client,
                EMBRYO_EDGE_DETECTION_PROMPT,
                img_path,
                max_tokens=128
            )

            # Parse response
            lines = response.strip().split('\n')
            answer_line = lines[0].strip().lower()
            embryo_present = 'yes' in answer_line

            print(f"Claude: {'YES (embryo visible)' if embryo_present else 'NO (empty)'}")

        except Exception as e:
            print(f"Claude API error, using auto-detection... ", end='')
            embryo_present = detect_embryo_presence_auto(img)

        # Track last position where embryo was visible
        if embryo_present:
            last_visible_position = galvo_pos
        else:
            # Embryo disappeared - we found the bottom edge!
            print(f"\n  ✓ EMBRYO DISAPPEARED at galvo = {galvo_pos:+.3f}°")
            print(f"  ✓ LAST VISIBLE POSITION was galvo = {last_visible_position:+.3f}°")
            break

    detected_position = last_visible_position

    # Add tolerance margin (absolute degrees beyond detected edge)
    position_with_tolerance = detected_position + EMBRYO_EDGE_TOLERANCE_DEG  # Move more positive (lower down)

    # Calculate corresponding piezo positions
    detected_piezo = HEURISTIC_CALIBRATION_SLOPE * detected_position + HEURISTIC_CALIBRATION_OFFSET
    piezo_with_tolerance = HEURISTIC_CALIBRATION_SLOPE * position_with_tolerance + HEURISTIC_CALIBRATION_OFFSET

    print(f"\n  Detected edge:")
    print(f"    Galvo: {detected_position:+.3f}° → Piezo: {detected_piezo:+.1f} µm")
    print(f"  With {EMBRYO_EDGE_TOLERANCE_DEG:.2f}° tolerance:")
    print(f"    Galvo: {position_with_tolerance:+.3f}° → Piezo: {piezo_with_tolerance:+.1f} µm")
    print(f"    (This ensures at least {abs(detected_position - position_with_tolerance)/EDGE_DETECTION_STEP:.0f} empty frames)")

    return {
        'galvo_position': detected_position,
        'galvo_position_with_tolerance': position_with_tolerance,
        'piezo_position': detected_piezo,
        'detection_images': images
    }


# ============================================================================
# PHASE 2 & 3: AUTOMATED FOCUS CALIBRATION
# ============================================================================

def perform_focus_sweep_embryo(core, fig, ax, galvo_name, galvo_angle,
                                center_pos, sweep_range, sweep_step, phase_name="PHASE 2/3"):
    """
    Perform focus sweep for embryo at specified galvo position.

    Args:
        core: Micro-Manager core
        fig, ax: Matplotlib figure and axis
        galvo_name: "TOP" or "BOTTOM"
        galvo_angle: Galvo Y angle in degrees
        center_pos: Center position for sweep (µm)
        sweep_range: Range around center (µm)
        sweep_step: Step size (µm)
        phase_name: Phase label for display (default: "PHASE 2/3")

    Returns:
        dict: {
            'optimal_position': float,
            'optimal_image': np.ndarray,
            'all_positions': list,
            'all_images': list
        }
    """
    print(f"\n{'='*70}")
    print(f"FOCUS SWEEP - {galvo_name}")
    print(f"{'='*70}")
    print(f"Galvo Y: {galvo_angle:+.3f}°")
    print(f"Center: {center_pos:.1f} µm")
    print(f"Range: ±{sweep_range:.1f} µm")
    print(f"Step: {sweep_step} µm")

    # Set piezo as focus device
    core.setFocusDevice(PIEZO_DEVICE)

    # Set galvo position
    set_galvo_y_position(core, galvo_angle)

    # Generate sweep positions
    sweep_start = center_pos - sweep_range
    sweep_end = center_pos + sweep_range
    positions = np.arange(sweep_start, sweep_end + sweep_step/2, sweep_step)

    print(f"\nScanning {len(positions)} positions: {sweep_start:.1f} to {sweep_end:.1f} µm")

    # Perform sweep
    images = []
    for i, pos in enumerate(positions):
        # Move piezo
        core.setPosition(float(pos))
        core.waitForDevice(PIEZO_DEVICE)
        time.sleep(0.15)

        # Capture image
        img_full = capture_image(core)

        # Select best camera view (crop to single view)
        img = select_best_camera_view(img_full)
        images.append(img)

        # Save cropped image to disk
        save_image(img, pos, f"{galvo_name.lower()}_sweep")

        # Update display with cropped image
        update_matplotlib_image(fig, ax, img, {
            'phase': phase_name,
            'stage': f'{galvo_name} Focus Sweep',
            'position': f'{pos:.2f}',
            'galvo': f'{galvo_angle:+.2f}',
            'note': f'{i+1}/{len(positions)}'
        })

        print(f"  [{i+1}/{len(positions)}] {pos:.2f} µm captured")

    print(f"\n  ✓ Sweep complete. Captured {len(images)} images.")

    # ===== HYBRID FOCUS DETECTION =====
    # Primary: FFT Bandpass algorithm with Gaussian curve fitting
    # Validation: Claude vision confirmation

    # Compute FFT bandpass scores for all images
    print(f"\n[FFT ANALYSIS] Computing FFT bandpass focus scores...")
    print(f"  Frequency band: {FFT_LOWER_CUTOFF*100:.1f}% - {FFT_UPPER_CUTOFF*100:.1f}% of max frequency")

    fft_scores = []
    for i, img in enumerate(images):
        score = compute_fft_bandpass_score(img)
        fft_scores.append(score)
        print(f"  [{i+1}/{len(images)}] Position {positions[i]:.2f} µm: FFT score = {score:.2e}")

    # Fit Gaussian curve to focus scores
    print(f"\n[GAUSSIAN FIT] Fitting Gaussian curve to focus scores...")
    fit_result = fit_gaussian_curve(positions, fft_scores)

    if fit_result['success']:
        r_squared = fit_result['r_squared']
        best_position_fft = fit_result['best_position']
        peak_in_center = fit_result['peak_in_center']

        print(f"  ✓ Gaussian fit successful")
        print(f"    Peak position: {best_position_fft:.2f} µm")
        print(f"    R² = {r_squared:.3f}")
        print(f"    Sigma = {fit_result['params']['sigma']:.2f} µm")
        print(f"    Peak in center 80%: {peak_in_center}")

        # Validate fit quality
        if r_squared >= MINIMUM_R_SQUARED and peak_in_center:
            print(f"\n  ✓ HIGH CONFIDENCE: R² >= {MINIMUM_R_SQUARED} and peak in center")
            print(f"    Using FFT-based optimal position: {best_position_fft:.2f} µm")

            # Find closest actual position to fitted peak
            best_idx = np.argmin(np.abs(positions - best_position_fft))
            optimal_position = positions[best_idx]
            optimal_image = images[best_idx]

            confidence = "HIGH"
            selection_method = "FFT_Gaussian"

        elif r_squared >= MINIMUM_R_SQUARED:
            print(f"\n  ⚠ MEDIUM CONFIDENCE: R² >= {MINIMUM_R_SQUARED} but peak near edge")
            print(f"    Using FFT result with caution: {best_position_fft:.2f} µm")

            best_idx = np.argmin(np.abs(positions - best_position_fft))
            optimal_position = positions[best_idx]
            optimal_image = images[best_idx]

            confidence = "MEDIUM"
            selection_method = "FFT_Gaussian_EdgeWarning"

        else:
            print(f"\n  ⚠ LOW CONFIDENCE: R² = {r_squared:.3f} < {MINIMUM_R_SQUARED}")
            print(f"    Focus curve is too flat or noisy for reliable fitting")
            print(f"    Using position with maximum FFT score as fallback")

            best_idx = np.argmax(fft_scores)
            optimal_position = positions[best_idx]
            optimal_image = images[best_idx]

            confidence = "LOW"
            selection_method = "FFT_MaxScore_Fallback"
    else:
        print(f"  ✗ Gaussian fit failed: {fit_result['error_message']}")
        print(f"    Using position with maximum FFT score")

        best_idx = np.argmax(fft_scores)
        optimal_position = positions[best_idx]
        optimal_image = images[best_idx]

        confidence = "FALLBACK"
        selection_method = "FFT_MaxScore_FitFailed"

    print(f"\n  → ALGORITHMIC SELECTION: Position #{best_idx+1} ({optimal_position:.2f} µm)")
    print(f"    Method: {selection_method}")
    print(f"    Confidence: {confidence}")

    # Create montage for visualization and Claude validation
    print(f"\n[MONTAGE] Creating analysis montage...")
    montage = create_image_montage(images, positions, f"{galvo_name} Sweep")

    # Save montage
    montage_filename = f"{galvo_name.lower()}_sweep_montage.png"
    montage_path = IMAGE_DIR / montage_filename
    montage.save(montage_path)
    print(f"  ✓ Montage saved: {montage_path}")

    # Display montage
    montage_array = np.array(montage)
    display_montage_matplotlib(fig, ax, montage_array, f"{galvo_name} Montage", phase=phase_name)

    # Claude validation (not selection!)
    print(f"\n[CLAUDE VALIDATION] Validating FFT-selected position...")
    print(f"  Asking Claude to confirm that position #{best_idx+1} looks sharp")

    validation_prompt = f"""You are an expert microscopist examining a diSPIM light sheet microscope image montage of an embryo.

The montage shows {len(positions)} focus positions from a Z-sweep. Each position is labeled with its position number (1 to {len(positions)}).

An FFT-based autofocus algorithm has selected POSITION #{best_idx+1} as the optimal focus.

Your task: Validate this selection by visually confirming the embryo boundary is sharp at this position.

Respond with ONLY ONE LINE in this format:
CONFIRM - Position #{best_idx+1} shows sharp embryo boundaries
OR
REJECT - Position #{best_idx+1} is not optimal, suggest position #[NUMBER] instead

Be specific and direct. Focus on embryo boundary sharpness."""

    try:
        response = ask_claude_with_image(
            _anthropic_client,
            validation_prompt,
            montage_path,
            max_tokens=128
        )

        print(f"\n  Claude's validation:")
        print(f"  {response}")

        # Check if Claude confirms or suggests alternative
        if "CONFIRM" in response.upper():
            print(f"\n  ✓ VALIDATED: Claude confirms position #{best_idx+1}")
            validation_status = "CONFIRMED"
        elif "REJECT" in response.upper():
            print(f"\n  ⚠ DISAGREEMENT: Claude suggests different position")
            print(f"    Using FFT-selected position anyway (algorithmic priority)")
            validation_status = "REJECTED_OVERRIDE"
        else:
            print(f"\n  ? UNCLEAR: Claude response not parseable, assuming confirmation")
            validation_status = "ASSUMED"

    except Exception as e:
        print(f"  ERROR calling Claude API: {e}")
        print(f"  Proceeding with FFT-selected position (no validation)")
        validation_status = "NO_VALIDATION"

    # Move to optimal position
    core.setPosition(float(optimal_position))
    core.waitForDevice(PIEZO_DEVICE)
    time.sleep(0.2)

    # Update display with optimal image
    update_matplotlib_image(fig, ax, optimal_image, {
        'phase': phase_name,
        'stage': f'{galvo_name} Optimal Focus',
        'position': f'{optimal_position:.2f}',
        'galvo': f'{galvo_angle:+.2f}',
        'note': 'SELECTED'
    })

    return {
        'optimal_position': optimal_position,
        'optimal_image': optimal_image,
        'all_positions': positions.tolist(),
        'all_images': images,
        'fft_scores': fft_scores,
        'fit_result': fit_result,
        'confidence': confidence,
        'selection_method': selection_method,
        'validation_status': validation_status
    }


# ============================================================================
# PHASE 4: CALIBRATION CALCULATION
# ============================================================================

def calculate_calibration(galvo_top, piezo_top, galvo_bottom, piezo_bottom):
    """Calculate 2-point linear calibration."""
    print(f"\n{'='*70}")
    print("PHASE 4: CALCULATING CALIBRATION")
    print(f"{'='*70}")

    # Calculate slope (µm/°)
    slope = (piezo_top - piezo_bottom) / (galvo_top - galvo_bottom)

    # Calculate offset
    galvo_center = (galvo_top + galvo_bottom) / 2
    piezo_center = (piezo_top + piezo_bottom) / 2
    offset = piezo_center - (slope * galvo_center)

    print(f"\nCalibration points:")
    print(f"  TOP:    Galvo Y = {galvo_top:+.3f}° → Piezo = {piezo_top:.2f} µm")
    print(f"  BOTTOM: Galvo Y = {galvo_bottom:+.3f}° → Piezo = {piezo_bottom:.2f} µm")
    print(f"\nCalibration formula:")
    print(f"  piezo_position (µm) = {slope:.3f} × galvo_angle (°) + {offset:.3f}")

    # Verify
    piezo_top_check = slope * galvo_top + offset
    piezo_bottom_check = slope * galvo_bottom + offset
    print(f"\nVerification:")
    print(f"  TOP:    {piezo_top:.2f} µm (measured) vs {piezo_top_check:.2f} µm (formula)")
    print(f"  BOTTOM: {piezo_bottom:.2f} µm (measured) vs {piezo_bottom_check:.2f} µm (formula)")

    calibration = {
        'slope_um_per_deg': float(slope),
        'offset_um': float(offset),
        'galvo_top_deg': float(galvo_top),
        'galvo_bottom_deg': float(galvo_bottom),
        'piezo_top_um': float(piezo_top),
        'piezo_bottom_um': float(piezo_bottom),
        'sample_type': 'embryo',
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'device_piezo': PIEZO_DEVICE,
        'device_galvo': GALVO_DEVICE
    }

    return calibration


def save_calibration(calibration, filename=CALIBRATION_FILE):
    """Save calibration to JSON file."""
    with open(filename, 'w') as f:
        json.dump(calibration, f, indent=2)
    print(f"\n✓ Calibration saved to: {filename}")


# ============================================================================
# PHASE 5: VOLUME ACQUISITION TEST
# ============================================================================

def test_volume_acquisition(core, fig, ax, calibration, galvo_y_top, galvo_y_bottom):
    """
    Phase 5: Test volume acquisition using the new calibration.

    This is a simplified version based on hardware_triggered_scan.py

    NOTE: Volume acquisition captures FULL frames (both camera views) to preserve
    all data. The calibration phases use cropped single-view images for analysis,
    but the final volume contains both views for maximum flexibility in post-processing.

    Args:
        core: Micro-Manager core
        fig, ax: Matplotlib figure and axis
        calibration: Calibration dict with slope and offset
        galvo_y_top: Detected top galvo position (degrees)
        galvo_y_bottom: Detected bottom galvo position (degrees)
    """
    print(f"\n{'='*70}")
    print("PHASE 5: VOLUME ACQUISITION TEST")
    print(f"{'='*70}")
    print("  (Acquiring full dual-view frames for complete data preservation)")

    # Calculate galvo parameters from detected embryo extent
    # Use the detected galvo range to determine piezo range via calibration
    slope = calibration['slope_um_per_deg']
    offset = calibration['offset_um']

    # Galvo center is midpoint of detected range
    galvo_center = (galvo_y_top + galvo_y_bottom) / 2

    # Galvo amplitude is half the detected range
    detected_galvo_range = galvo_y_bottom - galvo_y_top
    galvo_amplitude = detected_galvo_range / 2

    # Calculate corresponding piezo parameters
    # Convert galvo positions to piezo using calibration formula
    piezo_top = slope * galvo_y_top + offset
    piezo_bottom = slope * galvo_y_bottom + offset
    detected_piezo_range = piezo_bottom - piezo_top

    PIEZO_CENTER_UM = (piezo_top + piezo_bottom) / 2
    PIEZO_AMPLITUDE_UM = detected_piezo_range / 2

    print(f"\nAdaptive volume parameters (based on detected embryo extent):")
    print(f"  Detected galvo range: {galvo_y_top:+.3f}° to {galvo_y_bottom:+.3f}° ({detected_galvo_range:.3f}° total)")
    print(f"  Corresponding piezo range: {piezo_top:.1f} to {piezo_bottom:.1f} µm ({detected_piezo_range:.1f} µm total)")
    print(f"  Galvo: center={galvo_center:+.4f}°, amplitude=±{galvo_amplitude:.4f}°")
    print(f"  Piezo: center={PIEZO_CENTER_UM:.1f} µm, amplitude=±{PIEZO_AMPLITUDE_UM:.1f} µm")
    print(f"  Slices: {VOLUME_NUM_SLICES}")
    print(f"  Exposure: {VOLUME_EXPOSURE_MS} ms")
    print(f"\n  Benefits: Scan range optimized for embryo size!")
    print(f"  (vs fixed 98µm range, this saves {98.0 - detected_piezo_range:.1f} µm of unnecessary scanning)")

    try:
        # Configure camera for hardware trigger
        print(f"\n[1/5] Configuring camera for hardware trigger...")
        core.setCameraDevice(CAMERA_NAME)
        core.setProperty(CAMERA_NAME, "TRIGGER SOURCE", "EXTERNAL")
        core.setProperty(CAMERA_NAME, "SENSOR MODE", "PROGRESSIVE")  # CRITICAL!
        core.setProperty(CAMERA_NAME, "TRIGGER ACTIVE", "EDGE")
        core.setExposure(CAMERA_NAME, VOLUME_EXPOSURE_MS)
        time.sleep(0.1)
        print(f"  ✓ Camera configured for EXTERNAL trigger (PROGRESSIVE mode)")

        # Configure galvo for SPIM
        print(f"\n[2/5] Configuring galvo for SPIM...")
        core.setProperty(GALVO_DEVICE, "SPIMState", "Idle")
        time.sleep(0.2)

        core.setProperty(GALVO_DEVICE, "LaserOutputMode", "shutter + side")
        core.setProperty(GALVO_DEVICE, "BeamEnabled", "No")

        # X-axis scanning (light sheet width)
        core.setProperty(GALVO_DEVICE, "SingleAxisXAmplitude(deg)", 8.0)
        core.setProperty(GALVO_DEVICE, "SingleAxisXOffset(deg)", 0.0005)
        core.setProperty(GALVO_DEVICE, "SingleAxisXPattern", "1 - Triangle")

        # Y-axis positioning (synchronized with piezo)
        core.setProperty(GALVO_DEVICE, "SingleAxisYAmplitude(deg)", float(galvo_amplitude))
        core.setProperty(GALVO_DEVICE, "SingleAxisYOffset(deg)", float(galvo_center))
        core.setProperty(GALVO_DEVICE, "SingleAxisYPattern", "1 - Triangle")

        # SPIM timing
        core.setProperty(GALVO_DEVICE, "SPIMDelayBeforeScan(ms)", 6.75)
        core.setProperty(GALVO_DEVICE, "SPIMNumScansPerSlice", 1)
        core.setProperty(GALVO_DEVICE, "SPIMScanDuration(ms)", 5.5)
        core.setProperty(GALVO_DEVICE, "SPIMDelayBeforeLaser(ms)", 8.0)
        core.setProperty(GALVO_DEVICE, "SPIMLaserDuration(ms)", 5.0)
        core.setProperty(GALVO_DEVICE, "SPIMDelayBeforeCamera(ms)", 8.0)
        core.setProperty(GALVO_DEVICE, "SPIMCameraDuration(ms)", 1.0)

        core.setProperty(GALVO_DEVICE, "SPIMNumSlices", VOLUME_NUM_SLICES)
        core.setProperty(GALVO_DEVICE, "SPIMNumSlicesPerPiezo", 1)
        core.setProperty(GALVO_DEVICE, "SPIMNumSides", 1)
        core.setProperty(GALVO_DEVICE, "SPIMFirstSide", "A")
        print(f"  ✓ Galvo configured for SPIM")

        # Configure piezo for SPIM
        print(f"\n[3/5] Configuring piezo for SPIM...")
        core.setFocusDevice(PIEZO_DEVICE)
        core.setProperty(PIEZO_DEVICE, "SingleAxisAmplitude(um)", float(PIEZO_AMPLITUDE_UM))
        core.setProperty(PIEZO_DEVICE, "SingleAxisOffset(um)", float(PIEZO_CENTER_UM))
        core.setProperty(PIEZO_DEVICE, "SingleAxisPattern", "1 - Triangle")
        core.setProperty(PIEZO_DEVICE, "SPIMNumSlices", VOLUME_NUM_SLICES)
        core.setProperty(PIEZO_DEVICE, "SPIMState", "Armed")
        time.sleep(0.3)
        print(f"  ✓ Piezo armed")

        # Start acquisition
        print(f"\n[4/5] Starting hardware-triggered acquisition...")
        core.clearCircularBuffer()

        # Set buffer capacity
        buffer_capacity = core.getBufferTotalCapacity()
        if buffer_capacity < VOLUME_NUM_SLICES:
            core.setCircularBufferMemoryFootprint(512)
            time.sleep(0.1)

        core.prepareSequenceAcquisition(CAMERA_NAME)
        time.sleep(0.1)

        core.startSequenceAcquisition(CAMERA_NAME, VOLUME_NUM_SLICES, 0, True)
        time.sleep(0.1)

        # Trigger SPIM state machine
        core.setProperty(GALVO_DEVICE, "SPIMState", "Running")
        print(f"  ✓ Acquisition started")

        # Collect images
        print(f"\n  Waiting for {VOLUME_NUM_SLICES} images...")
        images = []
        timeout_s = VOLUME_NUM_SLICES * VOLUME_SLICE_PERIOD_MS / 1000.0 * 2
        start_time = time.time()

        while core.getRemainingImageCount() > 0 or core.isSequenceRunning():
            if core.getRemainingImageCount() > 0:
                img = core.popNextImage()

                try:
                    import rpyc
                    img = rpyc.classic.obtain(img)
                except (ImportError, AttributeError):
                    pass

                images.append(img)

                if len(images) % 10 == 0:
                    print(f"    Received {len(images)}/{VOLUME_NUM_SLICES} images...")

                # Update display
                if len(images) % 10 == 0:
                    update_matplotlib_image(fig, ax, img, {
                        'phase': 'PHASE 5: VOLUME TEST',
                        'stage': 'Acquiring Stack',
                        'position': '',
                        'galvo': '',
                        'note': f'Slice {len(images)}/{VOLUME_NUM_SLICES}'
                    })

            if time.time() - start_time > timeout_s:
                print(f"\n  WARNING: Timeout!")
                break

            time.sleep(0.01)

        # Stop sequence
        if core.isSequenceRunning():
            core.stopSequenceAcquisition()

        print(f"\n  ✓ Acquisition complete! Received {len(images)} images")

        # Save volume
        print(f"\n[5/5] Saving volume...")
        if len(images) > 0:
            import tifffile

            volume = np.array(images)
            print(f"  Volume shape: {volume.shape} (Z, Y, X)")

            tifffile.imwrite(
                TEST_VOLUME_FILE,
                volume,
                metadata={
                    'axes': 'ZYX',
                    'piezo_center_um': float(PIEZO_CENTER_UM),
                    'piezo_amplitude_um': float(PIEZO_AMPLITUDE_UM),
                    'galvo_center_deg': float(galvo_center),
                    'galvo_amplitude_deg': float(galvo_amplitude),
                    'num_slices': VOLUME_NUM_SLICES,
                    'sample_type': 'embryo',
                    'hardware_triggered': True
                }
            )

            print(f"  ✓ Saved: {TEST_VOLUME_FILE}")
            print(f"\n  ✓ VOLUME ACQUISITION SUCCESS!")

            # Display middle slice
            if len(images) > 0:
                mid_slice = images[len(images)//2]
                update_matplotlib_image(fig, ax, mid_slice, {
                    'phase': 'PHASE 5: VOLUME TEST',
                    'stage': 'Complete - Middle Slice',
                    'position': '',
                    'galvo': '',
                    'note': f'Slice {len(images)//2}/{len(images)}'
                })

        else:
            print(f"\n  ✗ No images captured!")

        # Reset camera to internal trigger
        core.setProperty(CAMERA_NAME, "TRIGGER SOURCE", "INTERNAL")
        core.setProperty(GALVO_DEVICE, "SPIMState", "Idle")
        core.setProperty(PIEZO_DEVICE, "SPIMState", "Idle")

    except Exception as e:
        print(f"\n  ✗ Volume acquisition failed: {e}")
        import traceback
        traceback.print_exc()

        # Try to reset
        try:
            core.stopSequenceAcquisition()
            core.setProperty(CAMERA_NAME, "TRIGGER SOURCE", "INTERNAL")
            core.setProperty(GALVO_DEVICE, "SPIMState", "Idle")
            core.setProperty(PIEZO_DEVICE, "SPIMState", "Idle")
        except:
            pass


# ============================================================================
# CLEANUP
# ============================================================================

def cleanup(core):
    """Reset devices to safe state."""
    print(f"\n{'='*70}")
    print("CLEANUP")
    print(f"{'='*70}")

    try:
        core.setProperty(GALVO_DEVICE, "SingleAxisYOffset(deg)", 0.0)
        print("  ✓ Galvo Y reset to center")
    except Exception as e:
        print(f"  Could not reset galvo: {e}")

    try:
        core.setConfig("Laser", "ALL OFF")
        print("  ✓ Lasers OFF")
    except Exception as e:
        print(f"  Could not turn off lasers: {e}")


# ============================================================================
# MAIN WORKFLOW
# ============================================================================

def main():
    """Main embryo calibration workflow."""
    global _anthropic_client, _core

    print("="*70)
    print("EMBRYO-BASED PIEZO-GALVO CALIBRATION")
    print("Automated with Claude Vision API")
    print("="*70)

    fig, ax = None, None

    try:
        # Initialize Anthropic API
        print("\n[0/7] Initializing Claude API...")
        _anthropic_client = initialize_anthropic_client()
        print("  ✓ Claude API ready")

        # Setup matplotlib
        fig, ax = setup_matplotlib_figure()
        _core = get_mmc()

        # System startup
        print("\n[1/7] System startup...")
        _core.setConfig("System", "Startup")
        _core.waitForConfig("System", "Startup")
        print("  ✓ System configured")

        # Lasers on
        print("\n[2/7] Enabling lasers...")
        _core.setConfig("Laser", "488 and 561")
        _core.waitForConfig("Laser", "488 and 561")
        print("  ✓ Lasers ON")

        # Configure camera and galvo
        print("\n[3/7] Configuring camera and galvo...")
        configure_camera(_core, CAMERA_EXPOSURE_MS)
        configure_galvo_for_calibration(_core)

        IMAGE_DIR.mkdir(exist_ok=True)

        # Load heuristic calibration for edge detection (if available)
        print("\n[3.5/8] Loading heuristic calibration for edge detection...")
        global HEURISTIC_CALIBRATION_SLOPE, HEURISTIC_CALIBRATION_OFFSET
        try:
            # Try loading bead calibration first (more reliable)
            cal_file = Path("piezo_galvo_calibration.json")
            if not cal_file.exists():
                # Fall back to embryo calibration
                cal_file = Path("piezo_galvo_calibration_embryo.json")

            with open(cal_file, 'r') as f:
                prev_cal = json.load(f)
                HEURISTIC_CALIBRATION_SLOPE = prev_cal['slope_um_per_deg']
                HEURISTIC_CALIBRATION_OFFSET = prev_cal['offset_um']
                print(f"  ✓ Loaded from {cal_file.name}:")
                print(f"    Slope: {HEURISTIC_CALIBRATION_SLOPE:.1f} µm/deg")
                print(f"    Offset: {HEURISTIC_CALIBRATION_OFFSET:.3f} µm")
        except FileNotFoundError:
            print(f"  No previous calibration found, using defaults:")
            print(f"    Slope: {HEURISTIC_CALIBRATION_SLOPE:.1f} µm/deg (empirical)")
            print(f"    Offset: {HEURISTIC_CALIBRATION_OFFSET:.3f} µm")

        # ===== PHASE 1: CENTERING CHECK =====
        print("\n[4/8] Phase 1: Centering verification...")
        embryo_centered = verify_embryo_centering(_core, fig, ax)

        if not embryo_centered:
            print("\n✗ CALIBRATION ABORTED: Embryo not visible or centered")
            print("Please adjust sample position and try again.")
            return

        # ===== PHASE 1.5: DETECT EMBRYO Z EXTENT =====
        print("\n[4.5/8] Phase 1.5: Detecting embryo Z extent...")
        print("  This will adaptively find where the embryo starts and ends in Z")
        print("  to optimize scan range and reduce wasted acquisition time.\n")

        # Detect top edge (where embryo first appears from above)
        top_edge_result = detect_embryo_top_edge(_core, fig, ax)
        GALVO_Y_TOP_DETECTED = top_edge_result['galvo_position']
        GALVO_Y_TOP_EDGE = top_edge_result['galvo_position_with_tolerance']

        # Detect bottom edge (where embryo disappears from below)
        bottom_edge_result = detect_embryo_bottom_edge(_core, fig, ax)
        GALVO_Y_BOTTOM_DETECTED = bottom_edge_result['galvo_position']
        GALVO_Y_BOTTOM_EDGE = bottom_edge_result['galvo_position_with_tolerance']

        detected_galvo_range = GALVO_Y_BOTTOM_EDGE - GALVO_Y_TOP_EDGE

        print(f"\n{'─'*70}")
        print("EDGE DETECTION COMPLETE")
        print(f"{'─'*70}")
        print(f"\n  Detected embryo boundaries (for volume scan range):")
        print(f"    TOP edge: {GALVO_Y_TOP_EDGE:+.3f}° (sparse embryo, fading region)")
        print(f"    BOTTOM edge: {GALVO_Y_BOTTOM_EDGE:+.3f}° (sparse embryo, fading region)")
        print(f"    Total range: {detected_galvo_range:.3f}° (~{detected_galvo_range * 100:.1f} µm)")

        # ===== SELECT INTERIOR CALIBRATION POSITIONS =====
        # CRITICAL: Don't calibrate at edges where embryo is sparse!
        # Move inward to find positions with good morphology for accurate focus detection
        print(f"\n{'─'*70}")
        print("SELECTING CALIBRATION POSITIONS")
        print(f"{'─'*70}")
        print(f"  Strategy: Calibrate at INTERIOR positions (not edges)")
        print(f"  Reason: Edges have sparse/fading embryo → poor focus detection → bad calibration")
        print(f"  Interior has sharp morphology → accurate focus detection → good calibration\n")

        inset_amount = detected_galvo_range * CALIBRATION_INSET_FRACTION
        GALVO_Y_TOP_CALIB = GALVO_Y_TOP_EDGE + inset_amount  # Move inward (more positive)
        GALVO_Y_BOTTOM_CALIB = GALVO_Y_BOTTOM_EDGE - inset_amount  # Move inward (more negative)

        print(f"  Moving {CALIBRATION_INSET_FRACTION*100:.0f}% inward from each edge:")
        print(f"    Inset distance: {inset_amount:.3f}° (~{inset_amount * 100:.1f} µm)")
        print(f"\n  TOP calibration position:")
        print(f"    Edge at: {GALVO_Y_TOP_EDGE:+.3f}° → Calibrate at: {GALVO_Y_TOP_CALIB:+.3f}° (good morphology!)")
        print(f"  BOTTOM calibration position:")
        print(f"    Edge at: {GALVO_Y_BOTTOM_EDGE:+.3f}° → Calibrate at: {GALVO_Y_BOTTOM_CALIB:+.3f}° (good morphology!)")

        calib_range = GALVO_Y_BOTTOM_CALIB - GALVO_Y_TOP_CALIB
        print(f"\n  Calibration range: {calib_range:.3f}° (interior positions with sharp features)")
        print(f"  Volume scan range: {detected_galvo_range:.3f}° (full detected extent)")
        print(f"  ✓ Calibration will be based on high-quality interior images,")
        print(f"    then applied to full volume range for best focus throughout!")

        # Update heuristic piezo positions for calibration (not edges!)
        HEURISTIC_TOP_PIEZO = GALVO_Y_TOP_CALIB * HEURISTIC_CALIBRATION_SLOPE
        HEURISTIC_BOTTOM_PIEZO = GALVO_Y_BOTTOM_CALIB * HEURISTIC_CALIBRATION_SLOPE

        print(f"\n  Heuristic piezo positions for calibration:")
        print(f"    Top: {HEURISTIC_TOP_PIEZO:.1f} µm")
        print(f"    Bottom: {HEURISTIC_BOTTOM_PIEZO:.1f} µm")

        # ===== PHASE 2: TOP CALIBRATION =====
        print(f"\n{'='*70}")
        print("[5/8] Phase 2: TOP calibration at INTERIOR position")
        print(f"{'='*70}")
        print(f"  Calibration position: {GALVO_Y_TOP_CALIB:+.3f}° (INTERIOR, good morphology)")
        print(f"  Edge is at: {GALVO_Y_TOP_EDGE:+.3f}° (not used for calibration)")
        print(f"  This ensures sharp, high-contrast images for accurate focus detection\n")

        set_galvo_y_position(_core, GALVO_Y_TOP_CALIB)

        # Move to heuristic position
        print(f"  Moving to heuristic piezo position: {HEURISTIC_TOP_PIEZO:.1f} µm")
        _core.setFocusDevice(PIEZO_DEVICE)
        _core.setPosition(float(HEURISTIC_TOP_PIEZO))
        _core.waitForDevice(PIEZO_DEVICE)
        time.sleep(0.3)

        # Perform focus sweep at interior position
        top_result = perform_focus_sweep_embryo(
            _core, fig, ax, "TOP", GALVO_Y_TOP_CALIB,
            HEURISTIC_TOP_PIEZO, SWEEP_RANGE_UM, SWEEP_STEP_UM,
            phase_name="PHASE 2: TOP CALIBRATION"
        )

        piezo_top_final = top_result['optimal_position']
        print(f"\n  ✓ TOP optimal position: {piezo_top_final:.2f} µm")

        # ===== PHASE 3: BOTTOM CALIBRATION =====
        print(f"\n{'='*70}")
        print("[6/8] Phase 3: BOTTOM calibration at INTERIOR position")
        print(f"{'='*70}")
        print(f"  Calibration position: {GALVO_Y_BOTTOM_CALIB:+.3f}° (INTERIOR, good morphology)")
        print(f"  Edge is at: {GALVO_Y_BOTTOM_EDGE:+.3f}° (not used for calibration)")
        print(f"  This ensures sharp, high-contrast images for accurate focus detection\n")

        set_galvo_y_position(_core, GALVO_Y_BOTTOM_CALIB)

        # Move to heuristic position
        print(f"  Moving to heuristic piezo position: {HEURISTIC_BOTTOM_PIEZO:.1f} µm")
        _core.setPosition(float(HEURISTIC_BOTTOM_PIEZO))
        _core.waitForDevice(PIEZO_DEVICE)
        time.sleep(0.3)

        # Perform focus sweep at interior position
        bottom_result = perform_focus_sweep_embryo(
            _core, fig, ax, "BOTTOM", GALVO_Y_BOTTOM_CALIB,
            HEURISTIC_BOTTOM_PIEZO, SWEEP_RANGE_UM, SWEEP_STEP_UM,
            phase_name="PHASE 3: BOTTOM CALIBRATION"
        )

        piezo_bottom_final = bottom_result['optimal_position']
        print(f"\n  ✓ BOTTOM optimal position: {piezo_bottom_final:.2f} µm")

        # ===== PHASE 4: CALCULATE CALIBRATION =====
        # Use INTERIOR positions for calibration (where focus detection was accurate)
        calibration = calculate_calibration(
            GALVO_Y_TOP_CALIB, piezo_top_final,
            GALVO_Y_BOTTOM_CALIB, piezo_bottom_final
        )

        # Add metadata about edge detection and calibration strategy
        calibration['edge_top_deg'] = float(GALVO_Y_TOP_EDGE)
        calibration['edge_bottom_deg'] = float(GALVO_Y_BOTTOM_EDGE)
        calibration['calib_inset_fraction'] = float(CALIBRATION_INSET_FRACTION)
        calibration['calib_strategy'] = 'interior_positions'

        # Save calibration
        save_calibration(calibration)

        # Summary with detailed explanation
        print(f"\n{'='*70}")
        print("✓ CALIBRATION COMPLETE")
        print(f"{'='*70}")
        print(f"\nCalibration strategy:")
        print(f"  Edge detection found: {GALVO_Y_TOP_EDGE:+.3f}° to {GALVO_Y_BOTTOM_EDGE:+.3f}° (embryo boundaries)")
        print(f"  Calibration performed at: {GALVO_Y_TOP_CALIB:+.3f}° to {GALVO_Y_BOTTOM_CALIB:+.3f}° (interior)")
        print(f"  Inset: {CALIBRATION_INSET_FRACTION*100:.0f}% inward from edges (for sharp features)")
        print(f"\nCalibration parameters (from interior positions):")
        print(f"  SLOPE = {calibration['slope_um_per_deg']:.3f} µm/°")
        print(f"  OFFSET = {calibration['offset_um']:.3f} µm")
        print(f"\nApplication:")
        print(f"  Volume will scan: {GALVO_Y_TOP_EDGE:+.3f}° to {GALVO_Y_BOTTOM_EDGE:+.3f}° (full detected range)")
        print(f"  Using calibration from: {GALVO_Y_TOP_CALIB:+.3f}° to {GALVO_Y_BOTTOM_CALIB:+.3f}° (sharp interior)")
        print(f"  Result: Accurate calibration applied to full range → good focus throughout!")
        print(f"\nFiles:")
        print(f"  Calibration: {CALIBRATION_FILE}")
        print(f"  Images: {IMAGE_DIR}/")

        # ===== PHASE 5: VOLUME ACQUISITION TEST =====
        print(f"\n{'='*70}")
        print("[8/8] Phase 5: Volume acquisition test")
        print(f"{'='*70}")
        print(f"  Scan range: {GALVO_Y_TOP_EDGE:+.3f}° to {GALVO_Y_BOTTOM_EDGE:+.3f}° (full detected extent)")
        print(f"  Calibration from: {GALVO_Y_TOP_CALIB:+.3f}° to {GALVO_Y_BOTTOM_CALIB:+.3f}° (interior positions)")
        print(f"  Expected result: Excellent focus throughout entire volume!\n")
        test_volume_acquisition(_core, fig, ax, calibration, GALVO_Y_TOP_EDGE, GALVO_Y_BOTTOM_EDGE)

        print(f"\n{'='*70}")
        print("✓ ALL PHASES COMPLETE")
        print(f"{'='*70}")
        print(f"\nNext steps:")
        print(f"  1. Review images in: {IMAGE_DIR}/")
        print(f"  2. Check volume: {TEST_VOLUME_FILE}")
        print(f"  3. Use calibration file: {CALIBRATION_FILE}")

        print(f"\nMatplotlib figure is still open for review.")
        print(f"Close the figure window or press Ctrl+C to exit.")

        # Keep matplotlib window open
        plt.ioff()
        plt.show()

    except KeyboardInterrupt:
        print("\n\nCalibration interrupted by user")

    except Exception as e:
        print(f"\n{'='*70}")
        print("ERROR")
        print("="*70)
        print(f"{e}")
        import traceback
        traceback.print_exc()

    finally:
        if _core is not None:
            cleanup(_core)


if __name__ == "__main__":
    main()
