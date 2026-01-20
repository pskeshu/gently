# ASI diSPIM Camera Hardware Triggering - Complete Reference

**Author:** Claude Code + microscope-control-expert agent
**Date:** 2025-10-14
**Purpose:** Comprehensive analysis of ASI diSPIM Java plugin camera triggering system for debugging hardware-triggered acquisition

---

## Table of Contents

1. [Executive Summary](#executive-summary)
2. [Critical Bug Discovery](#critical-bug-discovery)
3. [Camera Configuration by Manufacturer](#camera-configuration-by-manufacturer)
4. [Tiger Controller SPIM Properties](#tiger-controller-spim-properties)
5. [Complete Acquisition Workflow](#complete-acquisition-workflow)
6. [Timing Calculations](#timing-calculations)
7. [Hardware Trigger Output Generation](#hardware-trigger-output-generation)
8. [Debugging Guide](#debugging-guide)
9. [Recommended Python Implementation](#recommended-python-implementation)
10. [Java Source Code Reference](#java-source-code-reference)

---

## Executive Summary

The ASI diSPIM microscope uses a **hardware-triggered acquisition system** where the Tiger controller's SPIM state machine generates TTL pulses to trigger the camera and lasers synchronously. This document provides a complete reference for configuring and debugging this system.

**Key Components:**

- **Camera**: Must be configured for EXTERNAL trigger mode (EDGE, LEVEL, or OVERLAP)
- **Tiger Controller**: Micro-mirror card generates TTL pulses via SPIM state machine
- **SPIM State Machine**: Idle → Armed → Running state flow generates synchronized triggers
- **TTL Outputs**: BNC2 (TTL1) = Camera/Laser trigger, BNC4 (TTL3) = Side indicator

**Workflow:**
1. Configure camera for external triggering
2. Set Tiger controller SPIM timing properties
3. Start camera sequence acquisition (waits for triggers)
4. Set SPIM state to "Running" (generates TTL pulses)
5. Camera captures images on each trigger pulse

---

## Quick Start - Verified Working Configuration

**Tested System:** ASI diSPIM with Hamamatsu Flash4 camera
**Performance:** 100 slices @ 59.1 fps (1.7 seconds total)
**Status:** ✅ **FULLY WORKING** (2025-01-14)

### Minimal Working Example

```python
from client import get_mmc
import numpy as np

core = get_mmc()

# Device names (adjust for your system)
camera_name = "HamCam1"
galvo_device = "Scanner:AB:33"

# Acquisition parameters
num_slices = 100
camera_exposure_ms = 5.0

# 1. Apply system configuration
core.setConfig("System", "Startup")
core.waitForConfig("System", "Startup")

# 2. Turn on lasers
core.setConfig("Laser", "488 and 561")
core.waitForConfig("Laser", "488 and 561")

# 3. Configure camera - CRITICAL: Use PROGRESSIVE mode!
core.setCameraDevice(camera_name)
core.setProperty(camera_name, "TRIGGER SOURCE", "EXTERNAL")
core.setProperty(camera_name, "SENSOR MODE", "PROGRESSIVE")  # NOT "AREA"!
core.setProperty(camera_name, "TRIGGER ACTIVE", "EDGE")
core.setExposure(camera_name, camera_exposure_ms)

# 4. Configure Tiger controller
core.setProperty(galvo_device, "SPIMState", "Idle")
time.sleep(0.2)

# CRITICAL: Set laser output mode
core.setProperty(galvo_device, "LaserOutputMode", "shutter + side")

# Configure galvo X-axis (light sheet)
core.setProperty(galvo_device, "SingleAxisXAmplitude(deg)", 2.0)
core.setProperty(galvo_device, "SingleAxisXOffset(deg)", 0.0)
core.setProperty(galvo_device, "SingleAxisXPattern", "1 - Triangle")
core.setProperty(galvo_device, "SingleAxisXMode", "3 - Enabled with axes synced")

# Configure galvo Y-axis (slice stepping)
core.setProperty(galvo_device, "SingleAxisYAmplitude(deg)", 0.04)
core.setProperty(galvo_device, "SingleAxisYOffset(deg)", 0.0)
core.setProperty(galvo_device, "SingleAxisYPattern", "1 - Triangle")
core.setProperty(galvo_device, "SingleAxisYMode", "3 - Enabled with axes synced")

# CRITICAL: Set SPIM timing properties explicitly
core.setProperty(galvo_device, "SPIMNumSlices", num_slices)
core.setProperty(galvo_device, "SPIMNumSides", 1)
core.setProperty(galvo_device, "SPIMDelayBeforeScan(ms)", 11.0)
core.setProperty(galvo_device, "SPIMScanDuration(ms)", 5.5)
core.setProperty(galvo_device, "SPIMDelayBeforeLaser(ms)", 13.0)
core.setProperty(galvo_device, "SPIMLaserDuration(ms)", 5.0)
core.setProperty(galvo_device, "SPIMDelayBeforeCamera(ms)", 10.0)
core.setProperty(galvo_device, "SPIMCameraDuration(ms)", 1.0)  # MUST BE > 0!

# 5. Configure circular buffer
core.clearCircularBuffer()
core.setCircularBufferMemoryFootprint(1200)  # MB

# 6. Start camera sequence
core.prepareSequenceAcquisition(camera_name)
time.sleep(0.1)
core.startSequenceAcquisition(camera_name, num_slices, 0, True)

# 7. Trigger SPIM
core.setProperty(galvo_device, "SPIMState", "Running")

# 8. Wait for images
while core.getRemainingImageCount() < num_slices:
    time.sleep(0.01)

# 9. Retrieve images
images = []
for i in range(num_slices):
    img = core.popNextImage()
    try:
        import rpyc
        img = rpyc.classic.obtain(img)
    except:
        pass
    images.append(img)

volume = np.array(images)
print(f"Success! Acquired volume: {volume.shape}")

# 10. Cleanup
core.stopSequenceAcquisition()
core.setProperty(galvo_device, "SPIMState", "Idle")
core.setConfig("Laser", "ALL OFF")
```

### Critical Requirements for Success

1. ✅ **SENSOR MODE = "PROGRESSIVE"** - AREA mode will NOT work
2. ✅ **SPIMCameraDuration(ms) > 0** - Must be set explicitly
3. ✅ **LaserOutputMode = "shutter + side"** - Enables TTL outputs
4. ✅ **Circular buffer configured** - Needs ~1200MB for 100 slices
5. ✅ **Galvo Y-axis for slice stepping** - NOT the piezo!

**Full working script:** See `gently/test_volume_acq.py`

---

## Critical Bug Discovery and Solution

### ⚠️ Issue #1: Timing Properties Not Written to Tiger Controller

**The ASI diSPIM Java plugin has a critical issue in "Simple Timing" mode:**

The plugin calculates these timing values in `getTimingFromPeriodAndLightExposure()`:
- `scanDelay` - Delay before starting scan mirror
- `scanPeriod` - Scan mirror sweep duration
- `laserDelay` - Delay before laser trigger
- `laserDuration` - Laser TTL pulse width
- `cameraDelay` - Delay before camera trigger
- `cameraDuration` - Camera TTL pulse width

**However, in `ControllerUtils.prepareControllerForAquisition()`, only `SPIM_DURATION_SCAN` is explicitly written to the Tiger controller!**

The other timing properties are **NEVER written**:
- ❌ `SPIMDelayBeforeScan(ms)` - NOT set
- ❌ `SPIMDelayBeforeLaser(ms)` - NOT set
- ❌ `SPIMLaserDuration(ms)` - NOT set
- ❌ `SPIMDelayBeforeCamera(ms)` - NOT set
- ❌ `SPIMCameraDuration(ms)` - **NOT set** ← **If 0, NO triggers generated!**

**Impact:**
- Tiger controller uses stale values from previous acquisitions or defaults (possibly 0)
- If `SPIMCameraDuration(ms) = 0`, **no TTL pulses are generated**
- This explains why camera never receives hardware triggers

**Workarounds:**
1. Use "Advanced Timing" mode in diSPIM plugin (UI spinners write directly to controller)
2. Manually set properties via Micro-Manager or serial commands before acquisition
3. **In Python: Explicitly set all timing properties before starting SPIM state machine** ✅ **VERIFIED WORKING**

**Source Code Reference:**
- File: `ControllerUtils.java`, line 434 - Only sets `SPIM_DURATION_SCAN`
- File: `ControllerUtils.java`, lines 103-537 - Missing property writes
- File: `AcquisitionPanel.java`, lines 446-499 - UI spinners for Advanced Timing mode

### ⚠️ Issue #2: SENSOR MODE Must Be PROGRESSIVE (CRITICAL!)

**Discovered during Python implementation debugging:**

The Hamamatsu Flash4 camera has two sensor readout modes:
- **AREA mode**: Standard split readout (top/bottom simultaneous) - **DOES NOT WORK** with external triggers in SPIM mode
- **PROGRESSIVE mode**: Rolling shutter for light sheet - **REQUIRED** for hardware-triggered SPIM

**Symptoms when using AREA mode:**
- Camera sequence starts but immediately stops (`isSequenceRunning()` returns False)
- No images captured (buffer remains empty)
- No error messages, just silent failure

**Solution:**
```python
# CRITICAL: Must use PROGRESSIVE mode for hardware-triggered SPIM
core.setProperty(camera_name, "SENSOR MODE", "PROGRESSIVE")
core.setProperty(camera_name, "TRIGGER ACTIVE", "EDGE")
```

**Why PROGRESSIVE mode is required:**
- PROGRESSIVE mode implements rolling shutter synchronized with galvo scan
- AREA mode's split readout conflicts with external trigger timing
- Nature Protocols diSPIM paper (nprot.2014.172) explicitly uses rolling shutter mode
- Maximum exposure in PROGRESSIVE mode: ~10-12ms (vs much higher in AREA mode)

**Tested and verified:** 100-slice volume acquired at 59.1 fps with PROGRESSIVE mode ✅

### ⚠️ Issue #3: Piezo vs Galvo for Slice Stepping

**Common misconception:** The piezo (PiezoStage:P:34/Q:35) is used for SPIM slice scanning.

**Reality:**
- The **galvo Y-axis** (on Scanner:AB:33) handles all slice stepping in SPIM mode
- The **piezo** is only for objective focus positioning, NOT for SPIM scanning
- Setting piezo properties like `SA_AMPLITUDE` will fail - piezo doesn't support these properties

**Correct configuration:**
```python
# Galvo Y-axis handles slice stepping
core.setProperty(galvo_device, "SingleAxisYAmplitude(deg)", y_amplitude)
core.setProperty(galvo_device, "SingleAxisYOffset(deg)", 0.0)
core.setProperty(galvo_device, "SingleAxisYPattern", "1 - Triangle")
core.setProperty(galvo_device, "SingleAxisYMode", "3 - Enabled with axes synced")

# No piezo configuration needed for SPIM scanning!
```

---

## Camera Configuration by Manufacturer

### Hamamatsu Flash4 (HamCam)

**Property Names and Values:**

```python
# TRIGGER_SOURCE property
"INTERNAL"  # For live/snap mode
"EXTERNAL"  # For hardware-triggered acquisition

# SENSOR_MODE property
"AREA"         # Standard split readout (top/bottom simultaneous)
"PROGRESSIVE"  # Rolling shutter for light sheet (slower but better for SPIM)

# TRIGGER_ACTIVE property
"EDGE"        # Edge trigger - single pulse starts exposure
"LEVEL"       # Level trigger - TTL high duration = exposure time
"SYNCREADOUT" # Overlap mode - synchronous readout

# TriggerPolarity property
"POSITIVE"    # Trigger on rising edge (required)
```

**Configuration Sequence (from Cameras.java:231-260):**

```python
# Step 1: Set trigger source
core.setProperty(camera_name, "TRIGGER SOURCE", "EXTERNAL")

# Step 2: Set sensor mode
core.setProperty(camera_name, "SENSOR MODE", "PROGRESSIVE")  # For light sheet
# OR
core.setProperty(camera_name, "SENSOR MODE", "AREA")  # For standard acquisition

# Step 3: Set trigger type
core.setProperty(camera_name, "TRIGGER ACTIVE", "EDGE")  # Most common

# Step 4: Set exposure (for bookkeeping, not hardware in EDGE mode)
core.setExposure(camera_name, 10.0)  # milliseconds
```

**Camera Mode Recommendations:**
- **EDGE mode**: Camera latches exposure time internally, TTL pulse is just a trigger
- **LEVEL mode**: TTL high duration = exposure time (not recommended for fast acquisition)
- **PROGRESSIVE mode**: Maximum exposure ~10-12ms, use for light sheet rolling shutter
- **AREA mode**: Higher exposure range, split readout from center out

---

### PCO Edge (PCOCam)

**Property Names and Values:**

```python
# Triggermode property
"Internal"          # For live/snap mode
"External"          # Edge trigger mode
"External Exp. Ctrl." # Level trigger mode

# PixelRate property
"slow scan"  # Slower pixel readout, higher quality
"fast scan"  # Faster readout, affects timing calculations
```

**Configuration Sequence (from Cameras.java:262-283):**

```python
# For edge trigger
core.setProperty(camera_name, "Triggermode", "External")

# For level trigger
core.setProperty(camera_name, "Triggermode", "External Exp. Ctrl.")
```

---

### Andor Zyla (AndorCam)

**Property Names and Values:**

```python
# TriggerMode property
"Internal (Recommended for fast acquisitions)"  # Live mode
"External"                                       # Edge trigger
"External Exposure"                             # Level trigger

# Overlap property
"On"   # Overlap mode enabled
"Off"  # Standard mode

# LightScanPlus-SensorReadoutMode property
"Centre Out Simultaneous"  # Standard split readout
"Bottom Up Sequential"     # Rolling shutter for light sheet
"Bottom Up Simultaneous"   # Split readout from bottom
```

**Configuration Sequence (from Cameras.java:284-331):**

```python
# For light sheet mode
core.setProperty(camera_name, "TriggerMode", "External")
core.setProperty(camera_name, "Overlap", "Off")
core.setProperty(camera_name, "LightScanPlus-SensorReadoutMode", "Bottom Up Sequential")

# For edge trigger mode
core.setProperty(camera_name, "TriggerMode", "External")
core.setProperty(camera_name, "Overlap", "Off")
```

---

### Photometrics Prime (PVCAM)

**Property Names and Values:**

```python
# TriggerMode property
"Internal Trigger"  # Live mode
"Edge Trigger"      # Hardware triggered

# ClearMode property
"Never"         # No clearing between frames
"Pre-Exposure"  # Clear before each exposure
"Pre-Sequence"  # Clear once before sequence
```

**Configuration Sequence (from Cameras.java:332-349):**

```python
core.setProperty(camera_name, "TriggerMode", "Edge Trigger")
```

---

## Tiger Controller SPIM Properties

### Critical SPIM State Machine Properties

All properties are set on the **Micro-mirror (Galvo) card device**:

```python
# Device name format: "Scanner:AB:33" or "TigerCommHub-Axis:M@TigerComm-COM5"
galvo_device = "Scanner:AB:33"  # Your device name

# SPIM State Machine Control
"SPIMState"           # Values: "Idle", "Armed", "Running"
"SPIMNumSlices"       # Number of slices per side (e.g., 100)
"SPIMNumSides"        # 1 = single side, 2 = dual view
"SPIMFirstSide"       # "A" or "B" - which side starts

# SPIM Timing Properties (ALL IN MILLISECONDS, 0.25ms resolution)
"SPIMDelayBeforeScan(ms)"      # Delay before scan mirror starts
"SPIMScanDuration(ms)"         # Scan mirror sweep duration
"SPIMDelayBeforeLaser(ms)"     # Delay before laser trigger
"SPIMLaserDuration(ms)"        # Laser TTL pulse width
"SPIMDelayBeforeCamera(ms)"    # Delay before camera trigger
"SPIMCameraDuration(ms)"       # Camera TTL pulse width ← CRITICAL!

# SPIM Multi-Acquisition Properties
"SPIMNumRepeats"               # Volumes per trigger (for hardware timepoints)
"SPIMDelayBeforeRepeat(ms)"    # Delay between volumes
"SPIMDelayBeforeSide(ms)"      # Delay between sides (for dual view)
"SPIMNumScansPerSlice"         # Usually 1
"SPIMNumSlicesPerPiezo"        # For multichannel slice-by-slice
"SPIMAlternateDirectionsEnable" # "Yes" or "No"
"SPIMInterleaveSidesEnable"    # For interleaved stage scan
"SPIMPiezoHomeDisable"         # For stage scan mode

# Scan Mirror Properties
"SingleAxisXAmplitude(deg)"    # Light sheet width (X-axis)
"SingleAxisXOffset(deg)"       # Light sheet position offset
"SingleAxisXPattern"           # "0 - Ramp", "1 - Triangle"
"SingleAxisXMode"              # "0 - Disabled", "1 - Enabled", "3 - Enabled with axes synced"

"SingleAxisYAmplitude(deg)"    # Slice stepping amplitude (Y-axis)
"SingleAxisYOffset(deg)"       # Slice position offset
"SingleAxisYPattern"           # "0 - Ramp", "1 - Triangle"
"SingleAxisYMode"              # "0 - Disabled", "3 - Enabled with axes synced"

# Critical Output Configuration
"LaserOutputMode"              # "shutter + side" ← MUST BE SET FOR TRIGGERS!
"BeamEnabled"                  # "Yes" or "No" - disable during SPIM acquisition
```

### Piezo (Z-drive) Properties

```python
# Device name format: "Piezo:A:37" or "TigerCommHub-Axis:Z@TigerComm-COM5"
piezo_device = "Piezo:A:37"  # Your device name

# Piezo Sweep Properties
"SPIMState"                    # "Idle", "Armed" ← Must arm before galvo trigger
"SPIMNumSlices"                # Number of Z positions
"SA_AMPLITUDE"                 # Sweep amplitude in micrometers
"SA_OFFSET"                    # Center position in micrometers
"SA_PATTERN"                   # "0 - Ramp", "1 - Triangle"
"SA_MODE_Z"                    # "0 - Disabled", "1 - Enabled", "3 - Enabled with axes synced"
```

### PLogic Card Properties (Optional)

```python
# Device name format: "PLogic:E:36"
plogic_device = "PLogic:E:36"  # Your device name

# PLogic Control
"PLogicMode"                   # "Disp. Seq. positions"
"PLogicOutputChannel"          # "6,7" for lasers on BNC6 & BNC7
"PoLogicPreset"                # "3 - cell 1 high" during acquisition

# Note: PLogic adds 0.25ms delay to all TTL outputs
```

---

## Complete Acquisition Workflow

### Phase 1: Camera Configuration

```python
def configure_camera_for_hardware_trigger(core, camera_name, camera_mode="EDGE", exposure_ms=10.0):
    """
    Configure camera for hardware-triggered acquisition.

    Args:
        camera_name: e.g., "HamCam1"
        camera_mode: "EDGE", "LEVEL", "OVERLAP", or "LIGHT_SHEET"
        exposure_ms: Exposure time in milliseconds
    """
    core.setCameraDevice(camera_name)

    # For Hamamatsu Flash4
    core.setProperty(camera_name, "TRIGGER SOURCE", "EXTERNAL")

    if camera_mode == "LIGHT_SHEET":
        core.setProperty(camera_name, "SENSOR MODE", "PROGRESSIVE")
        core.setProperty(camera_name, "TRIGGER ACTIVE", "EDGE")
    elif camera_mode == "EDGE":
        core.setProperty(camera_name, "SENSOR MODE", "AREA")
        core.setProperty(camera_name, "TRIGGER ACTIVE", "EDGE")
    elif camera_mode == "LEVEL":
        core.setProperty(camera_name, "SENSOR MODE", "AREA")
        core.setProperty(camera_name, "TRIGGER ACTIVE", "LEVEL")
    elif camera_mode == "OVERLAP":
        core.setProperty(camera_name, "SENSOR MODE", "AREA")
        core.setProperty(camera_name, "TRIGGER ACTIVE", "SYNCREADOUT")

    # Set exposure time (for internal bookkeeping)
    core.setExposure(camera_name, exposure_ms)

    # Verify configuration
    time.sleep(0.1)
    trigger_source = core.getProperty(camera_name, "TRIGGER SOURCE")
    sensor_mode = core.getProperty(camera_name, "SENSOR MODE")
    trigger_active = core.getProperty(camera_name, "TRIGGER ACTIVE")

    print(f"Camera configured:")
    print(f"  TRIGGER SOURCE: {trigger_source}")
    print(f"  SENSOR MODE: {sensor_mode}")
    print(f"  TRIGGER ACTIVE: {trigger_active}")
    print(f"  Exposure: {core.getExposure(camera_name)} ms")
```

### Phase 2: Timing Calculation

```python
def calculate_spim_timing(camera_exposure_ms, camera_reset_ms, camera_readout_ms,
                          scan_laser_buffer_ms=0.25, scan_filter_freq_khz=0.2,
                          has_plogic=False):
    """
    Calculate SPIM timing parameters following ASI diSPIM plugin logic.

    Args:
        camera_exposure_ms: Desired light exposure time
        camera_reset_ms: Camera reset time (trigger to global exposure)
        camera_readout_ms: Camera readout time (frame transfer)
        scan_laser_buffer_ms: Safety buffer (default 0.25ms)
        scan_filter_freq_khz: Bessel filter frequency in kHz
        has_plogic: Whether PLogic card is present

    Returns:
        Dictionary of timing parameters
    """
    # Round to 0.25ms (Tiger controller resolution)
    def round_quarter_ms(val):
        return round(val * 4) / 4.0

    def ceil_quarter_ms(val):
        return math.ceil(val * 4) / 4.0

    camera_readout_max = ceil_quarter_ms(camera_readout_ms)
    camera_reset_max = ceil_quarter_ms(camera_reset_ms)
    global_exposure_delay_max = camera_readout_max + camera_reset_max

    laser_duration = round_quarter_ms(camera_exposure_ms)
    scan_duration = laser_duration + 2 * scan_laser_buffer_ms

    # Account for Bessel filter delay and PLogic delay
    scan_delay_filter = 0.39 / scan_filter_freq_khz
    if has_plogic:
        scan_delay_filter -= 0.25  # PLogic adds 0.25ms delay

    timing = {
        'scanDelay': global_exposure_delay_max - scan_laser_buffer_ms - scan_delay_filter,
        'scanPeriod': scan_duration,
        'laserDelay': global_exposure_delay_max,
        'laserDuration': laser_duration,
        'cameraDelay': camera_readout_max,
        'cameraDuration': 1.0,  # Short pulse for EDGE mode
        'cameraExposure': camera_exposure_ms + 0.1,  # Add safety margin
        'sliceDuration': max(scan_duration, laser_duration, camera_readout_max + camera_exposure_ms)
    }

    # Round all values to 0.25ms
    for key in timing:
        timing[key] = round_quarter_ms(timing[key])

    return timing
```

### Phase 3: Tiger Controller Configuration

```python
def configure_tiger_controller_for_spim(core, galvo_device, piezo_device,
                                        num_slices=100, num_sides=1, first_side_a=True,
                                        timing=None):
    """
    Configure Tiger controller SPIM state machine.

    Args:
        galvo_device: e.g., "Scanner:AB:33"
        piezo_device: e.g., "Piezo:A:37"
        num_slices: Number of slices per volume
        num_sides: 1 or 2 for single/dual view
        first_side_a: True if side A starts first
        timing: Dictionary from calculate_spim_timing()
    """
    # Ensure SPIM is idle
    core.setProperty(galvo_device, "SPIMState", "Idle")
    core.setProperty(piezo_device, "SPIMState", "Idle")
    time.sleep(0.2)

    # Disable beam scanning (will be controlled by SPIM state machine)
    core.setProperty(galvo_device, "BeamEnabled", "No")

    # CRITICAL: Set laser output mode to enable TTL outputs
    core.setProperty(galvo_device, "LaserOutputMode", "shutter + side")

    # Configure scan mirror X-axis (light sheet width)
    core.setProperty(galvo_device, "SingleAxisXAmplitude(deg)", 2.0)
    core.setProperty(galvo_device, "SingleAxisXOffset(deg)", 0.0)
    core.setProperty(galvo_device, "SingleAxisXPattern", "1 - Triangle")
    core.setProperty(galvo_device, "SingleAxisXMode", "3 - Enabled with axes synced")

    # Configure scan mirror Y-axis (slice stepping) - optional for stage scan
    # For piezo scan, Y-axis can provide additional slice stepping
    core.setProperty(galvo_device, "SingleAxisYAmplitude(deg)", 0.04)
    core.setProperty(galvo_device, "SingleAxisYOffset(deg)", 0.0)
    core.setProperty(galvo_device, "SingleAxisYPattern", "1 - Triangle")
    core.setProperty(galvo_device, "SingleAxisYMode", "3 - Enabled with axes synced")

    # Configure piezo sweep
    slice_step_um = 1.0  # Micrometers per slice
    piezo_amplitude = (num_slices - 1) * slice_step_um / 2.0  # Half range
    piezo_center = 0.0  # Center position

    core.setProperty(piezo_device, "SA_AMPLITUDE", piezo_amplitude)
    core.setProperty(piezo_device, "SA_OFFSET", piezo_center)
    core.setProperty(piezo_device, "SPIMNumSlices", num_slices)

    # ARM the piezo (ready to move on SPIM trigger)
    core.setProperty(piezo_device, "SPIMState", "Armed")

    # Set SPIM state machine parameters
    core.setProperty(galvo_device, "SPIMNumSlices", num_slices)
    core.setProperty(galvo_device, "SPIMNumSides", num_sides)
    core.setProperty(galvo_device, "SPIMFirstSide", "A" if first_side_a else "B")
    core.setProperty(galvo_device, "SPIMNumRepeats", 1)
    core.setProperty(galvo_device, "SPIMAlternateDirectionsEnable", "No")

    # ⚠️ CRITICAL: Set ALL timing properties explicitly
    if timing:
        core.setProperty(galvo_device, "SPIMDelayBeforeScan(ms)", timing['scanDelay'])
        core.setProperty(galvo_device, "SPIMScanDuration(ms)", timing['scanPeriod'])
        core.setProperty(galvo_device, "SPIMDelayBeforeLaser(ms)", timing['laserDelay'])
        core.setProperty(galvo_device, "SPIMLaserDuration(ms)", timing['laserDuration'])
        core.setProperty(galvo_device, "SPIMDelayBeforeCamera(ms)", timing['cameraDelay'])
        core.setProperty(galvo_device, "SPIMCameraDuration(ms)", timing['cameraDuration'])
    else:
        # Use safe defaults
        core.setProperty(galvo_device, "SPIMDelayBeforeScan(ms)", 0.0)
        core.setProperty(galvo_device, "SPIMScanDuration(ms)", 10.0)
        core.setProperty(galvo_device, "SPIMDelayBeforeLaser(ms)", 8.0)
        core.setProperty(galvo_device, "SPIMLaserDuration(ms)", 5.0)
        core.setProperty(galvo_device, "SPIMDelayBeforeCamera(ms)", 5.0)
        core.setProperty(galvo_device, "SPIMCameraDuration(ms)", 1.0)  # Must be > 0!

    core.setProperty(galvo_device, "SPIMDelayBeforeSide(ms)", 0.0)
    core.setProperty(galvo_device, "SPIMDelayBeforeRepeat(ms)", 0.0)

    # Verify configuration
    print(f"Tiger controller configured:")
    print(f"  SPIMNumSlices: {core.getProperty(galvo_device, 'SPIMNumSlices')}")
    print(f"  SPIMNumSides: {core.getProperty(galvo_device, 'SPIMNumSides')}")
    print(f"  SPIMScanDuration(ms): {core.getProperty(galvo_device, 'SPIMScanDuration(ms)')}")
    print(f"  SPIMCameraDuration(ms): {core.getProperty(galvo_device, 'SPIMCameraDuration(ms)')}")
    print(f"  LaserOutputMode: {core.getProperty(galvo_device, 'LaserOutputMode')}")
    print(f"  Piezo SPIMState: {core.getProperty(piezo_device, 'SPIMState')}")
```

### Phase 4: Start Camera Sequence Acquisition

```python
def start_camera_sequence(core, camera_name, num_images):
    """
    Start camera in sequence acquisition mode (waiting for external triggers).

    Args:
        camera_name: e.g., "HamCam1"
        num_images: Expected number of images (num_slices * num_sides)
    """
    # Prepare sequence acquisition (allocates camera buffer)
    core.prepareSequenceAcquisition(camera_name)
    time.sleep(0.1)

    # Start sequence (camera enters WAITING state)
    # Parameters: device, numImages, intervalMs, stopOnOverflow
    core.startSequenceAcquisition(camera_name, num_images, 0, True)

    print(f"Camera sequence started:")
    print(f"  Sequence running: {core.isSequenceRunning(camera_name)}")
    print(f"  Buffer capacity: {core.getBufferTotalCapacity()}")
    print(f"  Images in buffer: {core.getRemainingImageCount()}")
```

### Phase 5: Trigger SPIM State Machine

```python
def trigger_spim_acquisition(core, galvo_device):
    """
    Start SPIM state machine to generate TTL trigger pulses.

    Args:
        galvo_device: e.g., "Scanner:AB:33"
    """
    # Set SPIM state to "Running" (starts TTL pulse generation)
    core.setProperty(galvo_device, "SPIMState", "Running")
    time.sleep(0.1)

    state = core.getProperty(galvo_device, "SPIMState")
    print(f"SPIM triggered:")
    print(f"  SPIMState: {state}")

    if state != "Running":
        raise Exception(f"Failed to start SPIM state machine (state={state})")
```

### Phase 6: Wait for Images and Retrieve

```python
def wait_for_images(core, camera_name, num_expected, timeout_sec=30.0):
    """
    Wait for hardware-triggered images to accumulate in buffer.

    Args:
        camera_name: e.g., "HamCam1"
        num_expected: Expected number of images
        timeout_sec: Maximum wait time

    Returns:
        List of numpy arrays
    """
    import rpyc

    start = time.time()
    last_print = start

    print(f"Waiting for {num_expected} hardware-triggered images...")

    while core.getRemainingImageCount() < num_expected:
        elapsed = time.time() - start

        if elapsed > timeout_sec:
            count = core.getRemainingImageCount()
            print(f"Timeout after {elapsed:.1f}s - only got {count}/{num_expected} images")
            break

        # Print status every 0.5s
        if (time.time() - last_print) >= 0.5:
            count = core.getRemainingImageCount()
            seq_running = core.isSequenceRunning(camera_name)
            print(f"  t={elapsed:.1f}s: images={count}/{num_expected}, seq={seq_running}")
            last_print = time.time()

        time.sleep(0.01)

    # Retrieve images
    count = core.getRemainingImageCount()
    images = []

    print(f"\nRetrieving {count} images...")
    for i in range(count):
        img = core.popNextImage()
        img = rpyc.classic.obtain(img)  # For rpyc remote objects
        images.append(img)
        print(f"  Image {i+1}/{count}: shape={img.shape}, range=[{img.min()}, {img.max()}]")

    return images
```

### Complete Acquisition Function

```python
def acquire_spim_volume(core, camera_name, galvo_device, piezo_device,
                       num_slices=100, camera_exposure_ms=5.0):
    """
    Complete hardware-triggered SPIM volume acquisition.

    Args:
        core: Micro-Manager Core instance
        camera_name: e.g., "HamCam1"
        galvo_device: e.g., "Scanner:AB:33"
        piezo_device: e.g., "Piezo:A:37"
        num_slices: Number of Z slices
        camera_exposure_ms: Light exposure time

    Returns:
        numpy array of shape (num_slices, height, width)
    """
    try:
        # Phase 1: Configure camera
        configure_camera_for_hardware_trigger(core, camera_name,
                                              camera_mode="EDGE",
                                              exposure_ms=camera_exposure_ms)

        # Phase 2: Calculate timing
        timing = calculate_spim_timing(
            camera_exposure_ms=camera_exposure_ms,
            camera_reset_ms=3.0,      # Hamamatsu Flash4 typical
            camera_readout_ms=10.0,   # Depends on ROI and scan mode
            has_plogic=True
        )

        print("\nCalculated timing:")
        for key, val in timing.items():
            print(f"  {key}: {val} ms")

        # Phase 3: Configure Tiger controller
        configure_tiger_controller_for_spim(core, galvo_device, piezo_device,
                                           num_slices=num_slices,
                                           num_sides=1,
                                           first_side_a=True,
                                           timing=timing)

        # Phase 4: Start camera sequence
        start_camera_sequence(core, camera_name, num_slices)

        # Phase 5: Trigger SPIM
        trigger_spim_acquisition(core, galvo_device)

        # Phase 6: Wait for images
        expected_time = num_slices * timing['sliceDuration'] / 1000.0
        timeout = expected_time * 2 + 10.0
        images = wait_for_images(core, camera_name, num_slices, timeout)

        if len(images) == num_slices:
            print(f"\n✓ SUCCESS! Acquired {len(images)} images")
            return np.array(images)
        else:
            print(f"\n✗ FAILED - Got {len(images)}/{num_slices} images")
            return None

    finally:
        # Cleanup
        try:
            if core.isSequenceRunning(camera_name):
                core.stopSequenceAcquisition(camera_name)
            core.setProperty(galvo_device, "SPIMState", "Idle")
            core.setProperty(piezo_device, "SPIMState", "Idle")
        except:
            pass
```

---

## Timing Calculations

### Timing Diagram

```
SLICE TIMING FOR EDGE TRIGGER MODE (from AcquisitionPanel.java:1105-1240)
═══════════════════════════════════════════════════════════════════════════

Time →

┌────────────────────────────────────────────────────────────────────┐
│ SLICE PERIOD (sliceDuration)                                       │
│ = max(scan time, laser time, camera time)                          │
└────────────────────────────────────────────────────────────────────┘

Camera Readout  ◄──────────────►
(from previous    cameraDelay
 frame)           (cameraReadout_max)

                                  Camera Reset ◄───────────►
                                  (trigger to   cameraReset_max
                                   global exp)

                                               Camera Exposure ◄──────►
                                                                exposure

Scan Mirror:    ┌─────────────────────────────────────┐
  Position      │   scanPeriod                        │
                │   (scanDuration)                     │
             ┌──┴──┐                                 ┌─┴──┐
             │delay│                                 │    │
             └─────┘                                 └────┘
                ▲                                      ▲
             scanDelay                            slice ends

Laser TTL:           ┌─────────────────────┐
  (BNC2/TTL1)        │  laserDuration      │
                     │                      │
                ─────┴──────────────────────┴────────
                     ▲
                  laserDelay

Camera TTL:                 ▲
  (BNC2/TTL1)               │ cameraDuration (1ms pulse)
                ────────────┴────────────────
                            ▲
                         cameraDelay

                                  ▲──────────────► Camera actually exposing
                               Global Exposure
                                   Starts

TIMING CONSTRAINTS:
• All values rounded to 0.25ms (Tiger controller resolution)
• scanDelay accounts for:
  - Bessel filter delay: 0.39 / scanFilterFreq (in kHz)
  - PLogic delay: -0.25ms (if PLogic card present)
• Camera must finish readout before next trigger
• Laser turns on when camera reaches global exposure
• Scan starts 0.25ms before laser to ensure steady beam
• Tiger controller firmware ensures TTL timing precision
```

### Key Timing Parameters

**Camera Reset Time** (`cameraResetTime`):
- Hamamatsu Flash4: ~3ms (AREA mode), ~2ms (PROGRESSIVE mode)
- Time from trigger rising edge to global exposure start
- Measured experimentally or from camera specs

**Camera Readout Time** (`cameraReadoutTime`):
- Depends on ROI size and scan mode (slow vs fast)
- Hamamatsu Flash4 full frame: ~10ms
- Time to transfer previous frame from sensor to buffer
- Must wait this duration before next trigger

**Bessel Filter Delay**:
- Galvo mirror has Bessel filter for smooth motion
- Delay = 0.39 / filter_frequency_kHz
- Typical filter frequency: 0.2 kHz → delay = 1.95ms
- Must start scan mirror early to account for this lag

**PLogic Delay**:
- PLogic card adds 0.25ms to all TTL outputs
- If present, reduce scanDelay by 0.25ms to compensate

**Timing Formula** (from AcquisitionPanel.java:1161-1193):

```python
# Round to Tiger controller resolution (0.25ms)
cameraReadout_max = ceil_to_quarter_ms(cameraReadoutTime)
cameraReset_max = ceil_to_quarter_ms(cameraResetTime)

# Total delay before camera reaches global exposure
globalExposureDelay_max = cameraReadout_max + cameraReset_max

# Laser duration = desired exposure time
laserDuration = round_to_quarter_ms(desiredLightExposure)

# Scan duration includes buffer time before/after laser
scanLaserBufferTime = 0.25  # ms safety margin
scanDuration = laserDuration + 2 * scanLaserBufferTime

# Account for filter and PLogic delays
scanDelayFilter = 0.39 / scanFilterFreqKHz
if hasPLogic:
    scanDelayFilter -= 0.25  # Compensate for PLogic delay

# Calculate timing parameters
scanDelay = globalExposureDelay_max - scanLaserBufferTime - scanDelayFilter
laserDelay = globalExposureDelay_max
cameraDelay = cameraReadout_max
cameraDuration = 1.0  # Short pulse for EDGE mode
```

---

## Hardware Trigger Output Generation

### Tiger Micro-Mirror Card TTL Outputs

When `LaserOutputMode = "shutter + side"`, the micro-mirror card generates TTL signals on its BNC outputs:

```
┌──────────────────────────────────────────────────────┐
│  Tiger Micro-Mirror Card (Scanner:AB:33)             │
│                                                       │
│  BNC1 (TTL0): Not used by diSPIM                     │
│  BNC2 (TTL1): Camera/Laser trigger ← SPIM controlled │
│  BNC3 (TTL2): Not used by diSPIM                     │
│  BNC4 (TTL3): Side indicator (high=A, low=B)         │
│                                                       │
└──────────────────────────────────────────────────────┘
         │                        │
         │ TTL1                   │ TTL3
         │ (Camera/Laser)         │ (Side)
         ↓                        ↓
    ┌─────────┐           ┌─────────────┐
    │ Camera  │           │ Side Logic  │
    │ Trigger │           │ (optional)  │
    │ Input   │           └─────────────┘
    └─────────┘
```

**TTL1 (Camera/Laser Trigger) Pulse Timing:**
```
For each slice:
  1. Wait scanDelay ms
  2. Start scan mirror sweep (scanPeriod ms)
  3. Wait laserDelay ms
  4. Output TTL1 HIGH for laserDuration ms (laser trigger)
  5. Wait cameraDelay ms
  6. Output TTL1 pulse for cameraDuration ms (camera trigger)
  7. Move piezo to next position
  8. Repeat for next slice
```

**TTL3 (Side Indicator):**
- HIGH = Side A active
- LOW = Side B active
- Used for hardware channel switching in dual-view systems

### SPIM State Machine Firmware Logic

When `SPIMState` is set to `"Running"`, the Tiger controller firmware executes:

```c
// Pseudocode based on ASI firmware behavior

void SPIM_StateMachine() {
    if (SPIMState != RUNNING) return;

    // Initialize
    if (!SPIMPiezoHomeDisable) {
        movePiezoToStartPosition();
    }

    // For each volume repeat
    for (int vol = 0; vol < SPIMNumRepeats; vol++) {
        if (vol > 0) {
            delay_ms(SPIMDelayBeforeRepeat);
        }

        // For each side
        for (int side = 0; side < SPIMNumSides; side++) {
            if (side > 0) {
                delay_ms(SPIMDelayBeforeSide);
            }

            // Set side indicator
            if (SPIMNumSides == 2) {
                TTL3 = (currentSide == A) ? HIGH : LOW;
            }

            // For each slice
            for (int slice = 0; slice < SPIMNumSlices; slice++) {
                // Scan mirror sweep
                delay_ms(SPIMDelayBeforeScan);
                startScanMirrorSweep(SPIMScanDuration);

                // Laser trigger
                delay_ms(SPIMDelayBeforeLaser);
                TTL1 = HIGH;
                delay_ms(SPIMLaserDuration);
                TTL1 = LOW;

                // Camera trigger
                delay_ms(SPIMDelayBeforeCamera);
                TTL1 = HIGH;
                delay_ms(SPIMCameraDuration);  // ← If 0, no pulse!
                TTL1 = LOW;

                // Move to next slice position
                movePiezoNextSlice();

                waitForScanMirrorComplete();
            }
        }
    }

    // Return to idle
    SPIMState = IDLE;
}
```

**Critical Points:**
- If `SPIMCameraDuration = 0`, TTL1 toggles HIGH then immediately LOW (no pulse!)
- If `SPIMDelayBeforeCamera` is too small, camera hasn't finished readout yet
- If `SPIMScanDuration` is too small, scan mirror doesn't stabilize
- All delays must be ≥ 0, rounded to 0.25ms resolution

### PLogic Card Routing (Optional)

If a PLogic card is present, TTL signals can be routed to additional outputs:

```
┌────────────────────────────────────────────────┐
│  PLogic Card (PLogic:E:36)                     │
│                                                 │
│  Input: TTL1 from micro-mirror card            │
│                                                 │
│  Logic cells can:                              │
│  - Route to BNC5-8 outputs                     │
│  - Implement hardware channel switching        │
│  - Add 0.25ms delay to all outputs             │
│                                                 │
│  BNC5: Laser 405nm (optional)                  │
│  BNC6: Laser 488nm                             │
│  BNC7: Laser 561nm                             │
│  BNC8: Laser 640nm (optional)                  │
└────────────────────────────────────────────────┘
```

**Hardware Channel Switching:**
- PLogic uses counters to detect which channel is active
- Routes TTL1 to appropriate laser BNC based on channel preset
- Controlled by `PLogicMode = "Disp. Seq. positions"`
- See `ControllerUtils.java:691-815` for configuration logic

---

## Debugging Guide

### Step 1: Verify Camera Configuration

```python
# Check camera is in external trigger mode
camera_name = "HamCam1"

trigger_source = core.getProperty(camera_name, "TRIGGER SOURCE")
print(f"TRIGGER SOURCE: {trigger_source}")
assert trigger_source == "EXTERNAL", "Camera not in external trigger mode!"

trigger_active = core.getProperty(camera_name, "TRIGGER ACTIVE")
print(f"TRIGGER ACTIVE: {trigger_active}")
assert trigger_active in ["EDGE", "LEVEL", "SYNCREADOUT"], "Invalid trigger type!"

sensor_mode = core.getProperty(camera_name, "SENSOR MODE")
print(f"SENSOR MODE: {sensor_mode}")

exposure = core.getExposure(camera_name)
print(f"Exposure: {exposure} ms")
```

**Expected output:**
```
TRIGGER SOURCE: EXTERNAL
TRIGGER ACTIVE: EDGE
SENSOR MODE: PROGRESSIVE
Exposure: 10.0 ms
```

### Step 2: Verify Tiger Controller Timing Properties

**This is the most critical step!**

```python
galvo_device = "Scanner:AB:33"

# Check laser output mode (MUST be "shutter + side")
laser_mode = core.getProperty(galvo_device, "LaserOutputMode")
print(f"LaserOutputMode: {laser_mode}")
assert laser_mode == "shutter + side", "Wrong laser output mode - no triggers will be generated!"

# Check SPIM timing properties
print("\nSPIM Timing Properties:")
print(f"  SPIMDelayBeforeScan(ms): {core.getProperty(galvo_device, 'SPIMDelayBeforeScan(ms)')}")
print(f"  SPIMScanDuration(ms): {core.getProperty(galvo_device, 'SPIMScanDuration(ms)')}")
print(f"  SPIMDelayBeforeLaser(ms): {core.getProperty(galvo_device, 'SPIMDelayBeforeLaser(ms)')}")
print(f"  SPIMLaserDuration(ms): {core.getProperty(galvo_device, 'SPIMLaserDuration(ms)')}")
print(f"  SPIMDelayBeforeCamera(ms): {core.getProperty(galvo_device, 'SPIMDelayBeforeCamera(ms)')}")
print(f"  SPIMCameraDuration(ms): {core.getProperty(galvo_device, 'SPIMCameraDuration(ms)')}")

# CRITICAL CHECK
camera_duration = float(core.getProperty(galvo_device, 'SPIMCameraDuration(ms)'))
if camera_duration <= 0:
    print("\n⚠️  WARNING: SPIMCameraDuration is 0 - NO TRIGGERS WILL BE GENERATED!")
    print("   You must set this property to > 0 (typically 1.0 ms)")

# Check SPIM state machine parameters
print("\nSPIM State Machine:")
print(f"  SPIMNumSlices: {core.getProperty(galvo_device, 'SPIMNumSlices')}")
print(f"  SPIMNumSides: {core.getProperty(galvo_device, 'SPIMNumSides')}")
print(f"  SPIMFirstSide: {core.getProperty(galvo_device, 'SPIMFirstSide')}")
print(f"  SPIMState: {core.getProperty(galvo_device, 'SPIMState')}")
```

**Expected output:**
```
LaserOutputMode: shutter + side

SPIM Timing Properties:
  SPIMDelayBeforeScan(ms): 10.0
  SPIMScanDuration(ms): 10.0
  SPIMDelayBeforeLaser(ms): 13.0
  SPIMLaserDuration(ms): 5.0
  SPIMDelayBeforeCamera(ms): 10.0
  SPIMCameraDuration(ms): 1.0       ← MUST BE > 0!

SPIM State Machine:
  SPIMNumSlices: 100
  SPIMNumSides: 1
  SPIMFirstSide: A
  SPIMState: Idle
```

### Step 3: Verify Piezo Configuration

```python
piezo_device = "Piezo:A:37"

print("Piezo Configuration:")
print(f"  SPIMState: {core.getProperty(piezo_device, 'SPIMState')}")
print(f"  SPIMNumSlices: {core.getProperty(piezo_device, 'SPIMNumSlices')}")
print(f"  SA_AMPLITUDE: {core.getProperty(piezo_device, 'SA_AMPLITUDE')} µm")
print(f"  SA_OFFSET: {core.getProperty(piezo_device, 'SA_OFFSET')} µm")

# Piezo should be Armed before galvo is set to Running
piezo_state = core.getProperty(piezo_device, 'SPIMState')
assert piezo_state == "Armed", f"Piezo should be Armed, not {piezo_state}"
```

### Step 4: Test TTL Output with Oscilloscope

**Hardware verification (highly recommended):**

1. Connect oscilloscope probe to **BNC2 (TTL1)** on Tiger micro-mirror card
2. Set trigger: Rising edge, 2V threshold
3. Run this test code:

```python
# Configure for manual test
core.setProperty(galvo_device, "LaserOutputMode", "shutter + side")
core.setProperty(galvo_device, "SPIMNumSlices", 10)
core.setProperty(galvo_device, "SPIMNumSides", 1)
core.setProperty(galvo_device, "SPIMCameraDuration(ms)", 2.0)  # 2ms pulse
core.setProperty(galvo_device, "SPIMDelayBeforeCamera(ms)", 10.0)
core.setProperty(galvo_device, "SPIMScanDuration(ms)", 15.0)

# Arm piezo
core.setProperty(piezo_device, "SPIMState", "Armed")

# Trigger SPIM
core.setProperty(galvo_device, "SPIMState", "Running")

# You should see 10 TTL pulses on oscilloscope:
# - Pulse width: 2.0ms
# - Interval: ~15ms (scan duration)
# - Amplitude: 5V (TTL level)
```

**Expected oscilloscope trace:**
```
      ┌─┐    ┌─┐    ┌─┐    ┌─┐    ┌─┐
5V    │ │    │ │    │ │    │ │    │ │
      │ │    │ │    │ │    │ │    │ │
0V ───┘ └────┘ └────┘ └────┘ └────┘ └───
      ◄─►    ◄─►    ◄─►    ◄─►    ◄─►
      2ms    2ms    2ms    2ms    2ms

      ◄────►◄────►◄────►◄────►
       15ms  15ms  15ms  15ms
```

**If no pulses appear:**
- Check `LaserOutputMode` - must be "shutter + side"
- Check `SPIMCameraDuration(ms)` - must be > 0
- Check physical cable connection to BNC2
- Check Tiger controller power and communication
- Verify SPIMState transitions to "Running"

### Step 5: Monitor SPIM State During Acquisition

```python
# During acquisition, monitor state
galvo_state = core.getProperty(galvo_device, "SPIMState")
print(f"SPIMState during acquisition: {galvo_state}")

# Should be "Running" during acquisition
# Will return to "Idle" when complete
```

### Step 6: Check Camera Buffer

```python
# During/after acquisition
buffer_capacity = core.getBufferTotalCapacity()
buffer_free = core.getBufferFreeCapacity()
images_in_buffer = core.getRemainingImageCount()

print(f"Buffer capacity: {buffer_capacity}")
print(f"Buffer free: {buffer_free}")
print(f"Images in buffer: {images_in_buffer}")

# If images_in_buffer == 0 after acquisition:
# - Camera never received triggers
# - Check all above steps
```

### Common Error Messages and Solutions

#### Error: "Cannot set property 'TriggerActive'"
**Cause:** Wrong property name
**Solution:** Use `"TRIGGER ACTIVE"` (with space), not `"TriggerActive"`

#### Error: "Exposure is X ms but should be Y ms"
**Cause:** Camera mode has exposure limits (PROGRESSIVE max ~10-12ms)
**Solution:** Use lower exposure or switch to AREA mode

#### Error: "index was: 0, count was: 0" from Hamamatsu adapter
**Cause:** Camera sequence buffer empty - no triggers received
**Solutions:**
1. Check `SPIMCameraDuration(ms) > 0`
2. Check `LaserOutputMode = "shutter + side"`
3. Verify physical cable connection
4. Test TTL output with oscilloscope

#### Error: "No device with label 'Piezo:A:37'"
**Cause:** Wrong device name for your system
**Solution:** Use correct device names:
- Piezo: `"PiezoStage:P:34"` or `"PiezoStage:Q:35"`
- Scanner: `"Scanner:AB:33"` (common, but check your config)
- Camera: `"HamCam1"` (or your camera name)

**Note:** For SPIM scanning, you don't actually need to configure the piezo!

#### Error: "Cannot set property SA_AMPLITUDE to 49.5"
**Cause:** Trying to set piezo sweep properties for SPIM scanning
**Solution:** Don't configure piezo for SPIM! The galvo Y-axis handles slice stepping:
```python
# WRONG - piezo doesn't have SA_AMPLITUDE for SPIM
core.setProperty("PiezoStage:P:34", "SA_AMPLITUDE", 49.5)

# CORRECT - use galvo Y-axis
core.setProperty("Scanner:AB:33", "SingleAxisYAmplitude(deg)", 0.04)
```

#### Camera sequence stops immediately (isSequenceRunning() = False)
**Cause:** Using AREA sensor mode with external triggers
**Solution:** ✅ **CRITICAL FIX** - Switch to PROGRESSIVE mode:
```python
# WRONG - AREA mode doesn't work with external triggers
core.setProperty(camera_name, "SENSOR MODE", "AREA")

# CORRECT - PROGRESSIVE mode required for SPIM
core.setProperty(camera_name, "SENSOR MODE", "PROGRESSIVE")
```

**This was the root cause preventing hardware-triggered acquisition!**

#### Buffer capacity = 0
**Cause:** Circular buffer not configured
**Solution:** Set buffer memory footprint before starting acquisition:
```python
core.setCircularBufferMemoryFootprint(1200)  # MB for ~118 images
```

#### Images in buffer but all black
**Cause:** Lasers not synchronized with camera exposure
**Solutions:**
1. Check `SPIMLaserDuration(ms)` > 0
2. Verify laser trigger cable connections
3. Check `SPIMDelayBeforeLaser(ms)` timing
4. Ensure lasers are turned ON in config

#### SPIMState stuck in "Running"
**Cause:** SPIM state machine not completing (firmware issue or timeout)
**Solutions:**
1. Wait longer - full acquisition time = numSlices * scanDuration
2. Reset: Set `SPIMState = "Idle"` to abort
3. Verify all SPIM timing properties are valid (>0, <10000)
4. Check LaserOutputMode is set correctly

---

## Recommended Python Implementation

### Complete Test Script

```python
#!/usr/bin/env python3
"""
Fixed SPIM hardware-triggered acquisition with explicit timing property configuration.

This script addresses the critical bug where timing properties are calculated but not
written to the Tiger controller, resulting in no camera trigger pulses.
"""

import time
import numpy as np
from client import get_mmc

# Configuration
core = get_mmc()
camera_name = "HamCam1"
galvo_device = "Scanner:AB:33"
piezo_device = "Piezo:A:37"

num_slices = 100
camera_exposure_ms = 5.0  # Light exposure time
camera_reset_ms = 3.0     # Hamamatsu Flash4 typical
camera_readout_ms = 10.0  # Depends on ROI

print("=" * 70)
print("FIXED SPIM HARDWARE TRIGGERED ACQUISITION")
print("=" * 70)

try:
    # Step 1: Apply system startup config
    print("\nStep 1: Applying System Startup configuration...")
    core.setConfig("System", "Startup")
    core.waitForConfig("System", "Startup")

    # Step 2: Turn on lasers
    print("\nStep 2: Turning on lasers...")
    core.setConfig("Laser", "488 and 561")
    core.waitForConfig("Laser", "488 and 561")

    # Step 3: Configure camera for hardware trigger
    print("\nStep 3: Configuring camera for hardware trigger...")
    core.setCameraDevice(camera_name)
    core.setProperty(camera_name, "TRIGGER SOURCE", "EXTERNAL")
    core.setProperty(camera_name, "SENSOR MODE", "PROGRESSIVE")
    core.setProperty(camera_name, "TRIGGER ACTIVE", "EDGE")
    core.setExposure(camera_name, camera_exposure_ms)
    time.sleep(0.1)

    # Verify camera config
    print(f"  Camera: {camera_name}")
    print(f"  TRIGGER SOURCE: {core.getProperty(camera_name, 'TRIGGER SOURCE')}")
    print(f"  SENSOR MODE: {core.getProperty(camera_name, 'SENSOR MODE')}")
    print(f"  TRIGGER ACTIVE: {core.getProperty(camera_name, 'TRIGGER ACTIVE')}")
    print(f"  Exposure: {core.getExposure(camera_name)} ms")

    # Step 4: Calculate SPIM timing
    print("\nStep 4: Calculating SPIM timing parameters...")

    # Round to 0.25ms
    def round_quarter_ms(val):
        return round(val * 4) / 4.0

    def ceil_quarter_ms(val):
        import math
        return math.ceil(val * 4) / 4.0

    camera_readout_max = ceil_quarter_ms(camera_readout_ms)
    camera_reset_max = ceil_quarter_ms(camera_reset_ms)
    global_exposure_delay_max = camera_readout_max + camera_reset_max

    scan_laser_buffer_ms = 0.25
    laser_duration = round_quarter_ms(camera_exposure_ms)
    scan_duration = laser_duration + 2 * scan_laser_buffer_ms

    scan_filter_freq_khz = 0.2
    has_plogic = True
    scan_delay_filter = 0.39 / scan_filter_freq_khz
    if has_plogic:
        scan_delay_filter -= 0.25

    scan_delay = global_exposure_delay_max - scan_laser_buffer_ms - scan_delay_filter
    laser_delay = global_exposure_delay_max
    camera_delay = camera_readout_max
    camera_duration = 1.0  # Short pulse for EDGE mode

    print(f"  Timing parameters:")
    print(f"    scanDelay: {scan_delay} ms")
    print(f"    scanDuration: {scan_duration} ms")
    print(f"    laserDelay: {laser_delay} ms")
    print(f"    laserDuration: {laser_duration} ms")
    print(f"    cameraDelay: {camera_delay} ms")
    print(f"    cameraDuration: {camera_duration} ms")

    # Step 5: Configure Tiger controller
    print("\nStep 5: Configuring Tiger controller SPIM state machine...")

    # Ensure idle
    core.setProperty(galvo_device, "SPIMState", "Idle")
    core.setProperty(piezo_device, "SPIMState", "Idle")
    time.sleep(0.2)

    # CRITICAL: Set laser output mode
    core.setProperty(galvo_device, "LaserOutputMode", "shutter + side")
    print(f"  ✓ LaserOutputMode: {core.getProperty(galvo_device, 'LaserOutputMode')}")

    # Disable beam scanning
    core.setProperty(galvo_device, "BeamEnabled", "No")

    # Configure scan mirror X-axis (light sheet)
    core.setProperty(galvo_device, "SingleAxisXAmplitude(deg)", 2.0)
    core.setProperty(galvo_device, "SingleAxisXOffset(deg)", 0.0)
    core.setProperty(galvo_device, "SingleAxisXPattern", "1 - Triangle")
    core.setProperty(galvo_device, "SingleAxisXMode", "3 - Enabled with axes synced")

    # Configure scan mirror Y-axis (optional slice stepping)
    core.setProperty(galvo_device, "SingleAxisYAmplitude(deg)", 0.04)
    core.setProperty(galvo_device, "SingleAxisYOffset(deg)", 0.0)
    core.setProperty(galvo_device, "SingleAxisYPattern", "1 - Triangle")
    core.setProperty(galvo_device, "SingleAxisYMode", "3 - Enabled with axes synced")

    # Configure piezo
    slice_step_um = 1.0
    piezo_amplitude = (num_slices - 1) * slice_step_um / 2.0
    core.setProperty(piezo_device, "SA_AMPLITUDE", piezo_amplitude)
    core.setProperty(piezo_device, "SA_OFFSET", 0.0)
    core.setProperty(piezo_device, "SPIMNumSlices", num_slices)
    core.setProperty(piezo_device, "SPIMState", "Armed")
    print(f"  ✓ Piezo Armed: amplitude={piezo_amplitude} µm")

    # Set SPIM parameters
    core.setProperty(galvo_device, "SPIMNumSlices", num_slices)
    core.setProperty(galvo_device, "SPIMNumSides", 1)
    core.setProperty(galvo_device, "SPIMFirstSide", "A")
    core.setProperty(galvo_device, "SPIMNumRepeats", 1)
    core.setProperty(galvo_device, "SPIMAlternateDirectionsEnable", "No")

    # ⚠️ CRITICAL: Explicitly set ALL timing properties
    print("  ✓ Setting SPIM timing properties...")
    core.setProperty(galvo_device, "SPIMDelayBeforeScan(ms)", scan_delay)
    core.setProperty(galvo_device, "SPIMScanDuration(ms)", scan_duration)
    core.setProperty(galvo_device, "SPIMDelayBeforeLaser(ms)", laser_delay)
    core.setProperty(galvo_device, "SPIMLaserDuration(ms)", laser_duration)
    core.setProperty(galvo_device, "SPIMDelayBeforeCamera(ms)", camera_delay)
    core.setProperty(galvo_device, "SPIMCameraDuration(ms)", camera_duration)
    core.setProperty(galvo_device, "SPIMDelayBeforeSide(ms)", 0.0)
    core.setProperty(galvo_device, "SPIMDelayBeforeRepeat(ms)", 0.0)

    # Verify timing properties are set
    print(f"    SPIMCameraDuration(ms): {core.getProperty(galvo_device, 'SPIMCameraDuration(ms)')} ← MUST BE > 0!")

    camera_duration_check = float(core.getProperty(galvo_device, 'SPIMCameraDuration(ms)'))
    if camera_duration_check <= 0:
        raise Exception("SPIMCameraDuration is 0 - triggers will not be generated!")

    # Step 6: Start camera sequence
    print("\nStep 6: Starting camera sequence acquisition...")
    core.prepareSequenceAcquisition(camera_name)
    time.sleep(0.1)
    core.startSequenceAcquisition(camera_name, num_slices, 0, True)

    print(f"  Sequence running: {core.isSequenceRunning(camera_name)}")
    print(f"  Buffer capacity: {core.getBufferTotalCapacity()}")

    # Step 7: Trigger SPIM state machine
    print("\nStep 7: Triggering SPIM state machine...")
    core.setProperty(galvo_device, "SPIMState", "Running")
    time.sleep(0.1)

    spim_state = core.getProperty(galvo_device, "SPIMState")
    print(f"  SPIMState: {spim_state}")

    if spim_state != "Running":
        raise Exception(f"Failed to start SPIM (state={spim_state})")

    # Step 8: Wait for images
    print("\nStep 8: Waiting for hardware-triggered images...")
    expected_time = num_slices * scan_duration / 1000.0
    timeout = expected_time * 2 + 10.0
    print(f"  Expected time: {expected_time:.1f}s, timeout: {timeout:.1f}s")

    start = time.time()
    last_print = start

    while core.getRemainingImageCount() < num_slices:
        elapsed = time.time() - start

        if elapsed > timeout:
            print(f"  Timeout after {elapsed:.1f}s")
            break

        if (time.time() - last_print) >= 0.5:
            count = core.getRemainingImageCount()
            print(f"  t={elapsed:.1f}s: images={count}/{num_slices}")
            last_print = time.time()

        time.sleep(0.01)

    # Step 9: Retrieve images
    count = core.getRemainingImageCount()
    print(f"\n{'='*70}")

    if count >= num_slices:
        print(f"✓ SUCCESS! Acquired {count} images")
        print(f"{'='*70}")

        import rpyc
        images = []
        print(f"\nRetrieving {count} images...")
        for i in range(count):
            img = core.popNextImage()
            img = rpyc.classic.obtain(img)
            images.append(img)
            if i < 5 or i >= count - 5:  # Print first and last 5
                print(f"  Image {i+1}: shape={img.shape}, range=[{img.min()}, {img.max()}]")

        volume = np.array(images)
        print(f"\nVolume shape: {volume.shape}")

        # Save as TIFF
        from PIL import Image
        img_list = [Image.fromarray(img.astype(np.uint16)) for img in images]
        img_list[0].save('spim_volume_fixed.tif', save_all=True, append_images=img_list[1:])
        print(f"Saved to: spim_volume_fixed.tif")

    else:
        print(f"✗ FAILED - Got {count}/{num_slices} images")
        print(f"{'='*70}")

        # Diagnostic output
        print("\nDiagnostics:")
        print(f"  SPIMState: {core.getProperty(galvo_device, 'SPIMState')}")
        print(f"  LaserOutputMode: {core.getProperty(galvo_device, 'LaserOutputMode')}")
        print(f"  SPIMCameraDuration(ms): {core.getProperty(galvo_device, 'SPIMCameraDuration(ms)')}")
        print(f"  Camera trigger: {core.getProperty(camera_name, 'TRIGGER SOURCE')}")
        print("\nPossible issues:")
        print("  - Check physical BNC cable connection to camera")
        print("  - Verify TTL output with oscilloscope")
        print("  - Check camera firmware and trigger settings")

finally:
    # Cleanup
    print(f"\n{'='*70}")
    print("CLEANUP")
    print(f"{'='*70}")

    try:
        if core.isSequenceRunning(camera_name):
            core.stopSequenceAcquisition(camera_name)
            print("  Stopped camera sequence")
    except:
        pass

    try:
        core.setProperty(galvo_device, "SPIMState", "Idle")
        core.setProperty(piezo_device, "SPIMState", "Idle")
        print("  Reset SPIM to Idle")
    except:
        pass

    try:
        core.setConfig("Laser", "ALL OFF")
        print("  Lasers OFF")
    except:
        pass
```

---

## Java Source Code Reference

### Key Files Analyzed

1. **Cameras.java** (870 lines)
   - Location: `micro-manager/plugins/ASIdiSPIM/src/main/java/org/micromanager/asidispim/data/Cameras.java`
   - Contains: `setCameraTriggerMode()` method (lines 228-260)
   - Camera-specific property configuration for all supported cameras

2. **ControllerUtils.java** (886 lines)
   - Location: `micro-manager/plugins/ASIdiSPIM/src/main/java/org/micromanager/asidispim/utils/ControllerUtils.java`
   - Contains: `prepareControllerForAquisition()` (lines 103-223)
   - Contains: `prepareControllerForAquisition_Side()` (lines 293-537)
   - Contains: `triggerControllerStartAcquisition()` (lines 823-851)
   - **BUG LOCATION**: Line 434 - only sets `SPIM_DURATION_SCAN`, missing other timing properties

3. **AcquisitionPanel.java** (3525 lines)
   - Location: `micro-manager/plugins/ASIdiSPIM/src/main/java/org/micromanager/asidispim/AcquisitionPanel.java`
   - Contains: `getTimingFromPeriodAndLightExposure()` (lines 1105-1240)
   - Contains: Acquisition workflow orchestration (lines 2200-3000)
   - UI spinner bindings for timing properties (lines 446-499)

4. **Properties.java** (798 lines)
   - Location: `micro-manager/plugins/ASIdiSPIM/src/main/java/org/micromanager/asidispim/data/Properties.java`
   - Contains: All property name constants (lines 78-175)
   - Contains: All property value constants (lines 293-369)

5. **CameraModes.java** (289 lines)
   - Location: `micro-manager/plugins/ASIdiSPIM/src/main/java/org/micromanager/asidispim/data/CameraModes.java`
   - Defines camera trigger modes: EDGE, LEVEL, OVERLAP, PSEUDO_OVERLAP, LIGHT_SHEET, INTERNAL

6. **SliceTiming.java** (98 lines)
   - Location: `micro-manager/plugins/ASIdiSPIM/src/main/java/org/micromanager/asidispim/utils/SliceTiming.java`
   - Data structure for timing parameters

### Critical Code Sections

**Camera Configuration (Hamamatsu):**
```java
// Cameras.java:231-260
case HAMCAM:
    props_.setPropValue(devKey, Properties.Keys.TRIGGER_SOURCE,
            ((mode == CameraModes.Keys.INTERNAL)
                ? Properties.Values.INTERNAL
                : Properties.Values.EXTERNAL));

    props_.setPropValue(devKey, Properties.Keys.SENSOR_MODE,
            ((mode == CameraModes.Keys.LIGHT_SHEET)
                ? Properties.Values.PROGRESSIVE
                : Properties.Values.AREA));

    switch (mode) {
        case EDGE:
        case LIGHT_SHEET:
            props_.setPropValue(devKey, Properties.Keys.TRIGGER_ACTIVE,
                    Properties.Values.EDGE);
            break;
        case LEVEL:
            props_.setPropValue(devKey, Properties.Keys.TRIGGER_ACTIVE,
                    Properties.Values.LEVEL);
            break;
        case OVERLAP:
            props_.setPropValue(devKey, Properties.Keys.TRIGGER_ACTIVE,
                    Properties.Values.SYNCREADOUT);
            break;
    }
    break;
```

**Tiger Controller Configuration (Missing Properties):**
```java
// ControllerUtils.java:434 - ONLY sets scan duration!
props_.setPropValue(galvoDevice, Properties.Keys.SPIM_DURATION_SCAN,
        settings.sliceTiming.scanPeriod, skipScannerWarnings);

// ⚠️ MISSING: These timing properties are NEVER set:
// props_.setPropValue(galvoDevice, Properties.Keys.SPIM_DELAY_SCAN, ...);
// props_.setPropValue(galvoDevice, Properties.Keys.SPIM_DELAY_LASER, ...);
// props_.setPropValue(galvoDevice, Properties.Keys.SPIM_DURATION_LASER, ...);
// props_.setPropValue(galvoDevice, Properties.Keys.SPIM_DELAY_CAMERA, ...);
// props_.setPropValue(galvoDevice, Properties.Keys.SPIM_DURATION_CAMERA, ...); ← CRITICAL!
```

**Timing Calculation:**
```java
// AcquisitionPanel.java:1161-1193
final float cameraReadoutTime = computeCameraReadoutTime();
final float cameraResetTime = computeCameraResetTime();
final float cameraReadout_max = MyNumberUtils.ceilToQuarterMs(cameraReadoutTime);
final float cameraReset_max = MyNumberUtils.ceilToQuarterMs(cameraResetTime);
final float globalExposureDelay_max = cameraReadout_max + cameraReset_max;

final float laserDuration = MyNumberUtils.roundToQuarterMs(acqSettings.desiredLightExposure);
final float scanDuration = laserDuration + 2 * scanLaserBufferTime;

float scanDelayFilter = 0.39f / scanFilterFreq;
if (devices_.isValidMMDevice(Devices.Keys.PLOGIC)) {
    scanDelayFilter -= 0.25f;
}

s.scanDelay = globalExposureDelay_max - scanLaserBufferTime - scanDelayFilter;
s.laserDelay = globalExposureDelay_max;
s.cameraDelay = cameraReadout_max;
s.cameraDuration = 1;  // EDGE mode
```

**SPIM Trigger:**
```java
// ControllerUtils.java:843-844
props_.setPropValue(galvoDevice, Properties.Keys.SPIM_STATE,
        Properties.Values.SPIM_RUNNING, getSkipScannerWarnings(galvoDevice));
```

---

## Conclusion

This document provides a complete reference for understanding and debugging the ASI diSPIM hardware-triggered acquisition system. The critical discovery is that **timing properties must be explicitly set** on the Tiger controller, as the Java plugin does not write them automatically in Simple Timing mode.

**Key takeaways:**

1. **Always set `SPIMCameraDuration(ms) > 0`** - If 0, no TTL pulses are generated
2. **Verify `LaserOutputMode = "shutter + side"`** - Required for TTL output
3. **Explicitly set all timing properties** before starting acquisition
4. **Use oscilloscope** to verify TTL outputs if camera not receiving triggers
5. **Account for PLogic delay** (-0.25ms) in timing calculations if present

For debugging, start with the diagnostic commands in the Debugging Guide section to verify each component's configuration before attempting acquisition.

---

## Lessons Learned - Debugging Summary

This section documents the complete debugging journey from non-working to fully functional hardware-triggered SPIM acquisition.

### Timeline of Discovery

1. **Initial Problem**: Camera not receiving hardware triggers, "index was 0 count was 0" error
2. **First Investigation**: Analyzed ASI diSPIM Java plugin source code to understand timing and trigger mechanisms
3. **First Fix**: Added missing SPIM timing property configuration (especially `SPIMCameraDuration(ms)`)
4. **Property Name Error**: Fixed "TriggerActive" → "TRIGGER ACTIVE" (with space)
5. **Exposure Limit Discovery**: Found PROGRESSIVE mode has ~10-12ms max exposure (from Nature Protocols paper)
6. **Device Name Error**: Fixed "Piezo:A:37" → "PiezoStage:P:34" (but piezo not needed!)
7. **Piezo Misconception**: Realized galvo Y-axis handles slice stepping, not piezo
8. **Buffer Configuration**: Added circular buffer setup (1200MB footprint)
9. **CRITICAL FIX**: Switched from AREA to PROGRESSIVE sensor mode ← **This was the key!**
10. **SUCCESS**: 100 slices acquired at 59.1 fps!

### Root Causes Identified

**Primary Issue:** SENSOR MODE must be PROGRESSIVE
- AREA mode silently fails with external triggers
- Camera sequence stops immediately (`isSequenceRunning()` returns False)
- No error messages, just empty buffer
- **Solution:** `core.setProperty(camera_name, "SENSOR MODE", "PROGRESSIVE")`

**Secondary Issues:**
1. SPIM timing properties not set (controller uses stale values)
2. Property name case sensitivity ("TRIGGER ACTIVE" not "TriggerActive")
3. Circular buffer not configured (capacity = 0)
4. Wrong device names for system
5. Attempting to configure piezo for SPIM (not needed)

### What the Java Plugin Does Differently

The ASI diSPIM Java plugin works because:
1. It always sets SENSOR MODE based on camera mode selection (PROGRESSIVE for light sheet)
2. UI spinners in "Advanced Timing" mode write properties directly to controller
3. Camera configuration happens before any acquisition setup
4. Proper state machine flow is enforced

### Key Insights

1. **Hardware triggering is NOT plug-and-play** - Many properties must be set explicitly
2. **Sensor mode matters critically** - PROGRESSIVE vs AREA makes all the difference
3. **Silent failures are common** - Camera can reject external triggers without error messages
4. **Tiger controller is stateful** - Properties persist between acquisitions
5. **Galvo does it all** - Both light sheet scanning (X) and slice stepping (Y) happen on the galvo
6. **Timing is precise** - All values rounded to 0.25ms (Tiger firmware resolution)

### Performance Achieved

**Test Configuration:**
- 100 slices
- 2304 × 2304 pixels per slice
- 5ms exposure
- PROGRESSIVE sensor mode
- Galvo Y-axis slice stepping

**Results:**
- Acquisition time: 1.7 seconds
- Frame rate: 59.1 fps
- Total data: 5.08 GB (100 × 2304 × 2304 × 2 bytes)
- Success rate: 100% (all 100 slices captured)
- Image quality: Good (mean ~104, range 26-189)

### Recommendations for Future Work

1. **Start with PROGRESSIVE mode** - Don't waste time with AREA mode
2. **Always set timing properties explicitly** - Never rely on defaults
3. **Verify LaserOutputMode early** - Check it's "shutter + side"
4. **Configure buffer first** - Set memory footprint before starting sequence
5. **Test with fewer slices initially** - Start with 5-10 slices to debug quickly
6. **Use oscilloscope if available** - Verify TTL outputs directly
7. **Read the Nature Protocols paper** - Contains critical timing information
8. **Trust the working script** - Use `test_volume_acq.py` as reference

---

**Document Version:** 2.0
**Last Updated:** 2025-01-14
**Status:** ✅ Verified working implementation
**Maintainer:** dispim@shrofflab

**Acknowledgments:**
- Analysis based on Micro-Manager ASI diSPIM Java plugin source code
- Nature Protocols paper: Wu et al., nprot.2014.172
- Debugging collaboration with Claude Code microscope-control-expert agent
