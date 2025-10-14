# How the Light Sheet is Created - Detailed Explanation

**Question:** How does the Java code create a light sheet? Is it just by setting SPIM state, or is there more to it?

**Answer:** The light sheet is created by **two separate galvo axes working together**, not just by arming SPIM. Let me explain.

---

## The Key Insight: Two Axes, Two Jobs

The galvo scanner has **two perpendicular axes** (X and Y):

1. **X-axis ("Sheet-generating axis")**: Creates the light sheet by **rapidly scanning** perpendicular to the detection path
2. **Y-axis ("Slice-selecting axis")**: Selects which **Z-plane** to image by changing the light sheet angle

```
Top view of sample:

   Camera          Objective
      ↑               ↓
      |           ___/
      |       ___/  Light sheet (created by X-axis scan)
      |   ___/
      |__/

Side view:

   Sample          Light sheet position
      █            (controlled by Y-axis tilt)
      █         ↗
      █       ↗   ← Y-axis changes this angle
      █     ↗
      █   ↗
      █ ↗
```

---

## How It Works: The X-Axis Creates the Sheet

### Normal (Static) Galvo Position

When you just move the galvo to a position with `M X=500 Y=200`, the laser beam points to **one spot**. That's not a sheet - that's a point!

```
Laser →  O  ← Single point
```

### Light Sheet Mode: Continuous X-Axis Scanning

To create a **sheet** of light, the X-axis **continuously scans** back and forth very rapidly:

```
Laser →  ||||||||||||  ← Light sheet!
         X-axis scanning
```

This is done using **Single Axis Mode (SAM)** on the X-axis.

---

## The Configuration in Java Code

Let me trace through ControllerUtils.java to show exactly how this works.

### Step 1: Configure Light Sheet Width and Offset

**File:** `ControllerUtils.java` lines 422-437

```java
// send sheet width/offset
float sheetWidth = getSheetWidth(settings.cameraMode, cameraDevice, side);
float sheetOffset = getSheetOffset(settings.cameraMode, side);

if (settings.cameraMode == CameraModes.Keys.LIGHT_SHEET) {
    // adjust sheet width and offset to account for settle time
    final float settleTime = props_.getPropValueFloat(Devices.Keys.PLUGIN,
                                                       Properties.Keys.PLUGIN_LS_SCAN_SETTLE);
    final float readoutTime = settings.sliceTiming.laserDuration - 0.25f;
    sheetOffset -= (sheetWidth * settleTime/readoutTime)/2;
    sheetWidth += (sheetWidth * settleTime/readoutTime);
}

// THIS IS KEY: Set X-axis amplitude and offset for sheet scanning
props_.setPropValue(galvoDevice, Properties.Keys.SA_AMPLITUDE_X_DEG, sheetWidth);
props_.setPropValue(galvoDevice, Properties.Keys.SA_OFFSET_X_DEG, sheetOffset);
```

**What this does:**
- `SA_AMPLITUDE_X_DEG`: How wide the light sheet is (X-axis scan range)
- `SA_OFFSET_X_DEG`: Where the light sheet is centered

These correspond to the **Single Axis Mode** properties for the X-axis!

### Step 2: Configure Y-Axis for Slice Selection

**File:** `ControllerUtils.java` lines 335-340

```java
props_.setPropValue(galvoDevice, Properties.Keys.SPIM_DURATION_SCAN,
      settings.sliceTiming.scanPeriod, skipScannerWarnings);
props_.setPropValue(galvoDevice, Properties.Keys.SA_AMPLITUDE_Y_DEG,
      sliceAmplitude, skipScannerWarnings);
props_.setPropValue(galvoDevice, Properties.Keys.SA_OFFSET_Y_DEG,
      sliceCenter, skipScannerWarnings);
```

**What this does:**
- `SA_AMPLITUDE_Y_DEG`: Range of Y-axis motion (determines Z-range covered)
- `SA_OFFSET_Y_DEG`: Center position of Y-axis (center of Z-stack)
- `SPIM_DURATION_SCAN`: How long the scan takes per slice

The Y-axis **steps** through positions for each Z-slice, while the X-axis **continuously scans** to create the sheet.

---

## The Complete Picture

### What Each Property Does

| Property | Axis | Purpose | Example Value |
|----------|------|---------|---------------|
| `SA_AMPLITUDE_X_DEG` | X-axis | Width of light sheet | 2.0° (covers 2mm field) |
| `SA_OFFSET_X_DEG` | X-axis | Center position of sheet | 0.0° (centered) |
| `SA_AMPLITUDE_Y_DEG` | Y-axis | Z-range covered | 5.0° (100 slices × 0.5µm) |
| `SA_OFFSET_Y_DEG` | Y-axis | Center Z position | 1.5° (offset from home) |
| `SPIM_DURATION_SCAN` | Both | Scan duration per slice | 10.0 ms |

### Timeline of One Slice Acquisition

```
Time (ms):  0    2    4    6    8   10
            |----|----|----|----|----|
X-axis:     \\\\\\\\\\\\\\\\\\\\\\\\\\  ← Continuously scanning (creates sheet)
Y-axis:     ──────────────────────────  ← Held at slice position
Laser:           ████████████           ← On during exposure
Camera:          [──EXPOSE──]           ← Capturing image
```

**During this 10ms:**
- X-axis scans back and forth ~10 times (creating light sheet)
- Y-axis stays at one angle (selecting Z-plane)
- Laser pulses when camera is exposing
- Camera integrates light over exposure time

### What SPIM State Machine Does

When you set `SPIMState = "Armed"` then `SPIMState = "Running"`:

```
For each slice (Z-position):
    1. Move Y-axis to slice angle
    2. Start X-axis scanning (create light sheet)
    3. Wait for settling (galvo stabilizes)
    4. Trigger laser
    5. Trigger camera
    6. Wait for exposure
    7. Turn off laser
    8. Repeat for next slice
```

The **X-axis scanning happens automatically** during SPIM mode because you've set `SA_AMPLITUDE_X_DEG` and `SA_OFFSET_X_DEG`!

---

## The Critical Configuration Code

Here's the minimal code needed to create a light sheet:

### Java (from ASIdiSPIM plugin)

```java
// Configure X-axis for sheet generation
props_.setPropValue(galvoDevice, Properties.Keys.SA_AMPLITUDE_X_DEG,
                    2.0f);  // 2 degree sheet width
props_.setPropValue(galvoDevice, Properties.Keys.SA_OFFSET_X_DEG,
                    0.0f);  // Centered

// Configure Y-axis for slice selection
props_.setPropValue(galvoDevice, Properties.Keys.SA_AMPLITUDE_Y_DEG,
                    5.0f);  // 5 degree range (Z-stack)
props_.setPropValue(galvoDevice, Properties.Keys.SA_OFFSET_Y_DEG,
                    2.5f);  // Center of Z-stack

// Configure SPIM parameters
props_.setPropValue(galvoDevice, Properties.Keys.SPIM_NUM_SLICES, 100);
props_.setPropValue(galvoDevice, Properties.Keys.SPIM_DURATION_SCAN, 10.0f);
props_.setPropValue(galvoDevice, Properties.Keys.SPIM_DELAY_CAMERA, 0.5f);
props_.setPropValue(galvoDevice, Properties.Keys.SPIM_DURATION_CAMERA, 9.0f);

// Arm and trigger
props_.setPropValue(galvoDevice, Properties.Keys.SPIM_STATE,
                    Properties.Values.SPIM_ARMED);
// ... then trigger via TTL or:
props_.setPropValue(galvoDevice, Properties.Keys.SPIM_STATE,
                    Properties.Values.SPIM_RUNNING);
```

### Python (for Gently)

```python
from gently.mmcore_wrapper import (
    GentlyProperties, GentlyDevices,
    DeviceKeys, PropertyKeys, PropertyValues
)

# Initialize
devices = GentlyDevices(core)
devices.set_device(DeviceKeys.GALVO_A, "Scanner:AB:33")
props = GentlyProperties(core, devices)

# Configure X-axis for sheet generation (THE KEY PART!)
props.set_property(DeviceKeys.GALVO_A, PropertyKeys.SA_AMPLITUDE_X_DEG, 2.0)
props.set_property(DeviceKeys.GALVO_A, PropertyKeys.SA_OFFSET_X_DEG, 0.0)

# Configure Y-axis for slice selection
props.set_property(DeviceKeys.GALVO_A, PropertyKeys.SA_AMPLITUDE_Y_DEG, 5.0)
props.set_property(DeviceKeys.GALVO_A, PropertyKeys.SA_OFFSET_Y_DEG, 2.5)

# Configure SPIM parameters
props.set_property(DeviceKeys.GALVO_A, PropertyKeys.SPIM_NUM_SLICES, 100)
props.set_property(DeviceKeys.GALVO_A, PropertyKeys.SPIM_DURATION_SCAN, 10.0)
props.set_property(DeviceKeys.GALVO_A, PropertyKeys.SPIM_DELAY_CAMERA, 0.5)
props.set_property(DeviceKeys.GALVO_A, PropertyKeys.SPIM_DURATION_CAMERA, 9.0)

# Arm and trigger
props.set_property(DeviceKeys.GALVO_A, PropertyKeys.SPIM_STATE,
                   PropertyValues.SPIM_ARMED)
# ... then trigger via TTL or:
props.set_property(DeviceKeys.GALVO_A, PropertyKeys.SPIM_STATE,
                   PropertyValues.SPIM_RUNNING)
```

---

## Where Does Sheet Width Come From?

Looking at `ControllerUtils.java` lines 531-570:

### Method 1: Automatic from Camera ROI

```java
if (autoSheet) {
    Rectangle roi = core_.getROI(devices_.getMMDevice(cameraDevice));

    // Calculate based on camera height and calibration slope
    final float sheetSlope = 2;  // millidegrees per pixel
    sheetWidth = roi.height * sheetSlope / 1000f;  // Convert to degrees
    sheetWidth *= 1.1f;  // 10% margin
}
```

**Translation:** If your camera ROI is 512 pixels high, and the calibration is 2 millidegrees/pixel:
- Sheet width = 512 × 2 / 1000 × 1.1 = 1.126 degrees

### Method 2: Manual Setting

```java
else {
    // User-specified sheet width from setup panel
    sheetWidth = props_.getPropValueFloat(Devices.Keys.PLUGIN, widthProp);
}
```

---

## Summary: The Answer to Your Question

**Question:** "Is the light sheet created automatically by setting SPIM state?"

**Answer:** **Almost, but not quite!**

The light sheet is created by the **combination** of:

1. **Setting `SA_AMPLITUDE_X_DEG`** (sheet width) - **This makes X-axis scan**
2. **Setting `SA_OFFSET_X_DEG`** (sheet position)
3. **Arming SPIM** (`SPIMState = "Armed"`)
4. **Running SPIM** (`SPIMState = "Running"`)

**When SPIM runs:**
- The firmware **automatically** starts the X-axis scanning based on `SA_AMPLITUDE_X_DEG`
- The firmware **automatically** moves the Y-axis through slice positions based on `SA_AMPLITUDE_Y_DEG`
- The firmware **automatically** synchronizes laser/camera with the scanning

**You must configure the amplitude properties BEFORE arming SPIM!**

If you only set `SPIMState = "Running"` without setting `SA_AMPLITUDE_X_DEG`, you'll get **no light sheet** (or a point instead of a sheet).

---

## Practical Checklist for Gently

When implementing volume scanning, you **must** configure these properties:

### Required for Light Sheet

- ✅ `SA_AMPLITUDE_X_DEG` - Width of light sheet (X-axis scan range)
- ✅ `SA_OFFSET_X_DEG` - Position of light sheet (usually 0)

### Required for Z-Stack

- ✅ `SA_AMPLITUDE_Y_DEG` - Z-range (Y-axis total movement)
- ✅ `SA_OFFSET_Y_DEG` - Center Z-position (Y-axis center)
- ✅ `SPIM_NUM_SLICES` - Number of Z-slices

### Required for Timing

- ✅ `SPIM_DURATION_SCAN` - Scan duration per slice
- ✅ `SPIM_DELAY_CAMERA` - Delay before camera trigger
- ✅ `SPIM_DURATION_CAMERA` - Camera exposure time

### Required for Triggering

- ✅ `SPIM_STATE` - Arm → Running

**Only after ALL of these are set** will you get a proper light-sheet volume scan!

---

**Document Version:** 1.0
**Last Updated:** 2025-10-12
