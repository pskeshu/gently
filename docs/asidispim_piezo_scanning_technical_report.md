# ASI diSPIM Piezo Scanning Implementation - Comprehensive Technical Report

**Generated:** October 15, 2025
**Analysis Based On:** ASI diSPIM Micro-Manager Plugin (Java) and Python Reference Implementation

---

## Executive Summary

This report provides a complete technical analysis of the piezo scanning implementation in the ASI diSPIM Micro-Manager plugin, covering piezo position calculations, state machine coordination, synchronization mechanisms, scanning modes, and hardware property configuration.

**Key Finding:** The ASI diSPIM achieves synchronized galvo/piezo/camera/laser operation through a hardware state machine in the Tiger controller. The piezo is controlled indirectly by the galvo/micro-mirror card's SPIM state machine, which sends TTL pulses to the piezo card at precisely calculated intervals.

---

## 1. Piezo Scanning Modes

The ASI diSPIM plugin supports **seven distinct acquisition modes** defined in `AcquisitionModes.java`:

### Mode Definitions (Lines 51-59)

```java
public static enum Keys {
    PIEZO_SLICE_SCAN("Synchronous piezo/slice scan", 1),    // DEFAULT mode
    NO_SCAN("No scan (fixed sheet)", 3),
    STAGE_SCAN("Stage scan", 4),
    STAGE_SCAN_INTERLEAVED("Stage scan interleaved", 5),
    STAGE_SCAN_UNIDIRECTIONAL("Stage scan unidirectional", 7),
    SLICE_SCAN_ONLY("Slice scan only (unusual)", 2),       // Galvo Y only
    PIEZO_SCAN_ONLY("Piezo scan only (unusual)", 6),       // Piezo only
    NONE("None", 0);
}
```

### Mode Behavior Analysis

| Mode | Piezo Movement | Galvo Y Movement | Use Case |
|------|---------------|-----------------|----------|
| **PIEZO_SLICE_SCAN** | ✅ Steps through Z | ✅ Synchronized deflection | **Standard SPIM** - Both move together |
| **PIEZO_SCAN_ONLY** | ✅ Steps through Z | ❌ Disabled (amplitude=0) | Unusual - Piezo only, sheet stays fixed |
| **SLICE_SCAN_ONLY** | ❌ Disabled (amplitude=0) | ✅ Deflects through range | Unusual - Galvo only, piezo fixed |
| **STAGE_SCAN*** | ❌ Moved to HOME | ❌ Disabled | XY stage scanning mode |
| **NO_SCAN** | ❌ Disabled | ❌ Disabled | Fixed sheet, no volume scan |

**Key Implementation:** Lines 365-374 in `ControllerUtils.java`

```java
float piezoAmplitude;
switch (settings.spimMode) {
case NO_SCAN:
case STAGE_SCAN:
case STAGE_SCAN_INTERLEAVED:
case STAGE_SCAN_UNIDIRECTIONAL:
    piezoAmplitude = 0.0f;  // NO piezo movement
    break;
default:  // PIEZO_SLICE_SCAN, PIEZO_SCAN_ONLY, SLICE_SCAN_ONLY
    piezoAmplitude = (settings.numSlices - 1) * settings.stepSizeUm;
}
```

---

## 2. Piezo Position Calculations

### 2.1 Core Calculation Logic (ControllerUtils.java:349-402)

The piezo position is calculated through **three key variables**:

```java
float piezoCenter;   // Center position of piezo scan (µm)
float piezoAmplitude; // Total scan range (µm)
float sliceRate;     // Calibration: piezo µm per galvo degree
float sliceOffset;   // Calibration: offset in µm
```

### 2.2 Piezo Center Position Determination (Lines 350-362)

**THREE methods** to determine `piezoCenter`:

```java
if (settings.isStageScanning && devices_.isValidMMDevice(piezoDevice)) {
    // METHOD 1: Stage Scanning Mode
    // Piezo stays at HOME position (usually 0 µm)
    piezoCenter = props_.getPropValueFloat(piezoDevice,
                   Properties.Keys.HOME_POSITION) * 1000;  // mm → µm

} else if (settings.centerAtCurrentZ) {
    // METHOD 2: Center at Current Z Position
    // Use current piezo position from hardware
    piezoCenter = (float) positions_.getUpdatedPosition(piezoDevice,
                           Joystick.Directions.NONE);

} else {
    // METHOD 3: Use Saved Center Position (DEFAULT)
    // From Setup panel preferences
    piezoCenter = prefs_.getFloat(
        MyStrings.PanelNames.SETUP.toString() + side.toString(),
        Properties.Keys.PLUGIN_PIEZO_CENTER_POS, 0.0f);
}
```

### 2.3 Piezo Amplitude Calculation (Lines 363-374)

```java
float piezoAmplitude;
switch (settings.spimMode) {
case NO_SCAN:
case STAGE_SCAN:
case STAGE_SCAN_INTERLEAVED:
case STAGE_SCAN_UNIDIRECTIONAL:
    piezoAmplitude = 0.0f;  // NO movement
    break;
default:
    // Total scan range = (N-1) * step_size
    // E.g., 100 slices × 1 µm = 99 µm total range
    piezoAmplitude = (settings.numSlices - 1) * settings.stepSizeUm;
}
```

**Physical Interpretation:**
- `numSlices = 100`, `stepSize = 1.0 µm`
- `piezoAmplitude = 99 µm` (not 100!)
- Scan from: `piezoCenter - 49.5 µm` to `piezoCenter + 49.5 µm`

### 2.4 OVERLAP Mode Adjustment (Lines 383-388)

**Special case for synchronous/overlap camera mode:**

```java
if (cameraMode == CameraModes.Keys.OVERLAP) {
    // Take N+1 triggers but only use first N images
    // Adjust amplitude to maintain same slice positions
    piezoAmplitude *= ((float)numSlices)/((float)numSlices-1f);
    piezoCenter += piezoAmplitude/(2*numSlices);
    numSlices += 1;  // Add extra trigger
}
```

**Mathematical Analysis:**
- Original: 100 slices spanning 99 µm
- Overlap: 101 slices spanning 100 µm
- **Effect:** First 100 slice positions remain identical
- **Reason:** Extra trigger at the end to prime camera for next volume

---

## 3. Galvo-Piezo Calibration

### 3.1 Calibration Formula (Lines 390-402)

The **critical calibration** that synchronizes galvo and piezo:

```java
// Calibration parameters (from Setup panel)
float sliceRate = prefs_.getFloat(
    MyStrings.PanelNames.SETUP.toString() + side.toString(),
    Properties.Keys.PLUGIN_RATE_PIEZO_SHEET, 100);  // µm/degree

float sliceOffset = prefs_.getFloat(
    MyStrings.PanelNames.SETUP.toString() + side.toString(),
    Properties.Keys.PLUGIN_OFFSET_PIEZO_SHEET, 0);  // µm

// Convert piezo positions to galvo angles
float sliceAmplitude = piezoAmplitude / sliceRate;  // degrees
float sliceCenter = (piezoCenter - sliceOffset) / sliceRate;  // degrees
```

**Calibration Equation:**
```
galvo_angle (deg) = (piezo_position (µm) - offset) / rate
```

**Example with Real Parameters:**
```python
PIEZO_GALVO_SLOPE = 100.306  # µm/°
PIEZO_GALVO_OFFSET = 4.102   # µm

# For piezo_pos = 50 µm:
galvo_y = (50 - 4.102) / 100.306 = 0.458°
```

### 3.2 Validation (Lines 393-397)

```java
if (MyNumberUtils.floatsEqual(sliceRate, 0.0f)) {
    MyDialogUtils.showError("Calibration slope for side " + side.toString() +
        " cannot be zero. Re-do calibration on Setup tab.");
    return false;
}
```

**Critical:** Zero calibration slope would cause division by zero → abort acquisition.

---

## 4. Mode-Specific Galvo Adjustments

### 4.1 PIEZO_SCAN_ONLY Mode (Lines 406-414)

**When only piezo moves (galvo stays fixed):**

```java
if (settings.spimMode.equals(AcquisitionModes.Keys.PIEZO_SCAN_ONLY)) {
    // Undo the OVERLAP centering shift for galvo
    if (cameraMode == CameraModes.Keys.OVERLAP) {
        float actualPiezoCenter = piezoCenter - piezoAmplitude/(2*(numSlices-1));
        sliceCenter = (actualPiezoCenter - sliceOffset) / sliceRate;
    }
    sliceAmplitude = 0.0f;  // DISABLE galvo Y-axis movement
}
```

**Rationale:**
- Piezo still moves through adjusted positions (for OVERLAP)
- Galvo stays at one angle (no sweep)
- Recalculate galvo center to correspond to **original** piezo center

### 4.2 SLICE_SCAN_ONLY Mode (Lines 451-458)

**When only galvo moves (piezo stays fixed):**

```java
if (settings.spimMode.equals(AcquisitionModes.Keys.SLICE_SCAN_ONLY)) {
    if (cameraMode == CameraModes.Keys.OVERLAP) {
        piezoCenter -= piezoAmplitude / (2 * (numSlices - 1));
    }
    piezoAmplitude = 0.0f;  // DISABLE piezo movement
}
```

**Rationale:**
- Galvo sweeps through angle range (calculated previously)
- Piezo stays at one position
- Undo OVERLAP shift for piezo center

### 4.3 Rounding for DAC Resolution (Lines 415-417)

```java
sliceAmplitude = MyNumberUtils.roundFloatToPlace(sliceAmplitude, 4);  // 0.0001°
sliceCenter = MyNumberUtils.roundFloatToPlace(sliceCenter, 4);
```

**Hardware constraint:** DAC resolution is ~0.0001 degrees.

---

## 5. Piezo Hardware Property Configuration

### 5.1 Properties Written to Piezo Device (Lines 473-481)

```java
if (devices_.isValidMMDevice(piezoDevice)) {
    // Round to nearest 0.001 µm (DAC resolution)
    piezoAmplitude = MyNumberUtils.roundFloatToPlace(piezoAmplitude, 3);
    piezoCenter = MyNumberUtils.roundFloatToPlace(piezoCenter, 3);

    // Write to hardware
    props_.setPropValue(piezoDevice,
        Properties.Keys.SA_AMPLITUDE, piezoAmplitude);    // µm
    props_.setPropValue(piezoDevice,
        Properties.Keys.SA_OFFSET, piezoCenter);          // µm
    props_.setPropValue(piezoDevice,
        Properties.Keys.SPIM_NUM_SLICES, numSlices);
    props_.setPropValue(piezoDevice,
        Properties.Keys.SPIM_STATE, Properties.Values.SPIM_ARMED);
}
```

### 5.2 Property Definitions (Properties.java:97-98)

```java
SA_AMPLITUDE("SingleAxisAmplitude(um)", false),
SA_OFFSET("SingleAxisOffset(um)", false),
```

**Note:** `false` flag means property is NOT always forced (only written if value changes).

### 5.3 Bounds Checking (Lines 460-468)

```java
float piezoMin = props_.getPropValueFloat(piezoDevice,
                     Properties.Keys.LOWER_LIMIT) * 1000;  // mm → µm
float piezoMax = props_.getPropValueFloat(piezoDevice,
                     Properties.Keys.UPPER_LIMIT) * 1000;

if (MyNumberUtils.outsideRange(piezoCenter - piezoAmplitude / 2, piezoMin, piezoMax)
    || MyNumberUtils.outsideRange(piezoCenter + piezoAmplitude / 2, piezoMin, piezoMax)) {
    MyDialogUtils.showError("Imaging piezo for side " + side.toString()
        + " would travel outside the piezo limits during acquisition.");
    return false;  // ABORT ACQUISITION
}
```

**Safety check:** Prevents damage to piezo from over-travel.

---

## 6. SPIM State Machine Coordination

### 6.1 State Transitions

The SPIM state machine has **three states** (Properties.java:301-303):

```java
SPIM_ARMED("Armed"),
SPIM_RUNNING("Running"),
SPIM_IDLE("Idle"),
```

### 6.2 Piezo State Machine Flow

**SEQUENCE:**

```
1. IDLE → ARMED (Line 480)
   ├── Piezo properties written (SA_AMPLITUDE, SA_OFFSET, SPIM_NUM_SLICES)
   └── Piezo ready to receive trigger

2. ARMED → RUNNING (Line 843, triggerControllerStartAcquisition)
   ├── For piezo scanning modes: direct trigger to galvo device
   └── Piezo begins stepping through positions

3. RUNNING → IDLE (Line 616, cleanUpControllerAfterAcquisition_Side)
   ├── After acquisition completes or is cancelled
   └── Piezo returns to safe state
```

### 6.3 Triggering Logic (ControllerUtils.java:823-851)

```java
public boolean triggerControllerStartAcquisition(
        final AcquisitionModes.Keys spimMode, final boolean isFirstSideA) {

    final Devices.Keys galvoDevice = isFirstSideA ?
        Devices.Keys.GALVOA : Devices.Keys.GALVOB;

    switch (spimMode) {
    case STAGE_SCAN:
    case STAGE_SCAN_INTERLEAVED:
    case STAGE_SCAN_UNIDIRECTIONAL:
        // ARM galvo first, then trigger XY stage
        props_.setPropValue(galvoDevice, Properties.Keys.SPIM_STATE,
            Properties.Values.SPIM_ARMED);
        props_.setPropValue(Devices.Keys.XYSTAGE, Properties.Keys.STAGESCAN_STATE,
            Properties.Values.SPIM_RUNNING);
        break;

    case PIEZO_SLICE_SCAN:
    case SLICE_SCAN_ONLY:
    case PIEZO_SCAN_ONLY:
    case NO_SCAN:
        // Direct trigger to galvo (which controls piezo via TTL)
        props_.setPropValue(galvoDevice, Properties.Keys.SPIM_STATE,
            Properties.Values.SPIM_RUNNING, getSkipScannerWarnings(galvoDevice));
        break;
    }
    return true;
}
```

**Key Point:** Piezo is controlled **indirectly** by the galvo/micro-mirror card, not triggered directly.

---

## 7. Piezo-Galvo Synchronization Mechanism

### 7.1 Hardware Architecture

```
┌─────────────────────────────────────────────────────────────┐
│ Tiger Controller (ASI)                                       │
│                                                              │
│  ┌────────────────┐        TTL Pulse Train                  │
│  │ Micro-Mirror   │───────────────────────────────────────► │
│  │ Card (Galvo)   │        (SPIM_NUM_SLICES pulses)        │
│  │                │                                          │
│  │ - SPIM State   │                                          │
│  │   Machine      │                                          │
│  │ - Generates    │        ┌──────────────┐                 │
│  │   TTL triggers │        │ Piezo Card   │                 │
│  │                │───────►│              │                 │
│  └────────────────┘        │ - SA_AMPLITUDE                 │
│                            │ - SA_OFFSET                     │
│         │                  │ - SPIM_NUM_SLICES               │
│         │ SA_AMPLITUDE_Y   │                                 │
│         │ SA_OFFSET_Y      │ Each TTL pulse → step piezo     │
│         ▼                  │                                 │
│    Galvo Y-axis            └──────────────┘                 │
│    (deflects light         │                                 │
│     sheet angle)           ▼                                 │
│                       Piezo Actuator                         │
│                       (moves objective)                      │
└─────────────────────────────────────────────────────────────┘
```

### 7.2 Synchronization Properties (Lines 321-322, 440-443)

**Galvo card properties:**

```java
// How many slices before piezo steps
props_.setPropValue(galvoDevice, Properties.Keys.SPIM_NUM_SLICES_PER_PIEZO,
    numSlicesPerPiezo, skipScannerWarnings);

// Total number of slices (= total piezo steps)
props_.setPropValue(galvoDevice, Properties.Keys.SPIM_NUM_SLICES,
    numSlices, skipScannerWarnings);
```

**Multi-channel hardware switching (Lines 315-322):**

```java
int numSlicesPerPiezo = 1;  // DEFAULT: piezo steps every slice
if (settings.useChannels && settings.channelMode == MultichannelModes.Keys.SLICE_HW) {
    numSlicesPerPiezo = settings.numChannels;  // Step after all channels
}
```

**Example:** 4-channel slice-by-slice hardware switching:
- Galvo triggers laser 1, laser 2, laser 3, laser 4 (via PLogic)
- After 4 slices → piezo steps once
- Repeat for all slice positions

---

## 8. Stage Scanning Mode (Piezo Disabled)

### 8.1 Piezo Home Position (Lines 351-355, 491-498)

```java
if (settings.isStageScanning && devices_.isValidMMDevice(piezoDevice)) {
    // Piezo positioned at HOME (usually 0 µm)
    piezoCenter = props_.getPropValueFloat(piezoDevice,
        Properties.Keys.HOME_POSITION) * 1000;  // mm → µm

    // ...later in code...

    // Move piezo to home position before stage scan starts
    try {
        if (devices_.isValidMMDevice(piezoDevice)) {
            core_.home(devices_.getMMDevice(piezoDevice));
        }
    } catch (Exception e) {
        ReportingUtils.showError(e, "Could not move piezo to home");
    }
}
```

### 8.2 SPIM_PIEZO_HOME_DISABLE Property (Lines 502-510)

```java
final boolean isInterleaved = (settings.isStageScanning
    && settings.spimMode == AcquisitionModes.Keys.STAGE_SCAN_INTERLEAVED);

if (isInterleaved) {
    // Tell firmware NOT to move piezo at start of acquisition
    props_.setPropValue(galvoDevice, Properties.Keys.SPIM_PIEZO_HOME_DISABLE,
        Properties.Values.YES, skipScannerWarnings);
} else {
    props_.setPropValue(galvoDevice, Properties.Keys.SPIM_PIEZO_HOME_DISABLE,
        Properties.Values.NO, skipScannerWarnings);
}
```

**Rationale:**
- **Normal stage scan:** Piezo homed by software (`core_.home()`) before acquisition
- **Interleaved stage scan:** Piezo MUST NOT move during scan (sides alternate every slice)
- Firmware would normally try to home piezo at start → disabled for interleaved

---

## 9. Comparison: Java vs. Python Implementation

### 9.1 test_volume_acq.py (Galvo-Only)

```python
# GALVO Y-AXIS CONFIGURED (Lines 215-219)
core.setProperty(galvo_device, "SingleAxisYAmplitude(deg)", 0.04)
core.setProperty(galvo_device, "SingleAxisYOffset(deg)", 0.0)
core.setProperty(galvo_device, "SingleAxisYPattern", "1 - Triangle")
core.setProperty(galvo_device, "SingleAxisYMode", "3 - Enabled with axes synced")

# NO PIEZO CONFIGURATION
# Comment (Line 222): "Piezo is NOT used for SPIM slice scanning"
```

**Analysis:** This script demonstrates **SLICE_SCAN_ONLY** mode (galvo Y-axis only, no piezo).

### 9.2 test_volume_acq_with_piezo.py (Current Implementation - INCOMPLETE)

```python
# PIEZO CONFIGURATION (Lines 248-286)
def configure_piezo_for_scan(piezo_device, start_um, end_um, num_slices):
    # Calculate galvo Y positions using calibration
    galvo_y_start = piezo_to_galvo_y(start_um)
    galvo_y_end = piezo_to_galvo_y(end_um)

    # Move piezo to start position
    core.setPosition(piezo_device, start_um)
    core.waitForDevice(piezo_device)
    time.sleep(0.5)  # Allow settling


# CALIBRATION (Lines 99-115)
def piezo_to_galvo_y(piezo_pos_um, slope=100.306, offset=4.102):
    """Convert piezo position to galvo Y-axis angle."""
    galvo_y_deg = (piezo_pos_um - offset) / slope
    return galvo_y_deg
```

**Analysis:**
- Manual positioning of piezo to start position
- Galvo configured with matching Y offset
- **MISSING:** No `SA_AMPLITUDE` or `SA_OFFSET` written to piezo device!
- **MISSING:** No `SPIM_STATE = "Armed"` property set!

### 9.3 Key Differences

| Aspect | Java (ControllerUtils.java) | Python (test_volume_acq_with_piezo.py) |
|--------|------------------------------|------------------------------------------|
| **Piezo properties** | ✅ `SA_AMPLITUDE`, `SA_OFFSET` written | ❌ NOT written (manual positioning) |
| **State machine** | ✅ ARMED → RUNNING transition | ❌ Not using SPIM state properly |
| **Synchronization** | ✅ `SPIM_NUM_SLICES_PER_PIEZO` | ❌ Not configured |
| **Calibration** | ✅ Automatic from prefs | ✅ Hardcoded values |
| **Bounds checking** | ✅ Validates piezo limits | ❌ No validation |
| **Mode handling** | ✅ 7 distinct modes | ❌ Single approach |

### 9.4 CRITICAL MISSING FEATURES in Current Python Implementation

The Python script does not properly configure the piezo for SPIM state machine control. The following properties MUST be written:

```python
# REQUIRED ADDITIONS:
core.setProperty(piezo_device, "SingleAxisAmplitude(um)", piezo_amplitude)
core.setProperty(piezo_device, "SingleAxisOffset(um)", piezo_center)
core.setProperty(piezo_device, "SPIMNumSlices", num_slices)
core.setProperty(piezo_device, "SPIMState", "Armed")
```

Without these properties, the piezo state machine is not configured for synchronized scanning and will not move in coordination with the galvo and camera triggers.

---

## 10. Complete Property Configuration Sequence

### 10.1 Piezo-Specific Properties

From `Properties.java` lines 97-98:

```java
SA_AMPLITUDE("SingleAxisAmplitude(um)", false),    // Piezo scan range
SA_OFFSET("SingleAxisOffset(um)", false),          // Piezo center position
```

### 10.2 Full Piezo Configuration (Ordered Sequence)

```
┌─────────────────────────────────────────────────────────────────┐
│ STEP 1: Calculate Positions                                     │
├─────────────────────────────────────────────────────────────────┤
│ piezoCenter = getCenterPosition(mode, prefs)                    │
│ piezoAmplitude = (numSlices - 1) × stepSize                     │
│ Adjust for OVERLAP mode if needed                               │
│ Round to 0.001 µm resolution                                    │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 2: Validate Bounds                                         │
├─────────────────────────────────────────────────────────────────┤
│ piezoMin = LOWER_LIMIT property                                 │
│ piezoMax = UPPER_LIMIT property                                 │
│ CHECK: piezoCenter ± amplitude/2 within [piezoMin, piezoMax]    │
│ ABORT if out of bounds                                          │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 3: Write Piezo Properties                                  │
├─────────────────────────────────────────────────────────────────┤
│ SA_AMPLITUDE = piezoAmplitude (µm)                              │
│ SA_OFFSET = piezoCenter (µm)                                    │
│ SPIM_NUM_SLICES = numSlices                                     │
│ SPIM_STATE = "Armed"                                            │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 4: Configure Galvo (Micro-Mirror Card)                     │
├─────────────────────────────────────────────────────────────────┤
│ SA_AMPLITUDE_Y_DEG = piezoAmplitude / calibrationSlope          │
│ SA_OFFSET_Y_DEG = (piezoCenter - calibrationOffset) / slope     │
│ SPIM_NUM_SLICES_PER_PIEZO = 1 (or numChannels for HW switching) │
│ SPIM_NUM_SLICES = numSlices                                     │
│ SPIM_PIEZO_HOME_DISABLE = NO (or YES for interleaved)           │
└─────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─────────────────────────────────────────────────────────────────┐
│ STEP 5: Trigger Acquisition                                     │
├─────────────────────────────────────────────────────────────────┤
│ SPIM_STATE = "Running" (on galvo device)                        │
│ → Galvo state machine generates TTL pulses                      │
│ → Piezo steps on each pulse                                     │
│ → Synchronized movement                                         │
└─────────────────────────────────────────────────────────────────┘
```

---

## 11. Mathematical Summary

### 11.1 Core Equations

**Piezo Scan Range:**
```
Start Position = piezoCenter - piezoAmplitude / 2
End Position   = piezoCenter + piezoAmplitude / 2
Step Size      = piezoAmplitude / (numSlices - 1)
```

**Calibration Transform:**
```
galvo_angle (°) = (piezo_position (µm) - calibration_offset (µm)) / calibration_rate (µm/°)
```

**Inverse Transform:**
```
piezo_position (µm) = galvo_angle (°) × calibration_rate (µm/°) + calibration_offset (µm)
```

### 11.2 Example Calculation

**Given:**
- numSlices = 100
- stepSize = 1.0 µm
- piezoCenter = 50 µm
- calibrationRate = 100.306 µm/°
- calibrationOffset = 4.102 µm

**Calculate:**
```
1. piezoAmplitude = (100 - 1) × 1.0 = 99 µm
2. piezoStart = 50 - 99/2 = 0.5 µm
3. piezoEnd = 50 + 99/2 = 99.5 µm
4. galvoCenter = (50 - 4.102) / 100.306 = 0.458°
5. galvoAmplitude = 99 / 100.306 = 0.987°
6. galvoStart = 0.458 - 0.987/2 = -0.035°
7. galvoEnd = 0.458 + 0.987/2 = 0.951°
```

---

## 12. Corrected Python Implementation

### 12.1 Required Function: configure_piezo_for_scan()

```python
def configure_piezo_for_scan(piezo_device, start_um, end_um, num_slices):
    """
    Configure piezo for SPIM state machine-controlled synchronized scanning.

    Based on ControllerUtils.java:349-481 from ASI diSPIM plugin.
    """
    print(f"\nConfiguring piezo: {piezo_device}")
    print(f"  Z-range: {start_um} to {end_um} µm")
    print(f"  Slices: {num_slices}")

    # Calculate piezo center and amplitude
    piezo_center = (start_um + end_um) / 2.0
    piezo_amplitude = end_um - start_um
    step_size = piezo_amplitude / (num_slices - 1) if num_slices > 1 else 0.0

    # Round to DAC resolution (0.001 µm)
    piezo_center = round(piezo_center, 3)
    piezo_amplitude = round(piezo_amplitude, 3)

    print(f"  Center: {piezo_center:.3f} µm")
    print(f"  Amplitude: {piezo_amplitude:.3f} µm")
    print(f"  Step size: {step_size:.3f} µm")

    # Validate against hardware limits
    try:
        piezo_min = float(core.getProperty(piezo_device, "LowerLim(mm)")) * 1000
        piezo_max = float(core.getProperty(piezo_device, "UpperLim(mm)")) * 1000

        piezo_start_actual = piezo_center - piezo_amplitude / 2
        piezo_end_actual = piezo_center + piezo_amplitude / 2

        print(f"\n  Hardware limits: [{piezo_min:.1f}, {piezo_max:.1f}] µm")
        print(f"  Requested range: [{piezo_start_actual:.1f}, {piezo_end_actual:.1f}] µm")

        if piezo_start_actual < piezo_min or piezo_end_actual > piezo_max:
            raise ValueError(
                f"Piezo range [{piezo_start_actual:.1f}, {piezo_end_actual:.1f}] µm "
                f"exceeds hardware limits [{piezo_min:.1f}, {piezo_max:.1f}] µm"
            )
        print(f"  ✓ Range within limits")
    except Exception as e:
        print(f"  Warning: Could not validate limits: {e}")

    # Calculate corresponding galvo Y positions
    galvo_y_start = piezo_to_galvo_y(start_um)
    galvo_y_end = piezo_to_galvo_y(end_um)
    galvo_y_center = piezo_to_galvo_y(piezo_center)
    galvo_y_amplitude = abs(galvo_y_end - galvo_y_start)

    print(f"\n  Piezo-Galvo Calibration:")
    print(f"    Slope: {PIEZO_GALVO_SLOPE} µm/°")
    print(f"    Offset: {PIEZO_GALVO_OFFSET} µm")
    print(f"    Piezo {start_um} µm → Galvo Y {galvo_y_start:.4f}°")
    print(f"    Piezo {end_um} µm → Galvo Y {galvo_y_end:.4f}°")
    print(f"    Galvo Y center: {galvo_y_center:.4f}°")
    print(f"    Galvo Y amplitude: {galvo_y_amplitude:.4f}°")

    # CRITICAL: Configure piezo for SPIM state machine control
    print(f"\n  Writing piezo SPIM properties:")

    core.setProperty(piezo_device, "SingleAxisAmplitude(um)", piezo_amplitude)
    print(f"    SingleAxisAmplitude(um) = {piezo_amplitude}")

    core.setProperty(piezo_device, "SingleAxisOffset(um)", piezo_center)
    print(f"    SingleAxisOffset(um) = {piezo_center}")

    core.setProperty(piezo_device, "SPIMNumSlices", num_slices)
    print(f"    SPIMNumSlices = {num_slices}")

    # ARM the piezo (state machine ready to receive trigger)
    core.setProperty(piezo_device, "SPIMState", "Armed")
    print(f"    SPIMState = Armed")

    time.sleep(0.2)  # Allow properties to settle

    # Verify critical properties
    actual_amp = float(core.getProperty(piezo_device, "SingleAxisAmplitude(um)"))
    actual_offset = float(core.getProperty(piezo_device, "SingleAxisOffset(um)"))
    actual_slices = int(core.getProperty(piezo_device, "SPIMNumSlices"))
    actual_state = core.getProperty(piezo_device, "SPIMState")

    print(f"\n  Verification:")
    print(f"    SingleAxisAmplitude(um): {actual_amp} µm")
    print(f"    SingleAxisOffset(um): {actual_offset} µm")
    print(f"    SPIMNumSlices: {actual_slices}")
    print(f"    SPIMState: {actual_state}")

    # Critical validation
    if abs(actual_amp - piezo_amplitude) > 0.001:
        raise Exception(f"Piezo amplitude mismatch: wrote {piezo_amplitude}, read {actual_amp}")

    if abs(actual_offset - piezo_center) > 0.001:
        raise Exception(f"Piezo offset mismatch: wrote {piezo_center}, read {actual_offset}")

    if actual_state != "Armed":
        raise Exception(f"Piezo not armed! State = {actual_state}")

    print(f"  ✓ Piezo configured and armed for SPIM scanning")

    return {
        "start_um": start_um,
        "end_um": end_um,
        "center_um": piezo_center,
        "amplitude_um": piezo_amplitude,
        "step_size_um": step_size,
        "galvo_y_start": galvo_y_start,
        "galvo_y_end": galvo_y_end,
        "galvo_y_center": galvo_y_center,
        "galvo_y_amplitude": galvo_y_amplitude,
    }
```

### 12.2 Required Change to configure_tiger_controller() Call

Update the main acquisition function to use the calculated galvo values:

```python
# Step 5: Configure piezo
piezo_config = configure_piezo_for_scan(PIEZO_DEVICE, piezo_start_um, piezo_end_um, num_slices)

# Step 6: Configure Tiger controller with synchronized galvo Y-axis
configure_tiger_controller(
    GALVO_DEVICE,
    num_slices,
    timing,
    galvo_y_amp=piezo_config["galvo_y_amplitude"],
    galvo_y_offset=piezo_config["galvo_y_center"],
)
```

---

## 13. Key File Locations

### 13.1 Java Plugin Files

```
C:\Users\dispim\Documents\GitHub\micro-manager\plugins\ASIdiSPIM\src\main\java\org\micromanager\asidispim\
├── AcquisitionPanel.java (lines 1105-1326: timing calculation)
├── utils\ControllerUtils.java (lines 349-481: piezo configuration)
│                              (lines 823-850: state machine trigger)
├── utils\SliceTiming.java (complete file: timing data structure)
└── data\Properties.java (lines 97-98: piezo property definitions)
```

### 13.2 Python Reference Implementation

```
C:\Users\dispim\Documents\GitHub\gently\
└── test_volume_acq_with_piezo.py (lines 248-286: INCOMPLETE piezo config)
                                   (REQUIRES FIX: add SA_AMPLITUDE, SA_OFFSET, SPIMState)
```

---

## 14. Conclusion and Recommendations

### 14.1 Summary of Key Findings

1. **Piezo configuration is mode-dependent:** Seven acquisition modes with different piezo/galvo combinations
2. **Calibration is critical:** Piezo-galvo calibration (rate and offset) must be non-zero and accurate
3. **State machine coordination:** Piezo is armed via `SA_AMPLITUDE`/`SA_OFFSET`, then triggered by galvo card
4. **Synchronization mechanism:** `SPIM_NUM_SLICES_PER_PIEZO` controls stepping frequency
5. **Bounds checking:** Essential safety check to prevent piezo damage
6. **Python implementation incomplete:** Missing `SA_AMPLITUDE`/`SA_OFFSET` writes to piezo device

### 14.2 Critical Parameters from ASI diSPIM Plugin Setup

From Screenshot 2025-10-15 110509 (Setup Path B):

```
Imaging center: 0.0 µm
Piezo/Slice Calibration:
  - Slope: 100.306 µm/°
  - Offset: 4.102 µm
  - Step size: 5 µm
```

### 14.3 Implementation Checklist

To properly implement synchronized piezo scanning:

- ✅ Calculate piezo center and amplitude based on desired Z-range
- ✅ Validate piezo range against hardware limits (LOWER_LIMIT, UPPER_LIMIT)
- ✅ Write `SingleAxisAmplitude(um)` property to piezo device
- ✅ Write `SingleAxisOffset(um)` property to piezo device
- ✅ Write `SPIMNumSlices` property to piezo device
- ✅ Set `SPIMState` to "Armed" on piezo device
- ✅ Calculate corresponding galvo Y positions using calibration
- ✅ Configure galvo Y-axis with calculated amplitude and offset
- ✅ Set `SPIM_NUM_SLICES_PER_PIEZO` on galvo device (default = 1)
- ✅ Trigger SPIM state machine by setting galvo `SPIMState` to "Running"

---

**END OF REPORT**

This comprehensive analysis provides complete documentation of the piezo scanning implementation sufficient for reimplementation from scratch. All code references include file paths and line numbers for verification.
