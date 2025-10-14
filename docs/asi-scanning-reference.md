# ASI Plugin Galvo Scanning Reference

This document provides a comprehensive reference for understanding how the MicroManager ASI Tiger plugin implements galvo scanning for light-sheet microscopy (SPIM/diSPIM). This information is essential for implementing volume scanning functionality in Gently.

**Date Created:** 2025-10-12
**Research Source:** ASI Tiger Device Adapter (`micro-manager` repository)

---

## Table of Contents

1. [Overview](#overview)
2. [Source Code Locations](#source-code-locations)
3. [Communication Architecture](#communication-architecture)
4. [Serial Commands Reference](#serial-commands-reference)
5. [SPIM Properties and Configuration](#spim-properties-and-configuration)
6. [Single Axis Mode (SAM)](#single-axis-mode-sam)
7. [Ring Buffer Mode](#ring-buffer-mode)
8. [Typical Volume Scan Workflow](#typical-volume-scan-workflow)
9. [Mapping to Gently/MMCore](#mapping-to-gentlymmcore)

---

## Overview

The ASI Tiger plugin implements two main approaches for galvo-based volume scanning:

### 1. **Hardware SPIM Mode** (Firmware-Timed)
- Uses ASI Tiger controller's built-in SPIM state machine
- Controller coordinates: piezo z-movement → galvo scanning → camera trigger → laser pulse
- All timing handled in firmware via serial commands
- Extremely fast, synchronized 3D acquisition
- Triggered via TTL or software command

### 2. **Single Axis Mode (SAM)** (Software-Coordinated)
- Galvo performs continuous sweep patterns (ramp, triangle, sine)
- Software triggers camera at appropriate times during sweep
- More flexible but requires software coordination
- Useful for custom scanning patterns

---

## Source Code Locations

### ASI Tiger Device Adapter Files

Base directory: `C:\Users\christensenr\Documents\GitHub\micro-manager\mmCoreAndDevices\DeviceAdapters\ASITiger\`

| File | Purpose |
|------|---------|
| `ASIScanner.cpp` | Main scanner/galvo implementation |
| `ASIScanner.h` | Scanner class definition and property declarations |
| `ASITiger.h` | Defines, constants, and property name strings |
| `ASIHub.cpp` | Serial communication hub |

### Key Code Sections

**Scanner Initialization:** `ASIScanner.cpp:89-556`
- Reads unit multipliers, limits, home positions
- Creates SPIM properties (lines 420-556)
- Creates SAM properties (lines 1640-1800)

**SPIM State Control:** `ASIScanner.cpp:3605-3680`
- `OnSPIMState()` - Handles Idle/Armed/Running states
- Uses `SN` serial command

**Position Control:** `ASIScanner.cpp:686-739`
- `SetPosition(double x, double y)` - Move galvo mirrors
- Uses `M` serial command

---

## Communication Architecture

### Serial Communication Flow

```
MMCore Property Set/Get
    ↓
CScanner::OnPropertyX() handler
    ↓
hub_->QueryCommandVerify(command, expected_response)
    ↓
ASI Tiger Controller (Serial RS-232/USB)
    ↓
Returns: ":A" (acknowledge) or ":A <value>"
```

### Key Hub Methods

```cpp
hub_->QueryCommandVerify(command_str, expected_response)  // Send and verify
hub_->QueryCommand(command_str)                          // Send without verify
hub_->ParseAnswerAfterEquals(output_variable)            // Parse ":A X=123"
hub_->GetAnswerCharAtPosition3(output_char)              // Get state char
```

---

## Serial Commands Reference

### Basic Motion Commands

| Command | Description | Example |
|---------|-------------|---------|
| `M X=<val> Y=<val>` | Move galvo to position (in millidegrees) | `M X=500 Y=-200` |
| `W X` | Query X position | Returns `:A X=500.0` |
| `W Y` | Query Y position | Returns `:A Y=-200.0` |
| `UM X?` | Query unit multiplier for X | Returns `:A X=1000` |
| `HM X?` | Query home position (shutter position) | Returns `:A X=0.0` |

### SPIM Configuration Commands (`NR` - Number Repeats)

| Command | Description | Typical Values |
|---------|-------------|----------------|
| `NR X=<n>` | SPIMNumScansPerSlice | 1 (one scan per slice) |
| `NR Y=<n>` | SPIMNumSlices (piezo positions) | 10-200 (z-stack size) |
| `NR F=<n>` | SPIMNumSlicesPerPiezo | 1 (one slice per piezo move) |
| `NR R=<n>` | SPIMNumRepeats | 1-1000 (time points) |
| `NR Z=<n>` | SPIM mode byte (encoded settings) | See mode byte section |

### SPIM Timing Commands (`2SCAN`)

All timing values in **milliseconds**.

| Command | Property Name | Description | Typical Values |
|---------|---------------|-------------|----------------|
| `2SCAN X=<ms>` | SPIMDelayBeforeScan | Delay before scan starts | 0-10 ms |
| `2SCAN Y=<ms>` | SPIMDelayBeforeSide | Delay when switching sides | 0-50 ms |
| `2SCAN Z=<ms>` | SPIMDelayBeforeRepeat | Delay between repeats | 0-1000 ms |
| `2SCAN F=<ms>` | SPIMDelayBeforeCamera | Camera trigger delay | 0.1-1 ms |
| `2SCAN R=<ms>` | SPIMDelayBeforeLaser | Laser trigger delay | 0-1 ms |
| `2SCAN T=<ms>` | SPIMCameraDuration | Camera exposure time | 5-50 ms |
| `2SCAN O=<ms>` | SPIMLaserDuration | Laser on time | 5-50 ms |
| `2SCAN D=<ms>` | SPIMScanDuration (fw 3.14+) | Galvo scan duration | 5-20 ms |

### SPIM State Control (`SN`)

| Command | Description | Controller Response |
|---------|-------------|---------------------|
| `SN X?` | Query SPIM state | `:A <state_char>` |
| `SN X=80` | Stop/Idle (ASCII 'P') | `:A` |
| `SN X=97` | Arm for TTL trigger (ASCII 'a') | `:A` |
| `SN` | Start SPIM acquisition | `:A` |

**State Characters:**
- `'I'` (73) - Idle
- `'A'` (65) - Armed (waiting for trigger)
- `'S'` (83) - Running
- `'P'` (80) - Stopped

### Single Axis Mode Commands

| Command | Description | Values |
|---------|-------------|--------|
| `SAM X=<mode>` | Set single axis mode | 0=disabled, 1=enabled, 2=TTL trigger, 3=synced |
| `SAP X=<pattern>` | Set scan pattern | 0=ramp, 1=triangle, 2=square, 3=sine |
| `SAA X=<val>` | Set amplitude (millidegrees) | e.g., 500 = 0.5° |
| `SAO X=<val>` | Set offset (millidegrees) | e.g., 0 = center |
| `SAF X=<ms>` | Set period (milliseconds) | e.g., 10 = 100 Hz |

### Limit Setting Commands

| Command | Description |
|---------|-------------|
| `SL X=<val>` | Set lower limit for X axis |
| `SU X=<val>` | Set upper limit for X axis |
| `SL Y=<val>` | Set lower limit for Y axis |
| `SU Y=<val>` | Set upper limit for Y axis |

### Ring Buffer Commands

| Command | Description |
|---------|-------------|
| `RM X=0` | Reset ring buffer |
| `LD X=<val> Y=<val>` | Load point into ring buffer |

---

## SPIM Properties and Configuration

### SPIM Mode Byte (`NR Z`)

The mode byte encodes multiple settings using bit positions:

```
Bit 7: Smooth slice enable (1 = constant galvo scan)
Bit 6: (reserved)
Bit 5: Alternate directions (1 = bidirectional scanning)
Bit 4: Interleave sides (1 = alternate A/B per slice)
Bit 3-2: Laser output mode
Bit 1-0: Side configuration
```

**Side Configuration (bits 0-1):**
- `00` (0) - Side B only
- `01` (1) - Side A only (default)
- `10` (2) - Both sides, A first
- `11` (3) - Both sides, B first

**Example Mode Bytes:**
- `1` = Side A only, standard mode
- `2` = Both sides, A first
- `17` (0x11) = Side A, interleave enabled
- `33` (0x21) = Side A, alternate directions

### SPIM Property Mappings

| MMCore Property Name | Serial Command | Description |
|---------------------|----------------|-------------|
| `SPIMNumSlices` | `NR Y=<n>` | Number of piezo positions |
| `SPIMNumSlicesPerPiezo` | `NR F=<n>` | Slices per piezo position |
| `SPIMNumScansPerSlice` | `NR X=<n>` | Scans per slice |
| `SPIMNumSides` | `NR Z` (bits 0-1) | 1 or 2 sides |
| `SPIMFirstSide` | `NR Z` (bits 0-1) | "A" or "B" |
| `SPIMState` | `SN X=<state>` | Idle/Armed/Running |
| `SPIMDelayBeforeScan(ms)` | `2SCAN X=<ms>` | Pre-scan delay |
| `SPIMDelayBeforeCamera(ms)` | `2SCAN F=<ms>` | Camera trigger delay |
| `SPIMDelayBeforeLaser(ms)` | `2SCAN R=<ms>` | Laser trigger delay |
| `SPIMCameraDuration(ms)` | `2SCAN T=<ms>` | Camera exposure |
| `SPIMLaserDuration(ms)` | `2SCAN O=<ms>` | Laser on time |
| `SPIMScanDuration(ms)` | `2SCAN D=<ms>` | Scan duration (fw 3.14+) |

---

## Single Axis Mode (SAM)

Single Axis Mode allows continuous periodic scanning of the galvo mirror, useful for:
- Continuous light-sheet sweeping during volume acquisition
- Synchronized scanning with external triggers
- Custom scanning patterns

### SAM Properties

| Property | Values | Description |
|----------|--------|-------------|
| `SingleAxisXMode` | 0-3 | 0=disabled, 1=enabled, 2=armed for TTL, 3=synced |
| `SingleAxisXPattern` | 0-3 | 0=ramp, 1=triangle, 2=square, 3=sine |
| `SingleAxisXAmplitude(deg)` | 0-10° | Peak-to-peak amplitude |
| `SingleAxisXOffset(deg)` | -10 to +10° | Center position |
| `SingleAxisXPeriod(ms)` | 1-10000 | Scan period |

### SAM Code Example

From `ASIScanner.cpp:1720-1765`:

```cpp
// Get current mode
command << "SAM " << axisLetterX_ << "?";
hub_->QueryCommandVerify(command.str(), ":A");
hub_->ParseAnswerAfterEquals(tmp);

// Set mode
command << "SAM " << axisLetterX_ << "=" << mode_value;
hub_->QueryCommandVerify(command.str(), ":A");
```

### SAM Pattern Descriptions

**Ramp (0):** Sawtooth wave, fast retrace
```
  /|  /|  /|
 / | / | / |
```

**Triangle (1):** Symmetric bidirectional scan
```
 /\  /\  /\
/  \/  \/  \
```

**Square (2):** Step function
```
--  --  --
  |   |   |
  --  --  --
```

**Sine (3):** Smooth sinusoidal (firmware 3.14+)
```
  ---       ---
-     -   -     -
```

---

## Ring Buffer Mode

Ring buffer mode allows pre-programmed scanning patterns for phototargeting and polygon scanning.

### Ring Buffer Properties

| Property | Description |
|----------|-------------|
| `RingBufferMode` | "1 - One Point", "2 - Play Once", "3 - Repeat" |
| `RingBufferDelayBetweenPoints(ms)` | Time at each point |
| `RingBufferTrigger` | Manual trigger control |

### Ring Buffer Workflow

```cpp
// 1. Reset buffer
command << addressChar_ << "RM X=0";
hub_->QueryCommandVerify(command.str(), ":A");

// 2. Load points
for (each polygon vertex) {
    command << "LD " << axisLetterX_ << "=" << x*unitMultX_
            << " " << axisLetterY_ << "=" << y*unitMultY_;
    hub_->QueryCommandVerify(command.str(), ":A");
}

// 3. Execute
// Via MMCore: AddPolygonVertex(), LoadPolygons(), RunPolygons()
```

---

## Typical Volume Scan Workflow

### Hardware SPIM Volume Scan

```
1. Configure SPIM parameters
   - Set number of slices (NR Y=100)
   - Set scan duration (2SCAN D=10)
   - Set camera timing (2SCAN F=0.2, 2SCAN T=9.5)
   - Set sides (NR Z=1 for side A only)

2. Arm SPIM state machine
   - Send SN X=97 (arm for TTL)
   - Controller reports state 'A' (armed)

3. Trigger acquisition
   - External TTL pulse triggers start, OR
   - Software command SN (start immediately)

4. Acquisition runs
   - Controller automatically:
     * Moves piezo to each z position
     * Scans galvo for light sheet
     * Triggers camera
     * Triggers laser
     * Repeats for all slices

5. Completion
   - Controller returns to idle state
   - Query status with SN X?
```

### Software-Coordinated Scan (SAM)

```
1. Configure SAM pattern
   - Set pattern (SAP X=1 for triangle)
   - Set amplitude (SAA X=500 for 0.5°)
   - Set period (SAF X=10 for 10ms)

2. Enable SAM
   - SAM X=1 (start continuous scanning)

3. Software loop
   - For each z position:
     * Move piezo
     * Calculate galvo sync position
     * Trigger camera when galvo at correct phase
     * Acquire image

4. Disable SAM
   - SAM X=0 (stop scanning)
```

---

## Mapping to Gently/MMCore

### MMCore Property Interface

In Gently, access these settings via MMCore properties instead of direct serial commands:

```python
import pymmcore

core = pymmcore.CMMCore()

# SPIM configuration via properties
core.setProperty("Scanner:AB:33", "SPIMNumSlices", 100)
core.setProperty("Scanner:AB:33", "SPIMNumScansPerSlice", 1)
core.setProperty("Scanner:AB:33", "SPIMScanDuration(ms)", 10.0)
core.setProperty("Scanner:AB:33", "SPIMCameraDuration(ms)", 9.5)
core.setProperty("Scanner:AB:33", "SPIMDelayBeforeCamera(ms)", 0.2)

# Arm SPIM
core.setProperty("Scanner:AB:33", "SPIMState", "Armed")

# Query state
state = core.getProperty("Scanner:AB:33", "SPIMState")
print(f"SPIM State: {state}")  # "Idle", "Armed", or "Running"

# Start SPIM
core.setProperty("Scanner:AB:33", "SPIMState", "Running")
```

### Direct Serial Command Access (if needed)

```python
# For commands not exposed as properties
core.setProperty("TigerCommHub", "SerialCommand", "NR Y=100")
response = core.getProperty("TigerCommHub", "SerialResponse")
```

### Gently Device Class Integration

For Gently's `DiSPIMScanner` class, add property access methods:

```python
class DiSPIMScanner:
    """Enhanced with SPIM properties"""

    def set_spim_num_slices(self, num_slices: int):
        """Set number of z-slices for SPIM"""
        self.core.setProperty(self.device_name, "SPIMNumSlices", num_slices)

    def set_spim_timing(self, scan_ms: float, camera_ms: float, delay_ms: float):
        """Configure SPIM timing"""
        self.core.setProperty(self.device_name, "SPIMScanDuration(ms)", scan_ms)
        self.core.setProperty(self.device_name, "SPIMCameraDuration(ms)", camera_ms)
        self.core.setProperty(self.device_name, "SPIMDelayBeforeCamera(ms)", delay_ms)

    def arm_spim(self):
        """Arm SPIM for TTL trigger"""
        self.core.setProperty(self.device_name, "SPIMState", "Armed")

    def start_spim(self):
        """Start SPIM acquisition"""
        self.core.setProperty(self.device_name, "SPIMState", "Running")

    def stop_spim(self):
        """Stop SPIM acquisition"""
        self.core.setProperty(self.device_name, "SPIMState", "Idle")

    def get_spim_state(self) -> str:
        """Query current SPIM state"""
        return self.core.getProperty(self.device_name, "SPIMState")
```

---

## Additional Resources

### ASI Tiger Serial Command Documentation
- [ASI Tiger Controller Manual](http://asiimaging.com/docs/products/tiger)
- Firmware version checking: Use `BU X` command to query build info

### MicroManager Source Code
- Main repository: `https://github.com/micro-manager/micro-manager`
- ASI adapter location: `mmCoreAndDevices/DeviceAdapters/ASITiger/`

### Gently Implementation Files
- Scanner device: `gently/devices.py:556-645` (`DiSPIMScanner` class)
- Coordinate transforms: `gently/coordinates.py:78-126` (piezo-galvo calibration)
- Focus plans: `gently/plans.py` (template for volume scan plans)

---

## Notes

1. **Unit Conversions**:
   - Galvo positions are in degrees
   - Serial commands use millidegrees (multiply by `unitMultX_`, typically 1000)
   - MMCore properties handle conversion automatically

2. **Timing Constraints**:
   - Camera exposure must be ≤ scan duration
   - Allow settling time before camera trigger (typically 0.1-0.5 ms)
   - Total slice time = scan + camera + delays

3. **Firmware Versions**:
   - SPIM features require firmware ≥ 2.8
   - Some features require 2.84+ (delay properties)
   - Sine pattern requires 3.14+
   - Check firmware with `BU X` command

4. **Synchronization**:
   - Hardware SPIM mode uses controller's state machine for tight synchronization
   - Software mode requires manual coordination but offers more flexibility
   - TTL triggering provides sub-millisecond timing precision

---

**Document Version:** 1.0
**Last Updated:** 2025-10-12
**Author:** Research from ASI Tiger plugin source code analysis
