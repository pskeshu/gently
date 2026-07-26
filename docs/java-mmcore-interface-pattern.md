# Java ASIdiSPIM Plugin to MMCore Interface Pattern

This document explains how the Java ASIdiSPIM plugin interfaces with the C++ ASI Tiger device adapter through MMCore, and provides a template for implementing the same pattern in Python for Gently.

**Date Created:** 2025-10-12

---

## Table of Contents

1. [Overview](#overview)
2. [Java Plugin Architecture](#java-plugin-architecture)
3. [The Properties Wrapper Class](#the-properties-wrapper-class)
4. [The Devices Management Class](#the-devices-management-class)
5. [Usage in Volume Acquisition](#usage-in-volume-acquisition)
6. [Python Implementation for Gently](#python-implementation-for-gently)
7. [Complete Example](#complete-example)

---

## Overview

### The Key Insight

The Java ASIdiSPIM plugin **never directly accesses serial commands**. Instead, it:

1. Uses `CMMCore.setProperty()` and `CMMCore.getProperty()` to interact with devices
2. Wraps MMCore property access in a type-safe `Properties` class
3. Uses enums for device keys and property keys to avoid string typos
4. Optimizes by only setting properties when values actually change

**This is exactly what Gently needs to do.**

### Architecture Layers

```
┌────────────────────────────────────────────────────────┐
│       Java Plugin (ASIdiSPIM.java)                     │
│       - UI, acquisition orchestration                   │
└───────────────────────┬────────────────────────────────┘
                        │
┌───────────────────────▼────────────────────────────────┐
│       Properties.java + Devices.java                   │
│       - Type-safe property wrapper                      │
│       - Enum-based device/property keys                 │
│       - Change detection optimization                   │
└───────────────────────┬────────────────────────────────┘
                        │
┌───────────────────────▼────────────────────────────────┐
│       CMMCore (Micro-Manager Core)                     │
│       core.setProperty(device, property, value)        │
│       core.getProperty(device, property)               │
└───────────────────────┬────────────────────────────────┘
                        │
┌───────────────────────▼────────────────────────────────┐
│       C++ Device Adapter (ASITiger)                    │
│       - Property handlers convert to serial commands   │
│       - Hub manages serial communication               │
└───────────────────────┬────────────────────────────────┘
                        │
┌───────────────────────▼────────────────────────────────┐
│       ASI Tiger Controller (Hardware)                  │
└────────────────────────────────────────────────────────┘
```

---

## Java Plugin Architecture

### File Structure

**Core files:**
- `Properties.java` - Wrapper for MMCore property access
- `Devices.java` - Device management and enum keys
- `ControllerUtils.java` - Controller configuration logic
- `AcquisitionPanel.java` - Volume acquisition UI and orchestration

**Key classes:**
- `Properties.Keys` - Enum of all property names
- `Properties.Values` - Enum of common property values
- `Devices.Keys` - Enum of device roles (GALVOA, PIEZOA, etc.)
- `Devices.DeviceData` - Associates device role with MMCore device name

---

## The Properties Wrapper Class

### Purpose

The `Properties` class wraps `CMMCore.setProperty()` and `CMMCore.getProperty()` to provide:

1. **Type safety** - Enums instead of strings
2. **Optimization** - Only set if value changed
3. **Error handling** - Optional error suppression
4. **Type conversion** - Automatic string/int/float conversion
5. **Consistency** - Single interface for all property access

### Property Keys Enum

**File:** `Properties.java` lines 71-290

```java
public static enum Keys {
    // SPIM Properties
    SPIM_NUM_SIDES("SPIMNumSides"),
    SPIM_NUM_SLICES("SPIMNumSlices"),
    SPIM_NUM_REPEATS("SPIMNumRepeats"),
    SPIM_DELAY_REPEATS("SPIMDelayBeforeRepeat(ms)"),
    SPIM_NUM_SCANSPERSLICE("SPIMNumScansPerSlice"),
    SPIM_INTERLEAVE_SIDES("SPIMInterleaveSidesEnable"),
    SPIM_PIEZO_HOME_DISABLE("SPIMPiezoHomeDisable"),
    SPIM_ALTERTATE_DIRECTIONS("SPIMAlternateDirectionsEnable"),
    SPIM_NUM_SLICES_PER_PIEZO("SPIMNumSlicesPerPiezo"),
    SPIM_DELAY_SIDE("SPIMDelayBeforeSide(ms)"),
    SPIM_DELAY_SCAN("SPIMDelayBeforeScan(ms)"),
    SPIM_DELAY_LASER("SPIMDelayBeforeLaser(ms)"),
    SPIM_DURATION_SCAN("SPIMScanDuration(ms)"),
    SPIM_DURATION_LASER("SPIMLaserDuration(ms)"),
    SPIM_DELAY_CAMERA("SPIMDelayBeforeCamera(ms)"),
    SPIM_DURATION_CAMERA("SPIMCameraDuration(ms)"),
    SPIM_FIRSTSIDE("SPIMFirstSide"),
    SPIM_STATE("SPIMState"),

    // Single Axis Mode Properties
    SA_AMPLITUDE("SingleAxisAmplitude(um)", false),
    SA_OFFSET("SingleAxisOffset(um)", false),
    SA_AMPLITUDE_X_DEG("SingleAxisXAmplitude(deg)", false),
    SA_OFFSET_X_DEG("SingleAxisXOffset(deg)", false),
    SA_OFFSET_X("SingleAxisXOffset(um)", false),
    SA_MODE_X("SingleAxisXMode", false),
    SA_PATTERN_X("SingleAxisXPattern", false),
    SA_PERIOD_X("SingleAxisXPeriod(ms)", false),
    SA_AMPLITUDE_Y_DEG("SingleAxisYAmplitude(deg)", false),
    SA_OFFSET_Y_DEG("SingleAxisYOffset(deg)", false),
    SA_OFFSET_Y("SingleAxisYOffset(um)", false),

    // Scanner Properties
    SCANNER_FILTER_X("FilterFreqX(kHz)"),
    SCANNER_FILTER_Y("FilterFreqY(kHz)"),
    MAX_DEFLECTION_X("MaxDeflectionX(deg)"),
    MIN_DEFLECTION_X("MinDeflectionX(deg)"),
    BEAM_ENABLED("BeamEnabled", false),

    // ... many more
    ;

    private final String text;
    private final boolean forceSet;

    Keys(String text) {
        this.text = text;
        this.forceSet = true;  // Default: always set
    }

    Keys(String text, boolean forceSet) {
        this.text = text;
        this.forceSet = forceSet;  // Some properties only set if changed
    }

    @Override
    public String toString() {
        return text;  // Returns the actual MMCore property name
    }

    public boolean doForceSet() {
        return forceSet;
    }
}
```

**Key Features:**
- Each enum value maps to actual MMCore property name
- `forceSet` flag controls optimization (some properties always set, others only if changed)
- `toString()` returns the string to pass to MMCore

### Property Values Enum

**File:** `Properties.java` lines 293-369

```java
public static enum Values {
    YES("Yes"),
    NO("No"),
    SPIM_ARMED("Armed"),
    SPIM_RUNNING("Running"),
    SPIM_IDLE("Idle"),
    SAM_DISABLED("0 - Disabled"),
    SAM_ENABLED("1 - Enabled"),
    SAM_RAMP("0 - Ramp"),
    SAM_TRIANGLE("1 - Triangle"),
    // ... many more
    ;

    private final String text;

    Values(String text) {
        this.text = text;
    }

    @Override
    public String toString() {
        return text;
    }
}
```

### Core Methods

#### Setting Properties

**File:** `Properties.java` lines 417-672

**String values:**
```java
public void setPropValue(Devices.Keys device, Properties.Keys name, String strVal,
      boolean ignoreError) {
    if (device == Devices.Keys.PLUGIN) {
        // Special case: plugin "properties" stored in prefs
        prefs_.putString(PLUGIN_PREF_NODE, name, strVal);
    } else {
        String mmDevice = null;
        try {
            mmDevice = devices_.getMMDeviceException(device);

            // Optimization: only set if value changed (or forceSet is true)
            if (name.doForceSet()
                  || !core_.getProperty(mmDevice, name.toString()).equals(strVal)) {
                core_.setProperty(mmDevice, name.toString(), strVal);
            }
        } catch (Exception ex) {
            if (ignoreError) {
                ReportingUtils.logMessage("Device " + mmDevice +
                      " does not have property: " + name.toString());
            } else {
                MyDialogUtils.showError(ex, "Error setting string property " +
                      name.toString() + " to " + strVal + " in device " + mmDevice);
            }
        }
    }
}
```

**Integer values:**
```java
public void setPropValue(Devices.Keys device, Properties.Keys name, int intVal,
      boolean ignoreError) {
    // ... similar to string version
    if (name.doForceSet()
          || intVal != NumberUtils.coreStringToInt(
                core_.getProperty(mmDevice, name.toString()))) {
        core_.setProperty(mmDevice, name.toString(), intVal);
    }
}
```

**Float values:**
```java
public void setPropValue(Devices.Keys device, Properties.Keys name, float floatVal,
      boolean ignoreError) {
    // ... similar to string version
    if (name.forceSet
          || !MyNumberUtils.floatsEqual(floatVal, (float)NumberUtils.coreStringToDouble(
                core_.getProperty(mmDevice, name.toString())))) {
        core_.setProperty(mmDevice, name.toString(), floatVal);
    }
}
```

**Enum values:**
```java
public void setPropValue(Devices.Keys device, Properties.Keys name, Properties.Values val,
      boolean ignoreError) {
    setPropValue(device, name, val.toString(), ignoreError);
}
```

**Overloaded versions:**
- With/without `ignoreError` parameter (default false)
- Single device or array of devices
- All type combinations

#### Getting Properties

**File:** `Properties.java` lines 674-768

```java
private String getPropValue(Devices.Keys device, Properties.Keys name) {
    String val;
    if (device == Devices.Keys.PLUGIN) {
        val = prefs_.getString(PLUGIN_PREF_NODE, name, "");
    } else {
        String mmDevice = devices_.getMMDevice(device);
        val = "";  // Empty string to avoid null pointer exceptions
        if (mmDevice != null) {
            try {
                val = core_.getProperty(mmDevice, name.toString());
            } catch (Exception ex) {
                // Silently return empty string on error
            }
        }
    }
    return val;
}

public String getPropValueString(Devices.Keys device, Properties.Keys name) {
    return getPropValue(device, name);
}

public int getPropValueInteger(Devices.Keys device, Properties.Keys name) {
    int val = 0;
    if (device == Devices.Keys.PLUGIN) {
        val = prefs_.getInt(PLUGIN_PREF_NODE, name, 0);
    } else {
        String strVal = getPropValue(device, name);
        if (!strVal.equals("")) {
            val = NumberUtils.coreStringToInt(strVal);
        }
    }
    return val;
}

public float getPropValueFloat(Devices.Keys device, Properties.Keys name) {
    float val = 0;
    if (device == Devices.Keys.PLUGIN) {
        val = prefs_.getFloat(PLUGIN_PREF_NODE, name, 0);
    } else {
        String strVal = getPropValue(device, name);
        if (!strVal.equals("")) {
            val = (float)NumberUtils.coreStringToDouble(strVal);
        }
    }
    return val;
}
```

**Key features:**
- Read errors are silently ignored (return 0 or empty string)
- Type conversion handled automatically
- Plugin "properties" stored in preferences

---

## The Devices Management Class

### Purpose

The `Devices` class manages the mapping between:
- **Device roles** (GALVOA, PIEZOA, etc.) - what the device does
- **MMCore device names** ("Scanner:AB:33", "PiezoStage:P:34") - actual device name

### Device Keys Enum

**File:** `Devices.java` lines 93-105

```java
public static enum Keys {
    NONE, CORE, PLUGIN,
    CAMERAA, CAMERAB, MULTICAMERA, CAMERALOWER, CAMERAPREVIOUS,
    PIEZOA, PIEZOB, GALVOA, GALVOB, XYSTAGE, LOWERZDRIVE, UPPERZDRIVE,
    PLOGIC,
    TIGERCOMM,
    UPPERHDRIVE,
    SHUTTERLOWER,
    // ...
}
```

**Key features:**
- One enum for each device role in the system
- Side-specific devices (A/B for dual-view diSPIM)
- Special keys: NONE, CORE, PLUGIN

### Device Data Structure

**File:** `Devices.java` lines 154-194

```java
public static class DeviceData {
    Keys key;                // Device role
    String mmDevice;         // MMCore device name
    String displayName;      // GUI display name
    Sides side;              // A, B, or NONE
    String axisLetter;       // ASI axis letter(s)
    DeviceType type;         // MMCore device type
    Libraries deviceLibrary; // Device adapter library
    boolean saveInPref;      // Save to preferences?
}
```

### Core Methods

**File:** `Devices.java` lines 337-431

```java
// Check if device is assigned
public boolean isValidMMDevice(Devices.Keys key) {
    return (key != Devices.Keys.NONE && getMMDevice(key) != null);
}

// Get MMCore device name from role
public String getMMDevice(Devices.Keys key) {
    String mmDevice = deviceInfo_.get(key).mmDevice;
    if (mmDevice == null || mmDevice.equals("")) {
        return null;
    }
    return mmDevice;
}

// Get MMCore device name (throws exception if not set)
public String getMMDeviceException(Devices.Keys key) throws Exception {
    String mmDevice = getMMDevice(key);
    if (mmDevice == null || mmDevice.equals("")) {
        throw (new Exception("No device set for " + key.toString()));
    }
    return mmDevice;
}

// Check if device has a property
public boolean hasProperty(Devices.Keys devKey, Properties.Keys propKey) {
    try {
        return core_.hasProperty(getMMDevice(devKey), propKey.toString());
    } catch (Exception e) {
        return false;
    }
}

// Check if device is from ASI Tiger
public boolean isTigerDevice(Devices.Keys devKey) {
    return (deviceInfo_.get(devKey).deviceLibrary == Devices.Libraries.ASITIGER);
}
```

---

## Usage in Volume Acquisition

### Example: Configuring SPIM for Acquisition

**File:** `ControllerUtils.java` lines 102-536

This method shows the **exact pattern** for configuring the ASI controller for volume acquisition:

```java
public static int prepareControllerForAquisition(
      final Devices devices,
      final Properties props,
      final Prefs prefs,
      boolean skipScannerWarnings) {

    // Get device keys for this side
    Devices.Keys galvoDevice = Devices.Keys.GALVOA;  // or GALVOB
    Devices.Keys piezoDevice = Devices.Keys.PIEZOA;  // or PIEZOB

    // Calculate acquisition parameters
    int numSlices = 100;
    int numVolumesPerTrigger = 1;
    float sliceAmplitude = 5.0f;  // degrees
    float piezoAmplitude = 50.0f; // microns
    float scanDuration = 10.0f;   // ms

    // Configure galvo SPIM properties
    props.setPropValue(galvoDevice, Properties.Keys.SPIM_NUM_SLICES_PER_PIEZO,
                       numSlicesPerPiezo, skipScannerWarnings);
    props.setPropValue(galvoDevice, Properties.Keys.SPIM_NUM_REPEATS,
                       numVolumesPerTrigger, skipScannerWarnings);
    props.setPropValue(galvoDevice, Properties.Keys.SA_AMPLITUDE_Y_DEG,
                       sliceAmplitude, skipScannerWarnings);
    props.setPropValue(galvoDevice, Properties.Keys.SPIM_DELAY_SCAN,
                       delayBeforeScan, skipScannerWarnings);
    props.setPropValue(galvoDevice, Properties.Keys.SPIM_DURATION_SCAN,
                       scanDuration, skipScannerWarnings);
    props.setPropValue(galvoDevice, Properties.Keys.SPIM_DELAY_CAMERA,
                       delayBeforeCamera, skipScannerWarnings);
    props.setPropValue(galvoDevice, Properties.Keys.SPIM_DURATION_CAMERA,
                       cameraDuration, skipScannerWarnings);
    props.setPropValue(galvoDevice, Properties.Keys.SPIM_DELAY_LASER,
                       delayBeforeLaser, skipScannerWarnings);
    props.setPropValue(galvoDevice, Properties.Keys.SPIM_DURATION_LASER,
                       laserDuration, skipScannerWarnings);

    // Configure piezo SPIM properties
    props.setPropValue(piezoDevice, Properties.Keys.SA_AMPLITUDE,
                       piezoAmplitude);
    props.setPropValue(piezoDevice, Properties.Keys.SPIM_NUM_SLICES_PER_PIEZO,
                       numSlicesPerPiezo);

    // Disable beam during setup
    props.setPropValue(galvoDevice, Properties.Keys.BEAM_ENABLED,
                       Properties.Values.NO, true);

    // Arm SPIM state machine
    props.setPropValue(galvoDevice, Properties.Keys.SPIM_STATE,
                       Properties.Values.SPIM_ARMED, skipScannerWarnings);

    return 0;  // Success
}
```

**Key patterns:**
1. **No serial commands** - Everything through properties
2. **Type-safe enums** - Device keys and property keys
3. **Error control** - `skipScannerWarnings` parameter
4. **Logical grouping** - Galvo properties, then piezo properties
5. **State management** - Disable beam, configure, then arm

### Example: Reading Property Values

```java
// Get current number of slices
int numSlices = props.getPropValueInteger(Devices.Keys.GALVOA,
                                          Properties.Keys.SPIM_NUM_SLICES);

// Get current amplitude
float amplitude = props.getPropValueFloat(Devices.Keys.GALVOA,
                                          Properties.Keys.SA_AMPLITUDE_Y_DEG);

// Get current state
String state = props.getPropValueString(Devices.Keys.GALVOA,
                                        Properties.Keys.SPIM_STATE);
```

---

## Python Implementation for Gently

### Design Goals

Replicate the Java pattern in Python:

1. ✅ Enum-based device and property keys
2. ✅ Type-safe property wrapper
3. ✅ Optimization to only set when value changes
4. ✅ Optional error suppression
5. ✅ Automatic type conversion
6. ✅ Single interface for all property access

### Proposed File Structure

```
gently/
├── devices.py              # Existing device classes
├── mmcore_wrapper.py       # NEW: Property wrapper classes
│   ├── PropertyKeys        # Enum of property names
│   ├── PropertyValues      # Enum of common values
│   ├── DeviceKeys          # Enum of device roles
│   ├── GentlyProperties    # Property wrapper class
│   └── GentlyDevices       # Device management class
└── plans.py                # Bluesky plans using wrapper
```

### Implementation

**File:** `gently/mmcore_wrapper.py` (new file)

```python
"""
MMCore property access wrapper for Gently.

This module provides type-safe, optimized access to MMCore device properties,
following the pattern used by the Java ASIdiSPIM plugin.

Design principles:
- Use enums for type safety (avoid string typos)
- Optimize by only setting properties when values change
- Provide consistent error handling
- Support automatic type conversion
- Single interface for all property access

Author: [Your Name]
Date: 2025-10-12
"""

from enum import Enum, auto
from typing import Union, Optional, List
import logging

logger = logging.getLogger(__name__)


# ============================================================================
# Property Keys Enum
# ============================================================================


class PropertyKeys(Enum):
    """
    Enum of all device adapter property names used in Gently.

    The enum value (all caps) is used in Python code. The string value
    is the actual property name used by the MMCore device adapter.

    Some properties have force_set=False, meaning they will only be set
    if the value actually changed (optimization for slow serial operations).
    """

    # SPIM Mode Properties
    SPIM_NUM_SIDES = ("SPIMNumSides", True)
    SPIM_NUM_SLICES = ("SPIMNumSlices", True)
    SPIM_NUM_REPEATS = ("SPIMNumRepeats", True)
    SPIM_DELAY_REPEATS = ("SPIMDelayBeforeRepeat(ms)", True)
    SPIM_NUM_SCANS_PER_SLICE = ("SPIMNumScansPerSlice", True)
    SPIM_INTERLEAVE_SIDES = ("SPIMInterleaveSidesEnable", True)
    SPIM_PIEZO_HOME_DISABLE = ("SPIMPiezoHomeDisable", True)
    SPIM_ALTERNATE_DIRECTIONS = ("SPIMAlternateDirectionsEnable", True)
    SPIM_NUM_SLICES_PER_PIEZO = ("SPIMNumSlicesPerPiezo", True)
    SPIM_DELAY_SIDE = ("SPIMDelayBeforeSide(ms)", True)
    SPIM_DELAY_SCAN = ("SPIMDelayBeforeScan(ms)", True)
    SPIM_DELAY_LASER = ("SPIMDelayBeforeLaser(ms)", True)
    SPIM_DURATION_SCAN = ("SPIMScanDuration(ms)", True)
    SPIM_DURATION_LASER = ("SPIMLaserDuration(ms)", True)
    SPIM_DELAY_CAMERA = ("SPIMDelayBeforeCamera(ms)", True)
    SPIM_DURATION_CAMERA = ("SPIMCameraDuration(ms)", True)
    SPIM_FIRST_SIDE = ("SPIMFirstSide", True)
    SPIM_STATE = ("SPIMState", True)

    # Single Axis Mode Properties
    SA_AMPLITUDE = ("SingleAxisAmplitude(um)", False)
    SA_OFFSET = ("SingleAxisOffset(um)", False)
    SA_AMPLITUDE_X_DEG = ("SingleAxisXAmplitude(deg)", False)
    SA_OFFSET_X_DEG = ("SingleAxisXOffset(deg)", False)
    SA_OFFSET_X = ("SingleAxisXOffset(um)", False)
    SA_MODE_X = ("SingleAxisXMode", False)
    SA_PATTERN_X = ("SingleAxisXPattern", False)
    SA_PERIOD_X = ("SingleAxisXPeriod(ms)", False)
    SA_AMPLITUDE_Y_DEG = ("SingleAxisYAmplitude(deg)", False)
    SA_OFFSET_Y_DEG = ("SingleAxisYOffset(deg)", False)
    SA_OFFSET_Y = ("SingleAxisYOffset(um)", False)

    # Scanner Properties
    SCANNER_FILTER_X = ("FilterFreqX(kHz)", True)
    SCANNER_FILTER_Y = ("FilterFreqY(kHz)", True)
    MAX_DEFLECTION_X = ("MaxDeflectionX(deg)", True)
    MIN_DEFLECTION_X = ("MinDeflectionX(deg)", True)
    BEAM_ENABLED = ("BeamEnabled", False)

    # Stage/Piezo Properties
    UPPER_LIMIT = ("UpperLim(mm)", True)
    LOWER_LIMIT = ("LowerLim(mm)", True)
    JOYSTICK_ENABLED = ("JoystickEnabled", True)
    PIEZO_MODE = ("PiezoMode", True)

    # Hub Properties
    SERIAL_COMMAND = ("SerialCommand", True)
    SERIAL_RESPONSE = ("SerialResponse", True)
    SAVE_CARD_SETTINGS = ("SaveCardSettings", True)

    def __init__(self, property_name: str, force_set: bool):
        self._property_name = property_name
        self._force_set = force_set

    @property
    def property_name(self) -> str:
        """The actual MMCore property name string."""
        return self._property_name

    @property
    def force_set(self) -> bool:
        """Whether to always set this property (True) or only when changed (False)."""
        return self._force_set

    def __str__(self) -> str:
        """Returns the MMCore property name."""
        return self._property_name


# ============================================================================
# Property Values Enum
# ============================================================================


class PropertyValues(Enum):
    """
    Enum of common property values.

    Provides type safety for frequently-used property values.
    """

    YES = "Yes"
    NO = "No"

    # SPIM States
    SPIM_ARMED = "Armed"
    SPIM_RUNNING = "Running"
    SPIM_IDLE = "Idle"

    # Single Axis Mode
    SAM_DISABLED = "0 - Disabled"
    SAM_ENABLED = "1 - Enabled"
    SAM_RAMP = "0 - Ramp"
    SAM_TRIANGLE = "1 - Triangle"

    # Joystick Inputs
    JS_NONE = "0 - none"
    JS_X = "2 - joystick X"
    JS_Y = "3 - joystick Y"
    JS_RIGHT_WHEEL = "22 - right wheel"
    JS_LEFT_WHEEL = "23 - left wheel"

    def __str__(self) -> str:
        """Returns the property value string."""
        return self.value


# ============================================================================
# Device Keys Enum
# ============================================================================


class DeviceKeys(Enum):
    """
    Enum of device roles in the Gently system.

    Each key represents a device role (e.g., GALVO_A, PIEZO_A) that
    maps to an actual MMCore device name (e.g., "Scanner:AB:33").
    """

    NONE = auto()
    CORE = auto()

    # Cameras
    CAMERA_A = auto()
    CAMERA_B = auto()

    # Stages and Piezos
    PIEZO_A = auto()
    PIEZO_B = auto()
    GALVO_A = auto()
    GALVO_B = auto()
    XY_STAGE = auto()
    Z_STAGE = auto()

    # Controllers
    TIGER_COMM = auto()
    PLOGIC = auto()


# ============================================================================
# Gently Devices Class
# ============================================================================


class GentlyDevices:
    """
    Manages the mapping between device roles (DeviceKeys) and MMCore device names.

    This class maintains the association between logical device roles
    (e.g., GALVO_A) and actual MMCore device names (e.g., "Scanner:AB:33").

    Example:
        devices = GentlyDevices(core)
        devices.set_device(DeviceKeys.GALVO_A, "Scanner:AB:33")
        devices.set_device(DeviceKeys.PIEZO_A, "PiezoStage:P:34")

        # Get device name
        galvo_name = devices.get_device(DeviceKeys.GALVO_A)
    """

    def __init__(self, core):
        """
        Initialize device manager.

        Args:
            core: MMCore instance (pymmcore.CMMCore or RPyC proxy)
        """
        self.core = core
        self._device_map = {}  # DeviceKeys -> MMCore device name

    def set_device(self, device_key: DeviceKeys, mm_device_name: str):
        """
        Associate a device role with an MMCore device name.

        Args:
            device_key: Device role (e.g., DeviceKeys.GALVO_A)
            mm_device_name: MMCore device name (e.g., "Scanner:AB:33")
        """
        self._device_map[device_key] = mm_device_name

    def get_device(self, device_key: DeviceKeys) -> Optional[str]:
        """
        Get the MMCore device name for a device role.

        Args:
            device_key: Device role

        Returns:
            MMCore device name, or None if not set
        """
        return self._device_map.get(device_key)

    def get_device_exception(self, device_key: DeviceKeys) -> str:
        """
        Get the MMCore device name for a device role, raising exception if not set.

        Args:
            device_key: Device role

        Returns:
            MMCore device name

        Raises:
            ValueError: If device role is not assigned
        """
        device_name = self.get_device(device_key)
        if device_name is None:
            raise ValueError(f"No device set for {device_key}")
        return device_name

    def is_valid_device(self, device_key: DeviceKeys) -> bool:
        """
        Check if a device role has been assigned.

        Args:
            device_key: Device role

        Returns:
            True if device is assigned, False otherwise
        """
        return device_key != DeviceKeys.NONE and self.get_device(device_key) is not None

    def has_property(self, device_key: DeviceKeys, property_key: PropertyKeys) -> bool:
        """
        Check if a device has a specific property.

        Args:
            device_key: Device role
            property_key: Property to check

        Returns:
            True if device has the property, False otherwise
        """
        try:
            device_name = self.get_device(device_key)
            if device_name is None:
                return False
            return self.core.hasProperty(device_name, str(property_key))
        except Exception as e:
            logger.warning(f"Error checking property {property_key} on {device_key}: {e}")
            return False


# ============================================================================
# Gently Properties Class
# ============================================================================


class GentlyProperties:
    """
    Type-safe wrapper for MMCore property access.

    This class provides the main interface for setting and getting device
    properties. It follows the pattern used by the Java ASIdiSPIM plugin:

    - Uses enums for type safety
    - Optimizes by only setting properties when values change
    - Provides consistent error handling
    - Supports automatic type conversion

    Example:
        props = GentlyProperties(core, devices)

        # Set property
        props.set_property(DeviceKeys.GALVO_A, PropertyKeys.SPIM_NUM_SLICES, 100)
        props.set_property(DeviceKeys.GALVO_A, PropertyKeys.SPIM_STATE, PropertyValues.SPIM_ARMED)

        # Get property
        num_slices = props.get_property_int(DeviceKeys.GALVO_A, PropertyKeys.SPIM_NUM_SLICES)
        amplitude = props.get_property_float(DeviceKeys.GALVO_A, PropertyKeys.SA_AMPLITUDE_Y_DEG)
        state = props.get_property_string(DeviceKeys.GALVO_A, PropertyKeys.SPIM_STATE)
    """

    def __init__(self, core, devices: GentlyDevices):
        """
        Initialize property wrapper.

        Args:
            core: MMCore instance (pymmcore.CMMCore or RPyC proxy)
            devices: GentlyDevices instance for device name lookup
        """
        self.core = core
        self.devices = devices

    def set_property(
        self,
        device_key: DeviceKeys,
        property_key: PropertyKeys,
        value: Union[str, int, float, PropertyValues],
        ignore_error: bool = False,
    ):
        """
        Set a device property via MMCore.

        This method:
        - Converts the device key to an MMCore device name
        - Converts value to string (if enum)
        - Only sets if value changed (unless property_key.force_set is True)
        - Optionally suppresses errors

        Args:
            device_key: Device role
            property_key: Property to set
            value: New value (string, int, float, or PropertyValues enum)
            ignore_error: If True, log errors instead of raising

        Raises:
            Exception: If error occurs and ignore_error is False
        """
        try:
            # Get MMCore device name
            mm_device = self.devices.get_device_exception(device_key)

            # Convert enum value to string
            if isinstance(value, PropertyValues):
                value = str(value)

            # Get property name
            prop_name = str(property_key)

            # Check if we should set (optimization)
            should_set = property_key.force_set
            if not should_set:
                try:
                    current_value = self.core.getProperty(mm_device, prop_name)
                    should_set = str(value) != str(current_value)
                except Exception:
                    # If we can't read current value, set it anyway
                    should_set = True

            # Set property
            if should_set:
                self.core.setProperty(mm_device, prop_name, value)
                logger.debug(f"Set {device_key}.{property_key} = {value}")

        except Exception as e:
            if ignore_error:
                logger.warning(f"Error setting {device_key}.{property_key} = {value}: {e}")
            else:
                logger.error(f"Error setting {device_key}.{property_key} = {value}: {e}")
                raise

    def set_property_multiple(
        self,
        device_keys: List[DeviceKeys],
        property_key: PropertyKeys,
        value: Union[str, int, float, PropertyValues],
        ignore_error: bool = False,
    ):
        """
        Set a property on multiple devices.

        Args:
            device_keys: List of device roles
            property_key: Property to set
            value: New value
            ignore_error: If True, log errors instead of raising
        """
        for device_key in device_keys:
            self.set_property(device_key, property_key, value, ignore_error)

    def get_property_string(self, device_key: DeviceKeys, property_key: PropertyKeys) -> str:
        """
        Get a property value as a string.

        Args:
            device_key: Device role
            property_key: Property to get

        Returns:
            Property value as string, or empty string if error
        """
        try:
            mm_device = self.devices.get_device(device_key)
            if mm_device is None:
                return ""

            prop_name = str(property_key)
            return self.core.getProperty(mm_device, prop_name)

        except Exception as e:
            logger.warning(f"Error getting {device_key}.{property_key}: {e}")
            return ""

    def get_property_int(self, device_key: DeviceKeys, property_key: PropertyKeys) -> int:
        """
        Get a property value as an integer.

        Args:
            device_key: Device role
            property_key: Property to get

        Returns:
            Property value as int, or 0 if error
        """
        try:
            value_str = self.get_property_string(device_key, property_key)
            if value_str:
                return int(value_str)
            return 0
        except (ValueError, TypeError) as e:
            logger.warning(f"Error parsing int from {device_key}.{property_key}: {e}")
            return 0

    def get_property_float(self, device_key: DeviceKeys, property_key: PropertyKeys) -> float:
        """
        Get a property value as a float.

        Args:
            device_key: Device role
            property_key: Property to get

        Returns:
            Property value as float, or 0.0 if error
        """
        try:
            value_str = self.get_property_string(device_key, property_key)
            if value_str:
                return float(value_str)
            return 0.0
        except (ValueError, TypeError) as e:
            logger.warning(f"Error parsing float from {device_key}.{property_key}: {e}")
            return 0.0
```

---

## Complete Example

### Volume Scan Configuration in Gently

Here's how to use the wrapper classes to configure a volume scan:

```python
"""
Example: Configure ASI diSPIM controller for volume acquisition using MMCore wrapper.
"""

from gently.mmcore_wrapper import (
    GentlyDevices,
    GentlyProperties,
    DeviceKeys,
    PropertyKeys,
    PropertyValues,
)


def configure_spim_volume_scan(
    core,
    num_slices: int = 100,
    slice_step_um: float = 0.5,
    galvo_amplitude_deg: float = 5.0,
    scan_duration_ms: float = 10.0,
    camera_exposure_ms: float = 8.0,
):
    """
    Configure ASI Tiger controller for SPIM volume acquisition.

    This function follows the pattern from Java ASIdiSPIM plugin's
    ControllerUtils.prepareControllerForAquisition().

    Args:
        core: MMCore instance (local or RPyC proxy)
        num_slices: Number of slices in volume
        slice_step_um: Piezo step size in microns
        galvo_amplitude_deg: Galvo scan amplitude in degrees
        scan_duration_ms: Galvo scan duration in milliseconds
        camera_exposure_ms: Camera exposure time in milliseconds
    """

    # Initialize device and property managers
    devices = GentlyDevices(core)
    props = GentlyProperties(core, devices)

    # Map device roles to MMCore device names
    # (these would typically come from configuration file)
    devices.set_device(DeviceKeys.GALVO_A, "Scanner:AB:33")
    devices.set_device(DeviceKeys.PIEZO_A, "PiezoStage:P:34")
    devices.set_device(DeviceKeys.TIGER_COMM, "TigerCommHub")

    # Calculate parameters
    piezo_amplitude_um = num_slices * slice_step_um

    # Timing parameters (simplified for example)
    delay_before_scan = 0.0
    delay_before_camera = 0.5
    delay_before_laser = 0.25
    laser_duration = camera_exposure_ms
    camera_duration = camera_exposure_ms

    # ========================================================================
    # Step 1: Disable beam during configuration
    # ========================================================================
    props.set_property(
        DeviceKeys.GALVO_A, PropertyKeys.BEAM_ENABLED, PropertyValues.NO, ignore_error=True
    )

    # ========================================================================
    # Step 2: Configure galvo scanner SPIM properties
    # ========================================================================

    # Number of slices
    props.set_property(DeviceKeys.GALVO_A, PropertyKeys.SPIM_NUM_SLICES, num_slices)

    # Scan amplitude (determines slice coverage)
    props.set_property(DeviceKeys.GALVO_A, PropertyKeys.SA_AMPLITUDE_Y_DEG, galvo_amplitude_deg)

    # Timing - scan
    props.set_property(DeviceKeys.GALVO_A, PropertyKeys.SPIM_DELAY_SCAN, delay_before_scan)
    props.set_property(DeviceKeys.GALVO_A, PropertyKeys.SPIM_DURATION_SCAN, scan_duration_ms)

    # Timing - camera
    props.set_property(DeviceKeys.GALVO_A, PropertyKeys.SPIM_DELAY_CAMERA, delay_before_camera)
    props.set_property(DeviceKeys.GALVO_A, PropertyKeys.SPIM_DURATION_CAMERA, camera_duration)

    # Timing - laser
    props.set_property(DeviceKeys.GALVO_A, PropertyKeys.SPIM_DELAY_LASER, delay_before_laser)
    props.set_property(DeviceKeys.GALVO_A, PropertyKeys.SPIM_DURATION_LASER, laser_duration)

    # ========================================================================
    # Step 3: Configure piezo SPIM properties
    # ========================================================================

    # Piezo amplitude (total range of travel)
    props.set_property(DeviceKeys.PIEZO_A, PropertyKeys.SA_AMPLITUDE, piezo_amplitude_um)

    # Number of slices (shared with galvo)
    props.set_property(DeviceKeys.PIEZO_A, PropertyKeys.SPIM_NUM_SLICES_PER_PIEZO, num_slices)

    # ========================================================================
    # Step 4: Arm SPIM state machine
    # ========================================================================

    props.set_property(DeviceKeys.GALVO_A, PropertyKeys.SPIM_STATE, PropertyValues.SPIM_ARMED)

    print(f"✓ Configured SPIM volume scan:")
    print(f"  - Slices: {num_slices}")
    print(f"  - Step: {slice_step_um} µm")
    print(f"  - Galvo amplitude: {galvo_amplitude_deg}°")
    print(f"  - Scan duration: {scan_duration_ms} ms")
    print(f"  - State: ARMED")


def read_spim_status(core):
    """
    Read current SPIM configuration and status.

    Example of reading properties using the wrapper.
    """

    # Initialize managers
    devices = GentlyDevices(core)
    props = GentlyProperties(core, devices)

    # Map devices
    devices.set_device(DeviceKeys.GALVO_A, "Scanner:AB:33")
    devices.set_device(DeviceKeys.PIEZO_A, "PiezoStage:P:34")

    # Read properties
    num_slices = props.get_property_int(DeviceKeys.GALVO_A, PropertyKeys.SPIM_NUM_SLICES)
    amplitude = props.get_property_float(DeviceKeys.GALVO_A, PropertyKeys.SA_AMPLITUDE_Y_DEG)
    state = props.get_property_string(DeviceKeys.GALVO_A, PropertyKeys.SPIM_STATE)

    print(f"Current SPIM configuration:")
    print(f"  - Slices: {num_slices}")
    print(f"  - Amplitude: {amplitude}°")
    print(f"  - State: {state}")

    return {"num_slices": num_slices, "amplitude": amplitude, "state": state}


# ============================================================================
# Integration with existing Gently device classes
# ============================================================================


def integrate_with_existing_devices():
    """
    Example of how to integrate the wrapper with existing Gently device classes.

    You can either:
    1. Add the wrapper as a member of existing device classes
    2. Use the wrapper directly in Bluesky plans
    3. Gradually migrate existing code to use the wrapper
    """

    from gently.devices import DiSPIMScanner, DiSPIMPiezo

    # Option 1: Add wrapper to existing device class
    class EnhancedDiSPIMScanner(DiSPIMScanner):
        """DiSPIMScanner with MMCore wrapper for type-safe property access."""

        def __init__(self, name, core):
            super().__init__(name, core)

            # Initialize wrapper
            self.devices = GentlyDevices(core)
            self.devices.set_device(DeviceKeys.GALVO_A, name)
            self.props = GentlyProperties(core, self.devices)

        def configure_spim(self, num_slices, amplitude_deg, scan_duration_ms):
            """Configure SPIM parameters using type-safe wrapper."""
            self.props.set_property(DeviceKeys.GALVO_A, PropertyKeys.SPIM_NUM_SLICES, num_slices)
            self.props.set_property(
                DeviceKeys.GALVO_A, PropertyKeys.SA_AMPLITUDE_Y_DEG, amplitude_deg
            )
            self.props.set_property(
                DeviceKeys.GALVO_A, PropertyKeys.SPIM_DURATION_SCAN, scan_duration_ms
            )

        def arm_spim(self):
            """Arm SPIM state machine."""
            self.props.set_property(
                DeviceKeys.GALVO_A, PropertyKeys.SPIM_STATE, PropertyValues.SPIM_ARMED
            )

        def get_spim_state(self) -> str:
            """Get current SPIM state."""
            return self.props.get_property_string(DeviceKeys.GALVO_A, PropertyKeys.SPIM_STATE)


if __name__ == "__main__":
    # Example usage (assuming RPyC connection)
    import rpyc

    # Connect to MMCore server
    conn = rpyc.connect("localhost", 18861)
    core = conn.root

    # Configure volume scan
    configure_spim_volume_scan(
        core,
        num_slices=100,
        slice_step_um=0.5,
        galvo_amplitude_deg=5.0,
        scan_duration_ms=10.0,
        camera_exposure_ms=8.0,
    )

    # Read status
    status = read_spim_status(core)
    print(f"Configuration complete: {status}")
```

---

## Summary

### Key Takeaways

1. **Java ASIdiSPIM plugin uses MMCore properties, not direct serial commands**
   - All communication through `core.setProperty()` and `core.getProperty()`
   - C++ device adapter translates properties to serial commands internally

2. **Type safety via enums**
   - `Properties.Keys` enum for property names
   - `Properties.Values` enum for common values
   - `Devices.Keys` enum for device roles
   - Eliminates string typos and provides autocomplete

3. **Optimization for performance**
   - Only set property if value actually changed
   - Configurable via `forceSet` flag
   - Important for slow serial operations

4. **Clean separation of concerns**
   - `Devices` class: Device role ↔ MMCore device name mapping
   - `Properties` class: Type-safe property access wrapper
   - Application code: Uses enums, never raw strings

5. **Error handling flexibility**
   - Read errors: Silently return default value (0, empty string)
   - Write errors: Optionally suppress with `ignoreError` parameter
   - Useful for optional properties or missing devices

### Implementation Steps for Gently

1. **Create `gently/mmcore_wrapper.py`** with:
   - `PropertyKeys` enum (all SPIM/SAM properties)
   - `PropertyValues` enum (common values)
   - `DeviceKeys` enum (GALVO_A, PIEZO_A, etc.)
   - `GentlyProperties` class (property wrapper)
   - `GentlyDevices` class (device management)

2. **Update existing device classes** to use wrapper:
   - Add `props` member (GentlyProperties instance)
   - Add SPIM configuration methods
   - Migrate property access to use enums

3. **Update Bluesky plans** to use wrapper:
   - Configure SPIM via `props.set_property()`
   - Arm state machine
   - Trigger acquisition

4. **Testing**:
   - Unit tests for wrapper classes
   - Integration tests with actual hardware
   - Compare behavior to Java plugin

### Benefits

✅ **Type safety** - Autocomplete and compile-time checking
✅ **Performance** - Only set properties when values change
✅ **Maintainability** - Central definition of all properties
✅ **Consistency** - Same pattern as proven Java plugin
✅ **Debuggability** - Clear logging of all property access
✅ **Documentation** - Self-documenting via enum names

---

**Document Version:** 1.0
**Last Updated:** 2025-10-12
