# ASI diSPIM Architecture Documentation

This directory contains comprehensive documentation about the ASI Tiger plugin architecture and how to implement volume scanning in Gently based on the MicroManager approach.

**Created:** 2025-10-12

---

## Document Overview

### 1. [ASI Scanning Reference](./asi-scanning-reference.md) (~600 lines)

**Purpose:** Complete reference for ASI Tiger serial commands and SPIM properties.

**Key content:**
- Serial command syntax (NR, 2SCAN, SN, SAM)
- SPIM property reference with typical values
- Mode byte encoding for SPIM state machine
- Parameter ranges and constraints
- MMCore property interface examples

**When to use:** When you need to understand what a specific serial command does, what values are valid, or how MMCore properties map to hardware commands.

---

### 2. [ASI Plugin Architecture](./asi-plugin-architecture.md) (~600 lines)

**Purpose:** Explains the Hub-and-Spoke architecture of the C++ device adapter.

**Key content:**
- ASIHub class (serial communication hub)
- ASIPeripheralBase template (device base class)
- Device hierarchy and inheritance chain
- Extended device naming convention (e.g., "Scanner:AB:33")
- Shared properties mechanism
- Communication flow diagrams
- Comparison: ASI plugin vs Gently architecture

**When to use:** When you need to understand the abstraction layers, how the hub pattern works, or why devices are named the way they are.

---

### 3. [Java-MMCore Interface Pattern](./java-mmcore-interface-pattern.md) (~1000 lines)

**Purpose:** Shows how the Java ASIdiSPIM plugin interfaces with the C++ device adapter through MMCore, and provides a complete Python implementation template.

**Key content:**
- Java plugin architecture (Properties.java, Devices.java)
- Properties wrapper class pattern
- Enum-based type safety (device keys, property keys, values)
- Optimization strategy (only set when changed)
- Actual usage examples from ControllerUtils.java
- **Complete Python implementation** for Gently
- Integration examples with existing Gently code

**When to use:** This is the **most important document** for implementing volume scanning in Gently. It provides the exact pattern and complete Python code to follow.

---

### 4. [How Light Sheets Are Created](./lightsheet-creation-explained.md) ⭐ **NEW**

**Purpose:** Answers the critical question: "How does the Java code create a light sheet? Is it automatic when you set SPIM state?"

**Key content:**
- Two-axis galvo operation explained (X creates sheet, Y selects Z-plane)
- How `SA_AMPLITUDE_X_DEG` creates the light sheet by continuous X-axis scanning
- How `SA_AMPLITUDE_Y_DEG` selects Z-planes by Y-axis stepping
- Complete configuration checklist (what you MUST set before arming SPIM)
- Timeline diagrams showing X/Y axis coordination
- Practical code examples for both Java and Python

**When to use:** **Read this if you're confused about how the light sheet is actually generated!** This is essential for understanding what properties create the sheet vs. what properties control the Z-stack.

---

## Quick Start for Gently Implementation

### Step 1: Read the Java-MMCore Interface Pattern

Start with [java-mmcore-interface-pattern.md](./java-mmcore-interface-pattern.md) - it contains everything you need including complete Python code.

### Step 2: Understand the Key Insight

The Java plugin **never uses direct serial commands**. Instead:

```java
// Java ASIdiSPIM plugin
props.setPropValue(Devices.Keys.GALVOA, Properties.Keys.SPIM_NUM_SLICES, 100);
props.setPropValue(Devices.Keys.GALVOA, Properties.Keys.SPIM_STATE, Properties.Values.SPIM_ARMED);
```

Gently should do the same:

```python
# Gently (Python)
props.set_property(DeviceKeys.GALVO_A, PropertyKeys.SPIM_NUM_SLICES, 100)
props.set_property(DeviceKeys.GALVO_A, PropertyKeys.SPIM_STATE, PropertyValues.SPIM_ARMED)
```

### Step 3: Implement the Wrapper

Create `gently/mmcore_wrapper.py` using the template in the Java-MMCore Interface Pattern document. It includes:

- ✅ Complete enum definitions
- ✅ GentlyProperties class (property wrapper)
- ✅ GentlyDevices class (device management)
- ✅ Full example usage
- ✅ Integration with existing Gently code

### Step 4: Use in Bluesky Plans

```python
from gently.mmcore_wrapper import GentlyProperties, GentlyDevices, DeviceKeys, PropertyKeys

def volume_scan_plan(core, num_slices=100, slice_step_um=0.5):
    # Initialize
    devices = GentlyDevices(core)
    devices.set_device(DeviceKeys.GALVO_A, "Scanner:AB:33")
    devices.set_device(DeviceKeys.PIEZO_A, "PiezoStage:P:34")

    props = GentlyProperties(core, devices)

    # Configure SPIM
    props.set_property(DeviceKeys.GALVO_A, PropertyKeys.SPIM_NUM_SLICES, num_slices)
    props.set_property(DeviceKeys.GALVO_A, PropertyKeys.SA_AMPLITUDE_Y_DEG, 5.0)
    props.set_property(DeviceKeys.PIEZO_A, PropertyKeys.SA_AMPLITUDE, num_slices * slice_step_um)

    # Arm
    props.set_property(DeviceKeys.GALVO_A, PropertyKeys.SPIM_STATE, PropertyValues.SPIM_ARMED)

    # Trigger and acquire...
```

---

## Architecture Summary

### C++ Device Adapter (ASITiger)

```
TigerCommHub (ASIHub)
    ↓ Serial communication
CScanner, CPiezo (ASIPeripheralBase)
    ↓ Property handlers
ASI Tiger Controller Hardware
```

**Characteristics:**
- Direct serial communication
- Hub manages all I/O
- Properties map to serial commands internally

### Java ASIdiSPIM Plugin

```
AcquisitionPanel
    ↓ Uses
Properties + Devices wrapper classes
    ↓ Calls
CMMCore.setProperty() / getProperty()
    ↓ Communicates with
C++ Device Adapter
```

**Characteristics:**
- **No direct serial access**
- Type-safe enum-based API
- Optimization (only set if changed)
- Clean separation of concerns

### Gently (Python)

```
Bluesky Plans
    ↓ Uses
GentlyProperties + GentlyDevices wrapper classes
    ↓ Calls (via RPyC)
MMCore.setProperty() / getProperty()
    ↓ Communicates with
C++ Device Adapter
```

**Characteristics:**
- Same pattern as Java plugin
- Remote MMCore access via RPyC
- Ophyd device abstraction
- Bluesky orchestration

---

## File References

### C++ Device Adapter Files

Located in: `micro-manager/mmCoreAndDevices/DeviceAdapters/ASITiger/`

**Key files:**
- `ASIHub.h` / `ASIHub.cpp` - Serial communication hub
- `ASIPeripheralBase.h` - Template base class for devices
- `ASIScanner.h` / `ASIScanner.cpp` - Galvo scanner implementation
- `ASIPiezo.cpp` - Piezo stage implementation
- `ASITiger.h` - Property name constants

### Java Plugin Files

Located in: `micro-manager/plugins/ASIdiSPIM/src/main/java/org/micromanager/asidispim/`

**Key files:**
- `data/Properties.java` - Property wrapper class ⭐
- `data/Devices.java` - Device management class ⭐
- `utils/ControllerUtils.java` - SPIM configuration logic ⭐
- `AcquisitionPanel.java` - Volume acquisition UI

⭐ = Most important files to study

### Gently Files

Located in: `gently/`

**Existing:**
- `devices.py` - Device classes (DiSPIMScanner, DiSPIMPiezo)
- `plans.py` - Bluesky plans
- `coordinates.py` - Piezo-galvo calibration

**To create:**
- `mmcore_wrapper.py` - Properties and Devices wrapper classes (complete code provided in docs)

---

## Next Steps

1. **Review** [java-mmcore-interface-pattern.md](./java-mmcore-interface-pattern.md)
2. **Create** `gently/mmcore_wrapper.py` using provided template
3. **Update** existing device classes to use wrapper
4. **Implement** volume scan Bluesky plan
5. **Test** with hardware

---

## Questions?

If you need clarification on:
- **Serial commands** → See [ASI Scanning Reference](./asi-scanning-reference.md)
- **Hub architecture** → See [ASI Plugin Architecture](./asi-plugin-architecture.md)
- **Implementation pattern** → See [Java-MMCore Interface Pattern](./java-mmcore-interface-pattern.md)
- **How light sheet is created** → See [Light Sheet Creation Explained](./lightsheet-creation-explained.md) ⭐

All four documents are cross-referenced and work together to provide complete understanding.

---

**Documentation Version:** 1.0
**Last Updated:** 2025-10-12
