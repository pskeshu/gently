# ASI Plugin Architecture Overview

This document explains the abstraction layers and design patterns used in the MicroManager ASI Tiger plugin, which helps understand how to map similar concepts to Gently.

**Date Created:** 2025-10-12

---

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Hub Pattern](#hub-pattern)
3. [Peripheral Base Class](#peripheral-base-class)
4. [Device Hierarchy](#device-hierarchy)
5. [Communication Flow](#communication-flow)
6. [Shared Properties](#shared-properties)
7. [Comparison to Gently](#comparison-to-gently)

---

## Architecture Overview

The ASI plugin uses a **Hub-and-Spoke** architecture with three main abstraction layers:

```
┌─────────────────────────────────────────────────┐
│         MMCore (Micro-Manager Core)             │
└──────────────────┬──────────────────────────────┘
                   │
┌──────────────────▼──────────────────────────────┐
│              TigerCommHub                        │
│         (Inherits from ASIHub)                  │
│  - Serial port management                       │
│  - Device registry (deviceMap_)                 │
│  - Shared property propagation                  │
└──────────────────┬──────────────────────────────┘
                   │
       ┌───────────┼───────────┬───────────┐
       │           │           │           │
┌──────▼─────┐ ┌──▼─────┐ ┌──▼─────┐ ┌──▼─────┐
│ CScanner   │ │CPiezo  │ │CZStage │ │ CLED   │
│ (Galvo)    │ │        │ │        │ │        │
└────────────┘ └────────┘ └────────┘ └────────┘
       │           │           │           │
       └───────────┴───────────┴───────────┘
                   │
       ┌───────────▼───────────┐
       │  ASIPeripheralBase    │
       │  - Hub connection     │
       │  - Address mgmt       │
       │  - Common utilities   │
       └───────────────────────┘
```

---

## Hub Pattern

### ASIHub Class

**File:** `ASIHub.h`, `ASIHub.cpp`

The hub provides centralized serial communication and device management.

### Key Responsibilities

1. **Serial Communication**
   ```cpp
   int QueryCommand(command)
   int QueryCommandVerify(command, expectedResponse)
   int QueryCommandLongReply(command)  // For INFO command
   ```

2. **Response Parsing**
   ```cpp
   int ParseAnswerAfterEquals(value)        // ":A X=123"
   int ParseAnswerAfterPosition3(value)     // ":A I45.6"
   int GetAnswerCharAtPosition3(char)       // ":A S"
   ```

3. **Device Registry**
   ```cpp
   void RegisterPeripheral(deviceLabel, addressChar)
   void UnRegisterPeripheral(deviceLabel)
   ```

4. **Shared Property Management**
   ```cpp
   int UpdateSharedProperties(addressChar, propName, value)
   ```

5. **Firmware Build Information**
   ```cpp
   int GetBuildInfo(addressLetter, build)
   bool IsDefinePresent(build, defineToLookFor)
   ```

### Hub Properties

Exposed to users via MMCore:

| Property Name | Purpose |
|---------------|---------|
| `SerialCommand` | Send arbitrary serial commands |
| `SerialResponse` | Read last response |
| `SerialTerminator` | Configure line terminator |
| `SerialCommandRepeatDuration` | Repeat command for duration |

**Example Usage:**
```cpp
// User can send direct commands via hub
hub->SetProperty("SerialCommand", "NR Y=100")
response = hub->GetProperty("SerialResponse")
```

### Thread Safety

```cpp
MMThreadLock threadLock_;  // Protects serial transactions

int ASIHub::QueryCommand(...)
{
   MMThreadGuard g(threadLock_);  // RAII lock guard
   // ... serial communication ...
}
```

---

## Peripheral Base Class

### ASIPeripheralBase Template

**File:** `ASIPeripheralBase.h`

A **template class** that peripheral devices inherit from:

```cpp
template <template <typename> class TDeviceBase, class UConcreteDevice>
class ASIPeripheralBase : public ASIBase<TDeviceBase, UConcreteDevice>
```

### Template Parameters

- `TDeviceBase`: MM device interface (e.g., `CStageBase`, `CGalvoBase`)
- `UConcreteDevice`: Concrete implementation (e.g., `CScanner`, `CPiezo`)

### Key Features

#### 1. Hub Connection
```cpp
int PeripheralInitialize(bool skipFirmware = false)
{
   // Get hub from parent
   MM::Hub* genericHub = this->GetParentHub();
   hub_ = dynamic_cast<ASIHub*>(genericHub);

   // Register with hub
   hub_->RegisterPeripheral(deviceLabel, addressChar_);
}
```

#### 2. Address Management

Each device has an address on the Tiger controller:

```cpp
std::string addressString_;  // Hex format: "31", "33", "81"
std::string addressChar_;    // Raw character (can be extended ASCII)
```

**Address Encoding:**
- Cards 1-9: ASCII `'1'` to `'9'` (hex 0x31-0x39)
- Cards 10+: Extended ASCII 0x81-0xF5
- In device names: Represented as hex strings (e.g., "Scanner:AB:33")

#### 3. Extended Name Parsing

Device names encode address and axis information:

**Format:** `DeviceType:AxisLetters:HexAddress[:Channel]`

**Examples:**
- `Scanner:AB:33` - Scanner on card 0x33, axes A and B
- `PiezoStage:P:34` - Piezo on card 0x34, axis P
- `ZStage:Z:32` - Z stage on card 0x32, axis Z
- `LED:X:31:2` - LED on card 0x31, channel 2

**Parsing Functions:**
```cpp
static bool IsExtendedName(const char* name)
static std::string GetAxisLetterFromExtName(name, position)
static std::string GetHexAddrFromExtName(name)
static int GetChannelFromExtName(name)
```

#### 4. Firmware Version Query

Each peripheral queries its own firmware:

```cpp
// Query version
command << addressChar_ << "V";
hub_->QueryCommandVerify(command.str(), ":A v");
hub_->ParseAnswerAfterPosition(4, firmwareVersion_);

// Query build date
command << addressChar_ << "CD";

// Query build name
command << addressChar_ << "BU";
```

---

## Device Hierarchy

### Inheritance Chain

```cpp
// Example: CScanner (Galvo)

CScanner
  ↓
ASIPeripheralBase<CGalvoBase, CScanner>
  ↓
ASIBase<CGalvoBase, CScanner>
  ↓
CGalvoBase (MM device interface)
  ↓
CDeviceBase<CGalvoBase>
```

### Common Peripheral Devices

| Class | Base Interface | Address Example | Purpose |
|-------|---------------|-----------------|---------|
| `CScanner` | `CGalvoBase` | Scanner:AB:33 | Galvo mirrors |
| `CPiezo` | `CStageBase` | PiezoStage:P:34 | Piezo stage |
| `CZStage` | `CStageBase` | ZStage:Z:32 | Z focus stage |
| `CXYStage` | `CXYStageBase` | XYStage:XY:35 | XY stage |
| `CLED` | `CShutterBase` | LED:X:31 | LED illumination |
| `CPLogic` | `CGenericBase` | PLogic:37 | Programmable logic |
| `CCRISP` | `CAutoFocusBase` | CRISP:F:38 | Autofocus |

### Device Construction

When MM creates a device, it parses the extended name:

```cpp
CScanner::CScanner(const char* name) :
   ASIPeripheralBase< ::CGalvoBase, CScanner >(name),
   axisLetterX_(g_EmptyAxisLetterStr),
   axisLetterY_(g_EmptyAxisLetterStr),
   // ...
{
   if (IsExtendedName(name))
   {
      axisLetterX_ = GetAxisLetterFromExtName(name);      // "A"
      axisLetterY_ = GetAxisLetterFromExtName(name, 1);   // "B"
   }
}
```

---

## Communication Flow

### Typical Property Set Operation

**Example:** Setting SPIM number of slices

```
1. User/Software
   core.setProperty("Scanner:AB:33", "SPIMNumSlices", 100)
       ↓
2. MMCore
   Calls CScanner::OnSPIMNumSlices(pProp, MM::AfterSet)
       ↓
3. CScanner
   command << addressChar_ << "NR Y=" << num_slices;
   hub_->QueryCommandVerify(command.str(), ":A")
       ↓
4. ASIHub
   QueryCommand("3NR Y=100")  // '3' is addressChar_ for card 0x33
       ↓
5. Serial Port
   Sends: "3NR Y=100\r"
   Receives: ":A \r\n"
       ↓
6. ASI Tiger Controller
   Card 0x33 receives command
   Sets parameter
   Sends acknowledgment
```

### Response Verification

```cpp
int QueryCommandVerify(command, expectedPrefix)
{
   RETURN_ON_MM_ERROR( QueryCommand(command) );

   // Check if response starts with expected prefix
   if (serialAnswer_.substr(0, len) != expectedPrefix)
   {
      return ParseErrorReply();  // Parse ":N-<error_code>"
   }
   return DEVICE_OK;
}
```

### Error Handling

ASI controllers return error codes as `:N-<code>`

```cpp
int ASIHub::ParseErrorReply()
{
   if (serialAnswer_.substr(0, 2) == ":N")
   {
      int errNo = atoi(serialAnswer_.substr(3).c_str());
      return ERR_ASICODE_OFFSET + errNo;  // Map to MM error
   }
   return ERR_UNRECOGNIZED_ANSWER;
}
```

**Common Error Codes:**
- `:N-1` - Unknown command
- `:N-2` - Unknown axis
- `:N-3` - Missing parameter
- `:N-4` - Parameter out of range
- `:N-5` - Operation failed

---

## Shared Properties

### Problem

Multiple devices on same card share certain properties (e.g., SPIM parameters for Scanner and Piezo on same controller).

### Solution: UpdateSharedProperties()

```cpp
int ASIHub::UpdateSharedProperties(addressChar, propName, value)
{
   updatingSharedProperties_ = true;

   // Iterate through all registered devices
   for (auto it : deviceMap_)
   {
      if (addressChar == it.second)  // Same card address
      {
         // Update property on this device
         GetCoreCallback()->SetDeviceProperty(
            it.first.c_str(),   // Device label
            propName.c_str(),   // Property name
            value.c_str()       // New value
         );
      }
   }

   updatingSharedProperties_ = false;
}
```

### Usage in Property Handlers

```cpp
int CScanner::OnSPIMNumSlices(...)
{
   if (eAct == MM::AfterSet)
   {
      pProp->Get(tmp);
      command << addressChar_ << "NR Y=" << tmp;
      RETURN_ON_MM_ERROR( hub_->QueryCommandVerify(command.str(), ":A") );

      // Update all devices on same card
      command.str(""); command << tmp;
      RETURN_ON_MM_ERROR(
         hub_->UpdateSharedProperties(
            addressChar_,
            pProp->GetName(),
            command.str()
         )
      );
   }
}
```

### Preventing Infinite Loops

```cpp
if (hub_->UpdatingSharedProperties())
   return DEVICE_OK;  // Skip if we're being called from UpdateSharedProperties
```

---

## Comparison to Gently

### ASI Plugin Architecture

```
TigerCommHub (ASIHub)
    ↓ (hub_ pointer)
CScanner (ASIPeripheralBase)
    ↓ (direct serial via hub)
ASI Tiger Controller
```

**Characteristics:**
- C++ implementation
- Direct serial communication
- Hub manages all serial I/O
- Peripherals hold hub pointer
- Shared properties via device registry

### Gently Architecture

```
MMCore Server (RPyC)
    ↓ (network)
MMCore (pymmcore)
    ↓ (Python bindings)
TigerCommHub (ASIHub)
    ↓ (hub_ pointer)
CScanner (ASIPeripheralBase)
    ↓ (serial)
ASI Tiger Controller
```

**Characteristics:**
- Python implementation
- MMCore accessed remotely via RPyC
- Bluesky orchestration layer
- Ophyd device abstraction
- No direct serial access

### Mapping Concepts

| ASI Plugin | Gently Equivalent | Notes |
|------------|-------------------|-------|
| `ASIHub` | MMCore via RPyC | Hub abstraction hidden in MMCore |
| `hub_->QueryCommand()` | `core.setProperty()` | Property-based interface |
| `ASIPeripheralBase` | `DiSPIMScanner` | Ophyd-compatible device class |
| Device address | Device name string | "Scanner:AB:33" |
| Shared properties | Manual coordination | Need to implement if needed |
| Serial commands | MMCore properties | Commands exposed as properties |

### Key Differences

1. **Communication Layer**
   - ASI: Direct serial via hub
   - Gently: MMCore properties via RPyC

2. **Language**
   - ASI: C++
   - Gently: Python

3. **Orchestration**
   - ASI: Property callbacks
   - Gently: Bluesky plans

4. **Device Model**
   - ASI: MM device interfaces
   - Gently: Ophyd devices

5. **Synchronization**
   - ASI: Shared properties mechanism
   - Gently: Manual coordination or Bluesky plan logic

---

## Implications for Gently

### 1. Hub Abstraction is Hidden

In Gently, you don't directly access the hub. Instead:

```python
# ASI plugin (C++)
hub_->QueryCommandVerify("NR Y=100", ":A")

# Gently (Python)
core.setProperty("Scanner:AB:33", "SPIMNumSlices", 100)
```

### 2. No Shared Property Mechanism

If multiple Gently devices need coordinated properties, implement manually:

```python
class VolumeScanCoordinator:
    """Coordinates SPIM properties across scanner and piezo"""

    def configure_spim(self, scanner, piezo, num_slices):
        scanner.set_spim_num_slices(num_slices)
        piezo.set_spim_num_slices(num_slices)  # If piezo has SPIM
```

### 3. Direct Serial Access (If Needed)

```python
# Send arbitrary command via hub
core.setProperty("TigerCommHub", "SerialCommand", "NR Y=100")
response = core.getProperty("TigerCommHub", "SerialResponse")
```

### 4. Device Naming Convention

Use same extended name format:

```python
scanner = DiSPIMScanner("Scanner:AB:33", core)  # Parse address from name
piezo = DiSPIMPiezo("PiezoStage:P:34", core)
```

---

## Summary

The ASI plugin uses a **well-designed hub-and-spoke architecture** with:

1. **Centralized serial communication** via ASIHub
2. **Template-based peripheral classes** for code reuse
3. **Address-based device identification** encoded in names
4. **Shared property propagation** for multi-device coordination
5. **Clean abstraction** between MM interfaces and hardware commands

For Gently, this architecture is largely **hidden behind MMCore**, but understanding it helps when:
- Accessing SPIM/SAM properties correctly
- Debugging serial communication issues
- Understanding why certain properties exist
- Coordinating multiple devices for volume scanning

---

**Document Version:** 1.0
**Last Updated:** 2025-10-12
