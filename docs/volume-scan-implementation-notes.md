# Volume Scan Implementation Notes for Gently

This document outlines the strategy for implementing volume (z-stack) scanning in Gently based on the ASI Tiger plugin's approach, adapted for Gently's Bluesky/Ophyd architecture.

**Date Created:** 2025-10-12
**Status:** Planning Phase

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture Comparison](#architecture-comparison)
3. [Implementation Approaches](#implementation-approaches)
4. [Device Layer Extensions](#device-layer-extensions)
5. [Bluesky Plan Design](#bluesky-plan-design)
6. [Synchronization Strategy](#synchronization-strategy)
7. [Testing and Validation](#testing-and-validation)
8. [Implementation Roadmap](#implementation-roadmap)

---

## Overview

### Goal
Implement synchronized volume scanning in Gently that coordinates:
- **Piezo** (objective focus) - z-axis movement
- **Galvo** (scanner) - light sheet positioning/scanning
- **Camera** - image acquisition
- **Laser** - illumination control

### Design Principles
1. **Device-Agnostic**: Use Bluesky's device abstraction
2. **Composable**: Volume scans should compose with focus/detection plans
3. **MMCore-Based**: Leverage existing MMCore infrastructure
4. **Calibration-Driven**: Use piezo-galvo calibration from `coordinates.py`
5. **Pure Python**: No C++ extensions, work through MMCore API

---

## Architecture Comparison

### ASI Plugin (C++ Device Adapter)

```
┌─────────────────────────────────────┐
│   MMCore (Micro-Manager Core)       │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│   ASIScanner.cpp Device Adapter     │
│   - Direct serial commands          │
│   - Hardware state machine          │
│   - Firmware timing control         │
└──────────────┬──────────────────────┘
               │ Serial (RS-232/USB)
┌──────────────▼──────────────────────┐
│   ASI Tiger Controller              │
│   - SPIM state machine (firmware)   │
│   - Galvo/piezo/camera sync         │
│   - Hardware timing (sub-ms)        │
└─────────────────────────────────────┘
```

### Gently (Python Bluesky/Ophyd)

```
┌─────────────────────────────────────┐
│   Bluesky RunEngine                 │
│   - Executes plans (generators)     │
│   - Event model / data collection   │
└──────────────┬──────────────────────┘
               │
┌──────────────▼──────────────────────┐
│   Ophyd Device Classes               │
│   (DiSPIMScanner, DiSPIMPiezo, etc) │
│   - Device abstraction               │
│   - Status objects                   │
└──────────────┬──────────────────────┘
               │ RPyC (network)
┌──────────────▼──────────────────────┐
│   RPyC Server (start_server.py)     │
│   - Exposes MMCore via network      │
└──────────────┬──────────────────────┘
               │ Python bindings
┌──────────────▼──────────────────────┐
│   MMCore (pymmcore)                 │
│   - Device property access          │
│   - Position/trigger control        │
└──────────────┬──────────────────────┘
               │ C++ API / Serial
┌──────────────▼──────────────────────┐
│   ASI Tiger Controller              │
│   - Same hardware state machine     │
│   - Same SPIM capabilities          │
└─────────────────────────────────────┘
```

**Key Difference**: Gently adds Python-based orchestration layers (Bluesky/Ophyd/RPyC) on top of MMCore, but still has access to all hardware features through MMCore properties.

---

## Implementation Approaches

### Approach 1: Hardware SPIM Mode (Recommended for Speed)

**Advantages:**
- Leverages hardware state machine
- Sub-millisecond timing precision
- Minimal software overhead
- Proven ASI implementation

**How It Works:**
1. Configure SPIM parameters via MMCore properties
2. Arm SPIM state machine
3. Trigger (TTL or software)
4. Hardware handles synchronization
5. Collect images from buffer

**Implementation in Gently:**
```python
# Configure via MMCore properties
core.setProperty("Scanner:AB:33", "SPIMNumSlices", num_slices)
core.setProperty("Scanner:AB:33", "SPIMState", "Armed")
# Trigger and let hardware run
core.setProperty("Scanner:AB:33", "SPIMState", "Running")
```

**Challenges:**
- Requires understanding SPIM property mappings
- Less flexible than software control
- Debugging is hardware-level

### Approach 2: Software-Coordinated Sequential (Recommended for Flexibility)

**Advantages:**
- Full Python control
- Easy debugging and modification
- Works with existing Bluesky patterns
- Composable with other plans

**How It Works:**
1. Software loop over z positions
2. Move piezo to each position
3. Calculate synchronized galvo position
4. Set galvo position (via calibration)
5. Trigger camera acquisition
6. Collect image

**Implementation in Gently:**
```python
@bpp.run_decorator(md=metadata)
def volume_scan_sequential(piezo, scanner, camera, z_positions, calib):
    for i, z_pos in enumerate(z_positions):
        # Move piezo
        yield from bps.mv(piezo, z_pos)

        # Sync galvo (using calibration)
        galvo_pos = piezo_to_galvo(z_pos, calib.slope, calib.offset)
        yield from bps.mv(scanner, [galvo_pos, 0])

        # Acquire
        yield from bps.trigger_and_read([camera, piezo, scanner])
```

**Challenges:**
- Slower than hardware mode
- Software timing overhead
- Requires tight calibration

### Approach 3: Single Axis Mode (SAM) Continuous Scan

**Advantages:**
- Continuous galvo scanning (smooth motion)
- Good for fast acquisition
- Software triggers camera during scan

**How It Works:**
1. Configure SAM pattern (triangle/sine)
2. Start continuous galvo scan
3. Software synchronizes camera triggers
4. Piezo can move between scans or stay fixed

**Implementation in Gently:**
```python
# Configure SAM via MMCore
core.setProperty("Scanner:AB:33", "SingleAxisXMode", "1")  # Enabled
core.setProperty("Scanner:AB:33", "SingleAxisXPattern", "3")  # Sine
core.setProperty("Scanner:AB:33", "SingleAxisXAmplitude(deg)", 0.5)

# Software loop with synchronized triggering
for z_pos in z_positions:
    yield from bps.mv(piezo, z_pos)
    # Trigger camera at correct phase of galvo scan
    yield from bps.trigger_and_read([camera])
```

**Challenges:**
- Requires precise timing synchronization
- More complex than sequential
- Phase calculation needed

---

## Device Layer Extensions

### 1. Enhanced DiSPIMScanner Class

**File:** `gently/devices.py`

**Current Implementation:** Lines 556-645
- Basic position control via `set()` and `read()`
- XY interface for AB axes

**Proposed Extensions:**

```python
class DiSPIMScanner:
    """Enhanced scanner with SPIM and SAM support"""

    # ... existing code ...

    # SPIM Hardware Mode Methods
    def configure_spim(self, num_slices: int, scan_duration_ms: float,
                       camera_duration_ms: float, delay_ms: float = 0.2):
        """Configure hardware SPIM parameters"""
        self.core.setProperty(self.device_name, "SPIMNumSlices", num_slices)
        self.core.setProperty(self.device_name, "SPIMScanDuration(ms)", scan_duration_ms)
        self.core.setProperty(self.device_name, "SPIMCameraDuration(ms)", camera_duration_ms)
        self.core.setProperty(self.device_name, "SPIMDelayBeforeCamera(ms)", delay_ms)

    def arm_spim(self):
        """Arm SPIM for TTL or software trigger"""
        self.core.setProperty(self.device_name, "SPIMState", "Armed")

    def start_spim(self):
        """Start SPIM acquisition"""
        self.core.setProperty(self.device_name, "SPIMState", "Running")

    def stop_spim(self):
        """Stop/idle SPIM"""
        self.core.setProperty(self.device_name, "SPIMState", "Idle")

    def get_spim_state(self) -> str:
        """Query SPIM state: Idle/Armed/Running"""
        return self.core.getProperty(self.device_name, "SPIMState")

    # Single Axis Mode Methods
    def configure_sam(self, axis: str, mode: int, pattern: int,
                      amplitude_deg: float, offset_deg: float, period_ms: float):
        """Configure Single Axis Mode for continuous scanning"""
        axis_prop = "X" if axis.upper() in ["X", "A"] else "Y"
        self.core.setProperty(self.device_name, f"SingleAxis{axis_prop}Mode", str(mode))
        self.core.setProperty(self.device_name, f"SingleAxis{axis_prop}Pattern", str(pattern))
        self.core.setProperty(self.device_name, f"SingleAxis{axis_prop}Amplitude(deg)", amplitude_deg)
        self.core.setProperty(self.device_name, f"SingleAxis{axis_prop}Offset(deg)", offset_deg)
        self.core.setProperty(self.device_name, f"SingleAxis{axis_prop}Period(ms)", period_ms)

    def enable_sam(self, axis: str, enabled: bool = True):
        """Enable/disable Single Axis Mode"""
        axis_prop = "X" if axis.upper() in ["X", "A"] else "Y"
        mode = "1" if enabled else "0"
        self.core.setProperty(self.device_name, f"SingleAxis{axis_prop}Mode", mode)
```

### 2. Enhanced DiSPIMPiezo Class

**File:** `gently/devices.py`

**Current Implementation:** Lines 474-553

**Proposed Extensions:**

```python
class DiSPIMPiezo:
    """Enhanced piezo with SPIM coordination"""

    # ... existing code ...

    def configure_spim_piezo(self, num_slices: int):
        """Configure piezo for SPIM mode (if applicable)"""
        # Some piezo cards have SPIM properties
        try:
            self.core.setProperty(self.device_name, "SPIMNumSlices", num_slices)
        except:
            pass  # Not all piezo cards support SPIM properties

    def get_spim_piezo_state(self) -> str:
        """Query piezo SPIM state if supported"""
        try:
            # Query using serial command via hub
            return self.core.getProperty("TigerCommHub", "SerialResponse")
        except:
            return "N/A"
```

---

## Bluesky Plan Design

### Volume Scan Plan Family

**File:** `gently/plans.py` (new additions)

#### 1. Sequential Volume Scan

```python
def volume_scan_sequential(piezo, scanner, camera, z_positions: List[float],
                           calibration: ReferenceMap,
                           galvo_axis: str = 'A',
                           metadata: Optional[Dict] = None):
    """
    Device-agnostic sequential volume scan

    Coordinates piezo movement with galvo positioning using calibration.

    Parameters
    ----------
    piezo : Ophyd positioner
        Piezo device for z-axis
    scanner : Ophyd 2D positioner
        Galvo scanner device
    camera : Ophyd detector
        Camera for image acquisition
    z_positions : List[float]
        Z positions to scan (in micrometers)
    calibration : ReferenceMap
        Piezo-galvo calibration data
    galvo_axis : str
        'A' or 'B' - which galvo axis to use
    metadata : Dict, optional
        Additional metadata

    Yields
    ------
    Msg
        Bluesky messages
    """
    from .coordinates import piezo_to_galvo

    md = {
        'plan_name': 'volume_scan_sequential',
        'piezo': piezo.name,
        'scanner': scanner.name,
        'camera': camera.name,
        'num_slices': len(z_positions),
        'z_range': (min(z_positions), max(z_positions)),
        'calibration_slope': calibration.piezo_galvo_slope,
        'calibration_offset': calibration.piezo_galvo_offset,
    }
    if metadata:
        md.update(metadata)

    @bpp.run_decorator(md=md)
    def inner():
        for i, z_pos in enumerate(z_positions):
            print(f"Volume scan: slice {i+1}/{len(z_positions)}, z={z_pos:.2f} µm")

            # Move piezo to z position
            yield from bps.mv(piezo, z_pos)

            # Calculate synchronized galvo position
            galvo_pos = piezo_to_galvo(
                z_pos,
                calibration.piezo_galvo_slope,
                calibration.piezo_galvo_offset
            )

            # Move galvo (maintain other axis at 0)
            if galvo_axis.upper() == 'A':
                yield from bps.mv(scanner, [galvo_pos, 0.0])
            else:
                yield from bps.mv(scanner, [0.0, galvo_pos])

            # Acquire image
            yield from bps.trigger_and_read([camera, piezo, scanner],
                                          name=f'volume_slice_{i:04d}')

    yield from inner()


def volume_scan_bidirectional(piezo, scanner, camera, z_start: float, z_end: float,
                               num_slices: int, calibration: ReferenceMap,
                               metadata: Optional[Dict] = None):
    """
    Bidirectional volume scan (forward then reverse)

    Scans forward, then reverse to reduce motion time.
    """
    # Create forward positions
    z_forward = np.linspace(z_start, z_end, num_slices)

    # Forward scan
    yield from volume_scan_sequential(
        piezo, scanner, camera, z_forward, calibration,
        metadata={**(metadata or {}), 'scan_direction': 'forward'}
    )

    # Reverse scan
    z_reverse = z_forward[::-1]
    yield from volume_scan_sequential(
        piezo, scanner, camera, z_reverse, calibration,
        metadata={**(metadata or {}), 'scan_direction': 'reverse'}
    )


def volume_scan_continuous_sam(piezo, scanner, camera, z_positions: List[float],
                                sam_config: Dict, metadata: Optional[Dict] = None):
    """
    Volume scan with continuous Single Axis Mode galvo scanning

    Parameters
    ----------
    sam_config : Dict
        SAM configuration: {'pattern': 3, 'amplitude': 0.5, 'period': 10, ...}
    """
    md = {
        'plan_name': 'volume_scan_continuous_sam',
        'sam_config': sam_config,
    }
    if metadata:
        md.update(metadata)

    @bpp.run_decorator(md=md)
    def inner():
        # Configure SAM
        scanner.configure_sam(
            axis='A',
            mode=1,  # Enabled
            pattern=sam_config['pattern'],
            amplitude_deg=sam_config['amplitude'],
            offset_deg=sam_config.get('offset', 0.0),
            period_ms=sam_config['period']
        )

        try:
            # Enable SAM
            scanner.enable_sam('A', True)

            # Scan through z positions
            for i, z_pos in enumerate(z_positions):
                yield from bps.mv(piezo, z_pos)

                # Trigger camera during continuous galvo scan
                # TODO: Add phase synchronization logic here
                yield from bps.trigger_and_read([camera, piezo])

        finally:
            # Always disable SAM when done
            scanner.enable_sam('A', False)

    yield from inner()
```

#### 2. Hardware SPIM Volume Scan

```python
def volume_scan_hardware_spim(scanner, piezo, camera,
                               num_slices: int,
                               scan_duration_ms: float,
                               camera_duration_ms: float,
                               metadata: Optional[Dict] = None):
    """
    Hardware-timed SPIM volume scan using ASI controller state machine

    This is the fastest approach, leveraging firmware timing.

    Parameters
    ----------
    scanner : DiSPIMScanner
        Scanner device with SPIM support
    piezo : DiSPIMPiezo
        Piezo device
    camera : DiSPIMCamera
        Camera device
    num_slices : int
        Number of z-slices
    scan_duration_ms : float
        Duration of each galvo scan
    camera_duration_ms : float
        Camera exposure time
    """
    md = {
        'plan_name': 'volume_scan_hardware_spim',
        'num_slices': num_slices,
        'scan_duration_ms': scan_duration_ms,
        'camera_duration_ms': camera_duration_ms,
    }
    if metadata:
        md.update(metadata)

    @bpp.run_decorator(md=md)
    def inner():
        # Configure SPIM parameters
        scanner.configure_spim(
            num_slices=num_slices,
            scan_duration_ms=scan_duration_ms,
            camera_duration_ms=camera_duration_ms,
            delay_ms=0.2
        )

        # Arm SPIM
        scanner.arm_spim()
        print(f"SPIM armed, state: {scanner.get_spim_state()}")

        # Start SPIM acquisition
        scanner.start_spim()
        print("SPIM started")

        # Wait for completion (poll state)
        while scanner.get_spim_state() == "Running":
            yield from bps.sleep(0.1)

        print(f"SPIM complete, state: {scanner.get_spim_state()}")

    yield from inner()
```

---

## Synchronization Strategy

### Piezo-Galvo Calibration

**Existing Infrastructure:** `gently/coordinates.py:78-168`

```python
from gently.coordinates import (
    piezo_to_galvo,
    galvo_to_piezo,
    calculate_piezo_galvo_calibration
)

# Use calibration in volume scan
galvo_pos = piezo_to_galvo(piezo_pos, slope, offset)
```

### Timing Synchronization

**Key Timing Parameters:**

```python
@dataclass
class VolumeScanTiming:
    """Timing configuration for volume scans"""
    # Sequential scan timing
    piezo_settle_time_ms: float = 50.0      # Time for piezo to settle
    galvo_settle_time_ms: float = 5.0       # Time for galvo to settle
    camera_exposure_ms: float = 10.0        # Camera exposure

    # Hardware SPIM timing
    spim_scan_duration_ms: float = 10.0     # Total scan time per slice
    spim_camera_duration_ms: float = 9.5    # Camera exposure
    spim_delay_before_camera_ms: float = 0.2  # Settling before trigger
    spim_delay_before_laser_ms: float = 0.0   # Laser trigger delay

    # SAM continuous scan timing
    sam_period_ms: float = 10.0             # Scan period
    sam_camera_phase: float = 0.25          # Trigger at 25% of cycle
```

### Multi-Device Coordination

```python
def create_volume_scan_devices(core: pymmcore.CMMCore) -> Dict:
    """
    Create coordinated device set for volume scanning

    Returns dict with scanner, piezo, camera all configured
    for synchronized operation.
    """
    from gently.devices import DiSPIMScanner, DiSPIMPiezo, DiSPIMCamera
    from gently.coordinates import load_reference_map

    # Load calibration
    calib = load_reference_map("calibration.json")

    # Create devices
    devices = {
        'scanner': DiSPIMScanner("Scanner:AB:33", core, name='lightsheet_scanner'),
        'piezo': DiSPIMPiezo("PiezoStage:P:34", core, name='objective_piezo'),
        'camera': DiSPIMCamera("HamCam1", core, name='spim_camera'),
        'calibration': calib
    }

    return devices
```

---

## Testing and Validation

### Unit Tests

**File:** `tests/test_volume_scan.py` (new)

```python
def test_volume_scan_sequential():
    """Test sequential volume scan plan"""
    # Mock devices
    # Execute plan
    # Verify positions, images acquired

def test_piezo_galvo_sync():
    """Test piezo-galvo synchronization"""
    # Given calibration
    # Test position calculations
    # Verify accuracy

def test_spim_configuration():
    """Test SPIM property setting"""
    # Configure SPIM params
    # Verify properties set correctly
```

### Integration Tests

**File:** `test_volume_scan_hardware.py` (new)

```python
def test_volume_scan_on_hardware():
    """
    Test volume scan on actual hardware

    Requires: Connected ASI Tiger system
    """
    # Connect to MMCore
    # Create devices
    # Run volume scan
    # Analyze acquired data
```

### Validation Criteria

1. **Positional Accuracy**
   - Piezo positions within 0.1 µm of target
   - Galvo positions within 0.01° of calculated

2. **Timing Precision**
   - Sequential scan: ±10% of expected slice time
   - Hardware SPIM: <1% timing jitter (firmware controlled)

3. **Image Quality**
   - Consistent focus across volume
   - No motion blur
   - Proper light sheet alignment

4. **Throughput**
   - Sequential: ~100-200 ms per slice
   - Hardware SPIM: ~10-50 ms per slice

---

## Implementation Roadmap

### Phase 1: Device Layer Extensions (1-2 days)
- [ ] Add SPIM methods to `DiSPIMScanner`
- [ ] Add SAM methods to `DiSPIMScanner`
- [ ] Test property access via MMCore
- [ ] Document available properties

### Phase 2: Sequential Volume Scan (2-3 days)
- [ ] Implement `volume_scan_sequential()` plan
- [ ] Integrate piezo-galvo calibration
- [ ] Add timing configuration
- [ ] Test on hardware

### Phase 3: Bidirectional Scanning (1 day)
- [ ] Implement `volume_scan_bidirectional()` plan
- [ ] Optimize motion paths
- [ ] Test speed improvements

### Phase 4: Hardware SPIM Mode (2-3 days)
- [ ] Implement `volume_scan_hardware_spim()` plan
- [ ] Configure SPIM properties correctly
- [ ] Test TTL triggering
- [ ] Benchmark speed

### Phase 5: SAM Continuous Scanning (3-4 days)
- [ ] Implement `volume_scan_continuous_sam()` plan
- [ ] Add phase synchronization
- [ ] Test timing accuracy
- [ ] Compare to sequential

### Phase 6: Integration and Testing (2-3 days)
- [ ] Write comprehensive tests
- [ ] Hardware validation
- [ ] Performance benchmarking
- [ ] Documentation

### Phase 7: Advanced Features (optional)
- [ ] Dual-side SPIM
- [ ] Multi-embryo volume scans
- [ ] Adaptive z-spacing
- [ ] Real-time focus correction

---

## Open Questions

1. **Camera Buffering**: How to efficiently collect images during fast SPIM scans?
   - Use MMCore sequence acquisition?
   - Buffer in memory vs. stream to disk?

2. **Calibration Stability**: How often to recalibrate piezo-galvo sync?
   - Daily? Weekly? Per experiment?
   - Auto-calibration routine?

3. **Error Handling**: What if devices lose sync?
   - Timeout detection?
   - Recovery procedures?
   - Safe stop mechanisms?

4. **Data Management**: How to organize volume scan data?
   - OME-Zarr format?
   - Integrate with Bluesky databroker?
   - Metadata standards?

---

## References

### Gently Files
- `gently/devices.py` - Device classes
- `gently/plans.py` - Bluesky plans
- `gently/coordinates.py` - Coordinate transformations
- `docs/asi-scanning-reference.md` - ASI plugin serial commands

### External Documentation
- [Bluesky Plans Documentation](https://blueskyproject.io/bluesky/plans.html)
- [Ophyd Device Documentation](https://blueskyproject.io/ophyd/)
- [ASI Tiger Manual](http://asiimaging.com/docs/products/tiger)
- [PyMMCore Documentation](https://github.com/micro-manager/pymmcore)

---

**Document Version:** 1.0
**Status:** Planning
**Next Steps:** Begin Phase 1 implementation
