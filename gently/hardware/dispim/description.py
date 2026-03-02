"""
diSPIM hardware description.

Prompt text describing the dual-view inverted Selective Plane Illumination
Microscopy system capabilities, parameters, and safety limits.
"""

HARDWARE_DESCRIPTION = """
# diSPIM System

Dual-view Inverted Selective Plane Illumination Microscopy (diSPIM) for high-speed 3D imaging.

## System Components

### 1. DiSPIMVolumeScanner
Synchronized galvo mirrors, piezo stage, and camera for 3D volume acquisition.

**Capabilities:**
- Acquires 3D volumes by scanning light sheet through sample
- Hardware-triggered for precise synchronization
- Speed: ~20-50 slices per second

**Parameters:**
- `num_slices`: Number of Z planes to acquire (range: 10-200, typical: 50-100)
- `exposure_ms`: Camera exposure time (range: 5-100ms, typical: 10ms)
- `galvo_amplitude`: Light sheet width (typically 8° for full FOV)
- `piezo_amplitude`: Z-range in microns (calibrated per embryo)

**Typical volume acquisition time:**
- 50 slices @ 10ms exposure = ~2.5 seconds
- 100 slices @ 10ms exposure = ~5 seconds

### 2. DiSPIMXYStage
Motorized stage for multi-position imaging.

**Capabilities:**
- Position multiple embryos across large area
- Precision: ~1 micron
- Speed: ~5 mm/s

**Limits:**
- X: 500 - 2500 μm
- Y: -1000 - 1000 μm

**Safety:** Always check positions are within limits to prevent collisions!

### 3. DiSPIMPiezo
Fast Z-positioning for light sheet.

**Capabilities:**
- Synchronized with galvo for volumetric scanning
- Response time: <1ms

**Limits:**
- Range: ±200 μm from center

### 4. DiSPIMLaserControl
488nm and 561nm lasers for fluorescence.

**Important:**
- ALWAYS turn off lasers between acquisitions to prevent photobleaching!
- Laser exposure is cumulative - minimize total dose

## Safety Limits and Best Practices

### Photobleaching Prevention
- Use minimum laser power needed
- Minimize exposure time
- Maximize intervals between timepoints
- Turn off lasers immediately after acquisition

### Sample Health
- Maximum continuous imaging: ~2 hours recommended
- If embryo development appears delayed, reduce imaging frequency
- Watch for signs of photodamage: developmental arrest, blebbing

### Hardware Constraints
- Minimum interval between volumes: 10 seconds (hardware settle time)
- Stage movement takes ~0.5 seconds, add settling time
- Don't exceed stage limits (samples can collide with objectives!)

### Typical Acquisition Strategies

**Normal Development Monitoring:**
- Interval: 2-5 minutes
- Slices: 50-80 (covers full embryo)
- Exposure: 10ms
- Duration: Until hatching (~14 hours)

**High Temporal Resolution (pre-hatching):**
- Interval: 30-60 seconds
- Slices: 80-100 (embryo elongates!)
- Exposure: 8-10ms
- Duration: 30-60 minutes

**Low Photobleaching (long-term):**
- Interval: 5-10 minutes
- Slices: 40-60
- Exposure: 10ms
- Duration: 24+ hours
"""
