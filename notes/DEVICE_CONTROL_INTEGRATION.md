# Device Control Integration - Phase 1 Complete

The MicroscopyCopilot can now directly control the diSPIM microscope hardware! This enables conversational hardware control where you can ask the agent to calibrate embryos, acquire volumes, and manage multi-embryo experiments.

## What's New

### New Agent Capabilities

The agent can now:
- **Calibrate embryos**: Run full piezo-galvo calibration workflows
- **Acquire volumes**: Take 3D images of specific embryos
- **Move stage**: Position the stage at any embryo
- **Start timelapse**: Launch multi-embryo time-lapse acquisitions
- **Pause/Resume**: Control running acquisitions
- **Detect embryos**: (Coming in Phase 2 with SAM)

### Architecture

```
User: "Calibrate embryo_001"
  ↓
MicroscopyCopilot (Claude API)
  ↓
Tool: calibrate_embryo
  ↓
Bluesky Plan: calibrate_piezo_galvo_plan()
  ↓
RunEngine → Ophyd Devices → Micro-Manager → Hardware
```

## Quick Start

### Option 1: Automatic Setup (Easiest)

```python
from pathlib import Path
from gently.agent import create_copilot_with_hardware, run_rich_cli
import asyncio

async def main():
    # Creates copilot with RunEngine and all devices automatically
    copilot = create_copilot_with_hardware(
        storage_path=Path("./experiment_data")
    )

    # Load embryos from database
    copilot.load_embryos_from_database(database)

    # Run interactive CLI
    await run_rich_cli(copilot)

asyncio.run(main())
```

### Option 2: Manual Setup (More Control)

```python
from pathlib import Path
from client import get_mmc
from bluesky import RunEngine
from gently.agent import MicroscopyCopilot, create_devices_from_mmcore, run_rich_cli
import asyncio

async def main():
    # Get Micro-Manager core
    core = get_mmc()

    # Create devices
    devices = create_devices_from_mmcore(core)

    # Create RunEngine
    RE = RunEngine({})

    # Create copilot with hardware control
    copilot = MicroscopyCopilot(
        storage_path=Path("./experiment_data"),
        run_engine=RE,
        devices=devices
    )

    # Load embryos
    copilot.load_embryos_from_database(database)

    # Run CLI
    await run_rich_cli(copilot)

asyncio.run(main())
```

## Available Device Control Tools

### 1. Calibrate Embryo

Runs full piezo-galvo calibration for one embryo.

**User:** "Calibrate embryo_001"

**What it does:**
1. Moves stage to embryo position
2. Runs edge detection
3. Performs focus sweeps
4. Fits piezo-galvo synchronization
5. Stores calibration parameters

**Tool Call:**
```json
{
  "tool_name": "calibrate_embryo",
  "tool_input": {
    "embryo_id": "embryo_001",
    "piezo_positions": [40.0, 60.0]  // optional
  }
}
```

### 2. Acquire Volume

Acquires a single 3D volume for an embryo.

**User:** "Take a high-resolution image of embryo_002"

**What it does:**
1. Moves stage to embryo
2. Applies calibration (if available)
3. Acquires hardware-triggered SPIM volume
4. Stores volume and runs detectors

**Tool Call:**
```json
{
  "tool_name": "acquire_volume",
  "tool_input": {
    "embryo_id": "embryo_002",
    "num_slices": 50,
    "exposure_ms": 10.0,
    "save": true
  }
}
```

### 3. Move to Embryo

Moves XY stage to center on embryo.

**User:** "Move to embryo_003"

**Tool Call:**
```json
{
  "tool_name": "move_to_embryo",
  "tool_input": {
    "embryo_id": "embryo_003"
  }
}
```

### 4. Start Multi-Embryo Timelapse

Starts multi-embryo time-lapse acquisition.

**User:** "Start monitoring all embryos for hatching"

**Tool Call:**
```json
{
  "tool_name": "start_multi_embryo_timelapse",
  "tool_input": {
    "embryo_ids": ["embryo_001", "embryo_002", "embryo_003"],
    "num_timepoints": 500,
    "interval_seconds": 120,
    "num_slices": 50,
    "exposure_ms": 10.0,
    "enable_detectors": true
  }
}
```

**Note:** Full workflow orchestration will be added in Phase 3.

### 5. Pause / Resume Acquisition

**User:** "Pause the acquisition" / "Resume"

Pauses or resumes the running acquisition via RunEngine.

### 6. Detect Embryos (Coming in Phase 2)

Will use Segment Anything Model to automatically detect embryos.

**User:** "Find all embryos automatically"

**Coming soon!**

## Example Conversations

### Calibration Workflow

```
You: Load the embryos from the database
Copilot: ✓ Loaded 3 embryos from database

You: Calibrate embryo_001
Copilot: Running calibration for embryo_001...
        [moves stage, runs focus sweeps]
        ✓ Calibration complete for embryo_001
        Slope: 8.234e-03 deg/µm
        Offset: 0.0012 deg
        RMSE: 1.234e-05 deg

You: Now calibrate the other two
Copilot: [Calibrates embryo_002 and embryo_003 sequentially]
        ✓ All embryos calibrated
```

### Imaging Workflow

```
You: Take a test image of embryo_002
Copilot: Acquiring volume for embryo_002...
        ✓ Volume acquired for embryo_002
        Shape: (50, 512, 2048)
        Slices: 50, Exposure: 10.0 ms
        Timepoint: 0
        Saved to storage

You: Start monitoring all three for 12 hours
Copilot: ✓ Starting multi-embryo time-lapse acquisition
        Embryos: embryo_001, embryo_002, embryo_003
        Timepoints: 360
        Interval: 120 seconds
        Slices: 50, Exposure: 10.0 ms
        Detectors: enabled
```

## Device Configuration

### Default Configuration

```python
{
    'xy_stage_name': 'XYStage:XY:31',
    'camera_name': 'HamCam1',
    'scanner_name': 'Scanner:AB:33',
    'piezo_name': 'PiezoStage:P:34',
    'bottom_camera_name': 'Bottom PCO',
    'led_name': 'LED:X:31'
}
```

### Custom Configuration

If your devices have different names:

```python
from gently.agent import create_devices_from_mmcore
from client import get_mmc

core = get_mmc()

custom_config = {
    'xy_stage_name': 'MyXYStage',
    'camera_name': 'MyCamera',
    # ... other custom names
}

devices = create_devices_from_mmcore(core, custom_config)
```

## Architecture Details

### Device Types Created

1. **xy_stage** (DiSPIMXYStage): Stage positioning
2. **volume_scanner** (DiSPIMVolumeScanner): Hardware-triggered volume acquisition
3. **bottom_camera** (DiSPIMBottomCamera): Bottom view for embryo detection
4. **lightsheet_snap** (DiSPIMLightSheetSnap): Single-frame lightsheet imaging
5. **scanner** (DiSPIMScanner): Direct galvo control
6. **piezo** (DiSPIMPiezo): Direct piezo control

### Bluesky Integration

- **RunEngine**: Executes plans and manages hardware state
- **Plans**: Device-agnostic workflows from `gently.plans`
- **Async Execution**: Tool executors run plans in thread pool using `asyncio.to_thread()`

### Tool Execution Flow

1. User sends message to copilot
2. Claude API chooses tool and parameters
3. Tool executor validates inputs
4. Creates Bluesky plan
5. Executes plan via RunEngine in background thread
6. Returns results to Claude
7. Claude responds to user

## Next Steps

### Phase 2: SAM Integration (In Progress)

- Automatic embryo detection with Segment Anything
- Replace manual clicking with computer vision
- Integration with calibration workflow

### Phase 3: Workflow Manager

- Full multi-embryo time-lapse orchestration
- Adaptive parameter changes based on detections
- Progress monitoring and reporting
- Background execution management

### Phase 4: Testing & Validation

- Test all device control tools
- Validate SAM detection accuracy
- Performance optimization
- Production deployment

## Troubleshooting

### "Hardware control not available"

**Cause:** RunEngine or devices not initialized

**Solution:** Use `create_copilot_with_hardware()` or manually provide RunEngine and devices

### "Required devices not available"

**Cause:** Device creation failed

**Solution:** Check Micro-Manager configuration and device names

### Import Errors

**Cause:** Missing dependencies

**Solution:**
```bash
pip install bluesky ophyd pymmcore
```

### Device Not Found

**Cause:** Device name mismatch

**Solution:** Check device names in Micro-Manager and provide custom config

## Files Modified

**Phase 1 Complete:**

- `gently/agent/tools.py` - Added 8 new device control tools
- `gently/agent/copilot.py` - Added 8 tool executors
- `gently/agent/device_factory.py` - New: Device creation helpers
- `gently/agent/__init__.py` - Exported device factory functions

**Total New Capabilities:**
- 8 conversational device control tools
- Direct RunEngine integration
- Automatic device creation from mmcore
- Production-ready error handling

---

Ready for Phase 2: SAM Integration!
