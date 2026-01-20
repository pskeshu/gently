# Quick Start: Hardware Control with Agent

Use the conversational agent to control your diSPIM microscope!

## Method 1: Interactive CLI (Recommended for Testing)

```bash
python test_agent_hardware_control.py
```

Then just chat with the agent:
```
> Take a test image of embryo_001

Copilot: Acquiring volume for embryo_001...
         ✓ Volume acquired for embryo_001
         Shape: (50, 512, 2048)
         Slices: 50, Exposure: 10.0 ms
         Timepoint: 0
         Saved to storage
```

## Method 2: Direct Tool Call (Programmatic)

```bash
python test_single_volume_with_agent.py
```

This runs a single volume acquisition without conversation.

## Method 3: Your Own Script

```python
import asyncio
from pathlib import Path
from gently.agent import create_copilot_with_hardware

async def acquire_volume():
    # Create copilot with hardware
    copilot = create_copilot_with_hardware(
        storage_path=Path("./my_experiment")
    )

    # Load your embryos
    copilot.load_embryos_from_database(your_database)

    # Acquire volume
    result = await copilot._tool_acquire_volume({
        'embryo_id': 'embryo_001',
        'num_slices': 50,
        'exposure_ms': 10.0,
        'save': True
    })

    print(result)

asyncio.run(acquire_volume())
```

## Available Commands

When using the interactive CLI:

### Volume Acquisition
- "Take a test image of embryo_001"
- "Acquire a 100-slice volume of embryo_002"
- "Image embryo_001 with 5ms exposure"

### Stage Control
- "Move to embryo_002"
- "Go to embryo_001"

### Calibration
- "Calibrate embryo_001"
- "Run calibration for embryo_002"

### Information
- "What is the status of embryo_001?"
- "Show me all embryos"
- "/embryos" (slash command)
- "/status" (slash command)

### Control
- "Pause the acquisition"
- "Resume"
- "/quit"

## Parameters

You can customize acquisition parameters:

```python
await copilot._tool_acquire_volume({
    'embryo_id': 'embryo_001',
    'num_slices': 100,        # Number of Z slices (10-200)
    'exposure_ms': 5.0,       # Camera exposure (5-100ms)
    'save': True              # Save to disk
})
```

Or ask naturally:
```
> "Take a 100-slice volume of embryo_001 with 5ms exposure"
```

## Calibration Parameters

If your embryo has no calibration, default parameters are used:
- `galvo_center`: 0.0 degrees
- `galvo_amplitude`: 0.5 degrees
- `piezo_center`: 50.0 µm
- `piezo_amplitude`: 25.0 µm

To add calibration, run:
```python
database['embryos']['embryo_001']['calibration'] = {
    'offset': 0.0,              # galvo center (deg)
    'galvo_amplitude': 0.5,     # galvo amplitude (deg)
    'piezo_center': 50.0,       # piezo center (µm)
    'piezo_amplitude': 25.0,    # piezo amplitude (µm)
}
```

Or use the calibration tool:
```
> "Calibrate embryo_001"
```

## Troubleshooting

### "Hardware control not available"
Make sure you're using `create_copilot_with_hardware()` which automatically sets up RunEngine and devices.

### "Device not found"
Check your Micro-Manager configuration. The default device names are:
- XY Stage: `XYStage:XY:31`
- Camera: `HamCam1`
- Scanner: `Scanner:AB:33`
- Piezo: `PiezoStage:P:34`

If your devices have different names, provide custom config:
```python
from gently.agent import create_devices_from_mmcore
from client import get_mmc
from bluesky import RunEngine
from gently.agent import MicroscopyCopilot

core = get_mmc()
devices = create_devices_from_mmcore(core, {
    'xy_stage_name': 'MyStage',
    'camera_name': 'MyCamera',
    # ... etc
})

RE = RunEngine({})
copilot = MicroscopyCopilot(
    storage_path=Path("./data"),
    run_engine=RE,
    devices=devices
)
```

### Import errors
Make sure you have all dependencies:
```bash
pip install bluesky ophyd pymmcore anthropic
```

## What Gets Saved

After volume acquisition:
- Volume data stored in: `storage_path/images/embryo_XXX/`
- Format: Max projection PNG + metadata JSON
- Automatically runs detectors if enabled
- Updates embryo timepoint counter

## Next Steps

1. **Test single acquisition**: Run `test_agent_hardware_control.py`
2. **Integrate with your workflow**: Use `create_copilot_with_hardware()` in your scripts
3. **Add SAM detection**: Coming in Phase 2
4. **Multi-embryo timelapse**: Coming in Phase 3

Happy imaging! 🔬
