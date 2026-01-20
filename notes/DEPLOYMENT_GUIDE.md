# Deployment Guide - Agent on Microscope

Quick guide to deploy the conversational agent with SAM detection on your microscope.

## Prerequisites

### 1. Install Dependencies

```bash
# Core dependencies (if not already installed)
pip install anthropic rich prompt-toolkit

# SAM dependencies
pip install segment-anything torch torchvision opencv-python

# Visualization (optional but recommended)
pip install napari[all]

# Bluesky/Ophyd (should already be installed)
pip install bluesky ophyd
```

### 2. Download SAM Model

```bash
# Download SAM checkpoint (one-time)
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

# Or use curl
curl -L -o sam_vit_b_01ec64.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

Place it in your working directory or specify the path when creating the detector.

### 3. Set API Key

```bash
# Set your Anthropic API key
export ANTHROPIC_API_KEY='your-key-here'

# Or add to your .bashrc/.zshrc for persistence
echo 'export ANTHROPIC_API_KEY="your-key-here"' >> ~/.bashrc
```

## Deployment Scripts

### Option 1: Simple Agent Control Script

Create `run_microscope_agent.py`:

```python
#!/usr/bin/env python3
"""
Microscopy Agent for diSPIM Control

Simple script to run the conversational agent on the microscope.
"""

import asyncio
from pathlib import Path
from gently.agent import create_copilot_with_hardware, run_rich_cli


async def main():
    # Create experiment directory
    experiment_dir = Path("./experiment_data")
    experiment_dir.mkdir(exist_ok=True)

    print("="*70)
    print("MICROSCOPY COPILOT - diSPIM Control")
    print("="*70)
    print("\nInitializing hardware control...")

    # Create copilot with hardware
    copilot = create_copilot_with_hardware(
        storage_path=experiment_dir
    )

    print("\n✓ Ready! Hardware control enabled")
    print("\nAvailable commands:")
    print('  • "Find all embryos" - Detect embryos with SAM + Claude')
    print('  • "Calibrate embryo_000" - Run calibration')
    print('  • "Take a test image of embryo_001" - Acquire volume')
    print('  • "Move to embryo_002" - Position stage')
    print('  • "/embryos" - List all embryos')
    print('  • "/status" - Show experiment status')
    print('  • "/quit" - Exit')
    print()

    # Run interactive CLI
    await run_rich_cli(
        copilot,
        history_file=experiment_dir / ".agent_history"
    )


if __name__ == "__main__":
    import os

    # Check for API key
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY not set")
        print("Set it with: export ANTHROPIC_API_KEY='your-key-here'")
        exit(1)

    asyncio.run(main())
```

### Option 2: Full Workflow Script

Create `run_full_experiment.py`:

```python
#!/usr/bin/env python3
"""
Complete Multi-Embryo Experiment Workflow

1. Detect embryos automatically
2. Review and confirm
3. Calibrate all
4. Start time-lapse monitoring
"""

import asyncio
from pathlib import Path
from gently.agent import create_copilot_with_hardware


async def main():
    experiment_name = input("Experiment name: ")
    experiment_dir = Path(f"./experiments/{experiment_name}")
    experiment_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*70)
    print(f"EXPERIMENT: {experiment_name}")
    print("="*70)

    # Initialize copilot
    copilot = create_copilot_with_hardware(storage_path=experiment_dir)

    # Step 1: Detect embryos
    print("\n[1/3] Detecting embryos...")
    response = await copilot.handle_message("Find all embryos automatically")
    print(response)

    # Confirm
    confirm = input("\nProceed with these detections? (y/n): ")
    if confirm.lower() != 'y':
        print("Aborting. You can run again or manually correct.")
        return

    # Step 2: Calibrate all
    print("\n[2/3] Calibrating all embryos...")
    embryo_ids = list(copilot.experiment.embryos.keys())

    for i, embryo_id in enumerate(embryo_ids):
        print(f"\nCalibrating {embryo_id} ({i+1}/{len(embryo_ids)})...")
        response = await copilot.handle_message(f"Calibrate {embryo_id}")
        print(response)

    # Step 3: Start time-lapse
    print("\n[3/3] Starting time-lapse...")

    num_timepoints = int(input("Number of timepoints (default 500): ") or "500")
    interval = int(input("Interval in seconds (default 120): ") or "120")

    response = await copilot.handle_message(
        f"Start monitoring all embryos for {num_timepoints} timepoints "
        f"with {interval} second intervals"
    )
    print(response)

    print("\n" + "="*70)
    print("EXPERIMENT STARTED")
    print("="*70)
    print(f"\nData saving to: {experiment_dir}")
    print("Monitor progress in the CLI...")


if __name__ == "__main__":
    import os

    if not os.getenv("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY not set")
        exit(1)

    asyncio.run(main())
```

## Usage

### Quick Test

```bash
# Simple test
python run_microscope_agent.py
```

Then try:
```
> Find all embryos
> Take a test image of embryo_000
```

### Full Experiment

```bash
# Full workflow
python run_full_experiment.py
```

This will:
1. Detect embryos (SAM + Claude)
2. Show napari window for review
3. Ask for confirmation
4. Calibrate all embryos
5. Start time-lapse

## Device Configuration

If your devices have different names, create a config file:

**config_devices.py:**
```python
DEVICE_CONFIG = {
    'xy_stage_name': 'XYStage:XY:31',      # Your XY stage name
    'camera_name': 'HamCam1',              # Main camera
    'scanner_name': 'Scanner:AB:33',       # Galvo scanner
    'piezo_name': 'PiezoStage:P:34',       # Piezo
    'bottom_camera_name': 'Bottom PCO',    # Bottom camera
    'led_name': 'LED:X:31'                 # LED (optional)
}
```

Then use:
```python
from config_devices import DEVICE_CONFIG
from gently.agent import create_devices_from_mmcore
from client import get_mmc

core = get_mmc()
devices = create_devices_from_mmcore(core, DEVICE_CONFIG)
```

## Troubleshooting

### "SAM checkpoint not found"

```bash
# Make sure it's in the current directory or specify path
ls sam_vit_b_01ec64.pth

# Or specify path in code:
detector = SAMEmbryoDetector(
    sam_checkpoint="/path/to/sam_vit_b_01ec64.pth"
)
```

### "Hardware control not available"

Make sure Micro-Manager is running and devices are configured:
```python
from client import get_mmc
core = get_mmc()

# Test devices
print(core.getXYStageDevice())
print(core.getCameraDevice())
```

### "napari not opening"

Napari is optional. Detection still works, just check saved images:
```bash
ls ./experiment_data/detection_results/detection_final.png
```

### GPU Not Available

SAM works on CPU (slower but fine for 1-2 images):
```python
detector = SAMEmbryoDetector(device="cpu")  # Explicit CPU
```

For GPU:
```bash
# Check CUDA
python -c "import torch; print(torch.cuda.is_available())"

# If False, install CUDA-enabled PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## File Organization

```
experiment_data/
├── images/                    # Acquired volumes
│   └── embryo_000/
│       ├── t0000_max_proj.png
│       └── t0000_metadata.json
├── detection_results/         # SAM detection outputs
│   ├── detection_initial.png
│   ├── detection_round1.png
│   └── detection_final.png
├── detector_registry.json     # Detector configurations
└── .agent_history            # Command history
```

## Next Steps

1. **Test detection**: `python run_microscope_agent.py` → "Find all embryos"
2. **Verify hardware**: Try "Move to embryo_000", "Take a test image"
3. **Run calibration**: "Calibrate embryo_000"
4. **Start experiment**: "Start monitoring all embryos"

## Safety Notes

- **Test first**: Try detection and single volume before full time-lapse
- **Check positions**: Verify detected positions with "Move to embryo_X"
- **Monitor first cycle**: Watch the first few timepoints
- **Backup regularly**: Copy experiment_data/ folder

Happy imaging! 🔬
