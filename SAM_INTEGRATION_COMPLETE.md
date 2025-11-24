# SAM Integration Complete! 🎉

Your SAM + Claude embryo detection is now fully integrated into the conversational agent.

## What You Get

### 📍 **Just the Points**

The detection returns a simple list of embryo positions:

```python
embryos = [
    {
        'embryo_id': 0,
        'pixel_x': 512.5,
        'pixel_y': 1024.0,
        'stage_x_um': 1000.5,
        'stage_y_um': 500.2,
        'bbox_pixel': (490, 1000, 45, 48),
        'area_pixels': 3500,
        'circularity': 0.85,
        'confidence': 0.95
    },
    # ... more embryos
]
```

**That's it!** Everything you need for calibration.

## Usage

### 1️⃣ **Interactive CLI (Easiest)**

```bash
python test_sam_agent_integration.py
```

Then just say:
```
> Find all embryos

[Napari window opens with detections]

Copilot: ✓ Detected 8 embryos using SAM + Claude Vision

         Initial (SAM): 12
         Final (after Claude review): 8

         Claude verification: Verified

         👁️ Napari viewer opened
         📸 Images saved to: ./test_sam_detection/detection_results/

         Detected embryo positions:
           0: (1000.5, 500.2) µm [pixel: (512, 1024), area: 3500px, conf: 0.95]
           1: (1200.3, 650.8) µm [pixel: (800, 1300), area: 4100px, conf: 0.93]
           ...

         ✓ Loaded 8 embryos into experiment
```

### 2️⃣ **Standalone Detector**

```python
from gently.agent import SAMEmbryoDetector

detector = SAMEmbryoDetector(sam_checkpoint="sam_vit_b_01ec64.pth")

results = await detector.detect_embryos(
    image=bottom_camera_image,
    stage_position=(stage_x, stage_y),
    use_claude_review=True
)

# Get the points
for embryo in results['embryos']:
    print(f"Embryo {embryo['embryo_id']}: "
          f"stage=({embryo['stage_x_um']}, {embryo['stage_y_um']}) µm")
```

### 3️⃣ **Through Agent**

```python
from gently.agent import create_copilot_with_hardware

copilot = create_copilot_with_hardware(storage_path=Path("./data"))

# Ask to detect
response = await copilot.handle_message("Find all embryos")

# Embryos are automatically loaded into experiment
for embryo_id, embryo in copilot.experiment.embryos.items():
    print(f"{embryo_id}: {embryo.position}")
```

## What Happens During Detection

1. **Capture**: Bottom camera image captured at current stage position
2. **SAM**: Initial segmentation → 12 candidates
3. **Claude Round 1**: Reviews all detections → removes 3 false positives, adds 1 missed
4. **Claude Round 2**: Verifies corrections → confirms all good
5. **Convert**: Pixel coords → stage coordinates
6. **Visualize**: Opens napari window (non-blocking)
7. **Load**: Embryos loaded into experiment with positions
8. **Save**: Annotated images saved to disk

## Files Created

- `gently/agent/sam_detection.py` - SAM + Claude detector module (750 lines)
- `gently/agent/copilot.py` - Updated with `detect_embryos` tool
- `test_sam_agent_integration.py` - Test script with 3 modes

## Semi-Automatic Workflow

The agent shows you the results and waits for confirmation:

```
You: "Find all embryos"
[Napari shows detections with green boxes]

Agent: "Found 8 embryos. Review the napari window."

You: [Sees detection 3 is debris] "Remove detection 3"

Agent: [Would need correction tool - not yet implemented]

You: "Looks good, calibrate all"

Agent: [Starts calibration for all 8 embryos]
```

## What's NOT Done Yet

**Correction Tools** (optional enhancement):
- `remove_embryo_detection` - Remove false positives
- `add_embryo_at_position` - Add missed embryos manually

These are for edge cases where Claude misses something. In practice, the 2-round review catches most issues.

## Test It

### Mode 1: Interactive (Full Experience)
```bash
python test_sam_agent_integration.py interactive
```

### Mode 2: Simple (Just Get Points)
```bash
python test_sam_agent_integration.py simple
```

### Mode 3: Standalone (No Agent)
```bash
python test_sam_agent_integration.py standalone
```

## Requirements

```bash
pip install segment-anything torch opencv-python anthropic napari[all]
```

Download SAM checkpoint:
```bash
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
```

## The Data You Get

After detection, each embryo has:

### Position (for calibration)
- `stage_x_um`, `stage_y_um` - Where to move the stage
- `pixel_x`, `pixel_y` - Where it was in the image

### Quality Metrics
- `confidence` - SAM's stability score (0-1)
- `circularity` - Shape metric (1.0 = perfect circle)
- `area_pixels` - Size of embryo

### Metadata
- `embryo_id` - Sequential ID (0, 1, 2, ...)
- `bbox_pixel` - Bounding box (x, y, w, h)

## Next Steps

1. **Test it**: Run `test_sam_agent_integration.py`
2. **Verify detections**: Check napari shows correct embryos
3. **Use for calibration**: Say "calibrate all" to start workflow
4. **Iterate**: If you see false positives/negatives, we can add correction tools

## Architecture

```
User: "Find all embryos"
  ↓
Agent Tool: detect_embryos
  ↓
SAMEmbryoDetector.detect_embryos()
  ├─ Capture bottom camera image
  ├─ SAM segmentation (initial candidates)
  ├─ Claude Round 1 (review & correct)
  ├─ Claude Round 2 (verify corrections)
  ├─ Convert pixel → stage coordinates
  └─ Show in napari (non-blocking)
  ↓
Returns: List of embryo positions
  ↓
Agent loads into experiment
  ↓
User reviews & proceeds
```

## Example Output

```json
{
  "embryos": [
    {
      "embryo_id": 0,
      "pixel_x": 512.5,
      "pixel_y": 1024.0,
      "stage_x_um": 1000.5,
      "stage_y_um": 500.2,
      "bbox_pixel": [490, 1000, 45, 48],
      "area_pixels": 3500,
      "circularity": 0.85,
      "confidence": 0.95
    }
  ],
  "initial_detections": 12,
  "final_detections": 8,
  "verification": {
    "verified": true,
    "verification_summary": "All corrections look good..."
  }
}
```

**Ready to detect embryos!** 🔬
