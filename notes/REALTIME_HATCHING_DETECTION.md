# Real-time Hatching Detection for diSPIM Microscopy

Integrates Claude Vision API for on-the-fly embryo hatching detection during live acquisition. Automatically stops acquiring individual embryos when hatched and ends the session when all embryos have hatched.

## Features

- **Real-time detection**: Analyzes images as they're acquired
- **Selective stopping**: Stops individual embryos when hatched
- **Smart termination**: Ends acquisition when all embryos hatched
- **Temporal context**: Sends recent timepoints to Claude for better accuracy
- **Confirmation mode**: Continues imaging for N timepoints after hatching for validation
- **Comprehensive logging**: Saves all detection results and timing data

## Architecture

### Components

1. **`image_processing_utils.py`**
   - Converts 3D volumes to Claude-compatible images
   - Extracts View A from twin-view diSPIM data
   - Computes max projections
   - Compresses images (JPEG) to stay under API limits
   - Maintains sliding window of recent images

2. **`realtime_hatching_detector.py`**
   - Manages hatching detection state
   - Integrates Claude Vision API
   - Tracks which embryos have hatched
   - Logs all detection attempts and results

3. **`run_multi_embryo_volumes_with_detection.py`**
   - Modified acquisition script with integrated detection
   - Decides which embryos to acquire
   - Calls detection after each volume
   - Implements early termination logic

## Installation

### Prerequisites

```bash
# Install required packages in gently environment
pip install anthropic pillow numpy tifffile

# Set Claude API key
export ANTHROPIC_API_KEY="your-key-here"
```

### Configuration

Edit `HATCHING_DETECTION_CONFIG` in `run_multi_embryo_volumes_with_detection.py`:

```python
HATCHING_DETECTION_CONFIG = {
    "enabled": True,  # Enable/disable detection
    "min_timepoints_before_detection": 50,  # Don't check before this (~100min at 2min/tp)
    "confidence_threshold": "HIGH",  # HIGH/MEDIUM/LOW
    "image_history_window": 10,  # Recent images to send to Claude
    "stop_when_all_hatched": True,  # End when all embryos hatched
    "continue_after_hatching": 5,  # Confirmation timepoints after hatching
    "save_processed_images": True,  # Save max projections for debugging
    "detection_log_file": "hatching_detection_log.json",
}
```

## Usage

### Basic Workflow

1. **Calibrate embryos** (if not done):
   ```bash
   python multi_embryo_calibration.py
   ```

2. **Run acquisition with detection**:
   ```bash
   python run_multi_embryo_volumes_with_detection.py
   ```

3. **Follow prompts**:
   - Number of slices per volume (default: 50)
   - Number of timepoints (default: 500 for ~16h)
   - Interval between timepoints (default: 2 min)

### What Happens

**For each timepoint:**
1. System determines which embryos to acquire (skip hatched ones)
2. For each active embryo:
   - Move stage to embryo position
   - Acquire 3D volume (diSPIM)
   - Save raw TIFF file
   - **Process for detection:**
     - Extract View A (left half of twin views)
     - Compute max projection
     - Compress to JPEG
     - Add to image history
   - **Run hatching detection** (if past minimum timepoints):
     - Send recent 10 timepoints to Claude
     - Get hatching status + confidence
     - Update embryo state
   - **Check if should stop embryo:**
     - If hatched with HIGH confidence
     - Continue for 5 more timepoints (confirmation)
     - Then skip in future timepoints
3. **Check if all embryos hatched:**
   - If yes and all confirmed → end acquisition
   - Otherwise → wait and continue to next timepoint

## Configuration Options

### Detection Timing

| Parameter | Default | Description |
|-----------|---------|-------------|
| `min_timepoints_before_detection` | 50 | Skip detection for first N timepoints (~100min at 2min/tp) |
| `image_history_window` | 10 | Number of recent images sent to Claude for temporal context |
| `continue_after_hatching` | 5 | Timepoints to continue after hatching for confirmation |

### Detection Behavior

| Parameter | Default | Description |
|-----------|---------|-------------|
| `enabled` | True | Enable/disable real-time detection |
| `confidence_threshold` | HIGH | Required confidence level (HIGH/MEDIUM/LOW) |
| `stop_when_all_hatched` | True | End acquisition when all embryos hatched |

### Data Management

| Parameter | Default | Description |
|-----------|---------|-------------|
| `save_processed_images` | True | Save max projections as PNG for debugging |
| `detection_log_file` | hatching_detection_log.json | Filename for detection log |

## Outputs

### Directory Structure

```
multi_embryo_volumes/
└── 20251122_143000/
    ├── embryo_001_embryo001_t0000_*.tif    # Raw volumes
    ├── embryo_001_embryo001_t0001_*.tif
    ├── ...
    ├── processed_images/                    # Max projections (if enabled)
    │   ├── embryo_001_t0000_maxproj.png
    │   ├── embryo_001_t0001_maxproj.png
    │   └── ...
    ├── acquisition_log.json                 # Acquisition summary
    └── hatching_detection_log.json          # Detection results
```

### Acquisition Log

```json
{
  "timestamp": "2025-11-22T14:30:00",
  "session_duration_hours": 12.5,
  "num_embryos": 9,
  "num_slices": 50,
  "interval_minutes": 2,
  "detection_config": {...},
  "hatching_summary": {
    "total_embryos": 9,
    "hatched_count": 8,
    "embryo_status": {
      "embryo_001": {
        "hatched": true,
        "timepoint": 266,
        "confidence": "HIGH",
        "timestamp": "2025-11-22T15:20:00"
      }
    }
  },
  "results": [...]
}
```

### Detection Log

```json
{
  "timestamp": "2025-11-22T14:30:00",
  "model": "claude-sonnet-4-5",
  "hatching_status": {...},
  "detection_history": {
    "embryo_001": [
      {
        "timepoint": 50,
        "hatched": false,
        "confidence": "HIGH",
        "api_duration": 3.2,
        "num_images": 10
      },
      {...}
    ]
  }
}
```

## Timing Considerations

### API Call Latency

- **Image processing**: ~0.5-1s per embryo
- **API call**: ~3-6s per embryo (depends on network, image size)
- **Total overhead**: ~4-7s per embryo per timepoint

### Example Timeline (9 embryos, 2min intervals)

| Timepoint | Activity | Duration |
|-----------|----------|----------|
| 0-49 | Acquire only (no detection) | ~90s |
| 50+ | Acquire + detect all embryos | ~150s |
| 266+ | Some embryos hatched | ~120s (fewer active) |
| 300+ | Most hatched | ~60s (minimal active) |

**Recommendation**: Use 2-3 minute intervals to accommodate detection overhead.

## Optimization Strategies

### Start Detection Late

Don't waste API calls on early development:
- C. elegans embryos hatch at ~8-12 hours
- At 2min/tp, this is timepoint ~240-360
- Set `min_timepoints_before_detection: 50` (100 minutes)
- Could even set to 200 (6.7 hours) if embryo age is known

### Adaptive Window Size

Reduce images sent to Claude as development progresses:
```python
# In realtime_hatching_detector.py
if timepoint < 100:
    window_size = 10  # Early: more context
elif timepoint < 200:
    window_size = 6  # Mid: less context needed
else:
    window_size = 4  # Late: minimal context
```

### Batch Processing (Advanced)

For very large embryo numbers, consider:
- Acquire all embryos first
- Process/detect in parallel during wait time
- Requires threading/async implementation

## Troubleshooting

### "Request too large" Error

**Cause**: Too many/large images sent to Claude

**Solutions**:
1. Reduce `image_history_window` (e.g., 6 instead of 10)
2. Increase JPEG compression quality (lower number)
3. Reduce max image size in `compress_image_for_api`

### Detection Taking Too Long

**Cause**: Exceeding timepoint interval

**Solutions**:
1. Increase interval (e.g., 3 min instead of 2 min)
2. Start detection later (increase `min_timepoints_before_detection`)
3. Reduce `image_history_window`

### False Positives

**Cause**: Detection too sensitive

**Solutions**:
1. Set `confidence_threshold: 'HIGH'`
2. Increase `continue_after_hatching` for more confirmation
3. Require multiple consecutive positive detections

### False Negatives

**Cause**: Missing hatching events

**Solutions**:
1. Lower `confidence_threshold` to 'MEDIUM'
2. Increase `image_history_window` for more context
3. Check image quality (contrast, brightness)

## API Costs

### Estimate

- **Images per embryo**: ~300 timepoints × 10 images/window = 3000 images
- **Claude Vision pricing**: ~$3 per 1000 images
- **Cost per embryo**: ~$9
- **9 embryos**: ~$81 per full experiment

**Cost optimization**:
- Start detection late: Save ~50% ($40)
- Reduce window size: Save ~40% ($32)
- Combined: ~$16 per experiment

## Advanced: Custom Detection Logic

### Modify Detection Prompt

Edit `_create_detection_content` in `realtime_hatching_detector.py`:

```python
# Add embryo-specific context
content.append(
    {
        "type": "text",
        "text": f"""
    This is embryo #{embryo_number} from position {position}.
    Previous detection showed pre-hatching signs.
    Focus on eggshell breach in upper-right quadrant.
    """,
    }
)
```

### Add Pre-hatching Detection

```python
def detect_pre_hatching(recent_images):
    """Detect pre-hatching signs"""
    # Modify prompt to look for:
    # - Embryo movement
    # - Eggshell thinning
    # - Positioning changes
    pass
```

### Multi-stage Detection

```python
# Stage 1: Coarse check (every 10 timepoints)
# Stage 2: Fine check (every timepoint if pre-hatching)
# Stage 3: Confirmation (multiple images if hatched)
```

## Testing

### Mock Mode

Test without hardware:

```python
# In run_multi_embryo_volumes_with_detection.py
MOCK_MODE = True  # Add this flag

if MOCK_MODE:
    volume = np.random.randint(0, 4096, size=(50, 512, 2048))
else:
    volume = acquire_volume_for_embryo(...)
```

### Validation

Compare with manual annotations:
```python
# Load manual annotations
annotations = load_manual_annotations()

# Compare with detector results
for embryo_id in annotations:
    manual_tp = annotations[embryo_id]["hatching_timepoint"]
    detected_tp = detector.get_hatching_timepoint(embryo_id)
    diff = abs(manual_tp - detected_tp) if detected_tp else None
    print(f"{embryo_id}: Manual={manual_tp}, Detected={detected_tp}, Diff={diff}")
```

## Future Enhancements

- [ ] Parallel API calls during wait time
- [ ] Adaptive detection frequency
- [ ] Multi-stage detection (coarse → fine)
- [ ] Pre-hatching event prediction
- [ ] Integration with automated analysis pipelines
- [ ] Real-time visualization dashboard

## References

- [Claude Vision API Documentation](https://docs.anthropic.com/claude/docs/vision)
- [diSPIM Microscopy](https://www.nature.com/articles/nmeth.2064)
- [C. elegans Embryo Development](https://www.wormatlas.org/hermaphrodite/embryo/frameset.html)
