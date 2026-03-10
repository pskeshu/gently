# Try Gently Without Hardware

Get the agent running in 10 minutes — no microscope needed.

## Prerequisites

- **Python 3.11+**
- **Node.js 18+** (for the terminal UI)
- An **Anthropic API key** (`ANTHROPIC_API_KEY` environment variable)

## Install

```bash
git clone https://github.com/pskeshu/gently.git
cd gently
pip install -r requirements.txt

# Build the TUI (one-time)
cd gently/tui
npm install
npm run build
cd ../..
```

## Launch

```bash
export ANTHROPIC_API_KEY=sk-ant-...
python launch_gently.py --offline
```

The `--offline` flag skips the hardware connection. The full agent launches — conversation, perception, plan mode, memory — just without microscope control.

## What You Can Do

### Talk to the Agent

The agent understands C. elegans biology, microscopy, and experimental design. Try:

- "What stages of C. elegans development can you identify?"
- "What are the key morphological features of the comma stage?"
- "How long does bean stage typically last?"
- "What are signs of embryo arrest?"

### Enter Plan Mode

Plan mode transforms the agent into a scientific collaborator that helps design experiments.

- Type `/plan` to enter plan mode
- "Design an experiment to track muscle development in C. elegans"
- "Search PubMed for C. elegans nerve ring formation"
- `/plan status` to see the current plan
- `/plan` again to exit plan mode

In plan mode, the agent can:
- Search literature (PubMed, bioRxiv, Google Scholar)
- Design multi-phase experimental campaigns
- Create imaging and bench-work specifications
- Track dependencies between tasks
- Save and version plans

### Explore Memory

The agent has persistent memory across sessions:

- "What campaigns do we have?"
- "What have we learned so far?"
- "Catch me up on where we left off"

### Resume Sessions

```bash
python launch_gently.py --resume            # interactive session picker
python launch_gently.py --resume latest     # most recent session
```

## How Perception Works Offline

Gently's perception engine uses Vision Language Models (Claude) to classify embryo developmental stages from microscopy images. The system includes:

1. **Reference images** in `gently/examples/stages/` — annotated examples for each developmental stage (early, bean, comma, 1.5fold, 2fold, pretzel, hatching, hatched)
2. **Few-shot prompting** — reference images are included as examples so the VLM can compare
3. **Structured output** — each classification returns observed features, contrastive reasoning (why *not* the adjacent stage), confidence, and a reasoning trace
4. **Multi-phase verification** — when confidence is below 0.7, independent subagents re-analyze the image

The perception engine works with any image data you provide. In online mode, images come from the microscope; offline, they can come from saved sessions or benchmark datasets.

## Run the Perception Benchmark

The benchmark evaluates classification accuracy against ground-truth annotations:

```bash
python -m benchmarks.perception.runner \
    --session /path/to/embryo_data/session_id \
    --ground-truth /path/to/ground_truth.json \
    --output results.json \
    --max-timepoints 50
```

Metrics include exact-match accuracy, adjacent-stage accuracy, mean confidence, tool usage statistics, and verification trigger rates.

Demo data for benchmarking is in `benchmarks/data/` if available, or you can use any session directory containing TIFF volumes with a corresponding ground truth JSON.

## Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `ANTHROPIC_API_KEY` | *(required)* | Claude API access |
| `GENTLY_MODEL_MAIN` | `claude-opus-4-6` | Main agent model |
| `GENTLY_MODEL_PERCEPTION` | `claude-opus-4-5-20251101` | VLM perception model |
| `GENTLY_MODEL_FAST` | `claude-haiku-4-5-20251001` | Fast reasoning (subagents) |
| `GENTLY_MODEL_MEDIUM` | `claude-sonnet-4-5-20250929` | Medium reasoning (verification) |
| `GENTLY_STORAGE_PATH` | `D:/Gently2` | Session and data storage |

## What's NOT Available Offline

These features require a connected microscope:

- **Live acquisition** — capturing new images
- **Embryo detection** — SAM-based segmentation from camera feed
- **Hardware control** — stage movement, focus, laser/LED control
- **Real-time timelapse** — adaptive multi-embryo imaging
- **Calibration** — piezo-galvo alignment workflows

Tools that require hardware will return a clear "Microscope not connected" message.

## Next Steps

- [What Gently Can Do](capabilities.md) — full capabilities overview
- [Build a Plugin](build-a-plugin.md) — add your own organism or hardware
- [Hardware Setup](hardware-setup.md) — connect a real microscope
