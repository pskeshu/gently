# Microscopy Copilot - Quick Start Guide

Get up and running with the conversational AI microscopy copilot in 5 minutes.

## Step 1: Install Dependencies

```bash
pip install anthropic jinja2
```

## Step 2: Set API Key

Get your API key from https://console.anthropic.com/

```bash
export ANTHROPIC_API_KEY='sk-ant-your-key-here'
```

Or on Windows:
```cmd
set ANTHROPIC_API_KEY=sk-ant-your-key-here
```

## Step 3: Try the Test Script

```bash
python test_copilot.py
```

This will:
1. Initialize the copilot
2. Load 3 example embryos
3. Run example conversations
4. Enter interactive mode

**Example output:**
```
User: What embryos do we have?

Copilot: We have 3 C. elegans embryos ready for imaging:
- embryo_001 at position (1000.0, 500.0) μm
- embryo_002 at position (1200.0, 600.0) μm
- embryo_003 at position (1400.0, 700.0) μm

Would you like me to set up an acquisition plan?
```

## Step 4: Understand the Conversation Flow

The copilot can:

### Generate Plans
```
You: "Monitor all embryos and detect hatching"

Copilot: [Uses generate_bluesky_plan tool]
"I've created an adaptive timelapse plan that will:
- Image every 2 minutes
- Use 50 Z-slices at 10ms exposure
- Analyze each volume for hatching
- Increase frame rate when hatching detected
- Stop imaging embryos after confirmed hatching

Should I start?"
```

### Answer Status Questions
```
You: "What's happening?"

Copilot: [Quick response from state, no API call]
"Experiment running for 32 minutes (timepoint 16/500)
- embryo_001: active, interval=120s
- embryo_002: active, interval=60s (high priority)
- embryo_003: hatched at t0012"
```

### Analyze Images
```
You: "Is embryo 2 hatching?"

Copilot: [Uses analyze_volume tool with Claude Vision]
"I just examined the latest image of embryo 2. The eggshell
is still intact with no visible breaches. The embryo shows
late 3-fold stage characteristics with active movement.
Based on typical timing, I predict hatching in 20-40 minutes."
```

### Adapt Parameters
```
You: "Focus more on embryo 2"

Copilot: [Uses modify_parameters tool]
"I've adjusted embryo_002 parameters:
- Sampling rate: 120s → 60s (doubled)
- Z-slices: 50 → 80 (extended range)
- Priority: normal → high
This will capture more detail as it approaches hatching."
```

## Step 5: Understand the Architecture

```
MicroscopyCopilot
├── conversation_history          # Full chat with user
├── experiment                     # Experiment state
│   ├── embryos                   # EmbryoState for each embryo
│   │   ├── recent_images         # Sliding window of images
│   │   ├── parameters            # num_slices, interval, etc.
│   │   └── analysis_results      # Cached Claude Vision results
│   ├── acquisition_status        # idle/running/paused
│   └── current_plan_name         # Active plan
├── image_manager                  # Image storage & compression
└── plan_synthesizer              # NL → Bluesky plan generation
```

## Step 6: Try Real Microscope Integration

```bash
python run_with_copilot.py
```

This loads your real `embryo_database.json` and generates plans for actual hardware.

## Step 7: Add to Your Workflow

### Option A: Standalone Script

```python
from gently.agent import MicroscopyCopilot
from pathlib import Path

# Initialize
copilot = MicroscopyCopilot(storage_path=Path("./data"))

# Load embryos
copilot.load_embryos_from_database(your_database)

# Chat
response = await copilot.handle_message("Start monitoring embryos")
print(response)
```

### Option B: FastAPI Backend

```python
# In backend/main.py
from backend.agent_api import router as agent_router, init_copilot

# Initialize copilot
copilot = init_copilot(storage_path=Path("./data"))

# Add routes
app.include_router(agent_router)
```

Then use WebSocket or REST API from frontend.

### Option C: Integrate with Bluesky

```python
from bluesky import RunEngine
from gently.agent import MicroscopyCopilot

# Setup
RE = RunEngine({})
copilot = MicroscopyCopilot(run_engine=RE, devices=your_devices)

# Generate plan
await copilot.handle_message("Monitor all embryos")
plan_code = copilot.experiment.plan_history[-1]['code']

# Execute plan (would need to exec the generated code)
# The plan calls copilot.on_volume_acquired() for each volume
```

## Key Concepts

### 1. Tool Calling
Copilot uses Claude's function calling to interact with hardware:
- `generate_bluesky_plan` - Create acquisition plans
- `query_embryo_status` - Get embryo info
- `analyze_volume` - Send images to Claude Vision
- `modify_parameters` - Change acquisition settings

### 2. State Management
Everything is tracked in `ExperimentState`:
- Embryo positions and calibrations
- Current acquisition parameters
- Recent images (sliding window)
- Analysis results (cached)

### 3. Quick vs. Complex Responses
- Simple queries answered from state (no API call)
- Complex queries go to Claude with full context

### 4. Plan Synthesis
Natural language → Template → Validation → Python code

### 5. Adaptive Acquisition
Plans call `copilot.decide_parameters()` each timepoint for dynamic adjustment.

## Common Questions

**Q: How much does this cost?**
A: Depends on usage. Typical session:
- Plan generation: ~$0.03 (one-time)
- Status queries: ~$0.00 (cached, no API call)
- Image analysis: ~$0.01 per image
- For 6 embryos over 12 hours with periodic checks: ~$2-5

**Q: Can I use a different model?**
A: Yes! Change the `model` parameter:
```python
copilot = MicroscopyCopilot(model="claude-haiku-3-5-20241022")  # Cheaper, faster
```

**Q: Does this work offline?**
A: No, it requires Anthropic API access. For offline, you'd need to:
1. Use local models (Ollama, etc.) - modify `copilot.py`
2. Pre-generate plans and run without copilot

**Q: How do I customize C. elegans knowledge?**
A: Edit `gently/agent/prompts.py`, section `CELEGANS_BIOLOGY`

**Q: Can I add my own analysis tools?**
A: Yes! Add tool definition in `tools.py` and handler in `copilot.py`

**Q: How do I clear conversation history?**
A: `copilot.conversation_history = []`

**Q: Can multiple users chat with same copilot?**
A: Currently single-user. For multi-user, you'd need to manage separate copilot instances per user or session.

## Troubleshooting

### "ANTHROPIC_API_KEY not set"
```bash
export ANTHROPIC_API_KEY='your-key'  # Linux/Mac
set ANTHROPIC_API_KEY=your-key      # Windows
```

### "Module not found: anthropic"
```bash
pip install anthropic
```

### "Copilot not responding"
1. Check internet connection
2. Verify API key is valid
3. Check Anthropic API status: https://status.anthropic.com/

### "Plan validation failed"
Check error message - usually parameter out of range:
- num_slices: 10-200
- exposure_ms: 5-100
- interval_seconds: minimum 10

### "No images to analyze"
Make sure you've called `copilot.on_volume_acquired()` after acquiring volumes.

## Next Steps

1. **Read the full docs**: `COPILOT_README.md`
2. **Explore examples**: `test_copilot.py` and `run_with_copilot.py`
3. **Try with your data**: Load real embryo database
4. **Customize prompts**: Add your specific biology knowledge
5. **Extend tools**: Add custom analysis functions
6. **Integrate with frontend**: Use WebSocket API

## Example Session

```python
# Start copilot
copilot = MicroscopyCopilot(storage_path=Path("./data"))
copilot.load_embryos_from_database(database)

# Natural language interaction
await copilot.handle_message("Monitor all embryos for hatching")
# → Generates adaptive timelapse plan

await copilot.handle_message("What's the status?")
# → Quick response from state

await copilot.handle_message("Is embryo 3 close to hatching?")
# → Analyzes latest image, predicts timing

await copilot.handle_message("Increase frame rate for embryo 3")
# → Modifies parameters, explains why

await copilot.handle_message("Why did you do that?")
# → Explains reasoning
```

Happy experimenting! 🔬🤖
