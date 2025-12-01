## Microscopy Copilot for C. elegans Imaging

A conversational AI copilot that acts as a scientific collaborator for diSPIM microscopy experiments. The copilot can understand natural language goals, generate acquisition plans, monitor experiments in real-time, and dynamically adapt imaging parameters.

## Architecture Overview

```
┌──────────┐    Natural Language    ┌─────────────┐    Plans/Commands    ┌────────────┐
│   USER   │ ←──────────────────→  │  AI AGENT   │ ←──────────────────→ │ MICROSCOPE │
└──────────┘                        └─────────────┘                       └────────────┘
                                           ↕
                                    Image Stream
                                      Analysis
                                    State Tracking
```

### Core Components

1. **MicroscopyCopilot** (`gently/agent/copilot.py`)
   - Main orchestrator
   - Manages conversation with Claude API
   - Maintains experiment state
   - Executes tool calls

2. **State Management** (`gently/agent/state.py`)
   - `EmbryoState`: Tracks individual embryo status, parameters, analysis results
   - `ExperimentState`: Global experiment state, all embryos
   - `ImageRecord`: Stored images with metadata

3. **Plan Synthesis** (`gently/agent/plan_synthesis.py`)
   - Converts natural language goals → Bluesky plans
   - Template-based generation with validation
   - Safety checking for parameters

4. **Image Management** (`gently/agent/image_manager.py`)
   - Stores volumes, generates max projections
   - Compresses for Claude Vision API
   - Maintains sliding window of recent images

5. **System Prompts** (`gently/agent/prompts.py`)
   - C. elegans developmental biology knowledge
   - diSPIM hardware capabilities and limits
   - Bluesky plan examples
   - Tool usage guidelines

6. **Tools** (`gently/agent/tools.py`)
   - Claude function calling definitions
   - Hardware control interface
   - Status queries and parameter modification

## Installation

### Prerequisites

```bash
# Python 3.9+
pip install anthropic fastapi uvicorn websockets
pip install numpy pillow tifffile
pip install jinja2  # For plan templates
pip install bluesky ophyd  # For microscope control
```

### API Key Setup

Get your Anthropic API key from: https://console.anthropic.com/

```bash
export ANTHROPIC_API_KEY='sk-ant-your-key-here'
```

## Usage

### 1. Test the Copilot (No Hardware)

```bash
python test_copilot.py
```

This runs an interactive session where you can:
- Ask about embryo status
- Request plan generation
- Test conversation flow
- See how tool calling works

**Example conversation:**
```
User: What embryos do we have?

Copilot: We have 3 C. elegans embryos loaded:

1. embryo_001 - Position: (1000.0, 500.0) μm - Not yet imaged
2. embryo_002 - Position: (1200.0, 600.0) μm - Not yet imaged
3. embryo_003 - Position: (1400.0, 700.0) μm - Not yet imaged

All embryos are configured and ready for imaging. Would you like me to set up an acquisition plan?

User: Yes, monitor them all and detect when they hatch

Copilot: I've created a timelapse acquisition plan for all 3 embryos:

- Image every 2 minutes
- 50-slice volumes with 10ms exposure
- Run for up to 500 timepoints (~16 hours)
- Automatically detect hatching using Claude Vision analysis
- When hatching detected, increase sampling rate for confirmation
- Skip embryos after hatching confirmed to reduce photobleaching

Should I start the acquisition?
```

### 2. Run with Real Microscope

```bash
python run_with_copilot.py
```

This integrates with your diSPIM hardware:
1. Loads embryos from `embryo_database.json`
2. Asks for your experimental goal
3. Generates appropriate Bluesky plan
4. Executes plan with real hardware
5. Streams volumes for real-time analysis
6. Allows interactive queries during acquisition

### 3. FastAPI Backend Integration

Add to your `backend/main.py`:

```python
from backend.agent_api import router as agent_router, init_copilot
from pathlib import Path

# Initialize copilot
copilot = init_copilot(
    storage_path=Path("./experiment_data"),
    run_engine=None,  # Your Bluesky RunEngine
    devices={}  # Your Ophyd devices
)

# Add routes
app.include_router(agent_router)
```

Then access via:
- WebSocket: `ws://localhost:8000/api/agent/ws/chat`
- REST: `POST http://localhost:8000/api/agent/chat`

## Features

### 1. Natural Language Plan Generation

**User:** "I want to monitor all embryos for hatching with minimal photobleaching"

**Copilot generates:**
- Appropriate imaging interval (5 min)
- Minimal exposure time (8ms)
- Optimal slice count (40-50)
- Hatching detection enabled
- Auto-stop when all hatched

### 2. Real-Time Monitoring

**During acquisition, ask:**
- "What's happening with embryo 3?"
- "Which embryo is developing fastest?"
- "Is anyone close to hatching?"
- "Why did you increase the frame rate?"

**Copilot responds with:**
- Current developmental stage
- Recent observations
- Predictions based on timing
- Explanations of decisions

### 3. Dynamic Parameter Adaptation

**User:** "Focus more on embryo 2"

**Copilot automatically:**
- Increases sampling rate (2min → 1min)
- Expands Z-coverage (50 → 80 slices)
- Raises priority (imaged first)
- Explains why it made these choices

### 4. Intelligent Decision Making

Copilot can:
- **Detect events:** Hatching, cell division, morphological changes
- **Adapt acquisition:** Interval, Z-range, exposure, priority
- **Skip embryos:** After hatching confirmed or if development arrested
- **Predict timing:** "Embryo 2 will hatch in ~30 minutes"
- **Troubleshoot:** "Focus seems off, running autofocus..."
- **Optimize resources:** "Skipping embryo 4 to allocate time to active ones"

### 5. Embryo Naming & Tracking

The copilot can assign nicknames for easier reference:

```python
# Copilot notices one embryo developing faster
copilot.assign_nickname("embryo_002", "the fast one")

# Then in conversation:
User: "How's the fast one doing?"
Copilot: "The fast one (embryo_002) is at the 3-fold stage..."
```

### 6. Temporal Context Analysis

When analyzing images, copilot can:
- Include recent timepoints for context
- Detect changes over time
- Track progression through developmental stages
- Identify trends (speeding up, slowing down)

## Tool Reference

### Available Tools (Claude Function Calling)

1. **generate_bluesky_plan**
   - Input: Goal, embryo IDs, parameters
   - Output: Validated Python code for Bluesky plan
   - Example: "Monitor all embryos for hatching"

2. **query_embryo_status**
   - Input: Embryo ID
   - Output: Full status (timing, parameters, analysis results)
   - Example: Query "embryo_003" or "embryo 3"

3. **analyze_volume**
   - Input: Embryo ID, analysis prompt, use_context
   - Output: Claude Vision analysis of latest image
   - Example: "Is this embryo hatching?"

4. **modify_parameters**
   - Input: Embryo ID, parameter changes, reason
   - Output: Updated parameters
   - Example: Change interval from 120s to 60s

5. **get_experiment_summary**
   - Input: None
   - Output: Full experiment overview
   - Example: All embryos status, timing, current plan

6. **skip_embryo / resume_embryo**
   - Input: Embryo ID, reason
   - Output: Skip status updated
   - Example: Skip after hatching confirmed

7. **assign_nickname**
   - Input: Embryo ID, nickname
   - Output: Nickname assigned
   - Example: "the slow developer"

## C. elegans Biology Knowledge

The copilot understands:

- **Developmental stages:** 1-cell → 2-cell → 4-cell → gastrulation → comma → elongation → hatching
- **Timing:** ~14 hours to hatching at 20°C
- **Observable features:** Cell divisions, morphology changes, eggshell integrity, movement
- **Temperature effects:** Faster at 25°C, slower at 15°C
- **Phenotypes:** Normal, delayed, arrested, abnormal morphology

## diSPIM Hardware Knowledge

The copilot knows:

- **Capabilities:** Volume acquisition speed (~20-50 slices/s), multi-position imaging
- **Limits:** Stage range, piezo range, exposure limits
- **Safety:** Photobleaching prevention, laser management, collision avoidance
- **Best practices:** Minimum intervals, typical parameters, sample health monitoring

## Advanced Use Cases

### 1. Automated Hypothesis Testing

**User:** "Is there correlation between division rate and hatching time?"

**Copilot:**
- Designs multi-embryo experiment
- Tracks division timing for each
- Records hatching times
- Reports correlations

### 2. Event-Triggered Acquisition

**User:** "When you see cell migration start, switch to high temporal resolution"

**Copilot:**
- Monitors for migration patterns
- Automatically increases frame rate
- Captures event in detail
- Returns to normal after

### 3. Predictive Scheduling

**User:** "We have 6 hours left, prioritize the most interesting samples"

**Copilot:**
- Assesses each embryo's state
- Ranks by scientific interest
- Allocates imaging time optimally
- Skips less interesting ones

### 4. Multi-Modal Analysis

**Copilot analyzes:**
- Morphology (brightfield)
- Fluorescence (reporter expression)
- Motion/activity (frame-to-frame diff)
- Custom user-defined features

### 5. Closed-Loop Perturbations

**Integration with automation:**
```python
# When copilot detects specific stage
if copilot.detect("gastrulation onset"):
    # Trigger drug addition
    perfusion_system.add_drug("cytochalasin D", concentration=10)
    # Increase imaging frequency
    copilot.modify_parameters("all", {"interval_seconds": 30})
```

## API Examples

### REST API

```python
import requests

# Send message
response = requests.post(
    "http://localhost:8000/api/agent/chat",
    json={"message": "What's the status?"}
)
print(response.json()['response'])

# Get experiment status
status = requests.get("http://localhost:8000/api/agent/status")
print(status.json())

# Load embryos
requests.post(
    "http://localhost:8000/api/agent/load-embryos",
    json={"database": embryo_database}
)
```

### WebSocket API

```python
import asyncio
import websockets
import json

async def chat():
    uri = "ws://localhost:8000/api/agent/ws/chat"
    async with websockets.connect(uri) as websocket:
        # Send message
        await websocket.send(json.dumps({
            "type": "message",
            "content": "Start monitoring all embryos"
        }))

        # Receive response
        response = await websocket.recv()
        data = json.loads(response)
        print(data['content'])

asyncio.run(chat())
```

## Extending the Copilot

### Add Custom Analysis

```python
# In your code
async def custom_analysis(volume, embryo_state):
    """Custom analysis function"""
    # Your analysis logic
    result = analyze_something(volume)

    # Store in embryo state
    embryo_state.custom_classifications['my_feature'] = result

    return result

# Register with copilot
copilot.custom_analyzers['my_feature'] = custom_analysis
```

### Add New Plan Templates

```python
# In plan_synthesis.py
new_template = PlanTemplate(
    name='my_custom_plan',
    description='Description of what it does',
    template_str='''
def my_custom_plan(...):
    # Your plan code using Jinja2 templates
    ...
'''
)

copilot.plan_synthesizer.library.templates['my_custom_plan'] = new_template
```

### Add Custom Tools

```python
# In tools.py, add new tool definition
{
    "name": "my_custom_tool",
    "description": "What this tool does",
    "input_schema": {...}
}

# In copilot.py, add handler
async def _tool_my_custom_tool(self, tool_input: Dict) -> str:
    # Your tool implementation
    return result
```

## Troubleshooting

### Copilot Not Responding

1. Check API key: `echo $ANTHROPIC_API_KEY`
2. Check network connection
3. Look for errors in console
4. Try simpler query first

### Plan Generation Fails

1. Check parameter limits (num_slices: 10-200, exposure: 5-100ms)
2. Ensure embryos are loaded
3. Check plan validator errors
4. Review generated code for syntax errors

### Analysis Not Working

1. Ensure images are being stored
2. Check image compression (should be <1MB)
3. Verify Claude Vision API access
4. Check for timeout errors

## Performance Tips

1. **Minimize API Calls:** Use quick responses for simple queries
2. **Cache Analysis Results:** Store in embryo.custom_classifications
3. **Batch Operations:** Modify multiple embryos at once
4. **Use Appropriate Models:** Haiku for quick queries, Sonnet for complex reasoning
5. **Manage Context:** Clear conversation history if getting too long

## Safety & Best Practices

1. **Always validate generated plans** before execution
2. **Monitor photobleaching** - copilot will try, but verify
3. **Have kill switches** - ability to halt hardware immediately
4. **Test in simulation** first with Bluesky simulator
5. **Keep conversation focused** - start fresh for new experiments
6. **Review decisions** - ask copilot to explain why it made changes

## Future Enhancements

- [ ] Multi-user collaboration (multiple people querying same experiment)
- [ ] Experiment summarization for lab meetings
- [ ] Automatic figure generation
- [ ] Cross-experiment learning (learn from previous experiments)
- [ ] Predictive models for hatching timing
- [ ] Integration with analysis pipelines
- [ ] Voice interface for hands-free operation
- [ ] Automated documentation generation

## Support

For issues or questions:
1. Check logs in `experiment_data/logs/`
2. Review conversation history
3. Test with `test_copilot.py` first
4. Check Anthropic API status

## License

See main repository LICENSE file.
