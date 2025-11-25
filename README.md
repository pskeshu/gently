# Gently - DiSPIM Microscope Control System

A comprehensive microscopy control system for diSPIM (dual-view Selective Plane Illumination Microscopy) with AI-powered automation, real-time analysis, and intelligent experiment management.

## Architecture

```
gently/
├── core/                 # Core infrastructure
│   ├── data_store.py     # UID-based data persistence (TiledStore)
│   ├── event_bus.py      # Pub/sub event system
│   └── service.py        # Service registry & discovery
├── session/              # Session management
│   ├── manager.py        # Session persistence & restore
│   └── state.py          # Session state definitions
├── agent/                # AI Copilot system
│   ├── copilot.py        # Main conversational AI
│   ├── tool_registry.py  # Plugin-based tool system
│   ├── tools_plugins.py  # 35 registered tools
│   └── rich_cli.py       # Interactive CLI
├── analysis/             # Analysis pipelines
│   ├── pipeline.py       # Composable pipeline builder
│   └── steps.py          # Analysis step implementations
├── visualization/        # Visualization
│   ├── server.py         # WebSocket visualization server
│   └── embryo_marker.py  # Napari integration
└── gently.py             # Main entry point
```

## Quick Start

```python
from gently import Gently

# Initialize system (data persists to D:/Gently by default)
g = Gently()

# Start a session
await g.start_session(name="My Experiment")

# Connect to microscope
await g.connect_microscope(host="localhost", port=18861)

# Store data with UID tracking
volume_uid = g.store(volume_data, "volume", metadata={"embryo": "001"})

# Run analysis pipeline
result = await g.analyze(volume_data, pipeline="embryo_detection")

# Start visualization server
await g.start_visualization_server(port=8080)
```

## Features

### Core Infrastructure

- **TiledStore**: Persistent data storage at `D:/Gently` with UID-based lineage tracking
- **EventBus**: Async pub/sub for decoupled component communication
- **ServiceRegistry**: Service discovery for microscope, SAM, and queue servers

### AI Copilot

- **35 Tools**: Plugin-based tool system with auto-generated Claude API schemas
- **Session Persistence**: Auto-save on significant actions, full conversation restore
- **Rich CLI**: Interactive terminal interface with autocomplete

### Analysis Pipelines

```python
from gently.analysis import PipelineBuilder

pipeline = (PipelineBuilder("my_analysis")
    .max_projection(axis=0)
    .threshold(method="otsu")
    .blob_detection(min_sigma=5)
    .build())

result = await pipeline.execute(volume)
```

### Visualization Server

Real-time web-based visualization at `http://localhost:8080`:
- WebSocket streaming of images
- Event log display
- Integration with EventBus for automatic updates

## Installation

```bash
# Clone repository
git clone https://github.com/pskeshu/gently.git
cd gently

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements_copilot.txt
```

## Running the Copilot

```bash
# Start the Rich CLI
python -c "from gently.agent import run_rich_cli; import asyncio; asyncio.run(run_rich_cli())"
```

Or use the batch file:
```bash
start_claude_microscope.bat
```

## Tool Categories

| Category | Tools |
|----------|-------|
| **Experiment** | `get_experiment_summary`, `generate_bluesky_plan` |
| **Embryo** | `query_embryo_status`, `skip_embryo`, `assign_nickname` |
| **Hardware** | `calibrate_embryo`, `acquire_volume`, `move_to_embryo` |
| **Detection** | `detect_embryos`, `add_detector`, `test_detector` |
| **Analysis** | `analyze_volume` (Claude Vision) |
| **Data** | `list_runs`, `get_run_data`, `search_runs` |

## Configuration

### Default Storage
Data is stored at `D:/Gently` with the following structure:
```
D:/Gently/
├── data/
│   ├── volume/YYYYMMDD/*.npy
│   └── projection/YYYYMMDD/*.npy
├── index/index.json
└── sessions/
```

### Services
- **Microscope Server**: localhost:18861 (rpyc)
- **SAM Server**: localhost:18862 (rpyc)
- **Queue Server**: localhost:60610 (HTTP)
- **Visualization**: localhost:8080 (WebSocket)

## Requirements

- Python 3.11+
- Micro-Manager 2.0
- CUDA-capable GPU (for SAM segmentation)

## License

See LICENSE file.
