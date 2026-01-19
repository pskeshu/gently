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
# Launch the Rich CLI
python launch_copilot.py

# Without microscope connection
python launch_copilot.py --no-microscope
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
│   ├── volume/YYYYMMDD/*.tif      # Raw 3D volumes (ImageJ compatible)
│   ├── image/YYYYMMDD/*.tif       # 2D images
│   └── volume_projection/         # Pre-computed projections
├── images/{session_id}/           # Session-specific volumes
│   └── {embryo_id}_{timestamp}.tif
├── sessions/{session_id}.json     # Session metadata
├── dataset.db                     # SQLite database index
├── logs/                          # Application logs
└── videos/                        # Exported videos
```

### Database Schema (dataset.db)

The SQLite database indexes all data and provides queryable access:

| Table | Description |
|-------|-------------|
| `sessions` | Session metadata (session_id, created_at, embryo_count) |
| `embryos` | Embryo records per session |
| `volumes` | Volume file paths and metadata |
| `images` | 2D image records |
| `ground_truth` | Human-annotated developmental stage labels |
| `perception_runs` | Benchmark run records |
| `predictions` | Model predictions with reasoning traces |

**Query examples:**
```python
import sqlite3
conn = sqlite3.connect('D:/Gently/dataset.db')
cur = conn.cursor()

# List sessions with ground truth
cur.execute('SELECT DISTINCT session_id FROM ground_truth')

# Get embryo volumes for a session
cur.execute('''
    SELECT embryo_id, COUNT(*) as vol_count
    FROM volumes WHERE session_id = ?
    GROUP BY embryo_id
''', ('59799c78',))

# Get ground truth annotations
cur.execute('''
    SELECT embryo_id, stage, start_timepoint
    FROM ground_truth WHERE session_id = ?
    ORDER BY embryo_id, start_timepoint
''', ('59799c78',))
```

### Finding Sessions with Ground Truth

```python
# Find sessions that have ground truth data
import sqlite3
conn = sqlite3.connect('D:/Gently/dataset.db')
cur = conn.cursor()
cur.execute('SELECT DISTINCT session_id FROM ground_truth')
sessions_with_gt = [r[0] for r in cur.fetchall()]
print(sessions_with_gt)  # e.g., ['59799c78']
```

Ground truth JSON files are also stored at: `benchmarks/data/ground_truth/{session_id}.json`

## Running Perception Benchmarks

The perception benchmark evaluates stage classification accuracy against ground truth.

### Basic Usage

```bash
python -m benchmarks.perception.runner \
    --session "D:/gently/images/{session_id}" \
    --ground-truth benchmarks/data/ground_truth/{session_id}.json \
    --output results.json \
    -v
```

### Example with Session 59799c78

```bash
# Full benchmark (all 4 embryos, ~192 timepoints each)
python -m benchmarks.perception.runner \
    --session "D:/gently/images/59799c78" \
    --ground-truth benchmarks/data/ground_truth/59799c78.json \
    --output results_59799c78.json -v

# Single embryo only
python -m benchmarks.perception.runner \
    --session "D:/gently/images/59799c78" \
    --ground-truth benchmarks/data/ground_truth/59799c78.json \
    --embryo embryo_1 \
    --output results.json -v

# Limit timepoints per embryo
python -m benchmarks.perception.runner \
    --session "D:/gently/images/59799c78" \
    --ground-truth benchmarks/data/ground_truth/59799c78.json \
    --max-timepoints 50 \
    --output results.json -v
```

### Benchmark Options

| Flag | Description |
|------|-------------|
| `--session` | Path to session directory containing TIF volumes |
| `--ground-truth` | Path to ground truth JSON file |
| `--output` | Path to save results JSON |
| `--embryo` | Run specific embryo(s) only (repeatable) |
| `--max-timepoints` | Limit timepoints per embryo |
| `--description` | Add description to the benchmark run |
| `-v, --verbose` | Enable verbose logging |

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
