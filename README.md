# Gently

Safe agentic microscopy with vision language models.

**Status**: Under active development.

![Safety Architecture](docs/images/safety_architecture.png)

## Vision

Smart microscopy has evolved dramatically, but remains fundamentally rule-based. Adaptive illumination, event-triggered acquisition, real-time segmentation: these systems don't *understand* what they're imaging. There's a semantic gap between what microscopes measure (pixels, intensities) and what biologists care about (developmental stages, cell health, experimental outcomes).

Vision language models can bridge this gap through semantic reasoning over images. But how do you integrate VLMs with microscope hardware?

## Two Approaches

There's a useful distinction between [workflows and agents](https://www.anthropic.com/engineering/building-effective-agents): workflows orchestrate AI through predefined code paths, while agents dynamically direct their own tool usage.

**Workflow approach**: VLMs at specific decision points (classification, quality checks, event detection) within a traditional control system. Predictable, but rigid.

**Agentic approach**: The microscope exposed as tools an agent calls autonomously. Flexible, but risky without safety guarantees.

Gently supports both. Our **orchestrator agent** (copilot) and **perception agent** operate agentically, while **calibration workflows** use VLMs at specific decision points (coverage detection, focus assessment). The safety architecture makes either pattern safe to experiment with.

## Safety Stack

Multiple independent layers of protection:

| Layer | Protection |
|-------|------------|
| **Process Isolation** | HTTP API separates copilot from device layer. Client crashes don't affect the microscope. |
| **Device Limits** | Hard bounds validated in `set()` before any motion. Stage, piezo, galvo all protected. |
| **Plan Constraints** | Bluesky plans use a restricted vocabulary of safe primitives. |
| **Templated Actions** | Agents work with `Embryo` objects, not raw coordinates. |
| **Automatic Cleanup** | Try-finally patterns ensure lasers off on any error. |

This means: **bring your risky code**. AI-generated plans, experimental perception, coding agents iterating on control logic. The device layer catches errors before they reach hardware.

## We Welcome Coding Agents

Gently is designed for AI-assisted development. The safety stack exists precisely so that coding agents can iterate rapidly without risking hardware.

![Agent Developing Agent](docs/images/agent_developing_agent.png)

Our **agent-developing-agent methodology**: coding agents generate perception systems, test against benchmarks, analyze reasoning traces to identify failures, and refine. AI improving AI, with humans providing ground truth and guidance.

## Current Implementation

- **Hardware**: Dual-view selective plane illumination microscope (diSPIM)
- **Sample**: *C. elegans* embryo development (8 morphological stages)
- **Perception**: VLM-based stage classification with full reasoning traces
- **Interface**: Natural language copilot for biologists

### Sample-Oriented Interface

The sample is the basic unit of data, not the image or the acquisition. Each sample carries:
- Live imagery and timelapse history
- Calibration state
- Perception traces exposing all classification reasoning
- Detector configurations and event history

This design makes AI decision-making fully observable, addressing a key barrier to AI adoption in scientific instrumentation.

Currently, the sample abstraction is the `Embryo` object for *C. elegans* work. The pattern generalizes to other sample types, though UI/UX for different samples remains to be explored.

The architecture is designed for generalization to other microscopy platforms.

## Quick Start

### Prerequisites

- Python 3.11+
- [Node.js](https://nodejs.org/) 18+ (for the Ink TUI)
- An `ANTHROPIC_API_KEY` environment variable

### Setup

```bash
# Clone and install Python dependencies
git clone https://github.com/pskeshu/gently.git
cd gently
pip install -r requirements.txt

# Build the TUI (one-time, rebuild after TUI code changes)
cd gently/tui
npm install
npm run build
cd ../..
```

### Launch

```bash
# 1. Start the device layer (hardware control + SAM detection)
python start_device_layer.py

# 2. Launch the copilot
python launch_copilot.py

# Or launch without hardware (for development / review)
python launch_copilot.py --offline

# Resume a previous session
python launch_copilot.py --resume            # interactive picker
python launch_copilot.py --resume latest     # most recent session
python launch_copilot.py --resume <id>       # specific session

# List saved sessions
python launch_copilot.py --sessions
```

## Architecture

```
gently/
├── agent/              # Copilot, tool registry, timelapse orchestrator
│   ├── perception/     # VLM-based perception with reasoning traces
│   ├── plan_mode/      # Campaign planning tools
│   └── tools/          # Tool definitions for the copilot
├── context/            # Persistent agent memory (learnings, campaigns, plan items)
├── organisms/          # Organism modules (C. elegans stages, biology, detectors)
├── hardware/           # Hardware modules (diSPIM description)
├── tui/                # Ink terminal UI (Node.js / React)
├── visualization/      # Web-based monitoring and viz server
├── store.py            # GentlyStore — unified data storage (SQLite + files)
├── device_layer.py     # Device layer server (MMCore + Bluesky + SAM)
├── devices.py          # Ophyd device wrappers with safety limits
├── plans.py            # Bluesky plans for acquisition workflows
└── imaging.py          # Projection and image compression utilities
```

## Contributing

We welcome contributions across the project:

**Core Infrastructure**
- **Devices**: Be careful. Changes here affect hardware safety. Add tests.
- **Plans**: Follow Bluesky conventions. Plans should be composable and device-agnostic.
- **Simulated microscopes**: Simulated hardware for testing across the stack without real instruments.
- **Testing**: Test coverage, integration tests, edge cases.
- **Error recovery**: Better failure modes, graceful degradation.
- **Performance**: Making things faster and more efficient.

**AI & Agents**
- **Agent/perception**: Experiment freely. The safety stack has your back.
- **Design patterns**: Reusable patterns for LLM/agentic control in microscopy. If it can be a module, even better.
- **Cognitive models**: Thinking cognitively about microscopy and implementing cognitive computing models.
- **Local LLMs**: We currently use cloud providers. Support for local models would be valuable.
- **Benchmark datasets**: Ground truth annotations for perception. The agent-developing-agent loop needs data.

**Architecture & Scope**
- **System architecture**: Ideas on how to structure agentic microscopy systems.
- **Sample abstractions**: The `Embryo` object is our first sample type. What works for cells, tissue, other specimens?
- **Other microscopy platforms**: Porting to confocal, widefield, other light-sheet systems. Electron microscopy?
- **Multi-modal integration**: Combining microscopy with other data sources (genomics, proteomics, etc.).

**Human Interface**
- **UI/UX**: The web interface, copilot experience, and visualization all need work.
- **HCI research**: How do biologists work with intelligent instruments?
- **Documentation**: Tutorials, examples, better explanations for newcomers.
- **Accessibility**: Making the interface accessible to users with disabilities.
- **Internationalization**: Supporting other languages for the copilot.

Coding agents are welcome contributors.

**Questions or ideas?** [Open an issue](https://github.com/pskeshu/gently/issues).

## Acknowledgements

Gently was developed collaboratively with members of the [Shroff Lab](https://www.janelia.org/lab/shroff-lab), [Magdalena Schneider](https://schneidermc.github.io/) (AI@HHMI), and [Subin Dev S](https://github.com/subindevs).

## Publications

These papers provide theoretical background for gently's approach:

- Kesavan, P.S. & Nordenfelt, P. "From observation to understanding: A multi-agent framework for smart microscopy." *Journal of Microscopy* (2025). [DOI: 10.1111/jmi.70063](https://onlinelibrary.wiley.com/doi/10.1111/jmi.70063)
- Kesavan, P.S. & Bohra, D. "deepthought: domain driven design for microscopy with applications in DNA damage responses." *bioRxiv* (2025). [DOI: 10.1101/2025.02.25.639997](https://doi.org/10.1101/2025.02.25.639997)

## The Dream

**One microscope, made intelligent.** gently gives a microscope perception and reasoning — it understands what it's imaging, not just what it's measuring. A biologist talks to it in natural language. The safety stack means you can trust it.

**Now multiply that.** Every microscope running gently is an autonomous agent — it can perceive, reason, and act on its own instrument. Each one is a node with local intelligence.

**Connect the nodes.** [gently-meta](https://github.com/pskeshu/gently-meta) is a registry where these agents discover each other. Not a central brain — a shared awareness. Each instrument advertises what it can do, what it's working on, what it has seen.

**Science stops being bottlenecked by single instruments.** A genomics facility in Cambridge finds something unexpected. Microscopes in Boston, Tokyo, and Heidelberg are roped in to validate it across diverse samples and imaging modalities — automatically. The discovery-to-validation loop that currently takes months of emails and facility bookings happens in hours.

**Instruments become a shared, coordinated resource.** Discoveries in one modality trigger experiments in another. No single lab needs to own every capability. The collective sees more than any individual.

## License

See [LICENSE](LICENSE) file.
