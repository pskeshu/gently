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
| **Process Isolation** | RPyC separates client from hardware server. Client crashes don't affect the microscope. |
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

```bash
# Clone and install
git clone https://github.com/pskeshu/gently.git
cd gently
pip install -r requirements_copilot.txt

# 1. Start the Micro-Manager control layer
python start_server.py

# 2. Start services (visualization, etc.)
start_services.bat  # Windows

# 3. Launch the copilot
python launch_copilot.py

# 4. Open the web interface
# http://localhost:8080
```

## Architecture

```
gently/
├── agent/              # Copilot and tool registry
│   ├── perception/     # VLM-based perception with reasoning traces
│   └── tools/          # Tool definitions for the copilot
├── core/               # Event bus, data store, service registry
├── devices.py          # Ophyd device wrappers with safety limits
├── plans.py            # Bluesky plans for acquisition workflows
├── dataset/            # Data management and exploration
├── session/            # Session state and persistence
└── visualization/      # Web-based monitoring
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

Gently was developed collaboratively with members of the [Shroff Lab](https://www.janelia.org/lab/shroff-lab).

## Publications

These papers provide theoretical background for gently's approach:

- Kesavan, P.S. & Nordenfelt, P. "From observation to understanding: A multi-agent framework for smart microscopy." *Journal of Microscopy* (2025). [DOI: 10.1111/jmi.70063](https://onlinelibrary.wiley.com/doi/10.1111/jmi.70063)
- Kesavan, P.S. & Bohra, D. "deepthought: domain driven design for microscopy with applications in DNA damage responses." *bioRxiv* (2025). [DOI: 10.1101/2025.02.25.639997](https://doi.org/10.1101/2025.02.25.639997)

## License

See [LICENSE](LICENSE) file.
