# Gently

Safe agentic microscopy with vision language models.

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

The architecture is designed for generalization to other microscopy platforms.

## Quick Start

```bash
# Clone and install
git clone https://github.com/pskeshu/gently.git
cd gently
pip install -r requirements_copilot.txt

# Start the hardware server (on microscope computer)
python start_server_simple.py

# Launch the copilot
python launch_copilot.py
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

We welcome contributions at all levels:

- **Agent/perception layer**: Experiment freely. This is where innovation happens.
- **Plan layer**: Follow Bluesky conventions. Plans should be composable and device-agnostic.
- **Device layer**: Be careful. Changes here affect hardware safety. Add tests.

Coding agents are welcome contributors. The safety architecture means AI-generated code can be tested safely.

## Publications

- Kesavan, P.S. & Nordenfelt, P. "From observation to understanding: A multi-agent framework for smart microscopy." *Journal of Microscopy* (2025). [DOI: 10.1111/jmi.70063](https://onlinelibrary.wiley.com/doi/10.1111/jmi.70063)
- Kesavan, P.S. & Bohra, D. "deepthought: domain driven design for microscopy with applications in DNA damage responses." *bioRxiv* (2025). [DOI: 10.1101/2025.02.25.639997](https://doi.org/10.1101/2025.02.25.639997)

## License

See [LICENSE](LICENSE) file.
