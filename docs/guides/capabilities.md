# What Gently Can Do

Gently is an agentic harness for microscopy. It gives a microscope perception and reasoning through Vision Language Models, and exposes instrument control through a natural language interface.

## The Agent

At the center is a conversational agent that understands biology and microscope hardware. You talk to it in natural language. It has access to 40+ tools spanning acquisition, calibration, detection, analysis, and experimental design.

The agent operates in two modes:

- **Execution mode** — controls the microscope, acquires images, runs timelapses, classifies stages
- **Plan mode** — designs experiments, searches literature, creates multi-phase campaigns

Mode switching is natural: say "let's plan an experiment" or type `/plan`.

## Perception

VLM-based observation with full reasoning traces. The perception engine doesn't just classify — it explains *why*.

**How it works:**
1. The agent sends microscopy images to Claude's Vision API
2. Few-shot reference examples provide morphological context
3. The model describes what it sees *first*, then classifies
4. Structured output includes observed features, contrastive reasoning, confidence, and transitional state
5. Low-confidence results trigger multi-phase verification with independent subagents

**What it classifies** (C. elegans): early, bean, comma, 1.5-fold, 2-fold, pretzel, hatching, hatched — plus arrested and no-object states.

**Why traces matter:** Every classification comes with a reasoning trace showing the model's observations, what alternatives it considered, and why it chose its answer. This makes AI decision-making transparent and auditable — critical for scientific instrumentation.

**Works offline** with reference images or saved session data.

## Detection

Runtime-configurable event detection with multiple confidence levels and automated responses.

**Built-in detectors:**
- Hatching detection
- Stage transitions (comma, pretzel, gastrulation, first division)

**Custom detectors:** Define any detection prompt — the agent uses Claude Vision to evaluate it against incoming images.

**Response modes:**
- **Passive** — log the event
- **Recommend** — suggest actions to the user
- **Auto** — execute actions automatically (e.g., speed up imaging intervals near hatching)

**Verification:** Critical detections pass through multi-strategy verification — adversarial analysis, independent re-classification, temporal comparison, and ensemble voting — to prevent false positives.

## Plan Mode

A scientific collaborator for experimental design.

**Capabilities:**
- Ask clarifying questions to understand research goals
- Search literature (PubMed, bioRxiv, Google Scholar)
- Query lab history for past sessions, learnings, and outcomes
- Design multi-phase experimental campaigns
- Create detailed imaging specifications (strain, exposure, intervals, stop conditions)
- Create bench-work tasks (reagent prep, genetic crosses)
- Define decision points where researcher input is needed
- Track dependencies between tasks
- Version and snapshot plans

**Data model:** Campaigns contain phases, which contain plan items (imaging, bench, genetics, analysis, or decision points). Items can depend on each other and inherit parameters from earlier sessions.

**Works offline** — plan mode requires only the Claude API, not hardware.

## Memory

Persistent agent mind that spans sessions.

**What it remembers:**
- **Campaigns** — long-running research goals with hierarchical phases
- **Learnings** — insights with confidence levels and evidence basis
- **Observations** — notable events linked to specific embryos and sessions
- **Expectations** — predictions the agent tracks against outcomes
- **Watchpoints** — items flagged for attention
- **Embryo understanding** — per-embryo stage, health, and history

**Cross-session continuity:** Resume any session and ask "catch me up" — the agent synthesizes active campaigns, recent learnings, pending decisions, and next actions.

## Timelapse Orchestration

Multi-embryo adaptive acquisition that runs in the background.

**Features:**
- Parallel imaging across multiple embryos with per-embryo parameters
- Adaptive intervals — speed up near predicted events
- Stop conditions: stage-based, duration, fixed timepoints, all-terminal, manual
- Per-embryo lifecycle: add, skip, resume, remove mid-acquisition
- Per-embryo detectors with automated responses
- Confirmation timepoints after detection to verify events
- Live VLM classification at each acquisition

## Mesh Networking

Multi-instrument coordination for distributed experiments.

**How it works:**
- UDP broadcast discovery finds other gently instances on the network
- Bluetooth-style pairing with PIN verification
- Scope-based access control (status, campaigns, data, ML models)
- Shared campaigns — claim and execute plan items across instruments
- Append-only audit log for security

**Commands:** `/peers`, `/pair`, `/campaign share`, `/join-campaign`, `/claim`

## Safety Stack

Five independent layers of protection:

| Layer | Protection |
|-------|------------|
| **Process isolation** | HTTP API separates agent from device layer. Agent crashes don't affect hardware. |
| **Device limits** | Hard bounds validated in `set()` before any motion. Stage, piezo, galvo all protected. |
| **Plan constraints** | Bluesky plans use a restricted vocabulary of safe primitives. |
| **Templated actions** | Agent works with `Embryo` objects, not raw coordinates. |
| **Automatic cleanup** | Try-finally blocks ensure lasers and LEDs off on any error. |

This design means experimental AI code — perception systems, coding agents, novel control strategies — can run safely. The device layer catches errors before they reach hardware.

## Tool Categories

| Category | Examples | Requires Hardware |
|----------|----------|:-:|
| **Acquisition** | acquire_volume, capture_lightsheet, batch_lightsheet | Yes |
| **Calibration** | calibrate_embryo, calibrate_all_embryos | Yes |
| **Detection** | detect_embryos, manual_mark_embryos | Yes |
| **Detectors** | list_detectors, add_detector, test_detector | Partial |
| **Timelapse** | start_adaptive_timelapse, modify_timelapse_embryo | Yes |
| **Stage & Focus** | move_to_embryo, fine_focus | Yes |
| **Analysis** | analyze_volume, classify_embryo_stage | No |
| **Experiment** | get_experiment_summary, query_embryo_status | No |
| **Session** | list_sessions, import_embryos_from_session | No |
| **Planning** | create_campaign, propose_plan, search_literature | No |
| **Research** | search_literature, read_paper, search_strains | No |

Full tool reference: [TOOLS.md](../TOOLS.md) · Slash commands: [COMMANDS.md](../COMMANDS.md)

## Next Steps

- [Try Offline](try-offline.md) — run the agent without hardware
- [Build a Plugin](build-a-plugin.md) — add your own organism or hardware
- [Hardware Setup](hardware-setup.md) — connect a diSPIM
