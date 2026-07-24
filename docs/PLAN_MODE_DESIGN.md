# Plan Mode — Experimental Design Collaborator

## Overview

Plan mode transforms the microscopy agent from a pure execution agent ("detect embryos, acquire volume, run timelapse") into a **scientific research collaborator** that helps design complete experimental plans before touching hardware.

When a researcher says "I want to investigate nerve ring formation," the agent should reason through the full scientific workflow: strains, reporters, controls, imaging strategy, bench validations, perturbation experiments, timeline — not just "start a timelapse."

The output is a **structured, living plan** stored in the data model (campaigns, plan items, imaging specs) that the agent actively tracks across weeks of work. When the researcher sits down to image, the agent already knows exactly what to do.

---

## Problem Statement

The current agent has two gaps:

1. **No upstream reasoning.** It jumps from a scientific question directly to microscope operations. There's no mode where it helps design the experiment — what strains to use, what controls to run, what developmental window to target, what complementary experiments to perform outside of imaging.

2. **No downstream tracking.** Even when the researcher has a plan in their head, the agent doesn't track progress across sessions. Each session starts fresh. There's no "you're on session 3 of 5 for the wild-type data collection" or "the genetic cross you started last week — is it done yet?"

The data model (Campaign with hierarchy, PlannedSession with parameters) already exists to hold this information. What's missing is the reasoning layer that populates it and the tracking layer that maintains it.

---

## What Plan Mode Is

A dedicated conversational mode where the agent acts as a scientific advisor. It has a different personality, different system prompt, and different tool set than execution mode.

**Execution mode:** "Act immediately. Call tools first, explain after. Don't describe what you would do — do it."

**Plan mode:** "Reason carefully. Ask questions. Search the literature. Understand before proposing. Challenge assumptions. Think about controls the researcher might not have considered."

Plan mode produces a structured research plan that decomposes into campaigns, phases, and plan items — both imaging sessions and non-imaging work (bench assays, genetic crosses, analysis). Each imaging item carries a complete specification (strain, acquisition parameters, timing, stop conditions, success criteria) so the agent can auto-configure when it's time to execute.

---

## What Plan Mode Is NOT

- Not a separate application — same message loop, different prompt and tools
- Not a rigid wizard — free-form conversation with natural phases
- Not imaging-only — reasons about the full experimental workflow
- Not a one-time output — produces a living document the agent maintains

---

## Entry Points

### 1. Natural Language Detection

The agent detects planning-level questions and offers to enter plan mode:

- "I want to investigate nerve ring formation"
- "Help me design an experiment for..."
- "How should I approach studying X?"
- "I need to compare wild-type and mutant at comma stage"

### 2. Slash Command

`/plan` explicitly enters plan mode. Subcommands for existing plans:

```
/plan              → Enter plan mode (new plan or modify existing)
/plan status       → Show current plan progress
/plan next         → What's the next action?
/plan update       → Report progress on non-imaging items
```

### 3. Startup Wizard

When creating a new campaign during onboarding, offer: "Want to design a full experimental plan for this?"

---

## Architecture

### Mode Switching

The agent gets a `self.mode` attribute:

```python
class AgentMode(str, Enum):
    EXECUTION = "execution"  # Current behavior
    PLAN = "plan"  # Experimental design mode
```

The main message loop (`_call_claude_stream()`) selects prompt and tools based on mode:

```python
if self.mode == AgentMode.PLAN:
    system_prompt = build_plan_prompt(context, lab_context)
    tools = get_plan_tools()
else:
    system_prompt = build_system_prompt(experiment_state, connection_status)
    tools = get_execution_tools(has_microscope)
```

### File Structure

```
gently/agent/plan_mode/
    __init__.py              # PlanMode class, entry/exit logic
    prompt.py                # Plan mode system prompt
    tools/
        __init__.py
        research.py          # search_literature, read_paper, search_strains
        lab_context.py       # query_lab_history, check_hardware_capability
        planning.py          # propose_plan, create_campaign, create_planned_session, add_plan_item
```

---

## Plan Mode System Prompt

```
You are a scientific research planner — the same microscopy agent,
but right now you're helping design an experiment, not run one.

Your role:
1. Understand the scientific question deeply
2. Identify what's known and what's unknown
3. Design a complete experimental plan — not just imaging
4. Think about strains, controls, validations, timeline
5. Be specific: name real strains, real genes, real assays
6. Challenge assumptions — suggest controls the researcher
   might not have thought of
7. Suggest experiments outside of imaging where appropriate

You have access to:
- The researcher (ask questions to understand their goals)
- Literature search (find relevant papers, methods, strains)
- Lab context (past sessions, existing campaigns, learnings)
- Paper analysis (read PDFs the researcher provides)
- Your knowledge of {organism} biology and {hardware} capabilities

DO NOT rush to a plan. Gather information first. Ask questions.
Search the literature. Understand before proposing.

Your output will be a structured experimental plan that decomposes
into campaigns, phases, and planned sessions — with complete
imaging specifications for every imaging session.

When proposing imaging parameters, be specific:
- Name the strain and reporter
- Specify num_slices, exposure, laser power, interval
- Define the developmental window (start stage → stop condition)
- Set adaptive intervals if the biology calls for it
- Define success criteria for each session
- Note what to compare against (controls, prior sessions)

For non-imaging items (bench work, genetics, analysis):
- Describe the protocol or approach
- List required reagents or strains
- Estimate timeline
- Define success criteria
- Note dependencies on other plan items

{organism_biology_knowledge}

{hardware_description}
```

The prompt includes organism biology knowledge and hardware description (from the existing modules) so the agent can reason about developmental timing and hardware capabilities.

---

## Plan Mode Tools

### Research Tools

| Tool | Description |
|------|-------------|
| `search_literature` | Search PubMed / Google Scholar / bioRxiv. Returns titles, abstracts, key findings. Input: query string, optional filters (organism, year range, journal). |
| `read_paper` | Ingest a PDF the researcher provides. Extract methodology, strains, imaging parameters, key findings, gaps. Returns structured summary. |
| `search_strains` | Query WormBase / CGC / FlyBase for available strains and reporters. Input: gene name, reporter type, organism. Returns strain names, genotypes, availability. |

### Lab Context Tools

| Tool | Description |
|------|-------------|
| `query_lab_history` | Search past sessions, campaigns, learnings in our data stores. "Have we imaged this reporter before? What parameters worked?" |
| `check_hardware_capability` | "Can our diSPIM resolve this?" Check against hardware specs — resolution, speed, wavelengths, stage limits. |

### Interaction Tools

| Tool | Description |
|------|-------------|
| `ask_researcher` | Ask a question using the interactive picker (existing `ask_user_choice` mechanism). For domain expertise, lab-specific knowledge, or decisions. |

### Planning Tools

| Tool | Description |
|------|-------------|
| `propose_plan` | Present a structured plan for review. Renders the full plan document in the TUI. The researcher can approve, request changes, or iterate. |
| `create_campaign` | Commit a campaign (or sub-campaign/phase) to the data model. |
| `create_plan_item` | Create a plan item (imaging, bench, genetics, analysis, decision_point) with full spec. |
| `update_plan_item` | Modify an existing plan item (status, spec, description). |
| `link_plan_items` | Set dependencies between plan items ("sax-3 imaging depends on genetic cross"). |

---

## Conversation Flow

Not a rigid sequence — a natural conversation with phases. The agent drives the conversation but follows the researcher's lead.

### Phase 1: UNDERSTAND

The agent asks questions to understand the scientific goal.

```
Researcher: "I want to investigate nerve ring formation"

Agent: "Interesting — nerve ring assembly is a great target for
live imaging. A few questions to help me design the right approach:

What's driving this? Is this:
  ● Preliminary exploration (seeing if we can image it)
  ● Building on published work (extending a specific paper)
  ● Hypothesis-driven (testing a specific mechanism)
  ● Data collection for a grant/paper"
```

Follow-up questions depending on the answer:
- "What reporters do you have available?"
- "Are you interested in wild-type characterization, or do you have mutants in mind?"
- "What's your timeline — weeks or months?"

### Phase 2: RESEARCH

The agent searches literature and lab history to inform the design.

```
Agent: [searches literature for "nerve ring formation C. elegans
live imaging"]

"I found some relevant work:

- Yeh et al. 2018 used a rab-3p::GFP reporter on spinning disk
  confocal, 1-min intervals through the fold stages. They could
  resolve individual pioneer axons.

- Rapti et al. 2017 showed UNC-6/Netrin and SAX-3/Robo are
  required for proper ring positioning — sax-3 mutants have
  displaced nerve rings.

- Your lab imaged rab-3p::GFP in session 20260201 — the signal
  looked decent at comma stage but you only captured 2 timepoints.

Based on this, I think we can adapt the Yeh approach for diSPIM.
Our temporal resolution is better (we can do sub-minute intervals)
and we get true 3D, which they didn't have."
```

### Phase 3: DESIGN

The agent proposes a complete experimental plan.

```
Agent: "Here's what I'm thinking — a 4-phase plan over about
8 weeks..."

[renders structured plan document — see Plan Document section]

"A few design decisions I made:
- I chose rab-3p::GFP over specific neuron markers because it
  shows the full ring structure
- Adaptive intervals (slow early, fast during fold stages) to
  minimize photobleaching while catching the critical window
- sax-3 as the perturbation because it has a clean published
  phenotype we can validate against
- 5 embryos per session to account for occasional arrests

What do you think? Anything you'd change?"
```

### Phase 4: REFINE

Iterate based on researcher feedback.

```
Researcher: "We don't have the sax-3 strain, but we have unc-6.
Also, can we skip the EM?"

Agent: "Good call on unc-6 — it's upstream in the same pathway,
and the nerve ring phenotype is well-characterized (Wadsworth et al.).

I'll swap sax-3 for unc-6 and remove the EM validation. Instead,
we can strengthen the fixed imaging validation with a double stain.

[updates plan, re-renders]

The genetic cross is simpler too — you already have the reporter
and the unc-6 mutant, so you just need to cross them. About 10
days for a homozygous line."
```

### Phase 5: COMMIT

Write the plan to the data model.

```
Agent: "Plan looks good. I'll save it now.

Created:
  Campaign: 'Nerve ring formation study' (nrf-2026)
    Phase 1: Reporter validation (2 items)
    Phase 2: Data collection (6 items)
    Phase 3: Perturbation (4 items)
    3 decision points

Your first session is a pilot — test rab-3p::GFP visibility
on the diSPIM. Want to schedule it?"
```

---

## Data Model

### PlanItem

The core tracking unit. Every task in the plan — imaging or not — is a PlanItem.

```python
@dataclass
class PlanItem:
    """A single item in an experimental plan."""

    id: str
    campaign_id: str  # Which campaign/phase
    type: str  # imaging, bench, genetics, analysis, decision_point
    title: str  # "Pilot — rab-3p::GFP visibility test"
    description: Optional[str] = None  # Detailed notes, protocols, what to watch for
    status: str = "planned"  # planned → in_progress → completed | skipped | blocked
    depends_on: List[str] = field(default_factory=list)  # PlanItem IDs
    outcome: Optional[str] = None  # What happened (filled after completion)

    # Specifications (type-dependent)
    imaging_spec: Optional[ImagingSpec] = None  # if type == "imaging"
    bench_spec: Optional[BenchSpec] = None  # if type in (bench, genetics, analysis)

    # Linking
    planned_session_id: Optional[str] = None  # → PlannedSession (for imaging items)
    session_id: Optional[str] = None  # → Actual session (once executed)
    inherit_from: Optional[str] = None  # PlanItem ID to inherit spec from

    # Ordering
    phase_order: int = 0

    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
```

### ImagingSpec

Complete specification for an imaging session. Everything needed to auto-configure.

```python
@dataclass
class ImagingSpec:
    """Complete specification for a planned imaging session."""

    # ── Sample ──────────────────────────────────────────
    strain: Optional[str] = None  # "OH904"
    genotype: Optional[str] = None  # "otIs355[rab-3p::2xNLS::TagRFP]"
    reporter: Optional[str] = None  # "rab-3p::GFP (pan-neuronal)"
    sample_prep: Optional[str] = None  # "Standard egg prep, poly-lysine pads"
    temperature_c: Optional[float] = None  # 20.0
    num_embryos: Optional[int] = None  # 4

    # ── Acquisition ─────────────────────────────────────
    num_slices: Optional[int] = None  # 80
    exposure_ms: Optional[float] = None  # 10.0
    laser_wavelength_nm: Optional[int] = None  # 488
    laser_power_pct: Optional[float] = None  # 10.0
    galvo_amplitude: Optional[float] = None  # 8.0
    piezo_amplitude_um: Optional[float] = None  # 50.0

    # ── Timing ──────────────────────────────────────────
    interval_s: Optional[int] = None  # 180
    adaptive_intervals: Optional[Dict[str, int]] = None
    # e.g. {"early_to_comma": 300, "comma_to_2fold": 60, "after_2fold": 180}

    # ── Developmental Window ────────────────────────────
    target_window: Optional[str] = None  # "comma → pretzel"
    start_stage: Optional[str] = None  # "comma"
    stop_condition: Optional[str] = None  # "pretzel"
    estimated_duration_h: Optional[float] = None  # 4.0

    # ── Detection ───────────────────────────────────────
    detectors: Optional[List[str]] = None  # ["comma", "pretzel"]
    pre_terminal_speedup: Optional[bool] = None  # True

    # ── Validation ──────────────────────────────────────
    success_criteria: Optional[str] = None
    # "Nerve ring visible in ≥3/4 embryos at 2-fold"
    comparison_to: Optional[str] = None
    # "Compare to WT session 1 — matched conditions"
```

### BenchSpec

Specification for non-imaging tasks (bench work, genetics, analysis).

```python
@dataclass
class BenchSpec:
    """Specification for bench/genetics/analysis tasks."""

    protocol: Optional[str] = None  # "Standard chemotaxis assay"
    reagents: Optional[List[str]] = None  # ["anti-UNC-33", "secondary 568"]
    strains: Optional[List[str]] = None  # ["OH904", "N2"]
    target_genotype: Optional[str] = None  # "unc-6(ev400); otIs355"
    estimated_days: Optional[int] = None  # 14
    success_criteria: Optional[str] = None  # "Homozygous GFP+ line established"
    notes: Optional[str] = None
```

### Parameter Inheritance

When multiple sessions should use "same settings," the spec supports inheritance:

```python
# Session 1: fully specified
session_1_item = PlanItem(
    imaging_spec=ImagingSpec(
        strain="OH904", num_slices=80, exposure_ms=10.0,
        interval_s=300, stop_condition="pretzel", ...
    )
)

# Sessions 2-5: inherit from session 1, override only what differs
session_2_item = PlanItem(
    inherit_from=session_1_item.id,
    imaging_spec=ImagingSpec(num_embryos=5),  # only override this
)
```

When loading, the agent resolves inheritance: merge parent spec with local overrides. If the researcher changes exposure in session 1's spec after the pilot, it propagates to all inherited sessions automatically.

---

## Plan Document Rendering

What `propose_plan` / `/plan status` renders in the TUI:

```
═══════════════════════════════════════════════════
 EXPERIMENTAL PLAN: Nerve Ring Formation
 Campaign: nrf-2026
═══════════════════════════════════════════════════

Goal: Characterize nerve ring assembly dynamics
      in live C. elegans embryos

Hypothesis: Nerve ring pioneer axons establish
the ring structure by the 2-fold stage, with
guidance dependent on UNC-6/Netrin signaling

── Phase 1: Reporter Validation (Week 1-2) ───────

 [IMAGING] Pilot — rab-3p::GFP visibility test
   Strain:     OH904 (rab-3p::GFP)
   Params:     80 slices, 10ms @ 488nm 10%
   Timing:     3min interval, comma → pretzel
   Embryos:    4
   Criteria:   Can we resolve individual axon tracts?

 [BENCH] Fixed staining — anti-UNC-33
   Protocol:   Standard immunostaining
   Reagents:   anti-UNC-33 primary, Alexa568 secondary
   Purpose:    Independent validation of ring timing

 [DECISION] Phase 1 gate
   Question:   Can we see nerve ring forming live?
   If yes:     Proceed to Phase 2
   If no:      Switch reporter or try confocal

── Phase 2: Data Collection (Week 3-6) ───────────

 [IMAGING] WT session 1-5 — 5 embryos each
   Strain:     OH904 (rab-3p::GFP)
   Params:     80 slices, 10ms @ 488nm 10%
   Timing:     5min until comma, 1min through 2-fold
   Stop:       pretzel
   Detectors:  comma, pretzel (auto-enabled)
   Speedup:    Pre-terminal at pretzel

 [IMAGING] Matched controls — unperturbed
   Same strain, same conditions, no imaging
   Purpose:    Verify imaging doesn't affect development

 [GENETICS] Cross OH904 × unc-6(ev400)
   Target:     unc-6(ev400); otIs355
   Timeline:   ~10 days for homozygous line
   Selection:  GFP+ F2s, confirm unc phenotype

── Phase 3: Perturbation (Week 7-10) ─────────────

 [IMAGING] unc-6 mutant time series
   Strain:     unc-6(ev400); otIs355
   Params:     Inherit from WT sessions
   Depends on: Genetic cross completion

 [BENCH] Chemotaxis assay — post-hatch
   Protocol:   Standard chemotaxis plate assay
   Compare:    Imaged vs unimaged, WT vs unc-6
   Purpose:    Functional validation

 [ANALYSIS] Quantify ring geometry WT vs unc-6
   Metrics:    Ring diameter, symmetry, completion time
   Method:     3D segmentation of nerve ring from volumes

── Decision Points ────────────────────────────────

 After Phase 1:  Can we see it? → proceed or switch
 After Phase 2:  Timing reproducible? → power analysis
 After Phase 3:  unc-6 phenotype? → expand or publish
```

---

## Progress Tracking

### At Startup

The gap assessment checks active plans and reports status:

```
Welcome back. Here's where we stand on the nerve ring study:

Phase 1 (Reporter Validation): ✓ Complete
  ✓ Pilot imaging — GFP visible, good SNR at comma/1.5-fold
  ✓ Fixed staining — timing matches published data

Phase 2 (Data Collection): In Progress — 2 of 5 sessions done
  ✓ WT session 1 — 4 embryos, all reached pretzel
  ✓ WT session 2 — 4 embryos, 1 arrested early
  ○ WT session 3 — ready to go
  ○ WT session 4
  ○ Controls
  ⏳ Genetic cross — started 5 days ago, check F2s?

Phase 3 (Perturbation): Blocked on genetic cross

Is today WT session 3?
```

### During a Session

When the agent finishes a timelapse:

```
→ Links session_id to the PlanItem
→ Marks PlanItem status = "completed"
→ Records outcome ("4 embryos imaged, 3 reached 2-fold,
   nerve ring visible in all 3")
→ Updates Campaign progress ("3/5 sessions")
→ Checks if phase is complete
→ If decision point dependencies are met, prompts researcher
```

### Non-Imaging Updates

The researcher reports bench work progress conversationally:

```
Researcher: "The cross is done, I have homozygous unc-6; rab-3p::GFP"

Agent:
  → Updates PlanItem "Cross OH904 × unc-6" → completed
  → outcome = "Homozygous line established"
  → Checks dependencies: Phase 3 imaging items are now unblocked
  → "Great — that unblocks the unc-6 imaging sessions in Phase 3.
     You still have 3 WT sessions to finish first. Want to continue
     with WT session 3 today?"
```

### Decision Points

When all dependencies of a decision point are met, the agent raises it:

```
Both Phase 1 items are done. Decision needed:

  "Can we resolve individual axon tracts with rab-3p::GFP on diSPIM?"

Based on your pilot data:
  - Session 20260215: SNR was decent at comma, degraded at pretzel
  - Fixed staining confirmed nerve ring timing at ~440 min

Options:
  ● Proceed to Phase 2 (signal is good enough)
  ● Switch reporter (try a sparser marker like AIY::GFP)
  ● Optimize parameters first (try higher laser, shorter interval)
  ● Modify the plan
```

The decision outcome is recorded on the PlanItem and downstream phases adjust.

### Auto-Configuration at Session Start

When the researcher starts a planned imaging session, the agent pre-loads everything from the ImagingSpec:

```
Starting: WT session 3 — Nerve Ring Formation Study, Phase 2

Pre-loaded from plan:
  Strain:       OH904 (rab-3p::GFP, pan-neuronal)
  Embryos:      5 (bumped from 4 after session 2 arrest)
  Slices:       80 (elongation during fold stages)
  Exposure:     10ms @ 488nm, 10% power
  Intervals:    5 min until comma, then 1 min through 2-fold
  Stop:         After pretzel stage
  Detectors:    comma, pretzel (auto-enabled)
  Speedup:      Pre-terminal speedup at pretzel
  Temperature:  20°C

  Success criteria: Nerve ring visible in ≥3 embryos

  Matched to: WT sessions 1 & 2 (same parameters)
  Note from session 2: "1 embryo arrested at bean stage —
  imaging 5 embryos as backup"

Ready to detect embryos and start?
```

### Learning Across Sessions

After each session, the agent can propose plan updates:

```
Session 2 complete. 3 of 4 embryos reached pretzel successfully.
One arrested at bean stage (possible mounting issue).

Suggestion: For remaining sessions 3-5, image 5 embryos instead
of 4 to account for occasional arrests.

Update the plan?
  ● Yes, bump to 5 embryos per session
  ● No, keep at 4
  ● Let me think about it
```

---

## Plan Evolution

Plans change. The agent handles modifications naturally:

```
Researcher: "Actually, let's skip the behavioral assay and add
             a temperature series instead"

Agent:
  → PlanItem "Chemotaxis assay" → status = "skipped"
  → Creates new PlanItem(type="imaging", title="Temperature series
    — 25°C matched conditions")
  → Records an Observation: "Plan modified: replaced behavioral
    assay with 25°C temperature series per researcher request"
  → "Updated. I've replaced the chemotaxis assay with a 25°C
     temperature series in Phase 3. I'll use the same imaging
     parameters but at 25°C — development will be ~30% faster,
     so I've adjusted the expected duration. The rest of the
     plan is unchanged."
```

No version history system needed — the current state of PlanItems is the plan. Observations record why things changed.

---

## Storage Schema

> **Note**: Storage has migrated from SQLite to file-based (Gently3).
> Plan items live in `D:/Gently3/agent/campaigns/{id}_{slug}/plan/current.yaml`.
> The schema below shows the original SQL design for reference; the
> `FileContextStore` provides the same API over YAML files.

Original table in the context database (`agent_mind.db`):

```sql
CREATE TABLE IF NOT EXISTS plan_items (
    id TEXT PRIMARY KEY,
    campaign_id TEXT NOT NULL,
    type TEXT NOT NULL,              -- imaging, bench, genetics, analysis, decision_point
    title TEXT NOT NULL,
    description TEXT,
    status TEXT DEFAULT 'planned',   -- planned, in_progress, completed, skipped, blocked
    outcome TEXT,                    -- What happened (post-completion)
    spec TEXT,                       -- JSON: ImagingSpec or BenchSpec
    inherit_from TEXT,               -- PlanItem ID to inherit spec from
    planned_session_id TEXT,
    session_id TEXT,
    phase_order INTEGER DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    FOREIGN KEY (campaign_id) REFERENCES campaigns(id),
    FOREIGN KEY (planned_session_id) REFERENCES planned_sessions(id),
    FOREIGN KEY (inherit_from) REFERENCES plan_items(id)
);

CREATE TABLE IF NOT EXISTS plan_item_dependencies (
    item_id TEXT NOT NULL,
    depends_on_id TEXT NOT NULL,
    PRIMARY KEY (item_id, depends_on_id),
    FOREIGN KEY (item_id) REFERENCES plan_items(id),
    FOREIGN KEY (depends_on_id) REFERENCES plan_items(id)
);
```

Spec resolution on load:

```python
def resolve_spec(item: PlanItem, all_items: Dict[str, PlanItem]) -> ImagingSpec:
    """Merge inherited spec with local overrides."""
    if not item.inherit_from:
        return item.imaging_spec

    parent = all_items[item.inherit_from]
    parent_spec = resolve_spec(parent, all_items)  # Recursive

    # Merge: local fields override parent fields
    merged = dataclasses.replace(parent_spec)
    if item.imaging_spec:
        for field in dataclasses.fields(ImagingSpec):
            local_val = getattr(item.imaging_spec, field.name)
            if local_val is not None:
                setattr(merged, field.name, local_val)
    return merged
```

---

## Example: Full Plan for Nerve Ring Formation

### Campaign Hierarchy

```
Campaign: "Nerve ring formation study" (id: nrf-2026)
  ├── Campaign: "Phase 1 — Reporter validation" (parent_id: nrf-2026)
  │     ├── PlanItem(type="imaging", title="Pilot — rab-3p::GFP")
  │     │     imaging_spec: ImagingSpec(strain="OH904", num_slices=80, ...)
  │     │     → PlannedSession(acquisition_params=...)
  │     ├── PlanItem(type="bench", title="Fixed staining — anti-UNC-33")
  │     │     bench_spec: BenchSpec(reagents=["anti-UNC-33", ...])
  │     └── PlanItem(type="decision_point", title="Phase 1 gate")
  │           depends_on: [pilot_id, staining_id]
  │
  ├── Campaign: "Phase 2 — Data collection" (parent_id: nrf-2026)
  │     ├── PlanItem(type="imaging", title="WT session 1", id: wt1)
  │     │     imaging_spec: ImagingSpec(full spec)
  │     ├── PlanItem(type="imaging", title="WT session 2")
  │     │     inherit_from: wt1
  │     ├── PlanItem(type="imaging", title="WT session 3")
  │     │     inherit_from: wt1
  │     ├── PlanItem(type="imaging", title="WT session 4")
  │     │     inherit_from: wt1
  │     ├── PlanItem(type="imaging", title="WT session 5")
  │     │     inherit_from: wt1
  │     ├── PlanItem(type="imaging", title="Matched controls")
  │     │     inherit_from: wt1, imaging_spec: ImagingSpec(comparison_to="WT sessions")
  │     └── PlanItem(type="genetics", title="Cross OH904 × unc-6")
  │           bench_spec: BenchSpec(target_genotype="unc-6; otIs355", ...)
  │
  └── Campaign: "Phase 3 — Perturbation" (parent_id: nrf-2026)
        ├── PlanItem(type="imaging", title="unc-6 mutant time series")
        │     inherit_from: wt1
        │     imaging_spec: ImagingSpec(strain="unc-6; otIs355")
        │     depends_on: [cross_item_id]
        ├── PlanItem(type="bench", title="Chemotaxis assay")
        ├── PlanItem(type="analysis", title="Quantify ring geometry")
        │     depends_on: [wt_sessions..., mutant_session_id]
        └── PlanItem(type="decision_point", title="Phase 3 gate")
              description: "unc-6 phenotype detected? → expand or publish"
```

---

## Integration with Execution Mode

When the researcher exits plan mode and starts executing:

1. The startup wizard checks for planned sessions matching today
2. If a match is found, pre-loads the ImagingSpec:
   - Configures acquisition parameters
   - Enables detectors
   - Sets stop conditions and adaptive intervals
3. The agent's session context includes the plan status
4. After the session, the agent updates the PlanItem and campaign progress
5. If a decision point is ready, the agent raises it at the next session start

The plan lives in the background during execution mode — the agent is always aware of where the researcher is in their experimental plan.

---

## Summary

Plan mode adds a **scientific reasoning layer** upstream of microscope control. The agent becomes a research collaborator that:

1. Helps design experiments from scientific questions
2. Reasons about strains, controls, validations, and non-imaging work
3. Produces structured plans with complete imaging specifications
4. Tracks progress across sessions and weeks
5. Manages dependencies and decision points
6. Auto-configures the microscope when it's time to execute
7. Learns from completed sessions and suggests plan adjustments

The key principle: **the plan is a living document**, not a one-time output. The agent maintains it, tracks it, and uses it to be a better collaborator over the lifetime of a research project.

---

## Plan-Session Integration — How Plan Context Flows

The plan is only useful if the agent knows about it during a session. This section documents how plan context is resolved, set, and used at runtime.

### Startup Flow

```
launch_gently.py
|
+-- load config, organism, hardware
+-- FileStore(D:/Gently3)
+-- session picker (Ink TUI or --resume)
+-- connect to device layer
|
+-- MicroscopyAgent(session_id, store)
|   +-- ExperimentState()
|   |     active_plan_item_id: None (or restored from snapshot)
|   +-- SessionManager.create/resume()
|   |     restores active_plan_item_id from session snapshot
|   +-- TimelapseOrchestrator()
|   +-- _update_system_prompt()
|
+-- FileContextStore(D:/Gently3/agent)
+-- agent.set_context_store(cs)
|   +-- AgentMemory(cs, session_id)
|         active_plan_item_id: None
|
+-- bridge.init_wizard(cs, claude)
|
+-- spawn Ink TUI --> connects via WebSocket
|
+-- /ws/agent handler:
    |
    +-- if wizard.needed (first launch only):
    |   +-- run wizard (asks organism, etc.)
    |
    +-- else (returning user):
    |   +-- bridge.get_session_briefing()
    |       |
    |       +-- PLAN CONTEXT RESOLUTION:
    |       |   |
    |       |   +-- if session resumed with active_plan_item_id:
    |       |   |     restore onto AgentMemory (already on ExperimentState)
    |       |   |
    |       |   +-- else: memory.resolve_plan_context()
    |       |       +-- scan active campaigns
    |       |       +-- find unblocked imaging items via dependency graph
    |       |       |
    |       |       +-- 1 item: auto-set active_plan_item_id
    |       |       |           link session to campaign
    |       |       |           briefing: "Next up: [title] + full spec"
    |       |       |
    |       |       +-- N items: briefing lists all with specs
    |       |       |            "Which imaging task today?"
    |       |       |
    |       |       +-- 0 items: minimal briefing
    |       |
    |       +-- invalidate prompt cache
    |
    +-- system prompt rebuilt (next message):
    |   +-- get_awareness_summary() includes:
    |       |
    |       |   # Your Memory
    |       |   - Active campaigns: "Nerve ring study" (3/8)
    |       |
    |       |   ## Active Plan Item: WT session 3
    |       |   Campaign: Nerve ring formation study
    |       |     Strain: OH904
    |       |     Slices: 80
    |       |     Exposure: 10ms
    |       |     Laser: 488nm at 10%
    |       |     Interval: 300s
    |       |     Stop condition: pretzel
    |       |     Detectors: comma, pretzel
    |       |
    |       |   Use this spec when configuring embryos and
    |       |   starting the timelapse.
    |
    +-- REPL loop
        agent knows what it's here to do
```

### Plan Mode Exit Flow

When the user creates a new plan in plan mode and exits:

```
agent.exit_plan_mode()
|
+-- memory.resolve_plan_context()
|   +-- scans newly created campaign
|   +-- finds unblocked imaging items
|
+-- if 1 item: auto-set active_plan_item_id
|              link session to campaign
|              return "Back to run mode. Active plan item: [title]"
|
+-- if N items: return "N imaging tasks ready: ... Which one?"
|
+-- if 0 items: return "Back to run mode."
|
+-- invalidate prompt cache
+-- rebuild system prompt (now includes ImagingSpec)
```

### Where active_plan_item_id Lives

```
ExperimentState.active_plan_item_id     in-memory, serialized to session snapshot
AgentMemory.active_plan_item_id         in-memory, drives system prompt injection
```

Both are set together. ExperimentState is the persistence layer (survives session
save/resume). AgentMemory is the prompt layer (drives what the agent sees).

### Data Flow: Plan Item --> Microscope

```
ImagingSpec (in plan)
  |
  +-- [system prompt] agent sees strain, slices, exposure, etc.
  |
  +-- [detect_embryos] agent offers to configure with plan settings
  |
  +-- [execute_plan_item or start_timelapse]
  |     +-- resolve ImagingSpec
  |     +-- configure embryo params (num_slices, exposure_ms)
  |     +-- enable detectors
  |     +-- set stop condition and interval
  |     +-- start timelapse
  |     +-- mark plan item in_progress
  |
  +-- [ACQUISITION_COMPLETED event]
        +-- (future: auto-complete plan item with outcome)
        +-- dependency graph advances
        +-- next startup shows next unblocked item
```
