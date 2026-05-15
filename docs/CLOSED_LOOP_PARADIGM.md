# Closed-Loop Paradigm: Notes on the Discussion

This document captures the design conversation that produced everything on the
`paradigm/closed-loop` branch: the schema split, the Map-as-embryo-home work,
the operator-action vocabulary, the eval substrate (capture / replay /
decisions / shadow), and the trajectory the system is on. It is a
distillation, not a transcript — a future-self / new-collaborator reference
for *why* this code looks the way it does and *where it is going*.

---

## 1. The Original Friction

The conversation started from a small, concrete observation by the operator:

> "It feels awkward that the operator has to go between the chat in the TUI
> and the viz server… or even to chat about detecting embryos."

That awkwardness is a symptom, not a defect. It surfaces a deeper design
question: **what is the orchestrator (the agent) actually for?** Today the
orchestrator does at least four jobs at once, and one of them — *tool router*
— is the one creating the friction.

### The four orchestrator roles

| Role | Description | Replaceable by a button? |
| --- | --- | --- |
| 1. Tool router | "Detect embryos" → `detect_embryos()` call | **Yes** — this is the friction surface |
| 2. Workflow runner | Timelapses, multi-embryo plans, perception loops | No |
| 3. Domain reasoner | Knows microscopy, embryos, safety constraints | No |
| 4. Session memory | Coherent narrative of what happened and why | No |

Routing a single click through chat for a routine action is the system
fighting against its own users. Routing a multi-step scientific decision
through Claude is using the right tool for the right job. The paradigm here
is: **shrink role 1 to its essentials, keep roles 2–4 first-class, and let
the UI carry the rest.**

---

## 2. Web ↔ Chat Reconciliation Patterns

Four ways to relate the web UI and the chat orchestrator. Each has a
distinct world model property:

### A. Chat-only intent (the old default)

Every action originates in chat. The web is observation + delegated subtasks
(e.g. the marking canvas is a delegation the orchestrator triggers).

* Cleanest record.
* Worst friction.
* Orchestrator's world model is "complete" because every change passes
  through it.

### B. Two parallel command surfaces

Operator clicks in web, web acts directly; orchestrator finds out by polling
state or doesn't find out at all.

* Lowest friction.
* Orchestrator's world model **drifts from reality** — fatal for role 4
  (session memory) and dangerous for role 3 (safety reasoning).

### C. Web acts, orchestrator subscribes *(the chosen direction)*

Operator clicks → web performs the action **and** publishes an event
(`OPERATOR_*`) → orchestrator's session memory ingests it.

* Chat log shows only human conversation.
* Orchestrator's working context shows chat + events as a single timeline.
* Phase 7 (operator events vocabulary + reactive candidate) is the first
  installment of this pattern.

### D. Cross-pattern hybrid

Different action classes use different patterns. Heavy / novel / composite
actions use chat (A); routine / clickable / contextual actions use web (C).
This is what the system actually drifts toward; pattern C is the substrate
that makes it possible.

The orchestrator's job shifts from being **a funnel for action** to being **a
brain that knows what's happening on every surface**.

---

## 3. The "Turn" is Wrong; the "Decision Moment" is Right

Chat-AI literature reasons in *turns* (user message → assistant response).
That model imports an assumption that does not hold here: the human is at
the keyboard continuously. In a microscopy experiment running 12+ hours, the
human checks in once, twice, maybe ten times. The agent is autonomous in
between.

The right unit is a **decision moment**, triggered by:

1. **User message** — rare, interrupting (classic chat turn).
2. **Critical event** — error, safety violation, lost focus, perception
   anomaly. Wake immediately; decide to act / abort / escalate.
3. **Phase boundary** — between timepoints, between embryos. Built-in
   checkpoint: review state, decide whether to continue.
4. **Periodic checkpoint** — every N minutes if nothing else happened.
   Catches slow drifts.

Between moments the agent is asleep. Plans execute autonomously. Events
accumulate on the bus and in the world model. When the next decision
moment fires, the agent reads:

* The trigger (why am I waking up?)
* The world snapshot (NOW state)
* The events digest (what happened since last wake)
* The conversation history (which might be hours old and less relevant
  than usual)

This is closer to a **supervisory controller** than a chat partner. The
conversation history matters less than usual; what matters more is the
**flight log** (events) plus the **current state snapshot**.

### Trigger model — concrete

A small router (in code, not Claude) sits between the bus and the brain:

```
   user input ─┐
   event bus ──┼─► wake-router ──► (compose context) ──► claude.messages.create
   schedule  ──┘
```

The router's responsibilities:
* Subscribe to a whitelist of "wake-worthy" event types.
* Hold a debounce / coalescing buffer (so a burst of events becomes one
  wake).
* Keep a heartbeat schedule (every N minutes if no other trigger fired).
* On wake, package: trigger, world snapshot, events digest, recent
  conversation tail.
* Surface the package to the brain.

The brain stays the brain (Claude). The router is cheap, deterministic,
debuggable code. It's the **meta-orchestrator** the operator mentioned —
**not as another LLM**, but as a control surface.

### Phase boundaries: hand-back vs subscribe

Two designs for letting the brain look in mid-plan:

* **Plan hands control back** at well-known points (between embryos, every
  5 timepoints). Cheaper, predictable, slightly less reactive.
* **Plan never pauses; brain subscribes to plan events** ("perception
  complete for embryo 3"). More reactive, more plan-coupling.

The first one composes better with the supervisory-controller framing and
is the recommended starting point.

### Idle ticks

If 30 min pass with no event and no user, should the agent wake to verify
everything's OK? Default to **yes — periodic ticks with a high action
threshold.** Most ticks should result in the agent doing nothing. The
purpose is catching slow drifts (focus, sample state, hardware
degradation) that don't trigger their own events.

---

## 4. World Model — Tiered, Not Monolithic

A common mistake is "summarise everything every turn." Better is a tiered
model where different freshness/density tiers carry different cadence
costs.

### Tier 1 — World snapshot

Structured, ~30 lines, computed from in-memory state (not events), every
wake.

Includes: live stage XY/Z, current session id, embryo list with
calibration state, current plan, acquisition status, recent operator
actions (one-line summary).

Cheap to build, fresh every time. Already mostly present in the
codebase — `agent.experiment.get_summary()` plus the cached
`DEVICE_STATE_UPDATE` payload is 80% of this.

### Tier 2 — Recent-events digest

Hand-written formatter over the events bus, filtered to wake-worthy types,
inserted as a system note at each wake.

Shape: `"Since last response: operator added embryo 4 via Map at 14:32; calibration completed for embryo 2; one perception trace pending."`

Hand-written because LLM summarisation here adds latency, cost, and
non-determinism for low value. Events are already structured.

### Tier 3 — Pull tools

For when reasoning needs depth: `get_recent_perceptions(embryo_id, n=5)`,
`get_session_timeline()`, `get_learnings(campaign_id)`, etc. The agent
calls these when it wants the detail.

### Tier 4 — Optional LLM summariser

Reserved for genuinely natural-language streams that resist rule-based
compression: accumulated CV reasoning chains, narrative observations,
cross-session learnings. Use a smaller, faster Claude model (Haiku is the
natural fit). Run lazily, when a tier-3 tool asks for "summarise the last
30 min for embryo 3."

### Why this shape

Decision moments are **rare** in autonomous mode. Token budget per wake
can be generous (it's mostly idle compute). What matters more than budget
is **cadence of waking** — saving 200 tokens per turn doesn't help if
you're waking up at the wrong moments.

---

## 5. Testing — Where Most Projects Fail

You cannot iterate on agent architecture without a way to compare
architectures. Microscopy makes this hard:

* Physical, non-deterministic, non-replayable in the trivial sense.
* "Correct" is fuzzy — biological judgements rarely have ground truth.
* Slow feedback (a timelapse takes hours).
* Can't always reset to a clean state (samples are consumed).

Five testing primitives, ranked by payoff per unit work (this ordering
informed Phase 6's build order):

### 5.1 Event replay *(built — Phase 6a/6b)*

Capture the full event stream during a real run. Offline, replay it
through any candidate architecture. Diff its decisions against
production's. **Foundation** — without it, every change to the
orchestrator is a flight test.

### 5.2 Shadow mode *(built — Phase 6d)*

During a real experiment, candidate architectures run alongside
production. They see the same events but their decisions are *logged,
not enacted*. Unique value over pure replay: shadow agents experience
real temporal cadence, so timing-sensitive things (drift, races) are
caught.

### 5.3 Synthetic event sequences

Hand-crafted streams: cascading errors, ambiguous perception,
contradictory operator actions, focus drift, network drop mid-acquisition.
Stress / chaos testing. The orchestrator is correct if it doesn't do
something catastrophic — much easier to score than biological
correctness.

Trivially built on top of 5.1 — write a `jsonl` by hand, replay it.

### 5.4 Decision-level micro-benchmarks

Specific judgements — "given this perception result and these recent
observations, should the agent re-focus?" — captured as
(input → expected decision) pairs labelled by a biologist. Regression
suite. Cheap with biologist time, expensive to bootstrap, very valuable
once you have a few hundred.

### 5.5 Multi-agent A/B in production

Two embryos in the same dish, one supervised by architecture A and one
by B (both honouring the firmware fence). Compare biological outcomes.
Slow (one timelapse per data point), but the **only thing that measures
biological correctness end-to-end.**

---

## 6. Embryo Schema: Coarse vs Fine

Foundational and quietly important. Each embryo carries:

* `position_coarse` — set by bottom-camera detection or manual Map
  placement. Always present.
* `position_fine` — set later by SPIM-objective alignment (workflow not
  yet built). Initially `{}`.
* `stage_position` — a *derived property*: `fine if fine else coarse`.
  Downstream motion / perception keeps reading this and stays agnostic
  about which calibration stage we're in.

This is the seed for a broader idea: **measurements have provenance and
calibration state**. The same embryo at the same nominal XY can have
different "true" positions depending on which sensor sighted it. Encode
that explicitly so any downstream decision can ask *"how confident is
this position?"* without needing to know the whole calibration history.

When the operator drags an embryo on the Map, the PUT clears `fine` —
overriding the sighting invalidates any SPIM-derived fine alignment
derived from the old coarse. `OPERATOR_EDITED_EMBRYO` carries
`fine_position_invalidated` so the candidate / future controller can
schedule a re-alignment without inferring it.

---

## 7. The Map as Collaborative World Model

The Devices > Map page is more than visualisation. It is the **first
collaborative surface** between operator and orchestrator: both can read
the embryo list; both can mutate it. The orchestrator subscribes; the
operator clicks.

Visual semantics matter:

* Coarse-only embryo → outlined ring + number. *Provisional.*
* SPIM-fine-calibrated → filled disc + number. *Committed.*

Calibration state is then visible at a glance across the slide — the
operator can scan and see "embryo 3 still needs alignment" without
opening anything.

The pick-up / drop interaction (Phase 5) deliberately rejects
click-to-add: the Map is a schematic, not a satellite view. Adding a
new sighting without a visual reference is guessing. New embryos go
through the bottom-camera marking canvas. The Map is for **editing what
already exists**.

### Future arc

* **Annotations beyond position**: operator marks "this is the control",
  "this one is dead, skip", "I think this is in 2-cell stage". These
  become first-class scientific observations through additional
  `OPERATOR_*` events.
* **Satellite tile**: render the live bottom-camera frame as an overlay
  on the Map at the current stage XY, scaled by um_per_pixel. Inside
  that tile, click-to-add becomes meaningful (you can see what you're
  picking). Outside, the Map stays schematic.

---

## 8. Revolutionary Trajectories

Some of these are reasonable extensions; some are genuinely new.

### 8.1 Plans-as-goals, not scripts

Operator specifies "characterise gut development for these four
embryos." Orchestrator translates this into a continuously adapted
imaging plan that changes based on what perception sees mid-run. The
plan isn't a fixed script handed to Bluesky — it's a negotiation the
orchestrator keeps in flight, with the world model as the substrate
for adaptation.

Requires: tier-1 + tier-2 world model, decent perception loop, a way
to express goals as predicates over the world model.

### 8.2 Compounding cross-session learning

`agent/learnings/` already exists. Today it's barely used. With replay
+ shadow, an architecture that proposes priors ("embryos at 3-fold
typically need slower piezo") becomes **A/B testable across sessions**.
Improvement gets *measurable*, which is the unlock — most "smart
microscopy" today is shallow because it has no measurement loop.

The right framing: each session is a **trial**, the orchestrator is the
**experimenter**, the world model is what carries learning between
trials.

### 8.3 Collaborative world model

The Map (operator edits embryos) is the first instance. Extend
everywhere:

* Operator annotates morphology on the Map → orchestrator updates
  hypothesis space.
* Operator marks a focus failure → orchestrator marks the calibration
  region as untrustworthy.
* Operator confirms a perception → orchestrator increases confidence in
  the perception predicate for similar inputs.

The point is making the operator's tacit knowledge **first-class data**
that the system can reason about, not just record.

### 8.4 Reverse-mode microscopy

"I want to know X — plan the imaging that answers X." The orchestrator
translates scientific goals into imaging plans. This is the
plans-as-goals idea taken to its conclusion: the operator describes
intent in scientific terms, the orchestrator owns the imaging strategy.

Tractable only once 8.1 and the goal language are built.

### 8.5 Continuous shadow / always-on critic

Run the production orchestrator + a shadow candidate continuously, and
log all decision divergences. Over weeks, the divergence log becomes a
**dataset of disagreements**. Each disagreement is either:

* Production was right, candidate was wrong → candidate needs a fix.
* Candidate was right, production was wrong → consider promotion or
  investigate why production picked differently.
* Both were defensible → annotate the case.

Free with the eval substrate (Phase 6); the only addition is a
divergence collator.

---

## 9. Concretely Built Today (`paradigm/closed-loop` branch)

| # | Commit | What |
| --- | --- | --- |
| 1 | `3e410581` | Schema split: `position_coarse` / `position_fine` / derived `stage_position`. |
| 2 | `617e54c9` | `ExperimentState.notify_embryos_changed()` observer → `EMBRYOS_UPDATE` broadcast. |
| 3 | `144d9fc9` | Map render layer — lavender rings (coarse) / discs (fine) / numbers. |
| 4 | `4fbb9edf` | `detect_embryos` flows through web Marking canvas; `auth.py` + `require_control`. |
| 5 | `8f6553e1` | Map pick-up / drop / Delete to edit embryos in place (control-gated PUT/DELETE). |
| 6 | `808fe813` | Side-fix: re-enable XY joystick at device-layer boot. |
| 7 | `f7a13d69` | Side-fix: image-anchored crosshair + scroll-to-zoom in camera panel. |
| 8 | `d69cc219` | `gently/eval/`: event capture / replay / shadow / decision log scaffolding. |
| 9 | `75d7c9db` | Production decision capture wired through `ConversationManager.call_claude`. |
| 10 | `0a97563e` | `OPERATOR_*` event vocabulary + `ReactiveCandidate` (first real shadow). |

### Per-session disk shape now

`D:\Gently3\sessions\{id}\`

* `events.jsonl` — captured event bus, telemetry-filtered.
* `decisions.jsonl` — every Claude turn (success + error).
* `interaction_log.jsonl` — pre-existing chat-shaped interactions.
* `timeline.jsonl` — pre-existing session timeline.
* Plus everything from the legacy FileStore layout.

### Eval CLI

`python scripts/replay_session.py {session_id_prefix} [--histogram] [--candidate {name}] [--real-time] [--time-scale N]`

---

## 10. What is *Not* Done Yet

These are the natural follow-ups; sketched as future-self breadcrumbs.

### Near-term (days)

* **Tier-1 world snapshot** as a system-prompt section the brain sees
  on every wake. Build the snapshot from `agent.experiment` plus the
  last cached `DEVICE_STATE_UPDATE`. ~30 lines of formatted prose, every
  wake.
* **Tier-2 events digest** — hand-written formatter that reads the
  bus's recent meaningful events (or the captured jsonl tail) and
  produces a one-paragraph "since last response" note.
* **Snapshot ingest into the brain's prompt** — `_update_system_prompt`
  already takes a `context_summary`; route tier-1 + tier-2 through it.

### Medium-term (weeks)

* **Wake-router** — the code-level scheduler from §3. Currently the
  brain only wakes on user message. Add: event-driven wake (subscribe
  to wake-worthy events), periodic-tick wake (heartbeat), debounce /
  coalesce buffer.
* **More operator events** — `OPERATOR_ANNOTATED_EMBRYO` ("this is the
  control", "skip, looks dead"), `OPERATOR_STARTED_TIMELAPSE`,
  `OPERATOR_INTERRUPTED_PLAN`, `OPERATOR_TOGGLED_CAMERA`. Whatever the
  Map / web UI lets the operator do should publish a typed event.
* **SPIM-fine alignment workflow** — populate `position_fine`. Tool +
  per-embryo state transition. Triggers `EMBRYOS_UPDATE` and a new
  `FINE_ALIGNMENT_COMPLETED` event the orchestrator can react to.
* **Continuous-shadow harness** — extend `ShadowRunner` to run a
  candidate alongside production in the live agent process (not just
  during replay). Collect divergences into a per-session
  `divergences.jsonl`.

### Longer arc (months)

* **A goal expression language** — predicates over the world model that
  let the operator say "image until 4-fold" or "follow the cell
  divisions in embryo 3 at high resolution." This is the substrate for
  §8.1 (plans-as-goals).
* **LLM-driven candidates** — once the rule-based `ReactiveCandidate`
  proves the substrate, add Claude-driven candidates (Haiku for cheap,
  Opus for thinking). Use the snapshot+digest as their input.
* **Cross-session learning loop** — wire the `learnings/` store into
  the world model as priors. Add a learning-write surface (a tool the
  orchestrator can call when it notices a pattern). Use shadow A/B to
  validate that learnings improve decisions.
* **Goal-driven evaluation** — once goals exist, "did the experiment
  achieve its goal" becomes a measurable end-to-end success rate. The
  ultimate metric is this, not turn-level decision diffs.

---

## 11. Principles That Surface Throughout

A few recurring design priors worth naming:

1. **Distill, don't dump.** Structured summaries beat raw logs in
   prompts. Hand-written formatters beat LLM summarisers for
   structured data. LLMs for prose, code for structure.
2. **Pull beats push when uncertain.** Default to tools the agent
   queries on demand, not data shoved into every prompt. Push only
   what's universally relevant (the world snapshot).
3. **Same shape for production and shadow.** If production writes a
   Decision with these fields, shadow candidates write Decisions with
   the same fields. Diff is then trivial.
4. **Events carry intent; state carries position.** `EMBRYOS_UPDATE`
   is state (the embryo list now). `OPERATOR_EDITED_EMBRYO` is intent
   (a human just did this). Both exist; they answer different
   questions.
5. **The brain doesn't move hardware.** All hardware action goes
   through tools that go through the device layer that goes through
   ophyd that goes through MMCore. Shadow candidates are constructively
   prevented from acting. Layers are not negotiable.
6. **No SaveCardSettings.** Firmware persistent state silently inherits
   between sessions; if it ever gets out of sync with code it's a
   debugging nightmare. Apply firmware config every boot, code wins.
7. **Localhost is the diSPIM box. Remote is view-only by default.**
   Auth surface stays tiny and explicit. Token upgrade is the seam,
   not user accounts.

---

## 12. Open Questions (Worth Revisiting Later)

* **Continuous vs episodic shadows.** Continuous always-on shadow
  captures divergence over time but multiplies cost (multiple LLM
  candidates running). Episodic shadow at decision moments is cheaper
  but misses timing-sensitive cases. Hybrid?
* **Is the conversation history the right substrate at all?** With
  decision moments hours apart, prior chat may be more distracting
  than useful. Maybe the brain shouldn't see chat history beyond N
  hours; the world model + events digest are the durable memory and
  chat is just for the active dialogue.
* **How much should the operator know about the orchestrator's plan?**
  Today the operator drives by asking. With autonomous mode, the
  orchestrator runs experiments largely on its own. Should there be a
  permanent "what is the orchestrator thinking right now" surface
  visible on the Map? An always-on intent display?
* **Failure semantics.** If a candidate would have made a different
  decision than production, and production's decision led to a bad
  outcome, the candidate "wins." How do we score? Define "bad outcome"
  rigorously enough that it can be measured?

These are not blockers. They are notes for the next iteration of this
document, after a few weeks of running on the substrate built here.
