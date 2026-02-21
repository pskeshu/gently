# Gently Changelog

What changed in each version and what we were thinking at the time.

---

## v0.4.0

Consolidated five overlapping storage systems into `GentlyStore`. Added
`EventBus` for async messaging. Set up the daemon architecture (context,
clock, agent core, capabilities).

We switched from RPyC to HTTP for the device layer — easier to debug and
process-isolated, so a crashed agent can't take down hardware. The event
bus became the way components talk to each other; publish/subscribe
instead of direct calls.

The embryo became the basic unit of the system, not the image. Each one
carries imagery, calibration state, perception traces, and detector
configs. Safety was layered: process isolation, device limits, templated
actions, automatic cleanup.

---

## v0.5.0

Replaced Rich CLI output with an Ink (React + Node.js) TUI connected via
WebSocket. The copilot stopped owning stdout.

- Persistent layout: header, scrolling chat, input bar, status bar.
- WebSocket transport so the TUI doesn't poll.
- Choice pickers for structured questions — the LLM proposes options,
  the human picks.
- 8 themes, switched client-side.
- Split monolithic `server.py` (2,159 lines) into 13 route modules.

Perception moved here too — VLM-based stage classification, three-view
projections (XY, XZ, YZ), trace persistence for timelapse.

Separating display from logic made the boundaries cleaner.
`CopilotBridge` handles async mechanics, the TUI handles presentation.

+7,923 / -7,151 lines, 66 files.

---

## v0.6.0

Added plan mode. Run mode is for real-time control ("what should we image
now"), plan mode is for experimental design ("how should we structure this
study"). They use different prompts, different tools, different thinking
budgets.

- Campaign/PlanItem/ImagingSpec/BenchSpec data model with dependency
  graphs.
- `ContextStore` for the agent's understanding (campaigns, learnings),
  separate from `GentlyStore` (raw data, images). Different lifecycles.
- Organism and hardware modules (`gently/organisms/celegans/`,
  `gently/hardware/dispim/`) to make the system backend-agnostic.
- Startup wizard for onboarding.
- Early research tools: `search_literature`, `search_strains`,
  `check_hardware_capability`.
- Extended thinking for complex operations.

We wanted the copilot to work at the same abstraction level as the
scientist — campaigns and research questions, not pixel coordinates.

+13,000 / -1,512 lines, 76 files.

---

## v0.6.1

Cleanup. Removed dead code, relocated configs, flattened backend
directory, refreshed docs. Removed DiSPIM-specific scaffolding.

+81 / -13,692 lines, 112 files. Mostly deletion.

---

## v0.7.0

Plan mode was a prototype in v0.6.0. This version made it actually
usable.

Research tools got real API integrations:
- PubMed via NCBI E-utilities (search + abstracts)
- Paper reading via PMC full text, Unpaywall, local PDFs, URL fetch
- WormBase and CGC for strain search
- NCBI Gene for gene information

Plan infrastructure:
- Versioning with JSON snapshots (snapshot/list/restore)
- Validation — hardware limits, stage order, duration estimates,
  dependency cycle detection
- Execution bridge linking plan items to running sessions
- Templates for reusable protocols
- Markdown export
- Reorganization tools (move, delete, reorder, phase management)
- References — plan items carry citations from research tools

Extended thinking: plan mode always uses it (30K token budget), run mode
uses 10K triggered by complexity.

TUI: human-readable tool labels, session resume, campaign resolution by
shorthand/name.

+8,046 / -644 lines, 28 files.

---

## v0.8.0

Added LAN peer-to-peer coordination. Instances find each other via UDP
broadcast and can share campaigns.

- UDP discovery on port 19547, zero config.
- HTTP peer client for remote campaign operations.
- 8 new mesh API endpoints (share, join, claim, export, etc).
- Each node advertises capabilities (GPU, SAM, storage).
- Campaign sharing: origin shares, peers join and claim items. Double-claim
  returns 409, re-claim is idempotent.
- `/peers` command in TUI. Status bar shows peer count.
- 27 tests for coordination flows.

+1,778 lines, 22 files.

---

## v0.8.1

Status polling was every 30 seconds, so mode changes (run to plan) took
a while to show up on peers. Added a nudge pattern:

1. Node changes mode -> `EventBus` emits `STATUS_CHANGED`
2. `MeshService` hears it -> UDP nudge broadcast
3. Peers receive nudge -> immediate HTTP refetch
4. Updates in ~1 second

The 30s poll stays as fallback. The nudge is just "come look at me" — no
payload, no ordering, no delivery guarantee. If a peer misses it, the
poll catches up.

5 files, +53 lines.

---

## Notes on how we think about this

Things we've learned building this, roughly in order:

- The embryo should be the unit, not the image. That's how biologists
  think about it.
- If the agent decided something, you should be able to see why.
  Perception traces, plan versions, thinking blocks.
- Real-time control and experimental design are different enough to need
  separate modes with separate tools.
- The agent's understanding (ContextStore) and raw data (GentlyStore) have
  different lifecycles and should be kept apart.
- Publish/subscribe keeps coupling low. Most things don't need to call
  each other directly.
- Safety should come from the architecture (process isolation, device
  limits), not from hoping the prompt is good enough.
- The system should work offline. Mesh discovery is nice when it's there,
  but not required.
