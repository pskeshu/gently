# Gently — Project Kanban

Parked ideas, design sketches, and follow-up work. Items get pulled into
real branches/commits when ready; otherwise they live here so we don't
forget.

---

## TODO (next up)

### StopWatcher / Supervisor (LLM-based stop decisions)

**Origin**: discussion 2026-05-23 after noticing calibration embryos
kept being imaged after they hatched. Fix B (role-based
`no_object_consecutive_terminal`) shipped as the immediate patch; the
watcher is the architecturally clean follow-up.

**Idea**: same pattern as the perceiver/classifier split — separate
the cheap deterministic checks from expensive contextual reasoning.
A separate component (the Watcher / Supervisor) reasons about whether
each embryo should keep being imaged, based on accumulated context.

**Architecture sketch**:
```
per-timepoint, cheap, deterministic
  Detector → IntervalRule / PowerRule / BurstRule
           → StopCondition (manual / duration / fixed)
           → role-based no_object terminal  (B — done)
                          │
                          │  trigger signals
                          ▼
intermittent, LLM, contextual
  StopWatcher
    inputs:  recent perceiver prose (last N timepoints)
             recent classifier outputs
             dose accumulated vs budget
             other embryos' state
             time elapsed in current phase
    output:  {action: stop | continue | pause,
              reason: prose,
              confidence: HIGH | MEDIUM | UNCERTAIN,
              confirm_required: bool}
```

**When to fire** (cheapest first):
1. Trigger-driven — a deterministic condition is in an ambiguous zone
   (e.g. `consecutive_no_object >= 1 but < role_threshold`). The
   deterministic layer says "I'm not sure," the watcher decides.
2. Schedule-driven — every N rounds (~30 min) per embryo, periodic
   "is this still worth imaging?" check. Catches slow degradation
   that rules miss.
3. User-driven — explicit `/check embryo_003` command.

**Cost estimate**: 6 embryos × 1 watcher call / 30min × 4 hours ≈
48 LLM calls per session. With prompt caching across embryos,
~$0.50/session. Negligible.

**Beyond stops** — the same component could reason about:
- Burst readiness ("structure is GOOD on embryo_005 but still
  developing; wait 5 min for stable signal?")
- Cross-embryo decisions ("embryos 003 and 005 are in the same
  phase, pause 005 to focus light budget on 003")
- Experiment-level ("all test embryos crossed onset; calibration
  embryos gone; recommend ending the session")

A more honest name: **Supervisor** or **Strategist** — it reasons
about the experiment as a whole, not just per-timepoint outcomes.

**Tradeoffs**:
- ✅ Catches what deterministic rules miss; gives audience-readable
  explanations.
- ✅ Decouples policy from detection — tune the prompt without
  touching the detector.
- ✅ Aligns with how a human microscopist actually monitors a long
  timelapse.
- ⚠️ Adds a moving part. Wrong stops are destructive. Mitigation:
  `confirm_required=True` on stop actions; require deterministic
  AND watcher agreement.
- ⚠️ Latency on stop decisions (~30 min). Mitigation: trigger-driven
  invocation for time-sensitive cases.

**Implementation order**:
1. Stub the StopWatcher framework. Invoke-only (no LLM yet, just
   record what it would have done). Get the plumbing right.
2. Wire the LLM call, scoped to the no_object-but-not-yet-terminal
   case first. Smallest blast radius.
3. Open up periodic schedule + cross-embryo context.

**Files this would touch**:
- `gently/app/orchestration/watcher.py` (new)
- `gently/app/orchestration/timelapse.py` — invoke from
  `_check_stop_condition` + scheduled task
- `gently/harness/state.py` — watcher history per embryo
- Possibly new event type `WATCHER_DECISION`

---

### Persistent per-embryo dose tracking (first-class)

**Origin**: discussion 2026-05-23 — `import_embryos_from_session`
currently reconstructs dose by walking `volumes/*.meta.yaml`. The
fields `exposure_count` / `total_exposure_ms` / `last_imaged` aren't
persisted to `embryo.yaml`; they only live in memory during a session.

**Idea**: every dose-emitting path (normal acquisition, calibration
sub-acquisitions, burst frames, verification volumes, manual
`acquire_volume`) goes through a single `record_dose()` API that
writes:
1. **Per-event log** `embryos/{id}/dose_log.jsonl` — full provenance
2. **Summary on `embryo.yaml`** — atomic `dose:` block:
   ```yaml
   dose:
     total_exposure_ms: 40500.0
     exposure_count: 81
     by_source:
       normal: 38000.0
       calibration: 2500.0
       burst: 0
       verify: 0
     last_imaged: '2026-05-23T16:22:15'
   ```

**Files**:
- `gently/core/dose.py` (new) — `record_dose(store, session_id,
  embryo_id, *, source, num_slices, exposure_ms, ...)`
- `gently/core/file_store.py` — `append_dose()` updates log + summary
- Acquisition call sites in `timelapse.py`, `exclusive.py`,
  `calibration/*`
- `import_embryos_from_session` reads `dose:` directly; falls back to
  meta-walking for legacy sessions

**Status**: TODO comment already in `agent.py:_compute_imported_dose`.

---

### Cross-session resume restores orchestrator state

**Origin**: discussion 2026-05-23 about testing the timelapse resume.

**The gap**: `save_state()` writes `timelapse.yaml` after every
acquisition (rules, applied_rules, burst_applied, dose budget,
current_round). But `resume_session` in
`harness/session/manager.py:191` restores only embryos + conversation,
not orchestrator state. `orchestrator.load_state()` exists and works
but is never called.

**Fix**: wire `orchestrator.load_state()` into the resume flow. After
embryo state is restored, call `load_state()` to restore rules,
applied flags, round counter, dose budget. Strip the in-place
"if no last_imaged, zero exposure" branch (already addressed for
import, not for resume).

**Files**: `gently/harness/session/manager.py`, possibly
`gently/app/agent.py` to thread the orchestrator reference through.

---

### Bounded perceiver follow-up (multi-turn, last N turns)

**Origin**: discussion 2026-05-23. Implemented + reverted on the same
day after user reconsidered. The fundamental tradeoff was
acknowledged: multi-turn perceiver gives the model temporal context
("the puncta I described earlier are now brighter") but adds a moving
part to the critical detector path.

**Status**: deliberately not pursuing for the dopaminergic experiment.
Revisit if a less critical task wants it.

**Note**: The architectural pattern (sliding-window of N user+assistant
turns, system= for instructions, text-only history with images dropped
from prior turns) IS the right shape if/when we do this. Sketch lives
in the conversation history.

---

### Low Damage Mode — Phase 1 (MVP)

**Origin**: discussions 2026-05-22 / 2026-05-23 — design notes
flagged a general damage-reduction tactic with per-embryo + global
scope, calibration exemption, agent + human triggering, and tight
coupling to the photodose audit story.

**Idea**: Toggleable acquisition overlay that reduces dose vectors
(slices, power, cadence). Reversible — snapshots baseline on
activation, restores on deactivation. Manual toggle via agent tools
for Phase 1; rule-driven activation comes in Phase 2 via the
`AcquisitionOverlay` primitive (see tactics-library entry below).

**Architecture sketch**:
```python
EmbryoState.low_damage_active: bool = False
EmbryoState.overlay_baseline: Optional[Dict] = None
TimelapseOrchestrator._global_low_damage_active: bool = False

effective_state = global OR per_embryo

_acquire_embryo:
  if effective_state and embryo.role != 'calibration':
    if not baseline: snapshot (num_slices, power_488_pct, interval_seconds)
    apply overlay:
      num_slices  × 0.25  (floor 5)
      power_488   - 1.0   (device-clamped 2-6)
      interval    × 1.5
  on disable: restore baseline, clear flag
```

**Agent tools**:
- `enable_low_damage_mode(embryo_id=None)` — None = global
- `disable_low_damage_mode(embryo_id=None)`
- `get_low_damage_status()`

**Files**:
- `gently/harness/state.py` — new EmbryoState fields
- `gently/app/orchestration/timelapse.py` — `_apply_low_damage_overlay`,
  `_restore_low_damage_baseline`, integration in `_acquire_embryo`,
  extend save/load state
- `gently/app/orchestration/timelapse_models.py` — new event types
  `LOW_DAMAGE_ACTIVATED` / `LOW_DAMAGE_DEACTIVATED`
- `gently/app/tools/timelapse_tools.py` — three new tools
- `gently/ui/web/timelapse_tracker.py` — track LDM events
- `gently/ui/web/static/js/devices.js` — badge LDM embryos (shield
  icon)

**Tradeoffs**:
- ✅ Reversible — snapshot/restore means no baseline values lost
- ✅ Calibration exempt — honors the sacrificial-embryo design
- ✅ Composes with adaptive_power / adaptive_interval — overlay applies
  on top of whatever cadence/power is currently active
- ⚠️ Hardcoded overlay constants in Phase 1 — params live in one
  Python dict; become yaml-configurable in Phase 2
- ⚠️ User toggle is sticky — auto-disable on dose-pressure-clear comes
  with Phase 2 rule-driven LDM

**Design notes**: full deep dive in conversation 2026-05-22; sections
to land in `docs/imaging_template_design.md` §5 / §10 as part of the
doc-update entry below.

**Target**: v0.21.0

---

### Campaign `imaging_template` loader (Path B)

**Origin**: design conversation 2026-05-22 — minimal `imaging_template`
block in `campaign.yaml` that auto-configures the orchestrator at
session start. Closes the "agent forgets to install monitoring mode"
gap that caused the 2026-05-22 dopaminergic run to image at base
cadence even after detector reported lit_up from t30+.

**Minimum schema**:
```yaml
imaging_template:
  schema_version: 1
  base_interval_s: 300.0
  monitoring_mode: expression_monitoring
  monitoring_mode_params:
    fast_interval_s: 60.0
    rampdown_step_pct: 1.0
    rampdown_floor_pct: 2.0
  stop_conditions:
    - kind: all_test_hatched
    - kind: duration
      duration_s: 86400
```

`start_adaptive_timelapse` reads the active campaign's template if
present and `monitoring_mode` wasn't passed explicitly; explicit kwargs
still override.

**Files**:
- `gently/harness/imaging_template.py` (new) — `ImagingTemplate`
  dataclass + `apply_to_orchestrator`
- `gently/harness/memory/file_store.py` — extend `Campaign` with
  optional `imaging_template`
- `gently/app/tools/timelapse_tools.py` — wire reading into
  `start_adaptive_timelapse`

**Migration**: existing campaigns continue to work — block is purely
additive and optional.

**Related**: `docs/imaging_template_design.md` is the full schema spec;
Path B is the minimum useful subset.

**Target**: v0.21.0

---

### Sacrificial vocabulary alias

**Origin**: design notes + biologist-cognition-consultant review
(2026-05-22). "Calibration" reads as engineering jargon (bead slides,
power meters); the lab uses "sacrificial." Vocabulary mismatch
will confuse biologists onboarding to gently.

**Idea**: Add user-facing `display_name="Sacrificial"` to
`EmbryoRole(calibration)`. Internal `role='calibration'` string stays
for backwards compat (touches role registry, embryo.yaml, FileStore
lookups, persisted session state — too invasive to rename). UI labels,
tool descriptions, prompt mentions render the display name.

**Files**:
- `gently/harness/roles.py` — add `display_name` field on `EmbryoRole`
- `gently/ui/web/static/js/devices.js`, `embryos.js`, `marking.js` —
  UI strings
- `gently/app/tools/timelapse_tools.py`, `experiment_tools.py` — tool
  descriptions
- `gently/harness/prompts/templates.py` — prompt mentions
- `docs/imaging_template_design.md` — naming map update

**Target**: v0.21.0

---

### Update `imaging_template_design.md` with design-notes follow-ups

**Origin**: design notes (2026-05-22) raised six items not yet
captured in the v2 schema doc that landed on the same day.

**Items**:
1. Memory / learning layer — campaign learnings as tactic input
   (new §3.5)
2. `low_damage_mode` as worked example in §5 (first-class primitive
   set) and §10 (worked examples)
3. Future detector kinds — VQA, raw_metric (new §11.5)
4. Mode-conflict resolution policy / higher-order volition (new §13
   open question #8)
5. Sacrificial vocabulary — §2 note + §12 naming map alias
6. Signals + priority as embryo-level state (brief mention in §10)

**Files**: `docs/imaging_template_design.md` — additive only, no
structural rewrite.

**Target**: v0.21.0

---

### Hardware-state streaming page

**Origin**: design notes — need a continuous canonical "what is
the microscope doing right now" view. Currently telemetry is scattered
across devices view, map view, embryos view; no single page shows the
full real-time state of the instrument.

**Idea**: New web UI tab streaming real-time telemetry — xyz positions,
laser power per channel, focus score, current acquisition status,
exposure-budget-remaining per embryo, LDM status, active monitoring
modes, queued bursts, recent detector findings. Designed for ambient
awareness during long unattended runs.

**Files**:
- `gently/ui/web/templates/index.html` — new tab definition
- `gently/ui/web/static/js/hardware-state.js` (new)
- `gently/ui/web/routes/data.py` — telemetry snapshot endpoint
- Device layer — expose additional metrics if not already available

**Target**: v0.21.0 or v0.22.0

---

### Interrupt / override formalization

**Origin**: design notes — the scenario of a human watching the
agent do something funny and wanting to override its decision.
Pattern not formalized today; user prompts redirect the agent but
there's no explicit override semantics.

**Idea**: Explicit interrupt + override UX. Override locks prevent the
agent from auto-reverting human-set state (manually paused embryo
stays paused even when an agent rule wants to resume; manually enabled
LDM stays on even when an agent's auto-trigger clears). When the agent
proposes an action that conflicts with a recent human override, it
surfaces the conflict rather than silently proceeding.

**Status**: design-first. Needs a separate doc —
`docs/agent_interrupt_design.md` — before implementation.

**Files**: TBD pending design.

**Target**: v0.22.0+

---

### Tactics library expansion (LDM Phase 2 + filters + state predicates)

**Origin**: `docs/imaging_template_design.md` §5 lists 7 first-class
primitives; only 4 currently EXIST (`adaptive_interval`,
`adaptive_power`, `burst_capture`, `stop_on`). The rest are PLANNED.

**Items** (each could become its own kanban entry when pulled):
- `AcquisitionOverlay` primitive — generalizes LDM Phase 1 into a
  configurable, reversible, multi-param overlay tactic
- `dose_threshold` state predicate — fires at X% of budget; the
  auto-trigger for LDM
- `runtime_state` predicate kind — query `EmbryoState` flags as states;
  foundation for state chaining
- `debounce_state` filter — N-of-K windowed consensus on detector
  states; addresses the WEAK-noise false-positive risk surfaced in the
  2026-05-22 run analysis
- `ensemble_detect` filter — multi-shot VLM with agreement threshold
- `raw_signal_crosscheck` filter — quantitative ROI mean above rolling
  baseline gating categorical state
- `anticipatory_ramp` tactic — project-memory-driven cadence ramp
  before predicted events (depends on memory layer)

**Files**: `gently/app/orchestration/timelapse_models.py`, new
`gently/app/orchestration/filters.py`, new
`gently/app/orchestration/states.py` (state predicate kinds).

**Target**: v0.22.0+

---

### Detector-layer expansion — VQA + raw_metric

**Origin**: design notes — Perception agent extension beyond
narrow categorical classifiers. VQA for arbitrary visual questions
("does this embryo look stressed?"); raw metrics (focus score, ROI
mean intensity, brightness percentile) as fallback "perception" without
VLM cost.

**Idea**: Two new detector kinds:
- VQA — returns free text + optional structured fields; useful when
  the categorical schema doesn't fit (anomaly detection, unusual
  morphology, "what's interesting in this image")
- raw_metric — returns numeric values for tactics to predicate on
  (e.g., trigger LDM if focus_score drops below threshold)

**Files**:
- `gently/app/detectors/vqa.py` (new)
- `gently/app/detectors/raw_metric.py` (new) — focus, intensity stats
- `gently/app/detectors/registry.py` — register new factories

**Target**: v0.22.0+

---

### Project-level memory layer

**Origin**: design notes — "the experience of the past
experiments can benefit the future experiments." Concrete example:
twitching-to-hatching interval distribution accumulated across past
sessions of the same campaign feeds the schedule for an
`anticipatory_ramp` tactic in a new session.

**Idea**: Tactics can read from `D:/Gently3/agent/campaigns/{id}/learnings/`
to parameterize themselves at session start. New design pattern + a
supporting API in `FileContextStore`. Learnings are written by
post-session summarization (separate concern, mostly already in
place — `learnings/{id}_{slug}.yaml` files exist).

**Files**:
- `gently/harness/memory/file_store.py` —
  `read_campaign_learnings(campaign_id, query)` API
- `gently/harness/imaging_template.py` (once Path B exists) —
  tactic-param-from-learning lookup
- `docs/imaging_template_design.md` — new §3.5 covers this layer

**Target**: v0.22.0+

---

### F-drive automation emulation

**Origin**: design notes — emulate the existing manual or scripted
data-export / processing pipeline that runs against F-drive output
today, so the agent can take it over.

**Status**: needs definition. Define the F-drive workflow end-to-end
before scoping. Likely candidates: post-acquisition data copy to
F-drive, format conversion, naming convention enforcement, metadata
export.

**Target**: v0.22.0+

---

## In Progress

_(nothing currently)_

---

## Done (recent)

- **2026-05-23** — Role-based `no_object_consecutive_terminal` (B from
  the watcher discussion). Calibration: 2 consecutive → stop.
  Test: 5. Unassigned: never.
- **2026-05-23** — Two-stage dopaminergic detector with perceiver +
  classifier split.
- **2026-05-23** — `/import-embryos` carries role + light budget.
- **2026-05-23** — Auto-burst rule on `(MEDIUM|STRONG) AND GOOD
  structure`, one-time per embryo.
- **2026-05-23** — Rule fan-out fix (`applies_to` now listen-filter,
  not target list).
- **2026-05-23** — Released v0.20.0.
