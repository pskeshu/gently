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
