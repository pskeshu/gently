# Operations — operation spine (D, redesigned) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Steps use `- [ ]`.

**Goal:** Replace the legacy swimlane with a data-driven **operation-spine** Operations view that shows tactics planned/in-use/used, renders any tactic case from a generic model, and is testable offline via scenario fixtures.

**Architecture:** A generic *operation* model (tactics with state + optional live readouts/phases); a rewritten `experiment-overview.js` overview = the spine renderer (audit fixes baked in); a repo scenario library (6 fixtures) + `?scenario=` dev mode; a backend `build_operation_view(snapshot)` transform served on `/strategy`; live event-driven refresh.

**Reference (the validated prototype — read it for the exact renderer, CSS, model, and fixtures):**
`/tmp/claude-1000/-home-dna-lab-projects-gently/3192c9a4-bc99-4c0a-9cbd-e094de730bcf/scratchpad/opsdesign/harness.html`

## Global Constraints
- Keep the Experiment→Operations rename (done) + the burst-`phase` event (done). DISCARD the swimlane band.
- Operation model: `{title, session, tactics:[{seq, state:'used'|'active'|'queued', name, kind, target?, summary?, desc?, trigger?, live?:{readouts:[{label,value,sub?,bar?}], phases?:[{name,state:'done'|'active'|'todo',count,pips:[...]}]}}]}`.
- Audit fixes (baked into the renderer): (1) queued spine is colored blue + first queued marked "next" (decided-plan must read as a cocked instrument, not empty); (2) active phase dominates + "awaiting lock"-style status is the phase headline; (3) FLATTEN — no card-in-card, values on the panel face + hairline dividers; (4) active row's whole left column amber + colored left edge per state (green/amber/blue); (5) copy "queued" (not "going to be used"), replace "first/second/last" with `trigger`. Keep mono instrument values + domain copy.
- Renderer must handle ALL six fixtures (phased protocol, readout monitor, one-shot, all-queued, idle) without error.
- No new endpoint: extend `/api/experiments/{session}/strategy` payload with an `operation` object.
- Frontend has no JS unit harness → `node --check` + Chrome-MCP audit across fixtures.
- Git hygiene: stage only your files by explicit path; never `git add -A`.

---

### Task 1: Discard the swimlane band
**Files:** Modify `gently/ui/web/static/js/experiment-overview.js` (+ its css) — revert the tactic-band rendering added in commit 2a3d8b0, returning the Overview to its pre-band baseline (the spine replaces it in Task 2). Keep everything else (Rules view, modes, etc.).
- [ ] **Step 1:** `git revert --no-commit 2a3d8b0` (or manually remove the band code + css if revert conflicts), keeping the rename (Task-1) and burst-phase (Task-2) commits intact. Confirm `git diff` shows only the band removal.
- [ ] **Step 2:** `node --check gently/ui/web/static/js/experiment-overview.js` (exit 0).
- [ ] **Step 3: commit** — `git add gently/ui/web/static/js/experiment-overview.js <css> && git commit -m "feat(operations): discard swimlane tactic-band (replaced by operation spine)"`

---

### Task 2: Scenario library + the operation-spine renderer (the core)
**Files:** Create `gently/ui/web/static/js/operations-scenarios.js` (the 6 fixtures, promoted from the harness); Modify `gently/ui/web/static/js/experiment-overview.js` (replace the Overview render path with the data-driven spine renderer from the harness, audit fixes applied) + the experiment css (spine/node/phase styles from the harness). Verify: `node --check` + Chrome-MCP audit.

**Interfaces:** Produces `OperationView.render(rootEl, operationModel)` (or fold into `ExperimentOverview`); consumes the operation model (§ Global Constraints) from the live snapshot (Task 3) or a scenario fixture; `OPERATIONS_SCENARIOS` (the fixture map) + a `?scenario=<name>` loader (dev mode).

- [ ] **Step 1: port the fixtures** — create `operations-scenarios.js` exposing `window.OPERATIONS_SCENARIOS = {temp_strain, expression_onset, pre_hatching, transmission_survey, decided_plan, idle}` — copy the fixture objects from the harness `SCENARIOS` (the `.op` values), as the repo test corpus. Apply the copy fixes: `state:'queued'` (not 'planned' label), add `trigger` fields ("when temp locks", "after #1", etc.) replacing "first/second/last".
- [ ] **Step 2: the renderer** — port the harness renderer (`renderOperation`/`renderTactic`/`renderReadout`/`renderPhase`) into `experiment-overview.js`, replacing the swimlane Overview path, WITH the audit fixes: state-colored spine connector (green/amber/blue-dashed), whole-left-column amber for active, colored left edge per state, flattened active card (remove inner card backgrounds; hairline divider between readouts and phases), active phase dominant + status headline, first-queued "next" marker, `STATE_LABEL.queued='queued'`. Keep mono labels.
- [ ] **Step 3: scenario dev mode** — on init, if `new URLSearchParams(location.search).get('scenario')` matches a fixture, render that fixture (no fetch); else fetch the live operation (Task 3). Guard so scenario mode is opt-in via the param.
- [ ] **Step 4: `node --check`** both JS files (exit 0).
- [ ] **Step 5: Chrome-MCP audit** — serve the static files, load `?scenario=temp_strain`, `expression_onset`, `decided_plan`, `idle`; screenshot each; verify the audit fixes (queued reads cocked, active dominates, flattened, amber left column, "queued" copy) and that all six render without error; fix flaws.
- [ ] **Step 6: commit** — `feat(operations): operation-spine renderer + scenario library (data-driven, 6 cases)`

---

### Task 3: Backend transform `build_operation_view`
**Files:** Modify `gently/ui/web/strategy_snapshot.py` (add `build_operation_view(snapshot)` mapping the snapshot → operation model; have `build_strategy_snapshot` include `operation` in its return, OR add it in the route). Test: `tests/test_operation_view.py`.

**Interfaces:** Produces `build_operation_view(snapshot: dict) -> dict` (the operation model); `/api/experiments/{session}/strategy` response gains `operation`.

- [ ] **Step 1: failing test** — feed a snapshot dict (one embryo with a `temp_protocol` band {start,end,target_setpoint_c}, `setpoint_changes`, phased bursts, and an active monitoring mode) to `build_operation_view`; assert the returned `operation.tactics` includes: a `used` monitor tactic, an `active` temp-change tactic whose `live.phases` reflect before/during/after, and the `state`s are correct. (Confirm the real snapshot shape from `tests/test_temp_protocol_snapshot.py`.)
- [ ] **Step 2: run, FAIL.**
- [ ] **Step 3: implement** `build_operation_view`: derive tactics from the snapshot — completed monitoring modes/bursts → `used`; the active `temp_protocol`/monitoring mode → `active` (with readouts from temp + current burst, phases from the protocol's before/during/after burst counts); planned/queued items → `queued`. Map fields into the §model. Include `operation` in the snapshot/route payload.
- [ ] **Step 4: run, PASS;** `pytest -q` clean.
- [ ] **Step 5: commit** — `feat(operations): build_operation_view transform on the strategy snapshot`

---

### Task 4: Live refresh
**Files:** Modify `experiment-overview.js` (`init` — subscribe + debounced refetch, mirroring `experiment-strip.js`). Verify: `node --check` + harness (simulate an event → refetch).
- [ ] **Step 1:** in `init`, after the initial live fetch, register `ClientEventBus.on(...)` for `TEMP_PROTOCOL_STARTED`,`TEMP_PROTOCOL_COMPLETED`,`TEMPERATURE_SETPOINT_CHANGED`,`BURST_START`,`BURST_COMPLETE`,`EMBRYO_CADENCE_CHANGED`,`POWER_RAMP_STEP`, each → a debounced (~500ms) refetch+render. Guard against double-registration; skip in scenario mode.
- [ ] **Step 2:** `node --check` (exit 0).
- [ ] **Step 3: harness** — dispatch a `ClientEventBus.emit('TEMP_PROTOCOL_STARTED', {...})` and confirm a (stubbed) refetch fires after the debounce.
- [ ] **Step 4: commit** — `feat(operations): live event-driven refresh`

---

## Self-Review
- Discard swimlane → Task 1; renderer+scenarios → Task 2 (audit fixes baked in); backend transform → Task 3; live refresh → Task 4. ✓
- Open confirmations: the real snapshot shape (Task 3, from C's snapshot test); the `experiment-overview.js` Overview render seam to replace (Task 2, read the harness for the target renderer); the css file for experiment styles (Task 1/2).
- Type consistency: the operation model fields are identical across the fixtures (Task 2), the transform (Task 3), and the renderer (Task 2).
