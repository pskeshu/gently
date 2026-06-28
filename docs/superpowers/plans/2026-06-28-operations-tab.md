# Operations Tab (D) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Make the composed temp-change tactic observable: rename Experiment→Operations, render the protocol band + setpoint markers + phased bursts on the swimlane, and make the tab refresh live.

**Architecture:** Reuse `GET /api/experiments/{session}/strategy` (C added `temp_protocol` + `setpoint_changes` per-embryo). One backend enabler (emit burst `phase`), the rest is `experiment-overview.js` rendering + live event subscriptions.

**Tech Stack:** vanilla-JS + hand-rolled SVG (`experiment-overview.js`), the EventBus→websocket relay, pytest for the small backend bit.

## Global Constraints
- No new endpoints; reuse `/strategy`. C owns the snapshot builder; D consumes its fields.
- Snapshot fields (per-embryo, from C Task 6): `temp_protocol: {start, end, target_setpoint_c, frames, bursts_before, bursts_after} | None`, `setpoint_changes: [{t, to}]`.
- Rename is label-only (`index.html:119` nav text, `:393` `<h2>`); leave ids + `TABS.EXPERIMENT` unchanged.
- Tactic name is a frontend constant keyed off `temp_protocol` (events carry no name).
- Build against the REAL emitter payloads, not the stale enum docstrings.
- Live: subscribe via `ClientEventBus.on(...)` (mirror `experiment-strip.js:23`), debounced (~500ms) `loadStrategy()`+`render()`. Relay already emits all events by `.name` — no backend change.
- Frontend: no JS unit harness → `node --check` + Chrome-MCP harness + UI audit; live rig verification deferred.
- Git hygiene: stage only your files by explicit path; never `git add -A`.

---

### Task 1: Rename Experiment → Operations (label-only)
**Files:** Modify `gently/ui/web/templates/index.html` (nav button text `:119`, `<h2 class="experiment-title">` `:393`). Test: none (pure copy) — verified by `grep`.
- [ ] **Step 1:** change the nav button text at `:119` from `Experiment` to `Operations`, and the `<h2>` at `:393` from `Experiment` to `Operations`. Leave `data-tab="experiment"`, `#experiment-content`, `#experiment-overview-root`, `TABS.EXPERIMENT` untouched.
- [ ] **Step 2:** `grep -n ">Operations<" gently/ui/web/templates/index.html` shows both; `grep -n "data-tab=\"experiment\"" ` still present (ids unchanged).
- [ ] **Step 3: commit** — `git add gently/ui/web/templates/index.html && git commit -m "feat(operations): rename Experiment tab to Operations (label only)"`

---

### Task 2: Emit burst `phase` + record it in the snapshot
**Files:** Modify `gently/app/orchestration/exclusive.py` (`BurstAcquisition.run` BURST_START emit ~line 127-135 add `phase`); Modify `gently/ui/web/strategy_snapshot.py` (`_replay_timeline` `burst_started` handler ~line 762 record `phase` on the burst phase dict). Test: `tests/test_burst_phase_snapshot.py`.

**Interfaces:** Produces `BURST_START` data with `"phase": getattr(self, "_phase", None)`; snapshot burst phase dict gains `"phase"`.

- [ ] **Step 1: failing test**
```python
# tests/test_burst_phase_snapshot.py
import json, os
from gently.ui.web.strategy_snapshot import build_strategy_snapshot

def _write(tmp_path, events):
    sd = tmp_path / "sess"; (sd / "embryos" / "e1").mkdir(parents=True)
    (sd / "session.yaml").write_text("session_id: s1\nname: t\nstarted_at: '2026-06-28T10:00:00'\n")
    (sd / "embryos" / "e1" / "embryo.yaml").write_text("embryo_id: e1\nrole: Test\n")
    with open(sd / "timeline.jsonl", "w") as f:
        for e in events: f.write(json.dumps(e) + "\n")
    return sd

def test_burst_phase_recorded(tmp_path):
    sd = _write(tmp_path, [
        {"event_subtype": "burst_started", "embryo_id": "e1", "timestamp": "2026-06-28T10:00:10",
         "data": {"embryo_id": "e1", "frames": 60, "mode": "1hz", "phase": "during"}},
        {"event_subtype": "burst_completed", "embryo_id": "e1", "timestamp": "2026-06-28T10:01:10",
         "data": {"embryo_id": "e1"}},
    ])
    snap = build_strategy_snapshot(str(sd), "s1")
    emb = next(e for e in snap["embryos"] if e["id"] == "e1")
    burst = next(p for p in emb["phases"] if p.get("mode") == "burst")
    assert burst.get("phase") == "during"
```
> Confirm the real `build_strategy_snapshot` session-dir layout + timeline line shape from C Task 6's test (`tests/test_temp_protocol_snapshot.py`) and match it (session.yaml keys, embryo.yaml, timeline event shape). Adapt `_write` to the real shape; keep the `phase` assertion.
- [ ] **Step 2: run, expect FAIL** (phase is None / key absent)
- [ ] **Step 3: implement** — in `exclusive.py` `BurstAcquisition.run`, add `"phase": getattr(self, "_phase", None)` to the `BURST_START` event `data` dict. In `strategy_snapshot.py` `burst_started` handler, add `"phase": data.get("phase")` to the appended burst phase dict (next to `frames`/`hz`).
- [ ] **Step 4: run, expect PASS**; `pytest -q` no new failures.
- [ ] **Step 5: commit** — `feat(operations): emit burst phase in BURST_START + record in snapshot`

---

### Task 3: Render the tactic band on the swimlane
**Files:** Modify `gently/ui/web/static/js/experiment-overview.js` (`_renderSwimlanes` ~line 332: insert a session-level band; helpers for the protocol pill + setpoint markers, reusing the chip idiom ~728-748). Verify: `node --check` + Chrome-MCP harness.

- [ ] **Step 1: build the band** — In `_renderSwimlanes`, before drawing embryo rows (which start at `TOP_AXIS_H`), aggregate the protocol from the embryos: find the embryo with a non-null `temp_protocol` (and its `setpoint_changes`). If present, insert a band of fixed height (e.g. 28px) and shift embryo rows down by that height:
  - a **tactic pill** rect spanning `xFor(temp_protocol.start)`..`xFor(temp_protocol.end ?? now)`, labeled `Temp-change burst → ${temp_protocol.target_setpoint_c}°C` (a frontend constant prefix);
  - for each `setpoint_changes[i]`: a vertical marker line at `xFor(.t)` + a `→ ${.to}°C` chip (reuse the existing chip-drawing helper used for cadence/power chips ~728-748);
  - tint the burst blocks (already drawn per-embryo ~558-575) by their `phase` (before/during/after) — e.g. a subtle hue or a 1-char badge — using the `phase` field added in Task 2.
  If no embryo has `temp_protocol`, draw no band (layout unchanged) — keep the calm empty state.
- [ ] **Step 2:** `node --check gently/ui/web/static/js/experiment-overview.js` (exit 0).
- [ ] **Step 3: Chrome-MCP harness** — build `scratchpad/operationsdemo/` (copy `event-bus.js`, `experiment-overview.js`, `main.css`); an `index.html` that stubs `fetch('/api/experiments/current/strategy')` to return a snapshot with one embryo carrying a `temp_protocol` band (start/end), `setpoint_changes:[{t,to:25}]`, and a few phased burst phases; mount `ExperimentOverview`; screenshot the band; run the UI audit; report deferral of live rig verification.
- [ ] **Step 4: commit** — `feat(operations): tactic band — protocol pill + setpoint markers + phased bursts`

---

### Task 4: Live refresh on events
**Files:** Modify `gently/ui/web/static/js/experiment-overview.js` (`init` — add subscriptions + a debounced refetch). Verify: `node --check` + harness (simulate an event → refetch fires).

- [ ] **Step 1: implement** — in `ExperimentOverview.init`, after the initial `loadStrategy()`, register (mirroring `experiment-strip.js:23`):
```javascript
const _debouncedReload = (() => { let t; return () => { clearTimeout(t); t = setTimeout(() => { this.loadStrategy().then(() => this.render()); }, 500); }; })();
['TEMP_PROTOCOL_STARTED','TEMP_PROTOCOL_COMPLETED','TEMPERATURE_SETPOINT_CHANGED','BURST_START','BURST_COMPLETE','EMBRYO_CADENCE_CHANGED','POWER_RAMP_STEP']
  .forEach(ev => ClientEventBus.on(ev, _debouncedReload));
```
(Adapt to the real `ExperimentOverview` method names for load/render + `this` binding — confirm from the file; if `loadStrategy` isn't promise-returning, wrap appropriately. Don't double-register if `init` can run twice — guard with a flag.)
- [ ] **Step 2:** `node --check` (exit 0).
- [ ] **Step 3: harness** — in the Task-3 harness, dispatch a `ClientEventBus.emit('TEMP_PROTOCOL_STARTED', {...})` and confirm a (stubbed) refetch is triggered (e.g. fetch call count increments after the debounce). Screenshot/log.
- [ ] **Step 4: commit** — `feat(operations): live event-driven refresh of the Operations tab`

---

## Self-Review
- Rename → Task 1; burst phase enabler → Task 2; tactic band render → Task 3; live refresh → Task 4. ✓
- Open confirmations: real `build_strategy_snapshot` layout (Task 2), the `_renderSwimlanes` insertion point + chip helper (Task 3), the real `ExperimentOverview` load/render method names + init-guard (Task 4).
- Type consistency: `temp_protocol {start,end,target_setpoint_c}` + `setpoint_changes [{t,to}]` + burst `phase` across Tasks 2/3/4.
