# Design: Operations tab — observable tactics (sub-project D)

Status: design decided 2026-06-28 (autonomous, from recon).
Base branch: off `feature/temp-change-tactic` (C) — D renders the `temp_protocol`/`setpoint_changes`
snapshot fields C adds, and the protocol events C emits.

## 0. Roadmap context
Track 2. The user's ask: "when at the beginning of an experiment, tactics have been decided, it
should reflect and be communicated in the experiments tab… tactics should be observable — what and
how tactics are used in operations." Today's Experiment tab is embryo/cadence-centric and has **no
named-tactic concept**, no temperature/setpoint representation, and **does not auto-refresh**
(fetches once per tab click). D makes the composed temp-change tactic observable and live.

## 1. Overview
Three changes, all small, no new endpoints (reuse `GET /api/experiments/{session}/strategy`):
1. **Rename** the tab "Experiment" → "Operations" (label-only).
2. **Surface the tactic** in the swimlane SVG: a session-level band above the embryo rows showing
   the named protocol pill (spanning its start→complete), setpoint-change markers, and burst
   grouping/labels (before/during/after).
3. **Live refresh**: subscribe the tab to the relevant EventBus events (already relayed to the
   browser) + debounced re-fetch, so it updates during a run instead of only on tab click.

## 2. Architecture

### 2.1 Rename (label-only)
`index.html:119` nav button text and `:393` `<h2 class="experiment-title">` text → "Operations".
Leave all ids (`data-tab="experiment"`, `#experiment-content`, `#experiment-overview-root`) and
`TABS.EXPERIMENT` (`app.js`) unchanged — routing/JS untouched.

### 2.2 Burst phase in the event (1-line backend enabler)
The driver sets `b._phase` ("before"/"during"/"after") on each `BurstAcquisition`, but
`BURST_START` does not emit it, so the timeline can't distinguish phases. In
`gently/app/orchestration/exclusive.py` `BurstAcquisition.run`, include `"phase":
getattr(self, "_phase", None)` in the `BURST_START` event data. Then `strategy_snapshot._replay_timeline`'s
`burst_started` handler records `phase` on the burst phase dict (next to `frames`/`hz`). (C Task 6
already adds the `temp_protocol` band + `setpoint_changes`; this adds per-burst phase labeling.)

### 2.3 Frontend — the tactic band (`experiment-overview.js`)
Consume the snapshot's new fields (`temp_protocol`: `{start, end, target_setpoint_c, ...}` and
`setpoint_changes`: `[{at, to}]`, from C Task 6) and render, in `_renderSwimlanes` (line 332), a
**session-level band inserted above the per-embryo rows** (the SVG currently starts embryo rows at
`TOP_AXIS_H` — insert one band-height there and shift rows down):
- a **tactic pill** spanning the protocol window: label `"Temp-change burst → {target_setpoint_c}°C"`
  (tactic name is a frontend constant keyed off the `temp_protocol` field — events carry no name).
- **setpoint markers** at each `setpoint_changes[].at`: a vertical line + a `"→ N°C"` chip, reusing
  the existing cadence/power-chip drawing idiom (lines ~728-748).
- **burst phase tint/labels**: the per-embryo burst blocks (lines ~558-575) already render; tint or
  badge those whose `phase` is before/during/after (from 2.2). Empty state unchanged when no tactic.

### 2.4 Frontend — live refresh (`experiment-overview.js`)
In `ExperimentOverview.init`, add `ClientEventBus.on(...)` subscriptions (mirroring
`experiment-strip.js:23`) for: `TEMP_PROTOCOL_STARTED`, `TEMP_PROTOCOL_COMPLETED`,
`TEMPERATURE_SETPOINT_CHANGED`, `BURST_START`, `BURST_COMPLETE`, `EMBRYO_CADENCE_CHANGED`,
`POWER_RAMP_STEP` — each calls a **debounced** (~500 ms) `loadStrategy()`+`render()`. The websocket
relay (`server.py:386`) already emits all of these by `.name`, so no backend change. Keep the
existing now-ticker.

## 3. Data flow
protocol/burst events → EventBus → websocket (by `.name`) → `ClientEventBus` → debounced
`loadStrategy()` → `GET /strategy` → `build_strategy_snapshot` (now with `temp_protocol` +
`setpoint_changes` + per-burst `phase`) → `_renderSwimlanes` draws the tactic band + markers + tinted
bursts.

## 4. Error handling
- No active session / no tactic → existing calm empty state; the band simply absent when
  `temp_protocol` is null.
- Snapshot missing the new fields (older session) → render without the band (defensive optional
  chaining).
- Event flood → the debounce coalesces; re-fetch is cheap (single GET).

## 5. Testing
- Backend: `_replay_timeline` records `phase` on burst phases when `BURST_START` carries it (extend
  C Task 6's snapshot test).
- Frontend: `node --check`; a Chrome-MCP harness (like A/B1) feeding a snapshot with a `temp_protocol`
  band + `setpoint_changes` + phased bursts → screenshot the tactic band; verify a simulated
  `TEMP_PROTOCOL_STARTED` event triggers a debounced re-fetch. UI audit. Live rig verification deferred.

## 6. Out of scope
- A general user-authored tactic library (G).
- New strategy endpoints or backend snapshot restructuring (reuse `/strategy`; C owns the builder).
- The Rules table (unchanged).
- Wiring the docstring fix for the temp EventTypes (the enum comments wrongly say
  `{old_temp,new_temp}`/`{protocol_name,success}`) — fix opportunistically; D builds against the real
  emitter payloads (`{to}` / `{locked,cancelled,error}`).
