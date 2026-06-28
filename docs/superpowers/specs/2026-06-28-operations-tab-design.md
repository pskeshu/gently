# Design: Operations tab — the operation spine (sub-project D, redesigned)

Status: redesigned 2026-06-28 after a from-scratch design analysis + audit (user directive:
"start from no assumptions of legacy design… rethink completely new"). Supersedes the earlier
swimlane-band approach (discarded).
Base branch: `feature/operations-tab` (off C). Keep: the Experiment→Operations rename and the
burst-`phase` event. **Discard: the swimlane tactic-band rendering.**

## 0. The directive

Operations must present **tactics in three states — planned ("queued"), in use, and used** — so a
scientist reads at a glance what the agent decided, what it's doing now, and what's next. It must be
**testable against realistic experiments** and render **different tactic cases** (monitor, burst,
temp-change protocol, transmission, …). It is **mission-control for an autonomous instrument**, not
a timeline and not a SaaS dashboard.

## 1. The model — a generic tactic plan

Operations renders one data structure, the **operation view**:

```
operation = { title, session, tactics: [ tactic ] }
tactic = {
  seq, state: 'used'|'active'|'queued', name, kind,
  target?,            # e.g. "→ 32.0 °C"
  summary?,           # collapsed line (used/queued): "22 min · ended on signal"
  desc?,              # one-line sub-description
  trigger?,           # queued: the condition that starts it, e.g. "when temp locks"
  live?: {            # active (or any with live data)
    readouts: [{ label, value, sub?, bar? }],   # temp gauge, current burst, signal, cadence…
    phases?:  [{ name, state:'done'|'active'|'todo', count, pips:[...] }]  # before/during/after (protocols)
  }
}
```

This single model renders **any** tactic kind: a phased protocol (temp-change → phases + temp
gauge), a reactive monitor (expression-onset → readouts, no phases), a one-shot (transmission burst
→ readouts), or a purely queued plan (all tactics `state:'queued'`).

## 2. Architecture (four units)

### 2.1 Backend transform — `build_operation_view(snapshot)`
A new function (in `gently/ui/web/strategy_snapshot.py` or a sibling) that maps the existing
strategy snapshot into the operation model: the per-embryo `temp_protocol` band + `setpoint_changes`
+ phased bursts (from C) become a temp-change **tactic** with before/during/after **phases**; active
`monitoring_modes` become monitor **tactics** with **readouts** (cadence, power, signal); completed
bursts/phases become **used** tactics; not-yet-started planned items become **queued** tactics.
Served via the existing `/api/experiments/{session}/strategy` route (extend its payload with
`operation`) or a new `GET /api/operations/{session}` — reuse `/strategy` to avoid a new route.

### 2.2 The renderer — rewrite `experiment-overview.js`
Replace the swimlane Overview with the **operation spine** (data-driven from §1): a vertical thread
of tactic nodes; **used** = solid green dot + collapsed one-line summary; **in use** = amber node,
the *whole left column amber* (dot + seq + "IN USE"), card bloomed open with live internals; **queued**
= blue dashed dot + dimmed card + trigger condition. The spine connector is **state-colored**
(green used → amber active → blue-dashed queued); the first queued node is marked **next**. (The
Rules view can remain as-is or be folded in later — out of scope here.)

**Audit fixes baked in** (from the design audit):
1. Queued/decided-plan state reads as a *cocked* instrument: colored blue queued thread + "next" marker.
2. The **active phase dominates** the stepper; **"awaiting lock" is the headline** of the active phase.
3. **Flatten** the active card — no card-in-card; values sit on the panel face, separated by a hairline.
4. Active row's **entire left column amber**; colored left edge per state (green/amber/blue).
5. Copy: **"queued"** (not "going to be used"); replace "first/second/last" with the **trigger**;
   keep mono instrument values + domain copy ("ended on signal", "awaiting lock"); show rate-of-change
   deltas ("+14% over 6 min") and promote the decision-driving readout per tactic kind.

### 2.3 Scenario test mode (the "test it for real experiments" requirement)
The renderer accepts an operation model from either the live endpoint **or** a **scenario fixture**.
A small dev affordance — `?scenario=<name>` URL param (and/or a dev-only scenario `<select>`) — loads
a named fixture from a **scenario library** (`gently/ui/web/static/js/operations-scenarios.js`):
`temp_strain`, `expression_onset`, `pre_hatching`, `transmission_survey`, `decided_plan`, `idle`.
This makes Operations developable/verifiable **offline against any tactic case**, with no rig. The
fixtures are the prototyped harness fixtures, promoted into the repo as test data + a Chrome-MCP
audit target. (Scenario mode is dev-gated; live mode is default.)

### 2.4 Live refresh
The tab subscribes (via `ClientEventBus.on`, mirroring `experiment-strip.js`) to
`TEMP_PROTOCOL_STARTED/COMPLETED`, `TEMPERATURE_SETPOINT_CHANGED`, `BURST_START/COMPLETE`,
`EMBRYO_CADENCE_CHANGED`, `POWER_RAMP_STEP`, each triggering a debounced (~500 ms) refetch+render. The
websocket relay already emits all of these by `.name`; no backend change.

## 3. Data flow
events → EventBus → websocket → `ClientEventBus` → debounced refetch → `/strategy` (now with
`operation`) → `build_operation_view` → spine renderer. In scenario mode: `?scenario=` → fixture →
same renderer (no fetch).

## 4. Error/empty
- No active operation / no tactics → calm **idle** state ("When the agent begins operating by tactic,
  the plan and live progress appear here"), not a dead screen.
- Snapshot lacks `operation` (older session) → renderer falls back to a minimal "no tactic data" note.
- Unknown tactic `kind` → renders generically from `name`/`summary`/`readouts` (the model is kind-agnostic).

## 5. Testing
- Backend: `build_operation_view` over a hand-written snapshot → asserts the tactic list states +
  the temp-protocol tactic's phases + a queued tactic's trigger.
- Frontend: `node --check`; the **scenario library** doubles as the test corpus — a Chrome-MCP audit
  across all six fixtures (the harness, promoted); UI audit per the audit fixes. Live rig verification deferred.
- Scenario mode itself is the regression harness: any tactic case can be loaded and visually checked.

## 6. Out of scope
- A general user-/agent-authored persisted tactic *library* (G) — D presents tactics; G lets users
  define/save them.
- The Rules view rework; multi-embryo concurrent-operation layout (single active operation for now,
  though the model + spine extend to a stack of operations later).
