# Design: Temperature Interface (persistence + live graph)

Status: design approved in brainstorm (2026-06-27).
Base branch: off `integration/ux2-all` (PR #58, the UX-v2 stack) — this feature builds on that UI.

## 0. Where this sits (execution roadmap)

This is **sub-project A** of a sequenced program of work, preparing gently for
production temperature-strain experiments next week. The full order (commit history is
meant to reflect this priority):

- **Track 1 — critical path for next week**
  - **A. Temperature interface** *(this spec)* — persist temperature + live graph.
  - **B. Manual mode** — write-enabled control surface (illumination from MMConfig presets,
    live camera view, volume/timelapse config + by-hand triggers, temperature setpoint
    control). Embeds A's graph.
- **Track 2 — fast-follow (repeatability + observability)**
  - **C. Temp-change burst tactic** — a `MonitoringMode` choreographing
    transmission-burst → setpoint-change → bursts-during → bursts-after.
  - **D. Operations-tab overhaul** — make composed tactics observable.
- **Track 4 — later infrastructure**
  - **G. Tactics library** (user-/agent-authored, persisted, reusable tactics), then
  - **F. Plan-mode files** (repo base plans + session↔plan link/delink UI).
- **Lowest**
  - **E. Presentation / README.**

Next week's first experiment round is **hand-driven** (A + B). A is foundational because
B embeds its graph and C/D both consume the persisted temperature series.

## 1. Overview

The temperature controller (ACUITYnano Peltier stage, `gently/hardware/temperature.py`) is
already integrated as a Bluesky device: `read()` returns water temp + setpoint + state live,
exposed over the device layer at `GET /api/temperature/status`, with `set_temperature` /
`get_temperature` agent tools. The web UI already has an SSE push channel
(`DEVICE_STATE_UPDATE`) and shows temperature read-only in `devices.js`.

**Two things are missing, and this sub-project supplies exactly those two:**

1. **Persistence** — temperature is live-only today. `ImagingSpec.temperature_c` is a
   *planned setpoint*, never a recorded readout. Nothing writes actual temperature into the
   session or burst files, so a burst cannot be correlated to "what was the temperature."
   This is the one real data hole for the experiment.
2. **A live graph** — a continuous chart of current temperature, setpoint, and the rise/fall
   trajectory over time.

### What we keep / reuse
- The controller's existing `read()` and the device-layer `GET /api/temperature/status` route.
- The existing SSE push mechanism (`websocket.js` / `event-bus.js` / `status-store.js`).
- The file-based, human-browsable store convention (`FileStore`, append-only jsonl like
  `timeline.jsonl` / `predictions.jsonl`).
- The hand-rolled-SVG charting style already used in `experiment-overview.js` /
  `occupancy3d.js` — **no new charting dependency.**
- The "calm empty state, never mock data" UI convention.

### Scope decision: session-scoped capture
Temperature is captured **only while an imaging session is active** (decided in brainstorm).
No always-on facility daemon, no between-session global log. The series belongs to the
session, which is where correlation matters.

## 2. Architecture — five well-bounded units

### 2.1 Temperature log store (persistence)
Append-only series, one file per session: `sessions/{session}/temperature.jsonl`.
Dedicated file — **not** folded into `timeline.jsonl`, because 1 Hz samples would drown the
event stream and a standalone series is trivial to read and plot.

API (methods on / adjacent to `FileStore`, `gently/core/file_store.py`):
- `append_temperature_sample(session_id, sample)` — atomic append of one line.
- `read_temperature_log(session_id, since=None)` — return samples, optionally filtered to
  those at/after a timestamp (for the graph's rolling window / backfill).

Sample schema (one jsonl line):
```json
{"t": "2026-06-27T10:31:04.512Z", "water_c": 28.4, "setpoint_c": 32.0, "state": "heating"}
```
`state` mirrors the controller's reported state string (e.g. heating / cooling / locked).

### 2.2 Temperature sampler (the capture loop)
A small background service in the **agent** process (persistence and session lifecycle are
agent-side). While a session is active it:
- polls the device layer's `GET /api/temperature/status` on a fixed cadence,
- appends each reading to the session's `temperature.jsonl` via 2.1,
- holds the **latest reading in memory** (so acquisition can stamp it without a blocking read),
- emits an SSE `TEMPERATURE_UPDATE` event for the live graph.

Lifecycle: started/stopped with the session. If no temperature device is configured (the
`temperature:` block is config-gated in `device_layer.py`), the sampler does not run.

Cadence: **1 Hz**, configurable via the existing `temperature:` config block. (Thermal change
is slow; 1 Hz gives a smooth trace without flooding the log.)

### 2.3 Acquisition stamp (image↔temperature correlation)
At volume / burst acquisition, write the sampler's latest reading into the acquisition's
metadata:
- volume: `embryos/{id}/volumes/t{NNNN}.meta.yaml`
- burst: each frame `.meta.yaml` and the `burst.yaml` manifest
  (`gently/app/orchestration/exclusive.py`).

Stamped block:
```yaml
temperature:
  water_c: 28.4
  setpoint_c: 32.0
  state: heating
  sampled_at: 2026-06-27T10:31:04.512Z   # timestamp of the reading, not of acquisition
```
We stamp the **latest sample** rather than a fresh blocking device read — at 1 Hz it is
current enough, and acquisition stays latency-free. If no sample exists yet (session just
started), omit the `temperature` block rather than writing stale/zero values.

### 2.4 Read API (graph backfill)
`GET /api/temperature/history?session=current&since=<iso>` returns the series for the initial
graph render and reloads (`session=current` resolves to the most-recently-touched session,
matching the existing experiments route convention). Live updates ride the existing SSE
channel via `TEMPERATURE_UPDATE`; the endpoint is only for backfill.

### 2.5 Frontend component (the graph)
`gently/ui/web/static/js/temperature-graph.js` — a hand-rolled SVG line chart, reusable and
self-mounting, surfaced as a **card on the Devices tab** (where temperature already shows
read-only). B (manual mode) re-mounts the same component.

Renders:
- **water-temp trace** (continuous line),
- **setpoint line** (distinct, stepped — so a commanded jump from 28→32 reads as a step while
  the water trace climbs toward it),
- **current numeric readouts** (water, setpoint, state) above the chart,
- a **rolling time window** (last few minutes) with full-history fetch on demand.

Data: backfills from 2.4 on mount, then appends from the `TEMPERATURE_UPDATE` SSE stream.

## 3. Data flow

```
device layer (owns controller)
        ▲  GET /api/temperature/status   (poll @ 1 Hz, while session active)
        │
   agent: TemperatureSampler ──► append temperature.jsonl   (persistence)
        │                    └─► keep latest-in-memory ──► acquisition stamps meta
        └─► emit SSE TEMPERATURE_UPDATE ──► browser: temperature-graph.js (live append)

browser on mount/reload: GET /api/temperature/history?session=current ──► backfill graph
```

## 4. Error handling & edge cases

- **No temperature device / device layer down** → sampler no-ops; graph shows a calm
  "no temperature data" empty state (matching the codebase convention). Never mock data.
- **A failed poll** → log a gap and continue; a sampler error never crashes the session.
- **Partial last jsonl line** (atomic append + a reader tolerant of a truncated final line) →
  the history endpoint skips an unparseable trailing line rather than erroring.
- **Acquisition before first sample** → omit the `temperature` block from meta (no stale/zero).
- **Setpoint change** → captured naturally: the next sample carries the new `setpoint_c`, so
  the stepped setpoint line moves on its own; no special event needed.

## 5. Testing (TDD)

- **Store**: `append_temperature_sample` / `read_temperature_log` round-trip; `since` filter;
  tolerance of a truncated trailing line.
- **Sampler**: against the existing mock controller scaffold
  (`gently/hardware/dispim/devices/test_temperature_controller.py`) — samples at cadence,
  appends, updates latest-in-memory, emits SSE; **no-device path** (sampler stays dormant);
  a poll failure leaves a gap without raising.
- **Acquisition stamp**: a fake volume + burst acquisition stamps the latest sample into meta;
  the no-sample-yet path omits the block.
- **History endpoint**: filters by `since`; `session=current` resolution.
- **Frontend**: a light render test for the SVG component — water trace + stepped setpoint
  geometry, and the empty state when history is empty.

## 6. Out of scope (deferred to later sub-projects)

- **Setpoint *control* from the UI** — belongs to B (manual mode), which embeds this graph.
- **Automated temp-change choreography** — belongs to C (the tactic).
- **Always-on / facility-wide temperature logging** — explicitly not now (session-scoped only).

## 7. Open defaults (easy to flip)

- **1 Hz** sampling cadence.
- Stamp the **latest sample** (not a fresh blocking read) at acquisition time.
