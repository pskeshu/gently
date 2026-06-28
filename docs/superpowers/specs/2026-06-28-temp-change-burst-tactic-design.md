# Design: Temperature-change burst tactic (sub-project C)

Status: design decided 2026-06-28 (autonomous, from recon; flag any change before the rig run).
Base branch: off `feature/manual-mode-live-view` (B1) — C composes A's temperature persistence/stamp
and B1's `set_laser_config("ALL OFF")` brightfield primitive.

## 0. Roadmap context

Track 2, fast-follow to A + B1. C **automates** the hand-driven experiment B1 enables: bursts
before a temperature setpoint change, bursts during the ramp, bursts after lock — all in
brightfield (LED, no laser). It is the repeatable form of next week's experiment. The Operations-tab
observability rework (D) builds on the timeline events C emits.

## 1. Overview

The experiment is a **scripted, wall-clock sequence**, not a reactive rule. Recon confirmed
gently's `MonitoringMode`s are reactive rule-installers with no timed loop, so C is built as **an
agent tool that launches an async timed sequence** (a thin `TimelapseOrchestrator` method so it can
reach `_emit_event` / `queue_burst` / the client), composing primitives that already exist:
- bursts with temperature-stamped persistence + `BURST_START/COMPLETE` timeline events
  (`exclusive.py` `BurstAcquisition`),
- `set_temperature` / `get_temperature` with the `'[ SYSTEM LOCKED ]'` state contract,
- `set_laser_config("ALL OFF")` + `set_led("Open")` for brightfield.

## 2. What must be built (four units)

### 2.1 Brightfield burst (`laser_config` through the burst op)
`BurstAcquisition.run` calls `client.acquire_burst(...)` but never passes `laser_config`, so it
can't force lasers off. Add an optional `laser_config: str | None = None` to
`BurstAcquisition.__init__` (`exclusive.py`) and thread it into the `acquire_burst(...)` call; add
`laser_config=None` to `TimelapseOrchestrator.queue_burst(...)` and pass it through. A brightfield
burst is then `laser_config="ALL OFF"`.

### 2.2 Wait-for-lock helper
No wait-for-lock exists. Add `async wait_for_temperature_lock(client, timeout_s, poll_s=2.0) -> bool`
(a small util or orchestrator method): poll `client.get_temperature()` until `"LOCKED" in state`,
return True; return False on timeout. Substring match on `'[ SYSTEM LOCKED ]'`.

### 2.3 The protocol driver
`TimelapseOrchestrator.run_temp_change_burst_protocol(self, embryo_id, target_setpoint_c, *,
frames=60, mode="1hz", num_slices=1, bursts_before=1, bursts_after=1, lock_timeout_s=600.0,
poll_s=2.0)` — an `async def`:
1. Brightfield on: `await client.set_laser_config("ALL OFF")`, `await client.set_led("Open")`.
2. Emit `TEMP_PROTOCOL_STARTED` (embryo, target, params).
3. **Before:** run `bursts_before` bursts sequentially — construct
   `BurstAcquisition(embryo_id, frames=frames, mode=mode, num_slices=num_slices,
   temperature_provider=self._temperature_provider, laser_config="ALL OFF")` and `await burst.run(self)`
   (direct sequential run — the protocol owns the sequence; no queue contention for a standalone
   experiment).
4. **Change setpoint:** `await client.set_temperature(target_setpoint_c)`; emit
   `TEMPERATURE_SETPOINT_CHANGED` (from, to).
5. **During:** loop — run one burst, then `await client.get_temperature()`; break when
   `"LOCKED" in state` or `lock_timeout_s` elapsed (event-loop clock).
6. **After:** run `bursts_after` bursts.
7. Emit `TEMP_PROTOCOL_COMPLETED` (counts, final temp).
Each burst is temperature-stamped automatically (A). Bursts emit `BURST_START/COMPLETE` → render
for free.

### 2.4 Observability events (timeline)
Add `EventType.TEMPERATURE_SETPOINT_CHANGED`, `TEMP_PROTOCOL_STARTED`, `TEMP_PROTOCOL_COMPLETED`.
Map each in `gently/harness/session/timeline.py`'s EventType→subtype map (so they persist to
`timeline.jsonl`). Handle the subtypes in `strategy_snapshot.py:_replay_timeline` — open a
`"temp_protocol"` band on STARTED, close on COMPLETED, and record setpoint-change markers — so the
Experiment/Operations tab shows the tactic. (`TEMPERATURE_UPDATE` stays high-volume/non-persisted;
only the discrete setpoint *change* is a timeline event.)

### 2.5 Agent tool
`run_temp_change_burst_protocol(embryo_id, target_setpoint_c, frames=60, bursts_before=1,
bursts_after=1, context=...)` in `gently/app/tools/` — resolves the orchestrator + client, launches
the driver via `asyncio.create_task` (the ramp is minutes; don't block the agent turn), returns a
"protocol started" message. Callable from chat (and later a manual-mode button / Operations tab).

## 3. Data flow
tool → `asyncio.create_task(orchestrator.run_temp_change_burst_protocol(...))` → brightfield set →
bursts (BurstAcquisition.run → client.acquire_burst(laser_config="ALL OFF") → persisted under
`bursts/{id}/` with temperature stamp) → setpoint change (client.set_temperature + sampler records
`temperature.jsonl`) → poll-until-lock → after-bursts → events to EventBus → `timeline.jsonl` →
`strategy_snapshot` → Experiment tab.

## 4. Error handling
- No client / no embryo → tool returns an error, no task launched.
- A burst failure mid-protocol → log, emit `TEMP_PROTOCOL_COMPLETED` with an `error` field, restore
  state (lasers already off). Never leave lasers on.
- Lock never reached → `during` loop exits on `lock_timeout_s`; protocol proceeds to after-bursts and
  notes `locked=false`.
- Cancellation (`asyncio.CancelledError`) → stop queuing, emit completion with `cancelled=true`.

## 5. Testing
- `BurstAcquisition` passes `laser_config` to a fake `acquire_burst` (unit).
- `wait_for_temperature_lock`: a fake client returning non-locked then locked → True; always
  non-locked → False on timeout (use a tiny timeout/poll).
- The driver against fakes (fake client + a `BurstAcquisition.run` patched to a no-op recorder):
  asserts the phase order (brightfield set → before bursts → setpoint change → during-until-lock →
  after bursts → completed), the emitted events, and `laser_config="ALL OFF"` on every burst.
- timeline map + `strategy_snapshot` replay: the three new subtypes open/close a temp_protocol band
  + setpoint marker.
- Rig-deferred: the real thermal ramp timing.

## 6. Out of scope
- The full Operations-tab visual rework (D) — C only emits the events D will render richly.
- A general user-authored tactic library (G).
- Manual-mode UI button to launch the protocol (nice follow-up; chat/tool is enough for next week).
