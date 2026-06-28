# Task 7 Report: tactic_id threading + start-edge marking in execution tools

## Call sites changed

### `gently/app/orchestration/exclusive.py`

**`BurstAcquisition.__init__`** (line ~95)
- Added `tactic_id: str | None = None` kwarg; stored as `self._tactic_id`.
- `BURST_START` event data (line ~127): added `"tactic_id": self._tactic_id`.
- `BURST_COMPLETE` event data (line ~229): added `"tactic_id": self._tactic_id`.

### `gently/app/orchestration/temperature_protocol.py`

**`run_temp_change_burst_protocol`** signature (line 22)
- Added `tactic_id=None` to the keyword-only params.
- `TEMP_PROTOCOL_STARTED` event data (line ~41): added `"tactic_id": tactic_id`.
- `TEMP_PROTOCOL_COMPLETED` event data (line ~67): added `"tactic_id": tactic_id`.

### `gently/app/orchestration/timelapse.py`

**`queue_burst`** method (line ~2056)
- Added `tactic_id: str | None = None` to signature.
- Passes `tactic_id=tactic_id` into `BurstAcquisition(...)` constructor (line ~2098).

### `gently/app/tools/timelapse_tools.py`

**`enable_monitoring_mode`** (line ~1025)
- Added `tactic_id: str | None = None`.
- On success: resolves `cs = getattr(agent, "context_store", None)`, `session = getattr(agent, "session_id", None)`; calls `cs.transition_tactic(session, tactic_id, "active")` (guarded).

**`queue_burst`** tool (line ~1150)
- Added `tactic_id: str | None = None`.
- Passes `tactic_id=tactic_id` to `orchestrator.queue_burst(...)`.
- After successful enqueue: calls `cs.transition_tactic(session, tactic_id, "active")` (guarded).

**`stop_timelapse`** (line ~270)
- Added `tactic_id: str | None = None`.
- After successful stop: calls `cs.transition_tactic(session, tactic_id, "done")` (guarded).

**`pause_timelapse`** (line ~293)
- Added `tactic_id: str | None = None`.
- After successful pause: calls `cs.transition_tactic(session, tactic_id, "paused")` (guarded).

### `gently/app/tools/temperature_protocol_tools.py`

**`run_temp_change_burst_protocol_tool`** (line 40)
- Added `tactic_id: str | None = None`.
- Before `asyncio.create_task`: calls `cs.transition_tactic(session, tactic_id, "active")` (guarded).
- Passes `tactic_id=tactic_id` into `_driver(...)` (the `run_temp_change_burst_protocol` coroutine).

## How the transition + event threading work

1. **Start edge (tool → store)**: Tool layer resolves the context store via `getattr(agent, "context_store", None)` and session via `getattr(agent, "session_id", None)` — the same pattern as `declare_operation_plan`. If both are present and `tactic_id` is truthy, `transition_tactic` flips the plan tactic to `"active"`. This happens synchronously (before the background driver spawns for the temp protocol, after enqueue for bursts).

2. **Event threading (orchestration → bus)**: The `tactic_id` is included in `BURST_START`, `BURST_COMPLETE`, `TEMP_PROTOCOL_STARTED`, and `TEMP_PROTOCOL_COMPLETED` event `data` dicts. Task 8's `OperationPlanUpdater` subscribes to these events and can resolve the tactic by `data["tactic_id"]` to flip it to `"done"`.

3. **Backward compatibility**: All `tactic_id` params default to `None`. Every transition call is guarded: `if tactic_id: cs = ...; if cs and session: cs.transition_tactic(...)`. When `tactic_id` is absent, no store access occurs and the tool behaves identically to before.

## TDD evidence

`tests/test_tool_tactic_linkage.py` — 20 tests, all pass.

Coverage:
- `enable_monitoring_mode` with/without tactic_id (+ no-cs guard)
- `queue_burst` (tool) with/without tactic_id; tactic_id forwarded to orchestrator
- `stop_timelapse` with/without tactic_id
- `pause_timelapse` with/without tactic_id
- `BurstAcquisition._tactic_id` stored; `BURST_START` and `BURST_COMPLETE` events contain `tactic_id`
- `run_temp_change_burst_protocol_tool` with/without tactic_id
- `TEMP_PROTOCOL_STARTED`/`TEMP_PROTOCOL_COMPLETED` carry tactic_id (including `None` backward-compat)

Full suite (excluding 3 pre-existing broken collection targets): **638 passed, 19 skipped** — same count as before Task 7; no new failures.

## Files changed

- `gently/app/orchestration/exclusive.py`
- `gently/app/orchestration/temperature_protocol.py`
- `gently/app/orchestration/timelapse.py`
- `gently/app/tools/timelapse_tools.py`
- `gently/app/tools/temperature_protocol_tools.py`
- `tests/test_tool_tactic_linkage.py` (new)

## Concerns

None. All changes are purely additive (new optional kwarg, guarded call). The `@tool` decorator wraps sync functions as async coroutines — tests call them with keyword args and `await`; microscope-required tools need a `"client"` in the test context dict (FakeMicroscope provided).
