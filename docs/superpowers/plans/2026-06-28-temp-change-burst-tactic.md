# Temp-Change Burst Tactic (C) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`) syntax.

**Goal:** A scripted temperature-change burst protocol — brightfield bursts before a setpoint change, during the ramp (until lock), and after — launchable as an agent tool, observable on the Experiment tab.

**Architecture:** A thin async `TimelapseOrchestrator.run_temp_change_burst_protocol` driver composing existing `BurstAcquisition` (extended to force lasers off), `set_temperature`/`get_temperature`, and brightfield primitives; new timeline EventTypes so the tactic + setpoint changes render; an agent tool that launches the driver via `asyncio.create_task`.

**Tech Stack:** Python asyncio, the gently EventBus + TimelineManager, pytest (`asyncio_mode=auto`).

## Global Constraints
- C composes A (temperature stamp/persistence) + B1 (`set_laser_config("ALL OFF")`). No new deps.
- Brightfield every burst: `laser_config="ALL OFF"`; lasers never left on, even on error/cancel.
- Lock contract: poll `client.get_temperature()["state"]` until `"LOCKED" in state` (the device reports `'[ SYSTEM LOCKED ]'`).
- Bursts are temperature-stamped automatically (A) and emit `BURST_START/COMPLETE` (render for free).
- New EventTypes use `auto()` (wire serializes `.name`).
- Tests: fakes for client + burst; `asyncio_mode=auto` (no decorator). Rig-deferred: real ramp timing.
- Git hygiene: stage only your files by explicit path; never `git add -A` (pre-existing untracked screenshots/mockups + uv.lock are not yours).

---

### Task 1: Timeline EventTypes for the tactic

**Files:** Modify `gently/core/event_bus.py` (3 new `auto()` members near the other domain events); Modify `gently/harness/session/timeline.py` (add 3 entries to the EventType→subtype map, near the `BURST_*` entries ~line 213-225). Test: `tests/test_temp_protocol_events.py`.

**Interfaces:** Produces `EventType.TEMPERATURE_SETPOINT_CHANGED`, `EventType.TEMP_PROTOCOL_STARTED`, `EventType.TEMP_PROTOCOL_COMPLETED`, each mapped to a timeline subtype (`setpoint_changed`, `temp_protocol_started`, `temp_protocol_completed`).

- [ ] **Step 1: failing test**
```python
# tests/test_temp_protocol_events.py
from gently.core.event_bus import EventType


def test_new_event_types_exist():
    for n in ("TEMPERATURE_SETPOINT_CHANGED", "TEMP_PROTOCOL_STARTED", "TEMP_PROTOCOL_COMPLETED"):
        assert getattr(EventType, n).name == n


def test_timeline_maps_the_subtypes():
    from gently.harness.session import timeline as tl

    src = open(tl.__file__, encoding="utf-8").read()
    for sub in ("temp_protocol_started", "temp_protocol_completed", "setpoint_changed"):
        assert sub in src
```
- [ ] **Step 2: run, expect FAIL** — `pytest tests/test_temp_protocol_events.py -v`
- [ ] **Step 3: implement** — in `event_bus.py`, alongside the burst events:
```python
TEMPERATURE_SETPOINT_CHANGED = auto()  # discrete setpoint change (timeline)
TEMP_PROTOCOL_STARTED = auto()  # temp-change burst protocol began
TEMP_PROTOCOL_COMPLETED = auto()  # protocol ended
```
In `timeline.py`'s map (mirror the `EventType.BURST_START: {...}` entries), add:
```python
        EventType.TEMPERATURE_SETPOINT_CHANGED: {"category": "temperature", "event_subtype": "setpoint_changed"},
        EventType.TEMP_PROTOCOL_STARTED: {"category": "tactic", "event_subtype": "temp_protocol_started"},
        EventType.TEMP_PROTOCOL_COMPLETED: {"category": "tactic", "event_subtype": "temp_protocol_completed"},
```
> Confirm the real map structure (keys/value shape) from the existing `BURST_START` entry and match it exactly.
- [ ] **Step 4: run, expect PASS**; `pytest -q` no new failures.
- [ ] **Step 5: commit** — `git add gently/core/event_bus.py gently/harness/session/timeline.py tests/test_temp_protocol_events.py && git commit -m "feat(tactic): timeline event types for temp-change burst protocol"`

---

### Task 2: Brightfield burst — thread `laser_config`

**Files:** Modify `gently/app/orchestration/exclusive.py` (`BurstAcquisition.__init__` ~line 95 add param; its `client.acquire_burst(...)` call ~line 159-168 pass it); Modify `gently/app/orchestration/timelapse.py` (`queue_burst` ~line 2058 add `laser_config=None`, pass to `BurstAcquisition`). Test: `tests/test_burst_laser_config.py`.

**Interfaces:** Consumes `client.acquire_burst(..., laser_config=...)` (already accepts it, `client.py:642`). Produces `BurstAcquisition(..., laser_config=None)` and `queue_burst(..., laser_config=None)`.

- [ ] **Step 1: failing test**
```python
# tests/test_burst_laser_config.py
import asyncio
from gently.app.orchestration.exclusive import BurstAcquisition


class FakeClient:
    def __init__(self):
        self.calls = []

    async def acquire_burst(self, **kw):
        self.calls.append(kw)
        return {"success": True, "request_id": "b1", "frames": []}


async def test_burst_passes_laser_config(monkeypatch):
    b = BurstAcquisition("emb1", frames=3, mode="1hz", num_slices=1, laser_config="ALL OFF")
    assert b._laser_config == "ALL OFF"
    # the run() path forwards _laser_config into client.acquire_burst kwargs:
    # (unit-level: assert the attribute + that run threads it — see note)
```
> NOTE: `BurstAcquisition.run` needs an orchestrator with a client + embryo lookup + persistence; a full run is heavy. Assert at minimum that `__init__` stores `_laser_config` and that the `acquire_burst` call site in `run` includes `laser_config=self._laser_config` (verify by reading; optionally add a focused test that monkeypatches the embryo/persistence to capture the `acquire_burst` kwargs). Keep the test honest — if a full run is impractical, assert the attribute and add an inline source check that `laser_config=self._laser_config` appears in the `acquire_burst(...)` call.
- [ ] **Step 2: run, expect FAIL**
- [ ] **Step 3: implement** — add `laser_config: str | None = None` to `__init__`, store `self._laser_config = laser_config`; in the `client.acquire_burst(...)` call add `laser_config=self._laser_config`. In `queue_burst`, add `laser_config: str | None = None` and pass `laser_config=laser_config` into the `BurstAcquisition(...)` construction.
- [ ] **Step 4: run, expect PASS**; `pytest -q` clean.
- [ ] **Step 5: commit** — `feat(tactic): thread laser_config through BurstAcquisition + queue_burst (brightfield bursts)`

---

### Task 3: `wait_for_temperature_lock` helper

**Files:** Create `gently/app/orchestration/temperature_protocol.py` (module for the helper + later the driver). Test: `tests/test_wait_for_lock.py`.

**Interfaces:** Produces `async def wait_for_temperature_lock(client, timeout_s, poll_s=2.0) -> bool`.

- [ ] **Step 1: failing test**
```python
# tests/test_wait_for_lock.py
from gently.app.orchestration.temperature_protocol import wait_for_temperature_lock


class FakeClient:
    def __init__(self, states):
        self.states = list(states)
        self.calls = 0

    async def get_temperature(self):
        i = min(self.calls, len(self.states) - 1)
        self.calls += 1
        return {"state": self.states[i]}


async def test_returns_true_when_locked():
    c = FakeClient(["[ IDLE ]", "[ HEATING ]", "[ SYSTEM LOCKED ]"])
    assert await wait_for_temperature_lock(c, timeout_s=5.0, poll_s=0.001) is True


async def test_returns_false_on_timeout():
    c = FakeClient(["[ HEATING ]"])
    assert await wait_for_temperature_lock(c, timeout_s=0.02, poll_s=0.001) is False
```
- [ ] **Step 2: run, expect FAIL**
- [ ] **Step 3: implement**
```python
# gently/app/orchestration/temperature_protocol.py
import asyncio, logging

logger = logging.getLogger(__name__)


async def wait_for_temperature_lock(client, timeout_s: float, poll_s: float = 2.0) -> bool:
    """Poll the controller until it reports a locked state, or timeout. Substring 'LOCKED'."""
    loop = asyncio.get_event_loop()
    t0 = loop.time()
    while True:
        try:
            resp = await client.get_temperature()
        except Exception as exc:
            logger.warning("wait_for_temperature_lock poll failed: %s", exc)
            resp = {}
        if "LOCKED" in str(resp.get("state", "")):
            return True
        if loop.time() - t0 >= timeout_s:
            return False
        await asyncio.sleep(poll_s)
```
- [ ] **Step 4: run, expect PASS**
- [ ] **Step 5: commit** — `feat(tactic): wait_for_temperature_lock poll helper`

---

### Task 4: The protocol driver

**Files:** Modify `gently/app/orchestration/temperature_protocol.py` (add the driver fn that takes the orchestrator). Test: `tests/test_temp_protocol_driver.py`.

**Interfaces:** Produces `async def run_temp_change_burst_protocol(orchestrator, embryo_id, target_setpoint_c, *, frames=60, mode="1hz", num_slices=1, bursts_before=1, bursts_after=1, lock_timeout_s=600.0, poll_s=2.0, burst_runner=None) -> dict`. `burst_runner` is an injectable `async (BurstAcquisition)->None` for tests (defaults to `lambda b: b.run(orchestrator)`).

- [ ] **Step 1: failing test**
```python
# tests/test_temp_protocol_driver.py
from gently.app.orchestration.temperature_protocol import run_temp_change_burst_protocol
from gently.core.event_bus import EventType


class FakeClient:
    def __init__(self):
        self.laser = None
        self.led = None
        self.setpoint = None
        self._poll = 0

    async def set_laser_config(self, c):
        self.laser = c

    async def set_led(self, s):
        self.led = s

    async def set_temperature(self, t):
        self.setpoint = t

    async def get_temperature(self):
        self._poll += 1
        return {"state": "[ SYSTEM LOCKED ]" if self._poll >= 2 else "[ HEATING ]"}


class FakeOrch:
    def __init__(self, client):
        self._client = client
        self._temperature_provider = lambda: None
        self.events = []

    @property
    def client(self):
        return self._client

    def _emit_event(self, et, data):
        self.events.append((et, data))


async def test_phase_order_and_brightfield(monkeypatch):
    client = FakeClient()
    orch = FakeOrch(client)
    bursts = []

    async def runner(b):
        bursts.append({"phase": getattr(b, "_phase", None), "laser": b._laser_config})

    res = await run_temp_change_burst_protocol(
        orch,
        "emb1",
        25.0,
        frames=3,
        bursts_before=1,
        bursts_after=1,
        lock_timeout_s=5.0,
        poll_s=0.001,
        burst_runner=runner,
    )
    assert client.laser == "ALL OFF" and client.led == "Open"
    assert client.setpoint == 25.0
    assert all(b["laser"] == "ALL OFF" for b in bursts)  # every burst brightfield
    assert len(bursts) >= 3  # before + >=1 during + after
    ets = [e[0] for e in orch.events]
    assert EventType.TEMP_PROTOCOL_STARTED in ets
    assert EventType.TEMPERATURE_SETPOINT_CHANGED in ets
    assert EventType.TEMP_PROTOCOL_COMPLETED in ets
    assert res["locked"] is True
```
- [ ] **Step 2: run, expect FAIL**
- [ ] **Step 3: implement** — append to `temperature_protocol.py`:
```python
from gently.app.orchestration.exclusive import BurstAcquisition
from gently.core.event_bus import EventType


async def run_temp_change_burst_protocol(
    orchestrator,
    embryo_id,
    target_setpoint_c,
    *,
    frames=60,
    mode="1hz",
    num_slices=1,
    bursts_before=1,
    bursts_after=1,
    lock_timeout_s=600.0,
    poll_s=2.0,
    burst_runner=None,
):
    client = orchestrator.client
    if burst_runner is None:

        async def burst_runner(b):
            await b.run(orchestrator)

    async def one_burst(phase):
        b = BurstAcquisition(
            embryo_id,
            frames=frames,
            mode=mode,
            num_slices=num_slices,
            temperature_provider=getattr(orchestrator, "_temperature_provider", None),
            laser_config="ALL OFF",
        )
        b._phase = phase
        await burst_runner(b)

    locked = False
    error = None
    cancelled = False
    try:
        await client.set_laser_config("ALL OFF")
        await client.set_led("Open")
        orchestrator._emit_event(
            EventType.TEMP_PROTOCOL_STARTED,
            {
                "embryo_id": embryo_id,
                "target_setpoint_c": target_setpoint_c,
                "frames": frames,
                "bursts_before": bursts_before,
                "bursts_after": bursts_after,
            },
        )
        for _ in range(bursts_before):
            await one_burst("before")
        await client.set_temperature(target_setpoint_c)
        orchestrator._emit_event(
            EventType.TEMPERATURE_SETPOINT_CHANGED,
            {"embryo_id": embryo_id, "to": target_setpoint_c},
        )
        loop = asyncio.get_event_loop()
        t0 = loop.time()
        while True:
            await one_burst("during")
            try:
                st = str((await client.get_temperature()).get("state", ""))
            except Exception:
                st = ""
            if "LOCKED" in st:
                locked = True
                break
            if loop.time() - t0 >= lock_timeout_s:
                break
        for _ in range(bursts_after):
            await one_burst("after")
    except asyncio.CancelledError:
        cancelled = True
        raise
    except Exception as exc:
        error = str(exc)
        logger.exception("temp-change burst protocol failed")
    finally:
        orchestrator._emit_event(
            EventType.TEMP_PROTOCOL_COMPLETED,
            {"embryo_id": embryo_id, "locked": locked, "cancelled": cancelled, "error": error},
        )
    return {"locked": locked, "cancelled": cancelled, "error": error}
```
- [ ] **Step 4: run, expect PASS**; `pytest -q` clean.
- [ ] **Step 5: commit** — `feat(tactic): temp-change burst protocol driver (brightfield before/during/after)`

---

### Task 5: Agent tool

**Files:** Create `gently/app/tools/temperature_protocol_tools.py` (or add to an existing tools module — follow the `@tool` pattern). Test: `tests/test_temp_protocol_tool.py`.

**Interfaces:** Produces a `run_temp_change_burst_protocol` agent tool that resolves orchestrator+client from `context`, launches the driver via `asyncio.create_task`, returns a started message; validates embryo/client presence.

- [ ] **Step 1: failing test** — assert the tool, given a context with a fake orchestrator/client, creates a task and returns a "started" string; given no client, returns an error without creating a task.
> Confirm the real `@tool` decorator + context helpers (`ctx_get(context,"client")`, `require_agent`, `require_timelapse_orchestrator`) from an existing tool (e.g. `gently/app/tools/temperature_tools.py`); mirror them. Write the test to the real registration shape.
- [ ] **Step 2: run, expect FAIL**
- [ ] **Step 3: implement** — mirror an existing tool: resolve `orchestrator` + `client`, guard None (return error dict/string), `asyncio.create_task(run_temp_change_burst_protocol(orchestrator, embryo_id, target_setpoint_c, frames=frames, bursts_before=bursts_before, bursts_after=bursts_after))`, return `f"Temp-change burst protocol started for {embryo_id} → {target_setpoint_c} C"`.
- [ ] **Step 4: run, expect PASS**
- [ ] **Step 5: commit** — `feat(tactic): run_temp_change_burst_protocol agent tool`

---

### Task 6: Observability in strategy_snapshot

**Files:** Modify `gently/ui/web/strategy_snapshot.py` (`_replay_timeline` ~line 627; mirror the burst-phase handling at ~762/777). Test: `tests/test_temp_protocol_snapshot.py`.

**Interfaces:** Consumes the timeline subtypes (`temp_protocol_started/completed`, `setpoint_changed`). Produces, in the snapshot: a `temp_protocol` band (open on started, close on completed) and a `setpoint_changes` list.

- [ ] **Step 1: failing test** — feed `_replay_timeline` (or `build_strategy_snapshot` over a temp `timeline.jsonl`) a sequence: `temp_protocol_started`, `setpoint_changed(to=25)`, `burst_started/completed`, `temp_protocol_completed`; assert the snapshot exposes a temp_protocol span and a setpoint change of 25.
> Confirm the real `_replay_timeline` input/output shape from the existing burst handling; match the snapshot dict structure it already produces.
- [ ] **Step 2: run, expect FAIL**
- [ ] **Step 3: implement** — in `_replay_timeline`, handle the three subtypes: on `temp_protocol_started` open a band (record start t + params); on `temp_protocol_completed` close it; on `setpoint_changed` append `{t, to}` to a `setpoint_changes` list in the snapshot. Surface them in the returned snapshot dict next to the existing phases.
- [ ] **Step 4: run, expect PASS**; `pytest -q` clean.
- [ ] **Step 5: commit** — `feat(tactic): surface temp-protocol band + setpoint changes in strategy snapshot`

---

## Self-Review
- §2.1 brightfield burst → Task 2; §2.2 wait-for-lock → Task 3; §2.3 driver → Task 4; §2.4 events → Tasks 1 & 6; §2.5 tool → Task 5. ✓
- Open confirmations (explicit): the timeline map value-shape (Task 1), the burst `run` acquire_burst call site (Task 2), the `@tool` + context helpers (Task 5), the `_replay_timeline` shape (Task 6). Each names a fallback.
- Type consistency: `laser_config="ALL OFF"` everywhere; event names `TEMPERATURE_SETPOINT_CHANGED`/`TEMP_PROTOCOL_STARTED`/`TEMP_PROTOCOL_COMPLETED` across Tasks 1/4/6; driver returns `{locked,cancelled,error}`.
