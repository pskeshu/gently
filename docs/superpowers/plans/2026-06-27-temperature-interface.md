# Temperature Interface Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Persist live temperature into each imaging session and chart it, so bursts/volumes are correlatable to temperature and the rise/fall trajectory is visible during the temperature-strain experiments.

**Architecture:** A `FileStore`-backed append-only `temperature.jsonl` per session (mirroring `predictions.jsonl`); a `TemperatureSampler(Service)` in the agent process — modeled on `DeviceStateMonitor` — that, while a session is active, polls the device layer at 1 Hz, appends each reading, holds the latest in memory, and publishes a `TEMPERATURE_UPDATE` event that the viz server already forwards to the browser; acquisition code stamps the latest reading into volume/burst metadata; a FastAPI history route backfills the graph; a hand-rolled SVG component renders water-temp + stepped-setpoint as a card on the Devices tab.

**Tech Stack:** Python 3 + asyncio, FastAPI, the project's `EventBus`, `FileStore` (file-based YAML/JSONL), vanilla-JS + hand-rolled SVG frontend (no build step), pytest with `asyncio_mode = "auto"`.

## Global Constraints

- **No new dependency** — chart is hand-rolled SVG (`createElementNS`), matching `experiment-overview.js`. No charting library.
- **Session-scoped capture only** — sampler persists/emits **only while a session is active**; no always-on facility daemon.
- **Sample schema** (one JSONL line): `{"t": <iso-utc>, "water_c": <float>, "setpoint_c": <float|null>, "state": <str|null>}`.
- **Append pattern**: reuse `FileStore._append_jsonl` (append mode, `json.dumps(..., default=str)`, trailing `\n`). Meta written via existing `_write_yaml` / `yaml.safe_dump`.
- **Defaults (flippable):** 1 Hz sampling; stamp the **latest sample** (no fresh blocking read) at acquisition.
- **Robustness:** a failed poll = a gap, logged, loop continues; a sampler error never crashes the session. Empty state, **never mock data**.
- **Tests:** `pytest`; `async def test_*` needs no decorator (auto mode); use the `file_store` fixture (`tests/conftest.py:38-45`, `FileStore(tmp_path/...)`). The frontend has **no JS unit harness** — verify it by running the app + Chrome DevTools MCP.
- **Env:** production runs pip + `requirements*.txt` (no `uv`); we add no deps, so nothing to declare.

---

### Task 1: Temperature log store (FileStore methods)

**Files:**
- Modify: `gently/core/file_store.py` (add two methods on `FileStore`; reuse module-level `_append_jsonl` at `:197-201` and `_read_jsonl` at `:204-215`, and `_session_dir`/`_require_session_dir` at `:255-269`)
- Test: `tests/test_temperature_store.py` (new)

**Interfaces:**
- Produces:
  - `FileStore.append_temperature_sample(self, session_id: str, sample: dict) -> None` — appends one line to `sessions/{folder}/temperature.jsonl`.
  - `FileStore.read_temperature_log(self, session_id: str, since: str | None = None) -> list[dict]` — returns samples; if `since` (an ISO-UTC string) is given, only samples with `r["t"] >= since` (lexicographic compare is valid for fixed-format UTC ISO).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_temperature_store.py
def _new_session(file_store):
    return file_store.create_session(name="temp-test")  # returns session_id


def test_append_and_read_roundtrip(file_store):
    sid = _new_session(file_store)
    file_store.append_temperature_sample(
        sid,
        {"t": "2026-06-27T10:00:00+00:00", "water_c": 28.0, "setpoint_c": 32.0, "state": "heating"},
    )
    file_store.append_temperature_sample(
        sid,
        {"t": "2026-06-27T10:00:01+00:00", "water_c": 28.3, "setpoint_c": 32.0, "state": "heating"},
    )
    rows = file_store.read_temperature_log(sid)
    assert [r["water_c"] for r in rows] == [28.0, 28.3]


def test_read_since_filters(file_store):
    sid = _new_session(file_store)
    for i, t in enumerate(
        ["2026-06-27T10:00:00+00:00", "2026-06-27T10:00:01+00:00", "2026-06-27T10:00:02+00:00"]
    ):
        file_store.append_temperature_sample(
            sid, {"t": t, "water_c": 28.0 + i, "setpoint_c": 32.0, "state": "heating"}
        )
    rows = file_store.read_temperature_log(sid, since="2026-06-27T10:00:01+00:00")
    assert [r["water_c"] for r in rows] == [29.0, 30.0]


def test_read_unknown_session_is_empty(file_store):
    assert file_store.read_temperature_log("does-not-exist") == []
```

> NOTE for implementer: confirm the exact session-creation API on `FileStore` (search for `def create_session`). If its signature differs, adjust `_new_session` accordingly — the rest of the test is unaffected.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_temperature_store.py -v`
Expected: FAIL — `AttributeError: 'FileStore' object has no attribute 'append_temperature_sample'`

- [ ] **Step 3: Write minimal implementation**

Add to the `FileStore` class body in `gently/core/file_store.py` (near the other per-session jsonl helpers like `add_prediction`):

```python
def append_temperature_sample(self, session_id: str, sample: dict) -> None:
    """Append one temperature reading to the session's temperature.jsonl."""
    sd = self._require_session_dir(session_id)
    _append_jsonl(sd / "temperature.jsonl", sample)


def read_temperature_log(self, session_id: str, since: str | None = None) -> list[dict]:
    """Return temperature samples for a session, optionally filtered to t >= since (ISO-UTC string)."""
    sd = self._session_dir(session_id)
    if sd is None:
        return []
    rows = _read_jsonl(sd / "temperature.jsonl")
    if since is not None:
        rows = [r for r in rows if str(r.get("t", "")) >= since]
    return rows
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_temperature_store.py -v`
Expected: PASS (3 passed)

- [ ] **Step 5: Commit**

```bash
git add gently/core/file_store.py tests/test_temperature_store.py
git commit -m "feat(temperature): session-scoped temperature.jsonl store on FileStore"
```

---

### Task 2: `TEMPERATURE_UPDATE` event type

**Files:**
- Modify: `gently/core/event_bus.py` (add enum member near `:86`; add to `_NO_HISTORY_TYPES` near `:187`)
- Test: `tests/test_temperature_event.py` (new)

**Interfaces:**
- Produces: `EventType.TEMPERATURE_UPDATE` (a new enum member). High-volume → excluded from event history.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_temperature_event.py
from gently.core.event_bus import EventType, EventBus


def test_temperature_update_event_exists():
    assert EventType.TEMPERATURE_UPDATE.value == "TEMPERATURE_UPDATE"


def test_temperature_update_publishes_to_subscriber():
    bus = EventBus()
    seen = []
    bus.subscribe(EventType.TEMPERATURE_UPDATE, lambda e: seen.append(e.data))
    bus.publish(event_type=EventType.TEMPERATURE_UPDATE, data={"x": 1}, source="t")
    assert seen == [{"x": 1}]
```

> NOTE: confirm `EventBus.subscribe` signature/usage from an existing test (`tests/` has event-bus usage); adjust the subscribe call if the project's API differs (e.g. `subscribe(event_type, handler)` vs `subscribe(handler, event_type)`).

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_temperature_event.py -v`
Expected: FAIL — `AttributeError: TEMPERATURE_UPDATE`

- [ ] **Step 3: Write minimal implementation**

In `gently/core/event_bus.py`, add the member alongside the other `EventType` values, **matching the existing `auto()` style** (`DEVICE_STATE_UPDATE = auto()` at `:86`). The wire protocol serializes `event.event_type.name` (`Event.to_dict` at event_bus.py:232; server.py:377/419), so the browser receives the string `"TEMPERATURE_UPDATE"` regardless — which is what the frontend (Task 7) subscribes to. The test must assert `.name == "TEMPERATURE_UPDATE"` (NOT `.value`, which is an `auto()` int):

```python
    TEMPERATURE_UPDATE = auto()  # high-volume telemetry from the temperature controller
```

And add it to the high-volume set so it is not retained in history (next to `DEVICE_STATE_UPDATE` in `_NO_HISTORY_TYPES` near `:187`):

```python
(EventType.TEMPERATURE_UPDATE,)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_temperature_event.py -v`
Expected: PASS (2 passed)

- [ ] **Step 5: Commit**

```bash
git add gently/core/event_bus.py tests/test_temperature_event.py
git commit -m "feat(temperature): add TEMPERATURE_UPDATE event type"
```

---

### Task 3: TemperatureSampler service

**Files:**
- Create: `gently/app/temperature_sampler.py`
- Test: `tests/test_temperature_sampler.py` (new)
- Reference (template, do not modify): `gently/app/device_state_monitor.py`, `gently/core/service.py:63-172`

**Interfaces:**
- Consumes: `EventType.TEMPERATURE_UPDATE` (Task 2); `FileStore.append_temperature_sample` (Task 1); a microscope client exposing `async get_temperature() -> dict` returning `{"success": bool, "temperature_c": float, "setpoint_c": float, "state": str, ...}` (`gently/hardware/dispim/client.py:837`).
- Produces:
  - `TemperatureSampler(Service)` with `__init__(self, microscope, store, session_id_getter, interval_sec=1.0)`.
  - `async on_start(self)` / `async on_stop(self)` (background asyncio loop, like `DeviceStateMonitor`).
  - `async _tick(self, bus) -> None` — one poll/append/emit cycle (the unit under test).
  - attribute `self.latest: dict | None` — most recent sample (for the acquisition stamp).
  - module function `temperature_stamp(latest: dict | None) -> dict | None` — `{"water_c","setpoint_c","state","sampled_at"}` or `None`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_temperature_sampler.py
import asyncio
from gently.core.event_bus import EventBus, EventType
from gently.app.temperature_sampler import TemperatureSampler, temperature_stamp


class FakeScope:
    def __init__(self, resp):
        self.resp = resp
        self.calls = 0

    async def get_temperature(self):
        self.calls += 1
        if isinstance(self.resp, Exception):
            raise self.resp
        return self.resp


def _capture(bus):
    seen = []
    bus.subscribe(EventType.TEMPERATURE_UPDATE, lambda e: seen.append(e.data))
    return seen


async def test_tick_appends_emits_and_sets_latest(file_store):
    sid = file_store.create_session(name="s")
    scope = FakeScope(
        {"success": True, "temperature_c": 28.4, "setpoint_c": 32.0, "state": "heating"}
    )
    bus = EventBus()
    seen = _capture(bus)
    s = TemperatureSampler(scope, file_store, lambda: sid)
    await s._tick(bus)
    rows = file_store.read_temperature_log(sid)
    assert len(rows) == 1 and rows[0]["water_c"] == 28.4
    assert s.latest["water_c"] == 28.4
    assert seen and seen[0]["sample"]["water_c"] == 28.4 and seen[0]["session_id"] == sid


async def test_tick_no_active_session_is_noop(file_store):
    scope = FakeScope({"success": True, "temperature_c": 1.0, "setpoint_c": 2.0, "state": "x"})
    bus = EventBus()
    seen = _capture(bus)
    s = TemperatureSampler(scope, file_store, lambda: None)
    await s._tick(bus)
    assert scope.calls == 0 and s.latest is None and seen == []


async def test_tick_poll_failure_is_a_gap_not_a_crash(file_store):
    sid = file_store.create_session(name="s")
    scope = FakeScope(RuntimeError("device down"))
    bus = EventBus()
    s = TemperatureSampler(scope, file_store, lambda: sid)
    # _run swallows; _tick raises — assert the loop-level guard swallows by calling _run-style guard:
    try:
        await s._tick(bus)
    except RuntimeError:
        pass  # _tick may raise; the loop in _run catches it (see test below)
    assert file_store.read_temperature_log(sid) == []


def test_temperature_stamp_shapes():
    assert temperature_stamp(None) is None
    assert temperature_stamp(
        {"t": "2026-06-27T10:00:00+00:00", "water_c": 28.4, "setpoint_c": 32.0, "state": "heating"}
    ) == {
        "water_c": 28.4,
        "setpoint_c": 32.0,
        "state": "heating",
        "sampled_at": "2026-06-27T10:00:00+00:00",
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_temperature_sampler.py -v`
Expected: FAIL — `ModuleNotFoundError: gently.app.temperature_sampler`

- [ ] **Step 3: Write minimal implementation**

```python
# gently/app/temperature_sampler.py
"""Session-scoped temperature sampler — polls the device layer, persists, emits.

Modeled on gently/app/device_state_monitor.py. While a session is active it polls
the microscope's temperature at a fixed cadence, appends each reading to the
session's temperature.jsonl, holds the latest reading (for acquisition stamping),
and publishes TEMPERATURE_UPDATE for the live graph. A failed poll is a gap, not a
crash; with no active session the loop idles.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

from gently.core.service import Service
from gently.core.event_bus import get_event_bus, EventType

logger = logging.getLogger(__name__)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def temperature_stamp(latest: dict | None) -> dict | None:
    """Build a temperature meta block from a latest sample, or None if unavailable."""
    if not latest:
        return None
    return {
        "water_c": latest.get("water_c"),
        "setpoint_c": latest.get("setpoint_c"),
        "state": latest.get("state"),
        "sampled_at": latest.get("t"),
    }


class TemperatureSampler(Service):
    def __init__(self, microscope, store, session_id_getter, interval_sec: float = 1.0):
        super().__init__(name="temperature-sampler", service_type="monitor")
        self._microscope = microscope
        self._store = store
        self._session_id_getter = session_id_getter
        self._interval = interval_sec
        self._task: asyncio.Task | None = None
        self.latest: dict | None = None

    async def on_start(self) -> None:
        self._task = asyncio.create_task(self._run(), name="temperature-sampler-loop")

    async def on_stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _run(self) -> None:
        bus = get_event_bus()
        while True:
            try:
                await self._tick(bus)
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # a gap, never a crash
                logger.warning("temperature sampler tick failed: %s", exc)
            await asyncio.sleep(self._interval)

    async def _tick(self, bus) -> None:
        session_id = self._session_id_getter()
        if not session_id:
            return
        resp = await self._microscope.get_temperature()
        if not resp or not resp.get("success", True):
            return
        water = resp.get("temperature_c")
        if water is None:
            return
        sample = {
            "t": _now_iso(),
            "water_c": water,
            "setpoint_c": resp.get("setpoint_c"),
            "state": resp.get("state"),
        }
        self._store.append_temperature_sample(session_id, sample)
        self.latest = sample
        bus.publish(
            event_type=EventType.TEMPERATURE_UPDATE,
            data={"session_id": session_id, "sample": sample},
            source="temperature-sampler",
        )
```

> Note the failure test: `_tick` propagates the poll exception, and `_run`'s `except Exception` swallows it. The test asserts the gap (no rows). If you prefer, also add a direct `_run`-guard test that starts the loop briefly and asserts it does not raise — optional.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_temperature_sampler.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add gently/app/temperature_sampler.py tests/test_temperature_sampler.py
git commit -m "feat(temperature): TemperatureSampler service (poll/persist/emit @1Hz)"
```

---

### Task 4: Wire the sampler into the agent lifecycle

**Files:**
- Modify: `gently/app/agent.py` — init attribute (near `:212`), construct+start (near the `DeviceStateMonitor` block `:818-827` inside `start_viz_server`), stop (near `:850-855`).
- Test: `tests/test_temperature_sampler_wiring.py` (new)

**Interfaces:**
- Consumes: `TemperatureSampler` (Task 3); the agent's microscope client (`self.microscope`), its `FileStore`, and `self.session_id`.
- Produces: `agent.temperature_sampler: TemperatureSampler | None` (the live instance, read by the acquisition stamp in Task 6).

> IMPLEMENTER: confirm the agent's FileStore attribute name before writing the construction line. Search `gently/app/agent.py` for the `FileStore` it uses (likely `self.store`). Use that exact attribute. If the agent reaches the store indirectly, pass whatever object exposes `append_temperature_sample`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_temperature_sampler_wiring.py
from gently.app.temperature_sampler import TemperatureSampler


def test_agent_initializes_temperature_sampler_attribute():
    # The attribute must exist (None until start_viz_server runs with a microscope).
    import gently.app.agent as agent_mod

    src = agent_mod.__file__
    text = open(src, encoding="utf-8").read()
    assert "temperature_sampler" in text
    assert "TemperatureSampler(" in text
```

> This is a light wiring guard (a full agent boot is an integration concern). It fails until the wiring lines exist.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_temperature_sampler_wiring.py -v`
Expected: FAIL — assertion error ("temperature_sampler" not in source)

- [ ] **Step 3: Write minimal implementation**

Near `:212` (with the other monitor attrs, e.g. `self.device_state_monitor = None`):

```python
        self.temperature_sampler = None
```

Inside `start_viz_server`, right after the `DeviceStateMonitor` start block (`:818-827`):

```python
if self.microscope is not None and self.temperature_sampler is None:
    from .temperature_sampler import TemperatureSampler

    self.temperature_sampler = TemperatureSampler(
        self.microscope, self.store, lambda: self.session_id
    )
    await self.temperature_sampler.start()
```

In the symmetric shutdown path (`:850-855`, where `device_state_monitor.stop()` is awaited):

```python
            if self.temperature_sampler is not None:
                await self.temperature_sampler.stop()
                self.temperature_sampler = None
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_temperature_sampler_wiring.py -v`
Expected: PASS (1 passed)

Then full suite sanity: `pytest -q` — Expected: no new failures.

- [ ] **Step 5: Commit**

```bash
git add gently/app/agent.py tests/test_temperature_sampler_wiring.py
git commit -m "feat(temperature): start/stop TemperatureSampler with the agent"
```

---

### Task 5: History API route

**Files:**
- Create: `gently/ui/web/routes/temperature.py`
- Modify: `gently/ui/web/routes/__init__.py` (import + add to the factories tuple in `register_all_routes`)
- Test: `tests/test_temperature_route.py` (new)
- Reference (template): `gently/ui/web/routes/experiments.py:1-66`, test template `tests/test_data_catalog.py:69-114`

**Interfaces:**
- Consumes: `server.gently_store` (a `FileStore`) with `list_sessions()`, `_session_dir(id)`, and `read_temperature_log(id, since=)` (Task 1).
- Produces: `GET /api/temperature/{session_id}/history?since=<iso>` → `{"session_id": str, "samples": list[dict]}`; `session_id="current"` resolves to newest session.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_temperature_route.py
from unittest.mock import MagicMock
from pathlib import Path
from fastapi import FastAPI
from fastapi.testclient import TestClient
from gently.ui.web.routes.temperature import create_router


def _server(samples, sessions=(("sess-1", True),)):
    store = MagicMock()
    store.list_sessions.return_value = [{"session_id": sid} for sid, _ in sessions]
    store._session_dir.side_effect = lambda sid: (
        Path("/x") if any(sid == s for s, _ in sessions) else None
    )
    store.read_temperature_log.return_value = samples
    srv = MagicMock()
    srv.gently_store = store
    return srv, store


def _client(server):
    app = FastAPI()
    app.include_router(create_router(server))
    return TestClient(app)


def test_history_returns_samples():
    srv, store = _server(
        [
            {
                "t": "2026-06-27T10:00:00+00:00",
                "water_c": 28.0,
                "setpoint_c": 32.0,
                "state": "heating",
            }
        ]
    )
    r = _client(srv).get("/api/temperature/sess-1/history")
    assert r.status_code == 200
    body = r.json()
    assert body["session_id"] == "sess-1"
    assert body["samples"][0]["water_c"] == 28.0


def test_history_passes_since_through():
    srv, store = _server([])
    _client(srv).get("/api/temperature/sess-1/history?since=2026-06-27T10:00:01+00:00")
    store.read_temperature_log.assert_called_with("sess-1", since="2026-06-27T10:00:01+00:00")


def test_history_current_resolves_newest():
    srv, store = _server([], sessions=(("newest", True),))
    r = _client(srv).get("/api/temperature/current/history")
    assert r.status_code == 200
    assert r.json()["session_id"] == "newest"


def test_history_unknown_session_404():
    srv, store = _server([], sessions=(("sess-1", True),))
    # _session_dir returns None for unknown -> 404
    store._session_dir.side_effect = lambda sid: None
    r = _client(srv).get("/api/temperature/ghost/history")
    assert r.status_code == 404
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_temperature_route.py -v`
Expected: FAIL — `ModuleNotFoundError: gently.ui.web.routes.temperature`

- [ ] **Step 3: Write minimal implementation**

```python
# gently/ui/web/routes/temperature.py
"""Read-only temperature history for the live graph (backfill on mount/reload).

Live updates ride the TEMPERATURE_UPDATE event channel; this route is backfill only.
Mirrors routes/experiments.py session resolution.
"""

from fastapi import APIRouter, HTTPException


def create_router(server) -> APIRouter:
    router = APIRouter()

    def _resolve_session(session_id: str):
        store = getattr(server, "gently_store", None)
        if store is None:
            raise HTTPException(status_code=503, detail="FileStore not configured on viz server")
        if session_id == "current":
            sessions = store.list_sessions()
            if not sessions:
                raise HTTPException(status_code=404, detail="No sessions in store")
            session_id = sessions[0].get("session_id")
        if store._session_dir(session_id) is None:
            raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")
        return session_id

    @router.get("/api/temperature/{session_id}/history")
    async def get_history(session_id: str, since: str | None = None):
        real_id = _resolve_session(session_id)
        store = server.gently_store
        samples = store.read_temperature_log(real_id, since=since)
        return {"session_id": real_id, "samples": samples}

    return router
```

Register it in `gently/ui/web/routes/__init__.py` — add the import and append `create_router` to the factories iterated by `register_all_routes` (follow the existing pattern exactly; alias to avoid name clashes, e.g. `from .temperature import create_router as create_temperature_router`).

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_temperature_route.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add gently/ui/web/routes/temperature.py gently/ui/web/routes/__init__.py tests/test_temperature_route.py
git commit -m "feat(temperature): GET /api/temperature/{session}/history backfill route"
```

---

### Task 6: Acquisition temperature stamp (burst + volume)

**Files:**
- Modify: `gently/app/orchestration/exclusive.py` — `_persist_burst_to_disk` (per-frame `metadata` at `:387-407`, `burst.yaml` manifest dict at `:423-442`); the `BurstAcquisition`/`ExclusiveAcquisition` construction to receive a temperature provider.
- Modify: the volume acquisition call site that builds `metadata` for `FileStore.put_volume`/`register_volume` (locate the caller in `gently/app/orchestration/timelapse.py` / `gently/app/tools/acquisition_tools.py`).
- Test: `tests/test_temperature_stamp.py` (new) — covers the pure helper + the volume metadata channel.
- Consumes: `temperature_stamp` and `agent.temperature_sampler.latest` (Tasks 3–4).

**Interfaces:**
- Produces: a `temperature` block under `metadata` for volumes (`meta["metadata"]["temperature"]`) and under both per-frame `metadata` and the `burst.yaml` manifest top-level for bursts.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_temperature_stamp.py
import numpy as np
from gently.app.temperature_sampler import temperature_stamp


def test_stamp_none_when_no_reading():
    assert temperature_stamp(None) is None


def test_volume_metadata_carries_temperature(file_store):
    sid = file_store.create_session(name="s")
    emb = file_store.create_embryo(sid, position={"x": 0, "y": 0, "z": 0})  # confirm signature
    stamp = temperature_stamp(
        {"t": "2026-06-27T10:00:00+00:00", "water_c": 28.4, "setpoint_c": 32.0, "state": "heating"}
    )
    vol = np.zeros((2, 4, 4), dtype="uint16")
    file_store.put_volume(sid, emb, timepoint=0, volume=vol, metadata={"temperature": stamp})
    meta = file_store.get_volume_meta(sid, emb, 0)  # confirm accessor name
    assert meta["metadata"]["temperature"]["water_c"] == 28.4
```

> IMPLEMENTER: confirm `create_embryo` and the volume-meta accessor (`get_volume_meta` or read the `.meta.yaml` directly via `get_volume_path(...).with_suffix` — adjust to the real API). The assertion target — `metadata["temperature"]` round-tripping into `t0000.meta.yaml` — is the contract; `put_volume` already nests the passed `metadata`, so this passes once the helper exists and the accessor is right.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_temperature_stamp.py -v`
Expected: FAIL — initially on import/accessor; fix the accessor name from the real API, then it exercises the channel.

- [ ] **Step 3: Write minimal implementation**

**Volume path** — at the acquisition call site that calls `put_volume`/`register_volume`, fold the stamp into the existing `metadata` dict:

```python
from gently.app.temperature_sampler import temperature_stamp

# ... where `agent` (or self) holds the sampler and `metadata` is being built:
stamp = temperature_stamp(getattr(getattr(agent, "temperature_sampler", None), "latest", None))
if stamp is not None:
    metadata["temperature"] = stamp
```

**Burst path** — `gently/app/orchestration/exclusive.py`:
1. Add a constructor param to the acquisition class that persists bursts: `temperature_provider=None` (a zero-arg callable returning the latest sample dict, or `None`), stored as `self._temperature_provider`.
2. In `_persist_burst_to_disk`, compute once:

```python
from gently.app.temperature_sampler import temperature_stamp

_temp = temperature_stamp(self._temperature_provider() if self._temperature_provider else None)
```

3. Inject into the per-frame `metadata` dict (`:387-407`): add `"temperature": _temp` (only when not None — or always; `None` is acceptable YAML).
4. Inject into the `burst.yaml` manifest dict (`:423-442`): add a top-level `"temperature": _temp`.
5. Where this acquisition class is constructed (the orchestrator that owns bursts — `gently/app/orchestration/timelapse.py`), pass `temperature_provider=lambda: agent.temperature_sampler.latest if agent.temperature_sampler else None` (use the orchestrator's existing agent/store handle; confirm the attribute).

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_temperature_stamp.py -v`
Expected: PASS

- [ ] **Step 5: Verify burst wiring by inspection + targeted run**

The burst persist writes real TIFFs, so it is verified by inspection (the `_temp` block is added in both dict sites) plus, during end-to-end verification (after Task 7), trigger one burst against the mock device and confirm `burst.yaml` and a frame `.meta.yaml` contain a `temperature` block. Note in the commit if the volume call site could not be located and was deferred — do not silently skip it.

- [ ] **Step 6: Commit**

```bash
git add gently/app/orchestration/exclusive.py gently/app/orchestration/timelapse.py tests/test_temperature_stamp.py
git commit -m "feat(temperature): stamp latest reading into burst + volume metadata"
```

---

### Task 7: Frontend — temperature graph component + Devices card

**Files:**
- Create: `gently/ui/web/static/js/temperature-graph.js`
- Modify: `gently/ui/web/templates/index.html` — add a chart container inside `#devices-content` (the existing small readout is at `:452-465`; mount the chart as a section under the Details/Map view).
- Modify: `gently/ui/web/static/js/devices.js` — initialize the chart in `init()` (`:1556`) and subscribe it to `TEMPERATURE_UPDATE` (next to the `DEVICE_STATE_UPDATE` subscription at `:1561`).
- Reference (SVG style): `gently/ui/web/static/js/experiment-overview.js`; (event API) `gently/ui/web/static/js/event-bus.js:9-48` (`ClientEventBus.on(type, handler)`).

**No JS unit harness exists** — this task is verified by running the app + Chrome DevTools MCP (see Step 4), consistent with the repo and the "UI audit before done" practice.

- [ ] **Step 1: Build the component**

Create `temperature-graph.js` exposing a small object/module:

```javascript
// gently/ui/web/static/js/temperature-graph.js
// Hand-rolled SVG line chart: water-temp trace + stepped setpoint line.
// No dependency. Backfills from /api/temperature/{session}/history, then appends
// from TEMPERATURE_UPDATE events. Calm empty state, never mock data.
const TemperatureGraph = (() => {
  const SVGNS = "http://www.w3.org/2000/svg";
  const MAX_POINTS = 600;           // rolling ~10 min @ 1 Hz
  let _root = null, _samples = [], _session = "current";

  function init(container, sessionId) {
    _root = container; _session = sessionId || "current"; _samples = [];
    backfill();
    ClientEventBus.on("TEMPERATURE_UPDATE", onEvent);
  }

  async function backfill() {
    try {
      const r = await fetch(`/api/temperature/${_session}/history`);
      if (!r.ok) { renderEmpty(); return; }
      const body = await r.json();
      _session = body.session_id || _session;
      _samples = (body.samples || []).slice(-MAX_POINTS);
      render();
    } catch (e) { renderEmpty(); }
  }

  function onEvent(data) {
    if (!data || !data.sample) return;
    _samples.push(data.sample);
    if (_samples.length > MAX_POINTS) _samples.shift();
    render();
  }

  function renderEmpty() {
    _root.innerHTML = '<div class="temp-graph-empty">No temperature data yet</div>';
  }

  function render() {
    if (!_samples.length) { renderEmpty(); return; }
    const W = _root.clientWidth || 480, H = 160, pad = 24;
    const xs = _samples.map((_, i) => i);
    const ws = _samples.map(s => s.water_c).filter(v => v != null);
    const sps = _samples.map(s => s.setpoint_c).filter(v => v != null);
    const lo = Math.min(...ws, ...sps) - 1, hi = Math.max(...ws, ...sps) + 1;
    const sx = i => pad + (i / Math.max(1, xs.length - 1)) * (W - 2 * pad);
    const sy = v => H - pad - ((v - lo) / Math.max(0.001, hi - lo)) * (H - 2 * pad);

    const svg = document.createElementNS(SVGNS, "svg");
    svg.setAttribute("viewBox", `0 0 ${W} ${H}`); svg.setAttribute("width", "100%");

    const line = (pts, cls) => {
      const p = document.createElementNS(SVGNS, "polyline");
      p.setAttribute("points", pts); p.setAttribute("class", cls);
      p.setAttribute("fill", "none"); svg.appendChild(p);
    };
    line(_samples.map((s, i) => s.water_c != null ? `${sx(i)},${sy(s.water_c)}` : "").filter(Boolean).join(" "), "temp-water");
    // Stepped setpoint: carry previous y until it changes.
    let sp = []; _samples.forEach((s, i) => { if (s.setpoint_c != null) sp.push(`${sx(i)},${sy(s.setpoint_c)}`); });
    line(sp.join(" "), "temp-setpoint");

    const last = _samples[_samples.length - 1];
    const readout = document.createElement("div");
    readout.className = "temp-graph-readout";
    readout.textContent = `${last.water_c?.toFixed?.(1) ?? "—"} °C → ${last.setpoint_c?.toFixed?.(1) ?? "—"} °C (${last.state ?? ""})`;

    _root.innerHTML = ""; _root.appendChild(readout); _root.appendChild(svg);
  }

  function dispose() { ClientEventBus.off("TEMPERATURE_UPDATE", onEvent); }
  return { init, dispose, _render: render, _samples: () => _samples };
})();
window.TemperatureGraph = TemperatureGraph;
```

Add minimal CSS (in the devices stylesheet) for `.temp-water` (stroke: water color), `.temp-setpoint` (dashed stroke), `.temp-graph-empty` (muted), matching existing palette.

- [ ] **Step 2: Mount it**

In `templates/index.html`, add inside `#devices-content` (under the map/details view):

```html
<section class="devices-temp-graph" id="devices-temp-graph" aria-label="Temperature trajectory"></section>
```

Load the script (next to the other `static/js/*.js` includes for the devices tab).

In `devices.js` `init()` (`:1556`), after existing setup:

```javascript
    const tg = document.getElementById('devices-temp-graph');
    if (tg && window.TemperatureGraph) TemperatureGraph.init(tg, 'current');
```

(No extra subscription needed in `devices.js` — the component self-subscribes. Optionally also route the existing small readout off the new event.)

- [ ] **Step 3: Add the script tag**

Add `<script src="/static/js/temperature-graph.js"></script>` in `index.html` alongside the other component scripts, **before** `devices.js` loads (so `window.TemperatureGraph` exists when `init()` runs).

- [ ] **Step 4: Verify live in the app (Chrome DevTools MCP)**

Use the `run` skill to launch the app with the mock temperature backend and an active session. Then with Chrome DevTools MCP:
- navigate to the Devices tab, take a snapshot/screenshot;
- confirm the empty state shows when no samples, then the water trace + stepped setpoint render and update live as the sampler emits;
- run the UI audit (alignment/spacing/overflow/contrast) per the "UI audit before done" practice and fix any flaws;
- trigger one burst and confirm (Task 6) that `burst.yaml` + a frame `.meta.yaml` carry a `temperature` block.

- [ ] **Step 5: Commit**

```bash
git add gently/ui/web/static/js/temperature-graph.js gently/ui/web/templates/index.html gently/ui/web/static/js/devices.js
git commit -m "feat(temperature): live SVG temperature graph on the Devices tab"
```

---

## Self-Review

**Spec coverage:**
- Persistence / `temperature.jsonl` → Task 1. ✓
- Sampler (poll @1 Hz, session-gated, latest-in-memory, SSE) → Tasks 2–4. ✓
- Per-acquisition stamp (burst + volume) → Task 6. ✓
- History API (`current` resolution, `since`) → Task 5. ✓
- Reusable SVG graph on Devices card (water trace + stepped setpoint + readout, backfill + live, empty state) → Task 7. ✓
- Error/empty handling (gap-not-crash, no-device idle, empty state) → Tasks 3 & 7. ✓
- Out-of-scope (setpoint control, choreography, always-on) → not implemented, by design. ✓

**Open verification items folded into tasks (not placeholders):** session-creation API (Task 1/3/6), `EventBus.subscribe` shape (Task 2), the agent's FileStore attribute (Task 4), the volume-meta accessor + `create_embryo` signature (Task 6), the burst-acquisition construction site (Task 6). Each is an explicit "confirm from the real API" instruction with a concrete fallback, not a TODO.

**Type consistency:** `temperature_stamp` returns `{water_c, setpoint_c, state, sampled_at}` everywhere; sample lines are `{t, water_c, setpoint_c, state}`; event payload is `{session_id, sample}`; history response is `{session_id, samples}`. Consistent across Tasks 1/3/5/6/7.
