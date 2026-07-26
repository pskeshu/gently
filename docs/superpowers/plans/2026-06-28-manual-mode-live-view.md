# Manual Mode — SPIM Live View (B1) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A Manual view in the Devices tab with a continuous SPIM single-slice brightfield live view (galvo/piezo/exposure controlled live), brightfield illumination, temperature, and by-hand burst/volume triggers — the hand-driven surface for next week's temperature-strain experiments.

**Architecture:** A device-layer lightsheet live streamer (MMCore continuous sequence acquisition + peek-latest, parked galvo/piezo), bridged to the browser by a `LightSheetStreamMonitor` (EventBus `LIGHTSHEET_FRAME`) over the existing base64-JPEG/SSE→WS transport; `require_control` FastAPI proxy routes; a Manual view that mirrors the bottom-camera panel. FPS is measured on the rig; a binary transport path is a conditional follow-up.

**Tech Stack:** Python asyncio + aiohttp (device layer), FastAPI (viz proxy), `pymmcore.CMMCore`, the project EventBus, vanilla-JS + canvas/SVG frontend (no build step), pytest (`asyncio_mode=auto`).

## Global Constraints

- **No new dependency.** Reuse `_encode_frame_for_stream` (OpenCV already present), the bottom-camera streamer pattern, A's portable temperature graph.
- **Single SPIM camera** today: `self.devices.get("camera")` (`HamCam1`). No side-A/B selector in B1 (deferred to B2).
- **Live = continuous sequence acquisition**, never a snap loop: `core.startContinuousSequenceAcquisition(0)` → peek `core.getLastImage()` (NEVER `popNextImage` — don't drain) → `stopSequenceAcquisition()` on exit. The core handle is `self.system.core` in the device layer; unwrap rpyc frames with `_safe_obtain`.
- **Park** before/under live: `piezo.setPosition(z)`, `scanner.sa_offset_y.setPosition(deg)` (or `scanner.set_y_offset(deg)`), `scanner.set_spim_state("Idle")`, `piezo.set_spim_state("Idle")`. galvo/piezo updates apply live (no restart); exposure change → stop→`setExposure`→restart.
- **Brightfield safety:** laser forced off in manual live (`setConfig(laser_group, "ALL OFF")` / `set_laser_power(...,0)`).
- **Concurrency:** the streamer honors `self._state_pause_counter > 0` (heavy plan owns MMCore → back off / stop sequence). Only one live stream at a time.
- **Lightsheet stream resolution/quality:** its own config — default `_ls_target_max_dim = 512`, `_ls_jpeg_quality = 70` (higher than the bottom-camera 360/55 thumbnail, for focus).
- **Transport stays JSON/`send_text`** (no binary path exists); a binary hop is Task 8, conditional on the FPS measurement.
- **Auth:** browser-facing writes are FastAPI proxy routes guarded by `Depends(require_control)`; device-layer aiohttp routes have no auth, so the browser must go through the proxy.
- **Tests:** `pytest`; `file_store`-style fixtures; FastAPI `TestClient` + mock client for routes; a fake core for the streamer. Frontend has no JS unit harness → `node --check` + Chrome-MCP harness + UI audit. Much of B1 needs the real rig; off-rig we cover streamer (fake core), routes, client, frontend harness, and defer live/FPS verification.

---

### Task 1: `LIGHTSHEET_FRAME` event type + frontend exclusion

**Files:**
- Modify: `gently/core/event_bus.py` (enum near `:88`; `_NO_HISTORY_TYPES` near `:186-195`)
- Modify: `gently/ui/web/static/js/websocket.js` (exclusion guard `:104-107`)
- Test: `tests/test_lightsheet_event.py`

**Interfaces:**
- Produces: `EventType.LIGHTSHEET_FRAME` (declared with `auto()`, matching `BOTTOM_CAMERA_FRAME`), in `_NO_HISTORY_TYPES`. Wire serialization uses `.name` (so the browser receives `"LIGHTSHEET_FRAME"`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_lightsheet_event.py
from gently.core.event_bus import EventType, EventBus, _NO_HISTORY_TYPES


def test_lightsheet_frame_event_exists():
    assert EventType.LIGHTSHEET_FRAME.name == "LIGHTSHEET_FRAME"


def test_lightsheet_frame_excluded_from_history():
    assert EventType.LIGHTSHEET_FRAME in _NO_HISTORY_TYPES


def test_lightsheet_frame_publishes_to_subscriber():
    bus = EventBus()
    seen = []
    bus.subscribe(EventType.LIGHTSHEET_FRAME, lambda e: seen.append(e.data))
    bus.publish(event_type=EventType.LIGHTSHEET_FRAME, data={"jpeg_b64": "x"}, source="t")
    assert seen == [{"jpeg_b64": "x"}]
```

> Confirm `EventBus.subscribe` signature / `_NO_HISTORY_TYPES` exportability against the real file; adapt the import/subscribe if needed, keep the assertions.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_lightsheet_event.py -v`
Expected: FAIL — `AttributeError: LIGHTSHEET_FRAME`

- [ ] **Step 3: Write minimal implementation**

In `gently/core/event_bus.py`, add the member next to `BOTTOM_CAMERA_FRAME` (`:88`):
```python
    LIGHTSHEET_FRAME = auto()  # Live JPEG frame from the SPIM lightsheet live stream
```
Add to `_NO_HISTORY_TYPES` (next to `BOTTOM_CAMERA_FRAME`):
```python
(EventType.LIGHTSHEET_FRAME,)  # high-volume live frames — keep out of history
```
In `gently/ui/web/static/js/websocket.js`, extend the exclusion guard (`:104-107`) so the frame skips the Events table but still reaches `ClientEventBus.emit`:
```javascript
        if (msg.event_type !== 'DEVICE_STATE_UPDATE' &&
            msg.event_type !== 'BOTTOM_CAMERA_FRAME' &&
            msg.event_type !== 'TEMPERATURE_UPDATE' &&
            msg.event_type !== 'LIGHTSHEET_FRAME') {
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_lightsheet_event.py -v` (3 passed) and `node --check gently/ui/web/static/js/websocket.js` (exit 0)

- [ ] **Step 5: Commit**

```bash
git add gently/core/event_bus.py gently/ui/web/static/js/websocket.js tests/test_lightsheet_event.py
git commit -m "feat(manual-mode): add LIGHTSHEET_FRAME event type + frontend exclusion"
```

---

### Task 2: Device-layer lightsheet live streamer (continuous sequence acquisition)

**Files:**
- Modify: `gently/hardware/dispim/device_layer.py` — add config attrs (near `:160-169`), `_park_lightsheet_sync`, `_grab_lightsheet_frame_sync`, `_lightsheet_streamer`, `_broadcast_lightsheet`, `handle_lightsheet_stream`, `handle_lightsheet_params`; register two routes (near `:2806`).
- Test: `tests/test_lightsheet_streamer.py`
- Reference (mirror): `_bottom_camera_streamer`/`_broadcast_camera`/`handle_bottom_camera_stream`/`_encode_frame_for_stream` (same file); sequence-acq calls in `devices/acquisition.py:186-254`.

**Interfaces:**
- Consumes: `self.system.core` (`pymmcore.CMMCore`), `self.devices["camera"]` / `["scanner"]` / `["piezo"]`, `self._state_pause_counter`, `self._encode_frame_for_stream` (reused verbatim), `_safe_obtain` (rpyc unwrap, imported as in `acquisition.py`).
- Produces: SSE `GET /api/lightsheet/stream`; `POST /api/lightsheet/live/params` `{galvo, piezo, exposure}`; in-process live param state `self._ls_params`.

> **Implementer confirmations (cannot be verified off-rig — confirm against the real code, do not guess silently):**
> 1. The device-layer core handle (`self.system.core`) and that `pymmcore.CMMCore` exposes `startContinuousSequenceAcquisition(float)`, `getLastImage()`, `stopSequenceAcquisition()`, `isSequenceRunning()` (standard CMMCore API). If `getLastImage` is unavailable, use `getLastImageMD`/`getNBeforeLastImage`.
> 2. The scanner/piezo park calls: `self.devices["scanner"].sa_offset_y.setPosition(deg)` and `self.devices["piezo"].setPosition(z)`, and `set_spim_state("Idle")` on both (from `devices/scanner.py:271`, `devices/piezo.py:226`, recon §3).
> 3. Whether brightfield live needs the scanner/beam enabled or a static park — confirm against `../micro-manager/plugins/ASIdiSPIM/src/main/java/org/micromanager/asidispim/SetupPanel.java`. Default: static park, laser "ALL OFF".

- [ ] **Step 1: Write the failing test (fake core + fake devices)**

```python
# tests/test_lightsheet_streamer.py
import asyncio, numpy as np, pytest
from gently.hardware.dispim.device_layer import DeviceLayer  # confirm class name/import


class FakeCore:
    def __init__(self):
        self.running = False
        self.exposure = None
        self.cam = None
        self._frame = np.full((64, 64), 1000, dtype=np.uint16)
        self.started = 0
        self.stopped = 0

    def setCameraDevice(self, n):
        self.cam = n

    def getCameraDevice(self):
        return self.cam

    def setExposure(self, n, ms):
        self.exposure = ms

    def startContinuousSequenceAcquisition(self, interval):
        self.running = True
        self.started += 1

    def stopSequenceAcquisition(self):
        self.running = False
        self.stopped += 1

    def isSequenceRunning(self):
        return self.running

    def getLastImage(self):
        return self._frame


class FakeAxis:
    def __init__(self):
        self.pos = None

    def setPosition(self, v):
        self.pos = v


class FakeScanner:
    def __init__(self):
        self.sa_offset_y = FakeAxis()
        self.name = "Scanner"
        self.state = None

    def set_spim_state(self, s):
        self.state = s


class FakePiezo(FakeAxis):
    def __init__(self):
        super().__init__()
        self.name = "Piezo"
        self.state = None

    def set_spim_state(self, s):
        self.state = s


def _streamer(dl):
    dl.system = type("S", (), {"core": FakeCore()})()
    dl.devices = {
        "camera": type("C", (), {"name": "HamCam1"})(),
        "scanner": FakeScanner(),
        "piezo": FakePiezo(),
    }
    return dl


async def test_grab_parks_and_peeks(monkeypatch):
    dl = _streamer(DeviceLayer.__new__(DeviceLayer))
    dl._state_pause_counter = 0
    dl._ls_target_max_dim = 512
    dl._ls_jpeg_quality = 70
    dl._ls_params = {"galvo": 1.5, "piezo": 40.0, "exposure": 20.0}
    dl._ls_seq_started = False
    dl._ls_applied = {}
    img = await asyncio.to_thread(dl._grab_lightsheet_frame_sync)
    assert img is not None and img.shape == (64, 64)
    assert dl.system.core.running is True  # sequence started
    assert dl.devices["piezo"].pos == 40.0  # piezo parked
    assert dl.devices["scanner"].sa_offset_y.pos == 1.5  # galvo parked


async def test_exposure_change_restarts_sequence():
    dl = _streamer(DeviceLayer.__new__(DeviceLayer))
    dl._state_pause_counter = 0
    dl._ls_target_max_dim = 512
    dl._ls_jpeg_quality = 70
    dl._ls_params = {"galvo": 0.0, "piezo": 50.0, "exposure": 10.0}
    dl._ls_seq_started = False
    dl._ls_applied = {}
    await asyncio.to_thread(dl._grab_lightsheet_frame_sync)
    starts = dl.system.core.started
    dl._ls_params["exposure"] = 30.0  # exposure change
    await asyncio.to_thread(dl._grab_lightsheet_frame_sync)
    assert dl.system.core.stopped >= 1 and dl.system.core.started == starts + 1
    assert dl.system.core.exposure == 30.0
```

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_lightsheet_streamer.py -v`
Expected: FAIL — `AttributeError: _grab_lightsheet_frame_sync` (or import).

- [ ] **Step 3: Implement the streamer**

Add config attrs in `__init__` (near `:169`, after the `_cam_*` block):
```python
# Lightsheet (SPIM) live stream — continuous sequence acquisition.
self._ls_subscribers: list[asyncio.Queue] = []
self._ls_task: asyncio.Task | None = None
self._ls_interval_sec: float = 0.0  # peek as fast as exposure allows
self._ls_target_max_dim: int = 512
self._ls_jpeg_quality: int = 70
self._ls_params: dict = {"galvo": 0.0, "piezo": 50.0, "exposure": 20.0}
self._ls_seq_started: bool = False
self._ls_applied: dict = {}  # last-applied galvo/piezo/exposure
```

Add the grab/park/peek (mirrors the sequence-acq calls from `acquisition.py`; uses `self.system.core`):
```python
def _park_lightsheet_sync(self) -> None:
    """Park scanner galvo + imaging piezo at the current live params (static sheet)."""
    p = self._ls_params
    scanner = self.devices.get("scanner")
    piezo = self.devices.get("piezo")
    if scanner is not None:
        try:
            scanner.set_spim_state("Idle")
        except Exception:
            pass
        scanner.sa_offset_y.setPosition(float(p["galvo"]))
    if piezo is not None:
        try:
            piezo.set_spim_state("Idle")
        except Exception:
            pass
        piezo.setPosition(float(p["piezo"]))


def _ensure_lightsheet_sequence_sync(self) -> None:
    """Start (or restart on exposure change) the continuous sequence on the SPIM camera."""
    core = self.system.core
    cam = self.devices.get("camera")
    if cam is None:
        raise RuntimeError("No lightsheet camera configured")
    p = self._ls_params
    need_restart = not self._ls_seq_started or self._ls_applied.get("exposure") != p["exposure"]
    if need_restart:
        if core.isSequenceRunning():
            core.stopSequenceAcquisition()
        if core.getCameraDevice() != cam.name:
            core.setCameraDevice(cam.name)
        core.setExposure(cam.name, float(p["exposure"]))
        core.startContinuousSequenceAcquisition(self._ls_interval_sec * 1000.0)
        self._ls_seq_started = True
        self._ls_applied["exposure"] = p["exposure"]


def _grab_lightsheet_frame_sync(self):
    """Park → ensure sequence running → peek the latest frame (never drain)."""
    try:
        self._park_lightsheet_sync()  # galvo/piezo applied live
        self._ensure_lightsheet_sequence_sync()  # start / restart on exposure
        from gently.hardware.dispim.devices.acquisition import _safe_obtain

        core = self.system.core
        img = core.getLastImage()
        try:
            img = _safe_obtain(img)
        except (ImportError, AttributeError):
            pass
        return np.asarray(img)
    except Exception as exc:
        logger.debug("Lightsheet grab failed: %s", exc)
        return None


def _stop_lightsheet_sequence_sync(self) -> None:
    try:
        if self.system.core.isSequenceRunning():
            self.system.core.stopSequenceAcquisition()
    except Exception:
        logger.debug("stop lightsheet sequence failed", exc_info=True)
    self._ls_seq_started = False
    self._ls_applied = {}
```

Add the streamer loop + broadcast (mirror `_bottom_camera_streamer`/`_broadcast_camera`, reusing `_encode_frame_for_stream`):
```python
async def _lightsheet_streamer(self):
    logger.info("Lightsheet streamer started")
    try:
        while self._ls_subscribers:
            if self._state_pause_counter > 0:
                # Heavy plan owns MMCore: release the sequence and back off.
                if self._ls_seq_started:
                    await asyncio.to_thread(self._stop_lightsheet_sequence_sync)
                await asyncio.sleep(0.1)
                continue
            tick = time.monotonic()
            img = await asyncio.to_thread(self._grab_lightsheet_frame_sync)
            payload = self._encode_frame_for_stream(img) if img is not None else None
            if payload is not None:
                await self._broadcast_lightsheet(payload)
            elapsed = time.monotonic() - tick
            # Pace to at least the exposure; peek-rate caps near the camera rate.
            floor = max(self._ls_interval_sec, self._ls_params["exposure"] / 1000.0)
            await asyncio.sleep(max(0.0, floor - elapsed))
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception("Lightsheet streamer crashed")
    finally:
        await asyncio.to_thread(self._stop_lightsheet_sequence_sync)
        logger.info("Lightsheet streamer exiting")


async def _broadcast_lightsheet(self, payload):
    if not self._ls_subscribers:
        return
    dead = []
    for q in self._ls_subscribers:
        try:
            q.put_nowait(payload)
        except asyncio.QueueFull:
            try:
                _ = q.get_nowait()
                q.put_nowait(payload)
            except Exception:
                dead.append(q)
    for q in dead:
        try:
            self._ls_subscribers.remove(q)
        except ValueError:
            pass
```

> `_encode_frame_for_stream` uses `self._cam_target_max_dim`/`self._cam_jpeg_quality`. To get the 512/70 lightsheet settings without duplicating the encoder, add an optional override: change its signature to `_encode_frame_for_stream(self, img, max_dim=None, quality=None)` defaulting to the `_cam_*` values, and call it `self._encode_frame_for_stream(img, self._ls_target_max_dim, self._ls_jpeg_quality)` from the lightsheet loop. (One-line change to the encoder; bottom-camera behavior unchanged.)

Add the SSE handler + params handler (mirror `handle_bottom_camera_stream`; the params handler updates `self._ls_params` — galvo/piezo apply on the next grab, exposure triggers the restart path):
```python
async def handle_lightsheet_stream(self, request):
    response = web.StreamResponse(
        status=200,
        reason="OK",
        headers={
            "Content-Type": "text/event-stream",
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
    await response.prepare(request)
    queue: asyncio.Queue = asyncio.Queue(maxsize=4)
    self._ls_subscribers.append(queue)
    if len(self._ls_subscribers) == 1 and (self._ls_task is None or self._ls_task.done()):
        self._ls_task = asyncio.create_task(self._lightsheet_streamer(), name="lightsheet-streamer")
    try:
        await response.write(b": connected\n\n")
        while True:
            try:
                payload = await asyncio.wait_for(queue.get(), timeout=10.0)
            except asyncio.TimeoutError:
                await response.write(b": keepalive\n\n")
                continue
            if payload is None:
                break
            await response.write(f"data: {json.dumps(payload)}\n\n".encode())
    except (asyncio.CancelledError, ConnectionResetError, ConnectionAbortedError):
        pass
    except Exception:
        logger.exception("Lightsheet SSE writer failed")
    finally:
        try:
            self._ls_subscribers.remove(queue)
        except ValueError:
            pass
    return response


async def handle_lightsheet_params(self, request):
    body = await request.json()
    for k in ("galvo", "piezo", "exposure"):
        if k in body and body[k] is not None:
            self._ls_params[k] = float(body[k])
    return web.json_response({"params": self._ls_params})
```

Register both routes near `:2806`:
```python
        self._app.router.add_get("/api/lightsheet/stream", self.handle_lightsheet_stream)
        self._app.router.add_post("/api/lightsheet/live/params", self.handle_lightsheet_params)
```

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/test_lightsheet_streamer.py -v`
Expected: PASS (2 passed). If `DeviceLayer.__new__` bypassing `__init__` leaves attrs unset, the test sets the needed ones explicitly (it does).

- [ ] **Step 5: Commit**

```bash
git add gently/hardware/dispim/device_layer.py tests/test_lightsheet_streamer.py
git commit -m "feat(manual-mode): device-layer lightsheet live streamer (continuous sequence acquisition)"
```

---

### Task 3: Client methods — `stream_lightsheet` + `set_lightsheet_live_params`

**Files:**
- Modify: `gently/hardware/dispim/client.py` (add two methods on `DiSPIMMicroscope`, near `stream_bottom_camera` `:913`)
- Test: `tests/test_lightsheet_client.py`

**Interfaces:**
- Produces: `async def stream_lightsheet(self, timeout=None)` (async generator over `GET /api/lightsheet/stream`, identical SSE parse to `stream_bottom_camera`); `async def set_lightsheet_live_params(self, galvo=None, piezo=None, exposure=None) -> dict` (`POST /api/lightsheet/live/params`).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_lightsheet_client.py
import pytest
from gently.hardware.dispim.client import DiSPIMMicroscope


async def test_set_params_posts_body(monkeypatch):
    m = DiSPIMMicroscope.__new__(DiSPIMMicroscope)
    sent = {}

    async def fake_post(path, body):
        sent["path"] = path
        sent["body"] = body
        return {"params": body}

    m._api_post = fake_post  # confirm the real low-level POST helper name
    res = await m.set_lightsheet_live_params(galvo=1.0, piezo=42.0, exposure=15.0)
    assert sent["path"] == "/api/lightsheet/live/params"
    assert sent["body"] == {"galvo": 1.0, "piezo": 42.0, "exposure": 15.0}
    assert res == {"params": {"galvo": 1.0, "piezo": 42.0, "exposure": 15.0}}


def test_stream_lightsheet_is_async_generator():
    m = DiSPIMMicroscope.__new__(DiSPIMMicroscope)
    import inspect

    assert inspect.isasyncgenfunction(m.stream_lightsheet)
```

> Confirm the real low-level POST helper (the recon shows `set_led` etc. POST via an internal helper — find whether it's `self._api_post(path, body)` or an inline `self._session.post`). Match it; if `set_lightsheet_live_params` should drop `None` keys, build the body from only the provided args (the test passes all three).

- [ ] **Step 2: Run to verify it fails**

Run: `pytest tests/test_lightsheet_client.py -v` → FAIL (no such methods).

- [ ] **Step 3: Implement**

```python
async def stream_lightsheet(self, timeout: float | None = None):
    """Async generator yielding JPEG frames from the lightsheet live SSE stream.

    Mirrors :meth:`stream_bottom_camera`; subscriber-gated on the device layer.
    """
    self._ensure_connected()
    client_timeout = aiohttp.ClientTimeout(total=None, sock_read=timeout, sock_connect=10.0)
    url = f"{self.http_url}/api/lightsheet/stream"
    async with self._session.get(url, timeout=client_timeout) as resp:
        resp.raise_for_status()
        buf = b""
        async for chunk in resp.content.iter_any():
            if not chunk:
                continue
            buf += chunk
            while b"\n\n" in buf:
                event_block, buf = buf.split(b"\n\n", 1)
                data_lines = []
                for line in event_block.splitlines():
                    if not line or line.startswith(b":"):
                        continue
                    if line.startswith(b"data:"):
                        data_lines.append(line[5:].lstrip())
                if not data_lines:
                    continue
                raw = b"\n".join(data_lines).decode("utf-8", errors="replace")
                try:
                    import json as _json

                    yield _json.loads(raw)
                except Exception as exc:
                    logger.warning("Malformed lightsheet SSE payload skipped: %s", exc)


async def set_lightsheet_live_params(self, galvo=None, piezo=None, exposure=None) -> dict:
    """POST live galvo/piezo/exposure to the device-layer lightsheet streamer."""
    body = {}
    if galvo is not None:
        body["galvo"] = float(galvo)
    if piezo is not None:
        body["piezo"] = float(piezo)
    if exposure is not None:
        body["exposure"] = float(exposure)
    return await self._api_post("/api/lightsheet/live/params", body)
```

> If the real POST helper isn't `_api_post`, adapt this one call site (and the test) to the real helper. `stream_lightsheet` copies `stream_bottom_camera` verbatim except the URL.

- [ ] **Step 4: Run to verify it passes**

Run: `pytest tests/test_lightsheet_client.py -v` (2 passed)

- [ ] **Step 5: Commit**

```bash
git add gently/hardware/dispim/client.py tests/test_lightsheet_client.py
git commit -m "feat(manual-mode): client stream_lightsheet + set_lightsheet_live_params"
```

---

### Task 4: `LightSheetStreamMonitor` (agent Service) + agent wiring

**Files:**
- Create: `gently/app/lightsheet_monitor.py`
- Modify: `gently/app/agent.py` (init attr; construct in `start_viz_server` near the bottom-camera monitor `:849-860`; stop in `stop_viz_server` near `:862-869`)
- Test: `tests/test_lightsheet_monitor.py`
- Reference (mirror verbatim, swapping names/event): `gently/app/bottom_camera_monitor.py`

**Interfaces:**
- Consumes: `microscope.stream_lightsheet()` (Task 3); `EventType.LIGHTSHEET_FRAME` (Task 1).
- Produces: `LightSheetStreamMonitor(Service)` with `running` property, `on_start`/`on_stop`; publishes `LIGHTSHEET_FRAME`. `agent.lightsheet_monitor` (constructed, not started; started via proxy in Task 5).

- [ ] **Step 1: Write the failing test**

```python
# tests/test_lightsheet_monitor.py
import asyncio
from gently.core.event_bus import EventType, get_event_bus
from gently.app.lightsheet_monitor import LightSheetStreamMonitor


class FakeScope:
    async def stream_lightsheet(self):
        for i in range(3):
            yield {"t": float(i), "jpeg_b64": f"f{i}"}
            await asyncio.sleep(0)


async def test_monitor_publishes_frames():
    bus = get_event_bus()
    seen = []
    bus.subscribe(EventType.LIGHTSHEET_FRAME, lambda e: seen.append(e.data))
    mon = LightSheetStreamMonitor(FakeScope(), reconnect_delay_sec=0.01)
    await mon.start()
    await asyncio.sleep(0.05)
    await mon.stop()
    assert any(d.get("jpeg_b64") == "f0" for d in seen)
    assert mon.running is False
```

- [ ] **Step 2: Run to verify it fails** — `pytest tests/test_lightsheet_monitor.py -v` → FAIL (no module).

- [ ] **Step 3: Implement** — copy `gently/app/bottom_camera_monitor.py` to `gently/app/lightsheet_monitor.py` and change: class → `LightSheetStreamMonitor`; `name="lightsheet-monitor"`; the `_run` loop calls `self.microscope.stream_lightsheet()` and publishes `EventType.LIGHTSHEET_FRAME` with `source="lightsheet-monitor"`. (Everything else — Service base, on_start/on_stop, reconnect loop, `_last_frame_ts`, `running` — is identical.)

Agent wiring in `gently/app/agent.py`: add `self.lightsheet_monitor = None` next to `self.bottom_camera_monitor = None`; in `start_viz_server` (after the bottom-camera monitor construction `:849-860`):
```python
if self.microscope is not None and self.lightsheet_monitor is None:
    try:
        from .lightsheet_monitor import LightSheetStreamMonitor

        self.lightsheet_monitor = LightSheetStreamMonitor(self.microscope)
        logger.info("Lightsheet monitor ready (not started)")
    except Exception as e:
        logger.warning(f"Failed to construct lightsheet monitor: {e}")
        self.lightsheet_monitor = None
```
In `stop_viz_server` (near `:862`):
```python
        if self.lightsheet_monitor is not None:
            try:
                await self.lightsheet_monitor.stop()
            except Exception:
                logger.exception("Failed to stop lightsheet monitor")
            self.lightsheet_monitor = None
```

- [ ] **Step 4: Run to verify it passes** — `pytest tests/test_lightsheet_monitor.py -v` (1 passed); `pytest -q` (no new failures).

- [ ] **Step 5: Commit**

```bash
git add gently/app/lightsheet_monitor.py gently/app/agent.py tests/test_lightsheet_monitor.py
git commit -m "feat(manual-mode): LightSheetStreamMonitor bridge + agent wiring"
```

---

### Task 5: Browser proxy routes (`require_control`)

**Files:**
- Modify: `gently/ui/web/routes/data.py` (add routes in `create_router`, mirroring the bottom-camera + room-light routes)
- Test: `tests/test_lightsheet_routes.py`
- Reference: bottom-camera start/stop/status (`data.py:260-311`), room-light proxy (`data.py:335-355`), `_resolve_client` (`:313`), `require_control` (`:10`).

**Interfaces:**
- Consumes: `agent.lightsheet_monitor` (Task 4); `client.set_lightsheet_live_params`, `set_led`, `set_laser_power`, `set_camera_led_mode`, `move_to_position`, `acquire_burst`, `acquire_volume`.
- Produces (all `Depends(require_control)` except GET status): `POST /api/devices/lightsheet/live/{start,stop}`, `GET /api/devices/lightsheet/live/status`, `POST /api/devices/lightsheet/live/params`, `POST /api/devices/led/set`, `POST /api/devices/laser/off`, `POST /api/devices/camera/led_mode`, `POST /api/devices/stage/move`, `POST /api/devices/acquire/{burst,volume}`.

- [ ] **Step 1: Write the failing test (TestClient + mock client/monitor)**

```python
# tests/test_lightsheet_routes.py
from unittest.mock import MagicMock, AsyncMock
from fastapi import FastAPI
from fastapi.testclient import TestClient
from gently.ui.web.routes.data import create_router
import gently.ui.web.auth as auth


def _app(client=None, monitor=None):
    server = MagicMock()
    server.agent_bridge.agent.client = client
    server.agent_bridge.agent.lightsheet_monitor = monitor
    app = FastAPI()
    app.include_router(create_router(server))
    # legacy localhost = CONTROL; TestClient client.host is "testclient" → force CONTROL:
    app.dependency_overrides[auth.require_control] = lambda: True
    return TestClient(app)


def test_live_params_forwards():
    client = MagicMock()
    client.set_lightsheet_live_params = AsyncMock(return_value={"params": {}})
    r = _app(client=client).post(
        "/api/devices/lightsheet/live/params", json={"galvo": 1.0, "piezo": 40.0, "exposure": 20.0}
    )
    assert r.status_code == 200
    client.set_lightsheet_live_params.assert_awaited_once_with(galvo=1.0, piezo=40.0, exposure=20.0)


def test_acquire_burst_forwards():
    client = MagicMock()
    client.acquire_burst = AsyncMock(return_value={"success": True, "request_id": "b1"})
    r = _app(client=client).post(
        "/api/devices/acquire/burst",
        json={"frames": 60, "mode": "1hz", "num_slices": 1, "exposure_ms": 5.0},
    )
    assert r.status_code == 200 and r.json().get("request_id") == "b1"


def test_live_start_requires_monitor():
    r = _app(monitor=None).post("/api/devices/lightsheet/live/start")
    assert r.status_code == 503
```

> Confirm the `require_control` override mechanism: in legacy mode `TestClient` requests are not localhost, so override the dependency (as above) to isolate route logic. A separate test can assert the gate by NOT overriding and expecting 403.

- [ ] **Step 2: Run to verify it fails** — `pytest tests/test_lightsheet_routes.py -v` → FAIL (routes 404).

- [ ] **Step 3: Implement** — in `create_router`, add (mirroring the referenced routes). Live start/stop/status copy the bottom-camera versions verbatim, swapping `bottom_camera_monitor` → `lightsheet_monitor`. Then:
```python
@router.post("/api/devices/lightsheet/live/params", dependencies=[Depends(require_control)])
async def lightsheet_live_params(payload: dict = Body(...)):  # noqa: B008
    client = _resolve_client()
    if client is None:
        raise HTTPException(status_code=503, detail="Microscope not connected")
    try:
        res = await client.set_lightsheet_live_params(
            galvo=payload.get("galvo"), piezo=payload.get("piezo"), exposure=payload.get("exposure")
        )
    except Exception as exc:
        logger.exception("lightsheet live params failed")
        raise HTTPException(status_code=502, detail=f"params failed: {exc}") from exc
    return res


@router.post("/api/devices/led/set", dependencies=[Depends(require_control)])
async def led_set(payload: dict = Body(...)):  # noqa: B008
    client = _resolve_client()
    if client is None:
        raise HTTPException(status_code=503, detail="Microscope not connected")
    try:
        return await client.set_led(str(payload.get("state", "Closed")))
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"led failed: {exc}") from exc


@router.post("/api/devices/laser/off", dependencies=[Depends(require_control)])
async def laser_off():
    client = _resolve_client()
    if client is None:
        raise HTTPException(status_code=503, detail="Microscope not connected")
    try:
        return await client.set_laser_power(488, 0)  # confirm signature; force off
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"laser off failed: {exc}") from exc


@router.post("/api/devices/camera/led_mode", dependencies=[Depends(require_control)])
async def camera_led_mode(payload: dict = Body(...)):  # noqa: B008
    client = _resolve_client()
    if client is None:
        raise HTTPException(status_code=503, detail="Microscope not connected")
    try:
        return await client.set_camera_led_mode(bool(payload.get("use_led", False)))
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"camera led mode failed: {exc}") from exc


@router.post("/api/devices/stage/move", dependencies=[Depends(require_control)])
async def stage_move(payload: dict = Body(...)):  # noqa: B008
    client = _resolve_client()
    if client is None:
        raise HTTPException(status_code=503, detail="Microscope not connected")
    try:
        return await client.move_to_position(float(payload["x"]), float(payload["y"]))
    except KeyError:
        raise HTTPException(status_code=400, detail="x and y required")
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"stage move failed: {exc}") from exc


@router.post("/api/devices/acquire/burst", dependencies=[Depends(require_control)])
async def acquire_burst(payload: dict = Body(...)):  # noqa: B008
    client = _resolve_client()
    if client is None:
        raise HTTPException(status_code=503, detail="Microscope not connected")
    try:
        return await client.acquire_burst(
            frames=int(payload.get("frames", 60)),
            mode=str(payload.get("mode", "1hz")),
            num_slices=int(payload.get("num_slices", 1)),
            exposure_ms=float(payload.get("exposure_ms", 5.0)),
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"burst failed: {exc}") from exc


@router.post("/api/devices/acquire/volume", dependencies=[Depends(require_control)])
async def acquire_volume(payload: dict = Body(...)):  # noqa: B008
    client = _resolve_client()
    if client is None:
        raise HTTPException(status_code=503, detail="Microscope not connected")
    try:
        return await client.acquire_volume(
            num_slices=int(payload.get("num_slices", 50)),
            exposure_ms=float(payload.get("exposure_ms", 10.0)),
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"volume failed: {exc}") from exc
```
Add the live start/stop/status routes by copying `start_bottom_camera_stream`/`stop_bottom_camera_stream`/`get_bottom_camera_status` (`data.py:260-311`) under `/api/devices/lightsheet/live/...` with `getattr(agent, "lightsheet_monitor", None)`.

- [ ] **Step 4: Run to verify it passes** — `pytest tests/test_lightsheet_routes.py -v` (3 passed); `pytest -q` (no new failures).

- [ ] **Step 5: Commit**

```bash
git add gently/ui/web/routes/data.py tests/test_lightsheet_routes.py
git commit -m "feat(manual-mode): require_control proxy routes for lightsheet live + illumination + acquire"
```

---

### Task 6: Manual view (frontend)

**Files:**
- Modify: `gently/ui/web/templates/index.html` — add a `data-view="manual"` button to `#devices-view-switcher` (`:445-449`); add a `#devices-view-manual` container with the live canvas + control rail (model the camera panel `:569-608`); load A's `temperature-graph.js` if not already loaded on this page.
- Modify: `gently/ui/web/static/js/devices.js` — add `'manual'` to `VIEWS` (`:17`); a `handleLightsheetFrame` (mirror `handleCameraFrame` `:1111`); live toggle (mirror `toggleCameraStream`); galvo/piezo/exposure controls (debounced POST `live/params`); illumination toggles; acquire buttons; FPS readout; embed `TemperatureGraph.init`; a `'v'`-style key handled only if not deferred — leave existing keys, add nothing conflicting.
- Modify: the devices stylesheet for the manual panel (reuse `.devices-camera-*` styles where possible).

**No JS unit harness** — verified by `node --check` + a Chrome-MCP harness (like A) + a UI audit.

- [ ] **Step 1: Markup** — add to `#devices-view-switcher`:
```html
                        <button class="view-btn" data-view="manual" title="Manual control">Manual</button>
```
Add the view container after `#devices-view-optical3d` (`:716`-ish), with: a live `<img id="devices-ls-img">` (or `<canvas>`) + placeholder + FPS/side overlay + Start/Stop toggle (`#devices-ls-toggle`); a right rail with exposure input, galvo slider (`#devices-ls-galvo`), piezo slider (`#devices-ls-piezo`), illumination toggles (LED `#devices-ls-led`, camera-LED-mode, room light, a static "Laser: OFF" indicator), a temperature setpoint + `<div id="devices-ls-tempgraph">`, and Snap-volume / Burst buttons + a `#devices-ls-lastcap` card. Mirror the `.devices-camera-*` class structure for the image stage so the existing zoom/pan inline machinery can be reused or replicated.

- [ ] **Step 2: JS — frame paint + controls**

Add `'manual'` to `VIEWS`. Add a frame handler mirroring `handleCameraFrame` (separate DOM ids `_lsImg`/`_lsMeta`, its own FPS window) and subscribe `ClientEventBus.on('LIGHTSHEET_FRAME', handleLightsheetFrame)` in `setupCameraWiring` (or a new `setupManualWiring`). Live toggle hits `/api/devices/lightsheet/live/start|stop`. Galvo/piezo/exposure inputs: on `input`, **debounce ~120 ms**, then `fetch('/api/devices/lightsheet/live/params', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({galvo, piezo, exposure})})`. Illumination toggles POST their routes. Acquire buttons POST `/api/devices/acquire/burst|volume` with the current params, show "acquiring…" then render the result ref in `#devices-ls-lastcap`. Embed A's graph: `if (window.TemperatureGraph) TemperatureGraph.init(document.getElementById('devices-ls-tempgraph'), 'current')`. FPS readout from the frame handler's window (reuse the `computeCameraFps` approach).

- [ ] **Step 3: `node --check`** — `node --check gently/ui/web/static/js/devices.js` (exit 0).

- [ ] **Step 4: Chrome-MCP harness verification** — build a standalone harness (like A's): copy `event-bus.js`, `temperature-graph.js`, the new manual JS, and `main.css`; stub `fetch` for `live/params` + status + acquire; feed simulated `LIGHTSHEET_FRAME` events (a moving synthetic gradient that shifts when galvo/piezo "params" change) to demonstrate the slide-and-see; screenshot; run the alignment/spacing/contrast UI audit and fix flaws. Live in-app + FPS verification is deferred to the rig.

- [ ] **Step 5: Commit**

```bash
git add gently/ui/web/templates/index.html gently/ui/web/static/js/devices.js gently/ui/web/static/css/main.css
git commit -m "feat(manual-mode): Manual view — SPIM live canvas, galvo/piezo/exposure, illumination, acquire, temp"
```

---

### Task 7: FPS measurement (rig) + conditional binary transport

**Files:**
- Create: `docs/superpowers/notes/2026-06-28-lightsheet-fps-measurement.md` (record the numbers)
- (Conditional) Modify: `gently/ui/web/connection_manager.py` + `agent_ws`/`server.py` + `websocket.js` for a binary frame path.

**This task is a measurement + a gate, not unconditional code.**

- [ ] **Step 1:** On the rig, start lightsheet live and record from the Manual-view FPS readout + device logs: **device grab rate**, **delivered rate**, **browser paint rate**, at 512 px/q70 and at a reduced 384 px/q60. Write them into the notes file with the exposure used.
- [ ] **Step 2: Diagnose.** If device grab < ~15 fps → limiter is exposure/readout/rpyc, not transport — tune exposure/size/quality, stop here. If device grab ≥ target but browser paint < target → transport is the bottleneck → do Step 3.
- [ ] **Step 3 (conditional): binary WebSocket path.** Add a `send_bytes`-based frame channel: device→agent SSE stays; on the agent→browser hop, push the raw JPEG via `websocket.send_bytes(prefix + jpeg)` (small type byte), bypassing base64 + the per-client `json.dumps` + the EventBus fan-out; browser `onmessage` binary → `createImageBitmap(new Blob([buf]))` → `ctx.drawImage`. Re-measure and record.
- [ ] **Step 4: Commit** the notes (and any binary-path code, if built).

```bash
git add docs/superpowers/notes/2026-06-28-lightsheet-fps-measurement.md
git commit -m "docs(manual-mode): lightsheet live FPS measurement + transport decision"
```

---

## Self-Review

**Spec coverage:** §2.1 streamer → Task 2; §2.2 monitor → Task 4; §2.3 proxy routes → Task 5; §2.4 client methods → Task 3; §2.5 Manual view → Task 6; §2.6 concurrency (`_state_pause_counter` back-off) → Task 2 streamer loop; §2.7 brightfield laser-off → Tasks 2 & 5 (`laser/off`); §3 transport baseline + measurement + binary escalation → Tasks 1/4 (baseline) + Task 7 (measure/escalate); §4 error handling → Tasks 2/5 (try/except, 503/502); §5 testing → each task's tests + Task 6 harness; LIGHTSHEET_FRAME event → Task 1. Single-camera (no side A/B) reflected throughout. ✓

**Open confirmations (explicit, not placeholders):** pymmcore sequence-acq method availability + core handle (Task 2); scanner/piezo park method names (Task 2); the SetupPanel scanner/beam question (Task 2); the client low-level POST helper name (Task 3); the `require_control` test-override mechanism (Task 5). Each names a concrete fallback.

**Type consistency:** frame payload `{t, shape, downsample, mime, jpeg_b64}` (reused encoder) across Tasks 2/3/4/6; live params `{galvo, piezo, exposure}` across Tasks 2/3/5/6; event `LIGHTSHEET_FRAME` across Tasks 1/4/6; `lightsheet_monitor` attr across Tasks 4/5. Consistent.
