# Design: Manual Mode — SPIM single-slice brightfield live view (sub-project B1)

Status: design approved in brainstorm (2026-06-28).
Base branch: off `feature/temperature-interface` (sub-project A) — B1 embeds A's now-portable
temperature graph and its acquisition temperature stamp.

## 0. Where this sits (execution roadmap)

Sub-project **B** of the temperature-strain experiment prep. B is split:

- **B1 (this spec) — the hand-driven experiment surface:** a **Manual view** in the Devices
  tab with a SPIM single-slice brightfield live view (one selectable side A/B), live
  galvo/piezo/exposure controls, brightfield illumination (LED, no laser), temperature
  setpoint + A's graph, and by-hand burst/volume triggers.
- **B2 (deferred):** laser channel-preset browser, full timelapse/volume config form,
  side-by-side dual view, richer camera selection.

This is **load-bearing for next week** — the experiments are imaged by hand first. The first
experiment round is hand-driven through this view.

## 1. Overview

Today the Devices tab is read-only and there is no human-driven imaging surface. The
temperature experiments image with the **SPIM cameras under LED/brightfield illumination (no
laser, DIC/BF-like)** — a *single plane*, not a scan — and the operator needs to slide the
**galvo (sheet position)** and **imaging piezo (focal plane)** and *see the effect live*, then
trigger bursts/volumes by hand around a temperature change.

### What already exists (reused)
- The MMCore core is a real `pymmcore.CMMCore`; **continuous sequence acquisition** is already
  proven in the volume path (`acquisition.py:199-231`: `startSequenceAcquisition`/`popNextImage`/
  `getRemainingImageCount`/`stopSequenceAcquisition`).
- `capture_lightsheet_image(piezo_position, galvo_position, exposure_ms)` (`client.py:493`) —
  a single-plane snap primitive (but it goes through the Bluesky queue → too heavy for a live
  loop; used only for one-shot snaps).
- The **bottom-camera live streamer** (`device_layer.py` `_bottom_camera_streamer`) — the
  structural template (subscriber-gated SSE, drop-oldest backpressure, `_state_pause_counter`
  heavy-plan back-off, JPEG/base64 encode, EventBus→WS transport).
- Direct device writes: `set_led`, `set_laser_power`, `set_camera_led_mode`, `set_room_light`,
  `set_temperature`, `move_to_position`, `acquire_volume`, `acquire_burst` (client methods).
- `require_control` auth (`auth.py`) — localhost = CONTROL on the rig; remote needs a token.
- The Devices view-switcher pattern (Map/Details/3D) to host a new "Manual" view.
- A's portable temperature-graph component + per-acquisition temperature stamp.

### What must be built
A device-layer **lightsheet live streamer** (continuous sequence + peek-latest), an agent-side
**LightsheetStreamMonitor**, **`require_control` browser proxy routes**, two new client methods,
and the **Manual view** UI. Transport reuses the JSON/base64 path first; a binary path is a
**conditional escalation gated on an FPS measurement**.

## 2. Architecture — seven units

### 2.1 Lightsheet live streamer (device layer, new)
`_lightsheet_live_streamer` task + `handle_lightsheet_stream` SSE handler in `device_layer.py`,
mirroring the bottom-camera streamer's skeleton (subscriber-gated lifecycle, per-subscriber
`asyncio.Queue` drop-oldest backpressure, `_state_pause_counter` back-off).

**Grab mechanism — the MM live pattern (replacing the snap loop):**
- On first subscriber: select the SPIM side camera (`core.setCameraDevice`), set exposure,
  **park** galvo + piezo at the current params, `core.startContinuousSequenceAcquisition(0)`.
- Loop: **peek the latest** frame (`getLastImage` / `getNBeforeLastTaggedImage`) — never
  `popNextImage` (don't drain). Encode → broadcast. Pace to `max(exposure, ~1/30 s)` up to the
  display cap (MM's clamp).
- On last subscriber: `stopSequenceAcquisition`, restore prior camera/state.

**Live param updates** (`POST /api/lightsheet/live/params {side, galvo, piezo, exposure}`):
- **galvo / piezo → applied live, no restart** (move scanner/imaging-piezo; next frames show
  the new plane — the slide-and-see response).
- **exposure / side → stop → reconfigure → restart** (camera params can't change mid-sequence).

**Resolution/quality:** the lightsheet stream gets its **own configurable size/quality**
(default ~512 px, JPEG q≈70) — higher than the bottom-camera 360 px thumbnail, since this view
is for judging focus. Resolution↔FPS tradeoff feeds the §3 measurement.

**rpyc caveat:** the core may be an rpyc proxy, so each `getLastImage` transfers a full frame
across that boundary *inside the device-layer process*; the streamer encodes to JPEG there, so
only a small payload leaves the process. rpyc bounds the *grab* rate, not the web payload.

**Implementer confirmations (flagged, not guesses):** the exact device methods to park the
galvo (scanner) + imaging piezo and the SPIM side-camera device names
(`device_factory.py`/`devices/scanner.py`/`optical.py`); and whether brightfield/LED live needs
the scanner/beam enabled at all or a fully static park — confirm against
`../micro-manager/plugins/ASIdiSPIM/.../SetupPanel.java`. B1 assumes static park, laser off.

### 2.2 Lightsheet stream monitor (agent Service, new)
Mirrors `BottomCameraStreamMonitor`: consumes the device SSE via a new client generator and
republishes frames on the EventBus as a new high-volume `EventType.LIGHTSHEET_FRAME`
(added to `_NO_HISTORY_TYPES` and the `websocket.js` events-table exclusion). Opt-in;
started/stopped via the proxy routes.

### 2.3 Browser proxy routes (viz server, `require_control`, new)
In `data.py`'s `create_router`, each `Depends(require_control)`, each resolving the live client
via `_resolve_client()`:

| Route | Calls |
|---|---|
| `POST /api/devices/lightsheet/live/start` | start `LightsheetStreamMonitor` |
| `POST /api/devices/lightsheet/live/stop` | stop the monitor |
| `GET /api/devices/lightsheet/live/status` | `{available, streaming, side, last_frame_ts, fps}` |
| `POST /api/devices/lightsheet/live/params` | `client.set_lightsheet_live_params(side,galvo,piezo,exposure)` |
| `POST /api/devices/led/set` | `client.set_led(state)` |
| `POST /api/devices/laser/off` | `client.set_laser_power(wl, 0)` / "ALL OFF" |
| `POST /api/devices/camera/led_mode` | `client.set_camera_led_mode(use_led)` |
| `POST /api/devices/room_light/set` | *(exists `data.py:335`)* |
| `POST /api/devices/stage/move` | `client.move_to_position(x,y)` |
| `POST /api/devices/acquire/burst` | `client.acquire_burst(...)` |
| `POST /api/devices/acquire/volume` | `client.acquire_volume(...)` |
| `POST /api/devices/temperature/set` | *(exists `data.py:381`)* |

The device-layer aiohttp routes have no auth; the browser must go through this proxy.

### 2.4 New client methods
`DiSPIMMicroscope.stream_lightsheet(...)` (async generator over `GET /api/lightsheet/stream`,
mirroring `stream_bottom_camera`) and `set_lightsheet_live_params(side, galvo, piezo, exposure)`
(`POST /api/lightsheet/live/params`). Everything else already exists.

### 2.5 Manual view (frontend, new `#devices-view-manual`)
A "Manual" entry in the Devices view-switcher → two-column layout:
- **Left:** the live image canvas (reusing the bottom-camera render + `zoom-pan.js` +
  crosshair), an FPS + side overlay, and a Start/Stop Live toggle; a "Last capture" card below.
- **Right control rail:** Camera (side A/B selector, exposure) · Focus & sheet (galvo + piezo
  sliders with numeric fields + nudge) · Illumination (LED, camera-LED-mode, room light, laser
  OFF indicator) · Temperature (setpoint + A's embedded graph) · Acquire (Snap volume / Burst
  buttons with a small param summary).

Interactions: galvo/piezo sliders → **debounced** `POST live/params` → canvas updates in a
couple of frames; acquire → "acquiring…" overlay → Last-capture card → live resumes; FPS
readout surfaces the §3 measurement. Toggle/setpoint widgets model on the existing room-light
and temp-set controls.

### 2.6 Concurrency rule
Live holds the camera/MMCore, so a triggered burst/volume **stops the live sequence → acquires
→ resumes** (via the existing `_state_pause_counter` back-off). Only **one live stream at a
time** (lightsheet vs bottom contend for MMCore). Acquisitions land with A's temperature stamp.

### 2.7 Brightfield safety
On entering manual/brightfield live, **laser is forced off**; `DiSPIMLightSource` power-limit
clamps already cap any laser power.

## 3. Transport & the FPS measure→escalate path (Approach ①)

**Baseline (build first):** lightsheet frames ride the existing path unchanged —
device base64-JPEG-in-JSON over SSE → `LightsheetStreamMonitor` → EventBus `LIGHTSHEET_FRAME`
→ `ConnectionManager.broadcast` (`json.dumps` + `send_text`) → browser canvas. Additions: the
new EventType (+ `_NO_HISTORY_TYPES` + `websocket.js` exclusion) and a browser paint handler.

**Instrumented measurement (the gate):** three counters — **device grab rate**, **delivered
rate**, **browser paint rate** (the FPS readout). Target **≥ ~15 fps usable** (stretch 25–30),
latency < ~150 ms. Run on the rig; record the numbers in the plan.

**Diagnosis fork:**
- device grab < target → limiter is exposure / readout / rpyc, *not* transport → tune
  exposure / size / quality; binary won't help.
- device grab ≥ target but browser paint < target → transport is the bottleneck → escalate.

**Escalation (conditional task):** a **binary WebSocket path on the agent→browser hop** —
`websocket.send_bytes(jpeg)` with a small type prefix, browser `createImageBitmap` → canvas —
bypassing base64 (+33%), the per-client `json.dumps`, and the EventBus fan-out. The device→agent
SSE stays (single consumer). Built **only if** the measurement points to transport.

## 4. Error handling & edges
- No SPIM camera / device down → `live/status {available:false}` → empty canvas, controls
  disabled. Sequence fails to start → surfaced, graceful (no crash).
- rpyc/slow grab → backpressure drops oldest; FPS readout shows reality.
- Param out of range → `DiSPIMLightSource` clamps + UI input bounds; laser forced off in
  brightfield.
- Only one live stream at a time; live is subscriber-gated → stops on view-switch / page-close.
- `require_control` denied (remote, no token) → 403 → controls shown disabled.

## 5. Testing
- **Device layer:** streamer against a fake core supporting sequence acquisition
  (start/stop/`getLastImage`) — param updates (galvo/piezo live vs exposure/side restart),
  heavy-plan pause/resume, subscriber lifecycle.
- **Proxy routes:** FastAPI TestClient + mock client — `require_control` 403 gate, param
  forwarding.
- **Client methods:** `stream_lightsheet` SSE parsing; `set_lightsheet_live_params` POST.
- **Frontend:** `node --check` + a Chrome-MCP harness (like A) for canvas + slide-and-see +
  states, plus a UI audit. (No JS unit harness in repo.)
- **FPS:** a documented **rig measurement step** (grab/deliver/paint fps); the binary path is a
  **conditional follow-up task gated on those numbers**.
- **Integration (rig):** live renders; slider changes the image; burst persists with
  temperature; live resumes.

Much of B1 needs the **real rig** (rpyc core, SPIM cameras, galvo/piezo) to verify; off-rig we
cover the streamer (fake core), proxy routes, client methods, and the frontend harness, and
defer live/FPS verification to the microscope — as with A.

## 6. Out of scope (B2)
Laser channel-preset browser; full timelapse/volume configuration form; side-by-side dual view;
richer camera selection.

## 7. Appendix: ASIdiSPIM two-camera model (B2 reference)

From the MM ASIdiSPIM plugin (`../micro-manager/plugins/ASIdiSPIM`), for when B2 adds side A/B:

- **Camera roles** (`data/Devices.java:93-121`): `CAMERAA` (side-A imaging), `CAMERAB`
  (side-B imaging), `CAMERALOWER` (bottom), `MULTICAMERA` (the MMCore *Utilities* "Multi Camera"
  fusion adapter). `SPIM_CAMERAS = {CAMERAA, CAMERAB}`; a `Sides` enum (A/B/NONE).
- **Live / single-side select** (`data/Cameras.java:163-186`, `setCamera`): `core.setCameraDevice(
  mmDevice)` bracketed by stop/start-live; `getCurrentCamera` (`:193-210`) treats the Core as the
  source of truth. Live can target one side **or** `MULTICAMERA` (dual live via the Utilities
  Multi-Camera device).
- **Multi-Camera discovery** (`DevicesPanel.java:167-180`): auto-detected by
  `getDeviceLibrary=="Utilities"` AND `getDeviceDescription=="Combine multiple physical cameras
  into a single logical camera"`.
- **Acquisition does NOT use the fusion device** (`AcquisitionPanel.java:2668-2671`): it runs
  **two parallel `startSequenceAcquisition`** on the physical side cameras into one shared buffer
  (equal ROI required) and **demuxes by the per-image `"Camera"` tag** (`:2736-2772`; side A → even
  channel indices, side B → odd). The Utilities Multi-Camera is reserved for **dual live**.

**Implication for gently:** B1's streamer already does `core.setCameraDevice(cam.name)` before the
continuous sequence — that IS the MM side-select point. To add side A/B in **B2**: register
`CAMERAA`/`CAMERAB` (and auto-discover the Utilities "Multi Camera") in `device_factory.py`; add a
camera-role selector that sets the device before starting the live sequence; for dual **live**,
point the sequence at the Multi-Camera device; for dual **acquisition**, either use the
Multi-Camera or replicate the two-sequence + `"Camera"`-tag demux. Correction to the working
assumption: ASIdiSPIM acquisition is **two independent sequences demuxed by tag**, not the
multicamera fusion — the fusion device is a live-only convenience.
