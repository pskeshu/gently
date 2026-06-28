# Lightsheet live-view FPS measurement + transport decision (B1 Task 7)

Status: **deferred to the rig** — the numbers below must be filled in on the microscope
(the streamer needs the real `pymmcore` core, SPIM camera, and rpyc transport; it cannot run
on the Linux dev box). This file is the measurement protocol + the decision gate.

## What to measure

With the Manual view open and lightsheet live running, record three rates:

| Metric | Where to read it |
|---|---|
| **device grab rate** (frames peeked/encoded /s) | device-layer log / instrument `_lightsheet_streamer` |
| **delivered rate** (frames broadcast /s) | device-layer `_broadcast_lightsheet` |
| **browser paint rate** | the Manual-view FPS readout (`computeLightsheetFps`) |

Record at two resolution/quality settings:
- default **512 px / JPEG q70** (`_ls_target_max_dim=512`, `_ls_jpeg_quality=70`)
- reduced **384 px / q60**

Note the exposure used (the peek floor is `max(exposure, 1/30 s)`).

## Target

**≥ ~15 fps usable for focus** (stretch 25–30), end-to-end latency < ~150 ms.

## Diagnosis rule (the gate)

- **device grab < target** → limiter is exposure / readout / rpyc, **not** transport. Tune
  exposure, the 512 px size, and JPEG quality. A binary transport path will NOT help — stop here.
- **device grab ≥ target but browser paint < target** → transport is the bottleneck → build the
  **binary WebSocket path** (below).

## Conditional escalation — binary WebSocket path

Only if the diagnosis points to transport. The current path is base64-JPEG-in-JSON over SSE →
EventBus → `ConnectionManager.broadcast` (`json.dumps` + `send_text`) — `connection_manager.py`
has **no `send_bytes` path** (confirmed). Escalation, on the **agent→browser hop** (where cost
multiplies per client):

- push raw JPEG bytes via `websocket.send_bytes(prefix + jpeg)` (a 1-byte type tag identifies a
  lightsheet frame), bypassing **base64 (+33%)**, the **per-client `json.dumps`**, and the
  **EventBus fan-out**;
- browser `onmessage` (binary) → `createImageBitmap(new Blob([buf]))` → `ctx.drawImage`;
- the device→agent SSE stays as-is (single consumer = the monitor, so its base64 cost is paid
  once, not per browser).

Re-measure after building; record the before/after numbers here.

## Results (fill in on the rig)

| setting | device grab fps | delivered fps | browser paint fps | exposure | notes |
|---|---|---|---|---|---|
| 512px/q70 | _TBD_ | _TBD_ | _TBD_ | _TBD_ | |
| 384px/q60 | _TBD_ | _TBD_ | _TBD_ | _TBD_ | |

Decision: _TBD (transport bottleneck? build binary path Y/N)_
