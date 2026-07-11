# Gently — Concurrency & Runtime Architecture

How the system runs temperature telemetry, device-state polling, experiments,
image acquisition, and perception/VLM "at the same time" without stepping on
itself. The short version: **almost nothing runs truly in parallel — the design
quarantines the blocking work and serializes the hardware, then keeps both event
loops responsive by offloading and decoupling everything slow.**

> Scope: the concurrency model. For storage layout see `CLAUDE.md`; for the
> device/hardware plugin model see `docs/asi-plugin-architecture.md`.

## 1. Two processes, bridged by HTTP + a shared filesystem

This split is the load-bearing fact of the whole design.

```
┌────────────────────────────────────────┐        ┌────────────────────────────────────────┐
│  APP / VIZ process   (FastAPI :8080)    │        │  DEVICE-LAYER process  (aiohttp :60610) │
│  gently/app/agent.py                    │        │  gently/hardware/dispim/device_layer.py │
│                                         │  HTTP  │                                          │
│  • agent + TimelapseOrchestrator        │◄──────►│  • the ONLY code touching hardware:      │
│  • Perceiver (gently_perception, VLM)   │  (DiSPIM│    MMCore/pymmcore, Ophyd, Bluesky RE    │
│  • AsyncAnthropic clients               │  Client)│  • plan queue + single executor          │
│  • EventBus + WebSocket ConnectionMgr   │        │  • 3 state pollers + camera/LS streamers │
│  • TemperatureSampler, DeviceStateMonitor│       │                                          │
└──────────────┬──────────────────────────┘        └───────────────┬──────────────────────────┘
               │                     shared filesystem (incoming/ TIFF staging, session store)
               └──────────────────────────────────────────────────┘
```

- The **device-layer process** (`device_layer.py`) is the *only* code that ever
  touches the microscope: MMCore/pymmcore, the Ophyd devices, and a Bluesky
  `RunEngine`. All truly blocking hardware work is quarantined here.
- The **app/viz process** (`app/agent.py`, FastAPI+uvicorn) owns everything
  cognitive and user-facing: the agent, the `TimelapseOrchestrator`, the
  `Perceiver` (external `gently_perception` VLM package), the Anthropic clients,
  the in-process `EventBus`, the WebSocket `ConnectionManager`, and the pollers.
- The app reaches hardware **only** through `DiSPIMClient` (`client.py`), over a
  single shared `aiohttp.ClientSession`. It never imports MMCore.

Because blocking hardware calls live in a separate process, the app-side loop
stays a light cooperative-asyncio world.

## 2. One asyncio event loop per process

- **Device loop:** HTTP routes + the RunEngine driver + the plan queue/executor +
  three state pollers (XY ~5 Hz, piezo/galvo ~1 Hz, full property cache ~15 s) +
  subscriber-gated camera/lightsheet SSE streamers.
- **App loop:** FastAPI + the `TimelapseOrchestrator` acquisition loop +
  `TemperatureSampler` (1 Hz) + `DeviceStateMonitor` + `Perceiver` calls + one
  coroutine per open browser WebSocket.

## 3. The core trick: hardware is serialized, not parallelized

Every Bluesky plan — a move, a snap, a volume/burst acquisition, a focus/calibration
sweep — is submitted as a `PlanRequest` onto **one `asyncio.Queue`
(`self._plan_queue`), drained by one `_plan_executor` task**. Only one plan owns
the hardware at a time.

- `submit_plan` enqueues and `await`s a per-request `asyncio.Future`; the executor
  sets the result/exception, which flows back to the awaiting HTTP handler. On
  failure the executor records it in `_plan_execution_log` and continues to the
  next queued plan.
- Underneath, **pymmcore's internal `g_core_lock` is the real mutex** serializing
  every actual core call across pollers *and* plans. The `DiSPIMSystem` facade is
  the single place the process touches MMCore.

A microscope has one stage and one camera; single-file execution is correct, not
a limitation.

## 4. How constant polling coexists with long experiments

Four cooperating mechanisms keep the loops responsive while a plan runs:

1. **Offload blocking reads.** Every MMCore read, camera grab, SAM call, and
   transient temperature probe goes through `asyncio.to_thread`; the camera
   sequence runs on its own `threading.Thread` returning an Ophyd `Status`. The
   loop itself never sits on I/O.
2. **Split pollers by cadence.** The three device-state pollers are independent
   tasks so a slow (~1.5 s) full-state-cache read cannot stall the ~5 Hz XY path.
3. **`pause_state_updates()` — a reference-counted async context manager.** Heavy
   plans (a `frozenset` of names) wrap execution in it, incrementing a counter;
   every poller/streamer checks `if self._state_pause_counter > 0` and *skips its
   MMCore read*, emitting only ~2 s heartbeats. The plan gets full serial/camera
   bandwidth instead of the pollers fighting it for `g_core_lock`. Nested heavy
   sections stack and unwind cleanly.
4. **Telemetry bypasses the plan queue.** `GET /api/temperature/status` reads the
   temperature Ophyd device directly (`temp.read()` over serial/MQTT — a device
   wholly separate from MMCore); `GET /api/devices/state` returns the cached
   `_state_latest` snapshot. Neither sits behind a running experiment, so status
   polls are never blocked by a long acquisition. (MMCore push callbacks also
   mirror joystick/property changes into `_state_latest` via
   `loop.call_soon_threadsafe` with ~50 ms debouncing.)

### Temperature specifically
The vendor SDK backend (serial or MQTT) runs its own **background daemon thread**
that ingests the controller's 500 ms telemetry broadcast into a cache
(`self.telemetry`); `get_water_temp()`/`get_system_state()` return the cached
value (non-blocking). `TemperatureController.read()` returns that cache, and
`TemperatureSampler` (`interval_sec=1.0`) polls it at 1 Hz → persists + emits
`TEMPERATURE_UPDATE`. So we ride the telemetry indirectly, resampled at 1 Hz.

## 5. Image data travels via the filesystem, not JSON

Arrays over ~1 MB are written as **TIFF into the shared `incoming/` staging dir**;
only a small `{__file_ref__, path, shape, dtype}` dict crosses HTTP (`serialize_value`).
The client resolves the ref with `tifffile` and hands the decoded array to
`register_volume`, which renames the file into the session store and stamps it
with the latest temperature sample — avoiding a multi-GB JSON blob and a double
decode.

## 6. Perception / VLM / events are decoupled (fire-and-forget)

- **Perception never gates acquisition.** The `TimelapseOrchestrator` does
  `asyncio.create_task(self._run_perception(...))` rather than awaiting it inline,
  so a slow VLM call doesn't hold up the next embryo. Inside the task the
  `Perceiver` and the Claude client are awaited cooperatively, and every Claude
  call is wrapped in `asyncio.wait_for(timeout=30)` returning a safe fallback
  instead of raising into the loop.
- **The `EventBus` publishes without awaiting handlers** (`event_bus.py`): sync
  handlers run inline, async handlers are scheduled via `asyncio.ensure_future` /
  `loop.call_soon_threadsafe`; high-volume telemetry types skip the bounded
  history deque, so one slow WebSocket client delays only its own broadcast.

## 7. Backpressure & failure isolation

- **SSE streams** use per-subscriber queues bounded at `maxsize=4`; device-state
  broadcasts drop slow subscribers, camera/lightsheet streams drop the *oldest*
  frame and push the newest so steady clients keep fresh frames.
- **WebSocket fan-out** (`ConnectionManager.broadcast`) sends per client under an
  `asyncio.Lock` and drops clients that error.
- **Failure = a gap, not a crash.** A failed temperature poll logs once and backs
  off (`1.0 s → min(interval*10, 30 s)`); a stalled SSE forces a watchdog
  reconnect after 60 s (chosen to tolerate the quiet windows during heavy plans);
  volume acquisition always disables lasers on error to protect the sample.
- **Config safety.** `POST /api/temperature/config` refuses (`409`) while the
  RunEngine is not idle or a ramp holds the controller lock; `health_check()` is
  read-only and never flips the connected flag, so a transient status-poll timeout
  can't disconnect an in-flight acquisition.

## 8. Known limits & bottlenecks

- **RunEngine on the loop.** `self.RE(plan)` is invoked synchronously on the
  device-layer loop, so while a plan runs the loop is largely occupied — which is
  exactly why heavy plans quiet the pollers and the app-side watchdog tolerates
  60 s of silence. Whether the telemetry HTTP handlers stay fully responsive
  mid-acquisition depends on Bluesky's internal threading (external package) and
  is not verified from repo code.
- **Unbounded in-flight perception.** `_perception_tasks` is a plain set that
  self-prunes; sustained fast acquisition against a slow VLM could grow concurrent
  Claude calls without an explicit cap. A semaphore is the obvious guard if
  cadence is ever pushed.
- **~~Synchronous O(n) prediction writes~~ — FIXED.** `store_prediction` used to
  re-parse the entire `predictions.jsonl` on every append to compute the next id.
  Now O(1) via a bounded tail read (`_last_jsonl_record`, `file_store.py`). Other
  `FileStore` JSONL writes remain synchronous on the app loop but are single-line
  appends.
- **External VLM internals.** How `gently_perception`'s `Perceiver` implements its
  VLM call (async httpx vs sync-in-thread) is not inspectable from this repo; the
  orchestrator awaits it, implying a coroutine.

## The model in one line

Two single-threaded event loops; hardware fenced into one process behind a
one-at-a-time plan queue; everything blocking pushed into threads; and the
slow/cognitive work (VLM, persistence, UI) decoupled with fire-and-forget tasks —
so the system *feels* concurrent while the microscope itself stays strictly
serialized.

---
*Generated from a code-grounded architecture pass (Claude Opus 4.8), 2026-07-01.*
