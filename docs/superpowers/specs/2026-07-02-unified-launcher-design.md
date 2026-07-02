# Unified Launcher — single entry point + device-layer process management — Design

Date: 2026-07-02
Status: **Initial design (RFC) — PARKED.** A starting point to build on, not a finished
design; direction + mockup done, implementation deferred to a future session.
Idea credit: **Magdalena** — the single unified launcher.
Branch: `feature/unified-launcher` (off `development`).

## Problem

Starting gently today is two commands in two terminals: `python start_device_layer.py`
(the hardware server) then `python launch_gently.py` (agent + web). gently can't
start or stop the device layer — it only connects to it as an HTTP client — so there's
no in‑app way to bring hardware up/down. Goal: make `launch_gently.py` the **single**
entry point, with a dead‑simple launch gate and the ability to start/stop the device
layer from gently.

## How the two processes relate (today)

- `start_device_layer.py` boots the hardware control server (aiohttp) on **port 60610**
  (`DEVICE_PORT`) and runs the configured hardware module's `create_device_layer()`.
  Independent OS process.
- `launch_gently.py` boots the agent + web/viz server and creates an HTTP client
  (`QueueServerClient → http://127.0.0.1:60610`); `client.is_connected` drives the
  "Device: ○ offline" banner. The two only talk over HTTP.

## The launch gate — BARE BASIC (the key decision)

The launch screen answers exactly two questions, nothing more:

1. **Microscope hardware — on/off.** On → gently starts + connects the device layer.
   Off → software‑only (analysis, planning, reviewing saved data).
2. **AI agent (API) — on/off.** On → chat, perception, planning (uses the API key).
   Off → UI‑only.

Then **Let's go →**. That's the whole screen. Everything else — organism, hardware
module, device port, SAM device, session resume — is a **default** (config / last‑used),
reachable behind a muted **"Advanced options"** disclosure and in **Settings**. It is
NOT on the gate: if you're not using hardware, you should never be asked which hardware.

Visual reference (dark, sharp, sleek): `docs/superpowers/mockups/2026-07-02-launcher-gate.html`
— two toggle cards (`Microscope hardware`, `AI agent`) + `Let's go →` + `Advanced
options · remembers your choice`.

## Process model — managed child subprocess

A **`DeviceLayerSupervisor`** in the gently process spawns `start_device_layer.py` as a
**child** (`subprocess.Popen([sys.executable, "start_device_layer.py", "--port", …,
"--sam-device", …])`), holds its handle, monitors liveness, captures its log, and stops
it. Ownership is what makes start/stop‑from‑UI and no‑orphans work.
- API: `start(config)`, `stop(force=False)`, `status()`, log tail, `atexit`/signal cleanup.
- **External device layer still supported:** if one is already running on 60610, gently
  connects and shows it as "external (not managed)" — it just won't stop what it didn't start.
- Rejected alternatives: independent processes + a `/shutdown` endpoint (weaker ownership,
  more parts); in‑process device layer (a hardware crash would take down the UI).

## Boot flow (defer‑init)

`launch_gently` boots a minimal web server → shows the **launch gate** (prefilled from
last choice) → on **Let's go**, it initializes per the two toggles: start the agent (if
API on) and start the device layer via the supervisor (if hardware on) → then the landing
("Good afternoon…") → dashboard. Deferring heavy init until the gate is submitted is what
lets a web screen control boot‑level behavior (agent on/off).

## Runtime control

A **device‑layer panel in the Devices tab** mirrors the gate's hardware block at runtime:
live status (running / stopped / external / crashed), Start / Stop, and a tail of the
device‑layer log — so you can bring hardware up/down without restarting gently.

## Stop safety (graceful + mid‑run guard)

Stop → if a plan/acquisition is active, warn + require explicit confirm ("hardware is
active — stop anyway?") → SIGTERM (clean shutdown) → SIGKILL fallback (~5 s). Reuses the
409 + `"blocked"` pattern from the thermalizer work. Startup failure (hardware off)
surfaces the existing plain‑language `_render_startup_failure` diagnosis, not a traceback.

## Persistence

`config/launch.local.json` (gitignored) remembers the two toggles (+ any advanced values)
so the gate is prefilled every boot.

## Non‑goals (YAGNI)

Auto‑restart‑on‑crash loops; multiple / remote device layers (mesh already covers
cross‑machine); managing an externally‑started device layer's lifecycle; putting
organism/hardware/port/SAM/session on the gate (they're defaults + Advanced/Settings).

## Phasing (when resumed)

1. `DeviceLayerSupervisor` + the runtime Devices start/stop panel + the **hardware toggle**
   on a minimal gate. Delivers "no more separate `start_device_layer.py`" immediately.
2. Defer‑init boot refactor + the **agent toggle** + persistence + "Advanced options".

## Open questions to fold in (this is initial work)

This is a starting point — the following must be worked through as it grows:

- **Background startup.** After "Let's go", the device‑layer startup should run in the
  **background (non‑blocking)** while gently proceeds to the dashboard — not block the gate
  until the device connects. (To confirm + design the progress signalling.)
- **Usable during startup.** gently must stay **usable while the device layer boots** —
  software features (planning, review, analysis) available immediately; hardware‑dependent
  actions gated behind a "device starting…" state until connected.
- **On‑demand startup‑sequence screen.** There must be a screen to watch the device
  **startup sequence** on demand — a device console (in the Devices panel) streaming the
  `start_device_layer` boot log + per‑stage connection progress.
- **Shutdown from that same screen.** That console is also the **shutdown control** — start
  the sequence, watch it, and stop the device from one place.
- **Guided shutdown housekeeping.** Shutdown is a **sequence, not just a kill** — e.g. it
  can prompt the operator to **reset the F drive** (post‑shutdown drive/housekeeping) as a
  step in the flow.

## Future direction — Windows desktop app (Electron)

Bigger fold‑in to weigh: package the unified launcher as a **desktop app for Windows**,
potentially **Electron**, so gently is a double‑click application instead of terminal
commands. Electron would *be* the launcher shell — it owns and spawns the Python backend
(`launch_gently`) and the device‑layer child, renders the existing web UI in a native
window (the launch gate is the first screen), and gets native process lifecycle
(children killed on quit), a tray/menu, and a real installer + auto‑update.

Tradeoffs for later: bundling/shipping the Python environment (embeddable Python /
PyInstaller), app size, auto‑update, and keeping the **web UI the single source of truth**
(Electron stays a thin shell — no UI logic moves into it). A lighter native wrapper
(**pywebview** or **Tauri**) is the fallback if Electron's footprint is too heavy. This is
a packaging/architecture decision, not required for the launcher's first cut — but the
managed‑child process model here is exactly what an Electron shell would take over.

## Decisions log (from brainstorming)

- Launcher model: **web startup screen + runtime panel**.
- Boot flow: **show the gate every boot, remember last choices** (one click to proceed).
- Stop: **graceful SIGTERM + mid‑run guard + SIGKILL fallback**.
- Scope: **fold in gently's own options** — but as a **bare‑basic 2‑question gate**
  (hardware, agent), with the rest behind Advanced/Settings.
