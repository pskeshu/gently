# Gently Desktop (Tauri shell)

A thin [Tauri](https://v2.tauri.app) desktop wrapper that turns gently into a
double-click Windows app. It **owns** the Python backend: on launch it spawns
`launch_gently.py --no-browser`, shows a splash while the server boots, then
renders the existing web UI in a native WebView2 window. No application logic
lives here — the web UI served by Python stays the single source of truth.

This is the desktop-packaging fold-in described in RFC #78
(`docs/superpowers/specs/2026-07-02-unified-launcher-design.md`).

## Why Tauri (and what it does / doesn't solve)

- **Tiny footprint** — uses the OS WebView2 (already present on Win 11), not a
  bundled Chromium.
- **Robust process ownership** — the shell puts the spawned Python into a
  Windows **Job Object** with `KILL_ON_JOB_CLOSE`. Python's own device-layer
  grandchild (via `DeviceLayerSupervisor`) inherits the job, so when the app
  quits *or crashes* the OS reaps the **entire** process tree — no orphaned
  device layer holding COM ports.
- **Not solved here:** bundling the Python environment (torch, anthropic,
  perception) into a redistributable installer. This build launches the repo's
  existing `.venv`. Shipping a self-contained `.msi`/`.exe` to another machine
  needs an embeddable-Python / PyInstaller sidecar — see *Bundling* below.

## Architecture

```
Tauri shell (Rust, WebView2 window)
  └─ spawns:  python launch_gently.py --no-browser      [in a kill-on-close Job]
                └─ spawns:  start_device_layer.py        (DeviceLayerSupervisor)
  ├─ shows splash/index.html while uvicorn boots
  └─ navigate() → http://localhost:8080  (the live gently UI)
```

- `src-tauri/src/main.rs` — spawn + Job Object teardown + wait-for-server + navigate.
- `splash/index.html` — boot splash (Rust updates its status line via `window.__gentlyStatus`).
- `src-tauri/tauri.conf.json` — one window, `frontendDist: ../splash`, NSIS bundle.

## Quitting (graceful handshake, issue #85)

Closing the window does **not** kill the backend outright. The shell first runs
a graceful handshake: it intercepts the close and `POST`s the backend's
loopback-only `/api/shutdown`, which stops a managed device layer via its clean
SIGTERM path and lets the backend drain state (e.g. session-replay final
batches) before exiting on its own; the shell waits a few seconds for the port
to go down, then quits. If an acquisition is running the backend answers `409`
and the shell asks for confirmation first — cancel and the window stays open.
If the backend is dead or hung, the shell just exits. Either way the Job-Object
kill-on-close (and best-effort `child.kill()`) remains the unchanged crash-safe
floor underneath.

## Prerequisites

- **Rust** (MSVC toolchain) — `rustup default stable-x86_64-pc-windows-msvc`
- **MSVC C++ build tools** (Visual Studio 2022 / Build Tools)
- **WebView2 runtime** (inbox on Windows 11)
- **Node** (only for the Tauri CLI: `npm install` in this folder)
- A working gently checkout with its Python **`.venv`** at the repo root

### Linux (dev machines)

The shell is cross-platform — the Job-Object code is `#[cfg(windows)]`, so on
Linux teardown is the best-effort `child.kill()` only (fine for dev; the
kill-on-close guarantee is a Windows-production feature). Verified on Manjaro
(KDE/Wayland, webkit2gtk 2.52):

- **Rust** — `rustup default stable`
- **WebKitGTK + GTK3** — `webkit2gtk-4.1` and `gtk3` dev packages
  (Arch/Manjaro: `pacman -S webkit2gtk-4.1`; Debian/Ubuntu:
  `apt install libwebkit2gtk-4.1-dev build-essential`)
- **Node** — as above
- The venv is found at **`.venv/bin/python`** (Unix layout) automatically;
  `GENTLY_PYTHON` overrides as usual

Same commands, bash syntax:

```bash
cd desktop && npm install
GENTLY_LAUNCH_ARGS="--no-api --offline --no-auth" npm run dev
```

`npm run build` is Windows-only as configured (`bundle.targets: ["nsis"]`) —
on Linux use `npm run dev`, or override targets if a Linux bundle is ever needed.

## Run (dev)

```powershell
cd desktop
npm install          # once — restores the Tauri CLI
npm run dev          # build + launch the shell (spawns the backend for you)
```

`npm run dev` (= `tauri dev`) compiles the shell, opens the window, spawns
`launch_gently.py --no-browser`, and navigates to the live UI once it's up.

UI-only (no hardware / API key / login) — handy for pure UI work:

```powershell
$env:GENTLY_LAUNCH_ARGS="--no-api --offline --no-auth"; npm run dev
```

## Making code changes — what reflects, and how

The window is just a WebView pointed at the Python server, so it splits in three:

| You edit… | Reflects by… | Rebuild? |
|---|---|---|
| **Web UI** (`gently/ui/web/templates`, `static/js`, `static/css`) | **Refresh the window** (Ctrl+R) | no — served live by Python |
| **Python backend** (`gently/**/*.py`, `launch_gently.py`) | restart the backend — or run with **`--reload`** (below), then Ctrl+R | process restart |
| **Rust shell / config** (`src-tauri/`, `splash/`) | `npm run dev` **auto-rebuilds + relaunches** | automatic (watched) |

**Python hot-reload.** Pass `--reload` and the backend auto-restarts whenever a
`gently/*.py` file changes (watchfiles) — then just Ctrl+R the window:

```powershell
$env:GENTLY_LAUNCH_ARGS="--reload --no-api --offline --no-auth"; npm run dev
# or, iterating in a plain browser instead of the shell:
uv run python launch_gently.py --reload
```

It restarts the *whole* backend (a few seconds), so use it for UI / backend dev,
not live hardware sessions.

## Build (installer)

```powershell
cd desktop
npm run build        # = tauri build — NSIS installer under src-tauri/target/release/bundle/
```

Then Gently installs like any Windows app (Start-Menu entry, double-click — no
terminal). The built app launches the **repo's** `.venv` + `launch_gently.py`
(paths baked at compile time / overridable — see env vars); it's a single-machine
build until Python bundling lands.

## Configuration (env vars read by the shell)

| Var | Default | Purpose |
|---|---|---|
| `GENTLY_HOME` | compile-time repo root | Directory to run the backend from |
| `GENTLY_PYTHON` | `<repo>/.venv/Scripts/python.exe` | Python interpreter to launch |
| `GENTLY_LAUNCH_ARGS` | *(none)* | Extra args appended to `launch_gently.py --no-browser` |
| `VIZ_PORT` | `8080` | Port the shell waits on and navigates to |
| `GENTLY_DEVICE_LAYER_SCRIPT` | `start_device_layer.py` | Alternate device-layer entry point (read by `DeviceLayerSupervisor`) |

## Bundling Python (the remaining long pole)

`tauri build` today does **not** package Python. To make a redistributable app:

1. Produce a self-contained backend (PyInstaller one-folder of `launch_gently`,
   or an embeddable-Python tree with the deps installed).
2. Ship it as a Tauri **sidecar** (or under the app's resource dir) and point
   `GENTLY_PYTHON` / `GENTLY_HOME` at it.
3. Expect a multi-GB artifact (torch/CUDA dominate) and plan auto-update
   accordingly.

The Job-Object ownership and boot/navigate flow are unchanged by bundling — only
*where Python comes from* changes.
