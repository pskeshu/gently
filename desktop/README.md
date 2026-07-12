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

## Prerequisites

- **Rust** (MSVC toolchain) — `rustup default stable-x86_64-pc-windows-msvc`
- **MSVC C++ build tools** (Visual Studio 2022 / Build Tools)
- **WebView2 runtime** (inbox on Windows 11)
- **Node** (only for the Tauri CLI: `npm install` in this folder)
- A working gently checkout with its Python **`.venv`** at the repo root

## Run (dev)

```bash
cd desktop
npm install                      # once — installs @tauri-apps/cli
# UI-only smoke test (no hardware, no API key, no login):
GENTLY_LAUNCH_ARGS="--no-api --offline --no-auth" npx tauri dev
# Full app (needs ANTHROPIC_API_KEY in the environment / repo .env):
npx tauri dev
```

`tauri dev` builds and launches the shell; it spawns the backend for you.

## Build (installer)

```bash
cd desktop
npx tauri build                  # produces an NSIS installer under src-tauri/target/release/bundle/
```

The built app launches the **repo's** `.venv` + `launch_gently.py` (paths baked
at compile time / overridable — see env vars). It is a single-machine build
until Python bundling lands.

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
