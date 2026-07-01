# Settings Panel — editable ACUITYnano thermalizer + config visibility — Design

Date: 2026-07-01
Status: Approved (build all phases)
Branch: feature/temperature-operations-all (→ #72)
Source: Opus audit workflow + two implementation-reference passes.

## Problem / audit

The gently "Settings" panel is **100% client-side display preferences** — every control writes browser `localStorage` (`gently-dashboard-config` + `gently-theme`); `settings.js` makes **zero** backend calls. The **ACUITYnano thermalizer connection** (serial COM / MQTT-HiveMQ / mock) is in **no GUI** — it lives in `config/config.yml` `temperature:`, read once at device-layer boot; changing transport/port/creds = edit YAML + restart. Naming trap: the Vitals "Temperature model (20/25 °C)" radio is a *developmental-timing reference curve*, not the hardware setpoint.

Two-process: viz (FastAPI :8080) proxies to the device layer (aiohttp :60610) via `DiSPIMClient`; the controller lives only in the device-layer process.

## Decisions (user)

- Build **all phases**.
- Apply mode: **try live hot-swap, fall back to restart-required**.
- **Mock-SIM: dev/debug only** — not selectable in the production panel.

## Design

New server-backed **"Hardware / Thermalizer"** section (separate from the localStorage panel; explicit "machine-wide, saved on the server" note). Fields grounded in `temperature.py`:
- Backend radio **Serial | MQTT (HiveMQ)** (Mock hidden unless a dev flag). Serial: `com_port` (required), `baud_rate` (115200). MQTT: `broker`, `port` (8883), `user`, `password` (write-only, `••••`, never echoed; blank = embedded HiveMQ SIM). Common: `stabilize_timeout` (600), `feedback_peltier`.
- **Test connection** (non-committing): build transient backend → `read()`/`get_system_state` → `close()` → report; never swaps the live device.
- **Apply**: live hot-swap — build the NEW controller first (`create_temperature_controller`), keep the old on failure, swap `self.devices["temperature"]`, `old.close()`. Guard: **409 if `self.RE.state != "idle"`** or a `set()` worker holds the controller lock (mid-ramp/mid-plan swap corrupts a live `bps.mv(temperature,…)`). If teardown/rebuild can't apply live, persist + "restart the device layer to apply" banner.
- **Persistence**: sidecar `config/config.local.yml` (`temperature:` block) merged over `config.yml` at device-layer boot — preserves `config.yml`'s comments; password written only when a new non-redacted value is submitted.

**Effective-config viewer** (Phase 2, read-only, secrets redacted): ports/hosts, model IDs, storage base_path + derived dirs, mmconfig/mmdirectory, organism/hardware, switchbot name, coverslip, XY safety envelope (from the live device-state stream), mesh instance-id + cert fingerprint, timeouts, ML params, `ux_v2`. **Never expose** (redact/omit): `ANTHROPIC_API_KEY`, `GENTLY_CONTROL_TOKEN`, `mesh_key.pem`, MQTT creds.

## Call chain (per new capability)
Browser `fetch('/api/devices/temperature/config…')` → FastAPI `routes/data.py` (`require_control` on mutations) → `_resolve_client().<m>()` → `client.py` `_api_*` → device-layer `handle_*` → `self.devices["temperature"]`.

## Phasing / changes

**Phase 0 (visibility + test):**
- device-layer: `GET /api/temperature/config` (current `temperature` block, password redacted, + live backend + `read()` state); `POST /api/temperature/config/test` (transient backend probe). Register in `on_start` (~:3394).
- client: `get_temperature_config`, `test_temperature_config`.
- viz: `GET /api/devices/temperature/config` (read-only), `POST /api/devices/temperature/config/test` (`require_control`).
- UI: read-only Hardware/Thermalizer section + Test button (a separate `ThermalizerSettings` JS object, isolated from `SettingsManager`); relabel the Vitals "Temperature model" field.

**Phase 1 (editable + live reconnect):**
- device-layer: `POST /api/temperature/config` (validate; 409 guard; build-new-before-swap; sidecar persist; return live state); boot-merge sidecar over `config.yml` `temperature` (after `yaml.safe_load` ~:227).
- client: `set_temperature_config`. viz: `POST /api/devices/temperature/config` (`require_control`).
- UI: editable Serial/MQTT form (Mock dev-only), Apply (no auto-save), applied-live vs restart-required banner.

**Phase 2 (visibility + prefs):**
- viz `GET /api/config/effective` (read-only, redacted) + a read-only "Effective config" viewer in the panel.
- Server-side dashboard-pref **defaults** + reset/export/import (viz route storing rig defaults in a file; `settings.js` layers over localStorage).
- Restart-required editors for SAFE `settings.py` knobs (timeouts, mesh timing, ML, `ux_v2`, NCBI) via a `config/settings.local.yml`/env override read at launch, with an explicit "restart required" path (never mutate the frozen `settings` singleton live). If the launcher-override mechanism proves out-of-scope, ship the viewer + pref-defaults and defer the editors.

## Safety
- Never echo the MQTT password (redact on GET; persist only on new value). Redact all secrets in the effective-config viewer.
- `require_control` on every config write/test proxy.
- 409 reconfigure guard while RE running / lock held; build-new-before-swap so a bad config never leaves the rig with no thermalizer.
- Keep the 0.0–99.9 °C clamp in both layers; GUI can't widen it.
- Sidecar persistence (not `config.yml` rewrite); restrict perms on any file holding the plaintext MQTT password.
- Restart-required for frozen `settings.py` values — write override + prompt restart, never live-mutate.

## Rig-only / honesty
The vendor SDK (`acuitynano_precision_thermalizer_*`) isn't on PyPI and isn't installed in the hardware-free shim, so **serial/MQTT construction, Test, and live-swap are rig-verified**; the shim path exercises routes + validation + the mock backend + UI flow. Live hot-swap's clean teardown per transport needs on-rig confirmation (fallback = restart banner).
