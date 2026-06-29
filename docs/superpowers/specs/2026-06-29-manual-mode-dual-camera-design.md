# Design: Manual mode B2 — dual-camera + laser-preset browser + timelapse form

Status: design 2026-06-29 (after recon). Branch `feature/manual-mode-dual-camera` (off F). Extends B1's
single-camera manual mode with three independent enhancements. Each has a HEADLESS-buildable part (built
+ tested now against a fake MMCore) and a RIG-DEFERRED part (live hardware — verified on the scope).

## 0. What B1 gives us (recon)
- Single-camera (HamCam1) lightsheet live streamer (`device_layer.py` `_ensure_lightsheet_sequence_sync`
  → `core.setCameraDevice(cam.name)`, the side-select point), the `#devices-view-manual` UI rail, the
  `require_control` proxy routes (`routes/data.py`), client methods.
- **Laser presets already enumerated end-to-end:** `DiSPIMLightSource._get_available_configs` → device-layer
  `GET /api/laser/configs` → `client.get_laser_configs()` → proxy `GET /api/devices/laser/configs`
  (returns the 7 `Laser`-group presets). Only the UI consumer + a set-preset proxy are missing.
- Volume/burst acquisition with full geometry params (`acquire_volume(num_slices, exposure_ms,
  galvo_amplitude/center, piezo_amplitude/center, laser_config, laser_power_*)`).
- ASIdiSPIM dual-camera reference: B1 spec §7 (CAMERAA/CAMERAB, Multi-Camera fusion is live-only;
  dual-acquire = two parallel sequences + `"Camera"`-tag demux).

## 1. Laser-preset browser (smallest — list already exists)
- **Backend (headless):** new `POST /api/devices/laser/config` (`require_control`) proxy → `client.set_laser_config(name)`.
  (`/laser/off` today only hardcodes "ALL OFF".)
- **UI (headless):** replace the static `#devices-ls-laser-status` "OFF (brightfield)" indicator in the
  Illumination group with a `<select>` populated from `GET /api/devices/laser/configs`; on change POST the
  chosen preset. Keep "ALL OFF" the safe default + the existing manual-view-entry laser-off safety (don't
  remove the I3 guard — selecting a preset is an explicit user action).
- **Rig-deferred:** the actual laser firing (the preset just calls `setConfig` on the rig).

## 2. Dual-camera config
- **Backend (headless):** register a second `DiSPIMCamera("HamCam2")` as `devices["camera_b"]` in
  `device_factory.py` — DEFENSIVELY (only if the camera is in the core's loaded devices; skip + log
  otherwise, so single-camera rigs still start). Add a `side` field ('A'|'B') to `_ls_params` +
  `handle_lightsheet_params`; `_ensure_lightsheet_sequence_sync` picks `camera` vs `camera_b` by side and
  restarts the sequence on side change (reuse the exposure-change restart path). New `GET /api/devices/cameras`
  endpoint listing available camera roles (A always; B if registered).
- **UI (headless):** a "Side A / B" selector in the manual rail → carries `side` on the live/params POST.
- **Rig-deferred:** live DUAL view via the "Multi Camera" fusion device (live-only) + dual-side acquisition
  (two parallel `startSequenceAcquisition` + tag demux). v1 = single live stream, switchable side.

## 3. Timelapse config form
- **Backend (headless):** new `POST /api/devices/timelapse/start` (`require_control`) proxy wrapping the
  agent path `start_adaptive_timelapse(embryo_ids, stop_condition, interval_seconds, condition_value,
  monitoring_mode)` (validate params; resolve the orchestrator like the agent tool does). The volume
  geometry (num_slices/exposure/galvo±/piezo±/laser_config/power) is captured in the form + passed through
  / persisted to per-embryo calibration where applicable.
- **UI (headless):** a collapsible "Timelapse" panel in the manual rail gathering cadence/stop/embryos/
  monitoring_mode + the volume geometry, reading `GET /api/devices/scan_geometry` + `/api/devices/laser/configs`
  for defaults. A "Start timelapse" submit → the new proxy.
- **Rig-deferred:** the actual timelapse run + galvo/piezo motion.

## 4. Out of scope / deferred
- Live Multi-Camera dual view + dual-side acquisition demux (rig).
- Saving timelapse configs as reusable presets (could reuse the tactic-library/plan-template later).
- Per-line laser power UI beyond the preset (the clamps in `optical.py` still apply).

## 5. Testing
- Laser-preset: the POST proxy (TestClient + mock client asserts `set_laser_config(name)`); `node --check`
  + Chrome audit of the dropdown populated from a stubbed configs endpoint.
- Dual-camera: device_factory registers camera_b against a FAKE core that has HamCam2 (and skips when
  absent); the `side` param threads into `_ls_params` + selects the camera; `/api/devices/cameras` lists
  roles; `node --check` + Chrome audit of the side selector.
- Timelapse: the start proxy validates + calls the orchestrator path (mock); the form gathers + posts the
  params; `node --check` + Chrome audit of the form.
- All three: backward compatible (single-camera rig unaffected; the laser-off safety intact).
