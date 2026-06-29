# Manual mode B2 — dual-camera + laser-preset browser + timelapse form Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Steps use `- [ ]`.

**Goal:** Extend B1's single-camera manual mode with a laser-preset browser, dual-camera (side A/B) config, and a manual timelapse-config form — building the headless parts now (live dual-view + real acquisition are rig-deferred).

**Architecture:** New `require_control` proxy routes wrapping existing client/device-layer + agent-tool paths; device_factory registers a second camera defensively; UI additions to `#devices-view-manual` / `devices.js`.

## Global Constraints
- HEADLESS-buildable parts only; mark RIG-DEFERRED parts (live dual view, real acquisition/timelapse) as noted in the spec — don't fake hardware.
- Backward compatible: single-camera rigs must still start (defensive HamCam2 registration); the manual-view-entry laser-off safety (B1 I3) stays intact.
- Laser preset list already exists: `GET /api/devices/laser/configs` (data.py:526). Reuse it; add only the set proxy + UI.
- Proxy routes mirror the existing `routes/data.py` `require_control` pattern (e.g. `/api/devices/laser/off` :508). Device-layer/client calls mirror existing ones.
- UI extends `#devices-view-manual` (index.html ~:720-835) + `DevicesManager` in `devices.js`.
- Git hygiene: stage only your files by explicit path; never `git add -A`.

---

### Task 1: Laser-preset browser
**Files:** Modify `gently/ui/web/routes/data.py` (add `POST /api/devices/laser/config` `require_control` → `client.set_laser_config(name)`, mirror `/laser/off` :508); `gently/ui/web/templates/index.html` (the Illumination group ~:800 — replace the static `#devices-ls-laser-status` indicator with a `<select id="devices-laser-preset">` + keep a status line); `gently/ui/web/static/js/devices.js` (populate the select from `GET /api/devices/laser/configs` on manual-view entry; on change `POST /api/devices/laser/config {config}`; default/show "ALL OFF"). Test: `tests/test_laser_preset_route.py`.
- [ ] Confirm `client.set_laser_config` + the proxy pattern (`/laser/off`) + how devices.js fetches/posts + caches DOM. Add the set proxy + the dropdown; keep the I3 laser-off-on-entry safety (selecting a preset is an explicit override).
- [ ] TDD: the POST proxy calls `set_laser_config(name)` (TestClient + mock client); bad/missing config → 400. `node --check devices.js`. Build a Chrome harness (stub `/configs` + `/config`) for the controller to audit the dropdown. `pytest tests/test_laser_preset_route.py -v`; `pytest -q` clean. Commit `feat(b2): laser-preset browser (set proxy + manual-mode dropdown)`.

### Task 2: Dual-camera config
**Files:** Modify `gently/hardware/dispim/device_factory.py` (register `DiSPIMCamera("HamCam2")` as `devices["camera_b"]` DEFENSIVELY — only if in the core's loaded devices; `default_config["camera_b_name"]="HamCam2"`); `gently/hardware/dispim/device_layer.py` (add `side` to `_ls_params` (default 'A') + `handle_lightsheet_params`; `_ensure_lightsheet_sequence_sync` selects `camera`/`camera_b` by side, restart on side change like exposure; a `handle_get_cameras` + route `GET /api/cameras`); `gently/ui/web/routes/data.py` (`GET /api/devices/cameras` proxy + thread `side` through `POST /api/devices/lightsheet/live/params`); `index.html` + `devices.js` (a Side A/B selector). Test: `tests/test_dual_camera_factory.py`, `tests/test_lightsheet_side_param.py`.
- [ ] Confirm device_factory camera creation (:103-107) + `_ls_params`/`_ensure_lightsheet_sequence_sync` (~:177, :1018-1037) + `handle_lightsheet_params` (:1169). Register camera_b only if present in `core` (skip+log otherwise). Thread `side`; restart sequence on side change.
- [ ] TDD (fake core with HamCam2 → camera_b created; fake core without → skipped, no crash; side param selects the camera + triggers restart). `node --check`. Chrome harness for the side selector. `pytest tests/test_dual_camera_factory.py tests/test_lightsheet_side_param.py -v`; `pytest -q` clean. Commit `feat(b2): dual-camera config (HamCam2 + side selector)`.

### Task 3: Timelapse config form
**Files:** Modify `gently/ui/web/routes/data.py` (add `POST /api/devices/timelapse/start` `require_control` → resolve the orchestrator/agent path and call `start_adaptive_timelapse(...)` with validated params: interval_seconds/stop_condition/embryo_ids/condition_value/monitoring_mode + the volume geometry passed through; mirror how the agent tool resolves the orchestrator); `index.html` (a collapsible "Timelapse" panel in the manual rail — cadence/stop/embryos/monitoring_mode + num_slices/exposure/galvo±/piezo±/laser_config); `devices.js` (read `GET /api/devices/scan_geometry` + `/api/devices/laser/configs` for defaults; submit → the new proxy). Test: `tests/test_timelapse_start_route.py`.
- [ ] Confirm `start_adaptive_timelapse` signature (timelapse_tools.py:61) + how the route can reach the orchestrator (the app server holds it — confirm the proxy's access, or call the device/agent path). Validate params; RIG-DEFERRED real execution — the route wires + validates, returns the orchestrator's result/string.
- [ ] TDD: the start proxy validates + calls the orchestrator path (mock) with the right params; bad params → 400. `node --check`. Chrome harness for the form. `pytest tests/test_timelapse_start_route.py -v`; `pytest -q` clean. Commit `feat(b2): manual timelapse-config form (start proxy + rail panel)`.

## Self-Review
- Laser-preset→T1; dual-camera→T2; timelapse-form→T3. ✓
- Open confirmations: set_laser_config proxy + devices.js patterns (T1); device_factory + _ls_params + side restart (T2); start_adaptive_timelapse reachability from a route (T3).
- Type consistency: the laser config name string across the proxy + UI; the `side` 'A'/'B' across _ls_params/params/UI; the timelapse param set across the proxy + form.
- Rig-deferred (explicit): live Multi-Camera dual view, dual-side acquisition, real timelapse execution + galvo/piezo motion.
