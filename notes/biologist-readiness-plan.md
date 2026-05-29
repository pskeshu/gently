# Gently — Biologist-Readiness Plan

> Engineering plan to make Gently more robust, easier for a non-programmer biologist to operate,
> and to evolve it into a multi-user, web-first microscope control system.
> Compiled from a codebase audit (architecture map, complexity audit of all >200-line files in `gently/`,
> robustness + UX review, frontend audit, startup/topology trace, and auth/multi-user ground-truth).

**Author:** engineering analysis · **Date:** 2026-05-28 · **Horizon:** 1 focused week + a multi-sprint convergence arc

---

## 0. Strategic decisions (already made)

These are settled and shape everything below:

1. **Frontend → converge on web-only.** The browser becomes the single surface (a floating agent chat window + the existing rich visuals). The Ink TUI becomes **legacy / maintenance-only** and is retired once the web reaches control parity. → *Do not invest in TUI refactors.*
2. **Processes → keep the two-process split, improve feedback.** The device layer (`start_device_layer.py`) stays a separate process from the agent (`launch_gently.py`) — this isolation is a safety feature, not an accident. Fix the *visibility* of its state, not the topology.
3. **Multi-user → LAN deployment, pluggable auth (no IT dependency to start).** Auth is a thin pluggable layer. Start with **Gently-managed accounts** (or shared/role tokens as an MVP) — needs nothing from institute IT. **Institute SSO (e.g. Janelia/HHMI login via a reverse proxy) is an optional later upgrade** that slots into the same layer if/when IT provides an endpoint. Gently owns the **control arbitration + roles + audit**, regardless of which login backend is used.
4. **Roles → viewers vs operators.** Anyone authenticated can **watch** (today's read-only experience, unchanged). Only **operators** can take control and drive the microscope. **Admins** can force-release and manage roles.
5. **Permission model → an explicit observable-vs-inputable classification.** Every endpoint/WS-message is tagged `observable` (read-only) or `inputable` (control). One registry drives all gating: viewer = observable set; operator-with-lock = observable + inputable. Adding a new action forces a classification; the audit log falls out of the `inputable` tag.
6. **Plan shape → balanced.** Interleave robustness/UX hardening with safe, high-value refactors. Bold-but-safe: refactor where features *won't* break; add tests *before* touching anything that might.

---

## 1. Executive summary

Gently is in **good architectural shape**. The hard parts (async acquisition state machine, hardware-safety code, the LLM loop) are well-factored. The problems that matter are **not "too complex"** — they are a handful of **silent, high-consequence failure modes**, an **opt-in/jargon UX that assumes a programmer**, and the **operational friction** of starting and using a multi-process, dual-frontend system. The web-only + multi-user direction resolves much of the friction *by construction* (e.g. it dissolves the embryo-marking hand-off and removes the Node dependency).

**Top priorities, in order:**

1. **Fix the verified, provable bugs** (status-tool KeyError, non-atomic writes, the silent device-down, the env-var split). Low risk, immediate value.
2. **Wire crash/restart auto-resume** — the single biggest data-loss risk; the code already exists but is never called.
3. **Harden transient-failure handling** (device hiccups, perception/Claude outages) so a brief blip doesn't silently end a run or image a dead embryo.
4. **Make state visible** — live device heartbeat, connection banner, liveness line, acquisition-settings panel, armed-rules display.
5. **Begin the web-only + multi-user arc** — browser agent chat, then the auth + single-driver control lock (the control lock must land *with* browser control, not after).

---

## 2. State of the codebase — legitimate vs. accidental complexity

Most large files are **legitimately large** (broad-but-cohesive domain modules), not tangled. Accidental complexity is concentrated and well-localized.

### Leave alone — legitimate complexity (high feature-break risk)
- `harness/state.py` (979L) — shared mutable `EmbryoState`/`ExperimentState`. Splitting *creates* the duplication the design avoids. **Riskiest refactor target in the repo.**
- `harness/conversation.py` (774L) — core LLM loop (asend-recursion, observed-failure guards).
- `hardware/dispim/devices/*` (stage/optical/scanner/acquisition/camera/piezo) — laser/stage safety constants + MMCore vocab.
- `hardware/dispim/plans/calibration.py` (958L) — irreducible multi-phase calibration state machine.
- `core/imaging.py`, `event_bus.py`, `service.py`; `app/device_state_monitor.py`; `organisms/celegans/stages.py`.

### Top refactor targets — accidental complexity worth fixing

| File | Verdict | Risk | Effort | The fix |
|---|---|---|---|---|
| `app/tools/timelapse_tools.py` (815L) | REFACTORABLE | low | ~4h | Contains the confirmed KeyError bug. `@timelapse_tool` decorator kills the 6-line preamble in 17 tools; stop reaching into `orchestrator._embryo_states`. |
| `app/tools/calibration_tools.py` (1504L) | REFACTORABLE | low | ~2h | Delete ~450 lines of **dead code** (`fast_calibrate_embryo`, `hybrid_focus_selection`, `binary_edge_search`, `_fine_focus_sweep`) — unregistered, uncalled, reference nonexistent agent attrs. |
| `harness/bridge.py` (2215L) | REFACTORABLE | med | ~10h | God-object: 720-line `handle_command` if/elif ladder + case-folding bug (lowercases session/embryo IDs). Dispatch table off `CommandRegistry`. **High value for web convergence** — the browser control surface leans on this. |
| `harness/detection/verifier.py` (1158L) | REFACTORABLE | med | ~6h | `verify()`/`verify_with_context()` + two `_evaluate_consensus*` are superset/subset dupes; 5 `_run_*` + 4 `_parse_*` copy-paste. ~250 lines. **Capture consensus truth-table fixtures first.** |
| `mesh/peer_client.py` (393L) | REFACTORABLE | low | ~4h | 11 near-identical authed methods → one `_authed_json` helper (~270→~80 lines). |
| `hardware/dispim/claude_client.py` (631L) | REFACTORABLE | low | ~3h | 4 vision methods copy-paste → one `_vision_call`. |
| `harness/memory/file_store.py` (2552L) | MIXED | med | ~10h | Mixin split + shared serde. Lower priority than deleting the SQLite twin. |

### The dominant *reduction* opportunity — ~4000 lines of dead duplicate code
The **legacy SQLite store stack** is a complete duplicate of the live file stores (CLAUDE.md says "No SQLite databases"):
- `core/store.py` (1064L) twins `core/file_store.py`
- `harness/memory/{store,_intentions,_plans,_understanding,_ml_pipelines}.py` (~2960L) twin `harness/memory/file_store.py`

Dead in production, pinned only by ~41 tests. Delete **after** migrating tests to the `file_context_store` fixture → ~4000 lines gone, zero runtime change. **Friday work** (gated on test migration).

---

## 3. Verified bugs (confirmed in source, not just inferred)

| # | Bug | Location | Impact |
|---|---|---|---|
| V1 | `get_timelapse_status` reads `next_embryo`/`next_acquisition_in_seconds` that `to_dict()` never emits → **KeyError every call**. Same dead keys in `detection_tools.py`. | `app/tools/timelapse_tools.py:145-146,154` | Biologist's primary "is it working?" tool is broken. |
| V2 | `load_state()` fully implemented, `save_state()` runs every acquisition — but `load_state()` has **zero callers**. | `app/orchestration/timelapse.py:1643` | **No crash/restart auto-resume.** Overnight crash = whole night lost. |
| V3 | `_write_yaml` does `unlink()` then `rename()`; `save_state()` writes with no temp file. | `core/file_store.py:123-125` | **Non-atomic on Windows** — a power blip corrupts the files `/resume` needs. |
| V4 | Launcher reads `GENTLY_STORAGE`; everything else uses `GENTLY_STORAGE_PATH`. | `launch_gently.py:121` | Logs and data silently split to different paths. |
| V5 | Device-layer-down is a `logger.debug` (invisible at default log level). | `launch_gently.py:209-212` | Biologist starts with scope off, gets a normal-looking startup, discovers it mid-conversation. |
| V6 | **XSS / HTML injection** — event key/value (perception prose, paths, agent text) assigned via `innerHTML` with no escaping. | `ui/web/static/js/events.js:69-77, 130-151, 237` | Real injection surface in the events table. `escapeHtml` exists and is used elsewhere. |
| V7 | `/ws/agent` has **no connection guard/lock**; conversation state is a single shared object. | `routes/agent_ws.py:128`, `bridge.py:565`, `agent.py:759` | Latent today (TUI is sole client); **becomes live corruption the moment a browser drives the agent.** Fixed by the control lock (§9). |
| V8 | `bridge.handle_command` does `command.strip().lower()` then branches on it. | `harness/bridge.py:647,696` | Case-sensitive args (session IDs, hostnames, embryo IDs) silently corrupted. |
| V9 | Embryo marking blocks forever; `wait_for_marking(timeout=None)`; TUI never shows the viz URL or signals a browser is needed. | `ui/web/embryo_marker.py:79`, `server.py:481`, `detection_tools.py` | **Worst operational friction** — hangs if no browser is open. Dissolved by web-only convergence. |
| V10 | Marking is global shared state broadcast to all `/ws` clients; any client's `marking_done` clobbers. | `server.py:459-472`, `websocket.py:164-188` | Two browsers marking simultaneously clobber each other. Fixed by driver-only gating (§9). |

---

## 4. Robustness gaps (ranked, for unattended multi-hour sessions)

1. **[CRITICAL] No crash/restart auto-resume** (V2). `_resume_session` (`manager.py:40-117`) restores embryos+conversation but never the orchestrator or runtime fields (stop_condition, cadence_phase, next_due_at, error_count).
2. **[CRITICAL] Device hiccup permanently drops embryos.** `_acquire_embryo` (`timelapse.py:712`) treats network/timeout as terminal; 3 strikes → `complete: errors`. No auto-reconnect in `client.py`.
3. **[CRITICAL] Silent perception/detector outage.** `_run_perception`/`_run_detector` are log-only, no retry, no event. A Claude outage silently freezes stage/hatching detection **while the laser keeps firing.**
4. **[HIGH] Non-atomic writes** (V3).
5. **[HIGH] No abort path for a hung device-layer plan** — one RunEngine, no abort endpoint; one stuck acquisition freezes the wheel each round.
6. **[HIGH] Disk-full silently stops persistence** — `save_state` failures are `logger.debug`.
7. **[HIGH] Orphaned volume TIFFs** — swallowed `register_volume` failure + 300s `cleanup_incoming` race deletes valid volumes.
8. **[HIGH] Unbounded fatal exception kills the whole session** — `_run_loop` top-level except → FAILED, no per-iteration recovery.
9. **[MEDIUM]** Perception task leak / no per-call timeout (`timelapse.py:2706, 2483`).
10. **[MEDIUM]** Startup picker / `wait_for_marking` block forever.
11. **[MEDIUM]** Advisory `session.lock`, no PID check.

---

## 5. Biologist usability gaps (ranked)

1. **[CRITICAL] "Microscope not connected" is silent** (V5). → persistent banner worded as consequence+fix; live heartbeat dot; `/reconnect`.
2. **[CRITICAL] Phototoxicity protection is opt-in, silent, expert-only** — only arms if Claude is passed `monitoring_mode='expression_monitoring'`. → make it **default** for reporter/hatching experiments; agent states plainly what it armed; show armed rules in plain English.
3. **[CRITICAL] No LLM-independent emergency stop** — pause/stop are only LLM tools. → `/stop` `/pause` that call the orchestrator directly (no API round-trip).
4. **[HIGH] Silent auto-complete / auto-pause** — biologist must inspect `completion_reason`. → push plain-language notice; distinguish hardware-error (offer retry) from biological endpoint.
5. **[HIGH] No liveness reassurance.** → "last volume 0:47 ago · next in 1:13" line, yellow/red when stalled.
6. **[HIGH] Marking blocks with no browser cue** (V9).
7. **[HIGH] Cryptic launch hard-stops** (`ANTHROPIC_API_KEY not set`, "TUI not available", Node/npm).
8. **[HIGH] First-run setup landmines** — stale model IDs (`settings.py:55-58`), env-var split, raw `ModuleNotFoundError` on bad organism, README version drift (v0.11.0 vs 0.20.0). → `--doctor` preflight.
9. **[MEDIUM]** Jargon mismatch (campaign/role=test/burst/SAM/photodose). → relabel human-facing strings.
10. **[MEDIUM]** Stop-condition vocabulary mismatch ("pretzel"/"2fold" shown but rejected as targets); casing drift.
11. **[MEDIUM]** Generic error strings (raw `str(e)`/tracebacks reach the biologist).

---

## 6. Frontend audit

### Web UI (`gently/ui/web`) — the future single surface
- **Stack:** vanilla JS, no build step, FastAPI + Jinja2, Three.js for 3D. ~15k JS / 21 `<script>` globals / ~12.5k CSS. Tabs: Embryos, Events, Calibration, Experiment, Plans, Sessions, Devices.
- **Good infra:** `ClientEventBus` decoupling, exponential-backoff WS reconnect, connection manager + presence, clean `devices.js` IIFE, and `experiment-overview.js`'s `el()/elText()/svgEl()` DOM-builders.
- **Refactor targets:** `embryos.js` (4,443L god-object — six concerns: state reconcile, four view renderers, copilot chat, markdown engine; cleanest cuts: `copilot-chat.js` + `markdown.js` low-risk ~½ day). Inline `innerHTML` everywhere (adopt the `el()` helpers). Dead code (`mode-toggle-preview.html`, the `#main-content` shortcut/help). 14 separate `DOMContentLoaded` handlers with global keydown collisions.
- **Bugs:** V6 (XSS), Experiment tab `init()` double-fire, event double-display race (no dedup by `event_id`).

### TUI (`gently/tui`) — LEGACY / maintenance-only
- **Stack:** Ink 5 + React 18 + Zustand, `tsc` build. Well-architected (Static/active message split, hooks, typed protocol). 
- **Decision: do NOT refactor.** Keep it working until the web reaches control parity, then retire the spawn from `launch_gently.py`. Skip the `StatusBar.tsx` split, dead-component cleanup, etc. — they're polish on a retiring component.
- **The bugs that matter are the ones that affect parity/migration:** the stale device-connection dot (B1 — `device_connected` captured once at launch, never updated) and the missing live-state push — but the *fix* belongs in the shared backend (see §8), not the TUI.

---

## 7. Acquisition-settings UX (in the web UI)

**Principle:** the sample is the primary object; acquisition settings are a collapsible "imaging recipe" attached to each sample, plus a global defaults header. A normal scope shows one global block; Gently's edge is *per-sample* settings — lean into that.

**Already in the data model + persisted** (`EmbryoState` state.py:139-148, per-volume `meta.yaml` at `bridge.py:1045`): interval, exposure, num_slices, 488 power, mode, timepoints, cumulative dose; z-geometry from the calibration dict. `FileStore.get_latest_volume_metadata()` (`file_store.py:707`) returns it in one read-only call.

**Surfaced today:** almost nothing. The web **Experiment tab** (`strategy_snapshot.py`) shows cadence/488-power/dose/timepoints; the Monitoring board is pure biology.

**Missing from the model (needs new fields):** channels/wavelengths + 561/405/637 per-line power; objective, binning, camera ROI (pixel size is a hardcoded constant in `imaging.py`).

**Design (Experiment tab):**
```
┌─ Imaging defaults ───────────────────────────────────────────┐
│ Objective 40× · z 100µm/50 slices (2µm) · base 90s            │
│ Channels: [488 ▓ 4%] [561 ▓ 6%]      ← colored wavelength chips│
└───────────────────────────────────────────────────────────────┘
▸ Sample e3   [comma stage · 0.82]  ●imaging          ← biology FIRST
   dose ████████░░ 64% · 142 tp · last 0:47 · next 1:13
   ▾ Imaging recipe (collapsed)                        ← settings SECOND
       interval 90s · exposure 10ms · 50 slices
       z: 50µm ± 25µm (step 2µm) · 488 @ 4%
```
- Wavelength chips colored by line (405 violet · 488 cyan · 561 yellow-green · 637 red) with live power %.
- Biology stays the headline; recipe is a collapsible secondary row.
- **Backend: mostly free** — add `num_slices` + `exposure_ms` to `_serialize_runtime_state` (`timelapse.py:1716`); also fixes the `_DEFAULT_PER_TP_MS=500` hardcode (`strategy_snapshot.py:46`). z-geometry + 488 power already on disk.

---

## 8. Generalization — beyond nuclear-stained *C. elegans*

The architecture is **already substantially organism-pluggable** (recent work removed hardcoded celegans imports from the harness):
- `OrganismProtocol` (`harness/protocols.py`), `load_organism()`/`get_organism()` driven by `config/config.yml`, a documented **build-a-plugin guide with a Drosophila example**.
- The whole "8 stages" assumption is **encapsulated in `gently/organisms/celegans/`**. The orchestrator reads stage names via `get_organism()` — no hardcoded stage strings.
- `SAMPLE_TERM`/`SAMPLE_TERM_PLURAL` already flows into prompts. **"Nuclear stain" is prose, not structure.** A non-nuclear reporter already runs via `dopaminergic_signal.py`.

**Three real gaps:**
1. 🔴 **External `gently_perception` is celegans-only.** `Perceiver()` takes no organism arg (`agent.py:129`); `plan_mode/tools/validation.py:115` imports `CELEGANS` directly. Crosses a repo boundary. *The one item that genuinely gates "any organism via the standard staging path."*
2. 🟡 **Startup wizard lies** — the organism picker (`startup_wizard.py:156`) stores a free-text learning but never calls `load_organism()` / writes config. Correctness bug, low-risk fix.
3. 🟡 **No channel/stain abstraction** — wavelengths exist only at the device layer + prose. **Same deliverable as §7's channel model** — build once, serves both.

**Build the new UI generalization-ready:** sample noun from `SAMPLE_TERM` (web `index.html` currently hardcodes "Embryos"); stage chips from `organism.STAGES`; channels rendered as data, not assumptions.

---

## 9. Startup, topology, and multi-user auth

### Process topology (factual)
Three processes; the agent is the hub:
- **Agent** (`launch_gently.py`) — asyncio loop + Claude agent; hosts the **uvicorn viz server in-process on :8080** (`/ws` browser, `/ws/agent` TUI/future-browser, `/api/mesh/*`); mesh UDP :19547; connects *out* to the device layer over HTTP.
- **Device layer** (`start_device_layer.py`) — separate **manual** launch, HTTP :60610 (MMCore/Bluesky/SAM). The agent connects to it; does not spawn it. **Isolation is a safety feature — keep it.**
- **Node TUI** — auto-spawned subprocess, connects back to `:8080/ws/agent`. (Retiring under web-only.)

Ports (all in `settings.py`, env-overridable): viz **8080** (`viz_host` defaults to `0.0.0.0`), device **60610**, mesh UDP **19547**, transfer 19548.

### Startup feedback fixes (keep 2 processes)
- **Fix V5** (silent device-down) → loud message + persistent banner: *"Microscope offline — start it with `python start_device_layer.py`. I can plan/review but can't image."*
- **`--doctor` preflight**: API key valid, device reachable + expected devices present, port 8080 free, storage writable + free space, model IDs current.
- **Live device heartbeat in the protocol** — periodic `state_update` carrying live `device_connected` (+ last-volume timestamp) from `agent_ws.py`. Lights up the connection dot/banner everywhere. *(Shared fix behind the stale-dot frontend bug.)*
- **`/reconnect`** so a late device-layer start doesn't force a relaunch.
- **Unify `GENTLY_STORAGE` → `GENTLY_STORAGE_PATH`** (V4).
- **Security:** with the reverse proxy in front, bind viz to `127.0.0.1`, lock CORS to the proxy origin, make mesh opt-in.

### Current auth posture
**None.** Plain HTTP on `0.0.0.0:8080`, wide-open CORS (`server.py:172-178`), anonymous self-asserted presence, **no guard on `/ws/agent`** (V7). The mesh has real auth (HMAC tokens + daily rotation + scopes + `Depends()` gating + audit + rate-limit) but it authenticates **machines** (instance_id, TLS cert, PIN ceremony), is localhost-exempt, and opens fully when no PairingManager exists.

### Multi-user design (LAN, pluggable auth)

Two separate problems: **authentication** (who are you) and **arbitration** (who drives the one microscope). Authentication is standard; arbitration is the hard, instrument-specific part.

**Layer 0 — Transport:** TLS required once credentials cross the LAN. Either terminate TLS at a reverse proxy (if one is used) or wire the existing mesh cert into `start_viz_server` (the constructor already accepts `ssl_certfile/keyfile` — `launch_gently.py:252` just never passes them). Bind viz to `127.0.0.1` if behind a proxy; lock CORS (`server.py:172-178`).

**Layer 1 — Authentication (PLUGGABLE — no IT dependency to start).** A thin auth abstraction with interchangeable backends, in rough order of effort/strength:
  - **A. Shared/role tokens (MVP):** an operator secret/link grants control; viewers just open the page (or a view token). Crude, zero external deps, works day one.
  - **B. Gently-managed accounts (recommended start):** a small user list (user → role → password/token) + a login page. No external system, fully under lab control. Fine for a trusted LAN; you own credential storage + TLS.
  - **C. Self-hosted login (Authelia/Authentik, or "Sign in with Google"):** your own user list or Google-Workspace login, run as a reverse proxy. Real identities, no institute IT.
  - **D. Institute SSO (Janelia/HHMI) via reverse proxy:** Gently trusts a verified `X-Forwarded-Email` header. Strongest, real verified identities, IT manages accounts — **but depends on an IT-provided SSO endpoint (OIDC/SAML) + a rig hostname/cert.** Optional later upgrade.
  
  *Decision: build the layer so A/B work now and D can slot in later. Do not block the project on institute IT.* No password storage if/when D is used.

**Layer 2 — Identity replaces anonymity:** swap presence's `Anonymous {id}` (`connection_manager.py`) for the authenticated user. Presence UI stays; identity becomes trustworthy.

**Layer 2.5 — The observable/inputable registry (permission backbone).** Classify every endpoint and WS-message as **`observable`** (read-only) or **`inputable`** (control), in one registry. All gating derives from this single source: viewer = observable set; operator-with-lock = observable + inputable. Adding an action forces a classification; the audit log falls out of the `inputable` tag. Inventory:
  - **OBSERVABLE:** live images/projections/3D volumes/galleries/filmstrips; monitoring board; events/system tables; reasoning/strategy panels + plots; device *status*/property tables/stage map; sessions/campaigns/plans (view) + acquisition-settings recipe (view); presence; the agent transcript (watch the stream).
  - **INPUTABLE:** the agent chat `/ws/agent` (master control); start/stop/pause timelapse, run plans, move stage; embryo marking (place/edit/roles); bottom-camera start/stop; detector config; campaign mutations (share/join/claim/status); emergency `/stop` `/pause`; perception follow-up chat (cost).

**Layer 3 — Control arbitration (the core): single-driver "control lock."**
```
        ┌─ FREE ──────────────────────────────┐
        │   any OPERATOR → "Take control"      │
        ▼                                       │
   HELD by user A  ──A "Release"──────────────► FREE
        │  ▲                                     ▲
        │  └── B "Request control" → A approves ─┘ (handoff)
        │
        └── A disconnects → GRACE (~60s, esp. mid-acquisition)
                              → reclaim if A returns, else FREE
                              → ADMIN can force-release
```
- Many browsers connect to `/ws/agent`; **only the lock-holder's messages drive the agent.** Everyone else watches the same shared conversation live ("🔒 Dr. Chen is driving — Request control").
- **Acquisition-aware:** don't drop the lock on a network blip during an unattended run (grace period); confirm before a handoff interrupts a running acquisition.
- **This is the missing guard** — fixes V7 (conversation corruption) and V10 (marking clobber → driver-only).

**Layer 4 — Roles (viewers vs operators — per the requirement):**
- **viewer** — authenticated, **read-only; today's rich watching experience unchanged** (images, volumes, monitoring, events). The default for most users.
- **operator** — viewer + can take the control lock and drive the scope.
- **admin** — operator + force-release + manage roles.
Gate the mutating set (`/ws/agent` chat, marking, bottom-camera, campaign mutations, perception chat-cost) with operator + lock, using the proven `Depends()` pattern. Read endpoints stay open to any authenticated user.

**Layer 5 — Audit:** reuse the mesh audit log so every hardware-affecting action is attributed to a user.

---

## 10. THE 5-DAY PLAN (next week, one engineer)

Quick wins and verified bugs first; the resume wiring gets its test net before it ships; the big deletion is last and gated on test migration. **Tests are added before any medium/high-risk change. No TUI refactors.**

### MONDAY — Verified bugs & quick wins *(no new risk)*
- Fix **V1** (`get_timelapse_status` KeyError) — align keys with `to_dict()`; audit `detection_tools.py`; add a smoke test.
- Fix **V3** (crash-safe writes) — temp-file → `flush`+`os.fsync` → `os.replace()`; keep one `.bak`; make `_read_yaml` raise on corruption for critical files.
- Fix **V4** (unify storage env var); log `Storage: <path>` at startup.
- Fix **V6** (XSS) — route event key/value through `escapeHtml` (`events.js`).
- Fix **V5** (silent device-down) — loud message at WARNING+.
- Delete dead calibration code (~450 lines, `calibration_tools.py`) — Grep-confirm zero callers.
- Friendly config errors (`organisms/__init__.py:46`, `hardware/__init__.py:46`).
- **Done when:** status tool returns clean in a smoke test; killing mid-write never corrupts YAML; data+logs share a path; events table renders `<` literally; calibration tests pass.

### TUESDAY — Acquisition auto-resume (#1 data-loss risk) + its test net
- **Tests FIRST:** round-trip `save_state()`/`load_state()` (populate orchestrator → reload fresh → assert cadence/stop/error fields).
- Wire **V2** — `orchestrator.resume_acquisition()` from `manager._resume_session`: restore runtime fields, re-point `_embryo_states` at shared `EmbryoState`, restart `_run_loop()` if saved status was RUNNING.
- Confirm-on-resume banner: *"Previous run was acquiring N embryos (last 3h ago). Resume? [Resume] [Review only]"*.
- **Done when:** round-trip test passes; simulated crash + relaunch restarts the loop with correct cadence; "Review only" stays stopped.

### WEDNESDAY — Device & perception failure hardening
- Distinguish transient vs terminal errors in `_acquire_embryo` (~712-878): network/timeout → pause + surfaced event; only genuine bad images count toward 3-strike completion.
- Client auto-reconnect (`client.py`) — exponential-backoff re-poll of `/api/status`.
- Retry + timeout on autonomous Claude calls (`_run_perception` ~2643, `_run_detector` ~2441, perceiver ~2706, detector.run ~2483): `asyncio.wait_for(...)` + bounded retry; emit `PERCEPTION_DEGRADED`; semaphore on concurrent perception.
- Per-iteration loop guard (`_run_loop` ~703-710): continue on per-embryo error; reserve FAILED for unrecoverable.
- **Done when:** simulated disconnect pauses (not drops) embryos and self-heals; simulated 429 retries then emits `PERCEPTION_DEGRADED`; one embryo's exception doesn't kill the others.

### THURSDAY — Visibility & trust (web + shared backend)
- **Live device heartbeat** in the protocol (periodic `state_update` w/ `device_connected` + last-volume ts) — the shared fix behind the stale dot.
- **Web connection banner + `/reconnect`** driven by the heartbeat; liveness line ("last volume 0:47 ago · next 1:13", red when stalled).
- **Acquisition-settings panel** on the web Experiment tab (§7) — add `num_slices`/`exposure_ms` to `_serialize_runtime_state`, render the recipe + wavelength chips. Mostly free.
- **LLM-independent `/stop` `/pause`** through the bridge (no API round-trip).
- **Surface armed adaptive rules** in plain English; warn if a precious sample has none.
- **Done when:** disconnecting the device shows the banner and `/reconnect` works; status shows live time-since-last-volume; the Experiment tab shows per-sample recipe + channels; `/stop` halts a run with the API mid-stream.

### FRIDAY — Safe, high-value reduction *(gated on test migration)*
- Migrate the ~41 SQLite-store tests onto the `file_context_store` fixture.
- Delete the SQLite stack (~4000 lines) once green; grep-confirm no live importer.
- If time: `peer_client.py` `_authed_json` extraction.
- **Done when:** full suite passes with SQLite files deleted; agent runs a session normally.

> **Note on scope:** the web-only convergence and multi-user auth (below) are a **multi-sprint arc**, not part of this week. Next week stabilizes the system; the arc transforms it.

---

## 11. The web-only + multi-user convergence roadmap (post-week)

Incremental — the backend protocol for agent control already exists (`/ws/agent`); the browser becomes a second client.

**Milestone A — Browser agent chat.** Floating chat window in the web UI connecting to `/ws/agent` with streaming (text/thinking/tool calls). Build on the copilot-chat scaffolding in `embryos.js`.

**Milestone B — Auth + control lock (MUST land with/before A in production).** TLS + a pluggable auth backend (start with Gently-managed accounts / tokens — §9 Layer 1) + the observable/inputable registry + the **single-driver control lock** enforced at `/ws/agent`. This is the minimum that makes browser control *safe* (fixes V7). Browser chat without arbitration = the corruption bug in production. (Institute SSO is a later drop-in, not a prerequisite.)

**Milestone C — Interactive flows in the browser.** Choice pickers (plan approval, session resolution), applied-spec cards, token/cost. Port slash commands to **GUI affordances** (a biologist shouldn't type `/timelapse`).

**Milestone D — Roles + audit.** viewer/operator/admin (§9 Layer 4), per-action gating, request-control handoff UX, admin override, attributed audit log.

**Milestone E — Retire the TUI.** Once parity is reached, drop the TUI spawn from `launch_gently.py`; delete the duplicated markdown/campaign/spec renderers. **Bonus: removes the Node/npm precondition entirely.**

**Parity checklist before retiring the TUI:** chat streaming · slash-command equivalents · choice pickers · startup wizard · session resolution/resume · applied-spec display · token/cost · campaign browse.

**Milestone F — Reshape `launch_gently.py` into a service launcher.** Once the TUI is gone, the launcher's job changes from "spawn a terminal app" to "start a server you point a browser at."
- *What it does:* `--doctor` preflight (API key, device reachable + expected devices, port free, storage writable + free space, auth configured) → start agent + viz server → **block on the server, not a TUI subprocess** → print a human banner (URL · device status · storage · "Ctrl-C to stop") → optionally `webbrowser.open` unless `--no-browser` → clean shutdown on Ctrl-C.
- *Delete/migrate:* the Node-on-PATH + `dist/index.js` checks (~155-163, **removes the Node dependency**); `run_ink_picker()` + the `--resume` Ink picker (session picking moves to the browser); `subprocess.Popen(["node", ...])` + `tui_proc.wait()` (~398-410, replaced by awaiting the server); repurpose `set_launch_info(...)` to feed the web connection banner/heartbeat.
- *Browser landing flow (replaces the TUI startup sequence):* a session picker/dashboard landing page (replaces the Ink `--resume` picker); the startup wizard as a web modal (hostable in the floating chat); **first-run auth bootstrap** — create an admin + print a one-time setup link in the terminal banner.
- *Modes preserved:* `--offline`, `--sessions`, `--resume <id>` (now pre-selects what the browser opens into), plus new `--doctor` and `--no-browser`.
- *Mental model shift:* "open it → a terminal app appears" → "open it → a server starts; you and labmates point browsers at it." Cleaner for a LAN multi-user instrument.

### Progress
- **[done] Milestone A (start):** floating agent-chat window in the web UI (`static/js/agent-chat.js`, `static/css/agent-chat.css`, wired into `index.html`) connecting to `/ws/agent` with streaming + choice pickers + applied-spec cards + slash-command routing, XSS-safe.
- **[done] Milestone B (seed):** single-driver control lock in `routes/agent_ws.py` (holder drives; observers get a "Take control" banner; control passes on disconnect). Fixes latent V7. *Not yet gated by auth — that's the next increment.*
- **[done] Milestone E/F (TUI retired + launcher reshaped):** `launch_gently.py` no longer spawns the Node TUI — it starts the agent + viz server, prints a launch banner (URL · device status [fixes V5] · storage · Ctrl-C), auto-opens the browser (`--no-browser` to suppress), and serves until interrupted. Node/dist requirement removed; `--resume`/bare `--resume` resolves to most-recent (interactive picker deferred to the web). TUI source kept in-tree (reversible), `run_ink_picker` retained for reference.
- **[done] Self-managed auth + roles:** `accounts.py` (PBKDF2 users, HMAC session cookies, first-run admin bootstrap), `auth.py` cookie-aware `resolve_role`, `/login` + `/api/auth/*`, `/ws/agent` authenticates and gates control by role. Viewing is **open** (anonymous watch); login elevates to control; admin/operator/viewer. `GENTLY_NO_AUTH=1` disables.
- **[done] Control-action gating pass:** perception-chat POST + `/ws` marking actions gated to control role (`require_control` / session cookie). Device-ingest + campaign-mesh routes deliberately left to their own machine/mesh auth (documented).
- **[done] Uniform session transcript:** user messages + agent stream broadcast to ALL `/ws/agent` clients (observers watch live); display history persisted to `<session>/chat_display.json` and replayed on connect so refreshes/late-joiners see the full conversation. Choice pickers interactive only for the holder.
- **[done] Verified bugs:** V1 `get_timelapse_status` KeyError, V3 crash-safe atomic writes (`os.replace`+fsync), V4 storage env-var unify, V6 events-table XSS escape. Plus the Windows Ctrl-C launcher fix.
- **[remaining] Not yet done (need tests / live verification):** V2 acquisition auto-resume (add round-trip tests first), dead-calibration-code + SQLite-stack deletion (grep + test migration), device/perception transient-failure hardening, friendly organism/hardware config errors, machine-token auth for device-ingest endpoints.

---

## 12. Backlog (valuable, not next week)

**Robustness:** device-layer abort/`RE.stop` endpoint + client calling it on timeout; disk free-space check → `STORAGE_LOW/FULL` event + auto-pause; crash-safe `register_volume` + `cleanup_incoming` quarantine; live-PID lock check; startup-picker/marking timeouts.

**Usability/setup:** phototoxicity protection **default** for reporter/hatching experiments; `--doctor`; startup model-ID preflight; lean core install (move torch/SAM to a `[device]` extra); README version single-sourcing; jargon relabeling; stop-condition vocabulary + perceiver/STOP_CONDITIONS self-check; consequence+next-step error wrapping.

**Generalization:** per-sample **channels/wavelengths/per-line-power model** (+ objective/binning) into `EmbryoState`/volume metadata — the "looks like microscope software" deliverable *and* the stain generalizer; **parameterize `gently_perception`** for non-celegans (cross-repo); wire startup wizard → `load_organism()`; route `validation.py` through `get_organism()`; reconcile the two `DevelopmentalStage` enums.

**Dual-view readiness (design constraint).** All current data is single-view; dual-view is the intended general capability (single-view = `n_views == 1`). Today view B is dropped only at the explicit 4D `(Views,Z,Y,X)` → `vol[0]` path in `generate_jpeg_projection`. **Rule: view count/layout is declared by acquisition in volume metadata — never inferred from pixel shape** (an aspect-ratio "dual-view" guess sliced centered single-view embryos in half; fixed by removing it). To go dual-view later (additive): (1) device layer records `n_views`/layout in the volume sidecar; (2) uid scheme extends `volume_{embryo}_t{NNNN}` → optional `…_vA`/`…_vB` (bare form stays back-compat); (3) `generate_jpeg_projection` emits a three-view per present view; (4) filmstrip/detail panel + perception become view-aware (toggle / side-by-side / feed one or both). Nothing in the single-view path blocks this.

**Refactors (after their tests; medium risk):** `bridge.py` dispatch table + case-fold fix (high value for web control); `verifier.py` consensus dedup (capture truth-table fixtures first; preserve `ensemble_size=50` + conservative on-error defaults); `timelapse.py` rule `to_dict()/from_dict()` then `TimelapseStatePersister` + `RuleEngine`; `file_store.py` mixin split; `device_layer.py` `@json_endpoint`; `claude_client.py` `_vision_call`; consolidate pixel→stage math (duplicated 4×) into `gently.core.coordinates`; web `embryos.js` split (`copilot-chat.js`, `markdown.js`, `embryo-views.js`).

---

## Key file reference
`app/orchestration/timelapse.py` (load_state:1643, _run_loop:703, _acquire_embryo:712, save_state:~1635, _serialize_runtime_state:1685/1716) · `app/tools/timelapse_tools.py:145-146` · `core/file_store.py:111-132, 707` · `harness/session/manager.py:40-117` · `launch_gently.py:121,209-212,252,394,398` · `harness/bridge.py:565,627-1346,647` · `hardware/dispim/client.py` · `ui/web/server.py:172-178,459-502,624-633` · `ui/web/routes/{agent_ws.py:126,websocket.py,chat.py}` · `ui/web/embryo_marker.py:79` · `ui/web/static/js/{embryos.js,events.js:69-151,marking.js}` · `mesh/{pairing.py,routes.py:35-90}` · `organisms/__init__.py`, `organisms/celegans/stages.py` · `harness/protocols.py` · `app/agent.py:129` · `settings.py:32-33,55-58,64`.
