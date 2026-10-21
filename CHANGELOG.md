# Gently Changelog

What changed in each version and what we were thinking at the time.

---

## v0.4.0

Consolidated five overlapping storage systems into `GentlyStore`. Added
`EventBus` for async messaging. Set up the daemon architecture (context,
clock, agent core, capabilities).

We switched from RPyC to HTTP for the device layer — easier to debug and
process-isolated, so a crashed agent can't take down hardware. The event
bus became the way components talk to each other; publish/subscribe
instead of direct calls.

The embryo became the basic unit of the system, not the image. Each one
carries imagery, calibration state, perception traces, and detector
configs. Safety was layered: process isolation, device limits, templated
actions, automatic cleanup.

---

## v0.5.0

Replaced Rich CLI output with an Ink (React + Node.js) TUI connected via
WebSocket. The copilot stopped owning stdout.

- Persistent layout: header, scrolling chat, input bar, status bar.
- WebSocket transport so the TUI doesn't poll.
- Choice pickers for structured questions — the LLM proposes options,
  the human picks.
- 8 themes, switched client-side.
- Split monolithic `server.py` (2,159 lines) into 13 route modules.

Perception moved here too — VLM-based stage classification, three-view
projections (XY, XZ, YZ), trace persistence for timelapse.

Separating display from logic made the boundaries cleaner.
`CopilotBridge` handles async mechanics, the TUI handles presentation.

+7,923 / -7,151 lines, 66 files.

---

## v0.6.0

Added plan mode. Run mode is for real-time control ("what should we image
now"), plan mode is for experimental design ("how should we structure this
study"). They use different prompts, different tools, different thinking
budgets.

- Campaign/PlanItem/ImagingSpec/BenchSpec data model with dependency
  graphs.
- `ContextStore` for the agent's understanding (campaigns, learnings),
  separate from `GentlyStore` (raw data, images). Different lifecycles.
- Organism and hardware modules (`gently/organisms/celegans/`,
  `gently/hardware/dispim/`) to make the system backend-agnostic.
- Startup wizard for onboarding.
- Early research tools: `search_literature`, `search_strains`,
  `check_hardware_capability`.
- Extended thinking for complex operations.

We wanted the copilot to work at the same abstraction level as the
scientist — campaigns and research questions, not pixel coordinates.

+13,000 / -1,512 lines, 76 files.

---

## v0.6.1

Cleanup. Removed dead code, relocated configs, flattened backend
directory, refreshed docs. Removed DiSPIM-specific scaffolding.

+81 / -13,692 lines, 112 files. Mostly deletion.

---

## v0.7.0

Plan mode was a prototype in v0.6.0. This version made it actually
usable.

Research tools got real API integrations:
- PubMed via NCBI E-utilities (search + abstracts)
- Paper reading via PMC full text, Unpaywall, local PDFs, URL fetch
- WormBase and CGC for strain search
- NCBI Gene for gene information

Plan infrastructure:
- Versioning with JSON snapshots (snapshot/list/restore)
- Validation — hardware limits, stage order, duration estimates,
  dependency cycle detection
- Execution bridge linking plan items to running sessions
- Templates for reusable protocols
- Markdown export
- Reorganization tools (move, delete, reorder, phase management)
- References — plan items carry citations from research tools

Extended thinking: plan mode always uses it (30K token budget), run mode
uses 10K triggered by complexity.

TUI: human-readable tool labels, session resume, campaign resolution by
shorthand/name.

+8,046 / -644 lines, 28 files.

---

## v0.8.0

Added LAN peer-to-peer coordination. Instances find each other via UDP
broadcast and can share campaigns.

- UDP discovery on port 19547, zero config.
- HTTP peer client for remote campaign operations.
- 8 new mesh API endpoints (share, join, claim, export, etc).
- Each node advertises capabilities (GPU, SAM, storage).
- Campaign sharing: origin shares, peers join and claim items. Double-claim
  returns 409, re-claim is idempotent.
- `/peers` command in TUI. Status bar shows peer count.
- 27 tests for coordination flows.

+1,778 lines, 22 files.

---

## v0.8.1

Status polling was every 30 seconds, so mode changes (run to plan) took
a while to show up on peers. Added a nudge pattern:

1. Node changes mode -> `EventBus` emits `STATUS_CHANGED`
2. `MeshService` hears it -> UDP nudge broadcast
3. Peers receive nudge -> immediate HTTP refetch
4. Updates in ~1 second

The 30s poll stays as fallback. The nudge is just "come look at me" — no
payload, no ordering, no delivery guarantee. If a peer misses it, the
poll catches up.

5 files, +53 lines.

---

## v0.8.2 – v0.8.4

Mesh security. The v0.8.0 mesh had no authentication — any node on the
LAN could query any other. These three versions added layered security:

**Phase 1 — Pairing (v0.8.2)**
Bluetooth-style pairing flow. One node runs `/pair <hostname>`, the other
sees a 6-digit PIN and runs `/pair accept`. Both sides must confirm the
same code before trust is established. Trusted peers are persisted in
`mesh_trusted_peers.json`. `/pair list`, `/pair unpair` for management.

**Phase 2 — TLS + Signed UDP (v0.8.3)**
- Self-signed TLS certificates generated per instance. Paired peers
  exchange certificate fingerprints during pairing.
- All HTTP calls between paired peers use HTTPS with cert pinning
  (`aiohttp.Fingerprint`). Fingerprint mismatch → connection refused.
- UDP heartbeats signed with HMAC-SHA256. Replay protection via
  monotonic sequence numbers. Unsigned packets from unknown peers still
  accepted for discovery (unpaired peers appear as "untrusted").
- Rate limiting on pairing endpoint (5 attempts per IP per 5 minutes).

**Phase 3 — Audit + Token Rotation (v0.8.4)**
- `MeshAuditLog` writes structured JSON-lines to `mesh_audit.jsonl`.
  Events: auth success/failure, cert pinning ok/fail, signature invalid,
  replay rejected, pairing lifecycle, rate limiting. Auto-rotates at 10k
  lines.
- Daily token rotation: `HMAC-SHA256(base_token, epoch_day)`. Both
  peers derive the same daily token independently — zero network
  coordination. Accepts current + previous day for midnight boundaries.
- Security events published to EventBus (`MESH_AUTH_FAILURE`,
  `MESH_CERT_PIN_FAILURE`) for TUI notifications.

+1,920 lines across 21 files.

---

## v0.8.5

Capability-scoped permissions and TUI status bar integration.

**Phase 4 — Scoped Permissions**
Three scopes: `status` (read mesh info), `campaigns` (join/claim/report),
`campaigns:admin` (share/unshare). New pairings get all three by default.
`/pair scopes <hostname> <scope_list>` to restrict.

Auth dependency factory pattern — `_make_auth_dep("campaigns")` creates
per-endpoint FastAPI dependencies. Scope denials logged to audit trail
and published as `MESH_SCOPE_DENIED` events.

**TUI Status Bar Integration**
Merged the navigable status bar browser from main with mesh security
notifications:
- Fixed notification protocol (`text` → `title`/`body`) so mesh events
  display correctly in the status bar.
- Peer discovery: "Peer joined: hostname" (trusted) or "New peer:
  hostname — Use /pair to connect" (untrusted).
- Peer loss: "Peer offline: hostname" warning.
- Pairing: PIN display in notification body, success confirmation.
- Security alerts: auth failures, certificate mismatches (MITM warning),
  scope denials pushed as warning/error notifications.
- Trust indicators in peer browser: green lock (trusted+TLS), yellow
  shield (trusted, no TLS), red ? (unpaired).

---

## v0.8.6

TUI: extracted campaign browser from StatusBar into a dedicated
`CampaignBrowser` component. StatusBar keeps a read-only summary,
`/campaign` opens the full interactive tree with actions (share,
pause/resume), subcampaign expansion, and keyboard navigation.

---

## v0.11.0

Library restructure — separated the agentic harness from the application.

**Four-Layer Architecture**
Gently is now organized into four layers with strict downward-only dependencies:
1. **Foundation** (`gently/core/`) — event bus, data stores, imaging, coordinates
2. **Harness** (`gently/harness/`) — reusable agent framework (tools, conversation,
   perception, memory, prompts, detection, session management)
3. **Domain Plugins** (`gently/organisms/`, `gently/hardware/`) — swappable organism
   and hardware knowledge
4. **Application** (`gently/app/`) — the microscopy agent product, domain tools,
   orchestration

**Key Moves**
- `gently/agent/` split: framework → `harness/`, app code → `app/`
- `gently/context/` → `harness/memory/` (agent's persistent mind lives with the harness)
- Root-level diSPIM files (`config.py`, `device_layer.py`, `plans.py`, `devices/`,
  etc.) → `hardware/dispim/` (they're plugin code, not framework)
- `gently/visualization/` → `gently/ui/web/`
- `gently/imaging.py`, `coordinates.py`, `store.py` → `gently/core/`

**Plugin Contracts**
- Added `harness/protocols.py` with `OrganismProtocol` and `HardwareProtocol`
  defining what plugins must export.
- Removed hardcoded `from gently.organisms.celegans...` imports from harness layer.
  All organism/hardware access now goes through `get_organism()`/`get_hardware()`.

**Naming**
- `copilot` → `agent` throughout (class names, files, routes)
- Backward-compat shims at old locations (`gently.agent`, `gently.context`)

317 tests pass.

---

## v0.10.0

Distributed ML, data reasoning, and quality-of-life fixes.

**Distributed ML Mesh**
- Verse map for mesh-wide data coordination — nodes advertise what data
  they have, so the mesh knows where to route ML jobs.
- Data reasoning engine: coverage assessment, quality scoring, gap
  planning. The agent can evaluate whether there's enough data to train
  and what's missing.
- ML engine: architecture registry, data loader, trainer, evaluation
  pipeline. Supports federated averaging across mesh peers.
- Bulk transfer protocol for moving volumes between nodes (chunked,
  resumable, tracked).

**Web UI Embryo Marking**
- Replaced napari-based embryo marking with a browser-based UI served
  from the viz server. No more native GUI dependency.

**Launch Fixes**
- Fixed TLS mismatch: viz server now uses the self-signed cert, so
  `wss://` connections from the TUI work correctly. Eliminates the
  "Invalid HTTP request" errors from uvicorn.
- Default log level changed from INFO to WARNING — quiet terminal.
- Added `-v`/`--verbose` (INFO) and `--debug` (DEBUG) CLI flags.
- Uvicorn warnings suppressed when not in verbose mode.

**Packaging**
- Moved device, ML, and testing deps from optional to core requirements.
- Added `requirements-cuda.txt` for GPU setups.

+8,500 lines, 68 files.

---

## v0.9.2

More dead code removal and a layer violation fix (P8).

- Deleted 4 orphaned files (~1,175 lines): `agent/logger.py`,
  `agent/visualization.py`, `dataset/trace_persister.py`,
  `analysis/algorithms.py` — all defined classes/functions that nothing
  imported.
- Fixed `visualization/ → agent/` layer violation: projection utilities
  (`projection_three_view`, `compute_crop_bounds`, etc.) lived in
  `agent/perception/projection.py` but were needed by 4 files in
  `visualization/`. Moved them into `gently/imaging.py` where they
  belong. Updated 9 import sites, deleted the old file.
- Deduplicated `dataset/explorer_server.py`: replaced 6 copy-pasted
  projection functions with imports from `gently.imaging`.
- Cleaned dead imports (`center_of_mass`, `OrderedDict`), fixed
  deprecated `scipy.ndimage.measurements` path.
- Fixed `__all__` in `__init__.py`: calibration plan names now
  conditionally added to match their conditional import.

---

## v0.9.0

Internal restructuring. No new user-facing features — this is about
making the codebase easier to work in.

Five refactoring passes (P1–P5):

**P1 — Module decomposition**
- Split `copilot.py` (1,600 lines) into 3 delegate classes:
  `ConversationManager`, `ToolDispatcher`, `ExperimentDelegate`.
- Split `hardware_tools.py` into 5 domain-specific tool modules.
- Split `context/store.py` into mixin modules by domain.
- Consolidated duplicated image encoding into `gently/imaging.py`.

**P2 — Logging and configuration**
- Replaced ~530 `print()` calls with structured logging.
- Centralized hardcoded config into `gently/settings.py` with env
  overrides.

**P3 — Service architecture**
- `VisualizationServer` and `DeviceLayerServer` now extend the `Service`
  base class — lifecycle state machine, health checks, double-start
  guards for free.
- Migrated `ServiceClient` from `httpx` to `aiohttp`, matching the rest
  of the codebase.

**P4 — Error handling and type safety**
- `gently/exceptions.py`: 16 domain exception classes under `GentlyError`
  (hardware, calibration, perception, storage, network, copilot).
- Converted ~25 bare `except Exception` handlers to specific types.
- Consolidated duplicate prompt strings in `claude_client.py`.
- Deleted orphaned `plans_qserver.py` (moved utility plans to `plans.py`).
- `gently/store_types.py`: 8 TypedDict definitions for `GentlyStore`
  return values.

**P5 — Packaging and documentation**
- Added `pyproject.toml` with setuptools packaging, optional dependency
  groups, and `gently` console script entry point.
- Updated `.gitignore` for mesh artifacts, LaTeX files, electron/.
- Synced version strings across 4 locations.
- Generated reference docs: `docs/COMMANDS.md` (24 slash commands),
  `docs/TOOLS.md` (68 run-mode + 27 plan-mode tools),
  `scripts/README.md`, `examples/README.md`.

The goal was to get the codebase to a state where you can grep for
something and find it in one place. Exceptions have types, services have
a lifecycle, config has a home, and the docs match the code.

---

## v0.9.1

Continued internal cleanup (P6–P7). Still no user-facing changes.

**P6 — Architectural fixes**
- Fixed layer violation: moved `device_factory.py` and `sam_detection.py`
  out of `agent/` (application layer) to the root package (infrastructure
  layer), where `device_layer.py` can import them without reaching upward.
- Split `devices.py` (1,813 lines, 12 Ophyd classes) into
  `gently/devices/` package — one module per device domain (stage, camera,
  piezo, scanner, optical, acquisition). Re-exports preserve existing
  import paths.
- Moved hardcoded mesh constants (port 8080, timeouts, stale/dead
  thresholds) into `settings.py` with `GENTLY_*` env var overrides.
- Removed dead `HTTPService` base class (never subclassed).

**P7 — Dead code removal**
- Deleted `gently/visualization.py` (245 lines) — shadowed by the
  `gently/visualization/` package, completely unreachable since the
  package was created.
- Deleted `gently/capabilities/` module (6 files, ~1,300 lines) —
  abandoned abstraction layer with zero external consumers.
- Removed deprecated `pixel_to_stage_offset()` from `coordinates.py` —
  all callers had been migrated to the replacement functions.
- Fixed broken visualization imports in `__init__.py` that silently failed
  on every import (requested symbols that neither the dead file nor the
  package exported).

Net: ~3,500 lines removed across P6–P7.

---

## v0.22.0

File-based storage, a redesigned web UI, new hardware, and the tooling to
keep it all honest.

**File-based storage (Gently3)**
Retired the SQLite databases. All state now lives as human-browsable files
under `D:\Gently3\` — sessions, embryos, volumes, projections, traces,
campaigns, learnings, agent memory, all YAML/JSONL/TIFF.
- `FileStore` replaces `GentlyStore`; `FileContextStore` replaces the
  `agent_mind.db` `ContextStore`. Drop-in API replacements.
- A root `gently.yaml` manifest documents the layout for humans and agents.
- YAML parses are cached in `FileContextStore` — fixes slow Plans/campaign
  loading.

**Web UI redesign**
- Agent chat became a docked, sliding side panel (overlay + pin-to-dock)
  instead of owning the screen.
- Added a Home landing tab; the chat no longer auto-runs the startup wizard.
- Login is non-blocking — a "Continue in view-only" escape hatch.
- Recent images aggregate across previous sessions.

**Hardware**
- Integrated the ACUITYnano temperature controller (config, web control,
  SDKs) with a live HiveMQ cloud SIM for hardware-free testing.
- Added the SPIM-head F-drive device, hard limits, and focus/align plans.
- Room-light toggle and a device-layer terminal UI.

**Agent + perception**
- Integrated the agent with perception: pull tool, prompt context, event
  bridge, wake-router.
- Live acquisition control with observable, permissioned autonomy and a
  refreshed prompt.
- Retired napari from the agent; added web-chat autocomplete and pruned
  dead tools.

**Tooling and environment**
- Added ruff lint/format tooling and fixed all violations.
- Adopted incremental mypy typing — config, CI, pre-commit wiring, and a
  documented policy in `CONTRIBUTING.md`; pinned mypy to 2.1.0.
- Switched environment setup to uv with an offline/UI-only launch path;
  pinned pymmcore to device-interface 70.
- Relicensed and updated the author list.

---

## v1.0.0.dev1

The first build cut for someone else to use. The first external user is the
reader; everything here exists so their feedback lands on something specific.

**The Atrium**

The canvas surface is now a real view in the web UI behind `?atrium=1`, not a
prototype in a docs folder. Windows on a pannable bench, gauges in a
screen-fixed courtyard, and one pressure primitive driving an information
release ladder across all three scales. The tabbed UI stays the default and is
untouched with the flag off.

It also arrived with a list of things that are not ours to settle —
`docs/atrium/OPEN-DECISIONS.md` is eight judgement calls found by an
adversarial critique, written down rather than guessed at. The startup warning
that EVENTS can never reach its `open` rung is one of them, left deliberately
audible.

**You can tell which build you are running**

The version was written in two places and reached the browser through exactly
one path — the agent-chat socket — so without an API key there was no version
on screen anywhere. Now `gently/_version.py` is the only literal, pyproject
derives it, and `build_id()` reports `1.0.0.dev1+g92816ea`, with `-dirty` when
the tree has uncommitted changes. It shows on the launch gate, in Settings and
at `/openapi.json`.

That suffix is the point. A bare version names every commit between two tags,
which is the whole window a reviewer works in; the commit names one tree, and
`-dirty` separates a real bug from a half-finished local edit.

**Four bugs from the walkthrough**

All four came out of frame-by-frame review of the 2026-08-07 recording rather
than from anyone in the room, and each one was a case of two things that should
have agreed and had nothing making them agree.

- Every failure message rendered in success green. `Delete failed (405)` looked
  exactly like `Volume acquired`.
- `operate.js` and `devices.js` held the same mutable roster array, so a push
  in one became a phantom row in the other — with no id, and a delete that
  405'd.
- Selecting a different embryo on the SPIM head changed the caption and left
  the previous embryo's pixels on screen. Caption and frame are now gated on
  the stage actually being there.
- The click target for a registered embryo was smaller than the ring it was
  drawn as, so aiming at an embryo and missing by a few pixels created a marker
  that Register turned into a duplicate.

**Known**

Three milestone issues need the microscope and are not done: camera ROI
readout, two-point calibration, and the SPIM laser/exposure controls. The
`pytest` suite has 17 pre-existing failures that no CI job has ever run (#143).

---

## v1.0.0rc1

The first build in which a biologist can take an experiment from a dish to a
running multi-embryo timelapse without leaving Devices › Operate, and get it
back after a restart. Seventy-nine pull requests since `v1.0.0.dev1`, nearly all
of them started as a sentence somebody said at the microscope.

A release candidate: everything below is merged and has been run on the
microscope. The list under **Known** is what is still open.

**The acquisition plan**

Operate's last pane was the one nobody had touched. It is now a configurator in
the shape of a classical multi-dimensional acquisition, minus what the
instrument already knows: positions are the roster and z is each embryo's
calibration, so what is asked is what is left. Cadence, two channels, and how
it ends (#194, #195).

- **Two channels.** SPIM volumes of every embryo every round, and a DIC
  overview: one bottom-camera frame of the whole field, on its own clock,
  taken from the centroid of the embryos or from a position you pin.
- **The overview says which light it is taken under** (#212): the room light,
  the LED, or the light as it is. The light goes on for the frame and off
  again before the volumes; a light that was already on is left on.
- **Endings per embryo.** The run has a default ending; any embryo can have
  its own. "Embryo 2 at hatching, the rest after twelve timepoints."
- **A duration ending is measured on the embryo's clock.** "After six hours"
  means six hours after the run began, whatever the software was doing in
  between. A pause or a restart does not stop an embryo developing, so it
  does not stop the clock.
- **The plan is said back as one sentence** beside Start, and that sentence is
  exactly what goes on the wire.
- **The run, embryo by embryo** (#196). Each embryo has a row with its count,
  its next time, its ending, and its own Stop. The Embryos tab shows the DIC
  frames as a strip you can open and step through (#200), and in the film
  they are the film's first row, on the same timeline as the embryos (#210).
- **Templates** (#197). A plan saves under a name and runs later as the
  sentence it was saved as.
- **A subset of embryos** can be targeted for a run (#161). Role decides what
  an embryo is for; selection decides which ones this run images. With
  several selected, each highlighted row says which it is, "selected" or
  "target", instead of being a lighter or a darker blue (#220).
- **Start is not offered while a run is going** (#217). The button went on
  saying "Run tactic" over a tactic that was running.
- **A run that was stopped says so** (#228). A run started from a saved
  tactic left that tactic "active" after Stop, with a Stop button that did
  nothing. However a run ends, its tactic is done, and the run records how
  it ended: a run stopped on purpose is no longer read as "interrupted" at
  the next start. A paused run could not be stopped at all.

Three things that were quietly wrong are fixed on the way. The laser preset
chosen on the pane was collected and never applied. Slices and exposure were
validated and dropped, so every timepoint ran at the defaults. And Start was
four different verbs behind one label (#156). A run on an uncalibrated embryo
is now refused rather than warned about (#154).

**A session survives a restart**

The orchestrator wrote a checkpoint every round and nothing ever read it
back. A resumed run took its counts from the conversation snapshot, which on
the rig said t1 while the checkpoint said t15, so a restarted run would have
numbered from t2 over the volumes on disk.

- The checkpoint is applied on resume, and the plan a run was started with is
  kept in the session as `acquisition.yaml` (#203).
- A restored run can be carried on: **Resume run** images every embryo still
  going, continues the numbering, and keeps the DIC clock (#204).
- The pane comes back on what the session was running: the plan, the selected
  embryos, and the mode (#208).
- The snapshot now follows the run and is written at shutdown (#206).

**Removing an embryo deletes nothing** (#219)

The × beside an embryo is for a false positive, and it sits one row from the
embryo that has been imaged all night. It asked nothing, and it deleted the
embryo's folder from disk: volumes, projections, traces, calibration.

- A removed embryo's folder is moved, whole, to the session's `removed`
  folder. Undo is on the toast, and after that the roster lists the removed
  embryos, each with Restore. It still does after a restart.
- An embryo that holds timepoints or a calibration is asked about first.
- An embryo the run is imaging is refused, with where to stop it.

**Calibration**

Calibration has its own pane (#158) and shows its sweep where the operator
started it (#145). It looks once before spending sixty exposures: on an empty
field it used to succeed, fitting a slope and a scan cuboid to noise (#176).
A running calibration can be aborted, which cancels the routine and halts the
axes including the piezo (#201).

**What a calibration looked at is kept** (#214). Its conclusion was stored, a
slope and an offset; its evidence was not. Every exposure, focus curve and
montage went to the browser's memory and was gone at the next restart. Each
run now leaves a folder under its embryo, whether it calibrated, was refused,
failed or was aborted, and the pane shows the latest run's plots.

The plots can be read (#220). They were drawn 600 px wide with 9 pt type and
shown 126 px wide, where that type is three pixels tall.

**The SPIM head, and stopping things**

- **HALT** stops every positioner, and is always enabled (#168). A HALT the
  controller refused now says which axis did not stop instead of reporting
  success (#173).
- **Raise head** sits beside it and takes the head to the F-drive's own top
  limit, the load height (#207).
- The light-sheet exposure the operator sets now sticks (#167).
- Where the head looks is measured instead of assumed, with every past centre
  kept on disk and restorable (#177, #178).
- A live view is off once you leave its surface and stays off. It used to
  restart on return, which started the SPIM camera streaming under a running
  timelapse (#209).

**The stage's fences**

The map's region is the XY safety envelope, and it was four constants pushed
into the controller at boot. It is now walked from the map, two corners at a
time, with the box following the stage (#170, #191, #192), and every edit is
kept (#186).

There are two fences and only one of them binds other people (#185). The
controller's own soft limits are enforced against every client, Micro-Manager
included; Gently's envelope binds only Gently. The controller's fence can be
taken down and stays down (#183), and "off" means the controller's own limits
rather than a conservative box of ours (#193). The green region on the map is
the one you edit, not whatever the controller happens to hold (#189).

**Settings**

Gently has about a hundred configurable things, kept in seven places. The
Settings page showed a third of them, and eight of its nineteen view settings
were read by nothing (#213).

- **A registry.** Each setting is declared once: what it is about, how far it
  reaches, when a change takes effect, where it is kept, and who reads it. A
  test fails if a declared reader does not read it.
- **Seven categories**, by subject, and a badge on every setting for its
  reach: this browser, this rig, needs restart.
- **A tab of the app**, not a page beside it. A view setting changes the view
  at once.
- **Recording** has its switch. It was on by default with five knobs and none
  of them in the UI. The section says what is kept, typed text included.
- **A history.** Every change to a setting is appended to
  `config/settings_history.jsonl` under the data folder and never pruned. A
  secret is never written.

**Light, and what a panel is**

`docs/architecture/PANELS.md` sets the policy: panels are standard surfaces
mounted in many places and reading one shared state, and they read back
rather than rendering a command as a fact (#148, #152). The Light panel is
the first. Illumination mode is its root, because LED and laser are the two
ways this instrument lights a sample and the workflow alternates between them
(#164). The laser's settings appear once a line is routed (#163). Device
properties can be read out, and the display range is a histogram panel (#153).

**Detection**

Bottom-camera detection produced seventeen false positives for sixteen
embryos, and the cause was not SAM but the candidate finder in front of it.
That is now a flat-field and blob finder (#165), and Claude classifies each
candidate crop, which is what removes the bright out-of-focus edges of
bubbles (#166). The boot banner says when SAM cannot run instead of promising
it (#147), and a failed detect says which failure it was (#198).

**Finding the files** (#221)

Everything Gently keeps is a file, and the way to one was to know the layout
and walk to it.

- **Show file** and **Open in Fiji** in both image viewers, for the image on
  screen. A timepoint is its volume: the viewer shows a projection, and what
  opens in Fiji is the stack.
- **Folder** on every session, on an embryo, on a calibration run, and for
  the logs, the recordings, the config and the settings history.
- Fiji is found where it is usually unpacked, or where Settings says it is.
  Micro-Manager's ImageJ is never used: starting it starts Micro-Manager,
  which takes the microscope's ports.
- The window opens on the computer Gently runs on. A browser on another
  computer is handed the path instead.

**The chrome**

- **The gate leads into the workspace** (#215). There was a page between
  them, and the only thing anyone pressed on it was Skip.
- **The gate offers the last sessions to carry on from** (#222). A new
  session is what is chosen, every time; carrying on is decided.
- **Diagnostics, a third switch on the gate** (#228). The recording of a run
  that needed working out stopped at 120 MB, partway through it. With
  Diagnostics on, everything on screen is recorded under larger limits and
  the log files are in detail. The workspace says when it is on.
- **Nobody is an anonymous eagle** (#228). A browser that has given no name
  is called by its account, or by where it is: "At the microscope" or
  "Guest".
- The rig moved into the header: device-layer state, start and stop, the log,
  water and room light are reachable from every tab (#174).
- A boot is a notification, not a bar to dismiss (#180).
- The build id copies in one click and says how old the build is (#172, #188).
- A button beside the session id opens the session's folder in the file
  manager (#205).
- **A projection shows the left channel** (#224). The camera's frame carries
  two channels side by side, and a projection of all of it was mostly empty
  field. Which channel is shown is a setting of the rig; the volume on disk
  is the whole frame. Projections already drawn keep what they were drawn
  with.
- Home's recent images update as volumes land, and open (#202). They are
  under their sessions, embryo by embryo (#223).
- A note's embryos are named with their session, `6f090787/embryo_1`. A note
  was drawn with twenty-eight one-letter tags: its embryos had been given as
  one string, and taken apart letter by letter (#218).
- A failure toast says why, not just which number (#146).
- `hidden` actually hides (#181).

**Under the floor**

- **Before the tag, the whole app was walked** (#225): every tab, view and
  viewer, in both themes, with every refused request written down. Two were
  refused on every rig, always. The Light panel read the LED's state from an
  address with no route, so its mode was "unknown" whatever the hardware was
  doing. And Restore in the Map's region history had a button and a handler
  with nothing between them. A test now checks that whatever a page asks
  for, something answers.

- **The DIC overview was never saved on the real microscope** (#210). A
  night's run logged 24 frames acquired and none were on disk: the capture
  reported no file path, and the run skipped filing without a word. The
  tests had passed because their camera was kinder than the real one. Frames
  are now filed, and a frame that cannot be is a warning in the log.
- **CI runs the tests** (#211). Nothing did before, so a pull request could
  merge with its own tests failing, and sixteen tests had failed unseen for
  months. Thirteen of those were device-safety tests checking a mock. The
  suite is green, and its first run on CI found a real bug: on Python 3.10 a
  closed browser tab was reported as an operator aborting a calibration.
- The test suite cannot reach the microscope's own data. It had written a
  fixture's region into the rig's config eighteen times (#190).
- `mypy` is clean in both runs. The last error was real: the GPU probe read
  an attribute torch does not have, so the node never listed its GPU (#206).
- One roster component and one marking surface where there were two of each
  (#157, #162).

**Known**

Perception, the detectors and calibration each still take their own part of
the camera's frame, by three different rules. Only the projection follows
the new setting.

Two issues stay on the 1.0.0 milestone: an operator-selectable camera
readout region on the SPIM view (#125), and the laser toggle reading back
the hardware's state rather than the request it sent (#106).

A plan derived for a session that predates `acquisition.yaml` shows its DIC
position as pinned rather than "centroid", because the checkpoint stores the
resolved position.

CI drives no browser. What the UI looks like and does is verified by hand,
on the microscope.

---

## Unreleased — the project layer

A root campaign is a project, and a project now holds what a biologist asks a
project to hold: its sessions, its plan, the ledger of what is next, predicted,
measured and open, the strains and references it rests on, and a view of how
every night in it went. Nothing moved on disk; the project is a filter over
the records that were already there, keyed by one new field.

**Sessions belong to a project**

- `session.yaml.project_id` is the one truth of membership. Every caller that
  used to write the session-intent record goes through one door,
  `gently/app/filing.py`: the agent when it runs a plan item, the Operate
  start through the resolution tools, and the Sessions tab's "File under…".
- A session has one home and can serve plan items in other projects; the
  Sessions tab says "also serves …" for those. Sessions filed before the
  field existed get their home from their plan links at startup, once.
- `GET /api/sessions?project_id=` returns the project's sessions with what the
  project view shows: errors and warnings from the event capture, the agent's
  turns and corrections from the interaction log. `?unfiled=1` lists the rest.

**The ledger**

- `GET /api/campaigns/{id}/ledger`: next (unblocked items), proposed, blocked,
  predicted (expectations), measured (outcomes and findings), open and
  answered questions. Expectations and questions carry the session they were
  made in, so a project owns them through its sessions.
- A plan item can be `proposed`: the agent's suggestion, never runnable until
  a person accepts it. Outside plan mode the agent's `create_plan_item` makes
  proposed items; in plan mode, where the plan is written together, planned.
- A finding is a notebook note of kind `finding` with a `projects` facet. The
  agent's `propose_finding` files one as proposed; a person confirms,
  supersedes or dismisses it on the Projects tab. `record_expectation` and
  `raise_question` give the other two columns a writer; a person adds the same
  things through `/api/context` and `/api/notebook`.
- When the live session changes, one deterministic "Session summary" note is
  written for the night that ends: embryos, timepoints, how the run ended,
  errors and warnings, agent turns. Nothing for an empty session.

**An intention becomes a spec**

- A plan item written as a sentence is read at once for what it states
  (embryos, cadence, channels, stop, temperature, strain, exposure, and the
  "to check…" clause as the success criterion), and Gently drafts the rest in
  the background: the kind of acquisition, settings inside the instrument's
  limits, a tactic outline the run can execute, a title, a rationale and its
  assumptions. The card shows the draft as a proposal with Accept, Edit and
  Ask again; nothing runs on it until a person accepts. Every field carries
  where it came from, and a person's field is never overwritten by a draft.

**The spec is the pane's shape**

- A plan item's acquisition is now the structure Devices › Operate runs,
  the same one a saved tactic carries and a session keeps: cadence, volumes
  or brightfield-only, slices and exposure, laser preset and per-line powers,
  the brightfield overview, the ending. The June-era flat fields are read into
  it, so nothing already written is lost. The card says it in the pane's own
  sentence.
- **Set up on Devices** on an item opens Devices › Operate with the
  Acquisition pane filled from it; Start files the run under the item and the
  plan records what actually ran. Gently's drafts arrive in this shape.

**References, through one search**

- A project carries `strains[]` and `references[]`, each `{source, key,
  title, url, subtitle}`. `GET /api/search/{source}?q=` answers one source at
  a time so the add popover can query five in parallel: the lab's own names
  (instant, offline), genes from the Alliance of Genome Resources (WormBase
  ids), CGC strains, papers from Europe PMC, WormBase life-stage and anatomy
  terms from the EBI Ontology Lookup Service. The agent's `search_references`
  and `add_reference` tools use the same functions.
- WormBase's own REST API answers scripts with a browser challenge, so the two
  WormBase lookups in the research tools, which had been failing silently,
  are gone; WormBase ids come through the Alliance and OLS, and every chip
  links to wormbase.org for a person. CGC's search page now links strains by
  absolute URL; the parser follows.
- Each C. elegans stage carries its WBls term (`gently/organisms/celegans/
  stages.py`); the session detail hands the map to the Sessions tab.

**Removed**

- The dormant `Project` dataclass, its writer and the `agent/projects/`
  directory; the never-written `summary.yaml` from the storage map.

---

## Unreleased — a day on the rig, 2026-10-21

The scan calibration's first day on hardware, and what the rig turned up
while it ran. Four embryos scanned in 37 s each with the slope agreeing to
2 µm/deg across the slide; the rest of this entry is what was wrong around it.

**A dark reporter is not an empty field**

- The reporter detector's "no puncta" was filed as `no_object`, the perception
  pipeline's word for an empty field, and the role's patience for empty
  fields stopped three nuclear-marker embryos after five rounds as "likely
  hatched". A reporter that has not switched on is now `no_signal`, and the
  reporter path never touches the empty-field counter
  (`reporter_pseudo_stage`, `gently/app/orchestration/timelapse.py`). The
  tab names and colours the reporter's readings as their own, off the stage
  ramp.

**The SPIM camera comes back full frame**

- Calibration and volumes set the camera to the light-sheet band and nothing
  put it back, so the SPIM camera view after a calibration was the 2048×512
  strip. The live view clears the ROI whenever it reconfigures
  (`DiSPIMCamera.clear_roi`).

**The Embryos tab**

- The selected embryo has the overview's stage: its projections large, a
  timeline coloured by each timepoint's call, play, scrub, compare, and
  following newest; the detail pane follows the stage and a chosen call puts
  its frame on the stage. The stage is a factory now (`createOverviewStage`);
  the overview keeps its instance.
- The embryo tiles sized themselves to their content and ran under the panel
  beside them; the rail has one shrinkable column.
- The header counted the orchestrator's acquisition counter as "rounds",
  which starts at −1. It counts volumes.

**The Sessions tab opens at once**

- The list parsed every brightfield sidecar of every session (a night is
  thousands) and walked every file for its size: over a minute for the rig's
  353 sessions. Frames are counted by name (`FileStore.count_snapshots`),
  sizes come from the cache or not at all, and the tab asks
  `/api/sessions/sizes` for the rest in batches once the list is drawn.

**The agent's marking request**

- Pressing the pane's own Register while the agent waited filed the embryos
  through a different door and answered the agent with nothing; it detected
  again. Register is the answer while a session is open, the tool names the
  embryos registered meanwhile rather than saying "0", and the request shows
  in the chat with Open the map and Register, where the eye is.
- `get_plan_status` accepts a plan item's reference, which is what the pane
  hands the agent. The LED command the pane sends as it opens is a 503 before
  the device layer is up, not a traceback.

**The map is told what a pixel is at the sample**

- The agent's detect tool handed the map its image with `pixel_size_um=6.5`,
  the camera's physical pixel; through the 10× bottom objective a pixel is
  0.65 µm. The map scaled every mark by it, so every embryo registered
  through the agent's marking session was filed ten times too far from
  where the frame was taken, and the stage drove there. The hand-off carries
  the sample-plane pixel (`get_um_per_pixel`), the server refuses to start a
  marking session with a scale no objective could give, and the pane ignores
  one. Embryos marked through the pane's own Detect were never affected.

**The default role is lineaging**

- An embryo nobody has said anything about was `test`: read by the
  dopaminergic reporter detector, never staged. It is `lineaging` now, the
  nuclear perception route, at every door — the state, the store, the detect
  tools, the pane's toggle and roster. `lineaging` keeps the old default's
  patience for an embryo out of frame (five empty fields, not two). `test`
  is the reporter run's role, chosen on purpose.

**Dev**

- `httpx2` in the dev group: Starlette's test client asks for it.

---

## Notes on how we think about this

Things we've learned building this, roughly in order:

- The embryo should be the unit, not the image. That's how biologists
  think about it.
- If the agent decided something, you should be able to see why.
  Perception traces, plan versions, thinking blocks.
- Real-time control and experimental design are different enough to need
  separate modes with separate tools.
- The agent's understanding (ContextStore) and raw data (GentlyStore) have
  different lifecycles and should be kept apart.
- Publish/subscribe keeps coupling low. Most things don't need to call
  each other directly.
- Safety should come from the architecture (process isolation, device
  limits), not from hoping the prompt is good enough.
- The system should work offline. Mesh discovery is nice when it's there,
  but not required.
