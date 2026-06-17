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
