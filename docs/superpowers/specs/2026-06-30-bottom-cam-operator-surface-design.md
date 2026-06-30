# Bottom-cam → SPIM Operator Surface ("Operate" view) — Design

Date: 2026-06-30
Status: Approved (proceed to implementation)
Branch: feature/temperature-operations-all

## Purpose

A single, professional, guided operator surface for the manual bottom-camera →
SPIM acquisition workflow, replacing the scattered current UX (bottom cam on the
Map view, SPIM controls on the Manual view, embryo marking on the Embryos page).
Mirrors how the operator physically works the rig:

1. Focus the bottom objective on the embryos.
2. Mark **all** embryos in one pass (a single FOV holds them).
3. Per embryo: center → lower the SPIM head → focus the SPIM (LED on) → acquire.

This is sub-project **A** (the spine). Two data flywheels hang off it and are
specified separately: **B** marking→localization labels (retire SAM), **C**
manual-focus→autofocus-validation. A exposes the hook points (confirm, focus
score) they will tap, but does not implement them.

## Settled decisions

- **Home:** a new `Operate` view in the device tab, alongside Map/Details/3D/Manual.
  One purpose per view: Map stays a passive spatial monitor; Manual stays raw
  knobs. The Detect/Center/enlarge controls already added to Map migrate to Operate;
  Map keeps only read-only embryo dots.
- **Focus control:** software **nudge** buttons (± fixed steps), hard-fenced to
  the axis limits (F-drive floor 30 µm). **No autofocus** (objective-crash risk).
  Live focus-score readout to assist.
- **Marking:** batch — mark all embryos on one frozen full-res frame. **Positions
  only, no roles.** Roles are a separate, later, optional concern.
- **SPIM focus step:** inline in Operate (lightsheet live + galvo/piezo/LED nudges),
  not a handoff to Manual.
- **Single source of truth:** the canonical `experiment.embryos` list (already wired
  via EMBRYOS_UPDATE / /api/embryos/current). Detect feeds it through a
  human-confirm step, not a side list.
- **UI quality:** treat as a design pass (frontend-design), not a port of the
  amateur marking canvas.

## Architecture

New device-tab view `operate` (devices.js view list becomes
`['operate','map','details','optical3d','manual']`), three vertical zones:

1. **Survey** — enlarged bottom-cam live; bottom-Z focus nudge (fenced) + live
   focus score; Detect (SAM → candidates) or click-to-mark on a frozen frame;
   Confirm.
2. **Embryos** — the one canonical list, each row with a state chip and select.
3. **Acquire** (selected embryo) — Center → Lower SPIM head (F-drive, fenced) →
   inline lightsheet live + galvo/piezo/LED nudges + focus score → Acquire volume.

Per-embryo state machine (client-side, keyed by embryo id; persistence deferred):

```
marked ──Center──▶ centered ──Lower SPIM + focus──▶ focused ──Acquire──▶ imaged
```

### Component reuse

| Need | Reuse | New |
|---|---|---|
| Bottom-cam live + enlarge | camera panel (built) | move into Operate |
| Mark-all on frozen frame | MarkingManager interaction logic | re-homed canvas, redesigned, positions-only |
| Embryo list + Center | SSOT list + stage/move (built) | per-embryo state chips |
| SPIM focus | Manual galvo/piezo/LED/lightsheet-live endpoints | inline placement |
| Acquire | /api/devices/acquire/volume | — |
| Bottom-Z + F-drive nudge | DiSPIMZstage / DiSPIMFDrive device classes | device-factory wiring, polling, fenced endpoints |
| Focus score | analysis/core.calculate_focus_score | inject into camera stream payloads |
| Register marks (agent-free) | experiment.add_embryo | register-on-confirm endpoint |

## New backend endpoints (web routes proxy → device layer)

- `GET /api/devices/stage/bottom_z` · `POST /api/devices/stage/bottom_z/nudge {delta}`
  — read + fenced nudge of the bottom-camera focus Z (DiSPIMZstage).
- `GET /api/devices/spim/fdrive` · `POST /api/devices/spim/fdrive/nudge {delta}`
  — read + fenced nudge of the SPIM-head F-drive (floor 30 µm; report distance-to-floor).
- `POST /api/devices/detect_embryos` (revised) — return SAM candidates
  `{embryos:[{pixel_x,pixel_y,stage_x_um,stage_y_um,confidence}], stage_position}`;
  no auto-register.
- `POST /api/devices/embryos/confirm {markers:[{pixel_x,pixel_y}], stage_position,
  pixel_size_um, objective_mag}` — pixel→stage, register each into experiment.embryos
  (role 'unassigned'), fire EMBRYOS_UPDATE. Agent-free.
- Focus score injected into existing bottom-cam + lightsheet SSE payloads
  (`focus_score` field), computed server-side on the full frame.

Device-layer additions: instantiate DiSPIMZstage in the device factory when
present; add bottom-Z and F-drive to the slow position poller; fenced nudge
handlers (clamp/reject out-of-range).

## Safety / error handling

- All Z moves fenced server-side; device classes hard-enforce limits; out-of-range
  → 4xx, surfaced in UI. Nudges are bounded single steps — no autonomous/repeated
  moves.
- Device layer down → 503; controls disabled with clear state.
- Frozen-frame capture failure → error toast, stay in Survey. Confirm with zero
  markers → disabled.
- F-drive: never below floor; show distance-to-floor.

## Testing

- Backend (TDD): pixel→stage in register-on-confirm (reuse coordinate tests);
  fenced Z endpoints reject out-of-range; focus-score payload shape.
- Frontend: launch with the gently_perception shim; Chrome MCP drives
  detect→mark→confirm→EMBRYOS_UPDATE, per-embryo state transitions, out-of-range
  nudge blocked; screenshots + UI audit against the professional bar.
- Rig-only: real Z moves, real SPIM focus, SAM on a live frame.

## Out of scope (separate sub-projects)

- B: persist (frame + pixel markers + roles) as localization labels; benchmark
  classical/trained detectors vs SAM.
- C: poll missing Z axes for passive focus-trace logging + offline validator.
  (A wires the Z axes for read/move, which C extends to logging.)

## MVP boundary

Ship the Operate view end-to-end with maximal reuse: Survey (live + fenced bottom-Z
+ Detect/mark-all + Confirm), the SSOT list with state chips + Center, Acquire zone
(F-drive nudge + inline SPIM focus controls + Acquire). Per-embryo state client-side.
Defer label/focus-trace persistence (B/C).
