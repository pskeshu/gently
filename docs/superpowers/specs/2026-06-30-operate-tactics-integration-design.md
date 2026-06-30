# Operate → Tactics/Timelapse Integration ("Phase C: Run") — Design

Date: 2026-06-30
Status: Approved (build all 3 phases)
Branch: feature/temperature-operations-all
Source: Opus expert workflow (tactics + timelapse + agent/resolution + UI study → 3 candidates → synthesis)

## Problem

The Operate view dead-ends: `confirmMarks()` registers positions-only embryos
(role `unassigned`) into `experiment.embryos` (the SSOT), and `onEmbryosUpdate`
auto-dives into the manual per-embryo loop. There is no path from "embryos
marked" to the agentic timelapse or to a tactic/plan.

Underneath, all four imaging surfaces (Operate manual loop, Manual-view timelapse
form, agent tools, Operations spine) already share **one engine**
(`TimelapseOrchestrator`) and **one language** (the per-session *Operation Plan*
of **tactics**). Operate is wired to none of them. `resolve_scope_embryos`
(role_scope.py) — the scope→embryos resolver — has zero callers; it was built for
exactly this hand-off.

## Decisions (user)

- **Build all three phases** (not just the first slice).
- **Adaptive timelapse default monitoring mode = `idle`** (pure fixed-cadence;
  operator opts into expression/pre-terminal monitoring explicitly).
- **Live run is monitored in-Operate** (the rail flips to a compact read-only
  run-spine), with a deep-link to the Operations tab.

## Design — Phase C "Run" on the Operator Spine

Stepper gains a third node: **① Focus → ② Mark → ③ Run.** After Confirm, instead
of auto-diving into the manual loop, the stepper advances to ③ Run and the right
rail shows a **Run chooser**. The **tactic** is the single unifying object: every
run mode emits exactly one tactic scoped to the marked set
(`scope.mode='embryos', embryo_ids=[marked]`).

| Mode | Behavior | Tactic kind |
|---|---|---|
| **A — Manual one-by-one** | the existing Phase-B b1–b5 loop, now reached from the chooser | `oneshot` (cosmetic; keeps the spine coherent) |
| **B — Adaptive timelapse** | inline form: interval, stop condition, monitoring_mode (default **idle**); reuses `/api/devices/timelapse/start` with explicit `embryo_ids=[marked]` | `standing_timelapse` (+ optional `reactive_monitor` layered on) |
| **C1 — From library** | apply a saved tactic, scope re-pointed to the marked set | template's kind, via `apply_tactic` |
| **C2 — Continue a plan** | resume_plan candidate → `execute_plan_item(item_ref, embryo_ids=[marked])` + `seed_operation_plan_from_plan_item` | tactics seeded from `ImagingSpec.tactics` |
| **C3 — Hand to agent** | open AgentChat preloaded with the roster; agent authors + starts the plan | agent-authored via `declare_operation_plan` |

**Running:** the rail flips to a compact read-only **run-spine** (reuse
`experiment-overview.js` `_renderOpsTactic`/`_renderOperationSpine`): state-colored
tactic cards + live readouts + **Pause/Stop** (`pause_timelapse`/`stop_timelapse`)
+ **Open in Operations** deep-link. Optional `set_autonomy('ask'|'auto')`.

**Roles wrinkle (load-bearing):** marking stays positions-only, but
`expression_monitoring` rules scope to `role=='test'` (subject) — so role-scoped
monitoring matches zero just-marked embryos unless roles are set. The Run chooser
shows a **role chip strip** (all marked default to **Subject**, flip any to
**Reference**); choosing a non-manual mode applies roles via a new thin
`POST /api/embryos/roles`. Roles are assigned **at Run, not at marking** —
consistent with the "marking is positions-only" rule.

## Keystone new component: the Tactic Executor

`gently/app/orchestration/tactic_executor.py` —
`execute_tactic(session, tactic)`: `resolve_scope_embryos(scope, roster)` →
dispatch by `kind` to `orchestrator.start` / `enable_monitoring_mode` /
`queue_burst` (later `acquire_volume` / temp protocols), threading `tactic_id` →
`transition_tactic('active')` + merge live binds. This makes the tactics language
*executable* (not just descriptive) and is `resolve_scope_embryos`'s first caller.
It centralizes the `kind`→tool mapping the agent also uses (no duplication).

## Integration points (verified code seams)

- `operate.js` `onEmbryosUpdate` — stop auto-selecting embryo 1; on first confirm
  advance stepper to ③ Run + render the chooser.
- `operate.js` `renderStep` single-driver — host Phase C chooser + live run-spine
  as new render branches (`data-active='c0'` / running) without disturbing a1/b*.
- `POST /api/devices/embryos/confirm` (`data.py`) — unchanged (positions-only SSOT).
- `POST /api/devices/timelapse/start` (`data.py`) — Mode B reuses verbatim with
  `embryo_ids=[marked]`; Phase 2 ADDS tactic seeding (closes the `data.py:~1004`
  plan-auto-link TODO). `volume_geometry` stays NOT forwarded (RIG-DEFERRED).
- `TimelapseOrchestrator.start/enable_monitoring_mode/queue_burst` — the engine
  the executor dispatches into; holds marked `EmbryoState` refs (zero copy).
- `resolve_scope_embryos` (role_scope.py) — Tactic Executor is its first caller.
- `start_adaptive_timelapse` (timelapse_tools.py) — add `tactic_id` for lifecycle
  symmetry (the other start/stop tools already have it).
- `OperationPlanUpdater` — already maps BURST_COMPLETE→done,
  EMBRYO_CADENCE_CHANGED/TRIGGER_FIRED→bind; drives the run-spine once tactics
  carry `tactic_id`.
- resolution dispatch (`bridge._dispatch_resolution_pick`) + `execute_plan_item`
  + `seed_operation_plan_from_plan_item` — Modes C2/C3 + session guard.
- `experiment-overview.js` `_renderOperationSpine`/`_renderOpsTactic` — reused for
  the in-Operate run-spine.
- `AgentChat.togglePanel`/`runCommand` — Mode C3.

## New backend

1. **Tactic Executor** (`gently/app/orchestration/tactic_executor.py`) — the keystone (above) + unit tests.
2. **`POST /api/embryos/roles`** (thin) — reuse `assign_embryo_roles` internals → set `EmbryoState.role` + fire EMBRYOS_UPDATE. Default marked→subject mandatory.
3. **Tactic seeding on `/api/devices/timelapse/start`** — declare+seed `standing_timelapse` (+ `reactive_monitor`) and transition active (closes the TODO). Additive, minimal, idempotent.
4. **`tactic_id` on `start_adaptive_timelapse`** — lifecycle symmetry.
5. **Tactic structure schema extension** (`operation_plan_tools._validate_tactics`) — add `stop_condition`/`condition_value`/`monitoring_mode`/`interval` to `standing_timelapse`/`reactive_monitor` so a tactic is self-describing for the executor.
6. **Session guard** — ensure a live session/orchestrator before Phase C runs (orchestrator is None without one); reuse `should_enter_resolution`/bootstrap.

## Phasing

- **Phase 1 (cheap slice):** Operate Phase C scaffold (③ Run node, stop auto-dive,
  Run chooser) + **Mode B** (reuses `/timelapse/start`, no new backend) + thin
  `POST /api/embryos/roles` + role chip strip + in-Operate run-spine. Working
  marking→adaptive-timelapse hand-off.
- **Phase 2 (tactics integrity):** tactic seeding on `/timelapse/start`;
  `tactic_id` on `start_adaptive_timelapse`; tactic structure schema extension;
  Mode A `oneshot`.
- **Phase 3 (keystone + breadth):** Tactic Executor (+ tests) powering Mode C1
  (library); Mode C2 (continue a plan); Mode C3 (hand to agent); `set_autonomy`
  in the run-spine.

## Testing

- Backend (TDD): Tactic Executor (scope resolution + kind dispatch, mocked
  orchestrator); roles route; schema validator extension.
- Frontend: shim + Chrome MCP — drive mark→Confirm→Run chooser→role assign→Mode B
  start→run-spine; verify stepper/chooser/run-spine; UI audit.
- Adversarial code-review workflow over the full diff before merge.

## Rig-only / honesty flags

Real stage motion + acquisition stay RIG-DEFERRED (orchestrator calls
`client.acquire_volume` directly; "Bluesky" framing is aspirational).
`volume_geometry` not forwarded by `/timelapse/start`. The `oneshot` manual tactic
is cosmetic (no orchestrator mechanism backs it). Timelapse start needs a live
session/orchestrator — unavailable in the hardware-free shim, so Mode B's actual
start is rig/session-verified; the UI flow + tactic emission are shim-verifiable.
