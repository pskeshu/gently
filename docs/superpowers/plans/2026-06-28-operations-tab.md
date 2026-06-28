# Operations — agent-authored Operation Plan (D v3) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Steps use `- [ ]`.

**Goal:** The agent emits a typed Operation Plan (its tactics, planned/active/done); Operations renders it ⊕ live telemetry. Built backend-first (the user-directed, low-rework part), then the renderer.

**Architecture:** A typed `OperationPlan` in `FileContextStore` (new domain), written by a forced-tool agent call, served on a route, rendered by a data-driven operation-spine generalized for tactic kinds, with live-telemetry binding + scenario test mode.

**Tech Stack:** Python (FileContextStore YAML, the `@tool`/forced-`tool_choice` pattern, EventBus `CONTEXT_UPDATED`), FastAPI route, vanilla-JS + SVG renderer, pytest.

## Global Constraints
- Source of truth = the agent's typed plan (NOT a backend reconstruction). Live telemetry binds onto declared tactics; the agent declares tactic identity, the system supplies live values.
- Plan schema (see spec §1): `{session_id,title,goal,tactics:[{id,name,kind,state,scope,rationale,structure,live_bind,relations}],updated_at,updated_reason}`. `kind∈{standing_timelapse,reactive_monitor,scripted_protocol,exclusive_burst,oneshot,custom}`, `state∈{planned,active,done}`. No round-robin.
- Forced typed output mirrors `gently/harness/memory/notebook_ask.py` (ASK_TOOL + `tool_choice={'type':'tool',...}` → validated `block.input`) OR the `@tool` auto-schema (`harness/tools/registry.py`).
- Store mirrors `session_intents`/`active/` domains; fire the existing `CONTEXT_UPDATED`.
- Route mirrors `gently/ui/web/routes/context.py` (`/api/context`).
- Renderer = the validated operation-spine (harness `scratchpad/opsdesign/harness.html`) with the audit fixes (queued reads cocked + "next" marker; active phase/status is the headline; flatten — no card-in-card; whole-left-column amber + colored left edge per state; copy "queued"; mono values). Generalize for the tactic kinds (standing→per-embryo cadence strip; reactive→watch/reaction/status; scripted→phases).
- Frontend: no JS harness → `node --check` + Chrome-MCP audit across scenario fixtures.
- Git hygiene: stage only your files by explicit path; never `git add -A`.

---

### Task 1: OperationPlan model + FileContextStore domain
**Files:** Create/extend `gently/harness/memory/model.py` (an `OperationPlan`/`Tactic` dataclass or a documented dict schema); Modify `gently/harness/memory/file_store.py` (`set_operation_plan(session_id, plan)` / `get_operation_plan(session_id)`, YAML under `agent/operation_plans/{session_id}.yaml`, fire `CONTEXT_UPDATED`). Test: `tests/test_operation_plan_store.py`.
- [ ] Confirm the real `FileContextStore` domain pattern from `set_session_intent`/`create_session_intent` (~file_store.py:758) + the `_notify_*`/`CONTEXT_UPDATED` emit. Mirror it.
- [ ] TDD: set→get round-trip preserves the tactics list + states; `CONTEXT_UPDATED` fired on set. `pytest tests/test_operation_plan_store.py -v`; `pytest -q` clean. Commit `feat(operations): OperationPlan store domain in FileContextStore`.

### Task 2: `declare_operation_plan` agent tool (forced typed output)
**Files:** Create `gently/app/tools/operation_plan_tools.py` (+ register in `tools/__init__`). Test: `tests/test_operation_plan_tool.py`.
- [ ] Confirm the `@tool` decorator + the forced-tool pattern (registry.py auto-schema, or a literal schema like notebook_ask.ASK_TOOL). The tool accepts the typed plan (tactics list) and writes it via `store.set_operation_plan`. Resolve the store from context (mirror existing memory tools).
- [ ] TDD: calling the tool with a plan persists it (get returns it) + returns a confirmation; missing store → error. `pytest tests/test_operation_plan_tool.py -v`; `pytest -q` clean. Commit `feat(operations): declare_operation_plan typed agent tool`.

### Task 3: Route `GET /api/operation_plan/{session_id}`
**Files:** Create `gently/ui/web/routes/operation_plan.py` (+ register in `routes/__init__.py`). Test: `tests/test_operation_plan_route.py`.
- [ ] Mirror `routes/context.py` / `routes/temperature.py` `_resolve_session` + `server.context_store`/`gently_store` resolution. Return the stored plan; 404/empty handled; `session=current` resolves newest.
- [ ] TDD (TestClient + mock store): returns the plan; empty when none. `pytest tests/test_operation_plan_route.py -v`; `pytest -q` clean. Commit `feat(operations): GET /api/operation_plan/{session} route`.

### Task 4: The operation-spine renderer + scenario library (frontend)
**Files:** Create `gently/ui/web/static/js/operations-scenarios.js` (plan fixtures: temp_strain, expression_onset, hatching_detect, transmission_survey, decided_plan, async_multi, idle); Rewrite the Overview path in `gently/ui/web/static/js/experiment-overview.js` to the data-driven operation-spine (generalized per tactic kind, audit fixes); port CSS to `experiment.css`. Reference: `scratchpad/opsdesign/harness.html`.
- [ ] Port the renderer (renderOperation/renderTactic/renderReadout/renderPhase) + add per-kind rendering: standing→per-embryo cadence strip; reactive→watch/reaction/status; scripted→phase stepper; exclusive/oneshot→compact. Audit fixes baked in. `?scenario=` dev mode loads a fixture.
- [ ] `node --check` both JS; build a Chrome-MCP harness at `scratchpad/opsv3/` (real repo files) for the controller to audit across fixtures. Commit `feat(operations): operation-spine renderer + plan scenario library (data-driven)`.

### Task 5: Live binding + refresh
**Files:** Modify `experiment-overview.js` (fetch `/api/operation_plan`, bind live telemetry from `/strategy`/get_status onto tactics by `live_bind`, subscribe `CONTEXT_UPDATED` + tactic events → debounced refetch).
- [ ] Bind temperature/current-burst/cadence/signal onto declared tactics' readouts/progress; live refresh on plan-change + telemetry events (debounced). `node --check`; harness check. Commit `feat(operations): live telemetry binding + event-driven refresh`.

## Self-Review
- Store→Task1; tool→Task2; route→Task3; renderer+scenarios→Task4 (audit fixes); live binding→Task5. ✓
- Open confirmations: FileContextStore domain pattern (T1), the forced-tool/@tool pattern (T2), the route store-resolution (T3), the Overview render seam + harness renderer (T4), the live telemetry sources (T5).
- Type consistency: the plan/tactic schema is identical across store (T1), tool (T2), route (T3), fixtures+renderer (T4), binding (T5).
