# Tactics Library (G) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Steps use `- [ ]`.

**Goal:** Save / list / apply typed tactics (D's tactic objects) as reusable templates — a near-copy of the plan-template pattern.

**Architecture:** A tactic-library domain in `FileContextStore` (save/list/get/apply), agent tools, and a read route — mirroring `save_plan_template`/`apply_plan_template`.

## Global Constraints
- Mirror the plan-template pattern: `_plans.py:551-667` (`save_plan_template`/`list_plan_templates`/`get_plan_template`/`apply_plan_template`) + `plan_mode/tools/templates.py`.
- A saved tactic is a TEMPLATE (planned form): `{id,name,slug,kind,structure,scope_hint?,description,params?,created_at,created_by}` — NO live state. `apply_tactic` returns a fresh planned tactic (new run id, state="planned", no `live`).
- Store domain `agent/tactic_library/{id}_{slug}.yaml`; fire `CONTEXT_UPDATED`.
- Git hygiene: stage only your files by explicit path; never `git add -A`.

---

### Task 1: Tactic-library store domain
**Files:** Modify `gently/harness/memory/file_store.py` (or `_plans.py` if templates live there) — add `save_tactic(tactic, name=None) -> str`, `list_tactics() -> list[dict]`, `get_tactic(id_or_name) -> dict | None`, `apply_tactic(id_or_name) -> dict | None`. Test: `tests/test_tactic_library_store.py`.
- [ ] Confirm the exact plan-template implementation (`_plans.py:551-667`) — the id/slug generation, the YAML write helper, the `_notify`/`CONTEXT_UPDATED` emit — and mirror it for tactics at `agent/tactic_library/`. `apply_tactic` returns a deep copy with a fresh `id` (e.g. `tac_<8hex>`), `state="planned"`, and `live`/run-state stripped.
- [ ] TDD: save a tactic → list shows it → get returns it → apply returns a fresh planned tactic (new id, no live, state=planned); get/apply unknown → None. `pytest tests/test_tactic_library_store.py -v`; `pytest -q` clean. Commit `feat(tactics-library): tactic-library store domain (save/list/get/apply)`.

### Task 2: Agent tools
**Files:** Create `gently/app/tools/tactic_library_tools.py` (+ register in `tools/__init__`). Test: `tests/test_tactic_library_tools.py`.
- [ ] Mirror `plan_mode/tools/templates.py` (the `@tool` usage + store resolution from context). Tools: `save_tactic(name, tactic, description="")` → `store.save_tactic`; `list_tactics()` → `store.list_tactics`; `apply_tactic(id_or_name)` → `store.apply_tactic` then append the planned tactic to the current Operation Plan (`get_operation_plan(session)` → append → `set_operation_plan`; create a minimal plan if none). Resolve the context store + session from the agent (as `declare_operation_plan` does). Register the module.
- [ ] TDD (fake context store + session): save persists; apply adds a planned tactic to the current Operation Plan; list returns the library; missing store → error. `pytest tests/test_tactic_library_tools.py -v`; `pytest -q` clean. Commit `feat(tactics-library): save/list/apply_tactic agent tools`.

### Task 3: Route `GET /api/tactic_library`
**Files:** Create `gently/ui/web/routes/tactic_library.py` (+ register in `routes/__init__.py`). Test: `tests/test_tactic_library_route.py`.
- [ ] Mirror `routes/operation_plan.py` — resolve `server.context_store`, return `{tactics: store.list_tactics()}`; empty list when none. Register the router.
- [ ] TDD (TestClient + mock store): returns the library; empty when none. `pytest tests/test_tactic_library_route.py -v`; `pytest -q` clean. Commit `feat(tactics-library): GET /api/tactic_library route`.

## Self-Review
- Store→Task1; tools→Task2; route→Task3. ✓
- Open confirmations: the plan-template implementation to mirror (T1), the `@tool` + store/session resolution (T2), the route store handle (T3).
- Type consistency: the saved-tactic dict shape is identical across store (T1), tools (T2), route (T3); `apply_tactic`'s fresh planned tactic matches D's tactic schema.
