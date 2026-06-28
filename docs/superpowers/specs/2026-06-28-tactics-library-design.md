# Design: Tactics library (sub-project G)

Status: design decided 2026-06-28. Base branch: off `feature/operations-tab` (D) — G reuses D's
typed tactic objects (the Operation Plan substrate). Mirrors the existing **plan-template** pattern
(`_plans.py:551-667`, `plan_mode/tools/templates.py`).

## 0. Why G is now small

D made tactics **typed, agent-authored objects** (kind/structure/scope/live/relations). So a "tactics
library" is just **save / list / apply those typed tactic objects** — a near-copy of how plan
templates are already saved/applied. No new conceptual model; it's the reuse layer over D.

## 1. The saved tactic

```
saved_tactic = {
  id, name, slug, kind,            # kind ∈ D's tactic kinds
  structure,                       # the tactic's structure (phases/cadence/watch) sans live state
  scope_hint?,                     # default scope (global/embryos)
  description, params?,            # human description + default params
  created_at, created_by           # 'agent' | 'human'
}
```
Stored at `agent/tactic_library/{id}_{slug}.yaml` (a new FileContextStore domain), mirroring the
plan-template store. A saved tactic is a *template* — the planned form of a tactic, with no live
state (no `live`, no `id`-per-run).

## 2. Architecture (three units, mirroring plan templates)

### 2.1 Store — `FileContextStore`
Add `save_tactic(tactic, name=None) -> id`, `list_tactics() -> list[dict]`, `get_tactic(id_or_name)`,
`apply_tactic(id_or_name) -> dict` (returns a fresh planned tactic — new run id, `state:"planned"`,
no `live`). Mirror `save_plan_template`/`list_plan_templates`/`get_plan_template`/`apply_plan_template`
(`_plans.py:551-667`) — same id/slug, YAML, atomic write. Fire `CONTEXT_UPDATED`.

### 2.2 Tools — `gently/app/tools/tactic_library_tools.py`
- `save_tactic(name, tactic, description="")` — persist a tactic template (the agent passes a tactic
  dict, e.g. lifted from the current Operation Plan, or authored fresh).
- `list_tactics()` — the library.
- `apply_tactic(id_or_name)` — instantiate a fresh planned tactic and **add it to the current
  Operation Plan** (`get_operation_plan` → append the planned tactic → `set_operation_plan`), so a
  saved tactic flows straight into the active plan. Mirror `plan_mode/tools/templates.py`.

### 2.3 Route — `GET /api/tactic_library`
Mirror `routes/operation_plan.py`; returns `{tactics:[...]}` from `list_tactics()`. For a future
browse UI; the tools are the primary surface for now.

## 3. Out of scope
- A dedicated library browse/edit UI (the tools + chat suffice for v1; a panel is a follow-on).
- Tactic *sets*/composites (save a whole plan as a named operation template) — a natural extension
  once single-tactic save/apply lands; defer.
- Sharing/export across machines.

## 4. Testing
- Store: save→list→get→apply round-trip; apply returns a fresh planned tactic (new id, no live state).
- Tools: save persists; apply adds a planned tactic to the current Operation Plan; list returns the library.
- Route: returns the library; empty when none.
