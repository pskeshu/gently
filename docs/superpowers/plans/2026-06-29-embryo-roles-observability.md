# Embryo roles/strain + multi-embryo Operations observability (D2) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Steps use `- [ ]`.

**Goal:** Add a per-embryo `strain` field, refine roles to *use* (add lineaging + a subject/reference `class`), and build the Operations roster lens that reads existing roles + strain. Detector stays on role this pass (full detector→strain separation is tracked future work — spec §4).

**Architecture:** Reuse gently's existing `role` (`roles.REGISTRY`, `EmbryoState.role`, `embryo.yaml`). Add `strain` alongside role; extend the registry; expose roles to the frontend; render a role+strain roster lens above the operation spine.

## Global Constraints
- REUSE the existing role concept — do NOT invent a new taxonomy. Extend `gently/harness/roles.py:REGISTRY` (`EmbryoRole` dataclass) and read role via the existing accessors (`EmbryoState.role`, `EmbryoInfo["role"]`, `get_role`/`REGISTRY.get`, `/api/embryos/positions`).
- Strain is a FREE-FORM string per embryo (no registry this pass). Coexists with plan-level strain/genotype overrides.
- Roles render with their REAL `REGISTRY` `ui_color`/`ui_icon` (magenta test, cyan calibration, lineaging's own) — never invented colors.
- `class` ('subject'|'reference') is an attribute ON `EmbryoRole`, derived from role; test/unassigned→subject, calibration/lineaging→reference.
- Backward compatible: embryos without strain → None; plans without role-scope render as D today.
- Detector stays wired to role THIS PASS (spec §4 documents the full separation as future work — do not implement it here).
- Git hygiene: stage only your files by explicit path; never `git add -A`.

---

### Task 1: Per-embryo `strain` field
**Files:** Modify `gently/core/store_types.py` (`EmbryoInfo` — add `strain: str | None`); `gently/harness/state.py` (`EmbryoState` — add `strain: str | None = None`); `gently/core/file_store.py` (`register_embryo` — accept `strain=None`, write/coalesce it in embryo.yaml like `role`); `gently/ui/web/routes/data.py` (`/api/embryos/positions` ~:689 — add `"strain": emb.get("strain")`). Test: `tests/test_embryo_strain.py`.
- [ ] Confirm how `role` is threaded through `register_embryo` (`file_store.py:507-580`, write at :576, coalesce at :564) and EmbryoState (`state.py:138`); mirror it for `strain`. Confirm the positions endpoint shape (`data.py:661-696`).
- [ ] TDD: register an embryo with `strain="pan-nuclear GFP"` → get_embryo returns it; update coalesces; absent → None; positions endpoint includes strain. `pytest tests/test_embryo_strain.py -v`; `pytest -q` clean. Commit `feat(d2): per-embryo strain field`.

### Task 2: Roles refined — lineaging + subject/reference class
**Files:** Modify `gently/harness/roles.py` (add `class_: str = "subject"` — or `klass`/`role_class` to avoid the `class` keyword — to `EmbryoRole`; add a `lineaging` entry to `REGISTRY`; set `class_` on each role: test/unassigned→subject, calibration/lineaging→reference; give lineaging its own ui_color/ui_icon/default_cadence/detector kept None or nuclear like calibration). Test: `tests/test_roles_registry.py` (extend if exists).
- [ ] Add the `class_` field (default "subject") to `EmbryoRole`; set it on all REGISTRY entries; add `lineaging` (reference, distinct ui_color e.g. a teal/green, ui_icon, cadence). Keep `detector_name` as-is on each role (staged). Confirm nothing else constructs EmbryoRole positionally in a way the new field breaks.
- [ ] TDD: `REGISTRY["lineaging"].class_=="reference"`; `REGISTRY["test"].class_=="subject"`; `REGISTRY["calibration"].class_=="reference"`; `get_role("lineaging")` works; existing roles/fields unchanged. `pytest tests/test_roles_registry.py -v`; `pytest -q` clean. Commit `feat(d2): roles-as-use — add lineaging + subject/reference class`.

### Task 3: `/api/roles` route + role-scoped tactic scope
**Files:** Create `gently/ui/web/routes/roles.py` (`GET /api/roles` → `{roles:[{name,description,class_,ui_color,ui_icon,default_cadence_seconds}]}` from `list_roles()`/REGISTRY; register in `routes/__init__.py`). Modify `gently/app/tools/operation_plan_tools.py` (allow `scope.mode=='role'` + `scope.role` in validation — accept a REGISTRY key); add a pure resolver `resolve_scope_embryos(scope, roster_or_embryos) -> list[str]` (in a small module or operation_plan_tools) mapping mode=role→embryo_ids by role. Test: `tests/test_roles_route.py`, `tests/test_role_scope.py`.
- [ ] `/api/roles` mirrors `routes/tactic_library.py` (simple list route, graceful). The resolver maps `{mode:'role',role:'test'}` against a list of embryos-with-roles → the matching ids; global→all, embryos→explicit. Validation accepts mode=role with a valid role key.
- [ ] TDD: route returns the registry incl. lineaging + class_; resolver resolves role→ids, global→all, embryos→explicit, unknown role→[]. `pytest tests/test_roles_route.py tests/test_role_scope.py -v`; `pytest -q` clean. Commit `feat(d2): /api/roles route + role-scoped tactic scope resolver`.

### Task 4: Operations roster lens (frontend)
**Files:** Modify `gently/ui/web/static/js/experiment-overview.js` (add a roster lens above the operation spine: fetch `/api/embryos/positions` + `/api/roles`, group embryos by role `class_` (Subjects foregrounded, References compact) then by role, each row `id · role chip (REGISTRY ui_color/ui_icon) · strain · cadence-phase chip · current tactic (from the plan's role-scoped tactics) · state`; render tactic-node scope by role using the resolver/role labels); `gently/ui/web/static/css/experiment.css` (the `.ops-roster*` classes, using the role colors from the API, not hardcoded). Reference: the validated prototype `scratchpad/d2proto/index.html` (regrounded to real role colors).
- [ ] Build the roster lens reading the real endpoints + role metadata (colors from `/api/roles`, not invented); class split → role groups → strain; spine nodes show role-scope ("→ test · E01.."). Backward compat: no embryos/roles → omit the lens, spine renders as D. `node --check`; build/refresh the opsv3 (or d2) Chrome harness with the real files for the controller to audit. Commit `feat(d2): Operations roster lens — embryos by role + strain`.

## Self-Review
- Strain→T1; roles/class→T2; roles route + role-scope→T3; roster lens→T4. ✓
- Open confirmations: register_embryo/EmbryoState role threading (T1), EmbryoRole construction sites (T2), the route/resolver pattern (T3), the embryos+roles endpoints + plan cross-reference for current-tactic (T4).
- Type consistency: `strain` str|None across model/store/endpoint; `class_` on EmbryoRole + in /api/roles + read by the renderer; role keys consistent across REGISTRY, scope.role, resolver, renderer.
- Staged: detector stays on role (spec §4 future work referenced, not implemented).
