# Design: Embryo roles/strain + multi-embryo Operations observability (sub-project D2)

Status: design 2026-06-29 (revised after recon + user steering). Extends D (the agent-authored
Operation Plan). Driven by the real temperature-strain run: **multiple embryos with roles**. Branch:
`feature/embryo-roles-observability` (off G / the operations stack).

## 0. Reuse what exists — gently ALREADY has embryo roles
Recon confirmed a first-class role concept; D2 builds on it, does NOT reinvent it:
- `role: str` on every embryo — `EmbryoInfo["role"]` in `embryo.yaml` (`gently/core/store_types.py:34`,
  written by `FileStore.register_embryo` `file_store.py:564/576`) and `EmbryoState.role`
  (`gently/harness/state.py:138`, default `"test"`).
- `gently/harness/roles.py:REGISTRY` — the taxonomy. Today: `test` (biological subject, magenta
  `#ff66cc`, star, dose ×1), `calibration` (reference, cyan `#00cccc`, diamond, dose ×10,
  nuclear-marker pipeline), `unassigned`. Each `EmbryoRole` carries `description,
  default_cadence_seconds, detector_name, photodose_budget_multiplier, ui_color, ui_icon,
  no_object_consecutive_terminal`.
- Already surfaced in the **Map view** (`marking.js` ROLE_CYCLE/ROLE_COLORS) + `GET
  /api/embryos/positions` (`routes/data.py:689`); already read for photodose scaling
  (`timelapse.py:426`). The **Embryos strip + Operations do NOT show role** — that is the gap D2 fills.

## 0a. The load-bearing insight — strain ⊥ use
Role and strain are ORTHOGONAL axes that gently currently conflates (the `calibration` role hardcodes
the nuclear-marker detector):
- **Strain** = the biological sample (e.g. pan-nuclear GFP — GFP in every nucleus; a dopaminergic
  reporter). It is what is *labeled* → it determines the **perception/detector**.
- **Use (role)** = what the embryo is FOR: subject (test), calibration, lineaging. Drives
  cadence / photodose / adaptive-vs-fixed tactics.
The SAME pan-nuclear strain is used for **lineaging** (C. elegans lineage tracing) OR **calibration**
(microscopist calibrating the lightsheet on the same nuclear structure); and **lineaging** can run on
*non*-pan-nuclear strains. So neither determines the other. Fix: make **strain** the per-embryo field
that drives the detector, and let **role** be purely *use*.

## 1. Schema changes

### 1.1 Per-embryo `strain` (NEW, free-form string)
Add `strain: str | None` to `EmbryoInfo` (`store_types.py`) + `EmbryoState` (`state.py`), persisted in
`embryo.yaml` via `register_embryo(..., strain=None)`. Free-form (e.g. `"pan-nuclear GFP"`); coexists
with the existing plan-level `strain`/`genotype`/`reporter` spec overrides
(`memory/file_store.py:1746`). No strain registry this pass.

### 1.2 Roles refined to USE — extend `roles.REGISTRY`
- Add `lineaging` to the REGISTRY (its own `ui_color`/`ui_icon`/`default_cadence_seconds`/policy).
  Distinct *use* from calibration even when it shares the pan-nuclear strain.
- Add a `class` attribute to `EmbryoRole`: `'subject' | 'reference'` — `test`→subject;
  `calibration`,`lineaging`→reference; `unassigned`→subject (safe default). Drives the Operations
  foregrounding; derived from role, not a new per-embryo field.
- Keep `detector_name` ON the role THIS PASS (staged — see §4). Document that it logically belongs to
  strain.

### 1.3 Role-scoped tactics (on D's Operation Plan)
`tactic.scope` gains `mode:'role'` + `role:<key>` (an existing REGISTRY key: test/calibration/lineaging).
`mode=role` resolves to all roster embryos with that role. Backward compatible (global/embryos unchanged).

## 2. Observability — Operations roster lens (reads existing role + strain)
A roster lens above the operation spine, reading the LIVE embryos (via the tracker / `EmbryoState.role`
/ `/api/embryos/positions`) + `roles.REGISTRY` metadata (real `ui_color`/`ui_icon` — magenta test, cyan
calibration, lineaging's color) — NOT new colors:
- Split FIRST by role `class`: **Subjects** (foregrounded — adaptive tactics/scenarios) then
  **References** (compact — calibration/lineaging holding steady).
- Within each class, group by role (use). Each embryo row: `id · role chip (REGISTRY color/icon) ·
  strain · cadence-phase chip · current tactic · state`.
- The spine's tactic nodes show scope by role ("→ test · E01,E02,E03"; "→ reference").
This answers "who's a subject vs a reference, what's each one's strain + use, what's happening to each."

## 3. Architecture (reuse D + existing role plumbing)
- Strain: field on EmbryoInfo/EmbryoState + register_embryo + the marking tool (`detection_tools.py`)
  accepts strain alongside role; `/api/embryos/positions` adds `strain`.
- Roles: extend `roles.py:REGISTRY` (+ `class` on EmbryoRole); reuse `get_role`/`REGISTRY.get`.
- Operations: roster lens in `experiment-overview.js` reading the embryos endpoint + REGISTRY; role-scope
  rendering on the spine. Role→embryo_ids resolver is a pure read over the roster.
- Tactic scope: `declare_operation_plan` accepts `scope.role`; seeding can set role-scoped tactics.

## 4. ⚠️ FULL SEPARATION — TRACKED FUTURE WORK (staged out of this pass)
This pass STAGES the model; the clean end-state (do later, do not lose):
1. **Detector follows strain, not role.** Move `detector_name`/perception selection off `EmbryoRole`
   onto strain (a strain→detector mapping / strain registry). `calibration` stops implying the
   nuclear-marker pipeline; a pan-nuclear *test* embryo would also get nuclear perception.
2. **Strain registry** (like `roles.py`): strain key → {label, markers, default detector, ui}. Lets
   strain drive detector cleanly + gives the UI consistent strain labels/colors.
3. **Role policy audit** once detector is off role: confirm role carries only cadence / photodose /
   adaptive-vs-fixed / class, nothing strain-derived.
4. Backfill: existing embryos have role but no strain — a migration/default for strain on legacy data.
These are real work items; they belong in a follow-on PR after D2's staged version lands.

## 5. Out of scope (D2)
- Per-role analysis/plots (results view, not Operations).
- A manual role/strain editor UI beyond what the Map marking tool already does.
- The full detector→strain decoupling (§4 — explicitly future).

## 6. Testing
- Strain field round-trips through embryo.yaml (register/get); `/api/embryos/positions` includes it.
- `roles.REGISTRY` has lineaging + `class`; `get_role` returns class; existing roles unchanged.
- Role-scope resolver: role→embryo_ids over a roster; global/embryos unchanged; backward compat.
- Renderer: roster lens (class split → role groups → strain) + role-scoped spine render across fixtures
  using REAL REGISTRY colors (node --check + Chrome audit). Roster-less/role-less plan renders as D today.
