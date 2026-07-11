# Wrap-up PR plan — the temperature-experiment + Operations goal

**Principle (user-stated):** capture the breadth of work as **separate PRs**, each a distinct unit of
work with its own identity/ownership and its own tests, **additive on top of PR #58**
(`integration/ux2-all`, the UX-v2 stack), stacked so they **compose into the final product** in order.
Each PR diffs cleanly against its parent in the stack; integrate the chain when rig-verified.

## The stack (base = PR #58 `integration/ux2-all`)

| # | Branch | Unit of work | Parent | Tests / review |
|---|--------|--------------|--------|----------------|
| 1 | `feature/temperature-interface` (A) | Temperature persistence (`append/read_temperature_sample`) + sampler service + SVG temperature graph w/ setpoint line | #58 | per-task + whole-branch review; graph Chrome-audited |
| 2 | `feature/manual-mode-live-view` (B1) | Manual-mode imaging: lightsheet brightfield live view (sequence acquisition), illumination control (LED/laser presets), galvo/piezo scan params + **4 acquire-safety fixes** (C1/I1/I2/I3) | A | per-task + whole-branch (opus) review |
| 3 | `feature/temp-change-tactic` (C) | Automated temp-change burst protocol (`wait_for_temperature_lock` + driver), burst-acquisition wiring, protocol events + agent tool | B1 | per-task + whole-branch review + fixes |
| 4 | `feature/operations-tab` (D) | **Operations: the agent-authored Operation Plan** — typed declare tool, store, route, execution-linkage (tactic_id + updater), plan-item seeding, operation-spine renderer + live binding | C | 10 tasks + whole-branch (opus) review + 6 fixes; 105 tests; Chrome-audited |
| 5 | `feature/tactics-library` (G) | Save / list / apply reusable typed tactics (on D's substrate), mirroring plan-templates; apply→Operation Plan | D | 3 tasks + whole-branch review + fixes; ~64 tests |
| 6 | `feature/embryo-roles-observability` (D2) | Per-embryo **strain** field + roles-as-**use** (lineaging + subject/reference) + multi-embryo Operations roster lens (role + strain) | G | 4 tasks + whole-branch review + fix; 54 tests; Chrome-audited |
| 7 | `feature/session-plan-linking` (F) | Session↔**plans** link/delink: multi-plan model (`unlink_plan_item_session` + reverse-query), link/delink endpoints, Plans-tab controls + session Linked-plans panel | D2 | 4 tasks + whole-branch review + fix; ~70 tests; both surfaces Chrome-audited |
| 8 | `feature/manual-mode-dual-camera` (B2) | Dual-camera config + laser-preset browser + timelapse config form (extends B1 manual mode) | F | TBD (SDD) |

Notes:
- Each branch is the natural unit of "distinct enough to own a PR." Sub-parts (e.g. B1's safety fixes)
  stay inside their branch — granular enough to capture the work, not so granular it's noise.
- A/B1/C are **kept-as-is pending rig verification** (not merged); D/G/D2 build on top. The stack is
  intact but unmerged — PRs can open stacked and merge the chain once verified on the rig.
- "Easy to put together into a final product" = the linear stack already composes; the integration
  point is #58 → `development`.

## At wrap-up
1. Verify each branch's tests pass (the SDD ledgers + whole-branch reviews are the evidence trail).
2. Open the stacked PRs in order (each targets its parent branch), each description capturing its unit.
3. Rebase/integrate the chain onto #58 when rig-verified; #58 → `development` as the final integration.
