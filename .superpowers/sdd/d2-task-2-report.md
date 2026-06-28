# D2 Task 2 Report — roles-as-use: lineaging + subject/reference class

## What was done

### 1. `role_class` field added to `EmbryoRole`

Added `role_class: str = "subject"` as the last field in the frozen dataclass
(`gently/harness/roles.py`, after `no_object_consecutive_terminal`). Default is
`"subject"` so any existing keyword-constructed `EmbryoRole` is safe without
change. Docstring in the module header updated to document the two values
(`"subject"` | `"reference"`) and their intent.

### 2. `role_class` set on all three existing REGISTRY entries

| Role        | role_class  | Rationale                                      |
|-------------|-------------|------------------------------------------------|
| `test`      | `"subject"` | The biological subject — the precious sample.  |
| `unassigned`| `"subject"` | Safe default: protect like a subject until resolved. |
| `calibration`| `"reference"` | Reference embryo for staging/calibration.   |

### 3. `lineaging` entry added to REGISTRY

```python
"lineaging": EmbryoRole(
    name="lineaging",
    description=(
        "Lineage-tracing reference — tracks nuclei/divisions; often a "
        "pan-nuclear strain but the strain is separate from this use."
    ),
    default_cadence_seconds=300.0,
    detector_name="perception",        # nuclear pipeline, same as calibration
    photodose_budget_multiplier=5.0,
    ui_color="#33cc88",                # teal-green — distinct from cyan/magenta
    ui_icon="triangle",
    no_object_consecutive_terminal=2,  # reference: gone == gone
    role_class="reference",
)
```

Color distinctness verified: `#33cc88` (teal-green) ≠ `#00cccc` (calibration
cyan) ≠ `#ff66cc` (test magenta).

### 4. EmbryoRole construction-site check

```
grep -rn "EmbryoRole(" gently/
```

Result: only `gently/harness/roles.py` — the three (now four) REGISTRY entries,
all using keyword arguments. No positional construction anywhere. The new field's
default makes it safe for any future keyword construction too.

### 5. What was NOT touched (detector — staged)

- No changes to detector/perception wiring, session logic, cadence policy,
  or any consumer of `EmbryoRole` fields.
- `detector_name="perception"` on `lineaging` mirrors `calibration` — the
  staged approach defers any strain→detector decoupling to spec §4 future work.
- `DEFAULT_ROLE`, `get_role`, `is_valid_role`, `list_roles` — all unchanged
  (they work generically over REGISTRY).

## TDD evidence

File: `tests/test_roles_registry.py` (created, 22 tests)

```
tests/test_roles_registry.py::test_test_role_class_is_subject PASSED
tests/test_roles_registry.py::test_unassigned_role_class_is_subject PASSED
tests/test_roles_registry.py::test_calibration_role_class_is_reference PASSED
tests/test_roles_registry.py::test_lineaging_role_exists_in_registry PASSED
tests/test_roles_registry.py::test_lineaging_role_class_is_reference PASSED
tests/test_roles_registry.py::test_lineaging_ui_color PASSED
tests/test_roles_registry.py::test_lineaging_ui_icon PASSED
tests/test_roles_registry.py::test_lineaging_description_contains_lineage PASSED
tests/test_roles_registry.py::test_lineaging_has_cadence PASSED
tests/test_roles_registry.py::test_get_role_lineaging_returns_correct_entry PASSED
tests/test_roles_registry.py::test_is_valid_role_lineaging_true PASSED
tests/test_roles_registry.py::test_is_valid_role_unknown_false PASSED
tests/test_roles_registry.py::test_list_roles_includes_lineaging PASSED
tests/test_roles_registry.py::test_list_roles_sorted PASSED
tests/test_roles_registry.py::test_get_role_unknown_raises_key_error PASSED
tests/test_roles_registry.py::test_test_role_fields_unchanged PASSED
tests/test_roles_registry.py::test_calibration_role_fields_unchanged PASSED
tests/test_roles_registry.py::test_unassigned_role_fields_unchanged PASSED
tests/test_roles_registry.py::test_default_role_is_in_registry PASSED
tests/test_roles_registry.py::test_default_role_is_test PASSED
tests/test_roles_registry.py::test_embryo_role_is_frozen PASSED
tests/test_roles_registry.py::test_embryo_role_default_role_class PASSED
22 passed in 0.12s
```

Full suite (excluding 3 pre-existing collection errors):
- Before task-2 changes: 41 pre-existing failures + 4 from this test file against old code = 45 total
- After task-2 changes: 32 failures, 765 passed — all 32 failures pre-existing (test_eval, test_mesh_service, test_prompt_manager and others), zero new failures introduced.

## Files changed

- `gently/harness/roles.py` — `role_class` field on dataclass + all 4 registry entries
- `tests/test_roles_registry.py` — new, 22 tests

## Concerns

None. The change is additive and isolated to `roles.py`. The frozen dataclass
with a defaulted field is backward-compatible. The `lineaging` detector choice
(`"perception"`) is intentionally the same as calibration — decoupling
detector-from-use is explicitly deferred to spec §4.
