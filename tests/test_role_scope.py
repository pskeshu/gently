"""Tests for resolve_scope_embryos resolver + role-scoped tactic validation (Task 3, D2).

Resolver lives at gently.app.orchestration.role_scope.resolve_scope_embryos.

Contracts:
- mode='global' → all embryo ids
- mode='embryos' + embryo_ids → exactly those ids
- mode='role' + role='test' → ids of embryos with that role
- mode='role' with unknown role → [] (not an error)
- missing/unknown mode → []
- embryo dicts use 'embryo_id' key (from /api/embryos/positions)

Scope validation in declare_operation_plan:
- A tactic with scope.mode='role' and a valid REGISTRY key passes validation.
"""

import pytest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_EMBRYOS = [
    {"embryo_id": "e1", "role": "test"},
    {"embryo_id": "e2", "role": "test"},
    {"embryo_id": "e3", "role": "calibration"},
    {"embryo_id": "e4", "role": "lineaging"},
    {"embryo_id": "e5", "role": "unassigned"},
]


# ---------------------------------------------------------------------------
# Resolver tests
# ---------------------------------------------------------------------------


def test_global_scope_returns_all_ids():
    from gently.app.orchestration.role_scope import resolve_scope_embryos

    ids = resolve_scope_embryos({"mode": "global"}, _EMBRYOS)
    assert set(ids) == {"e1", "e2", "e3", "e4", "e5"}


def test_embryos_scope_returns_explicit_ids():
    from gently.app.orchestration.role_scope import resolve_scope_embryos

    ids = resolve_scope_embryos({"mode": "embryos", "embryo_ids": ["e1", "e3"]}, _EMBRYOS)
    assert ids == ["e1", "e3"]


def test_role_scope_filters_by_role():
    from gently.app.orchestration.role_scope import resolve_scope_embryos

    ids = resolve_scope_embryos({"mode": "role", "role": "test"}, _EMBRYOS)
    assert set(ids) == {"e1", "e2"}


def test_role_scope_calibration():
    from gently.app.orchestration.role_scope import resolve_scope_embryos

    ids = resolve_scope_embryos({"mode": "role", "role": "calibration"}, _EMBRYOS)
    assert ids == ["e3"]


def test_role_scope_lineaging():
    from gently.app.orchestration.role_scope import resolve_scope_embryos

    ids = resolve_scope_embryos({"mode": "role", "role": "lineaging"}, _EMBRYOS)
    assert ids == ["e4"]


def test_role_scope_unknown_role_returns_empty():
    from gently.app.orchestration.role_scope import resolve_scope_embryos

    ids = resolve_scope_embryos({"mode": "role", "role": "nonexistent_role"}, _EMBRYOS)
    assert ids == []


def test_unknown_mode_returns_empty():
    from gently.app.orchestration.role_scope import resolve_scope_embryos

    ids = resolve_scope_embryos({"mode": "burst_only"}, _EMBRYOS)
    assert ids == []


def test_missing_mode_returns_empty():
    from gently.app.orchestration.role_scope import resolve_scope_embryos

    ids = resolve_scope_embryos({}, _EMBRYOS)
    assert ids == []


def test_none_scope_returns_empty():
    from gently.app.orchestration.role_scope import resolve_scope_embryos

    ids = resolve_scope_embryos(None, _EMBRYOS)
    assert ids == []


def test_empty_embryo_list():
    from gently.app.orchestration.role_scope import resolve_scope_embryos

    ids = resolve_scope_embryos({"mode": "global"}, [])
    assert ids == []


def test_embryo_id_key_used_not_id():
    """Resolver must read 'embryo_id', not 'id'."""
    from gently.app.orchestration.role_scope import resolve_scope_embryos

    # If resolver accidentally uses 'id', this returns []
    embryos = [{"embryo_id": "eX", "role": "test"}]
    ids = resolve_scope_embryos({"mode": "global"}, embryos)
    assert "eX" in ids


# ---------------------------------------------------------------------------
# declare_operation_plan: role-scoped tactic passes validation
# ---------------------------------------------------------------------------


class _FakeContextStore:
    def __init__(self):
        self._plans = {}

    def set_operation_plan(self, sid, plan):
        self._plans[sid] = plan

    def get_operation_plan(self, sid):
        return self._plans.get(sid)


class _FakeAgent:
    def __init__(self):
        self.context_store = _FakeContextStore()
        self.session_id = "sess_scope_test"


@pytest.mark.asyncio
async def test_role_scoped_tactic_passes_declare_validation():
    """A tactic with scope.mode='role' and a valid role key is accepted by
    declare_operation_plan."""
    from gently.app.tools.operation_plan_tools import declare_operation_plan

    role_scoped_tactic = {
        "id": "t-role",
        "name": "Test-embryo timelapse",
        "kind": "standing_timelapse",
        "state": "active",
        "scope": {"mode": "role", "role": "test"},
        "rationale": "Only image the test embryos",
        "structure": {"cadence_s": 120, "per_embryo": []},
        "live_bind": ["cadence"],
        "relations": {},
    }

    ctx = {"agent": _FakeAgent()}
    result = await declare_operation_plan(
        title="Role-scoped plan",
        goal="Image test embryos only",
        tactics=[role_scoped_tactic],
        updated_reason="role scope test",
        context=ctx,
    )

    assert "Error" not in result, f"Unexpected error: {result}"
    stored = ctx["agent"].context_store.get_operation_plan("sess_scope_test")
    assert stored is not None
    tactic = stored["tactics"][0]
    assert tactic["scope"]["mode"] == "role"
    assert tactic["scope"]["role"] == "test"


@pytest.mark.asyncio
async def test_role_scoped_tactic_all_registry_roles():
    """All four REGISTRY role keys are accepted as valid scope.role values."""
    from gently.app.tools.operation_plan_tools import declare_operation_plan

    for role_name in ("test", "calibration", "lineaging", "unassigned"):
        tactic = {
            "id": f"t-{role_name}",
            "name": f"{role_name} monitor",
            "kind": "standing_timelapse",
            "state": "planned",
            "scope": {"mode": "role", "role": role_name},
            "rationale": f"Target {role_name} embryos",
            "structure": {"cadence_s": 300},
            "live_bind": [],
            "relations": {},
        }
        ctx = {"agent": _FakeAgent()}
        result = await declare_operation_plan(
            title=f"Plan for {role_name}",
            goal="Role-specific imaging",
            tactics=[tactic],
            context=ctx,
        )
        assert "Error" not in result, f"Role {role_name!r} was rejected: {result}"
