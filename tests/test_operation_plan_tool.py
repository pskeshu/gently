"""
Tests for the declare_operation_plan agent tool (Task 2).

TDD — three concerns:
  1. Happy path: calling the tool persists the plan; store returns it.
  2. Error paths: missing store, missing session → error string, no write.
  3. Registration: tool is discoverable in the global registry after import.
"""

import pytest


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeContextStore:
    """Minimal in-memory stand-in for FileContextStore."""

    def __init__(self):
        self._plans: dict[str, dict] = {}
        self.calls: list[tuple] = []

    def set_operation_plan(self, session_id: str, plan: dict) -> None:
        self._plans[session_id] = plan
        self.calls.append(("set", session_id, plan))

    def get_operation_plan(self, session_id: str) -> dict | None:
        return self._plans.get(session_id)


class FakeAgent:
    """Minimal fake agent — carries context_store and session_id."""

    def __init__(self, *, context_store=None, session_id: str | None = "sess_test_01"):
        self.context_store = context_store
        self.session_id = session_id


def _make_context(*, with_store=True, with_session=True):
    store = FakeContextStore() if with_store else None
    sid = "sess_test_01" if with_session else None
    agent = FakeAgent(context_store=store, session_id=sid)
    return {"agent": agent}


# ---------------------------------------------------------------------------
# Minimal tactic fixture
# ---------------------------------------------------------------------------

_TACTICS_MINIMAL = [
    {
        "id": "t1",
        "name": "Baseline timelapse",
        "kind": "standing_timelapse",
        "state": "active",
        "scope": {"mode": "global"},
        "rationale": "Continuous pre-ramp imaging",
        "structure": {"cadence_s": 120, "per_embryo": []},
        "live_bind": ["cadence"],
        "relations": {},
    }
]

_TACTICS_MULTI = [
    {
        "id": "t1",
        "name": "Baseline timelapse",
        "kind": "standing_timelapse",
        "state": "done",
        "scope": {"mode": "global"},
        "rationale": "Establish pre-ramp cadence",
        "structure": {"cadence_s": 120, "per_embryo": []},
        "live_bind": ["cadence"],
        "relations": {},
    },
    {
        "id": "t2",
        "name": "Onset monitor",
        "kind": "reactive_monitor",
        "state": "active",
        "scope": {"mode": "global"},
        "rationale": "Detect onset crossing threshold",
        "structure": {"watch": "gfp_signal > 0.3", "reaction": "burst_capture", "status": "armed"},
        "live_bind": ["signal", "current_burst"],
        "relations": {"after": ["t1"]},
    },
    {
        "id": "t3",
        "name": "Final survey",
        "kind": "oneshot",
        "state": "planned",
        "scope": {"mode": "embryos", "embryo_ids": ["e1", "e2"]},
        "rationale": "Confirm recovery after ramp",
        "structure": {"note": "single high-res z-stack"},
        "live_bind": [],
        "relations": {"after": ["t2"]},
    },
]


# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_plan_is_persisted_to_store():
    """Calling the tool with a valid plan stores it; get_operation_plan returns it."""
    from gently.app.tools.operation_plan_tools import declare_operation_plan

    ctx = _make_context()
    result = await declare_operation_plan(
        title="Expression-onset survey",
        goal="Catch GFP onset under 25 C ramp",
        tactics=_TACTICS_MINIMAL,
        updated_reason="experiment started",
        context=ctx,
    )

    store = ctx["agent"].context_store
    stored = store.get_operation_plan("sess_test_01")
    assert stored is not None, "Plan should have been persisted"
    assert stored["title"] == "Expression-onset survey"
    assert stored["goal"] == "Catch GFP onset under 25 C ramp"
    assert stored["session_id"] == "sess_test_01"
    assert "Error" not in result


@pytest.mark.asyncio
async def test_confirmation_string_contains_session_and_title():
    """The return value identifies the session and plan title."""
    from gently.app.tools.operation_plan_tools import declare_operation_plan

    ctx = _make_context()
    result = await declare_operation_plan(
        title="Temperature-strain protocol",
        goal="Characterise onset timing under 25 C ramp",
        tactics=_TACTICS_MINIMAL,
        context=ctx,
    )

    assert "sess_test_01" in result
    assert "Temperature-strain protocol" in result


@pytest.mark.asyncio
async def test_tactics_list_preserved_in_store():
    """All tactics are stored with correct ids, kinds, and states."""
    from gently.app.tools.operation_plan_tools import declare_operation_plan

    ctx = _make_context()
    await declare_operation_plan(
        title="Multi-tactic plan",
        goal="Three-phase experiment",
        tactics=_TACTICS_MULTI,
        updated_reason="t1 done, t2 active",
        context=ctx,
    )

    stored = ctx["agent"].context_store.get_operation_plan("sess_test_01")
    assert stored is not None
    tactics = stored["tactics"]
    assert len(tactics) == 3
    assert [t["id"] for t in tactics] == ["t1", "t2", "t3"]
    assert [t["state"] for t in tactics] == ["done", "active", "planned"]
    assert [t["kind"] for t in tactics] == ["standing_timelapse", "reactive_monitor", "oneshot"]


@pytest.mark.asyncio
async def test_updated_at_is_stamped():
    """The plan stored in the store has a non-empty updated_at timestamp."""
    from gently.app.tools.operation_plan_tools import declare_operation_plan

    ctx = _make_context()
    await declare_operation_plan(
        title="Any plan",
        goal="Any goal",
        tactics=_TACTICS_MINIMAL,
        context=ctx,
    )

    stored = ctx["agent"].context_store.get_operation_plan("sess_test_01")
    assert stored is not None
    assert stored.get("updated_at"), "updated_at should be stamped"


@pytest.mark.asyncio
async def test_updated_reason_stored():
    """updated_reason is carried into the stored plan."""
    from gently.app.tools.operation_plan_tools import declare_operation_plan

    ctx = _make_context()
    await declare_operation_plan(
        title="Any plan",
        goal="Any goal",
        tactics=_TACTICS_MINIMAL,
        updated_reason="tactic t1 transitioned to active",
        context=ctx,
    )

    stored = ctx["agent"].context_store.get_operation_plan("sess_test_01")
    assert stored["updated_reason"] == "tactic t1 transitioned to active"


@pytest.mark.asyncio
async def test_set_called_exactly_once():
    """set_operation_plan is called exactly once per tool invocation."""
    from gently.app.tools.operation_plan_tools import declare_operation_plan

    ctx = _make_context()
    await declare_operation_plan(
        title="Any plan",
        goal="Any goal",
        tactics=_TACTICS_MINIMAL,
        context=ctx,
    )

    store = ctx["agent"].context_store
    set_calls = [c for c in store.calls if c[0] == "set"]
    assert len(set_calls) == 1


@pytest.mark.asyncio
async def test_overwrite_replaces_plan():
    """A second call replaces the first plan in the store."""
    from gently.app.tools.operation_plan_tools import declare_operation_plan

    ctx = _make_context()
    await declare_operation_plan(
        title="Initial plan",
        goal="First framing",
        tactics=_TACTICS_MINIMAL,
        context=ctx,
    )
    await declare_operation_plan(
        title="Revised plan",
        goal="Updated framing",
        tactics=_TACTICS_MULTI,
        updated_reason="added two tactics",
        context=ctx,
    )

    stored = ctx["agent"].context_store.get_operation_plan("sess_test_01")
    assert stored["title"] == "Revised plan"
    assert len(stored["tactics"]) == 3


# ---------------------------------------------------------------------------
# Unknown kind → clamped to 'custom'
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_unknown_kind_clamped_to_custom():
    """A tactic with an unrecognised kind is silently clamped to 'custom'."""
    from gently.app.tools.operation_plan_tools import declare_operation_plan

    ctx = _make_context()
    tactics = [
        {
            "id": "t1",
            "name": "Novel tactic",
            "kind": "future_kind_v99",
            "state": "planned",
            "scope": {"mode": "global"},
            "rationale": "Something new",
            "structure": {"note": "TBD"},
            "live_bind": [],
            "relations": {},
        }
    ]
    result = await declare_operation_plan(
        title="Any plan",
        goal="Any goal",
        tactics=tactics,
        context=ctx,
    )

    assert "Error" not in result
    stored = ctx["agent"].context_store.get_operation_plan("sess_test_01")
    assert stored["tactics"][0]["kind"] == "custom"


# ---------------------------------------------------------------------------
# Error paths — no write must occur
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_no_agent_returns_error_no_write():
    """No agent in context → error string, store is never called."""
    from gently.app.tools.operation_plan_tools import declare_operation_plan

    # No agent key at all
    ctx: dict = {}
    result = await declare_operation_plan(
        title="Any plan",
        goal="Any goal",
        tactics=_TACTICS_MINIMAL,
        context=ctx,
    )

    assert "error" in result.lower() or "Error" in result


@pytest.mark.asyncio
async def test_missing_store_returns_error_no_write():
    """Agent has no context_store → error string, nothing persisted."""
    from gently.app.tools.operation_plan_tools import declare_operation_plan

    ctx = _make_context(with_store=False)
    result = await declare_operation_plan(
        title="Any plan",
        goal="Any goal",
        tactics=_TACTICS_MINIMAL,
        context=ctx,
    )

    assert "Error" in result
    assert ctx["agent"].context_store is None  # store was never created


@pytest.mark.asyncio
async def test_missing_session_id_returns_error_no_write():
    """Agent has no session_id → error string, store.set is never called."""
    from gently.app.tools.operation_plan_tools import declare_operation_plan

    ctx = _make_context(with_session=False)
    result = await declare_operation_plan(
        title="Any plan",
        goal="Any goal",
        tactics=_TACTICS_MINIMAL,
        context=ctx,
    )

    assert "Error" in result
    store = ctx["agent"].context_store
    assert store.calls == [], "set_operation_plan must not be called without a session_id"


# ---------------------------------------------------------------------------
# Validation errors — no write on invalid tactics
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_missing_tactic_id_returns_error_no_write():
    """A tactic missing 'id' causes an error; nothing is persisted."""
    from gently.app.tools.operation_plan_tools import declare_operation_plan

    ctx = _make_context()
    bad_tactics = [{"name": "No id", "kind": "oneshot", "state": "planned"}]
    result = await declare_operation_plan(
        title="Bad plan",
        goal="Any goal",
        tactics=bad_tactics,
        context=ctx,
    )

    assert "Error" in result
    assert ctx["agent"].context_store.calls == []


@pytest.mark.asyncio
async def test_invalid_state_returns_error_no_write():
    """A tactic with a completely unrecognised state causes an error; nothing is persisted."""
    from gently.app.tools.operation_plan_tools import declare_operation_plan

    ctx = _make_context()
    bad_tactics = [
        {
            "id": "t1",
            "name": "Bad state",
            "kind": "oneshot",
            "state": "completely_unknown_xyz",  # not a valid state and not a known synonym
        }
    ]
    result = await declare_operation_plan(
        title="Bad plan",
        goal="Any goal",
        tactics=bad_tactics,
        context=ctx,
    )

    assert "Error" in result
    assert ctx["agent"].context_store.calls == []


# ---------------------------------------------------------------------------
# Normalization tests (Fix 2)
# ---------------------------------------------------------------------------


def test_normalize_kind_synonyms():
    """Kind synonyms are mapped to canonical values before validation."""
    from gently.app.tools.operation_plan_tools import _validate_tactics

    cases = [
        ("monitor", "reactive_monitor"),
        ("reactive", "reactive_monitor"),
        ("timelapse", "standing_timelapse"),
        ("standing", "standing_timelapse"),
        ("protocol", "scripted_protocol"),
        ("scripted", "scripted_protocol"),
        ("burst", "exclusive_burst"),
        ("reference", "standing_timelapse"),
    ]
    for raw_kind, expected in cases:
        tactics = [{"id": "t1", "name": "T", "kind": raw_kind, "state": "planned"}]
        result = _validate_tactics(tactics)
        assert result[0]["kind"] == expected, f"kind '{raw_kind}' should map to '{expected}'"


def test_normalize_kind_unknown_to_custom():
    """An unrecognised kind that is not a synonym is clamped to 'custom'."""
    from gently.app.tools.operation_plan_tools import _validate_tactics

    tactics = [{"id": "t1", "name": "T", "kind": "future_kind_v99", "state": "planned"}]
    result = _validate_tactics(tactics)
    assert result[0]["kind"] == "custom"


def test_normalize_scope_bare_list():
    """A bare list scope is converted to {mode:'embryos', embryo_ids:[...]}."""
    from gently.app.tools.operation_plan_tools import _validate_tactics

    tactics = [
        {
            "id": "t1",
            "name": "T",
            "kind": "standing_timelapse",
            "state": "planned",
            "scope": ["E01", "E02"],
        }
    ]
    result = _validate_tactics(tactics)
    assert result[0]["scope"] == {"mode": "embryos", "embryo_ids": ["E01", "E02"]}


def test_normalize_scope_global_string():
    """The string 'global' scope is converted to {mode:'global'}."""
    from gently.app.tools.operation_plan_tools import _validate_tactics

    tactics = [
        {
            "id": "t1",
            "name": "T",
            "kind": "standing_timelapse",
            "state": "planned",
            "scope": "global",
        }
    ]
    result = _validate_tactics(tactics)
    assert result[0]["scope"] == {"mode": "global"}


def test_normalize_scope_dict_missing_mode_with_embryo_ids():
    """A scope dict with embryo_ids but no mode gets mode:'embryos' injected."""
    from gently.app.tools.operation_plan_tools import _validate_tactics

    tactics = [
        {
            "id": "t1",
            "name": "T",
            "kind": "standing_timelapse",
            "state": "planned",
            "scope": {"embryo_ids": ["E01"]},
        }
    ]
    result = _validate_tactics(tactics)
    assert result[0]["scope"]["mode"] == "embryos"
    assert result[0]["scope"]["embryo_ids"] == ["E01"]


def test_normalize_scope_dict_missing_mode_with_role():
    """A scope dict with role but no mode gets mode:'role' injected."""
    from gently.app.tools.operation_plan_tools import _validate_tactics

    tactics = [
        {
            "id": "t1",
            "name": "T",
            "kind": "standing_timelapse",
            "state": "planned",
            "scope": {"role": "test"},
        }
    ]
    result = _validate_tactics(tactics)
    assert result[0]["scope"]["mode"] == "role"
    assert result[0]["scope"]["role"] == "test"


def test_normalize_scope_well_formed_dict_unchanged():
    """A correctly formed scope dict passes through unchanged."""
    from gently.app.tools.operation_plan_tools import _validate_tactics

    scope = {"mode": "embryos", "embryo_ids": ["E01", "E02"]}
    tactics = [
        {
            "id": "t1",
            "name": "T",
            "kind": "standing_timelapse",
            "state": "planned",
            "scope": scope,
        }
    ]
    result = _validate_tactics(tactics)
    assert result[0]["scope"] == scope


def test_normalize_phase_state_pending_to_todo():
    """Phase state 'pending' is normalized to 'todo'."""
    from gently.app.tools.operation_plan_tools import _validate_tactics

    tactics = [
        {
            "id": "t1",
            "name": "T",
            "kind": "scripted_protocol",
            "state": "planned",
            "live": {
                "readouts": [],
                "phases": [
                    {"name": "ramp", "state": "pending", "count": 0, "pips": []},
                    {"name": "hold", "state": "running", "count": 0, "pips": []},
                    {"name": "done_phase", "state": "completed", "count": 3, "pips": []},
                ],
            },
        }
    ]
    result = _validate_tactics(tactics)
    phases = result[0]["live"]["phases"]
    assert phases[0]["state"] == "todo"
    assert phases[1]["state"] == "active"
    assert phases[2]["state"] == "done"


def test_normalize_tactic_state_synonyms():
    """Tactic state synonyms are mapped to canonical values."""
    from gently.app.tools.operation_plan_tools import _validate_tactics

    cases = [
        ("in_progress", "active"),
        ("running", "active"),
        ("pending", "planned"),
        ("queued", "planned"),
        ("complete", "done"),
        ("completed", "done"),
    ]
    for raw_state, expected in cases:
        tactics = [{"id": "t1", "name": "T", "kind": "oneshot", "state": "planned"}]
        # Override state directly
        tactics[0]["state"] = raw_state
        result = _validate_tactics(tactics)
        assert result[0]["state"] == expected, (
            f"state '{raw_state}' should map to '{expected}'"
        )


def test_normalize_full_scenario():
    """Combined: kind synonym + bare-list scope + pending phase state + running tactic state."""
    from gently.app.tools.operation_plan_tools import _validate_tactics

    tactics = [
        {
            "id": "t1",
            "name": "Temp monitor",
            "kind": "monitor",  # synonym → reactive_monitor
            "state": "in_progress",  # synonym → active
            "scope": ["E01", "E02"],  # bare list → {mode:embryos, embryo_ids:[...]}
            "rationale": "Watch temp",
            "live": {
                "readouts": [{"label": "temp", "value": "25 °C"}],
                "phases": [
                    {"name": "watch", "state": "pending", "count": 0, "pips": []},
                ],
            },
        }
    ]
    result = _validate_tactics(tactics)
    t = result[0]
    assert t["kind"] == "reactive_monitor"
    assert t["state"] == "active"
    assert t["scope"] == {"mode": "embryos", "embryo_ids": ["E01", "E02"]}
    assert t["live"]["phases"][0]["state"] == "todo"
    assert t["live"]["readouts"] == [{"label": "temp", "value": "25 °C"}]


# ---------------------------------------------------------------------------
# Registration test
# ---------------------------------------------------------------------------


def test_tool_is_registered_in_global_registry():
    """declare_operation_plan is discoverable via the global tool registry after import."""
    import gently.app.tools  # noqa: F401 — triggers registration side effects
    from gently.harness.tools.registry import get_tool_registry

    registry = get_tool_registry()
    names = [t.name for t in registry.list_all()]
    assert "declare_operation_plan" in names, (
        f"Tool not found in registry. Registered tools: {names}"
    )


def test_tool_schema_has_required_fields():
    """The auto-generated Claude schema has the expected required fields."""
    import gently.app.tools  # noqa: F401
    from gently.harness.tools.registry import get_tool_registry

    registry = get_tool_registry()
    tool_def = registry.get("declare_operation_plan")
    assert tool_def is not None

    schema = tool_def.to_claude_schema()
    assert schema["name"] == "declare_operation_plan"
    required = schema["input_schema"]["required"]
    assert "title" in required
    assert "goal" in required
    assert "tactics" in required
    # updated_reason has a default, so it must NOT be in required
    assert "updated_reason" not in required
