"""
TDD tests for gently/app/tools/tactic_library_tools.py

Tests:
- save_tactic: calls store.save_tactic and returns a confirmation with the id
- list_tactics: returns readable summary from store.list_tactics
- apply_tactic: fetches template, appends fresh planned tactic to Operation Plan
- apply_tactic unknown id: returns error, does NOT call set_operation_plan
- missing context_store: returns error for all three tools
- tools are importable and registered in the tool registry
"""

import copy

import pytest

# ---------------------------------------------------------------------------
# Shared tactic fixture
# ---------------------------------------------------------------------------

_TACTIC = {
    "id": "t_original",
    "name": "Baseline timelapse",
    "kind": "standing_timelapse",
    "state": "active",
    "scope": {"mode": "global"},
    "rationale": "Continuous pre-ramp imaging",
    "structure": {"cadence_s": 120, "per_embryo": []},
    "live_bind": ["cadence"],
    "relations": {},
}


def _fresh_tactic():
    return copy.deepcopy(_TACTIC)


# ---------------------------------------------------------------------------
# Fake context store
# ---------------------------------------------------------------------------


class FakeContextStore:
    """Minimal context store double for tool tests."""

    def __init__(self):
        self._library: dict[str, dict] = {}  # id -> template
        self._plans: dict[str, dict] = {}  # session_id -> plan
        self.save_tactic_calls: list = []
        self.set_operation_plan_calls: list = []

    def save_tactic(self, tactic: dict, name: str | None = None) -> str:
        tid = f"tpl_{len(self._library) + 1}"
        name = name or tactic.get("name") or "unnamed"
        template = {
            "id": tid,
            "name": name,
            "kind": tactic.get("kind", "unknown"),
            "description": tactic.get("description") or tactic.get("rationale"),
        }
        self._library[tid] = template
        self.save_tactic_calls.append({"tactic": tactic, "name": name, "returned_id": tid})
        return tid

    def list_tactics(self) -> list[dict]:
        return list(self._library.values())

    def get_tactic(self, id_or_name: str) -> dict | None:
        for t in self._library.values():
            if t.get("id") == id_or_name or t.get("name") == id_or_name:
                return copy.deepcopy(t)
        return None

    def apply_tactic(self, id_or_name: str) -> dict | None:
        tmpl = self.get_tactic(id_or_name)
        if tmpl is None:
            return None
        # Return a fresh planned tactic with a new id
        fresh = copy.deepcopy(tmpl)
        fresh["id"] = f"run_{id_or_name}"
        fresh["state"] = "planned"
        return fresh

    def get_operation_plan(self, session_id: str) -> dict | None:
        return copy.deepcopy(self._plans.get(session_id))

    def set_operation_plan(self, session_id: str, plan: dict) -> None:
        self._plans[session_id] = copy.deepcopy(plan)
        self.set_operation_plan_calls.append(
            {"session_id": session_id, "plan": copy.deepcopy(plan)}
        )


# ---------------------------------------------------------------------------
# Fake agent
# ---------------------------------------------------------------------------


class FakeAgent:
    def __init__(self, session_id="sess_001", context_store=None):
        self.session_id = session_id
        self.context_store = context_store or FakeContextStore()


def _make_context(agent):
    return {"agent": agent}


# ---------------------------------------------------------------------------
# Import the tools under test
# ---------------------------------------------------------------------------


from gently.app.tools.tactic_library_tools import (  # noqa: E402
    apply_tactic,
    list_tactics,
    save_tactic,
)

# ---------------------------------------------------------------------------
# save_tactic tests
# ---------------------------------------------------------------------------


class TestSaveTactic:
    @pytest.mark.asyncio
    async def test_calls_store_save_tactic(self):
        """save_tactic must call context_store.save_tactic once."""
        cs = FakeContextStore()
        agent = FakeAgent(context_store=cs)
        await save_tactic(
            name="Baseline timelapse",
            tactic=_fresh_tactic(),
            context=_make_context(agent),
        )
        assert len(cs.save_tactic_calls) == 1

    @pytest.mark.asyncio
    async def test_passes_name_through(self):
        """The name passed to the tool must reach store.save_tactic."""
        cs = FakeContextStore()
        agent = FakeAgent(context_store=cs)
        await save_tactic(
            name="My custom tactic",
            tactic=_fresh_tactic(),
            context=_make_context(agent),
        )
        call = cs.save_tactic_calls[0]
        assert call["name"] == "My custom tactic"

    @pytest.mark.asyncio
    async def test_returns_confirmation_with_id(self):
        """Return value must contain the new template id."""
        cs = FakeContextStore()
        agent = FakeAgent(context_store=cs)
        result = await save_tactic(
            name="Baseline timelapse",
            tactic=_fresh_tactic(),
            context=_make_context(agent),
        )
        # The id returned by the fake store is "tpl_1"
        assert "tpl_1" in result

    @pytest.mark.asyncio
    async def test_description_merged_into_tactic(self):
        """If description is provided it should be in the tactic passed to the store."""
        cs = FakeContextStore()
        agent = FakeAgent(context_store=cs)
        await save_tactic(
            name="Baseline timelapse",
            tactic=_fresh_tactic(),
            description="Override description",
            context=_make_context(agent),
        )
        saved = cs.save_tactic_calls[0]["tactic"]
        assert saved.get("description") == "Override description"

    @pytest.mark.asyncio
    async def test_missing_context_store_returns_error(self):
        """If context_store is None, return an error string."""
        agent = FakeAgent(context_store=None)
        agent.context_store = None
        result = await save_tactic(
            name="x",
            tactic=_fresh_tactic(),
            context=_make_context(agent),
        )
        assert result.startswith("Error:")

    @pytest.mark.asyncio
    async def test_missing_agent_returns_error(self):
        result = await save_tactic(name="x", tactic=_fresh_tactic(), context={})
        assert result.startswith("Error:")


# ---------------------------------------------------------------------------
# list_tactics tests
# ---------------------------------------------------------------------------


class TestListTactics:
    @pytest.mark.asyncio
    async def test_returns_readable_summary(self):
        """list_tactics should include id, name, kind in the output."""
        cs = FakeContextStore()
        agent = FakeAgent(context_store=cs)
        # Pre-populate library
        cs.save_tactic(_fresh_tactic(), name="Baseline timelapse")
        result = await list_tactics(context=_make_context(agent))
        assert "tpl_1" in result
        assert "Baseline timelapse" in result
        assert "standing_timelapse" in result

    @pytest.mark.asyncio
    async def test_empty_library_message(self):
        cs = FakeContextStore()
        agent = FakeAgent(context_store=cs)
        result = await list_tactics(context=_make_context(agent))
        assert "empty" in result.lower() or "no saved" in result.lower()

    @pytest.mark.asyncio
    async def test_missing_context_store_returns_error(self):
        agent = FakeAgent(context_store=None)
        agent.context_store = None
        result = await list_tactics(context=_make_context(agent))
        assert result.startswith("Error:")

    @pytest.mark.asyncio
    async def test_missing_agent_returns_error(self):
        result = await list_tactics(context={})
        assert result.startswith("Error:")


# ---------------------------------------------------------------------------
# apply_tactic tests
# ---------------------------------------------------------------------------


class TestApplyTactic:
    @pytest.mark.asyncio
    async def test_appends_tactic_to_existing_plan(self):
        """apply_tactic must append the fresh tactic to the session's Operation Plan."""
        cs = FakeContextStore()
        agent = FakeAgent(context_store=cs)
        # Seed library + existing plan
        cs.save_tactic(_fresh_tactic(), name="Baseline timelapse")
        cs._plans["sess_001"] = {
            "session_id": "sess_001",
            "title": "Test plan",
            "goal": "Test goal",
            "tactics": [],
        }

        await apply_tactic(id_or_name="Baseline timelapse", context=_make_context(agent))

        assert len(cs.set_operation_plan_calls) == 1
        saved_plan = cs.set_operation_plan_calls[0]["plan"]
        assert len(saved_plan["tactics"]) == 1
        assert saved_plan["tactics"][0]["state"] == "planned"

    @pytest.mark.asyncio
    async def test_creates_minimal_plan_if_none_exists(self):
        """If no plan exists yet, apply_tactic creates a minimal one."""
        cs = FakeContextStore()
        agent = FakeAgent(context_store=cs)
        cs.save_tactic(_fresh_tactic(), name="Baseline timelapse")

        await apply_tactic(id_or_name="Baseline timelapse", context=_make_context(agent))

        assert len(cs.set_operation_plan_calls) == 1
        plan = cs.set_operation_plan_calls[0]["plan"]
        assert plan["session_id"] == "sess_001"
        assert len(plan["tactics"]) == 1

    @pytest.mark.asyncio
    async def test_appends_not_replaces(self):
        """apply_tactic appends; an existing tactic in the plan must remain."""
        cs = FakeContextStore()
        agent = FakeAgent(context_store=cs)
        cs.save_tactic(_fresh_tactic(), name="Baseline timelapse")
        cs._plans["sess_001"] = {
            "session_id": "sess_001",
            "title": "",
            "goal": "",
            "tactics": [
                {"id": "existing_t", "name": "Prior tactic", "kind": "oneshot", "state": "done"}
            ],
        }

        await apply_tactic(id_or_name="Baseline timelapse", context=_make_context(agent))

        plan = cs.set_operation_plan_calls[0]["plan"]
        assert len(plan["tactics"]) == 2
        ids = [t["id"] for t in plan["tactics"]]
        assert "existing_t" in ids

    @pytest.mark.asyncio
    async def test_unknown_id_returns_error_no_plan_write(self):
        """Unknown tactic id → error string, set_operation_plan must NOT be called."""
        cs = FakeContextStore()
        agent = FakeAgent(context_store=cs)

        result = await apply_tactic(id_or_name="nonexistent", context=_make_context(agent))

        assert result.startswith("Error:")
        assert len(cs.set_operation_plan_calls) == 0

    @pytest.mark.asyncio
    async def test_missing_context_store_returns_error(self):
        agent = FakeAgent(context_store=None)
        agent.context_store = None
        result = await apply_tactic(id_or_name="anything", context=_make_context(agent))
        assert result.startswith("Error:")

    @pytest.mark.asyncio
    async def test_missing_session_id_returns_error(self):
        cs = FakeContextStore()
        agent = FakeAgent(context_store=cs)
        agent.session_id = None
        result = await apply_tactic(id_or_name="anything", context=_make_context(agent))
        assert result.startswith("Error:")

    @pytest.mark.asyncio
    async def test_missing_agent_returns_error(self):
        result = await apply_tactic(id_or_name="anything", context={})
        assert result.startswith("Error:")

    @pytest.mark.asyncio
    async def test_return_value_names_tactic_and_session(self):
        """Confirmation message should mention the tactic name and session id."""
        cs = FakeContextStore()
        agent = FakeAgent(context_store=cs)
        cs.save_tactic(_fresh_tactic(), name="Baseline timelapse")

        result = await apply_tactic(id_or_name="Baseline timelapse", context=_make_context(agent))

        assert "Baseline timelapse" in result
        assert "sess_001" in result


# ---------------------------------------------------------------------------
# Registration smoke test
# ---------------------------------------------------------------------------


class TestRegistration:
    def test_tools_are_importable(self):
        """The three tools must import without error."""
        from gently.app.tools.tactic_library_tools import (
            apply_tactic,
            list_tactics,
            save_tactic,
        )

        assert callable(save_tactic)
        assert callable(list_tactics)
        assert callable(apply_tactic)

    def test_module_in_tools_package(self):
        """tactic_library_tools must be in tools.__init__ exports."""
        import gently.app.tools as tools_pkg

        assert hasattr(tools_pkg, "tactic_library_tools")

    def test_tools_registered_in_registry(self):
        """The three tools must appear by name in the tool registry."""
        import gently.app.tools  # noqa: F401 — ensure registration side-effect
        from gently.harness.tools.registry import get_tool_registry

        registry = get_tool_registry()
        names = {entry.name for entry in registry.list_all()}
        assert "save_tactic" in names, f"save_tactic not in registry: {names}"
        assert "list_tactics" in names, f"list_tactics not in registry: {names}"
        assert "apply_tactic" in names, f"apply_tactic not in registry: {names}"
