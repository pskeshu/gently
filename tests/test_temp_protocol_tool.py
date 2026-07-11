"""
Tests for the run_temp_change_burst_protocol agent tool (Task 5).

TDD: write failing tests first, then implement the tool.
"""

from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeClient:
    """Minimal fake microscope client."""

    async def set_temperature(self, t):
        return {"success": True, "temperature_c": t, "state": "[ HEATING ]"}

    async def get_temperature(self):
        return {"success": True, "temperature_c": 20.0, "state": "[ SYSTEM LOCKED ]"}


class FakeOrchestrator:
    """Minimal fake timelapse orchestrator with a client attribute."""

    def __init__(self, client):
        self.client = client

    def _emit_event(self, *args, **kwargs):
        pass


class FakeAgent:
    """Minimal fake agent — carries timelapse_orchestrator."""

    def __init__(self, orchestrator):
        self.timelapse_orchestrator = orchestrator


def _make_context(*, with_client=True, with_orchestrator=True):
    """Build a fake context dict."""
    client = FakeClient() if with_client else None
    orchestrator = FakeOrchestrator(client) if with_orchestrator else None
    agent = FakeAgent(orchestrator) if with_orchestrator else FakeAgent(None)
    return {"agent": agent, "client": client}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tool_creates_task_and_returns_started():
    """Happy path: context has agent+orchestrator+client → task is created, 'started' returned."""
    from gently.app.tools.temperature_protocol_tools import run_temp_change_burst_protocol_tool

    context = _make_context(with_client=True, with_orchestrator=True)
    created_tasks = []

    def fake_create_task(coro, **kwargs):
        # Cancel the coroutine immediately so there's no dangling task
        coro.close()
        mock_task = MagicMock()
        created_tasks.append(mock_task)
        return mock_task

    with patch("asyncio.create_task", side_effect=fake_create_task):
        result = await run_temp_change_burst_protocol_tool(
            embryo_id="emb1",
            target_setpoint_c=25.0,
            frames=30,
            bursts_before=1,
            bursts_after=1,
            context=context,
        )

    assert len(created_tasks) == 1, "Expected exactly one asyncio task to be created"
    assert "started" in result.lower(), f"Expected 'started' in result, got: {result!r}"
    assert "emb1" in result, f"Expected embryo_id in result, got: {result!r}"
    assert "25.0" in result or "25" in result, f"Expected setpoint in result, got: {result!r}"


@pytest.mark.asyncio
async def test_tool_no_client_returns_error_no_task():
    """No client in context → returns error string, no asyncio task created."""
    from gently.app.tools.temperature_protocol_tools import run_temp_change_burst_protocol_tool

    context = _make_context(with_client=False, with_orchestrator=True)
    created_tasks = []

    def fake_create_task(coro, **kwargs):
        coro.close()
        mock_task = MagicMock()
        created_tasks.append(mock_task)
        return mock_task

    with patch("asyncio.create_task", side_effect=fake_create_task):
        result = await run_temp_change_burst_protocol_tool(
            embryo_id="emb1",
            target_setpoint_c=25.0,
            context=context,
        )

    assert len(created_tasks) == 0, "No task should be created when client is absent"
    assert "error" in result.lower() or "not connected" in result.lower(), (
        f"Expected error message, got: {result!r}"
    )


@pytest.mark.asyncio
async def test_tool_no_orchestrator_returns_error_no_task():
    """No timelapse orchestrator → returns error string, no asyncio task created."""
    from gently.app.tools.temperature_protocol_tools import run_temp_change_burst_protocol_tool

    context = _make_context(with_client=True, with_orchestrator=False)
    created_tasks = []

    def fake_create_task(coro, **kwargs):
        coro.close()
        mock_task = MagicMock()
        created_tasks.append(mock_task)
        return mock_task

    with patch("asyncio.create_task", side_effect=fake_create_task):
        result = await run_temp_change_burst_protocol_tool(
            embryo_id="emb1",
            target_setpoint_c=25.0,
            context=context,
        )

    assert len(created_tasks) == 0, "No task should be created when orchestrator is absent"
    assert "error" in result.lower() or "not initialized" in result.lower(), (
        f"Expected error message, got: {result!r}"
    )


@pytest.mark.asyncio
async def test_tool_no_agent_context_returns_error():
    """No agent in context → returns error immediately."""
    from gently.app.tools.temperature_protocol_tools import run_temp_change_burst_protocol_tool

    context = {"client": FakeClient()}  # no "agent" key
    created_tasks = []

    def fake_create_task(coro, **kwargs):
        coro.close()
        mock_task = MagicMock()
        created_tasks.append(mock_task)
        return mock_task

    with patch("asyncio.create_task", side_effect=fake_create_task):
        result = await run_temp_change_burst_protocol_tool(
            embryo_id="emb1",
            target_setpoint_c=25.0,
            context=context,
        )

    assert len(created_tasks) == 0
    assert "error" in result.lower(), f"Expected error message, got: {result!r}"


def test_tool_is_registered():
    """The tool should be discoverable in the global tool registry after package import."""
    import gently.app.tools  # noqa: F401 — triggers registration
    from gently.harness.tools.registry import get_tool_registry

    registry = get_tool_registry()
    tool_names = [t.name for t in registry.list_all()]
    assert "run_temp_change_burst_protocol" in tool_names, (
        f"Tool not found in registry. Registered tools: {tool_names}"
    )


@pytest.mark.asyncio
async def test_tool_refuses_during_active_timelapse():
    """If orchestrator._status == RUNNING, refuse without creating a task."""
    from gently.app.orchestration.timelapse_models import TimelapseStatus
    from gently.app.tools.temperature_protocol_tools import run_temp_change_burst_protocol_tool

    context = _make_context(with_client=True, with_orchestrator=True)
    # Simulate an active timelapse
    context["agent"].timelapse_orchestrator._status = TimelapseStatus.RUNNING

    created_tasks = []

    def fake_create_task(coro, **kwargs):
        coro.close()
        mock_task = MagicMock()
        created_tasks.append(mock_task)
        return mock_task

    with patch("asyncio.create_task", side_effect=fake_create_task):
        result = await run_temp_change_burst_protocol_tool(
            embryo_id="emb1",
            target_setpoint_c=25.0,
            context=context,
        )

    assert len(created_tasks) == 0, "No task should be created when timelapse is running"
    assert "refusing" in result.lower(), f"Expected refusal message, got: {result!r}"
    assert "timelapse" in result.lower(), (
        f"Expected 'timelapse' in refusal message, got: {result!r}"
    )
