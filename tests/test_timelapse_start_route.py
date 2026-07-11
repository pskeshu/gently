"""Tests for B2 Task 3 — POST /api/devices/timelapse/start (manual timelapse proxy).

Verifies:
  - POST with valid params calls orchestrator.start with the right args → 200
  - monitoring_mode is forwarded to orchestrator.enable_monitoring_mode
  - interval_seconds <= 0 → 400
  - num_slices < 1 → 400
  - missing interval_seconds uses default 120.0 (no error)
  - orchestrator not initialised → 503
  - require_control gate (403 without override)

Orchestrator access path:
  server.agent_bridge.agent.timelapse_orchestrator
  (mirroring the `require_timelapse_orchestrator(agent)` helper in harness/tools/helpers.py)
"""

from unittest.mock import AsyncMock, MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

import gently.ui.web.auth as auth
from gently.ui.web.routes.data import create_router

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _app(orchestrator=None):
    """Build a TestClient wired with the data routes.

    The mock server exposes `agent_bridge.agent.timelapse_orchestrator`.
    `require_control` is overridden so route logic is isolated from auth.
    """
    server = MagicMock()
    server.agent_bridge.agent.timelapse_orchestrator = orchestrator
    # Satisfy other routes that call _resolve_client() or store
    server.agent_bridge.agent.client = MagicMock()
    server.agent_bridge.agent.lightsheet_monitor = None
    app = FastAPI()
    app.include_router(create_router(server))
    app.dependency_overrides[auth.require_control] = lambda: True
    return TestClient(app)


def _make_orchestrator(start_return="Timelapse started."):
    """Return a mock orchestrator with an async start and sync enable_monitoring_mode."""
    orch = MagicMock()
    orch.start = AsyncMock(return_value=start_return)
    orch.enable_monitoring_mode = MagicMock(return_value="Monitoring mode enabled.")
    return orch


# ---------------------------------------------------------------------------
# Happy path — minimal valid payload
# ---------------------------------------------------------------------------


def test_timelapse_start_minimal():
    """POST with just interval_seconds calls orchestrator.start; returns 200 with started=True."""
    orch = _make_orchestrator()
    r = _app(orch).post(
        "/api/devices/timelapse/start",
        json={"interval_seconds": 120},
    )
    assert r.status_code == 200
    body = r.json()
    assert body["started"] is True
    orch.start.assert_awaited_once()


def test_timelapse_start_calls_start_with_correct_args():
    """orchestrator.start receives the right interval, stop_condition, embryo_ids,
    condition_value."""
    orch = _make_orchestrator()
    _app(orch).post(
        "/api/devices/timelapse/start",
        json={
            "interval_seconds": 60,
            "stop_condition": "timepoints",
            "embryo_ids": ["e1", "e2"],
            "condition_value": 10,
        },
    )
    orch.start.assert_awaited_once_with(
        embryo_ids=["e1", "e2"],
        stop_condition="timepoints",
        base_interval_seconds=60.0,
        condition_value=10,
    )


def test_timelapse_start_uses_default_interval():
    """Omitting interval_seconds uses the default (120.0) without error."""
    orch = _make_orchestrator()
    r = _app(orch).post(
        "/api/devices/timelapse/start",
        json={},
    )
    assert r.status_code == 200
    _, kwargs = orch.start.call_args
    assert kwargs["base_interval_seconds"] == 120.0


def test_timelapse_start_result_in_response():
    """The orchestrator.start return value appears in the response."""
    orch = _make_orchestrator(start_return="Timelapse running — 3 embryos.")
    r = _app(orch).post(
        "/api/devices/timelapse/start",
        json={"interval_seconds": 90},
    )
    assert r.json()["result"] == "Timelapse running — 3 embryos."


# ---------------------------------------------------------------------------
# Monitoring mode
# ---------------------------------------------------------------------------


def test_timelapse_start_enables_monitoring_mode():
    """monitoring_mode != 'idle' triggers enable_monitoring_mode on the orchestrator."""
    orch = _make_orchestrator()
    r = _app(orch).post(
        "/api/devices/timelapse/start",
        json={"interval_seconds": 120, "monitoring_mode": "expression_monitoring"},
    )
    assert r.status_code == 200
    orch.enable_monitoring_mode.assert_called_once_with("expression_monitoring")
    assert r.json()["monitoring_mode_result"] == "Monitoring mode enabled."


def test_timelapse_start_idle_mode_skips_enable():
    """monitoring_mode='idle' does NOT call enable_monitoring_mode."""
    orch = _make_orchestrator()
    r = _app(orch).post(
        "/api/devices/timelapse/start",
        json={"interval_seconds": 120, "monitoring_mode": "idle"},
    )
    assert r.status_code == 200
    orch.enable_monitoring_mode.assert_not_called()


def test_timelapse_start_no_mode_skips_enable():
    """Omitting monitoring_mode does NOT call enable_monitoring_mode."""
    orch = _make_orchestrator()
    _app(orch).post("/api/devices/timelapse/start", json={"interval_seconds": 120})
    orch.enable_monitoring_mode.assert_not_called()


# ---------------------------------------------------------------------------
# Volume geometry passed through in response config
# ---------------------------------------------------------------------------


def test_timelapse_start_volume_geometry_in_config():
    """Volume geometry fields appear in response['config']['volume_geometry']."""
    orch = _make_orchestrator()
    r = _app(orch).post(
        "/api/devices/timelapse/start",
        json={
            "interval_seconds": 120,
            "num_slices": 80,
            "exposure_ms": 15.0,
            "galvo_amplitude": 0.7,
            "galvo_center": 0.1,
            "piezo_amplitude": 30.0,
            "piezo_center": 55.0,
            "laser_config": "488 only",
        },
    )
    assert r.status_code == 200
    vg = r.json()["config"]["volume_geometry"]
    assert vg["num_slices"] == 80
    assert vg["exposure_ms"] == 15.0
    assert vg["laser_config"] == "488 only"


# ---------------------------------------------------------------------------
# 400 — validation failures
# ---------------------------------------------------------------------------


def test_timelapse_start_interval_zero_returns_400():
    """interval_seconds = 0 → 400."""
    orch = _make_orchestrator()
    r = _app(orch).post(
        "/api/devices/timelapse/start",
        json={"interval_seconds": 0},
    )
    assert r.status_code == 400
    assert "interval_seconds" in r.json()["detail"]


def test_timelapse_start_negative_interval_returns_400():
    """interval_seconds < 0 → 400."""
    orch = _make_orchestrator()
    r = _app(orch).post(
        "/api/devices/timelapse/start",
        json={"interval_seconds": -10},
    )
    assert r.status_code == 400


def test_timelapse_start_num_slices_zero_returns_400():
    """num_slices = 0 → 400."""
    orch = _make_orchestrator()
    r = _app(orch).post(
        "/api/devices/timelapse/start",
        json={"interval_seconds": 120, "num_slices": 0},
    )
    assert r.status_code == 400
    assert "num_slices" in r.json()["detail"]


def test_timelapse_start_num_slices_negative_returns_400():
    """num_slices < 0 → 400."""
    orch = _make_orchestrator()
    r = _app(orch).post(
        "/api/devices/timelapse/start",
        json={"interval_seconds": 120, "num_slices": -5},
    )
    assert r.status_code == 400


def test_timelapse_start_non_numeric_interval_returns_400():
    """Non-numeric interval_seconds → 400."""
    orch = _make_orchestrator()
    r = _app(orch).post(
        "/api/devices/timelapse/start",
        json={"interval_seconds": "fast"},
    )
    assert r.status_code == 400


# ---------------------------------------------------------------------------
# 503 — orchestrator not reachable
# ---------------------------------------------------------------------------


def test_timelapse_start_no_orchestrator_returns_503():
    """orchestrator is None (agent not running / no session) → 503."""
    r = _app(orchestrator=None).post(
        "/api/devices/timelapse/start",
        json={"interval_seconds": 120},
    )
    assert r.status_code == 503


def test_timelapse_start_no_agent_bridge_returns_503():
    """agent_bridge missing entirely → 503."""
    server = MagicMock(spec=[])  # no agent_bridge attribute
    app = FastAPI()
    app.include_router(create_router(server))
    app.dependency_overrides[auth.require_control] = lambda: True
    r = TestClient(app).post(
        "/api/devices/timelapse/start",
        json={"interval_seconds": 120},
    )
    assert r.status_code == 503


# ---------------------------------------------------------------------------
# require_control gate
# ---------------------------------------------------------------------------


def test_timelapse_start_requires_control():
    """POST /api/devices/timelapse/start is gated by require_control (403 without override)."""
    orch = _make_orchestrator()
    server = MagicMock()
    server.agent_bridge.agent.timelapse_orchestrator = orch
    server.agent_bridge.agent.client = MagicMock()
    server.agent_bridge.agent.lightsheet_monitor = None
    app = FastAPI()
    app.include_router(create_router(server))
    # No dependency override — require_control will reject non-loopback hosts
    r = TestClient(app, raise_server_exceptions=False).post(
        "/api/devices/timelapse/start",
        json={"interval_seconds": 120},
    )
    assert r.status_code == 403
