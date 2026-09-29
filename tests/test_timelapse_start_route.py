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
    # The start route sets the run's laser preset on the controller when the
    # plan names one; the fake has to be awaitable for that.
    server.agent_bridge.agent.client.set_laser_config = AsyncMock(return_value={"success": True})
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


# ---------------------------------------------------------------------------
# The plan's two new axes: the DIC channel, and per-embryo termination
# ---------------------------------------------------------------------------


def test_a_plan_without_dic_calls_start_exactly_as_before():
    """A run with no DIC channel must not grow a `dic=` kwarg — the old
    assertions above are the contract, and this keeps them honest."""
    orch = _make_orchestrator()
    _app(orch).post("/api/devices/timelapse/start", json={"interval_seconds": 60})
    kwargs = orch.start.await_args.kwargs
    assert "dic" not in kwargs and "stop_conditions" not in kwargs


def test_dic_channel_is_forwarded_validated():
    orch = _make_orchestrator()
    r = _app(orch).post(
        "/api/devices/timelapse/start",
        json={
            "interval_seconds": 300,
            "dic": {
                "enabled": True,
                "every_seconds": 600,
                "position": {"x": -500, "y": -400},
                "exposure_ms": 8,
            },
        },
    )
    assert r.status_code == 200, r.text
    dic = orch.start.await_args.kwargs["dic"]
    assert dic == {
        "enabled": True,
        "every_seconds": 600.0,
        "position": {"x": -500.0, "y": -400.0},
        "exposure_ms": 8.0,
        "light": "room",
    }
    assert r.json()["dic"] == dic


def test_dic_disabled_is_the_same_as_absent():
    orch = _make_orchestrator()
    _app(orch).post(
        "/api/devices/timelapse/start", json={"dic": {"enabled": False, "every_seconds": 5}}
    )
    assert "dic" not in orch.start.await_args.kwargs


def test_dic_interval_must_be_positive():
    orch = _make_orchestrator()
    r = _app(orch).post(
        "/api/devices/timelapse/start", json={"dic": {"enabled": True, "every_seconds": 0}}
    )
    assert r.status_code == 400
    assert "every_seconds" in r.json()["detail"]
    orch.start.assert_not_awaited()


def test_dic_position_must_be_xy():
    orch = _make_orchestrator()
    r = _app(orch).post(
        "/api/devices/timelapse/start", json={"dic": {"enabled": True, "position": {"x": 1}}}
    )
    assert r.status_code == 400
    orch.start.assert_not_awaited()


def test_per_embryo_stop_conditions_are_forwarded():
    orch = _make_orchestrator()
    r = _app(orch).post(
        "/api/devices/timelapse/start",
        json={
            "stop_condition": "duration:12h",
            "stop_conditions": {
                "embryo_2": "hatching",
                "embryo_3": {"stop_condition": "timepoints", "condition_value": 3},
            },
        },
    )
    assert r.status_code == 200, r.text
    assert orch.start.await_args.kwargs["stop_conditions"] == {
        "embryo_2": "hatching",
        "embryo_3": {"stop_condition": "timepoints", "condition_value": 3},
    }


def test_a_malformed_stop_override_is_refused():
    orch = _make_orchestrator()
    r = _app(orch).post("/api/devices/timelapse/start", json={"stop_conditions": {"embryo_2": 7}})
    assert r.status_code == 400
    orch.start.assert_not_awaited()


# ---------------------------------------------------------------------------
# The run surface: status, and per-embryo control
# ---------------------------------------------------------------------------


def _running(orch):
    from datetime import datetime
    from types import SimpleNamespace

    sc = SimpleNamespace(describe=lambda: "timepoints:12")
    e1 = SimpleNamespace(
        timepoints_acquired=4,
        interval_seconds=300.0,
        cadence_phase="normal",
        next_due_at=datetime(2026, 9, 25, 12, 0, 0),
        is_complete=False,
        completion_reason=None,
        should_skip=False,
        stop_condition=sc,
        role="test",
        last_error=None,
    )
    state = SimpleNamespace(
        embryos={"embryo_1": e1}, to_dict=lambda: {"status": "running", "dic": {"frames": 2}}
    )
    orch.get_status = MagicMock(return_value=state)
    orch.stop_embryo = AsyncMock(return_value="Stopped imaging embryo_1 (reason: user_request)")
    orch.modify_embryo = AsyncMock(return_value="Modified embryo_1")
    return orch


def test_status_gives_the_pane_its_embryo_rows():
    orch = _running(_make_orchestrator())
    r = _app(orch).get("/api/devices/timelapse/status")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "running" and body["dic"] == {"frames": 2}
    row = body["embryos"]["embryo_1"]
    assert row["timepoints"] == 4
    assert row["stop_condition"] == "timepoints:12"
    assert row["next_due_at"].startswith("2026-09-25T12:00:00")


def test_status_without_an_orchestrator_is_503():
    r = _app(None).get("/api/devices/timelapse/status")
    assert r.status_code == 503


def test_one_embryo_can_be_stopped_while_the_run_carries_on():
    orch = _running(_make_orchestrator())
    r = _app(orch).post(
        "/api/devices/timelapse/embryo/embryo_1/stop", json={"reason": "done with it"}
    )
    assert r.status_code == 200, r.text
    orch.stop_embryo.assert_awaited_once_with("embryo_1", reason="done with it")


def test_one_embryos_termination_can_change_mid_run():
    orch = _running(_make_orchestrator())
    r = _app(orch).post(
        "/api/devices/timelapse/embryo/embryo_1/modify",
        json={"stop_condition": "timepoints", "condition_value": 20},
    )
    assert r.status_code == 200, r.text
    orch.modify_embryo.assert_awaited_once_with(
        "embryo_1", stop_condition="timepoints", condition_value=20
    )


def test_modify_needs_a_stop_condition():
    orch = _running(_make_orchestrator())
    r = _app(orch).post("/api/devices/timelapse/embryo/embryo_1/modify", json={})
    assert r.status_code == 400
    orch.modify_embryo.assert_not_awaited()


# ---------------------------------------------------------------------------
# The laser preset is set for the run, not merely recorded
# ---------------------------------------------------------------------------


def _app_with_client(orch, set_laser_config):
    """Like _app, but hands back the controller mock so the preset call can be asserted."""
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    server = MagicMock()
    server.agent_bridge.agent.timelapse_orchestrator = orch
    server.agent_bridge.agent.client = MagicMock()
    server.agent_bridge.agent.client.set_laser_config = set_laser_config
    server.agent_bridge.agent.lightsheet_monitor = None
    app = FastAPI()
    app.include_router(create_router(server))
    app.dependency_overrides[auth.require_control] = lambda: True
    return TestClient(app), server.agent_bridge.agent.client


def test_the_plans_laser_preset_is_set_on_the_controller_before_the_run():
    orch = _make_orchestrator()
    client_app, client = _app_with_client(orch, AsyncMock(return_value={"success": True}))
    r = client_app.post("/api/devices/timelapse/start", json={"laser_config": "488 and 561"})
    assert r.status_code == 200, r.text
    client.set_laser_config.assert_awaited_once_with("488 and 561")
    orch.start.assert_awaited_once()


def test_no_preset_means_the_controller_is_not_touched():
    orch = _make_orchestrator()
    client_app, client = _app_with_client(orch, AsyncMock(return_value={"success": True}))
    r = client_app.post("/api/devices/timelapse/start", json={"interval_seconds": 60})
    assert r.status_code == 200, r.text
    client.set_laser_config.assert_not_awaited()


def test_a_preset_the_controller_refuses_stops_the_start():
    """Every timepoint would otherwise image with the wrong lasers."""
    orch = _make_orchestrator()
    client_app, client = _app_with_client(
        orch, AsyncMock(side_effect=RuntimeError("no such preset"))
    )
    r = client_app.post("/api/devices/timelapse/start", json={"laser_config": "nope"})
    assert r.status_code == 502
    assert "no such preset" in r.json()["detail"]
    orch.start.assert_not_awaited()
    client.set_laser_config.assert_awaited_once_with("nope")
