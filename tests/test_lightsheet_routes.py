"""Tests for Task 5 lightsheet browser proxy routes.

Verifies:
  - live/start, live/stop, live/status — mirror bottom_camera routes
  - live/params — forwards to client.set_lightsheet_live_params
  - led/set, laser/off, camera/led_mode, stage/move
  - acquire/burst, acquire/volume
  - require_control gate (403 without override)
"""
from unittest.mock import MagicMock, AsyncMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from gently.ui.web.routes.data import create_router
import gently.ui.web.auth as auth


def _app(client=None, monitor=None):
    """Build a TestClient with the lightsheet routes wired up.

    Overrides require_control so that the TestClient (host="testclient")
    always gets the control role — isolates route logic from auth.
    """
    server = MagicMock()
    server.agent_bridge.agent.client = client
    server.agent_bridge.agent.lightsheet_monitor = monitor
    app = FastAPI()
    app.include_router(create_router(server))
    # TestClient host is "testclient" (not loopback) → override to force CONTROL
    app.dependency_overrides[auth.require_control] = lambda: True
    return TestClient(app)


# ---------------------------------------------------------------------------
# live start / stop / status
# ---------------------------------------------------------------------------

def test_live_status_no_monitor():
    """GET status returns available=False when monitor is None."""
    r = _app(monitor=None).get("/api/devices/lightsheet/live/status")
    assert r.status_code == 200
    assert r.json()["available"] is False


def test_live_status_with_monitor():
    monitor = MagicMock()
    monitor.running = True
    monitor._last_frame_ts = "2026-01-01T00:00:00"
    r = _app(monitor=monitor).get("/api/devices/lightsheet/live/status")
    assert r.status_code == 200
    body = r.json()
    assert body["available"] is True
    assert body["streaming"] is True


def test_live_start_requires_monitor():
    """POST start with no monitor → 503."""
    r = _app(monitor=None).post("/api/devices/lightsheet/live/start")
    assert r.status_code == 503


def test_live_start_calls_monitor_start():
    monitor = MagicMock()
    monitor.start = AsyncMock()
    monitor.running = True
    r = _app(monitor=monitor).post("/api/devices/lightsheet/live/start")
    assert r.status_code == 200
    monitor.start.assert_awaited_once()


def test_live_stop_no_monitor_returns_200():
    """POST stop with no monitor is idempotent — returns 200 streaming=False."""
    r = _app(monitor=None).post("/api/devices/lightsheet/live/stop")
    assert r.status_code == 200
    assert r.json()["streaming"] is False


def test_live_stop_calls_monitor_stop():
    monitor = MagicMock()
    monitor.stop = AsyncMock()
    r = _app(monitor=monitor).post("/api/devices/lightsheet/live/stop")
    assert r.status_code == 200
    monitor.stop.assert_awaited_once()


# ---------------------------------------------------------------------------
# live/params
# ---------------------------------------------------------------------------

def test_live_params_forwards():
    client = MagicMock()
    client.set_lightsheet_live_params = AsyncMock(return_value={"params": {}})
    r = _app(client=client).post(
        "/api/devices/lightsheet/live/params",
        json={"galvo": 1.0, "piezo": 40.0, "exposure": 20.0},
    )
    assert r.status_code == 200
    client.set_lightsheet_live_params.assert_awaited_once_with(
        galvo=1.0, piezo=40.0, exposure=20.0
    )


def test_live_params_no_client_503():
    server = MagicMock()
    server.agent_bridge.agent.client = None
    app = FastAPI()
    app.include_router(create_router(server))
    app.dependency_overrides[auth.require_control] = lambda: True
    r = TestClient(app).post(
        "/api/devices/lightsheet/live/params",
        json={"galvo": 1.0},
    )
    assert r.status_code == 503


# ---------------------------------------------------------------------------
# led/set
# ---------------------------------------------------------------------------

def test_led_set_forwards():
    client = MagicMock()
    client.set_led = AsyncMock(return_value={"success": True})
    r = _app(client=client).post("/api/devices/led/set", json={"state": "Open"})
    assert r.status_code == 200
    client.set_led.assert_awaited_once_with("Open")


# ---------------------------------------------------------------------------
# laser/off  — confirmed signature: set_laser_power(wavelength: int, pct: float)
# ---------------------------------------------------------------------------

def test_laser_off_calls_set_laser_power_488_0():
    """laser/off must call set_laser_power(488, 0) to force 488 nm line off."""
    client = MagicMock()
    client.set_laser_power = AsyncMock(return_value={"success": True})
    r = _app(client=client).post("/api/devices/laser/off")
    assert r.status_code == 200
    client.set_laser_power.assert_awaited_once_with(488, 0)


# ---------------------------------------------------------------------------
# camera/led_mode
# ---------------------------------------------------------------------------

def test_camera_led_mode_forwards():
    client = MagicMock()
    client.set_camera_led_mode = AsyncMock(return_value={"success": True})
    r = _app(client=client).post("/api/devices/camera/led_mode", json={"use_led": True})
    assert r.status_code == 200
    client.set_camera_led_mode.assert_awaited_once_with(True)


# ---------------------------------------------------------------------------
# stage/move
# ---------------------------------------------------------------------------

def test_stage_move_forwards():
    client = MagicMock()
    client.move_to_position = AsyncMock(return_value={"success": True})
    r = _app(client=client).post("/api/devices/stage/move", json={"x": 100.0, "y": 200.0})
    assert r.status_code == 200
    client.move_to_position.assert_awaited_once_with(100.0, 200.0)


def test_stage_move_missing_xy_400():
    client = MagicMock()
    client.move_to_position = AsyncMock(return_value={"success": True})
    r = _app(client=client).post("/api/devices/stage/move", json={"x": 1.0})
    assert r.status_code == 400


# ---------------------------------------------------------------------------
# acquire/burst
# ---------------------------------------------------------------------------

def test_acquire_burst_forwards():
    client = MagicMock()
    client.acquire_burst = AsyncMock(return_value={"success": True, "request_id": "b1"})
    r = _app(client=client).post(
        "/api/devices/acquire/burst",
        json={"frames": 60, "mode": "1hz", "num_slices": 1, "exposure_ms": 5.0},
    )
    assert r.status_code == 200 and r.json().get("request_id") == "b1"
    client.acquire_burst.assert_awaited_once_with(
        frames=60, mode="1hz", num_slices=1, exposure_ms=5.0
    )


# ---------------------------------------------------------------------------
# acquire/volume
# ---------------------------------------------------------------------------

def test_acquire_volume_forwards():
    client = MagicMock()
    client.acquire_volume = AsyncMock(return_value={"success": True, "request_id": "v1"})
    r = _app(client=client).post(
        "/api/devices/acquire/volume",
        json={"num_slices": 50, "exposure_ms": 10.0},
    )
    assert r.status_code == 200
    client.acquire_volume.assert_awaited_once_with(num_slices=50, exposure_ms=10.0)


# ---------------------------------------------------------------------------
# require_control gate — 403 WITHOUT override
# ---------------------------------------------------------------------------

def test_require_control_gate_403():
    """Without the dependency override, TestClient host is not loopback → 403."""
    server = MagicMock()
    server.agent_bridge.agent.client = MagicMock()
    server.agent_bridge.agent.lightsheet_monitor = None
    app = FastAPI()
    app.include_router(create_router(server))
    # Do NOT override require_control here
    r = TestClient(app).post("/api/devices/lightsheet/live/start")
    assert r.status_code == 403
