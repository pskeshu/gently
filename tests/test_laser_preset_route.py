"""Tests for B2 Task 1 — POST /api/devices/laser/config (set-preset proxy).

Verifies:
  - POST /api/devices/laser/config calls client.set_laser_config(config)
  - Missing config body key → 400
  - Empty string config → 400
  - No client (microscope not connected) → 503
  - require_control gate (403 without override)
"""

from unittest.mock import AsyncMock, MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

import gently.ui.web.auth as auth
from gently.ui.web.routes.data import create_router


def _app(client=None):
    """Build a TestClient with the data routes wired up.

    Overrides require_control so the TestClient always gets the control role —
    isolates route logic from auth, mirroring the pattern in test_lightsheet_routes.
    """
    server = MagicMock()
    server.agent_bridge.agent.client = client
    server.agent_bridge.agent.lightsheet_monitor = None
    app = FastAPI()
    app.include_router(create_router(server))
    app.dependency_overrides[auth.require_control] = lambda: True
    return TestClient(app)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_set_laser_config_calls_client():
    """POST /api/devices/laser/config must call client.set_laser_config(config)."""
    client = MagicMock()
    client.set_laser_config = AsyncMock(return_value={"success": True})
    r = _app(client=client).post(
        "/api/devices/laser/config",
        json={"config": "488 only"},
    )
    assert r.status_code == 200
    client.set_laser_config.assert_awaited_once_with("488 only")


def test_set_laser_config_all_off():
    """POST /api/devices/laser/config works with 'ALL OFF'."""
    client = MagicMock()
    client.set_laser_config = AsyncMock(return_value={"success": True})
    r = _app(client=client).post(
        "/api/devices/laser/config",
        json={"config": "ALL OFF"},
    )
    assert r.status_code == 200
    client.set_laser_config.assert_awaited_once_with("ALL OFF")


# ---------------------------------------------------------------------------
# 400 — missing / invalid config
# ---------------------------------------------------------------------------


def test_set_laser_config_missing_key_returns_400():
    """POST without 'config' key in body → 400."""
    client = MagicMock()
    r = _app(client=client).post(
        "/api/devices/laser/config",
        json={"wrong_key": "ALL OFF"},
    )
    assert r.status_code == 400


def test_set_laser_config_empty_string_returns_400():
    """POST with empty string config → 400."""
    client = MagicMock()
    r = _app(client=client).post(
        "/api/devices/laser/config",
        json={"config": ""},
    )
    assert r.status_code == 400


def test_set_laser_config_null_returns_400():
    """POST with null config → 400."""
    client = MagicMock()
    r = _app(client=client).post(
        "/api/devices/laser/config",
        json={"config": None},
    )
    assert r.status_code == 400


# ---------------------------------------------------------------------------
# 503 — no client
# ---------------------------------------------------------------------------


def test_set_laser_config_no_client_returns_503():
    """POST when microscope not connected → 503."""
    r = _app(client=None).post(
        "/api/devices/laser/config",
        json={"config": "ALL OFF"},
    )
    assert r.status_code == 503


# ---------------------------------------------------------------------------
# require_control gate
# ---------------------------------------------------------------------------


def test_set_laser_config_requires_control():
    """POST /api/devices/laser/config is gated by require_control."""
    server = MagicMock()
    client = MagicMock()
    client.set_laser_config = AsyncMock(return_value={"success": True})
    server.agent_bridge.agent.client = client
    server.agent_bridge.agent.lightsheet_monitor = None
    app = FastAPI()
    app.include_router(create_router(server))
    # No dependency override — require_control will reject non-loopback hosts
    r = TestClient(app, raise_server_exceptions=False).post(
        "/api/devices/laser/config",
        json={"config": "ALL OFF"},
    )
    assert r.status_code == 403
