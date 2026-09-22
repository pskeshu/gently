"""A HALT that did not stop the stage must not report success.

The device layer gets this right: when a controller refuses, it answers 502
with a per-axis ``errors`` map and lists what it did stop. Everything above it
threw that away. ``DiSPIMMicroscope._api_post`` returns ``resp.json()`` without
looking at the status, and the web route returned that body as-is, so the
browser saw HTTP 200 and the operator saw a green toast:

    one axis refuses   -> "Halted: fdrive, xy_stage"      (z_stage still moving)
    every axis refuses -> "Halted: nothing moving"        (nothing stopped)

On the control whose entire purpose is stopping the stage, that is the worst
available failure: a false all-clear. Proved end to end against the merged
code with a fake MMCore whose stop() raises, then fixed here.

The route converts ``success: false`` into 502 while keeping the body — the UI
needs to name which axes did not stop — and the toast says so.
"""

from __future__ import annotations

import re
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

import gently.ui.web.auth as auth
from gently.ui.web.routes.data import create_router

OPERATE_JS = (
    Path(__file__).resolve().parents[1] / "gently" / "ui" / "web" / "static" / "js" / "operate.js"
)


def _app(halt_result):
    client = MagicMock()
    client.halt_motion = AsyncMock(return_value=halt_result)
    server = MagicMock()
    server.agent_bridge.agent.client = client
    server.agent_bridge.agent.lightsheet_monitor = None
    app = FastAPI()
    app.include_router(create_router(server))
    app.dependency_overrides[auth.require_control] = lambda: True
    return TestClient(app)


def test_a_halt_that_stopped_everything_is_a_success() -> None:
    r = _app({"success": True, "halted": ["fdrive", "xy_stage"], "errors": {}}).post(
        "/api/devices/motion/halt"
    )
    assert r.status_code == 200
    assert r.json()["halted"] == ["fdrive", "xy_stage"]


def test_a_partial_halt_is_not_a_success() -> None:
    """One axis still moving is the case this exists for."""
    body = {
        "success": False,
        "halted": ["fdrive", "xy_stage"],
        "errors": {"z_stage": "ZStage:Z:32: HALT refused"},
    }
    r = _app(body).post("/api/devices/motion/halt")
    assert r.status_code == 502, (
        "a refused HALT reaches the browser as 200 again — the operator gets a "
        "green 'Halted' toast while an axis is still moving"
    )
    # The body must survive: the toast names the axes that did NOT stop.
    assert r.json()["errors"] == {"z_stage": "ZStage:Z:32: HALT refused"}
    assert r.json()["halted"] == ["fdrive", "xy_stage"]


def test_a_total_refusal_is_not_reported_as_nothing_moving() -> None:
    body = {"success": False, "halted": [], "errors": {"fdrive": "refused", "xy_stage": "refused"}}
    r = _app(body).post("/api/devices/motion/halt")
    assert r.status_code == 502
    assert set(r.json()["errors"]) == {"fdrive", "xy_stage"}


def test_the_toast_names_the_axes_that_did_not_stop() -> None:
    """CI runs no JavaScript, so the operator-facing half is pinned in source."""
    js = OPERATE_JS.read_text(encoding="utf-8")
    halt = re.search(r"async function haltMotion\(\) \{(.*?)\n    \}", js, re.S)
    assert halt, "haltMotion is gone"
    body = halt.group(1)
    assert "did NOT stop" in body, (
        "the failure toast no longer names the axes that refused; 'HALT failed' "
        "alone leaves the operator guessing which axis is still moving"
    )
    assert "e.data" in body, "the toast no longer reads the refusal body"
    assert "none are registered" in body, (
        "a HALT that reached no positioner reports as a plain success again — "
        "'Halted: nothing moving' reads as 'everything is stopped'"
    )
