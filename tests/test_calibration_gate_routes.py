"""The calibration gate must live on the routes, not in the browser.

The agent reaches these endpoints too, so a check in `operate.js` is a check
the agent walks past. These tests pin the refusal to the HTTP boundary.

409 rather than 400: the request is well formed, the instrument is simply not
in a state where it can be honoured.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from unittest.mock import AsyncMock, MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

import gently.ui.web.auth as auth
from gently.ui.web.routes.data import create_router


@dataclass
class _Embryo:
    calibration: dict[str, Any] = field(default_factory=dict)


FIT = {"slope_um_per_deg": 42.0, "offset_um": 1.0, "r_squared": 0.85}


def _app(embryos: dict[str, _Embryo]):
    """Routes wired against a roster, with a working orchestrator and client."""
    server = MagicMock()
    agent = server.agent_bridge.agent
    agent.experiment.embryos = embryos
    agent.timelapse_orchestrator.start_timelapse = AsyncMock(return_value={"success": True})
    agent.client.acquire_volume = AsyncMock(return_value={"success": True})
    app = FastAPI()
    app.include_router(create_router(server))
    app.dependency_overrides[auth.require_control] = lambda: True
    return TestClient(app)


ROSTER = {"e1": _Embryo(dict(FIT)), "e2": _Embryo({})}


# ── timelapse ────────────────────────────────────────────────────────────────


def test_timelapse_refuses_an_uncalibrated_embryo() -> None:
    r = _app(ROSTER).post(
        "/api/devices/timelapse/start",
        json={"interval_seconds": 120, "embryo_ids": ["e1", "e2"]},
    )
    assert r.status_code == 409, r.text
    detail = r.json()["detail"]
    assert "e2" in detail
    assert "e1" not in detail, "a calibrated embryo must not be named in the refusal"
    assert "allow_uncalibrated" in detail


def test_timelapse_allows_a_fully_calibrated_set() -> None:
    r = _app(ROSTER).post(
        "/api/devices/timelapse/start",
        json={"interval_seconds": 120, "embryo_ids": ["e1"]},
    )
    assert r.status_code != 409, r.text


def test_null_embryo_ids_still_checks_the_whole_roster() -> None:
    """`embryo_ids: null` means "all active embryos" — including the bad one."""
    r = _app(ROSTER).post("/api/devices/timelapse/start", json={"interval_seconds": 120})
    assert r.status_code == 409, r.text
    assert "e2" in r.json()["detail"]


def test_the_override_is_honoured() -> None:
    """Someone who means it can say so. The point is that they must say it."""
    r = _app(ROSTER).post(
        "/api/devices/timelapse/start",
        json={"interval_seconds": 120, "embryo_ids": ["e1", "e2"], "allow_uncalibrated": True},
    )
    assert r.status_code != 409, r.text


def test_the_gate_runs_before_the_interval_check_is_irrelevant() -> None:
    """A bad interval must still be a 400 — the gate does not mask validation."""
    r = _app(ROSTER).post(
        "/api/devices/timelapse/start",
        json={"interval_seconds": 0, "embryo_ids": ["e1"]},
    )
    assert r.status_code == 400, r.text


# ── single volume ────────────────────────────────────────────────────────────


def test_volume_refuses_a_named_uncalibrated_embryo() -> None:
    r = _app(ROSTER).post("/api/devices/acquire/volume", json={"embryo_id": "e2"})
    assert r.status_code == 409, r.text


def test_volume_without_an_embryo_id_is_not_gated() -> None:
    """Manual-mode snapping images the current stage position deliberately.

    Optional by design, not by oversight: a test shot with no embryo in mind is
    a real thing to want, and there is nothing to check it against.
    """
    r = _app(ROSTER).post("/api/devices/acquire/volume", json={"num_slices": 5})
    assert r.status_code != 409, r.text


# ── tactics ──────────────────────────────────────────────────────────────────


def test_run_tactic_refuses_an_uncalibrated_embryo() -> None:
    r = _app(ROSTER).post(
        "/api/operate/run-tactic",
        json={"library_id": "whatever", "embryo_ids": ["e2"]},
    )
    assert r.status_code == 409, r.text


# ── no roster ────────────────────────────────────────────────────────────────


def test_an_empty_roster_does_not_block() -> None:
    """With nothing known, there is nothing to assert — do not invent a refusal."""
    r = _app({}).post("/api/devices/timelapse/start", json={"interval_seconds": 120})
    assert r.status_code != 409, r.text
