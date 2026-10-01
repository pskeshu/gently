"""The Acquisition pane's "Acquire one volume" is the embryo's volume.

"the single volume runs do not work i think."

Two reasons it did not, that night and since. The one that was true on the
day: the device layer was stopped, so the route answered "Microscope not
connected" — which the pane says, and nothing here changes. The one that
is true whenever it is running: the route checked that the named embryo was
calibrated, and then acquired wherever the stage happened to be, with the
default scan geometry, and kept nothing. "Volume acquired", and nothing in
the Embryos tab, the filmstrip, or on disk to show for it.

The agent's `acquire_volume` tool does it right — moves to the embryo, scans
with its calibration, files the volume as a timepoint, pushes the projection
— so the route now runs that tool for the embryo. One way to take a volume,
whoever asks. Manual-mode snapping with no embryo named is left as it was.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient

import gently.ui.web.auth as auth
from gently.harness.state import ExperimentState
from gently.ui.web.routes.data import create_router

WEB = Path(__file__).resolve().parents[1] / "gently" / "ui" / "web"
OPERATE = (WEB / "static" / "js" / "operate.js").read_text(encoding="utf-8")

CAL = {
    "slope_um_per_deg": 50.0,
    "galvo_center": 0.3,
    "galvo_amplitude": 0.4,
    "piezo_center": 62.0,
    "piezo_amplitude": 18.0,
}


def _experiment(calibrated=True):
    ex = ExperimentState()
    ex.add_embryo(
        "embryo_2",
        position={"x": -200.0, "y": -600.0},
        calibration=dict(CAL) if calibrated else None,
    )
    return ex


def _app(experiment, client=None):
    server = MagicMock()
    agent = server.agent_bridge.agent
    agent.experiment = experiment
    agent.session_id = None
    agent.store = None
    agent.viz_server = None
    client = client or MagicMock()
    client.is_connected = True
    client.move_to_position = AsyncMock(return_value={"success": True})
    client.acquire_volume = AsyncMock(
        return_value={
            "success": True,
            "volume": np.zeros((3, 4, 4), dtype=np.uint16),
            "shape": (3, 4, 4),
        }
    )
    agent.client = client
    agent.lightsheet_monitor = None
    app = FastAPI()
    app.include_router(create_router(server))
    app.dependency_overrides[auth.require_control] = lambda: True
    return TestClient(app), agent, client


class TestForAnEmbryo:
    def test_it_goes_to_the_embryo_and_scans_with_its_calibration(self):
        ex = _experiment()
        app, agent, client = _app(ex)
        r = app.post(
            "/api/devices/acquire/volume",
            json={"embryo_id": "embryo_2", "num_slices": 40, "exposure_ms": 12},
        )
        assert r.status_code == 200, r.text
        client.move_to_position.assert_awaited()
        kw = client.acquire_volume.await_args.kwargs
        assert (kw["num_slices"], kw["exposure_ms"]) == (40, 12.0)
        assert (kw["galvo_center"], kw["galvo_amplitude"]) == (0.3, 0.4)
        assert (kw["piezo_center"], kw["piezo_amplitude"]) == (62.0, 18.0)

    def test_it_counts_as_a_timepoint_and_says_so(self):
        ex = _experiment()
        app, _, _ = _app(ex)
        r = app.post("/api/devices/acquire/volume", json={"embryo_id": "embryo_2"})
        assert r.status_code == 200, r.text
        assert ex.embryos["embryo_2"].timepoints_acquired == 1
        body = r.json()
        assert body["success"] is True and body["embryo_id"] == "embryo_2"
        assert body["message"].startswith("Acquired volume for embryo_2")

    def test_an_uncalibrated_embryo_is_still_refused(self):
        app, _, client = _app(_experiment(calibrated=False))
        r = app.post("/api/devices/acquire/volume", json={"embryo_id": "embryo_2"})
        assert r.status_code == 409
        client.acquire_volume.assert_not_awaited()

    def test_the_override_still_goes_through(self):
        app, _, client = _app(_experiment(calibrated=False))
        r = app.post(
            "/api/devices/acquire/volume",
            json={"embryo_id": "embryo_2", "allow_uncalibrated": True},
        )
        assert r.status_code == 200, r.text
        client.acquire_volume.assert_awaited()

    def test_a_failed_acquisition_is_a_failure_not_a_200(self):
        ex = _experiment()
        app, _, client = _app(ex)
        client.acquire_volume = AsyncMock(
            return_value={"success": False, "error": "camera timeout"}
        )
        r = app.post("/api/devices/acquire/volume", json={"embryo_id": "embryo_2"})
        assert r.status_code == 502 and "camera timeout" in r.json()["detail"]
        assert ex.embryos["embryo_2"].timepoints_acquired == 0

    def test_no_microscope_is_said(self):
        client = MagicMock()
        app, agent, client = _app(_experiment(), client)
        client.is_connected = False
        r = app.post("/api/devices/acquire/volume", json={"embryo_id": "embryo_2"})
        assert r.status_code == 503 and "not connected" in r.json()["detail"].lower()

    def test_an_embryo_there_is_not(self):
        app, _, client = _app(_experiment())
        r = app.post(
            "/api/devices/acquire/volume",
            json={"embryo_id": "embryo_9", "allow_uncalibrated": True},
        )
        assert r.status_code == 502
        client.acquire_volume.assert_not_awaited()


class TestWithoutAnEmbryo:
    def test_manual_snapping_is_left_as_it_was(self):
        """A test shot at the current position, with nothing to check it
        against: forwarded straight to the client, geometry and all."""
        app, _, client = _app(_experiment())
        r = app.post(
            "/api/devices/acquire/volume",
            json={
                "num_slices": 5,
                "exposure_ms": 20.0,
                "laser_config": "ALL OFF",
                "piezo_center": 55.0,
            },
        )
        assert r.status_code == 200, r.text
        client.move_to_position.assert_not_awaited()
        client.acquire_volume.assert_awaited_once_with(
            num_slices=5, exposure_ms=20.0, laser_config="ALL OFF", piezo_center=55.0
        )


class TestThePane:
    def test_the_toast_says_what_the_acquisition_said(self):
        branch = OPERATE[
            OPERATE.index("if (_mode === 'single') {") : OPERATE.index(
                "if (_mode === 'adaptive') {"
            )
        ]
        assert "const d = await postJSON('/api/devices/acquire/volume'" in branch
        assert "(d && d.message) || 'Volume acquired'" in branch
