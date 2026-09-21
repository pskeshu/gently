"""The exposure the panel asks for is the exposure the run uses.

"in light sheet view we have exposure time, but not sure we use it."

It was half true, in the worst way — the control looked live and behaved
differently on each path:

* the **live stream** did use it (the device layer reconfigures the camera
  sequence and floors the frame interval at the exposure);
* the **timelapse** did not. `POST /api/devices/timelapse/start` read
  `exposure_ms` from the body, validated it, packed it into a
  `volume_geometry` dict — and dropped it. `orchestrator.start()` takes no
  geometry; it reads `embryo.exposure_ms`, which nothing on that path set. So
  every timepoint ran at the 10 ms / 50 slice defaults no matter what the
  panel said.
* the **snap** path was worse than dropped: `capture_lightsheet_image()` was
  called with no exposure (client default 10 ms) and the dose was then
  recorded as a hard-coded 50 ms — the phototoxicity ledger wrong by 5x on
  every snap embryo, in a system whose point is not over-exposing the sample.

What must stay true: the settings land on exactly the embryos the run will
image, the ledger records what was actually used, and the galvo/piezo geometry
is NOT overridden from the panel — that comes from each embryo's calibration,
and a run that images a cuboid nobody measured is the failure the calibration
gate exists to prevent.
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import gently.ui.web.auth as auth
from gently.app.orchestration.timelapse import TimelapseOrchestrator
from gently.harness.state import EmbryoState
from gently.ui.web.routes.data import create_router


def _embryo(eid: str, **kw) -> EmbryoState:
    emb = EmbryoState(id=eid)
    emb.stage_position = {"x": 100.0, "y": 200.0}
    emb.calibration = {
        "slope_um_per_deg": 101.2,
        "r_squared": 0.97,
        "galvo_amplitude": 0.33,
        "galvo_center": 0.11,
        "piezo_amplitude": 22.0,
        "piezo_center": 44.0,
    }
    for k, v in kw.items():
        setattr(emb, k, v)
    return emb


def _app(embryos: dict[str, EmbryoState], orchestrator) -> TestClient:
    server = MagicMock()
    agent = server.agent_bridge.agent
    agent.timelapse_orchestrator = orchestrator
    agent.client = MagicMock()
    agent.lightsheet_monitor = None
    agent.experiment.embryos = embryos
    app = FastAPI()
    app.include_router(create_router(server))
    app.dependency_overrides[auth.require_control] = lambda: True
    return TestClient(app)


def _orch() -> MagicMock:
    orch = MagicMock()
    orch.start = AsyncMock(return_value="Timelapse started.")
    orch.enable_monitoring_mode = MagicMock(return_value="ok")
    return orch


# ---------------------------------------------------------------------------
# The route
# ---------------------------------------------------------------------------


def test_the_panels_exposure_reaches_the_embryos_that_will_be_imaged() -> None:
    embryos = {"embryo_1": _embryo("embryo_1"), "embryo_2": _embryo("embryo_2")}
    r = _app(embryos, _orch()).post(
        "/api/devices/timelapse/start",
        json={"interval_seconds": 120, "exposure_ms": 35.0, "num_slices": 12},
    )
    assert r.status_code == 200, r.text
    for emb in embryos.values():
        assert emb.exposure_ms == 35.0, (
            "the exposure was collected, validated and dropped again — the run "
            "will image at the 10 ms default whatever the panel said"
        )
        assert emb.num_slices == 12


def test_a_skipped_embryo_is_not_reconfigured() -> None:
    """The set that gets the settings is the set that gets imaged."""
    keep, skip = _embryo("embryo_1"), _embryo("embryo_2", should_skip=True)
    r = _app({"embryo_1": keep, "embryo_2": skip}, _orch()).post(
        "/api/devices/timelapse/start",
        json={"interval_seconds": 120, "exposure_ms": 35.0},
    )
    assert r.status_code == 200
    assert keep.exposure_ms == 35.0
    assert skip.exposure_ms == 10.0, "a skipped embryo was reconfigured by a run it is not in"


def test_named_embryos_only() -> None:
    a, b = _embryo("embryo_1"), _embryo("embryo_2")
    r = _app({"embryo_1": a, "embryo_2": b}, _orch()).post(
        "/api/devices/timelapse/start",
        json={
            "interval_seconds": 120,
            "exposure_ms": 35.0,
            "embryo_ids": ["embryo_1"],
            "allow_uncalibrated": True,
        },
    )
    assert r.status_code == 200
    assert a.exposure_ms == 35.0
    assert b.exposure_ms == 10.0


@pytest.mark.parametrize("bad", [0, -5, 10001, "soon"])
def test_an_exposure_the_camera_cannot_take_is_refused(bad: object) -> None:
    emb = _embryo("embryo_1")
    r = _app({"embryo_1": emb}, _orch()).post(
        "/api/devices/timelapse/start",
        json={"interval_seconds": 120, "exposure_ms": bad},
    )
    assert r.status_code == 400
    assert emb.exposure_ms == 10.0, "a refused request still rewrote the embryo"


def test_the_panel_does_not_override_the_measured_geometry() -> None:
    """Galvo/piezo come from the embryo's calibration, not from the panel."""
    emb = _embryo("embryo_1")
    before = dict(emb.calibration)
    r = _app({"embryo_1": emb}, _orch()).post(
        "/api/devices/timelapse/start",
        json={
            "interval_seconds": 120,
            "exposure_ms": 35.0,
            "galvo_amplitude": 9.9,
            "piezo_center": 999.0,
        },
    )
    assert r.status_code == 200
    assert emb.calibration == before, (
        "the panel overwrote a measured scan geometry; the run would image a "
        "cuboid nobody calibrated"
    )


# ---------------------------------------------------------------------------
# The snap path
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_a_snap_uses_the_embryos_exposure_and_records_what_it_used() -> None:
    emb = _embryo("embryo_1", acquisition_mode="snap", exposure_ms=35.0)
    experiment = MagicMock()
    experiment.embryos = {"embryo_1": emb}

    client = MagicMock()
    client.move_to_position = AsyncMock(return_value={"success": True})
    client.capture_lightsheet_image = AsyncMock(return_value={"success": True, "image": None})

    orch = TimelapseOrchestrator(microscope_client=client, experiment_state=experiment)
    await orch._acquire_embryo(emb, round_time=datetime.now())

    kwargs = client.capture_lightsheet_image.await_args.kwargs
    assert kwargs.get("exposure_ms") == 35.0, (
        "the snap ran at the client's default exposure, not the one configured for this embryo"
    )
    # The ledger must agree with the camera, or dose accounting is fiction.
    assert emb.total_exposure_ms == 35.0, f"recorded {emb.total_exposure_ms} ms for a 35 ms snap"
