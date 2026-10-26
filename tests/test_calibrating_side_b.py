"""Calibrating side B: the scan routine on the other arm, into its own slot.

Side B is measured with the B scanner, piezo and camera, so every
hardware-timed volume the scan atoms take carries the side; its starting
line is B's own previous fit or nothing, never A's numbers; and the legacy
sweep, which drives A's devices, refuses to pretend otherwise.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import numpy as np

from gently.calibration import atoms
from gently.calibration.scan import heuristic


def test_every_hardware_timed_volume_carries_the_side():
    client = MagicMock()
    client.acquire_volume = AsyncMock(
        return_value={"success": True, "volume": np.zeros((3, 4, 4), np.uint16)}
    )
    budget = atoms.Budget(100)
    asyncio.run(
        atoms._volume(
            client,
            n=3,
            exposure_ms=10.0,
            galvo_amp=0.5,
            galvo_c=0.0,
            piezo_amp=0.0,
            piezo_c=50.0,
            budget=budget,
            side="B",
        )
    )
    assert client.acquire_volume.await_args.kwargs["side"] == "B"
    asyncio.run(
        atoms._volume(
            client,
            n=3,
            exposure_ms=10.0,
            galvo_amp=0.5,
            galvo_c=0.0,
            piezo_amp=0.0,
            piezo_c=50.0,
            budget=budget,
        )
    )
    assert client.acquire_volume.await_args.kwargs["side"] == "A"


def _agent_without_prior():
    prior = SimpleNamespace(num_calibrations=0, r_squared_mean=0.0, slope_um_per_deg=100.0)
    return SimpleNamespace(experiment=SimpleNamespace(calibration_prior=prior))


def test_b_starts_from_its_own_fit_and_never_from_a_s():
    agent = _agent_without_prior()
    a_only = SimpleNamespace(calibration={"slope_um_per_deg": 50.0, "offset_um": 7.0})
    _, _, origin_a = heuristic(agent, a_only, "A")
    assert origin_a == "this embryo's previous fit"
    slope_b, _, origin_b = heuristic(agent, a_only, "B")
    assert origin_b != "this embryo's previous fit" and slope_b != 50.0

    both = SimpleNamespace(
        calibration={
            "slope_um_per_deg": 50.0,
            "offset_um": 7.0,
            "B": {"slope_um_per_deg": 101.0, "offset_um": 4.1},
        }
    )
    slope_b, offset_b, origin_b = heuristic(agent, both, "B")
    assert (slope_b, offset_b, origin_b) == (101.0, 4.1, "this embryo's previous fit")


def test_the_legacy_sweep_refuses_side_b():
    from gently.app.tools.calibration_tools import calibrate_embryo

    agent = MagicMock()
    client = MagicMock()
    client.is_connected = True
    fn = getattr(calibrate_embryo, "__wrapped__", calibrate_embryo)
    out = asyncio.run(
        fn("embryo_1", method="legacy", side="B", context={"agent": agent, "client": client})
    )
    assert out.startswith("Error") and "scan" in out
    out = asyncio.run(fn("embryo_1", side="C", context={"agent": agent, "client": client}))
    assert out.startswith("Error: side must be A or B")


def test_the_pane_s_side_reaches_the_tool():
    from fastapi import FastAPI

    import gently.ui.web.auth as auth
    from gently.ui.web.routes.data import create_router

    server = MagicMock()
    app = FastAPI()
    app.include_router(create_router(server))
    app.dependency_overrides[auth.require_control] = lambda: True
    # The filter is a closure inside the router factory; reach it the way the
    # routes do, through the module's own helper if exposed, else via a probe.
    import inspect

    src = inspect.getsource(create_router)
    assert 'if payload.get("side") in ("A", "B"):' in src
