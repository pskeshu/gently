"""The plan's laser settings reach every volume of the run.

The Acquisition pane asked for a laser preset and had nowhere to ask for
power. Adding the power found that the preset did not hold either: the start
route set it on the controller once, and every volume then routed its own
lines as it began — the volume plan's default, "488 and 561" — so a run
started on "488 only" imaged with both from its first volume on.

So the run carries both. The preset is the run's; the power is each embryo's,
beside the 488 power that was already there, because that is where the
saturation ramp-down looks for the number it steps.

- a power outside the device layer's hard limit refuses the START, rather
  than failing every embryo's first three volumes one by one
- a line the plan leaves empty keeps the power somebody gave it
- nothing is said to a volume about a line or a preset the plan did not set
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import gently.ui.web.auth as auth
from gently.app.orchestration.timelapse import TimelapseOrchestrator
from gently.hardware.dispim.devices.optical import DiSPIMLightSource
from gently.harness.state import ExperimentState
from gently.ui.web.routes.data import create_router

WEB = Path(__file__).resolve().parents[1] / "gently" / "ui" / "web"
FIT = {"galvo_center": 0.0, "slope_um_per_deg": 50.0}


def _experiment(n=2):
    ex = ExperimentState()
    for i in range(n):
        ex.add_embryo(f"embryo_{i + 1}", position={"x": 10.0 * i, "y": 0.0}, calibration=dict(FIT))
    return ex


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------


def _client():
    c = MagicMock()
    c.move_to_position = AsyncMock(return_value={"success": True})
    c.acquire_volume = AsyncMock(return_value={"success": True, "volume": None})
    return c


def _volumes(experiment, **start):
    orch = TimelapseOrchestrator(_client(), experiment, store=None, session_id=None)

    async def go():
        msg = await orch.start(base_interval_seconds=0.1, **start)
        assert msg.startswith("Started"), msg
        await asyncio.sleep(0.4)
        await orch.stop("test done")

    asyncio.run(go())
    calls = orch.client.acquire_volume.await_args_list
    assert calls, "no volume was taken"
    return orch, [c.kwargs for c in calls]


class TestEveryVolume:
    def test_the_preset_goes_with_every_volume_not_only_the_first(self):
        _, calls = _volumes(_experiment(), laser_config="488 only")
        assert len(calls) >= 3
        assert {c.get("laser_config") for c in calls} == {"488 only"}

    def test_each_embryo_is_imaged_at_its_own_power(self):
        ex = _experiment()
        ex.embryos["embryo_1"].laser_power_488_pct = 3.0
        ex.embryos["embryo_1"].laser_power_561_pct = 12.5
        ex.embryos["embryo_2"].laser_power_488_pct = 5.0
        orch, calls = _volumes(ex)
        moves = [c.args for c in orch.client.move_to_position.await_args_list]
        assert moves
        seen = {(c["laser_power_488_pct"], c.get("laser_power_561_pct")) for c in calls}
        assert seen == {(3.0, 12.5), (5.0, None)}

    def test_every_line_has_a_power(self):
        ex = _experiment(1)
        e = ex.embryos["embryo_1"]
        e.laser_power_405_pct, e.laser_power_488_pct = 1.0, 4.0
        e.laser_power_561_pct, e.laser_power_637_pct = 20.0, 30.0
        _, calls = _volumes(ex)
        for c in calls:
            assert (
                c["laser_power_405_pct"],
                c["laser_power_488_pct"],
                c["laser_power_561_pct"],
                c["laser_power_637_pct"],
            ) == (1.0, 4.0, 20.0, 30.0)

    def test_what_the_plan_did_not_set_is_not_mentioned(self):
        """None would do, the client drops it — but a volume is told only
        what somebody decided, and that is what the record of the call shows."""
        _, calls = _volumes(_experiment())
        for c in calls:
            assert "laser_config" not in c
            for wl in (405, 561, 637):
                assert f"laser_power_{wl}_pct" not in c
            assert c["laser_power_488_pct"] is None

    def test_the_next_run_does_not_inherit_the_preset(self):
        ex = _experiment()
        orch, _ = _volumes(ex, laser_config="561 only")
        orch.client.acquire_volume.reset_mock()

        async def again():
            await orch.start(base_interval_seconds=0.1)
            await asyncio.sleep(0.3)
            await orch.stop("test done")

        asyncio.run(again())
        assert all(
            "laser_config" not in c.kwargs for c in orch.client.acquire_volume.await_args_list
        )

    def test_a_brightfield_run_carries_no_preset(self):
        orch, _ = _volumes(_experiment(), laser_config="488 only")
        c = orch.client
        c.capture_bottom_image = AsyncMock(return_value={"image": None, "image_path": None})
        c.set_laser_config = AsyncMock(return_value={"success": True})
        c.set_led = AsyncMock(return_value={"success": True})

        async def go():
            await orch.start(volumes=False, dic={"enabled": True, "light": "none"})
            await orch.stop("test done")

        asyncio.run(go())
        assert orch._laser_config is None


class TestTheCheckpoint:
    def test_the_preset_and_the_powers_are_read_back(self):
        ex = _experiment()
        ex.embryos["embryo_1"].laser_power_488_pct = 3.0
        ex.embryos["embryo_1"].laser_power_561_pct = 12.5
        orch, _ = _volumes(ex, laser_config="488 and 561")
        doc = orch._serialize_runtime_state()
        assert doc["laser_config"] == "488 and 561"
        assert doc["embryos"]["embryo_1"]["laser_power_561_pct"] == 12.5

        fresh_ex = _experiment()
        fresh = TimelapseOrchestrator(_client(), fresh_ex, store=None, session_id=None)
        fresh._apply_runtime_state(doc)
        assert fresh._laser_config == "488 and 561"
        assert fresh_ex.embryos["embryo_1"].laser_power_488_pct == 3.0
        assert fresh_ex.embryos["embryo_1"].laser_power_561_pct == 12.5
        assert fresh_ex.embryos["embryo_2"].laser_power_561_pct is None

    def test_a_checkpoint_from_before_has_no_preset(self):
        fresh = TimelapseOrchestrator(_client(), _experiment(), store=None, session_id=None)
        fresh._laser_config = "488 only"
        fresh._apply_runtime_state({"base_interval_seconds": 60})
        assert fresh._laser_config is None


class TestTheEmbryo:
    def test_a_new_embryo_has_no_power_of_its_own(self):
        e = _experiment(1).embryos["embryo_1"]
        for wl in (405, 488, 561, 637):
            assert getattr(e, f"laser_power_{wl}_pct") is None

    def test_it_says_its_powers(self):
        e = _experiment(1).embryos["embryo_1"]
        e.laser_power_488_pct, e.laser_power_561_pct = 4.0, 12.5
        d = e.to_dict()
        assert (d["laser_power_488_pct"], d["laser_power_561_pct"]) == (4.0, 12.5)
        assert d["laser_power_405_pct"] is None and d["laser_power_637_pct"] is None


# ---------------------------------------------------------------------------
# The start route
# ---------------------------------------------------------------------------


def _route(experiment=None):
    orch = MagicMock()
    orch.start = AsyncMock(return_value="Started timelapse for 2 embryos")
    orch.enable_monitoring_mode = MagicMock(return_value="on")
    server = MagicMock()
    agent = server.agent_bridge.agent
    agent.timelapse_orchestrator = orch
    agent.experiment = experiment if experiment is not None else _experiment()
    agent.client = MagicMock()
    agent.client.set_laser_config = AsyncMock(return_value={"success": True})
    agent.lightsheet_monitor = None
    app = FastAPI()
    app.include_router(create_router(server))
    app.dependency_overrides[auth.require_control] = lambda: True
    return TestClient(app), orch, agent


def _start(app, **body):
    return app.post("/api/devices/timelapse/start", json={"interval_seconds": 300, **body})


class TestTheRoute:
    def test_the_powers_are_put_on_the_embryos_the_run_images(self):
        ex = _experiment()
        app, orch, _ = _route(ex)
        r = _start(app, laser_powers={"488": 4, "561": 12.5})
        assert r.status_code == 200, r.text
        for e in ex.embryos.values():
            assert (e.laser_power_488_pct, e.laser_power_561_pct) == (4.0, 12.5)
            assert e.laser_power_405_pct is None and e.laser_power_637_pct is None

    def test_only_the_embryos_named(self):
        ex = _experiment()
        app, _, _ = _route(ex)
        assert _start(app, embryo_ids=["embryo_2"], laser_powers={"488": 4}).status_code == 200
        assert ex.embryos["embryo_1"].laser_power_488_pct is None
        assert ex.embryos["embryo_2"].laser_power_488_pct == 4.0

    def test_a_line_left_empty_keeps_the_power_it_was_given(self):
        ex = _experiment()
        ex.embryos["embryo_1"].laser_power_488_pct = 3.0
        app, _, _ = _route(ex)
        for body in ({}, {"laser_powers": None}, {"laser_powers": {"488": None, "561": ""}}):
            assert _start(app, **body).status_code == 200
            assert ex.embryos["embryo_1"].laser_power_488_pct == 3.0
            assert ex.embryos["embryo_1"].laser_power_561_pct is None

    def test_the_limits_are_the_device_layers(self):
        lo, hi = DiSPIMLightSource.POWER_LIMITS_PCT[488]
        app, orch, _ = _route()
        assert _start(app, laser_powers={"488": lo}).status_code == 200
        assert _start(app, laser_powers={"488": hi}).status_code == 200
        r = _start(app, laser_powers={"488": hi + 0.1})
        assert r.status_code == 400
        assert f"{lo}-{hi}" in r.json()["detail"] and "488" in r.json()["detail"]

    @pytest.mark.parametrize(
        "powers",
        [{"488": 50}, {"488": 0}, {"488": -1}, {"561": 101}, {"488": "lots"}, {"488": True}],
    )
    def test_a_power_the_hardware_would_refuse_refuses_the_start(self, powers):
        ex = _experiment()
        app, orch, _ = _route(ex)
        r = _start(app, laser_powers=powers)
        assert r.status_code == 400 and "laser_powers" in r.json()["detail"]
        orch.start.assert_not_awaited()
        assert all(e.laser_power_488_pct is None for e in ex.embryos.values()), (
            "refused, and written onto the embryos anyway"
        )

    @pytest.mark.parametrize("powers", [{"532": 5}, {"green": 5}, [4, 12]])
    def test_a_line_there_is_not_or_a_shape_that_is_not_one(self, powers):
        app, orch, _ = _route()
        assert _start(app, laser_powers=powers).status_code == 400
        orch.start.assert_not_awaited()

    def test_one_bad_line_stops_the_good_one_being_applied(self):
        ex = _experiment()
        app, _, _ = _route(ex)
        assert _start(app, laser_powers={"561": 10, "488": 50}).status_code == 400
        assert all(e.laser_power_561_pct is None for e in ex.embryos.values())

    def test_the_preset_is_set_on_the_controller_and_carried_by_the_run(self):
        app, orch, agent = _route()
        assert _start(app, laser_config="488 only").status_code == 200
        agent.client.set_laser_config.assert_awaited_once_with("488 only")
        assert orch.start.await_args.kwargs["laser_config"] == "488 only"

    def test_with_no_preset_the_run_is_told_none(self):
        app, orch, agent = _route()
        assert _start(app).status_code == 200
        assert "laser_config" not in orch.start.await_args.kwargs
        agent.client.set_laser_config.assert_not_awaited()

    def test_a_brightfield_run_sets_no_power_whatever_it_is_sent(self):
        ex = _experiment()
        app, orch, _ = _route(ex)
        r = _start(
            app,
            volumes=False,
            dic={"enabled": True, "light": "led"},
            laser_powers={"488": 4},
            laser_config="488 only",
        )
        assert r.status_code == 200, r.text
        assert all(e.laser_power_488_pct is None for e in ex.embryos.values())
        assert "laser_config" not in orch.start.await_args.kwargs


# ---------------------------------------------------------------------------
# The Acquisition pane
# ---------------------------------------------------------------------------

INDEX = (WEB / "templates" / "index.html").read_text(encoding="utf-8")
OPERATE = (WEB / "static" / "js" / "operate.js").read_text(encoding="utf-8")


def _js(name: str) -> str:
    body = OPERATE[OPERATE.index(f"function {name}(") :]
    return body[: body.index("\n    }")]


class TestThePane:
    def test_the_rows_live_in_the_volume_channel(self):
        body = INDEX[INDEX.index('id="op-plan-spim-body"') :]
        body = body[: body.index('id="op-plan-dic"')]
        assert 'id="op-plan-powers"' in body
        assert body.index('id="op-plan-laser"') < body.index('id="op-plan-powers"')

    def test_which_rows_there_are_follows_the_preset(self):
        assert "AcquisitionPlan.linesOf(($('op-plan-laser') || {}).value)" in _js("renderPowerRows")

    def test_the_bounds_come_from_the_server(self):
        assert "getJSON('/api/devices/laser/limits')" in _js("loadLaserPresets")
        assert "_laserLimits[wl]" in _js("renderPowerRows")
        for bound in ("2.0", "6.0", 'min="2"', 'max="6"'):
            assert bound not in _js("renderPowerRows"), "a limit is written into the pane"

    def test_typing_is_not_interrupted_by_a_redraw(self):
        body = _js("renderPowerRows")
        assert "if (host.dataset.lines === key) return;" in body
        assert "typed[inp.dataset.planPower] = inp.value;" in body

    def test_the_form_reads_the_rows_and_a_restored_plan_clears_them(self):
        assert "laserPowers[inp.dataset.planPower] = inp.value;" in _js("readPlan")
        assert "inp.value = pct != null ? pct : '';" in _js("fillPlan")

    def test_the_plan_is_checked_against_the_limits_before_it_is_sent(self):
        assert "AcquisitionPlan.validate(plan, ids, _laserLimits)" in _js("startRun")
        assert "AcquisitionPlan.validate(plan, subjectIds(), _laserLimits)" in _js("renderPlan")
