"""A timelapse of the field in brightfield: bottom camera, no SPIM volumes.

A Gently timelapse has two channels — the SPIM volume of each embryo, and one
bottom-camera frame of the whole field — and the second could only ride along
with the first. A run followed in transmitted light alone could not be
started: the start asked for embryos, the gate asked for their calibration,
and the loop ended when the embryos did.

This is the second channel run by itself. What is tested is what follows from
there being no volumes, because each is a way the run could quietly do the
wrong thing on a microscope:

- nothing of the SPIM head is driven, and the lasers are put to ALL OFF
- no embryo has to exist, and none has to be calibrated
- the run ends on a count, a duration or a stop — never on a stage, which is
  read from a volume
- a frame that cannot be taken is the experiment failing, not a side channel
- every frame is lit the same: the LED's brightness is set before each one

Everything runs against a fake microscope, through the real loop.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import gently.ui.web.auth as auth
from gently.app.orchestration.timelapse import TimelapseOrchestrator
from gently.app.orchestration.timelapse_models import DicOverview, TimelapseStatus
from gently.harness.state import ExperimentState
from gently.ui.web.routes.data import create_router

WEB = Path(__file__).resolve().parents[1] / "gently" / "ui" / "web"
FRAME = (np.arange(64 * 64, dtype=np.uint16) % 4096).reshape(64, 64)
LED = {"enabled": True, "light": "led"}


@pytest.fixture(autouse=True, scope="module")
def _organism():
    """ "hatching" is looked up in the organism's stage table, which nothing
    loads in a bare test process."""
    from gently.organisms import load_organism

    load_organism("celegans")


def _client(*, frames=True, laser_off=True):
    """A microscope that answers at once, and writes down what it was asked."""
    c = MagicMock()
    c.log = []

    def said(name, result=None):
        async def call(*args, **kwargs):
            c.log.append((name, *args))
            return {"success": True} if result is None else result

        return AsyncMock(side_effect=call)

    async def capture(use_led=False, exposure_ms=None):
        c.log.append(("capture", exposure_ms))
        if frames is True or (callable(frames) and frames(len(c.log))):
            return {"image": FRAME, "image_path": None}
        # What the client hands back when the camera gave it nothing.
        return {"image": np.zeros((100, 100), dtype=np.uint16), "image_path": None}

    async def lasers(config):
        c.log.append(("lasers", config))
        if not laser_off:
            raise RuntimeError("PLogic did not answer")
        return {"success": True}

    c.capture_bottom_image = AsyncMock(side_effect=capture)
    c.set_laser_config = AsyncMock(side_effect=lasers)
    c.move_to_position = said("move")
    c.set_led = said("led")
    c.set_led_intensity = said("brightness")
    c.set_room_light = said("room")
    c.get_room_light_status = AsyncMock(return_value={"success": True, "state": "off"})
    c.acquire_volume = said("volume")
    c.capture_lightsheet_image = said("snap")
    return c


def _experiment(embryos=2, calibrated=False):
    ex = ExperimentState()
    for i in range(embryos):
        ex.add_embryo(
            f"embryo_{i + 1}",
            position={"x": 100.0 * i, "y": 40.0},
            # What the calibration gate reads as a fit: a slope.
            calibration={"galvo_center": 0.0, "slope_um_per_deg": 50.0} if calibrated else None,
        )
    return ex


def _orch(client=None, *, embryos=2):
    orch = TimelapseOrchestrator(
        client or _client(), _experiment(embryos), store=None, session_id=None
    )
    orch._dic_light_settle_s = 0.0
    return orch


def _names(orch):
    return [entry[0] for entry in orch.client.log]


async def _go(orch, seconds=0.5, *, stop=True, **kwargs):
    kwargs.setdefault("base_interval_seconds", 0.1)
    kwargs.setdefault("dic", dict(LED))
    msg = await orch.start(volumes=False, **kwargs)
    if not msg.startswith("Started"):
        return msg
    await asyncio.sleep(seconds)
    if stop:
        await orch.stop("test done")
    return msg


# ---------------------------------------------------------------------------
# What the run does, and does not, drive
# ---------------------------------------------------------------------------


class TestOnlyTheBottomCamera:
    def test_frames_are_taken_on_the_cadence(self):
        orch = _orch()
        msg = asyncio.run(_go(orch, 0.55))
        assert msg.startswith("Started brightfield timelapse"), msg
        assert "no SPIM volumes" in msg
        assert 3 <= _names(orch).count("capture") <= 8

    def test_the_spim_head_is_never_driven(self):
        orch = _orch()
        asyncio.run(_go(orch, 0.5))
        assert "volume" not in _names(orch)
        assert "snap" not in _names(orch)

    def test_the_lasers_are_put_to_all_off_before_the_first_frame(self):
        orch = _orch()
        asyncio.run(_go(orch, 0.3))
        names = _names(orch)
        assert ("lasers", "ALL OFF") in orch.client.log
        assert names.index("lasers") < names.index("capture")
        assert [e for e in orch.client.log if e[0] == "lasers"] == [("lasers", "ALL OFF")]

    def test_lasers_that_will_not_answer_do_not_stop_a_run_that_uses_none(self):
        orch = _orch(_client(laser_off=False))
        msg = asyncio.run(_go(orch, 0.3))
        assert msg.startswith("Started"), msg
        assert "capture" in _names(orch)

    def test_the_led_is_opened_for_the_frame_and_closed_after_every_time(self):
        orch = _orch()
        asyncio.run(_go(orch, 0.45))
        seq = [e for e in orch.client.log if e[0] in ("led", "capture")]
        frames = [i for i, e in enumerate(seq) if e[0] == "capture"]
        assert frames
        for i in frames:
            assert seq[i - 1] == ("led", "Open"), seq
            assert seq[i + 1] == ("led", "Closed"), seq
        # Dark between frames: the sample is lit for the exposure, not the night.
        assert seq[-1] == ("led", "Closed")

    def test_a_stop_while_the_led_is_coming_on_does_not_leave_it_on(self):
        """Between the LED opening and the frame there is a settle. A stop
        that landed in it left the LED open, with no run left to close it."""
        orch = _orch()
        orch._dic_light_settle_s = 5.0

        async def go():
            await orch.start(volumes=False, base_interval_seconds=60, dic=dict(LED))
            await asyncio.sleep(0.2)  # the LED is open and settling
            assert orch.client.log[-1] == ("led", "Open"), orch.client.log
            await orch.stop("operator")

        asyncio.run(go())
        assert orch.client.log[-1] == ("led", "Closed"), orch.client.log
        assert "capture" not in _names(orch)

    def test_the_status_says_what_kind_of_run_it_is(self):
        orch = _orch()

        async def go():
            await _go(orch, 0.35, stop=False)
            state = orch.get_status()
            await orch.stop("test done")
            return state

        state = asyncio.run(go())
        d = state.to_dict()
        assert d["volumes"] is False
        assert d["dic"]["frames"] >= 1
        assert d["seconds_until_next_round"] is not None, "the next frame has a time"
        assert d["active_embryos"] == 0

    def test_a_volume_run_still_says_it_takes_volumes(self):
        orch = TimelapseOrchestrator(
            _client(), _experiment(calibrated=True), store=None, session_id=None
        )

        async def go():
            msg = await orch.start(base_interval_seconds=0.1)
            assert msg.startswith("Started timelapse for 2 embryos"), msg
            await asyncio.sleep(0.3)
            state = orch.get_status().to_dict()
            await orch.stop("test done")
            return state

        state = asyncio.run(go())
        assert state["volumes"] is True
        assert "volume" in _names(orch)

    def test_a_volume_run_after_a_brightfield_one_takes_volumes_again(self):
        """The flag is the run's, not the orchestrator's."""
        orch = TimelapseOrchestrator(
            _client(), _experiment(calibrated=True), store=None, session_id=None
        )
        orch._dic_light_settle_s = 0.0

        async def go():
            await _go(orch, 0.2)
            orch.client.log.clear()
            await orch.start(base_interval_seconds=0.1)
            await asyncio.sleep(0.3)
            await orch.stop("test done")

        asyncio.run(go())
        assert "volume" in _names(orch)
        assert "capture" not in _names(orch), "the last run's overview came along"


# ---------------------------------------------------------------------------
# No embryo is a subject
# ---------------------------------------------------------------------------


class TestTheFieldNotTheEmbryos:
    def test_it_starts_with_no_embryo_registered(self):
        orch = _orch(embryos=0)
        msg = asyncio.run(_go(orch, 0.3))
        assert msg.startswith("Started"), msg
        assert "capture" in _names(orch)

    def test_with_no_embryo_the_stage_is_left_where_it_is(self):
        orch = _orch(embryos=0)
        asyncio.run(_go(orch, 0.3))
        assert "move" not in _names(orch)

    def test_embryos_say_only_where_the_field_is(self):
        orch = _orch(embryos=2)
        asyncio.run(_go(orch, 0.3))
        moves = [e for e in orch.client.log if e[0] == "move"]
        assert moves and set(moves) == {("move", 50.0, 40.0)}, "the centroid, every frame"

    def test_a_pinned_position_wins(self):
        orch = _orch(embryos=2)
        asyncio.run(_go(orch, 0.3, dic={**LED, "position": {"x": -7.0, "y": 9.0}}))
        assert {e for e in orch.client.log if e[0] == "move"} == {("move", -7.0, 9.0)}

    def test_uncalibrated_embryos_are_no_obstacle_and_are_left_untouched(self):
        orch = _orch(embryos=2)
        before = {
            e.id: (e.timepoints_acquired, e.is_complete) for e in orch.experiment.embryos.values()
        }
        asyncio.run(_go(orch, 0.3))
        after = {
            e.id: (e.timepoints_acquired, e.is_complete) for e in orch.experiment.embryos.values()
        }
        assert after == before


# ---------------------------------------------------------------------------
# How it ends
# ---------------------------------------------------------------------------


class TestHowItEnds:
    def test_after_a_count_of_frames(self):
        orch = _orch()
        asyncio.run(_go(orch, 0.9, stop=False, stop_condition="timepoints:3"))
        assert orch._status == TimelapseStatus.COMPLETED
        assert _names(orch).count("capture") == 3
        assert orch.get_status().dic["frames"] == 3

    def test_the_count_can_come_beside_the_word(self):
        orch = _orch()
        asyncio.run(_go(orch, 0.7, stop=False, stop_condition="timepoints", condition_value=2))
        assert orch._status == TimelapseStatus.COMPLETED
        assert _names(orch).count("capture") == 2

    def test_after_a_duration(self):
        orch = _orch()
        # 0.0001 h is 0.36 s.
        asyncio.run(_go(orch, 1.0, stop=False, stop_condition="duration:0.0001h"))
        assert orch._status == TimelapseStatus.COMPLETED
        assert 2 <= _names(orch).count("capture") <= 6

    def test_a_manual_run_goes_on_until_it_is_stopped(self):
        orch = _orch()

        async def go():
            await _go(orch, 0.4, stop=False)
            running = orch._status
            said = await orch.stop("operator")
            return running, said

        running, said = asyncio.run(go())
        assert running == TimelapseStatus.RUNNING
        assert said.startswith("Timelapse stopped")
        assert orch._status == TimelapseStatus.IDLE

    @pytest.mark.parametrize(
        "ending", ["hatching", "comma", "all_test_hatched", "hatching|timepoints:5"]
    )
    def test_it_cannot_end_on_a_stage(self, ending):
        orch = _orch()
        msg = asyncio.run(_go(orch, 0.1, stop_condition=ending))
        assert "cannot end on" in msg and "timepoints:N" in msg
        assert orch._status != TimelapseStatus.RUNNING
        assert orch.client.log == [], "refused before anything was driven"

    def test_without_the_channel_there_is_nothing_to_image(self):
        orch = _orch()
        for dic in (None, {"enabled": False}):
            msg = asyncio.run(_go(orch, 0.1, dic=dic))
            assert msg.startswith("Nothing to image"), msg
        assert orch.client.log == []

    def test_a_run_already_going_is_not_started_over(self):
        orch = _orch()

        async def go():
            await _go(orch, 0.1, stop=False)
            again = await orch.start(volumes=False, dic=dict(LED))
            await orch.stop("test done")
            return again

        assert asyncio.run(go()).startswith("Timelapse already running")


# ---------------------------------------------------------------------------
# A frame that fails
# ---------------------------------------------------------------------------


class TestAFrameThatFails:
    def test_three_in_a_row_stop_the_run_and_say_so(self):
        orch = _orch(_client(frames=False))
        asyncio.run(_go(orch, 0.8, stop=False))
        assert orch._status == TimelapseStatus.FAILED
        assert "3 brightfield frames in a row" in orch.get_status().error_message
        assert _names(orch).count("capture") == 3

    def test_a_frame_that_was_not_taken_is_not_counted(self):
        orch = _orch(_client(frames=False))
        asyncio.run(_go(orch, 0.8, stop=False, stop_condition="timepoints:2"))
        assert orch.get_status().dic["frames"] == 0

    def test_the_led_is_closed_even_after_a_frame_that_failed(self):
        orch = _orch(_client(frames=False))
        asyncio.run(_go(orch, 0.8, stop=False))
        leds = [e for e in orch.client.log if e[0] == "led"]
        assert leds and leds[-1] == ("led", "Closed")

    def test_one_bad_frame_is_not_the_end(self):
        # Every third call to the camera gives nothing; never three running.
        calls = {"n": 0}

        def sometimes(_):
            calls["n"] += 1
            return calls["n"] % 3 != 0

        orch = _orch(_client(frames=sometimes))
        asyncio.run(_go(orch, 0.9))
        assert orch._status == TimelapseStatus.IDLE, "stopped by the test, not by failing"
        assert orch.get_status().dic["frames"] >= 3

    def test_in_a_volume_run_a_failed_overview_is_still_only_logged(self):
        """The rule is the brightfield run's. Beside volumes the overview is a
        second channel, and must never take the volumes down."""
        orch = TimelapseOrchestrator(
            _client(frames=False), _experiment(calibrated=True), store=None, session_id=None
        )
        orch._dic_light_settle_s = 0.0

        async def go():
            await orch.start(base_interval_seconds=0.1, dic={**LED, "every_seconds": 0.1})
            await asyncio.sleep(0.7)
            status = orch._status
            await orch.stop("test done")
            return status

        assert asyncio.run(go()) == TimelapseStatus.RUNNING
        assert _names(orch).count("volume") >= 3


# ---------------------------------------------------------------------------
# The LED's brightness
# ---------------------------------------------------------------------------


class TestTheLedsBrightness:
    def test_it_is_set_before_every_frame_and_before_the_led_opens(self):
        orch = _orch()
        asyncio.run(_go(orch, 0.45, dic={**LED, "led_intensity_pct": 40}))
        seq = [e for e in orch.client.log if e[0] in ("brightness", "led", "capture")]
        frames = [i for i, e in enumerate(seq) if e[0] == "capture"]
        assert len(frames) >= 2
        for i in frames:
            assert seq[i - 2 : i] == [("brightness", 40), ("led", "Open")], seq

    def test_without_one_the_led_is_left_as_it_is(self):
        orch = _orch()
        asyncio.run(_go(orch, 0.3))
        assert "brightness" not in _names(orch)
        assert ("led", "Open") in orch.client.log

    def test_it_means_nothing_under_the_room_light(self):
        orch = _orch()
        asyncio.run(_go(orch, 0.3, dic={"enabled": True, "light": "room", "led_intensity_pct": 40}))
        assert "brightness" not in _names(orch)
        assert "led" not in _names(orch)

    def test_a_brightness_that_cannot_be_set_does_not_cost_the_frame(self):
        c = _client()
        c.set_led_intensity = AsyncMock(return_value={"success": False, "error": "no LED"})
        orch = _orch(c)
        asyncio.run(_go(orch, 0.3, dic={**LED, "led_intensity_pct": 40}))
        assert "capture" in _names(orch)

    def test_it_is_in_the_plan_and_survives_the_round_trip(self):
        dic = DicOverview.from_dict({**LED, "led_intensity_pct": 40})
        assert dic.led_intensity_pct == 40
        assert DicOverview.from_dict(dic.to_dict()).led_intensity_pct == 40

    @pytest.mark.parametrize("bad", [0, 101, -5, 12.5, "bright", True, None])
    def test_what_is_not_a_brightness_is_none(self, bad):
        assert DicOverview.from_dict({**LED, "led_intensity_pct": bad}).led_intensity_pct is None

    def test_a_plan_from_before_it_existed_has_none(self):
        assert DicOverview.from_dict({"enabled": True, "light": "led"}).led_intensity_pct is None


# ---------------------------------------------------------------------------
# The checkpoint, and carrying on
# ---------------------------------------------------------------------------


class TestCarryingOn:
    def _restored(self, orch):
        doc = orch._serialize_runtime_state()
        fresh = _orch()
        fresh._apply_runtime_state(doc)
        return doc, fresh

    def test_the_checkpoint_says_it_is_a_brightfield_run_and_how_it_ends(self):
        orch = _orch()
        asyncio.run(
            _go(orch, 0.3, stop_condition="timepoints:50", dic={**LED, "led_intensity_pct": 30})
        )
        doc, fresh = self._restored(orch)
        assert doc["volumes"] is False
        assert fresh._volumes is False
        assert fresh._run_stop.describe() == "50 timepoints"
        assert fresh._dic.led_intensity_pct == 30
        assert fresh._dic_frames == orch._dic_frames >= 1

    def test_a_checkpoint_from_before_is_a_volume_run(self):
        fresh = _orch()
        fresh._apply_runtime_state({"base_interval_seconds": 60})
        assert fresh._volumes is True
        assert fresh._run_stop is None

    def test_a_stopped_run_can_be_carried_on_and_keeps_its_count(self):
        orch = _orch()

        async def go():
            await _go(orch, 0.3, stop_condition="timepoints:50")
            before = orch._dic_frames
            assert orch.can_continue()
            orch.client.log.clear()
            said = await orch.continue_run()
            await asyncio.sleep(0.3)
            await orch.stop("test done")
            return before, said

        before, said = asyncio.run(go())
        assert said.startswith(f"Continued brightfield timelapse from frame {before}")
        assert orch._dic_frames > before
        assert ("lasers", "ALL OFF") in orch.client.log, "the lasers are put off again"
        assert "volume" not in _names(orch)

    def test_a_run_that_reached_its_ending_cannot(self):
        orch = _orch()
        asyncio.run(_go(orch, 0.7, stop=False, stop_condition="timepoints:2"))
        assert orch._status == TimelapseStatus.COMPLETED
        assert not orch.can_continue()


# ---------------------------------------------------------------------------
# The start route
# ---------------------------------------------------------------------------


def _route(experiment=None, start_return="Started brightfield timelapse"):
    orch = MagicMock()
    orch.start = AsyncMock(return_value=start_return)
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


BODY = {
    "interval_seconds": 300,
    "stop_condition": "timepoints:12",
    "volumes": False,
    "dic": {**LED, "led_intensity_pct": 40, "exposure_ms": 20},
}


class TestTheRoute:
    def test_it_starts_a_run_with_no_volumes(self):
        app, orch, _ = _route()
        r = app.post("/api/devices/timelapse/start", json=BODY)
        assert r.status_code == 200, r.text
        kwargs = orch.start.await_args.kwargs
        assert kwargs["volumes"] is False
        assert kwargs["dic"]["led_intensity_pct"] == 40
        assert kwargs["dic"]["light"] == "led"
        assert kwargs["stop_condition"] == "timepoints:12"

    def test_uncalibrated_embryos_do_not_refuse_it(self):
        """The gate is about scan geometry, and nothing here is scanned."""
        app, orch, _ = _route(_experiment(calibrated=False))
        assert app.post("/api/devices/timelapse/start", json=BODY).status_code == 200
        # The same embryos, asked for volumes, are refused.
        r = app.post("/api/devices/timelapse/start", json={"interval_seconds": 300})
        assert r.status_code == 409

    def test_it_sets_no_laser_preset_and_no_volume_settings(self):
        ex = _experiment()
        before = {e.id: (e.exposure_ms, e.num_slices) for e in ex.embryos.values()}
        app, orch, agent = _route(ex)
        body = {**BODY, "laser_config": "488 and 561", "exposure_ms": 77, "num_slices": 9}
        assert app.post("/api/devices/timelapse/start", json=body).status_code == 200
        agent.client.set_laser_config.assert_not_awaited()
        assert {e.id: (e.exposure_ms, e.num_slices) for e in ex.embryos.values()} == before

    def test_it_installs_no_stage_watching_and_no_per_embryo_endings(self):
        app, orch, _ = _route()
        body = {
            **BODY,
            "monitoring_mode": "expression_monitoring",
            "stop_conditions": {"embryo_1": "hatching"},
        }
        assert app.post("/api/devices/timelapse/start", json=body).status_code == 200
        orch.enable_monitoring_mode.assert_not_called()
        assert "stop_conditions" not in orch.start.await_args.kwargs

    @pytest.mark.parametrize("dic", [None, {"enabled": False}])
    def test_with_nothing_to_image_it_is_a_400(self, dic):
        app, orch, _ = _route()
        r = app.post("/api/devices/timelapse/start", json={**BODY, "dic": dic})
        assert r.status_code == 400 and "Nothing to image" in r.json()["detail"]
        orch.start.assert_not_awaited()

    @pytest.mark.parametrize("ending", ["hatching", "comma", "hatching+3", "timepoints:5|hatching"])
    def test_an_ending_read_from_a_stage_is_a_400(self, ending):
        app, orch, _ = _route()
        r = app.post("/api/devices/timelapse/start", json={**BODY, "stop_condition": ending})
        assert r.status_code == 400 and "cannot end on" in r.json()["detail"]
        orch.start.assert_not_awaited()

    @pytest.mark.parametrize(
        "ending", ["manual", "timepoints:5", "duration:6h", "timepoints:5|duration:6h"]
    )
    def test_the_endings_it_can_have(self, ending):
        app, orch, _ = _route()
        r = app.post("/api/devices/timelapse/start", json={**BODY, "stop_condition": ending})
        assert r.status_code == 200, r.text

    @pytest.mark.parametrize("pct", [0, 101, 12.5, "bright", True])
    def test_a_brightness_that_is_not_one_is_refused_not_dropped(self, pct):
        app, orch, _ = _route()
        body = {**BODY, "dic": {**LED, "led_intensity_pct": pct}}
        r = app.post("/api/devices/timelapse/start", json=body)
        assert r.status_code == 400 and "led_intensity_pct" in r.json()["detail"]
        orch.start.assert_not_awaited()

    @pytest.mark.parametrize("volumes", [None, True, "false", 0])
    def test_only_the_word_false_turns_the_volumes_off(self, volumes):
        app, orch, _ = _route(_experiment(calibrated=True))
        r = app.post(
            "/api/devices/timelapse/start", json={"interval_seconds": 300, "volumes": volumes}
        )
        assert r.status_code == 200, r.text
        assert "volumes" not in orch.start.await_args.kwargs


# ---------------------------------------------------------------------------
# The agent's tool
# ---------------------------------------------------------------------------


def _tool(name):
    import gently.app.tools.timelapse_tools  # noqa: F401  (registers on import)
    from gently.harness.tools.registry import get_tool_registry

    return get_tool_registry()._tools[name].handler


def _agent_context():
    agent = MagicMock()
    agent.timelapse_orchestrator.start = AsyncMock(return_value="Started brightfield timelapse")
    return {"agent": agent}, agent.timelapse_orchestrator


class TestTheTool:
    async def test_it_starts_a_run_with_no_volumes_under_the_led(self):
        ctx, orch = _agent_context()
        out = await _tool("start_brightfield_timelapse")(
            interval_seconds=300,
            stop_condition="duration",
            condition_value=12,
            led_intensity_pct=40,
            context=ctx,
        )
        assert out.startswith("Started brightfield timelapse")
        kwargs = orch.start.await_args.kwargs
        assert kwargs["volumes"] is False
        assert kwargs["base_interval_seconds"] == 300
        assert (kwargs["stop_condition"], kwargs["condition_value"]) == ("duration", 12)
        assert kwargs["dic"] == {
            "enabled": True,
            "light": "led",
            "exposure_ms": None,
            "led_intensity_pct": 40,
        }
        assert "embryo_ids" not in kwargs

    @pytest.mark.parametrize(
        "kwargs, says",
        [
            ({"light": "sunlight"}, "light must be"),
            ({"led_intensity_pct": 0}, "whole percent"),
            ({"led_intensity_pct": 150}, "whole percent"),
            ({"light": "room", "led_intensity_pct": 40}, "only applies under"),
            ({"interval_seconds": 0}, "must be positive"),
        ],
    )
    async def test_what_it_refuses_it_refuses_before_the_orchestrator(self, kwargs, says):
        ctx, orch = _agent_context()
        out = await _tool("start_brightfield_timelapse")(context=ctx, **kwargs)
        assert out.startswith("Error") and says in out
        orch.start.assert_not_awaited()

    def test_the_status_tool_does_not_report_an_idle_looking_run(self):
        orch = _orch()

        async def go():
            await _go(orch, 0.35, stop=False)
            agent = MagicMock()
            agent.timelapse_orchestrator = orch
            out = _tool("get_timelapse_status")(context={"agent": agent})
            await orch.stop("test done")
            return out

        out = asyncio.run(go())
        assert "Brightfield-only run" in out
        assert "Frames acquired:" in out
        assert "Active embryos" not in out
        assert "Total timepoints acquired" not in out


# ---------------------------------------------------------------------------
# The Acquisition pane
# ---------------------------------------------------------------------------

INDEX = (WEB / "templates" / "index.html").read_text(encoding="utf-8")
OPERATE = (WEB / "static" / "js" / "operate.js").read_text(encoding="utf-8")


def _js(name: str) -> str:
    body = OPERATE[OPERATE.index(f"function {name}(") :]
    return body[: body.index("\n    }")]


class TestThePane:
    def test_the_volume_channel_can_be_switched_off_and_starts_on(self):
        assert '<input type="checkbox" id="op-plan-spim" checked>' in INDEX

    def test_the_form_reads_it_and_a_page_without_it_is_a_volume_run(self):
        assert "volumes: !($('op-plan-spim') && !$('op-plan-spim').checked)" in _js("readPlan")

    def test_the_brightness_is_asked_for_and_read(self):
        assert 'id="op-plan-dic-led"' in INDEX and 'min="1" max="100"' in INDEX
        assert "dicLedPct: v('op-plan-dic-led')" in _js("readPlan")

    def test_a_restored_plan_clears_a_brightness_it_does_not_have(self):
        assert "led.value = plan.dic.ledPct != null ? plan.dic.ledPct : ''" in _js("fillPlan")

    def test_a_brightfield_run_is_not_refused_for_having_no_embryos(self):
        assert "if (plan.spim.enabled && !haveSubjects()) return;" in _js("startRun")

    def test_what_is_about_volumes_is_put_away_with_them(self):
        body = _js("renderChannels")
        for host in (
            "op-plan-spim-body",
            "op-plan-dic-every-field",
            "op-plan-overrides",
            "op-plan-watch-sec",
            "op-plan-watch-field",
        ):
            assert f"show('{host}', volumes)" in body, host
            assert f'id="{host}"' in INDEX, host

    def test_the_endings_on_offer_are_the_ones_the_run_can_have(self):
        body = _js("renderChannels")
        assert "AcquisitionPlan.BRIGHTFIELD_STOPS.includes(k)" in body
        assert "if (!ok(stop.value)) stop.value = 'manual';" in body

    def test_the_run_view_counts_frames_not_volumes(self):
        body = _js("renderRun")
        assert "st.volumes === false" in body
        assert "'brightfield, no SPIM volumes'" in body


# ---------------------------------------------------------------------------
# On disk
# ---------------------------------------------------------------------------


class TestOnDisk:
    """Against the real FileStore: a run is what it leaves behind."""

    @pytest.fixture
    def store(self, tmp_path):
        from gently.core.file_store import FileStore

        fs = FileStore(root=tmp_path)
        fs.create_session("s1")
        return fs

    def _run(self, store, **kwargs):
        orch = TimelapseOrchestrator(_client(), _experiment(), store=store, session_id="s1")
        orch._dic_light_settle_s = 0.0
        asyncio.run(_go(orch, 0.9, stop=False, **kwargs))
        return orch

    def test_every_frame_is_filed_and_says_how_it_was_lit(self, store):
        self._run(store, stop_condition="timepoints:3", dic={**LED, "led_intensity_pct": 40})
        frames = store.list_snapshots("s1", "dic")
        assert len(frames) == 3
        assert sorted(f["metadata"]["frame"] for f in frames) == [1, 2, 3]
        for f in frames:
            assert f["metadata"]["light"] == "led"
            assert f["metadata"]["led_intensity_pct"] == 40
            assert Path(f["file_path"]).exists()

    def test_no_embryo_folder_gets_a_volume(self, store, tmp_path):
        self._run(store, stop_condition="timepoints:2")
        assert not list(tmp_path.rglob("volumes/*.tif"))

    def test_the_checkpoint_is_read_back_as_the_run_it_was(self, store):
        ran = self._run(store, stop_condition="timepoints:2")
        fresh = TimelapseOrchestrator(_client(), _experiment(), store=store, session_id="s1")
        assert fresh.load_state().startswith("Restored")
        assert fresh._volumes is False
        assert fresh._dic_frames == ran._dic_frames == 2
        assert fresh._ended == "completed"
        assert not fresh.can_continue()
