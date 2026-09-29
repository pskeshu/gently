"""The DIC overview says which light it is taken under.

"on dic light source, perhaps that can be configured when setting up the dic
- where appropriate. usually we use the room light."

The bottom camera drives no light of its own (since June, by decision), so
the overview was taken under whatever light happened to be on: a night's
frames came out nearly dark. The plan now names the light. It goes on for the
frame and off again before the fluorescence volumes that follow; a light the
operator already had on is left on.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from gently.app.orchestration.timelapse import TimelapseOrchestrator
from gently.app.orchestration.timelapse_models import DicOverview
from gently.core.file_store import FileStore
from gently.harness.state import ExperimentState
from gently.ui.web import auth
from gently.ui.web.routes.data import create_router

FRAME = (np.arange(512 * 512, dtype=np.uint16) % 4096).reshape(512, 512)
WEB = Path(__file__).resolve().parents[1] / "gently" / "ui" / "web"
HTML = (WEB / "templates" / "index.html").read_text(encoding="utf-8")
OPERATE = (WEB / "static" / "js" / "operate.js").read_text(encoding="utf-8")


@pytest.fixture
def store(tmp_path):
    fs = FileStore(root=tmp_path)
    fs.create_session("s1")
    return fs


def _rig(room="off", room_works=True, off_works=True):
    """A microscope that records, in order, everything it is asked to do."""
    log: list[str] = []
    state = {"room": room}
    c = MagicMock()
    c.log = log

    async def move(x, y):
        log.append("move")
        return {"success": True}

    async def volume(*a, **k):
        log.append("volume")
        return {"success": True, "volume": None}

    async def capture(use_led=False, exposure_ms=None):
        log.append(f"capture(use_led={use_led}, room={state['room']})")
        return {"image": FRAME, "image_path": None}

    async def room_status():
        return {"success": True, "available": True, "state": state["room"]}

    async def set_room(s):
        log.append(f"room {s}")
        if s == "on" and not room_works:
            return {"success": False, "error": "room_light device not configured"}
        if s == "off" and not off_works:
            return {"success": False, "error": "BLE timeout"}
        state["room"] = s
        return {"success": True}

    async def set_led(s):
        log.append(f"led {s}")
        return {"success": True}

    c.move_to_position = AsyncMock(side_effect=move)
    c.acquire_volume = AsyncMock(side_effect=volume)
    c.capture_bottom_image = AsyncMock(side_effect=capture)
    c.get_room_light_status = AsyncMock(side_effect=room_status)
    c.set_room_light = AsyncMock(side_effect=set_room)
    c.set_led = AsyncMock(side_effect=set_led)
    return c


def _experiment():
    ex = ExperimentState()
    ex.add_embryo("embryo_1", position={"x": 0.0, "y": 0.0}, calibration={"galvo_center": 0.0})
    return ex


def _round(client, store, **dic):
    orch = TimelapseOrchestrator(client, _experiment(), store=store, session_id="s1")
    orch._dic_light_settle_s = 0.0

    async def go():
        msg = await orch.start(
            base_interval_seconds=100, dic=DicOverview(enabled=True, every_seconds=100, **dic)
        )
        assert msg.startswith("Started"), msg
        await asyncio.sleep(0.4)
        await orch.stop("test done")

    asyncio.run(go())
    return [e for e in client.log if e != "move"]


def test_the_room_light_is_the_default():
    assert DicOverview().light == "room"
    assert DicOverview.from_dict({"enabled": True}).light == "room"


def test_a_plan_from_before_the_light_existed_does_not_choose_the_led():
    """`use_led: true` was in every plan and did nothing. It must not start
    flashing the LED at a sample now."""
    old = DicOverview.from_dict({"enabled": True, "use_led": True})
    assert old.light == "room"
    assert "use_led" not in old.to_dict()


def test_an_unknown_light_is_the_room_light():
    assert DicOverview.from_dict({"enabled": True, "light": "sunlight"}).light == "room"


def test_the_light_survives_the_checkpoint():
    d = DicOverview(enabled=True, light="led").to_dict()
    assert d["light"] == "led" and DicOverview.from_dict(d).light == "led"


def test_the_room_light_goes_on_for_the_frame_and_off_before_the_volume(store):
    log = _round(_rig(room="off"), store)
    assert log[:3] == ["room on", "capture(use_led=False, room=on)", "room off"], log
    assert log.index("room off") < log.index("volume"), "the volume was imaged with the light on"


def test_a_room_light_that_was_already_on_is_left_on(store):
    log = _round(_rig(room="on"), store)
    assert "room on" not in log and "room off" not in log, log
    assert log[0] == "capture(use_led=False, room=on)"


def test_the_led_opens_for_the_frame_only(store):
    log = _round(_rig(), store, light="led")
    assert log[:3] == ["led Open", "capture(use_led=False, room=off)", "led Closed"], log
    assert not any(e.startswith("room") for e in log)


def test_as_it_is_touches_no_light(store):
    log = _round(_rig(), store, light="none")
    assert not any(e.startswith(("room", "led")) for e in log), log
    assert log[0].startswith("capture")


def test_the_light_goes_off_even_when_the_capture_fails(store):
    client = _rig()

    async def boom(use_led=False, exposure_ms=None):
        client.log.append("capture failed")
        raise RuntimeError("camera busy")

    client.capture_bottom_image = AsyncMock(side_effect=boom)
    log = _round(client, store)
    assert log[:3] == ["room on", "capture failed", "room off"], log
    assert "volume" in log, "a failed overview took the volumes down"


def test_no_room_light_on_this_rig_takes_the_frame_as_it_is(store, caplog):
    with caplog.at_level(logging.WARNING):
        log = _round(_rig(room_works=False), store)
    assert log[0] == "room on" and log[1].startswith("capture"), log
    assert "room off" not in log, "it switched off a light it never switched on"
    assert any("did not come on" in r.message for r in caplog.records)
    assert len(store.list_snapshots("s1", "dic")) == 1


def test_a_light_that_will_not_go_off_is_an_error_not_a_whisper(store, caplog):
    with caplog.at_level(logging.WARNING):
        log = _round(_rig(off_works=False), store)
    assert log.count("room off") == 2, "tried once and gave up"
    assert any("STILL ON" in r.message and r.levelno >= logging.ERROR for r in caplog.records)
    assert "volume" in log, "the run carries on; the volumes are the experiment"


# ── the route and the pane ───────────────────────────────────────────────


def _app(orch):
    server = MagicMock()
    server.agent_bridge.agent.timelapse_orchestrator = orch
    server.agent_bridge.agent.client.set_laser_config = AsyncMock(return_value={"success": True})
    app = FastAPI()
    app.include_router(create_router(server))
    app.dependency_overrides[auth.require_control] = lambda: True
    return TestClient(app)


def _orch():
    orch = MagicMock()
    orch.start = AsyncMock(return_value="Timelapse started.")
    return orch


@pytest.mark.parametrize("light", ["room", "led", "none"])
def test_the_route_forwards_the_light(light):
    orch = _orch()
    r = _app(orch).post(
        "/api/devices/timelapse/start",
        json={"interval_seconds": 300, "dic": {"enabled": True, "light": light}},
    )
    assert r.status_code == 200, r.text
    assert orch.start.await_args.kwargs["dic"]["light"] == light


def test_the_route_refuses_a_light_it_does_not_know():
    r = _app(_orch()).post(
        "/api/devices/timelapse/start",
        json={"interval_seconds": 300, "dic": {"enabled": True, "light": "sunlight"}},
    )
    assert r.status_code == 400 and "dic.light" in r.json()["detail"]


def test_the_pane_offers_the_three_lights_with_room_first():
    block = HTML[HTML.index('id="op-plan-dic-light"') :][:400]
    assert block.index('value="room" selected') < block.index('value="led"')
    assert 'value="none"' in block


def test_the_pane_reads_and_restores_the_light():
    assert "dicLight: v('op-plan-dic-light')," in OPERATE
    assert "set('op-plan-dic-light', plan.dic.light);" in OPERATE
