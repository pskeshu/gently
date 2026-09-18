"""The joystick lock is written to the controller, read back, and persisted.

Boot used to force JoystickEnabled=Yes unconditionally. With a lock in
Settings, boot must apply the operator's choice instead — otherwise every
device-layer restart silently unlocks the stage.
"""

from __future__ import annotations

import asyncio
import json

import pytest

pytest.importorskip("aiohttp")

from gently.hardware.dispim.device_layer import DeviceLayerServer  # noqa: E402
from gently.hardware.dispim.devices.stage import DiSPIMXYStage  # noqa: E402


class _Core:
    def __init__(self):
        self.props = {"JoystickEnabled": "Yes"}

    def setProperty(self, dev, prop, val):  # noqa: N802
        self.props[prop] = val

    def getProperty(self, dev, prop):  # noqa: N802
        return self.props[prop]


class _Req:
    def __init__(self, body):
        self._b = body

    async def json(self):
        return self._b


def _dl(tmp_path):
    core = _Core()
    dl = DeviceLayerServer.__new__(DeviceLayerServer)
    dl.devices = {"xy_stage": DiSPIMXYStage(name="XYStage:XY:31", core=core)}
    dl.config_path = str(tmp_path / "config.yml")
    return dl, core


def test_lock_writes_controller_and_persists(tmp_path):
    dl, core = _dl(tmp_path)
    resp = asyncio.run(dl.handle_set_joystick(_Req({"enabled": False})))
    assert resp.status == 200 and json.loads(resp.text)["enabled"] is False
    assert core.props["JoystickEnabled"] == "No"
    side = (tmp_path / "config.local.yml").read_text()
    assert "xy_joystick" in side and "enabled: false" in side


def test_get_reads_the_controller_not_a_memory(tmp_path):
    dl, core = _dl(tmp_path)
    core.props["JoystickEnabled"] = "No"  # e.g. set from Micro-Manager
    resp = asyncio.run(dl.handle_get_joystick(None))
    assert json.loads(resp.text)["enabled"] is False


def test_non_boolean_is_a_400(tmp_path):
    dl, _ = _dl(tmp_path)
    resp = asyncio.run(dl.handle_set_joystick(_Req({"enabled": "maybe"})))
    # bool("maybe") is True — a string must not silently unlock the stage
    assert resp.status == 400
