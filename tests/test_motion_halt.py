"""POST /api/motion/halt stops every positioner that exists (#109).

Ryan, first words on opening the SPIM head tab: "here we need a halt button."
A halt must reach the axis that is moving without being told which one, must
skip axes this rig does not have, and must not be silent when the controller
refuses.
"""

from __future__ import annotations

import asyncio
import json

import pytest

pytest.importorskip("aiohttp")

from gently.hardware.dispim.device_layer import DeviceLayerServer  # noqa: E402


class _Core:
    def __init__(self, fail: set[str] | None = None):
        self.stopped: list[str] = []
        self.fail = fail or set()

    def stop(self, label: str) -> None:
        if label in self.fail:
            raise RuntimeError(f"{label}: HALT refused")
        self.stopped.append(label)


def _dl(core: _Core, **devices) -> DeviceLayerServer:
    dl = DeviceLayerServer.__new__(DeviceLayerServer)
    dl.system = type("S", (), {"core": core})()
    dl.devices = {k: type("D", (), {"name": v})() for k, v in devices.items()}
    return dl


def _run(dl):
    resp = asyncio.run(dl.handle_halt_motion(None))
    return resp.status, json.loads(resp.text)


def test_halts_every_present_positioner_and_skips_absent_ones():
    core = _Core()
    dl = _dl(core, fdrive="ZStage:V:37", xy_stage="XYStage:XY:31")  # no z_stage
    status, body = _run(dl)
    assert status == 200 and body["success"] is True
    assert sorted(core.stopped) == ["XYStage:XY:31", "ZStage:V:37"]
    assert sorted(body["halted"]) == ["fdrive", "xy_stage"]


def test_a_refused_halt_is_reported_not_swallowed():
    core = _Core(fail={"ZStage:V:37"})
    dl = _dl(core, fdrive="ZStage:V:37", xy_stage="XYStage:XY:31")
    status, body = _run(dl)
    assert status == 502 and body["success"] is False
    assert "fdrive" in body["errors"] and body["halted"] == ["xy_stage"]
