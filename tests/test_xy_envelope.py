"""The XY safety envelope is operator-set, firmware-first, and persisted (#107).

Ryan: "we did notice that sometimes the embryos that we put on the coverslip
are outside the map region." The envelope used to be four module constants
that nothing above the device could change. Now the Map's Edit region wizard
sets it — but the controller is written and read back BEFORE the software
fence moves, and a stage sitting outside the new box is refused.
"""

from __future__ import annotations

import asyncio
import json

import pytest

pytest.importorskip("aiohttp")

from gently.hardware.dispim.device_layer import DeviceLayerServer  # noqa: E402
from gently.hardware.dispim.devices.stage import (  # noqa: E402
    XY_STAGE_X_MAX_UM,
    XY_STAGE_X_MIN_UM,
    DiSPIMXYStage,
)


class _Core:
    """Enough of MMCore for the XY stage: properties round-trip, position is fixed."""

    def __init__(self, xy_um=(0.0, 0.0)):
        self.props: dict[str, str] = {}
        self.xy = xy_um

    def setProperty(self, dev, prop, val):  # noqa: N802
        self.props[prop] = val

    def getProperty(self, dev, prop):  # noqa: N802
        return self.props[prop]

    def getXYPosition(self, *_):  # noqa: N802
        return self.xy

    def getXPosition(self, *_):  # noqa: N802
        return self.xy[0]

    def getYPosition(self, *_):  # noqa: N802
        return self.xy[1]


def _stage(core):
    st = DiSPIMXYStage(name="XYStage:XY:31", core=core)
    # read() goes through the real MMCore API surface; pin it to the fake.
    st.read = lambda: {st.name: {"value": core.xy}}  # type: ignore[method-assign]
    return st


def _dl(stage, tmp_path):
    dl = DeviceLayerServer.__new__(DeviceLayerServer)
    dl.devices = {"xy_stage": stage}
    dl.config_path = str(tmp_path / "config.yml")

    class _NoPause:
        async def __aenter__(self):
            return None

        async def __aexit__(self, *a):
            return False

    dl.pause_state_updates = lambda: _NoPause()  # type: ignore[method-assign]
    return dl


class _Req:
    def __init__(self, body):
        self._b = body

    async def json(self):
        return self._b


def _post(dl, body):
    resp = asyncio.run(dl.handle_set_envelope(_Req(body)))
    return resp.status, json.loads(resp.text)


def test_envelope_starts_at_the_code_defaults():
    st = _stage(_Core())
    assert st.x_limits == (XY_STAGE_X_MIN_UM, XY_STAGE_X_MAX_UM)


def test_software_fence_follows_a_verified_firmware_write(tmp_path):
    core = _Core(xy_um=(100.0, 100.0))
    st = _stage(core)
    dl = _dl(st, tmp_path)
    status, body = _post(dl, {"x_min": -500, "x_max": 1500, "y_min": -400, "y_max": 900})
    assert status == 200 and body["success"]
    assert st.x_limits == (-500.0, 1500.0) and st.y_limits == (-400.0, 900.0)
    assert float(core.props["UpperLimX(mm)"]) == pytest.approx(1.5)
    # and set() now fences against the new box, not the constants
    assert isinstance(st.set([1600.0, 0.0]).exception(), ValueError)
    sidecar = tmp_path / "config.local.yml"
    assert sidecar.exists() and "xy_envelope" in sidecar.read_text()


def test_stage_outside_the_new_box_is_refused_and_nothing_moves(tmp_path):
    core = _Core(xy_um=(5000.0, 5000.0))
    st = _stage(core)
    before = (st.x_limits, st.y_limits)
    status, body = _post(_dl(st, tmp_path), {"x_min": 0, "x_max": 100, "y_min": 0, "y_max": 100})
    assert status == 409 and "outside" in body["error"]
    assert (st.x_limits, st.y_limits) == before
    assert core.props == {}


def test_degenerate_box_is_refused(tmp_path):
    st = _stage(_Core())
    status, _ = _post(_dl(st, tmp_path), {"x_min": 10, "x_max": 10, "y_min": 0, "y_max": 1})
    assert status == 409


def test_missing_field_is_a_400(tmp_path):
    st = _stage(_Core())
    status, _ = _post(_dl(st, tmp_path), {"x_min": 0})
    assert status == 400
