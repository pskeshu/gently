"""The XY safety envelope is operator-set, persisted, and now TWO fences (#107).

Ryan: "we did notice that sometimes the embryos that we put on the coverslip
are outside the map region." The envelope used to be four module constants
that nothing above the device could change. The Map's region editor sets it.

The single envelope has since come apart into two, because they protect
against different things:

* the SOFTWARE fence bounds every move Gently commands — checked in `set()`
  before anything reaches the hardware, costs nothing, affects nobody else;
* the FIRMWARE fence is the same numbers written into the Tiger, whose only
  advantage is stopping a hand on the joystick — and whose cost is binding
  every other client of that controller, Micro-Manager included.

On a rig run by trained operators the second is opt-in, so applying a region
moves the software fence and leaves the controller alone unless enforcement
is switched on. What still holds either way: when the controller IS written
it is read back before the software fence follows, the software fence is
never wider than the firmware one, and an envelope that excludes where the
stage is standing is refused.
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


def _fence_error(status):
    """The error a refused ``set()`` reports, whichever Status class is in play.

    tests/test_dispim_device_safety.py installs MagicMock stand-ins for ophyd
    in ``sys.modules``, so in a full run ``stage.Status`` is a mock and
    ``status.exception()`` hands back another mock instead of the ValueError —
    this assertion passed alone and failed in the suite (#143's order
    dependence, in a new test). The fence itself is the same either way: the
    stage constructs a Status and calls ``set_exception`` with the error, so
    read it from the recorded call when the class is a mock.
    """
    exc = status.exception() if hasattr(status, "exception") else None
    if isinstance(exc, BaseException):
        return exc
    for call in getattr(getattr(status, "set_exception", None), "call_args_list", []):
        if call.args and isinstance(call.args[0], BaseException):
            return call.args[0]
    return None


def test_envelope_starts_at_the_code_defaults():
    st = _stage(_Core())
    assert st.x_limits == (XY_STAGE_X_MIN_UM, XY_STAGE_X_MAX_UM)


def test_a_region_binds_gently_without_touching_the_controller(tmp_path):
    """The default: the region is Gently's fence, and nobody else's."""
    core = _Core(xy_um=(100.0, 100.0))
    st = _stage(core)
    dl = _dl(st, tmp_path)
    status, body = _post(dl, {"x_min": -500, "x_max": 900, "y_min": -400, "y_max": 500})
    assert status == 200 and body["success"]
    assert st.x_limits == (-500.0, 900.0) and st.y_limits == (-400.0, 500.0)
    # set() fences against the new box, not the constants
    assert isinstance(_fence_error(st.set([950.0, 0.0])), ValueError)
    # ...and the controller was left alone, so a Micro-Manager user on this
    # Tiger still has the whole stage.
    assert core.props == {}, f"the controller was written without being asked: {core.props}"
    sidecar = tmp_path / "config.local.yml"
    assert sidecar.exists() and "xy_envelope" in sidecar.read_text()


def test_enforcing_writes_the_controller_first_then_the_software_fence(tmp_path):
    """When it IS asked for, the old ordering still holds."""
    core = _Core(xy_um=(100.0, 100.0))
    st = _stage(core)
    dl = _dl(st, tmp_path)
    dl.config = {"xy_envelope": {"enforced": True}}
    status, body = _post(dl, {"x_min": -500, "x_max": 900, "y_min": -400, "y_max": 500})
    assert status == 200 and body["success"]
    assert float(core.props["UpperLimX(mm)"]) == pytest.approx(0.9)
    assert st.x_limits == (-500.0, 900.0) and st.y_limits == (-400.0, 500.0)


def test_the_software_fence_is_never_wider_than_the_firmware_one(tmp_path):
    """Wider would command moves the controller then refuses.

    That reads as a mysterious hardware error rather than a limit, which is
    why the ordering in `initialize` is firmware first, region second.
    """
    core = _Core(xy_um=(100.0, 100.0))
    st = _stage(core)
    status, _ = _post(_dl(st, tmp_path), {"x_min": -500, "x_max": 900, "y_min": -400, "y_max": 500})
    assert status == 200
    sx_lo, sx_hi = st.x_limits
    assert sx_lo >= XY_STAGE_X_MIN_UM and sx_hi <= XY_STAGE_X_MAX_UM


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
