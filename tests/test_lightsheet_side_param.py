"""Tests for B2 Task 2 — lightsheet side param (handle_lightsheet_params +
_ensure_lightsheet_sequence_sync + handle_get_cameras).

Verifies:
  - handle_lightsheet_params accepts 'side' ('A'|'B'); invalid values ignored
  - side change in _ensure_lightsheet_sequence_sync selects the correct camera
    and triggers a sequence restart (mirroring the exposure-change path)
  - side B without camera_b falls back to A + no crash
  - handle_get_cameras returns ["A"] or ["A","B"] based on devices dict
"""
import json
import sys
from unittest.mock import AsyncMock, MagicMock

import pytest

# Patch heavy hardware deps before importing device_layer
for _mod in (
    "bluesky",
    "bluesky.run_engine",
    "ophyd",
    "ophyd.status",
    "pymmcore",
    "gently.hardware.console_ui",
):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

import bluesky as _bs  # noqa: E402

_bs.RunEngine = MagicMock(name="RunEngine")

from gently.hardware.dispim.device_layer import DeviceLayerServer  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeCore:
    """Minimal CMMCore stub tracking sequence-start/stop calls."""

    def __init__(self):
        self.running = False
        self.cam = None
        self.started = 0
        self.stopped = 0
        self.exposure = None

    def setCameraDevice(self, n):
        self.cam = n

    def getCameraDevice(self):
        return self.cam

    def setExposure(self, n, ms):
        self.exposure = ms

    def startContinuousSequenceAcquisition(self, interval):
        self.running = True
        self.started += 1

    def stopSequenceAcquisition(self):
        self.running = False
        self.stopped += 1

    def isSequenceRunning(self):
        return self.running

    def setCircularBufferMemoryFootprint(self, mb):
        pass  # no-op for camera-selection tests

    def clearCircularBuffer(self):
        pass  # no-op for camera-selection tests


class FakeAxisOffset:
    """Stub for DiSPIMScanner.sa_offset_y tracking setPosition calls."""

    def __init__(self):
        self.last_pos = None

    def setPosition(self, val: float) -> None:
        self.last_pos = val


class FakeScanner:
    """Stub scanner with sa_offset_y and set_spim_state."""

    def __init__(self, name: str):
        self.name = name
        self.sa_offset_y = FakeAxisOffset()
        self.spim_state = None

    def set_spim_state(self, state: str) -> None:
        self.spim_state = state


class FakePiezo:
    """Stub piezo tracking setPosition and set_spim_state calls."""

    def __init__(self, name: str):
        self.name = name
        self.last_pos = None
        self.spim_state = None

    def setPosition(self, val: float) -> None:
        self.last_pos = val

    def set_spim_state(self, state: str) -> None:
        self.spim_state = state


def _make_dl(
    with_camera_b: bool = True,
    with_scanner_b: bool = False,
    with_piezo_b: bool = False,
) -> DeviceLayerServer:
    """Build a minimal DeviceLayerServer with faked core + devices.

    mode is set to "continuous" so existing sequence-start assertions remain
    meaningful (snap mode is the production default but tested separately).
    """
    dl = DeviceLayerServer.__new__(DeviceLayerServer)
    dl.system = type("S", (), {"core": FakeCore()})()
    cam_a = type("C", (), {"name": "HamCam1"})()
    cam_b = type("C", (), {"name": "HamCam2"})()
    dl.devices = {
        "camera": cam_a,
        "scanner": FakeScanner("Scanner:AB:33"),
        "piezo": FakePiezo("PiezoStage:P:34"),
    }
    if with_camera_b:
        dl.devices["camera_b"] = cam_b
    if with_scanner_b:
        dl.devices["scanner_b"] = FakeScanner("Scanner:CD:33")
    if with_piezo_b:
        dl.devices["piezo_b"] = FakePiezo("PiezoStage:Q:35")
    dl._ls_params = {
        "galvo": 0.0,
        "piezo": 50.0,
        "exposure": 20.0,
        "side": "A",
        "mode": "continuous",  # explicit — snap mode tested in test_lightsheet_buffer.py
    }
    dl._ls_seq_started = False
    dl._ls_applied = {}
    dl._ls_interval_sec = 0.0
    dl._ls_parked = {}
    dl._ls_spim_idle = False
    return dl


# ---------------------------------------------------------------------------
# _ensure_lightsheet_sequence_sync — camera selection by side
# ---------------------------------------------------------------------------


def test_side_a_selects_camera_a():
    """Side A → sequence started on HamCam1."""
    dl = _make_dl()
    dl._ls_params["side"] = "A"
    dl._ensure_lightsheet_sequence_sync()
    assert dl.system.core.cam == "HamCam1"
    assert dl.system.core.started == 1


def test_side_b_selects_camera_b():
    """Side B with camera_b present → sequence started on HamCam2."""
    dl = _make_dl(with_camera_b=True)
    dl._ls_params["side"] = "B"
    dl._ensure_lightsheet_sequence_sync()
    assert dl.system.core.cam == "HamCam2"
    assert dl.system.core.started == 1


def test_side_b_fallback_to_a_when_no_camera_b():
    """Side B without camera_b → graceful fallback to A (HamCam1), no crash."""
    dl = _make_dl(with_camera_b=False)
    dl._ls_params["side"] = "B"
    dl._ensure_lightsheet_sequence_sync()
    assert dl.system.core.cam == "HamCam1"
    assert dl.system.core.started == 1


def test_side_change_triggers_restart():
    """Switching from A to B stops the running sequence and restarts on HamCam2."""
    dl = _make_dl(with_camera_b=True)
    # First call — start on A
    dl._ls_params["side"] = "A"
    dl._ensure_lightsheet_sequence_sync()
    assert dl.system.core.started == 1
    assert dl.system.core.cam == "HamCam1"

    # Switch to B — must restart
    dl._ls_params["side"] = "B"
    dl._ensure_lightsheet_sequence_sync()
    assert dl.system.core.stopped >= 1
    assert dl.system.core.started >= 2
    assert dl.system.core.cam == "HamCam2"


def test_no_restart_when_side_unchanged():
    """Same side, same exposure → sequence already running, no extra restart."""
    dl = _make_dl()
    dl._ls_params["side"] = "A"
    dl._ensure_lightsheet_sequence_sync()
    starts_after_first = dl.system.core.started

    # Manually mark the applied state (as the real code does)
    dl._ls_applied["exposure"] = 20.0
    dl._ls_applied["side"] = "A"

    dl._ensure_lightsheet_sequence_sync()
    assert dl.system.core.started == starts_after_first


# ---------------------------------------------------------------------------
# handle_lightsheet_params — side accepted / validated
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_handle_lightsheet_params_accepts_side_b():
    """POST {side: 'B'} → _ls_params['side'] updated to 'B'."""
    dl = _make_dl()
    req = MagicMock()
    req.json = AsyncMock(return_value={"side": "B"})
    await dl.handle_lightsheet_params(req)
    assert dl._ls_params["side"] == "B"


@pytest.mark.asyncio
async def test_handle_lightsheet_params_accepts_side_a():
    """POST {side: 'A'} when already A → _ls_params['side'] stays 'A'."""
    dl = _make_dl()
    req = MagicMock()
    req.json = AsyncMock(return_value={"side": "A"})
    await dl.handle_lightsheet_params(req)
    assert dl._ls_params["side"] == "A"


@pytest.mark.asyncio
async def test_handle_lightsheet_params_ignores_invalid_side():
    """Invalid side value (not 'A'|'B') → _ls_params['side'] unchanged."""
    dl = _make_dl()
    dl._ls_params["side"] = "A"
    req = MagicMock()
    req.json = AsyncMock(return_value={"side": "C"})
    await dl.handle_lightsheet_params(req)
    assert dl._ls_params["side"] == "A"


@pytest.mark.asyncio
async def test_handle_lightsheet_params_accepts_exposure_and_side():
    """POST with both exposure and side → both applied."""
    dl = _make_dl()
    req = MagicMock()
    req.json = AsyncMock(return_value={"exposure": 50.0, "side": "B"})
    await dl.handle_lightsheet_params(req)
    assert dl._ls_params["exposure"] == 50.0
    assert dl._ls_params["side"] == "B"


# ---------------------------------------------------------------------------
# handle_get_cameras — roles endpoint
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_handle_get_cameras_with_camera_b():
    """Both A and B listed when camera_b is registered."""
    dl = _make_dl(with_camera_b=True)
    req = MagicMock()
    resp = await dl.handle_get_cameras(req)
    body = json.loads(resp.body)
    assert body["cameras"] == ["A", "B"]


@pytest.mark.asyncio
async def test_handle_get_cameras_without_camera_b():
    """Only A listed when camera_b is absent."""
    dl = _make_dl(with_camera_b=False)
    req = MagicMock()
    resp = await dl.handle_get_cameras(req)
    body = json.loads(resp.body)
    assert body["cameras"] == ["A"]


# ---------------------------------------------------------------------------
# _park_lightsheet_sync — side selects correct scanner/piezo
# ---------------------------------------------------------------------------


def test_side_a_parks_scanner_a():
    """Side A → _park_lightsheet_sync drives scanner A's galvo."""
    dl = _make_dl()
    dl._ls_params["side"] = "A"
    dl._park_lightsheet_sync()
    assert dl.devices["scanner"].sa_offset_y.last_pos is not None


def test_side_b_parks_scanner_b():
    """Side B with scanner_b present → _park_lightsheet_sync drives scanner B."""
    dl = _make_dl(with_scanner_b=True)
    dl._ls_params["side"] = "B"
    dl._park_lightsheet_sync()
    assert dl.devices["scanner_b"].sa_offset_y.last_pos is not None
    # Side A scanner must be untouched
    assert dl.devices["scanner"].sa_offset_y.last_pos is None


def test_side_b_scanner_b_absent_falls_back_to_scanner_a():
    """Side B without scanner_b → graceful fallback to scanner A, no crash."""
    dl = _make_dl(with_scanner_b=False)
    dl._ls_params["side"] = "B"
    dl._park_lightsheet_sync()
    # Fallback: scanner A gets the park
    assert dl.devices["scanner"].sa_offset_y.last_pos is not None


def test_side_a_parks_piezo_a():
    """Side A → _park_lightsheet_sync drives piezo A."""
    dl = _make_dl()
    dl._ls_params["side"] = "A"
    dl._park_lightsheet_sync()
    assert dl.devices["piezo"].last_pos is not None


def test_side_b_parks_piezo_b():
    """Side B with piezo_b present → _park_lightsheet_sync drives piezo B."""
    dl = _make_dl(with_scanner_b=True, with_piezo_b=True)
    dl._ls_params["side"] = "B"
    dl._park_lightsheet_sync()
    assert dl.devices["piezo_b"].last_pos is not None
    # Side A piezo must be untouched
    assert dl.devices["piezo"].last_pos is None


def test_side_b_piezo_b_absent_falls_back_to_piezo_a():
    """Side B without piezo_b → graceful fallback to piezo A, no crash."""
    dl = _make_dl(with_piezo_b=False)
    dl._ls_params["side"] = "B"
    dl._park_lightsheet_sync()
    assert dl.devices["piezo"].last_pos is not None
