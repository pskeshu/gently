# tests/test_lightsheet_streamer.py
import sys
from unittest.mock import MagicMock

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

# Patch bluesky.RunEngine specifically
import bluesky as _bs  # noqa: E402

_bs.RunEngine = MagicMock(name="RunEngine")

import asyncio  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from gently.hardware.dispim.device_layer import DeviceLayerServer  # noqa: E402


class FakeCore:
    def __init__(self):
        self.running = False
        self.exposure = None
        self.cam = None
        self._frame = np.full((64, 64), 1000, dtype=np.uint16)
        self.started = 0
        self.stopped = 0

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

    def getLastImage(self):
        return self._frame


class FakeAxis:
    def __init__(self):
        self.pos = None

    def setPosition(self, v):
        self.pos = v


class FakeScanner:
    def __init__(self):
        self.sa_offset_y = FakeAxis()
        self.name = "Scanner"
        self.state = None

    def set_spim_state(self, s):
        self.state = s


class FakePiezo(FakeAxis):
    def __init__(self):
        super().__init__()
        self.name = "Piezo"
        self.state = None

    def set_spim_state(self, s):
        self.state = s


def _make_dl():
    dl = DeviceLayerServer.__new__(DeviceLayerServer)
    dl.system = type("S", (), {"core": FakeCore()})()
    dl.devices = {
        "camera": type("C", (), {"name": "HamCam1"})(),
        "scanner": FakeScanner(),
        "piezo": FakePiezo(),
    }
    # Park-guard state (park-guard fix: these are checked by _park_lightsheet_sync)
    dl._ls_parked = {}
    dl._ls_spim_idle = False
    return dl


@pytest.mark.asyncio
async def test_grab_parks_and_peeks(monkeypatch):
    dl = _make_dl()
    dl._state_pause_counter = 0
    dl._ls_target_max_dim = 512
    dl._ls_jpeg_quality = 70
    # Explicit continuous mode: this test verifies the sequence-start + getLastImage path.
    dl._ls_params = {"galvo": 1.5, "piezo": 40.0, "exposure": 20.0, "mode": "continuous"}
    dl._ls_seq_started = False
    dl._ls_applied = {}
    dl._ls_interval_sec = 0.0
    img = await asyncio.to_thread(dl._grab_lightsheet_frame_sync)
    assert img is not None and img.shape == (64, 64)
    assert dl.system.core.running is True  # sequence started
    assert dl.devices["piezo"].pos == 40.0  # piezo parked
    assert dl.devices["scanner"].sa_offset_y.pos == 1.5  # galvo parked


@pytest.mark.asyncio
async def test_exposure_change_restarts_sequence():
    dl = _make_dl()
    dl._state_pause_counter = 0
    dl._ls_target_max_dim = 512
    dl._ls_jpeg_quality = 70
    # Explicit continuous mode: this test verifies restart on exposure change.
    dl._ls_params = {"galvo": 0.0, "piezo": 50.0, "exposure": 10.0, "mode": "continuous"}
    dl._ls_seq_started = False
    dl._ls_applied = {}
    dl._ls_interval_sec = 0.0
    await asyncio.to_thread(dl._grab_lightsheet_frame_sync)
    starts = dl.system.core.started
    dl._ls_params["exposure"] = 30.0  # exposure change
    await asyncio.to_thread(dl._grab_lightsheet_frame_sync)
    assert dl.system.core.stopped >= 1 and dl.system.core.started == starts + 1
    assert dl.system.core.exposure == 30.0
