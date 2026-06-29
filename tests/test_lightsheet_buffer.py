"""Tests for live-view buffer hardening.

Verifies:
  - snap mode (default) uses snapImage()+getImage() — no circular buffer
  - snap mode does NOT call startContinuousSequenceAcquisition
  - continuous mode calls setCircularBufferMemoryFootprint(256) before starting
  - continuous mode calls clearCircularBuffer() after each getLastImage()
  - _stop_lightsheet_sequence_sync calls clearCircularBuffer() in both modes
"""
import sys
from unittest.mock import MagicMock

import numpy as np
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
# Full-featured FakeCore tracking every buffer-related call
# ---------------------------------------------------------------------------


class BufferFakeCore:
    """CMMCore stub that records every buffer/snap/sequence call."""

    def __init__(self):
        self.cam = None
        self.running = False
        self.started = 0
        self.stopped = 0
        self.exposure = None
        self.footprint_set: int | None = None  # value passed to setCircularBufferMemoryFootprint
        self.clear_called = 0
        self.snap_called = 0
        self.get_image_called = 0
        self.get_last_image_called = 0

    def setCameraDevice(self, n):
        self.cam = n

    def getCameraDevice(self):
        return self.cam or ""

    def setExposure(self, n, ms):
        self.exposure = ms

    def isSequenceRunning(self):
        return self.running

    def startContinuousSequenceAcquisition(self, interval):
        self.running = True
        self.started += 1

    def stopSequenceAcquisition(self):
        self.running = False
        self.stopped += 1

    def setCircularBufferMemoryFootprint(self, mb: int):
        self.footprint_set = mb

    def clearCircularBuffer(self):
        self.clear_called += 1

    def snapImage(self):
        self.snap_called += 1

    def getImage(self):
        self.get_image_called += 1
        return np.zeros((4, 4), dtype=np.uint16)

    def getLastImage(self):
        self.get_last_image_called += 1
        return np.zeros((4, 4), dtype=np.uint16)


class FakeAxisOffset:
    def __init__(self):
        self.last_pos = None

    def setPosition(self, val):
        self.last_pos = val


class FakeScanner:
    def __init__(self, name):
        self.name = name
        self.sa_offset_y = FakeAxisOffset()

    def set_spim_state(self, state):
        pass


class FakePiezo:
    def __init__(self, name):
        self.name = name
        self.last_pos = None

    def setPosition(self, val):
        self.last_pos = val

    def set_spim_state(self, state):
        pass


def _make_dl_buffer(mode: str = "snap") -> DeviceLayerServer:
    """Build a minimal DL for buffer-hardening tests."""
    dl = DeviceLayerServer.__new__(DeviceLayerServer)
    core = BufferFakeCore()
    dl.system = type("S", (), {"core": core})()
    cam_a = type("C", (), {"name": "HamCam1"})()
    dl.devices = {
        "camera": cam_a,
        "scanner": FakeScanner("Scanner:AB:33"),
        "piezo": FakePiezo("PiezoStage:P:34"),
    }
    dl._ls_params = {
        "galvo": 0.0,
        "piezo": 50.0,
        "exposure": 20.0,
        "side": "A",
        "mode": mode,
    }
    dl._ls_seq_started = False
    dl._ls_applied = {}
    dl._ls_interval_sec = 0.0
    dl._ls_parked = {}
    dl._ls_spim_idle = False
    return dl


# ---------------------------------------------------------------------------
# Snap mode — snapImage + getImage, no circular buffer
# ---------------------------------------------------------------------------


def test_snap_mode_calls_snapimage_and_getimage():
    """snap mode: _grab_lightsheet_frame_sync calls snapImage() + getImage()."""
    dl = _make_dl_buffer(mode="snap")
    dl._grab_lightsheet_frame_sync()
    core = dl.system.core
    assert core.snap_called >= 1, "snapImage() must be called in snap mode"
    assert core.get_image_called >= 1, "getImage() must be called in snap mode"


def test_snap_mode_does_not_call_get_last_image():
    """snap mode: getLastImage() is NOT called (avoids the never-draining buffer)."""
    dl = _make_dl_buffer(mode="snap")
    dl._grab_lightsheet_frame_sync()
    assert dl.system.core.get_last_image_called == 0


def test_snap_mode_does_not_start_continuous_sequence():
    """snap mode: startContinuousSequenceAcquisition is never called."""
    dl = _make_dl_buffer(mode="snap")
    dl._ensure_lightsheet_sequence_sync()
    assert dl.system.core.started == 0, (
        "startContinuousSequenceAcquisition must not be called in snap mode"
    )


def test_snap_mode_does_not_start_continuous_on_grab():
    """snap mode grab loop never starts continuous acquisition."""
    dl = _make_dl_buffer(mode="snap")
    dl._grab_lightsheet_frame_sync()
    assert dl.system.core.started == 0


# ---------------------------------------------------------------------------
# Continuous mode — buffer cap + drain
# ---------------------------------------------------------------------------


def test_continuous_mode_sets_circular_buffer_footprint():
    """continuous mode: setCircularBufferMemoryFootprint(256) called before start."""
    dl = _make_dl_buffer(mode="continuous")
    dl._ensure_lightsheet_sequence_sync()
    core = dl.system.core
    assert core.footprint_set is not None, (
        "setCircularBufferMemoryFootprint must be called in continuous mode"
    )
    assert core.footprint_set == 256, (
        f"footprint must be 256 MB, got {core.footprint_set}"
    )


def test_continuous_mode_calls_clear_after_grab():
    """continuous mode: clearCircularBuffer() called after each getLastImage()."""
    dl = _make_dl_buffer(mode="continuous")
    dl._grab_lightsheet_frame_sync()
    core = dl.system.core
    assert core.clear_called >= 1, (
        "clearCircularBuffer() must be called after getLastImage() in continuous mode"
    )


def test_continuous_mode_starts_sequence():
    """continuous mode: startContinuousSequenceAcquisition is called."""
    dl = _make_dl_buffer(mode="continuous")
    dl._ensure_lightsheet_sequence_sync()
    assert dl.system.core.started == 1


# ---------------------------------------------------------------------------
# Stop sequence — clearCircularBuffer on every exit path
# ---------------------------------------------------------------------------


def test_stop_calls_clear_circular_buffer():
    """_stop_lightsheet_sequence_sync always calls clearCircularBuffer()."""
    dl = _make_dl_buffer(mode="continuous")
    # Start a sequence first so there's something to stop
    dl._ensure_lightsheet_sequence_sync()
    clear_before = dl.system.core.clear_called
    dl._stop_lightsheet_sequence_sync()
    assert dl.system.core.clear_called > clear_before, (
        "clearCircularBuffer must be called in _stop_lightsheet_sequence_sync"
    )


def test_stop_resets_seq_started():
    """_stop_lightsheet_sequence_sync resets _ls_seq_started to False."""
    dl = _make_dl_buffer(mode="continuous")
    dl._ensure_lightsheet_sequence_sync()
    assert dl._ls_seq_started is True
    dl._stop_lightsheet_sequence_sync()
    assert dl._ls_seq_started is False


def test_stop_clears_applied_state():
    """_stop_lightsheet_sequence_sync clears _ls_applied so next start does a full restart."""
    dl = _make_dl_buffer(mode="continuous")
    dl._ensure_lightsheet_sequence_sync()
    assert dl._ls_applied != {}
    dl._stop_lightsheet_sequence_sync()
    assert dl._ls_applied == {}
