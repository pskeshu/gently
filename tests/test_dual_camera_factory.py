"""Tests for B2 Task 2 — dual-camera device_factory (HamCam2 defensive registration).

Verifies:
  - FakeCore with "HamCam2" in getLoadedDevices() → devices["camera_b"] created (name == "HamCam2")
  - FakeCore without "HamCam2" → camera_b absent, no crash, devices["camera"] still created
  - camera_b registration uses core.getLoadedDevices() as the presence check
"""
import sys
from unittest.mock import MagicMock

# Patch heavy hardware deps before importing device_factory.
# All device classes use pymmcore / ophyd.status only (no ophyd base classes),
# so patching at the module level is sufficient.
for _mod in (
    "pymmcore",
    "ophyd",
    "ophyd.status",
    "bluesky",
    "bluesky.run_engine",
    "gently.hardware.console_ui",
):
    if _mod not in sys.modules:
        sys.modules[_mod] = MagicMock()

import bluesky as _bs  # noqa: E402

_bs.RunEngine = MagicMock(name="RunEngine")

from gently.hardware.dispim.device_factory import create_devices_from_mmcore  # noqa: E402


class FakeCore:
    """Minimal CMMCore stand-in with configurable loaded-device list."""

    def __init__(self, loaded_devices=None):
        self._loaded = list(loaded_devices or [])

    def getLoadedDevices(self):
        return self._loaded

    # stubs that DiSPIMCamera.__init__ may indirectly trigger via ophyd proxies
    def getProperty(self, dev, prop):
        return "0"

    def setProperty(self, dev, prop, val):
        pass

    def getCameraDevice(self):
        return ""

    def setCameraDevice(self, name):
        pass


# ---------------------------------------------------------------------------
# Happy path — HamCam2 present
# ---------------------------------------------------------------------------


def test_camera_b_created_when_hamcam2_loaded():
    """HamCam2 in getLoadedDevices → devices['camera_b'] created with name 'HamCam2'."""
    core = FakeCore(loaded_devices=["HamCam1", "HamCam2"])
    devices = create_devices_from_mmcore(core)
    assert "camera_b" in devices, "camera_b must be registered when HamCam2 is in loaded devices"
    assert devices["camera_b"].name == "HamCam2"


def test_camera_b_name_is_hamcam2():
    """The camera_b name attribute is exactly the configured camera_b_name."""
    core = FakeCore(loaded_devices=["HamCam1", "HamCam2", "Scanner:AB:33"])
    devices = create_devices_from_mmcore(core)
    assert devices["camera_b"].name == "HamCam2"


# ---------------------------------------------------------------------------
# Single-camera rig — HamCam2 absent
# ---------------------------------------------------------------------------


def test_camera_b_absent_when_hamcam2_not_loaded():
    """HamCam2 NOT in getLoadedDevices → camera_b absent, no crash."""
    core = FakeCore(loaded_devices=["HamCam1"])
    devices = create_devices_from_mmcore(core)
    assert "camera_b" not in devices, "camera_b must be absent on single-camera rigs"


def test_camera_still_created_without_hamcam2():
    """Primary camera (HamCam1) is created regardless of HamCam2 presence."""
    core = FakeCore(loaded_devices=["HamCam1"])
    devices = create_devices_from_mmcore(core)
    assert "camera" in devices
    assert devices["camera"].name == "HamCam1"


# ---------------------------------------------------------------------------
# Edge case — empty loaded-device list
# ---------------------------------------------------------------------------


def test_camera_b_absent_on_empty_device_list():
    """Empty loaded-device list → camera_b not attempted, no extra error."""
    core = FakeCore(loaded_devices=[])
    # DiSPIMCamera.__init__ only stores name/core; it succeeds even with no devices.
    # So "camera" is still created; "camera_b" is skipped.
    devices = create_devices_from_mmcore(core)
    assert "camera_b" not in devices
    assert "camera" in devices


# ---------------------------------------------------------------------------
# Side-B optics: scanner_b / piezo_b defensive registration
# ---------------------------------------------------------------------------


def test_scanner_b_created_when_present():
    """Scanner:CD:33 in getLoadedDevices → devices['scanner_b'] with that name."""
    core = FakeCore(loaded_devices=["HamCam1", "Scanner:CD:33"])
    devices = create_devices_from_mmcore(core)
    assert "scanner_b" in devices, "scanner_b must be registered when Scanner:CD:33 is loaded"
    assert devices["scanner_b"].name == "Scanner:CD:33"


def test_scanner_b_absent_when_not_loaded():
    """Scanner:CD:33 NOT loaded → scanner_b absent, no crash."""
    core = FakeCore(loaded_devices=["HamCam1"])
    devices = create_devices_from_mmcore(core)
    assert "scanner_b" not in devices


def test_piezo_b_created_when_present():
    """PiezoStage:Q:35 in getLoadedDevices → devices['piezo_b'] with that name."""
    core = FakeCore(loaded_devices=["HamCam1", "PiezoStage:Q:35"])
    devices = create_devices_from_mmcore(core)
    assert "piezo_b" in devices, "piezo_b must be registered when PiezoStage:Q:35 is loaded"
    assert devices["piezo_b"].name == "PiezoStage:Q:35"


def test_piezo_b_absent_when_not_loaded():
    """PiezoStage:Q:35 NOT loaded → piezo_b absent, no crash."""
    core = FakeCore(loaded_devices=["HamCam1"])
    devices = create_devices_from_mmcore(core)
    assert "piezo_b" not in devices


def test_scanner_b_piezo_b_both_present_on_dual_side_rig():
    """Full dual-side rig: both scanner_b and piezo_b created; side A unchanged."""
    core = FakeCore(loaded_devices=["HamCam1", "HamCam2", "Scanner:CD:33", "PiezoStage:Q:35"])
    devices = create_devices_from_mmcore(core)
    assert "scanner_b" in devices
    assert "piezo_b" in devices
    # Side-A optics must still be present
    assert "scanner" in devices
    assert "piezo" in devices


def test_scanner_b_piezo_b_both_absent_on_single_side_rig():
    """Single-side rig: neither scanner_b nor piezo_b created; A-side and camera OK."""
    core = FakeCore(loaded_devices=["HamCam1"])
    devices = create_devices_from_mmcore(core)
    assert "scanner_b" not in devices
    assert "piezo_b" not in devices
    assert "scanner" in devices
    assert "piezo" in devices
