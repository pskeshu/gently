"""
Tests for diSPIM device layer safety mechanisms.

Tests bounds checking, error cleanup, config validation, and state machine
enforcement using a mock CMMCore to avoid pymmcore/Micro-Manager dependency.
"""

import time
from unittest.mock import MagicMock

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Mock CMMCore — stands in for pymmcore.CMMCore
# ---------------------------------------------------------------------------


class MockCore:
    """Minimal mock of pymmcore.CMMCore for safety tests."""

    def __init__(self):
        self._positions = {}  # device_name -> position
        self._xy_position = (1000.0, 0.0)
        self._galvo_positions = {}  # device_name -> (a, b)
        self._properties = {}  # (device, prop) -> value
        self._configs = {}  # group -> current_config
        self._available_configs = {}  # group -> [configs]
        self._exposure = 10.0
        self._camera_device = ""
        self._focus_device = None
        self._circular_buffer = []
        self._sequence_running = False
        self._buffer_capacity = 1000

        # Track calls for verification
        self.call_log = []

    def setPosition(self, device, position):
        self.call_log.append(("setPosition", device, position))
        self._positions[device] = position

    def getPosition(self, device):
        return self._positions.get(device, 0.0)

    def waitForDevice(self, device):
        self.call_log.append(("waitForDevice", device))

    def setXYPosition(self, x, y):
        self.call_log.append(("setXYPosition", x, y))
        self._xy_position = (x, y)

    def getXYPosition(self):
        return self._xy_position

    def setGalvoPosition(self, device, a, b):
        self.call_log.append(("setGalvoPosition", device, a, b))
        self._galvo_positions[device] = (a, b)

    def getGalvoPosition(self, device):
        return self._galvo_positions.get(device, (0.0, 0.0))

    def setProperty(self, device, prop, value):
        self.call_log.append(("setProperty", device, prop, value))
        self._properties[(device, prop)] = value

    def getProperty(self, device, prop):
        return self._properties.get((device, prop), "")

    def setConfig(self, group, config):
        self.call_log.append(("setConfig", group, config))
        self._configs[group] = config

    def getCurrentConfig(self, group):
        return self._configs.get(group, "")

    def getAvailableConfigs(self, group):
        return self._available_configs.get(group, [])

    def waitForConfig(self, group, config):
        self.call_log.append(("waitForConfig", group, config))

    def setCameraDevice(self, name):
        self._camera_device = name

    def getCameraDevice(self):
        return self._camera_device

    def setFocusDevice(self, name):
        self._focus_device = name

    def setExposure(self, *args):
        if len(args) == 1:
            self._exposure = args[0]
        else:
            self._exposure = args[1]

    def getExposure(self):
        return self._exposure

    def snapImage(self):
        self.call_log.append(("snapImage",))

    def getImage(self):
        return np.zeros((512, 512), dtype=np.uint16)

    def setROI(self, device, x, y, w, h):
        self.call_log.append(("setROI", device, x, y, w, h))

    def clearCircularBuffer(self):
        self._circular_buffer = []

    def getBufferTotalCapacity(self):
        return self._buffer_capacity

    def setCircularBufferMemoryFootprint(self, mb):
        pass

    def prepareSequenceAcquisition(self, camera):
        self.call_log.append(("prepareSequenceAcquisition", camera))

    def startSequenceAcquisition(self, camera, count, interval, stopOnOverflow):
        self.call_log.append(("startSequenceAcquisition", camera, count))
        self._sequence_running = True
        self._sequence_expected = count
        # Pre-load images into buffer
        self._circular_buffer = [np.zeros((512, 512), dtype=np.uint16)] * count

    def stopSequenceAcquisition(self):
        self.call_log.append(("stopSequenceAcquisition",))
        self._sequence_running = False

    def isSequenceRunning(self):
        # Auto-stop when buffer is drained (mimics real camera behavior)
        if self._sequence_running and len(self._circular_buffer) == 0:
            self._sequence_running = False
        return self._sequence_running

    def getRemainingImageCount(self):
        return len(self._circular_buffer)

    def popNextImage(self):
        if self._circular_buffer:
            return self._circular_buffer.pop(0)
        return None


# ---------------------------------------------------------------------------
# Patch pymmcore and ophyd so device modules can be imported
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _patch_imports(monkeypatch):
    """Patch pymmcore and ophyd.status so device imports succeed."""
    import sys

    # Create minimal mock modules if not installed
    mock_modules = {}

    if "pymmcore" not in sys.modules:
        pymmcore_mock = MagicMock()
        pymmcore_mock.CMMCore = MockCore
        mock_modules["pymmcore"] = pymmcore_mock

    if "ophyd" not in sys.modules:
        ophyd_mock = MagicMock()
        mock_modules["ophyd"] = ophyd_mock

    if "ophyd.status" not in sys.modules:

        class _RealStatus:
            """Minimal Status that actually tracks finished/exception."""

            def __init__(self, obj=None, timeout=None):
                self._finished = False
                self._exception = None
                self._callbacks = []

            def set_finished(self):
                self._finished = True
                for cb in self._callbacks:
                    cb(self)

            def set_exception(self, exc):
                self._exception = exc
                self._finished = True
                for cb in self._callbacks:
                    cb(self)

            @property
            def done(self):
                return self._finished

            @property
            def success(self):
                return self._finished and self._exception is None

            def wait(self, timeout=None):
                # For sync tests: just spin briefly
                deadline = time.time() + (timeout or 5)
                while not self._finished and time.time() < deadline:
                    time.sleep(0.01)
                if self._exception:
                    raise self._exception

            def add_callback(self, cb):
                self._callbacks.append(cb)
                if self._finished:
                    cb(self)

        ophyd_status_mock = MagicMock()
        ophyd_status_mock.Status = _RealStatus
        mock_modules["ophyd.status"] = ophyd_status_mock

    for name, mod in mock_modules.items():
        monkeypatch.setitem(sys.modules, name, mod)


# ---------------------------------------------------------------------------
# Device constructors with mock core
# ---------------------------------------------------------------------------


def make_core():
    return MockCore()


def make_z_stage(core=None, limits=(50.0, 250.0)):
    from gently.hardware.dispim.devices.stage import DiSPIMZstage

    return DiSPIMZstage("ZStage", core or make_core(), limits=limits)


def make_xy_stage(core=None):
    from gently.hardware.dispim.devices.stage import DiSPIMXYStage

    return DiSPIMXYStage("XYStage", core or make_core())


def make_piezo(core=None, limits=(-200.0, 200.0)):
    from gently.hardware.dispim.devices.piezo import DiSPIMPiezo

    return DiSPIMPiezo("Piezo", core or make_core(), limits=limits)


def make_fdrive(core=None):
    from gently.hardware.dispim.devices.piezo import DiSPIMFDrive

    return DiSPIMFDrive("FDrive", core or make_core())


def make_scanner(core=None, limits=(-5.0, 5.0)):
    from gently.hardware.dispim.devices.scanner import DiSPIMScanner

    return DiSPIMScanner("Scanner", core or make_core(), limits=limits)


def make_led(core=None, available_configs=None):
    core = core or make_core()
    core._available_configs["LED"] = available_configs or ["Open", "Closed"]
    from gently.hardware.dispim.devices.optical import DiSPIMLED

    return DiSPIMLED(core, name="LED")


def make_laser(core=None, available_configs=None):
    core = core or make_core()
    configs = available_configs or ["488 and 561", "488 only", "561 only", "ALL OFF"]
    core._available_configs["Laser"] = configs
    from gently.hardware.dispim.devices.optical import DiSPIMLaserControl

    return DiSPIMLaserControl(core, name="Laser")


# ===========================================================================
# 1. BOUNDS CHECKING
# ===========================================================================


class TestZStageBounds:
    """Z stage must reject positions outside configured limits."""

    def test_valid_position_within_limits(self):
        stage = make_z_stage()
        status = stage.set(150.0)
        status.wait(timeout=2)
        assert stage.core._positions["ZStage"] == 150.0

    def test_position_at_lower_limit(self):
        stage = make_z_stage()
        status = stage.set(50.0)
        status.wait(timeout=2)
        assert stage.core._positions["ZStage"] == 50.0

    def test_position_at_upper_limit(self):
        stage = make_z_stage()
        status = stage.set(250.0)
        status.wait(timeout=2)
        assert stage.core._positions["ZStage"] == 250.0

    def test_position_below_lower_limit_raises(self):
        stage = make_z_stage()
        with pytest.raises(ValueError, match="outside limits"):
            stage.set(49.9)

    def test_position_above_upper_limit_raises(self):
        stage = make_z_stage()
        with pytest.raises(ValueError, match="outside limits"):
            stage.set(250.1)

    def test_custom_limits(self):
        stage = make_z_stage(limits=(0.0, 100.0))
        with pytest.raises(ValueError):
            stage.set(-1.0)
        status = stage.set(50.0)
        status.wait(timeout=2)
        assert stage.core._positions["ZStage"] == 50.0

    def test_core_not_called_on_rejected_position(self):
        """Safety: core.setPosition must NOT be called when position is invalid."""
        core = make_core()
        stage = make_z_stage(core=core)
        with pytest.raises(ValueError):
            stage.set(300.0)
        assert ("setPosition", "ZStage", 300.0) not in core.call_log


class TestXYStageBounds:
    """XY stage must reject positions outside configured limits for each axis."""

    def test_valid_xy_position(self):
        stage = make_xy_stage()
        x = (stage.x_limits[0] + stage.x_limits[1]) / 2.0
        y = (stage.y_limits[0] + stage.y_limits[1]) / 2.0
        status = stage.set([x, y])
        status.wait(timeout=2)
        assert stage.core._xy_position == (x, y)

    def test_x_below_lower_limit(self):
        stage = make_xy_stage()
        status = stage.set([stage.x_limits[0] - 1.0, 0.0])
        with pytest.raises(ValueError, match="outside hardware limits"):
            status.wait(timeout=2)

    def test_x_above_upper_limit(self):
        stage = make_xy_stage()
        status = stage.set([stage.x_limits[1] + 1.0, 0.0])
        with pytest.raises(ValueError, match="outside hardware limits"):
            status.wait(timeout=2)

    def test_y_below_lower_limit(self):
        stage = make_xy_stage()
        status = stage.set([0.0, stage.y_limits[0] - 1.0])
        with pytest.raises(ValueError, match="outside hardware limits"):
            status.wait(timeout=2)

    def test_y_above_upper_limit(self):
        stage = make_xy_stage()
        status = stage.set([0.0, stage.y_limits[1] + 1.0])
        with pytest.raises(ValueError, match="outside hardware limits"):
            status.wait(timeout=2)

    def test_core_not_called_on_invalid_x(self):
        core = make_core()
        stage = make_xy_stage(core=core)
        stage.set([stage.x_limits[0] - 1.0, 0.0])
        time.sleep(0.1)
        assert not any(c[0] == "setXYPosition" for c in core.call_log)


class TestPiezoBounds:
    """Piezo must reject positions outside ±200µm limits."""

    def test_valid_position(self):
        piezo = make_piezo()
        status = piezo.set(0.0)
        status.wait(timeout=2)
        assert piezo.core._positions["Piezo"] == 0.0

    def test_at_negative_limit(self):
        piezo = make_piezo()
        status = piezo.set(-200.0)
        status.wait(timeout=2)
        assert piezo.core._positions["Piezo"] == -200.0

    def test_below_negative_limit_raises(self):
        piezo = make_piezo()
        with pytest.raises(ValueError, match="outside limits"):
            piezo.set(-200.1)

    def test_above_positive_limit_raises(self):
        piezo = make_piezo()
        with pytest.raises(ValueError, match="outside limits"):
            piezo.set(200.1)


class TestFDriveBounds:
    """F-drive (SPIM head) must reject positions outside the 30-25000 µm
    hard limits (module constants F_DRIVE_MIN_UM / F_DRIVE_MAX_UM)."""

    def test_valid_position(self):
        fdrive = make_fdrive()
        status = fdrive.set(1000.0)
        status.wait(timeout=2)
        assert fdrive.core._positions["FDrive"] == 1000.0

    def test_below_lower_limit_raises(self):
        fdrive = make_fdrive()
        with pytest.raises(ValueError, match="outside hardware limits"):
            fdrive.set(19.9)

    def test_above_upper_limit_raises(self):
        fdrive = make_fdrive()
        with pytest.raises(ValueError, match="outside hardware limits"):
            fdrive.set(25001.0)

    def test_at_floor_allowed(self):
        fdrive = make_fdrive()
        status = fdrive.set(30.0)
        status.wait(timeout=2)
        assert fdrive.core._positions["FDrive"] == 30.0

    def test_at_ceiling_allowed(self):
        fdrive = make_fdrive()
        status = fdrive.set(25000.0)
        status.wait(timeout=2)
        assert fdrive.core._positions["FDrive"] == 25000.0

    def test_just_below_floor_raises(self):
        fdrive = make_fdrive()
        with pytest.raises(ValueError, match="outside hardware limits"):
            fdrive.set(29.99)

    def test_limits_property_is_module_constants(self):
        from gently.hardware.dispim.devices.piezo import (
            F_DRIVE_MAX_UM,
            F_DRIVE_MIN_UM,
        )

        fdrive = make_fdrive()
        assert fdrive.limits == (F_DRIVE_MIN_UM, F_DRIVE_MAX_UM)
        assert fdrive.limits == (30.0, 25000.0)


class TestScannerBounds:
    """Scanner must reject positions outside ±5° limits for both A and B axes."""

    def test_valid_ab_position(self):
        scanner = make_scanner()
        status = scanner.set([0.0, 0.0])
        status.wait(timeout=2)
        assert scanner.core._galvo_positions["Scanner"] == (0.0, 0.0)

    def test_a_axis_exceeds_limit(self):
        scanner = make_scanner()
        status = scanner.set([5.1, 0.0])
        with pytest.raises(ValueError, match="outside limits"):
            status.wait(timeout=2)

    def test_b_axis_exceeds_limit(self):
        scanner = make_scanner()
        status = scanner.set([0.0, -5.1])
        with pytest.raises(ValueError, match="outside limits"):
            status.wait(timeout=2)

    def test_both_axes_at_limits(self):
        scanner = make_scanner()
        status = scanner.set([5.0, -5.0])
        status.wait(timeout=2)
        assert scanner.core._galvo_positions["Scanner"] == (5.0, -5.0)

    def test_core_not_called_on_invalid_a(self):
        core = make_core()
        scanner = make_scanner(core=core)
        scanner.set([6.0, 0.0])
        time.sleep(0.1)
        assert not any(c[0] == "setGalvoPosition" for c in core.call_log)


# ===========================================================================
# 2. CONFIG VALIDATION
# ===========================================================================


class TestLEDConfigValidation:
    """LED must only accept configurations from the cached available set."""

    def test_valid_config_accepted(self):
        led = make_led()
        status = led.set("Open")
        status.wait(timeout=2)
        assert led.core._configs["LED"] == "Open"

    def test_invalid_config_rejected(self):
        led = make_led()
        with pytest.raises(ValueError, match="not available"):
            led.set("Strobe")

    def test_closed_config_accepted(self):
        led = make_led()
        status = led.set("Closed")
        status.wait(timeout=2)
        assert led.core._configs["LED"] == "Closed"


class TestLaserConfigValidation:
    """Laser must only accept configurations from the cached available set."""

    def test_valid_config_accepted(self):
        laser = make_laser()
        status = laser.set("488 and 561")
        status.wait(timeout=2)
        assert laser.core._configs["Laser"] == "488 and 561"

    def test_invalid_config_rejected(self):
        laser = make_laser()
        with pytest.raises(ValueError, match="not available"):
            laser.set("800nm femtosecond")

    def test_all_off_accepted(self):
        laser = make_laser()
        status = laser.set("ALL OFF")
        status.wait(timeout=2)
        assert laser.core._configs["Laser"] == "ALL OFF"


# ===========================================================================
# 3. STATE MACHINE / CONFIGURE-BEFORE-TRIGGER
# ===========================================================================


class TestVolumeScanner:
    """Volume scanner requires configure() before trigger()."""

    def _make_volume_scanner(self):
        core = make_core()
        # Set up laser available configs
        core._available_configs["Laser"] = ["488 and 561", "ALL OFF"]

        scanner = make_scanner(core=core)
        from gently.hardware.dispim.devices.camera import DiSPIMCamera

        camera = DiSPIMCamera("Camera", core)
        piezo = make_piezo(core=core)
        laser = make_laser(core=core)

        from gently.hardware.dispim.devices.acquisition import DiSPIMVolumeScanner

        return DiSPIMVolumeScanner(
            scanner=scanner, camera=camera, piezo=piezo, laser_control=laser, core=core
        )

    def test_trigger_before_configure_raises(self):
        vs = self._make_volume_scanner()
        with pytest.raises(RuntimeError, match="not configured"):
            vs.trigger()

    def test_configure_then_trigger_succeeds(self):
        vs = self._make_volume_scanner()
        vs.configure(
            num_slices=10,
            exposure_ms=5.0,
            galvo_amplitude=1.0,
            galvo_center=0.0,
            piezo_amplitude=50.0,
            piezo_center=0.0,
            laser_config="488 and 561",
        )
        assert vs._configured is True
        status = vs.trigger()
        status.wait(timeout=5)
        assert status.success


# ===========================================================================
# 4. HAPPY-PATH CLEANUP — LASER/LED OFF AFTER SUCCESS
# ===========================================================================


class TestVolumeScannerHappyPathCleanup:
    """Volume scanner must disable lasers and reset state after successful acquisition."""

    def _make_volume_scanner(self):
        core = make_core()
        core._available_configs["Laser"] = ["488 and 561", "ALL OFF"]

        scanner = make_scanner(core=core)
        from gently.hardware.dispim.devices.camera import DiSPIMCamera

        camera = DiSPIMCamera("Camera", core)
        piezo = make_piezo(core=core)
        laser = make_laser(core=core)

        from gently.hardware.dispim.devices.acquisition import DiSPIMVolumeScanner

        vs = DiSPIMVolumeScanner(
            scanner=scanner, camera=camera, piezo=piezo, laser_control=laser, core=core
        )
        vs.configure(
            num_slices=5,
            exposure_ms=5.0,
            galvo_amplitude=1.0,
            galvo_center=0.0,
            piezo_amplitude=50.0,
            piezo_center=0.0,
        )
        return vs, core

    def test_lasers_disabled_after_successful_acquisition(self):
        """Lasers must be turned off after normal acquisition completes."""
        vs, core = self._make_volume_scanner()
        status = vs.trigger()
        status.wait(timeout=5)
        assert status.success

        laser_off_calls = [
            c
            for c in core.call_log
            if c[0] == "setConfig" and c[1] == "Laser" and c[2] == "ALL OFF"
        ]
        assert len(laser_off_calls) >= 1, "Lasers must be disabled after successful acquisition"

    def test_spim_state_idle_after_successful_acquisition(self):
        """SPIM state machine must return to Idle after normal acquisition."""
        vs, core = self._make_volume_scanner()
        status = vs.trigger()
        status.wait(timeout=5)

        # Both scanner and piezo should be set to Idle
        idle_calls = [
            c
            for c in core.call_log
            if c[0] == "setProperty" and c[2] == "SPIMState" and c[3] == "Idle"
        ]
        assert len(idle_calls) >= 2, "Both scanner and piezo SPIM state must be reset to Idle"

    def test_camera_trigger_mode_reset_after_acquisition(self):
        """Camera must return to INTERNAL trigger after volume acquisition."""
        vs, core = self._make_volume_scanner()
        status = vs.trigger()
        status.wait(timeout=5)

        trigger_resets = [
            c
            for c in core.call_log
            if c[0] == "setProperty" and c[2] == "TRIGGER SOURCE" and c[3] == "INTERNAL"
        ]
        assert len(trigger_resets) >= 1, (
            "Camera must be reset to INTERNAL trigger after acquisition"
        )

    def test_laser_sequence_on_then_off(self):
        """Lasers must be enabled THEN disabled — verify ordering."""
        vs, core = self._make_volume_scanner()
        status = vs.trigger()
        status.wait(timeout=5)

        laser_calls = [c for c in core.call_log if c[0] == "setConfig" and c[1] == "Laser"]
        configs = [c[2] for c in laser_calls]
        assert configs == ["488 and 561", "ALL OFF"], f"Expected laser on then off, got: {configs}"


# ===========================================================================
# 5. TIMEOUT PROTECTION
# ===========================================================================


class TestVolumeScannerTimeout:
    """Volume scanner must not hang if image collection stalls."""

    def test_timeout_when_images_never_arrive(self):
        """If camera never delivers images, acquisition must timeout."""
        core = make_core()
        core._available_configs["Laser"] = ["488 and 561", "ALL OFF"]

        # Override: startSequenceAcquisition runs but delivers no images
        def stalling_start(camera, count, interval, stopOnOverflow):
            core.call_log.append(("startSequenceAcquisition", camera, count))
            core._sequence_running = True
            core._circular_buffer = []  # No images delivered

        # Override: sequence stays "running" forever
        def always_running():
            return core._sequence_running

        core.startSequenceAcquisition = stalling_start
        core.isSequenceRunning = always_running

        scanner = make_scanner(core=core)
        from gently.hardware.dispim.devices.camera import DiSPIMCamera

        camera = DiSPIMCamera("Camera", core)
        piezo = make_piezo(core=core)
        laser = make_laser(core=core)

        from gently.hardware.dispim.devices.acquisition import DiSPIMVolumeScanner

        vs = DiSPIMVolumeScanner(
            scanner=scanner, camera=camera, piezo=piezo, laser_control=laser, core=core
        )
        vs.configure(
            num_slices=10,
            exposure_ms=5.0,
            galvo_amplitude=1.0,
            galvo_center=0.0,
            piezo_amplitude=50.0,
            piezo_center=0.0,
        )

        # Shorten timeout (frozen dataclass — bypass with object.__setattr__)
        from gently.settings import settings

        original = settings.timeouts.volume_acquisition
        object.__setattr__(settings.timeouts, "volume_acquisition", 0.5)

        try:
            status = vs.trigger()
            with pytest.raises(TimeoutError):
                status.wait(timeout=5)

            # Even on timeout, lasers must be off
            laser_off = [
                c
                for c in core.call_log
                if c[0] == "setConfig" and c[1] == "Laser" and c[2] == "ALL OFF"
            ]
            assert len(laser_off) >= 1, "Lasers must be disabled even on timeout"
        finally:
            object.__setattr__(settings.timeouts, "volume_acquisition", original)


# ===========================================================================
# 6. ERROR CLEANUP — LASER/LED AUTO-DISABLE ON FAILURE
# ===========================================================================


class TestVolumeScannerErrorCleanup:
    """Volume scanner must disable lasers even when acquisition fails."""

    def _make_failing_volume_scanner(self, fail_at="sequence"):
        """Create volume scanner where core fails during acquisition."""
        core = make_core()
        core._available_configs["Laser"] = ["488 and 561", "ALL OFF"]

        if fail_at == "sequence":

            def failing_start(*args, **kwargs):
                core._sequence_running = False
                raise RuntimeError("Camera sequence failed")

            core.startSequenceAcquisition = failing_start

        scanner = make_scanner(core=core)
        from gently.hardware.dispim.devices.camera import DiSPIMCamera

        camera = DiSPIMCamera("Camera", core)
        piezo = make_piezo(core=core)
        laser = make_laser(core=core)

        from gently.hardware.dispim.devices.acquisition import DiSPIMVolumeScanner

        vs = DiSPIMVolumeScanner(
            scanner=scanner, camera=camera, piezo=piezo, laser_control=laser, core=core
        )
        vs.configure(
            num_slices=10,
            exposure_ms=5.0,
            galvo_amplitude=1.0,
            galvo_center=0.0,
            piezo_amplitude=50.0,
            piezo_center=0.0,
        )
        return vs, core

    def test_lasers_disabled_on_acquisition_error(self):
        """Lasers MUST be turned off even if acquisition fails."""
        vs, core = self._make_failing_volume_scanner()
        status = vs.trigger()
        with pytest.raises(RuntimeError):
            status.wait(timeout=5)

        # Verify lasers were turned off in cleanup
        laser_off_calls = [
            c
            for c in core.call_log
            if c[0] == "setConfig" and c[1] == "Laser" and c[2] == "ALL OFF"
        ]
        assert len(laser_off_calls) >= 1, "Lasers must be disabled on error"

    def test_spim_state_reset_on_error(self):
        """SPIM state machine must return to Idle on error."""
        vs, core = self._make_failing_volume_scanner()
        status = vs.trigger()
        with pytest.raises(RuntimeError):
            status.wait(timeout=5)

        # Verify scanner SPIM state was set to Idle in cleanup
        idle_calls = [
            c
            for c in core.call_log
            if c[0] == "setProperty" and c[2] == "SPIMState" and c[3] == "Idle"
        ]
        assert len(idle_calls) >= 1, "SPIM state must be reset to Idle on error"


# ===========================================================================
# 7. LED CLEANUP ON BOTTOM CAMERA
# ===========================================================================


class TestBottomCameraLEDCleanup:
    """Bottom camera must turn LED off even when capture fails."""

    def _make_bottom_camera(self, snap_fails=False):
        core = make_core()
        core._available_configs["LED"] = ["Open", "Closed"]

        if snap_fails:

            def failing_snap():
                raise RuntimeError("Camera snap failed")

            core.snapImage = failing_snap

        led = make_led(core=core)

        from gently.hardware.dispim.devices.camera import DiSPIMBottomCamera

        cam = DiSPIMBottomCamera("BottomCam", core, led_control=led)
        cam.use_led = True
        return cam, core

    def test_led_on_before_capture(self):
        cam, core = self._make_bottom_camera()
        status = cam.trigger()
        status.wait(timeout=5)

        # LED should have been opened then closed
        config_calls = [c for c in core.call_log if c[0] == "setConfig" and c[1] == "LED"]
        configs_set = [c[2] for c in config_calls]
        assert "Open" in configs_set, "LED must be turned on before capture"
        assert "Closed" in configs_set, "LED must be turned off after capture"

    def test_led_off_on_capture_error(self):
        """LED MUST be turned off even if snapImage fails."""
        cam, core = self._make_bottom_camera(snap_fails=True)
        status = cam.trigger()
        with pytest.raises(RuntimeError):
            status.wait(timeout=5)

        # Despite error, LED must have been closed
        config_calls = [c for c in core.call_log if c[0] == "setConfig" and c[1] == "LED"]
        closed_calls = [c for c in config_calls if c[2] == "Closed"]
        assert len(closed_calls) >= 1, "LED must be turned off even on capture error"

    def test_led_not_used_when_disabled(self):
        """When use_led=False, LED should not be touched."""
        cam, core = self._make_bottom_camera()
        cam.use_led = False
        status = cam.trigger()
        status.wait(timeout=5)

        led_calls = [c for c in core.call_log if c[0] == "setConfig" and c[1] == "LED"]
        assert len(led_calls) == 0, "LED should not be controlled when use_led=False"


# ===========================================================================
# 8. CALIBRATION PRIOR
# ===========================================================================


class TestCalibrationPrior:
    """CalibrationPrior tracks piezo-galvo relationship across embryos."""

    def test_default_state(self):
        from gently.hardware.dispim.calibration import CalibrationPrior

        prior = CalibrationPrior()
        assert prior.slope_um_per_deg == 100.0
        assert prior.num_calibrations == 0
        assert not prior.session_slope_locked
        assert not prior.is_ready_for_fast_calibration()

    def test_lock_session_slope(self):
        from gently.hardware.dispim.calibration import CalibrationPrior

        prior = CalibrationPrior()
        prior.lock_session_slope(slope=95.5, r_squared=0.98, embryo_id="e1")
        assert prior.session_slope_locked
        assert prior.slope_um_per_deg == 95.5
        assert prior.bootstrap_embryo_id == "e1"
        assert prior.is_ready_for_fast_calibration()

    def test_not_ready_with_low_r_squared(self):
        from gently.hardware.dispim.calibration import CalibrationPrior

        prior = CalibrationPrior()
        prior.lock_session_slope(slope=95.0, r_squared=0.5, embryo_id="e1")
        assert prior.session_slope_locked
        assert not prior.is_ready_for_fast_calibration()

    def test_ema_update(self):
        from gently.hardware.dispim.calibration import CalibrationPrior

        prior = CalibrationPrior()

        # First calibration — set directly
        prior.update_from_calibration(slope=95.0, offset=5.0, r_squared=0.95, extent_deg=0.3)
        assert prior.slope_um_per_deg == 95.0
        assert prior.num_calibrations == 1

        # Second calibration — EMA with alpha=0.3
        prior.update_from_calibration(slope=105.0, offset=10.0, r_squared=0.90, extent_deg=0.4)
        expected_slope = 0.3 * 105.0 + 0.7 * 95.0  # 98.0
        assert abs(prior.slope_um_per_deg - expected_slope) < 0.01
        assert prior.num_calibrations == 2

    def test_serialization_roundtrip(self):
        from gently.hardware.dispim.calibration import CalibrationPrior

        prior = CalibrationPrior()
        prior.lock_session_slope(slope=97.0, r_squared=0.96, embryo_id="e1")
        prior.update_from_calibration(slope=98.0, offset=3.0, r_squared=0.97, extent_deg=0.25)

        data = prior.to_dict()
        restored = CalibrationPrior.from_dict(data)

        assert restored.slope_um_per_deg == prior.slope_um_per_deg
        assert restored.session_slope_locked == prior.session_slope_locked
        assert restored.bootstrap_embryo_id == prior.bootstrap_embryo_id
        assert restored.num_calibrations == prior.num_calibrations


# ===========================================================================
# 9. ROUNDING / PRECISION
# ===========================================================================


class TestPositionRounding:
    """Devices should round to avoid floating-point precision issues."""

    def test_z_stage_rounds_to_hundredths(self):
        stage = make_z_stage()
        # Use 150.006 — avoids banker's rounding edge case with .005
        status = stage.set(150.006)
        status.wait(timeout=2)
        assert stage.core._positions["ZStage"] == 150.01

    def test_piezo_rounds_to_thousandths(self):
        piezo = make_piezo()
        # Use 50.0006 — unambiguous rounding
        status = piezo.set(50.0006)
        status.wait(timeout=2)
        assert piezo.core._positions["Piezo"] == 50.001

    def test_fdrive_rounds_to_hundredths(self):
        fdrive = make_fdrive()
        status = fdrive.set(1000.006)
        status.wait(timeout=2)
        assert fdrive.core._positions["FDrive"] == 1000.01
