"""
DiSPIM compound acquisition devices (volume scanner and light sheet snap).
"""

import logging
import time
from collections import OrderedDict

import numpy as np
import pymmcore
from ophyd.status import Status

from gently.settings import settings

from ._common import _safe_obtain
from .camera import DiSPIMCamera
from .optical import DiSPIMLaserControl
from .piezo import DiSPIMPiezo
from .scanner import DiSPIMScanner

logger = logging.getLogger(__name__)


class DiSPIMVolumeScanner:
    """
    Compound device for hardware-triggered SPIM volume acquisition.

    Orchestrates camera, scanner, piezo, and lasers for synchronized 3D volume capture.
    Handles all the complexity of circular buffer management, state machine
    coordination, and automatic laser enable/disable.

    Laser Management:
    - Automatically enables configured lasers before acquisition
    - Automatically disables lasers after acquisition (prevents photobleaching!)
    - Always disables lasers on error (critical for sample health)
    - Laser timing during scan controlled by scanner's LaserOutputMode property

    This device encapsulates the entire hardware-triggered acquisition workflow,
    allowing plans to simply call trigger_and_read([volume_scanner]).

    Parameters
    ----------
    scanner : DiSPIMScanner
        Scanner device for galvo control
    camera : DiSPIMCamera
        Camera device for image acquisition
    piezo : DiSPIMPiezo
        Piezo device for Z-axis scanning
    laser_control : DiSPIMLaserControl
        Laser control device for managing laser configs.
        Required to ensure explicit laser management (no accidental photobleaching!)
    core : pymmcore.CMMCore
        Micro-Manager core instance
    name : str
        Device name (default: "volume_scanner")
    """

    def __init__(
        self,
        scanner: DiSPIMScanner,
        camera: DiSPIMCamera,
        piezo: DiSPIMPiezo,
        laser_control: "DiSPIMLaserControl",
        core: pymmcore.CMMCore,
        name: str = "volume_scanner",
    ):
        """
        Initialize volume scanner with all required devices.

        Note: laser_control is required because proper SPIM operation always
        requires explicit laser management to avoid photobleaching and ensure
        reproducible illumination.
        """
        self.name = name
        self.parent = None  # Required for Bluesky
        self.scanner = scanner
        self.camera = camera
        self.piezo = piezo
        self.laser_control = laser_control
        self.core = core

        self._last_volume = None
        self._last_volume_time = None
        self._configured = False

        # Configuration cache
        self._num_slices: int | None = None
        self._exposure_ms: float | None = None
        self._laser_config: str | None = None

    def configure(
        self,
        num_slices: int,
        exposure_ms: float,
        galvo_amplitude: float,
        galvo_center: float,
        piezo_amplitude: float,
        piezo_center: float,
        laser_config: str = "488 and 561",
        laser_power_488_pct: float | None = None,
        laser_power_561_pct: float | None = None,
        laser_power_405_pct: float | None = None,
        laser_power_637_pct: float | None = None,
        timing_params: dict | None = None,
    ):
        """
        Configure all devices for hardware-triggered volume acquisition.

        Parameters
        ----------
        num_slices : int
            Number of Z slices in the volume
        exposure_ms : float
            Camera exposure time in milliseconds
        galvo_amplitude : float
            Galvo Y-axis amplitude in degrees (synchronized with piezo)
        galvo_center : float
            Galvo Y-axis center offset in degrees
        piezo_amplitude : float
            Piezo scanning amplitude in micrometers
        piezo_center : float
            Piezo center offset in micrometers
        laser_config : str
            Laser channel selection preset (default: "488 and 561").
            Common options: "488 and 561", "488 only", "561 only"
        laser_power_488_pct, laser_power_561_pct, laser_power_405_pct,
        laser_power_637_pct : float, optional
            Per-line laser power %. ``None`` leaves the current setpoint
            unchanged. Out-of-range values are rejected at the device-layer
            setter (see DiSPIMLightSource.POWER_LIMITS_PCT).
        timing_params : Dict, optional
            Custom SPIM timing parameters (uses defaults if None)
        """
        # Configure camera for hardware triggering
        self.camera.configure_for_volume_acquisition(exposure_ms)

        # Configure scanner for volume acquisition
        self.scanner.configure_for_volume_acquisition(
            galvo_amplitude=galvo_amplitude,
            galvo_center=galvo_center,
            num_slices=num_slices,
            timing_params=timing_params,
        )

        # Configure piezo for volume acquisition
        self.piezo.configure_for_volume_acquisition(
            amplitude_um=piezo_amplitude, offset_um=piezo_center, num_slices=num_slices
        )

        # Apply per-line laser power if specified. Each setter raises if the
        # value violates the hard safety bound — fail loudly, don't acquire
        # at a wrong power.
        power_settings = {
            488: laser_power_488_pct,
            561: laser_power_561_pct,
            405: laser_power_405_pct,
            637: laser_power_637_pct,
        }
        applied_powers = {}
        for wavelength, pct in power_settings.items():
            if pct is not None:
                self.laser_control.set_power_pct(wavelength, pct)
                applied_powers[wavelength] = pct

        self._num_slices = num_slices
        self._exposure_ms = exposure_ms
        self._laser_config = laser_config
        self._laser_powers_pct = applied_powers  # for metadata downstream
        self._configured = True

    def trigger(self):
        """
        Start hardware-triggered volume acquisition.

        Returns
        -------
        Status
            Ophyd status object that finishes when volume is acquired
        """
        if not self._configured:
            raise RuntimeError("Device not configured. Call configure() first.")

        status = Status(obj=self, timeout=120)

        def wait():
            try:
                # Enable lasers
                self.core.setConfig(self.laser_control.group_name, self._laser_config)
                self.core.waitForConfig(self.laser_control.group_name, self._laser_config)

                # Defensive stop: the lightsheet live streamer may have left the
                # camera in continuous acquisition.  Starting a new sequence on
                # an already-running camera triggers an MMCore error (and is
                # thread-unsafe).  Stop cleanly before reconfiguring.
                if self.core.isSequenceRunning():
                    self.core.stopSequenceAcquisition()
                    self.core.waitForDevice(self.camera.name)

                # Prepare circular buffer
                self.core.clearCircularBuffer()
                buffer_capacity = self.core.getBufferTotalCapacity()
                if buffer_capacity < self._num_slices:
                    self.core.setCircularBufferMemoryFootprint(512)
                    self.core.waitForDevice(self.camera.name)

                # Start sequence acquisition
                self.core.prepareSequenceAcquisition(self.camera.name)
                self.core.waitForDevice(self.camera.name)
                self.core.startSequenceAcquisition(self.camera.name, self._num_slices, 0, True)
                self.core.waitForDevice(self.camera.name)

                # Trigger SPIM state machine
                self.scanner.set_spim_state("Running")

                # Collect images
                images = []
                timeout_s = settings.timeouts.volume_acquisition
                start_time = time.time()

                while self.core.getRemainingImageCount() > 0 or self.core.isSequenceRunning():
                    if self.core.getRemainingImageCount() > 0:
                        img = self.core.popNextImage()

                        # Handle rpyc transfer
                        try:
                            img = _safe_obtain(img)
                        except (ImportError, AttributeError):
                            pass

                        images.append(img)

                    if time.time() - start_time > timeout_s:
                        raise TimeoutError(f"Volume acquisition timeout after {timeout_s:.1f}s")

                    time.sleep(0.01)

                # Stop sequence
                if self.core.isSequenceRunning():
                    self.core.stopSequenceAcquisition()

                # Reset hardware states
                self.camera.set_trigger_mode("INTERNAL")
                self.scanner.set_spim_state("Idle")
                self.piezo.set_spim_state("Idle")

                # Disable lasers (important for sample health!)
                self.core.setConfig(self.laser_control.group_name, "ALL OFF")
                self.core.waitForConfig(self.laser_control.group_name, "ALL OFF")

                # Store volume
                self._last_volume = np.array(images)
                self._last_volume_time = time.time()

            except Exception as exc:
                # Cleanup on error - always turn off lasers!
                try:
                    self.core.stopSequenceAcquisition()
                    self.camera.set_trigger_mode("INTERNAL")
                    self.scanner.set_spim_state("Idle")
                    self.piezo.set_spim_state("Idle")
                    # Critical: disable lasers even on error
                    self.core.setConfig(self.laser_control.group_name, "ALL OFF")
                    self.core.waitForConfig(self.laser_control.group_name, "ALL OFF")
                except Exception:
                    pass
                status.set_exception(exc)
            else:
                status.set_finished()

        import threading

        threading.Thread(target=wait).start()

        return status

    def read(self):
        """Read acquired volume data."""
        if self._last_volume is not None:
            data = OrderedDict()
            data[self.name] = {
                "value": self._last_volume,
                "timestamp": self._last_volume_time or time.time(),
            }
            return data
        else:
            return OrderedDict()

    def describe(self):
        """Describe volume data format."""
        data = OrderedDict()
        data[self.name] = {
            "source": self.name,
            "dtype": "array",
            "shape": getattr(self._last_volume, "shape", []),
            "units": "counts",
        }
        return data

    def read_configuration(self):
        """Required for Bluesky."""
        return OrderedDict()

    def describe_configuration(self):
        """Required for Bluesky."""
        return OrderedDict()


class DiSPIMLightSheetSnap:
    """
    Compound device for single light sheet image acquisition during calibration.

    Combines scanner and camera for synchronized single-image snapshots.
    Used during focus sweeps and piezo-galvo calibration.
    """

    def __init__(
        self,
        scanner: DiSPIMScanner,
        camera: DiSPIMCamera,
        name: str = "lightsheet_snap",
    ):
        self.name = name
        self.parent = None  # Required for Bluesky
        self.scanner = scanner
        self.camera = camera

        self._last_image = None
        self._last_image_time = None

    def configure(
        self,
        sheet_width_deg: float = 8.0,
        y_position_deg: float = 0.0,
        exposure_ms: float = 50.0,
    ):
        """
        Configure light sheet parameters for single snapshot.

        Parameters
        ----------
        sheet_width_deg : float
            Light sheet width (X-axis amplitude) in degrees (default: 8.0)
        y_position_deg : float
            Light sheet Y-position offset in degrees (default: 0.0)
        exposure_ms : float
            Camera exposure time in milliseconds (default: 50.0)
        """
        # Configure scanner for continuous light sheet
        self.scanner.configure_for_calibration()
        self.scanner.set_y_offset(y_position_deg)

        # Configure camera for single snapshot
        self.camera.configure_for_calibration(exposure_ms)

    def set_y_position(self, angle_deg: float):
        """
        Adjust light sheet Y-position for focus sweeps.

        Parameters
        ----------
        angle_deg : float
            Y-axis offset angle in degrees
        """
        self.scanner.set_y_offset(angle_deg)

    def trigger(self):
        """
        Capture single light sheet image.

        Returns
        -------
        Status
            Ophyd status object for the capture
        """
        return self.camera.trigger()

    def read(self):
        """Read captured image."""
        return self.camera.read()

    def describe(self):
        """Describe image data format."""
        return self.camera.describe()

    def read_configuration(self):
        """Required for Bluesky."""
        return OrderedDict()

    def describe_configuration(self):
        """Required for Bluesky."""
        return OrderedDict()
