"""
DiSPIM scanner/galvo mirror control devices.
"""

import logging
import time
from collections import OrderedDict

import numpy as np
import pymmcore
from ophyd.status import Status

from gently.exceptions import HardwareError, StageMovementError

logger = logging.getLogger(__name__)


class _ScannerAxisOffset:
    """
    Movable component for a single scanner axis offset.
    Compatible with bps.mv() for use in Bluesky plans.
    """

    def __init__(self, scanner, axis: str, property_name: str):
        self.scanner = scanner
        self.axis = axis
        self.property_name = property_name
        self.name = f"{scanner.name}_{axis}_offset"
        self.parent = scanner

    def set(self, value):
        """Move axis offset to specified position - called by bps.mv()"""
        status = Status(obj=self, timeout=5)

        def wait():
            try:
                self.scanner.core.setProperty(self.scanner.name, self.property_name, float(value))
                self.scanner.core.waitForDevice(self.scanner.name)
            except (RuntimeError, StageMovementError) as exc:
                status.set_exception(exc)
            else:
                status.set_finished()

        import threading

        threading.Thread(target=wait).start()
        return status

    def setPosition(self, value: float) -> None:
        """Synchronous set: writes the offset property and blocks until the
        device settles. For use outside Bluesky plans (or inside plans that
        prefer sync semantics) — equivalent to ``set(value).wait()`` minus
        the Status/thread plumbing. MMCore traffic stays inside this ophyd
        boundary.
        """
        self.scanner.core.setProperty(self.scanner.name, self.property_name, float(value))
        self.scanner.core.waitForDevice(self.scanner.name)

    def read(self):
        """Read current offset value"""
        try:
            value = float(self.scanner.core.getProperty(self.scanner.name, self.property_name))
        except (RuntimeError, HardwareError):
            value = 0.0

        return OrderedDict(
            {self.name: {"value": value, "timestamp": time.time(), "units": "degrees"}}
        )

    def describe(self):
        """Describe component"""
        return OrderedDict(
            {
                self.name: {
                    "source": self.name,
                    "dtype": "number",
                    "shape": [],
                    "units": "degrees",
                }
            }
        )


class DiSPIMScanner:
    """
    DiSPIM Scanner/Galvo control - works with bps.mv(scanner, [a_pos, b_pos])

    ASI Tiger Scanner (AB:33 or CD:33) - controls galvo mirrors for light sheet
    Device-agnostic: any plan that moves a 2D positioner will work with this device

    Individual axis offsets can be moved with:
        bps.mv(scanner.sa_offset_x, x_value)  # X-axis offset
        bps.mv(scanner.sa_offset_y, y_value)  # Y-axis offset (galvo position)
    """

    def __init__(
        self,
        name: str,
        core: pymmcore.CMMCore,
        limits: tuple[float, float] = (-5.0, 5.0),
    ):
        self.name = name
        self.core = core
        self.parent = None  # Required for Bluesky
        self._limits = limits

        # Create movable axis offset components for use with bps.mv()
        self.sa_offset_x = _ScannerAxisOffset(self, "x", "SingleAxisXOffset(deg)")
        self.sa_offset_y = _ScannerAxisOffset(self, "y", "SingleAxisYOffset(deg)")

    @property
    def limits(self):
        return self._limits

    def set(self, position):
        """Move scanner to position [a, b] - called by bps.mv()"""
        try:
            a_pos, b_pos = position
            a_pos = float(a_pos)
            b_pos = float(b_pos)

            # Safety checks
            if not (self._limits[0] <= a_pos <= self._limits[1]):
                raise ValueError(f"A position {a_pos} outside limits {self._limits}")
            if not (self._limits[0] <= b_pos <= self._limits[1]):
                raise ValueError(f"B position {b_pos} outside limits {self._limits}")

            status = Status(obj=self, timeout=5)

            def wait():
                try:
                    # Scanner uses galvo position interface for AB axes
                    self.core.setGalvoPosition(self.name, a_pos, b_pos)
                    self.core.waitForDevice(self.name)
                except (RuntimeError, StageMovementError) as exc:
                    status.set_exception(exc)
                else:
                    status.set_finished()

            import threading

            threading.Thread(target=wait).start()

            return status

        except (ValueError, TypeError) as e:
            status = Status(self)
            status.set_exception(e)
            return status

    def read(self):
        """Read current scanner positions - required for Bluesky"""
        try:
            # getGalvoPosition returns tuple (a, b) voltages for galvo device
            ab_pos = np.array(self.core.getGalvoPosition(self.name))
        except (RuntimeError, HardwareError) as e:
            logger.error("Failed to read scanner positions from %s: %s", self.name, e)
            ab_pos = np.array([0.0, 0.0])

        data = OrderedDict()
        data[self.name] = {"value": ab_pos, "timestamp": time.time(), "units": "volts"}
        return data

    def describe(self):
        """Describe scanner device - required for Bluesky"""
        data = OrderedDict()
        data[self.name] = {
            "source": self.name,
            "dtype": "array",
            "shape": [2],
            "units": "volts",
        }
        return data

    def read_configuration(self):
        """Required for Bluesky"""
        return OrderedDict()

    def describe_configuration(self):
        """Required for Bluesky"""
        return OrderedDict()

    # Hardware configuration methods for SPIM scanning
    def enable_beam(self, enabled: bool = True):
        """
        Enable or disable the laser beam.

        Parameters
        ----------
        enabled : bool
            True to enable beam, False to disable
        """
        self.core.setProperty(self.name, "BeamEnabled", "Yes" if enabled else "No")

    def set_laser_output_mode(self, mode: str):
        """
        Set laser output mode.

        Parameters
        ----------
        mode : str
            Laser output mode, e.g., "shutter + side" for side A imaging
        """
        self.core.setProperty(self.name, "LaserOutputMode", mode)

    def set_spim_state(self, state: str):
        """
        Set SPIM state machine state.

        Parameters
        ----------
        state : str
            'Idle' to stop, 'Armed' to prepare, 'Running' to trigger acquisition
        """
        self.core.setProperty(self.name, "SPIMState", state)
        if state == "Idle":
            self.core.waitForDevice(self.name)

    def configure_x_axis(
        self,
        amplitude_deg: float,
        offset_deg: float,
        pattern: str = "1 - Triangle",
        mode: str = "3 - Enabled with axes synced",
    ):
        """
        Configure galvo X-axis (light sheet width scanning).

        Parameters
        ----------
        amplitude_deg : float
            Scanning amplitude in degrees (typically 8.0 for full light sheet)
        offset_deg : float
            Center offset in degrees
        pattern : str
            Waveform pattern (default: "1 - Triangle")
        mode : str
            Scan mode (default: "3 - Enabled with axes synced")
        """
        self.core.setProperty(self.name, "SingleAxisXAmplitude(deg)", amplitude_deg)
        self.core.setProperty(self.name, "SingleAxisXOffset(deg)", offset_deg)
        self.core.setProperty(self.name, "SingleAxisXPattern", pattern)
        self.core.setProperty(self.name, "SingleAxisXMode", mode)

    def configure_y_axis(
        self,
        amplitude_deg: float,
        offset_deg: float,
        pattern: str = "1 - Triangle",
        mode: str = "3 - Enabled with axes synced",
    ):
        """
        Configure galvo Y-axis (light sheet Z-plane positioning).

        Parameters
        ----------
        amplitude_deg : float
            Scanning amplitude in degrees (synchronized with piezo for volume scanning)
        offset_deg : float
            Center offset in degrees (positions the light sheet vertically)
        pattern : str
            Waveform pattern (default: "1 - Triangle")
        mode : str
            Scan mode (default: "3 - Enabled with axes synced")
        """
        self.core.setProperty(self.name, "SingleAxisYAmplitude(deg)", amplitude_deg)
        self.core.setProperty(self.name, "SingleAxisYOffset(deg)", offset_deg)
        self.core.setProperty(self.name, "SingleAxisYPattern", pattern)
        self.core.setProperty(self.name, "SingleAxisYMode", mode)

    def set_y_offset(self, angle_deg: float):
        """
        Set Y-axis offset for light sheet positioning.

        Used during calibration to move the light sheet to different Z planes.

        Parameters
        ----------
        angle_deg : float
            Y-axis offset angle in degrees
        """
        self.core.setProperty(self.name, "SingleAxisYOffset(deg)", float(angle_deg))
        self.core.waitForDevice(self.name)

    def configure_spim_timing(
        self,
        scan_delay_ms: float = 6.75,
        num_scans_per_slice: int = 1,
        scan_duration_ms: float = 5.5,
        laser_delay_ms: float = 8.0,
        laser_duration_ms: float = 5.0,
        camera_delay_ms: float = 8.0,
        camera_duration_ms: float = 1.0,
    ):
        """
        Configure SPIM timing parameters for hardware-triggered acquisition.

        Parameters
        ----------
        scan_delay_ms : float
            Delay before starting galvo scan (default: 6.75ms)
        num_scans_per_slice : int
            Number of galvo scans per Z slice (default: 1)
        scan_duration_ms : float
            Duration of galvo scan (default: 5.5ms)
        laser_delay_ms : float
            Delay before laser pulse (default: 8.0ms)
        laser_duration_ms : float
            Duration of laser pulse (default: 5.0ms)
        camera_delay_ms : float
            Delay before camera trigger (default: 8.0ms)
        camera_duration_ms : float
            Duration of camera exposure (default: 1.0ms)
        """
        self.core.setProperty(self.name, "SPIMDelayBeforeScan(ms)", scan_delay_ms)
        self.core.setProperty(self.name, "SPIMNumScansPerSlice", num_scans_per_slice)
        self.core.setProperty(self.name, "SPIMScanDuration(ms)", scan_duration_ms)
        self.core.setProperty(self.name, "SPIMDelayBeforeLaser(ms)", laser_delay_ms)
        self.core.setProperty(self.name, "SPIMLaserDuration(ms)", laser_duration_ms)
        self.core.setProperty(self.name, "SPIMDelayBeforeCamera(ms)", camera_delay_ms)
        self.core.setProperty(self.name, "SPIMCameraDuration(ms)", camera_duration_ms)

    def configure_spim_parameters(
        self,
        num_slices: int,
        slices_per_piezo: int = 1,
        num_sides: int = 1,
        first_side: str = "A",
    ):
        """
        Configure SPIM acquisition parameters.

        Parameters
        ----------
        num_slices : int
            Total number of Z slices in volume
        slices_per_piezo : int
            Number of slices per piezo step (default: 1)
        num_sides : int
            Number of SPIM sides (1 or 2, default: 1)
        first_side : str
            First side to image ('A' or 'B', default: 'A')
        """
        self.core.setProperty(self.name, "SPIMNumSlices", num_slices)
        self.core.setProperty(self.name, "SPIMNumSlicesPerPiezo", slices_per_piezo)
        self.core.setProperty(self.name, "SPIMNumSides", num_sides)
        self.core.setProperty(self.name, "SPIMFirstSide", first_side)

    def configure_for_calibration(self):
        """
        Configure scanner for calibration (continuous light sheet for focus sweeps).

        Sets up:
        - Enabled beam
        - Laser output mode for triggering lasers
        - X-axis scanning (8° amplitude for full light sheet width)
        - Y-axis with minimal amplitude (will adjust offset for positioning)
        """
        self.enable_beam(True)
        self.set_laser_output_mode("shutter + side")  # Enable laser triggering
        self.configure_x_axis(amplitude_deg=8.0, offset_deg=0.0005)
        self.configure_y_axis(amplitude_deg=0.0001, offset_deg=0.0)
        self.core.waitForDevice(self.name)

    def configure_for_volume_acquisition(
        self,
        galvo_amplitude: float,
        galvo_center: float,
        num_slices: int,
        timing_params: dict | None = None,
    ):
        """
        Configure scanner for hardware-triggered volume acquisition.

        Sets up the complete SPIM state machine for synchronized piezo/galvo/camera scanning.

        Parameters
        ----------
        galvo_amplitude : float
            Galvo Y-axis amplitude in degrees (matched to piezo amplitude)
        galvo_center : float
            Galvo Y-axis center offset in degrees
        num_slices : int
            Number of Z slices
        timing_params : Dict, optional
            Custom timing parameters (uses defaults if None)
        """
        # Reset state machine
        self.set_spim_state("Idle")
        self.set_laser_output_mode("shutter + side")
        self.enable_beam(False)

        # Configure scanning axes
        self.configure_x_axis(amplitude_deg=8.0, offset_deg=0.0005)
        self.configure_y_axis(amplitude_deg=galvo_amplitude, offset_deg=galvo_center)

        # Configure timing (use defaults if not provided)
        if timing_params is None:
            self.configure_spim_timing()
        else:
            self.configure_spim_timing(**timing_params)

        # Configure acquisition parameters
        self.configure_spim_parameters(num_slices=num_slices)
