"""
DiSPIM piezo and F-drive positioner devices.
"""

import logging
import time
from collections import OrderedDict

import pymmcore
from ophyd.status import Status

from gently.exceptions import HardwareError, StageMovementError

logger = logging.getLogger(__name__)


# =========================================================================
# SPIM-HEAD F-DRIVE HARDWARE SAFETY LIMITS — absolute MMCore micrometres.
#
# Layer-1 software fence for the SPIM head: the ASI Tiger "ZStage:V:37"
# axis, which the ASIdiSPIM plugin labels "SPIM Head Height" (the F axis).
# This is the drive that LOWERS the objectives into the dish to hunt for
# embryos ("Start Hunting") and RAISES them clear for sample loading
# ("Load Sample"). Every F-drive move planned by any layer above (Bluesky
# plans, agent orchestrators, UI tools) is bounded here. These are NOT
# constructor kwargs and DiSPIMFDrive exposes no setter — no layer above
# can widen them.
#
#   F_DRIVE_MIN_UM  collision-critical FLOOR. Smaller F drives the head
#                   DOWN toward the sample/holder; this hard stop keeps the
#                   objectives off the dish.
#   F_DRIVE_MAX_UM  fully-raised "Load Sample" ceiling.
#
# Update only after physically verifying head travel on the rig (drive the
# F axis to each extreme by hand, confirm no collision, read the absolute
# MMCore position) and then editing the constants below.
#
# SCOPE: this is a SOFTWARE-ONLY fence. Unlike the XY stage (see
# devices/stage.py, which also pushes ASI Tiger firmware soft-limits) we do
# NOT write these to the controller, so a physical joystick move can still
# drive the head past these bounds — they bind code-issued moves only.
# =========================================================================
F_DRIVE_MIN_UM: float = 30.0
F_DRIVE_MAX_UM: float = 25000.0


class DiSPIMFDrive:
    """
    DiSPIM F-drive (SPIM Head motor) - works with bps.mv(fdrive, position)

    ASI Tiger "ZStage:V:37" axis — the ASIdiSPIM "SPIM Head Height" / F
    axis that lowers the objectives to hunt for embryos and raises them to
    load a sample. Device-agnostic: any plan that moves a positioner works.

    Hard travel bounds are the module-level F_DRIVE_MIN_UM / F_DRIVE_MAX_UM
    constants. They are not constructor kwargs and cannot be widened from
    above — see the safety-limit note above this class.
    """

    def __init__(self, name: str, core: pymmcore.CMMCore, move_timeout_s: float = 120.0):
        self.name = name
        self.core = core
        self.parent = None  # Required for Bluesky
        # Full-travel F moves (e.g. 25000 -> 5000 um: "Load Sample" -> approach)
        # are slow; the per-move Status timeout must comfortably exceed the
        # longest traverse or bps.mv would error mid-move while the stage is
        # still travelling. Configurable for unusually slow controllers.
        self._move_timeout_s = float(move_timeout_s)
        self.tolerance = 0.1  # µm

    @property
    def limits(self) -> tuple[float, float]:
        """Read-only view of the hardware safety limits (module constants)."""
        return (F_DRIVE_MIN_UM, F_DRIVE_MAX_UM)

    def set(self, position):
        """Move F-drive to position - called by bps.mv()"""
        position = float(position)
        position = round(position, 2)  # Round to 0.01 μm precision

        # Hardware safety check — pinned to the module-level F_DRIVE_*_UM
        # constants; nothing above this layer can widen them.
        if not (F_DRIVE_MIN_UM <= position <= F_DRIVE_MAX_UM):
            raise ValueError(
                f"F-drive position {position} outside hardware limits "
                f"[{F_DRIVE_MIN_UM}, {F_DRIVE_MAX_UM}]"
            )

        status = Status(obj=self, timeout=self._move_timeout_s)

        def wait():
            try:
                self.core.setPosition(self.name, position)
                self.core.waitForDevice(self.name)
            except (RuntimeError, StageMovementError) as exc:
                status.set_exception(exc)
            else:
                status.set_finished()

        import threading

        threading.Thread(target=wait).start()

        return status

    def read(self):
        """Read current F-drive position - required for Bluesky"""
        try:
            value = self.core.getPosition(self.name)
        except (RuntimeError, HardwareError) as e:
            logger.error("Failed to read position from %s: %s", self.name, e)

        data = OrderedDict()
        data[self.name] = {
            "value": float(value),
            "timestamp": time.time(),
            "units": "micrometers",
        }
        return data

    def describe(self):
        """Describe F-drive device - required for Bluesky"""
        data = OrderedDict()
        data[self.name] = {
            "source": self.name,
            "dtype": "number",
            "shape": [],
            "units": "micrometers",
        }
        return data

    def read_configuration(self):
        """Required for Bluesky"""
        return OrderedDict()

    def describe_configuration(self):
        """Required for Bluesky"""
        return OrderedDict()


class DiSPIMPiezo:
    """
    DiSPIM Piezo stage - works with bps.mv(piezo, position)

    ASI Tiger PiezoStage (P:34 or Q:35) - objective focus control
    Device-agnostic: any plan that moves a positioner will work with this device
    """

    def __init__(
        self,
        name: str,
        core: pymmcore.CMMCore,
        limits: tuple[float, float] = (-200, 200.0),
    ):
        self.name = name
        self.core = core
        self.parent = None  # Required for Bluesky
        self._limits = limits
        self.tolerance = 0.01  # µm

    @property
    def limits(self):
        return self._limits

    def set(self, position):
        """Move piezo to position - called by bps.mv()"""
        position = float(position)
        position = round(position, 3)  # Round to 0.001 μm precision

        # Safety check
        if not (self._limits[0] <= position <= self._limits[1]):
            raise ValueError(f"Position {position} outside limits {self._limits}")

        status = Status(obj=self, timeout=10)

        def wait():
            try:
                self.core.setPosition(self.name, position)
                self.core.waitForDevice(self.name)
            except (RuntimeError, StageMovementError) as exc:
                status.set_exception(exc)
            else:
                status.set_finished()

        import threading

        threading.Thread(target=wait).start()

        return status

    def read(self):
        """Read current piezo position - required for Bluesky"""
        try:
            value = self.core.getPosition(self.name)
        except (RuntimeError, HardwareError) as e:
            logger.error("Failed to read position from %s: %s", self.name, e)
            value = 0.0

        data = OrderedDict()
        data[self.name] = {
            "value": float(value),
            "timestamp": time.time(),
            "units": "micrometers",
        }
        return data

    def describe(self):
        """Describe piezo device - required for Bluesky"""
        data = OrderedDict()
        data[self.name] = {
            "source": self.name,
            "dtype": "number",
            "shape": [],
            "units": "micrometers",
        }
        return data

    def read_configuration(self):
        """Required for Bluesky"""
        return OrderedDict()

    def describe_configuration(self):
        """Required for Bluesky"""
        return OrderedDict()

    def setPosition(self, value: float) -> None:
        """Synchronous move: drive to ``value`` and block until the piezo
        reports settled. For use outside Bluesky plans — equivalent to
        ``set(value).wait()`` minus the Status/thread plumbing. Safety bounds
        match the Bluesky ``set()`` path.
        """
        value = float(value)
        if not (self._limits[0] <= value <= self._limits[1]):
            raise ValueError(f"Position {value} outside limits {self._limits}")
        self.core.setPosition(self.name, value)
        self.core.waitForDevice(self.name)

    # Hardware configuration methods for SPIM
    def set_as_focus_device(self):
        """Set this piezo as the Micro-Manager focus device."""
        self.core.setFocusDevice(self.name)

    def configure_amplitude_offset(
        self, amplitude_um: float, offset_um: float, pattern: str = "1 - Triangle"
    ):
        """
        Configure piezo amplitude and offset for scanning.

        Parameters
        ----------
        amplitude_um : float
            Scanning amplitude in micrometers
        offset_um : float
            Center offset in micrometers
        pattern : str
            Waveform pattern (default: "1 - Triangle")
        """
        self.core.setProperty(self.name, "SingleAxisAmplitude(um)", float(amplitude_um))
        self.core.setProperty(self.name, "SingleAxisOffset(um)", float(offset_um))
        self.core.setProperty(self.name, "SingleAxisPattern", pattern)

    def set_spim_state(self, state: str):
        """
        Set SPIM state for piezo.

        Parameters
        ----------
        state : str
            'Idle' to stop or 'Armed' to prepare for hardware triggering
        """
        self.core.setProperty(self.name, "SPIMState", state)
        if state == "Armed":
            self.core.waitForDevice(self.name)

    def configure_for_spim(self, num_slices: int):
        """
        Configure piezo for SPIM acquisition.

        Parameters
        ----------
        num_slices : int
            Number of Z slices for the volume
        """
        self.core.setProperty(self.name, "SPIMNumSlices", num_slices)

    def configure_for_volume_acquisition(
        self, amplitude_um: float, offset_um: float, num_slices: int
    ):
        """
        Configure piezo for hardware-triggered volume acquisition.

        Combines all necessary setup steps: sets as focus device, configures
        amplitude/offset, sets SPIM parameters, and arms the device.

        Parameters
        ----------
        amplitude_um : float
            Piezo scanning amplitude in micrometers
        offset_um : float
            Piezo center offset in micrometers
        num_slices : int
            Number of Z slices in volume
        """
        self.set_as_focus_device()
        self.configure_amplitude_offset(amplitude_um, offset_um)
        self.configure_for_spim(num_slices)
        self.set_spim_state("Armed")
