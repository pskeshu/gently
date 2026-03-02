"""
DiSPIM piezo and F-drive positioner devices.
"""

import time
import logging
from collections import OrderedDict
from typing import Tuple

from ophyd.status import Status
import pymmcore

from ..exceptions import HardwareError, StageMovementError

logger = logging.getLogger(__name__)


class DiSPIMFDrive:
    """
    DiSPIM F-drive (SPIM Head motor) - works with bps.mv(fdrive, position)

    ASI Tiger V:37 axis - controls F-axis module for lowering objectives
    Device-agnostic: any plan that moves a positioner will work with this device
    """

    def __init__(self, name: str, core: pymmcore.CMMCore,
                 limits: Tuple[float, float] = (20.0, 25000.0)):
        self.name = name
        self.core = core
        self.parent = None  # Required for Bluesky
        self._limits = limits
        self.tolerance = 0.1  # µm

    @property
    def limits(self):
        return self._limits

    def set(self, position):
        """Move F-drive to position - called by bps.mv()"""
        position = float(position)
        position = round(position, 2)  # Round to 0.01 μm precision

        # Safety check
        if not (self._limits[0] <= position <= self._limits[1]):
            raise ValueError(f"Position {position} outside limits {self._limits}")

        status = Status(obj=self, timeout=30)

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
            'value': float(value),
            'timestamp': time.time(),
            'units': 'micrometers'
        }
        return data

    def describe(self):
        """Describe F-drive device - required for Bluesky"""
        data = OrderedDict()
        data[self.name] = {
            'source': self.name,
            'dtype': 'number',
            'shape': [],
            'units': 'micrometers'
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

    def __init__(self, name: str, core: pymmcore.CMMCore,
                 limits: Tuple[float, float] = (-200, 200.0)):
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
            'value': float(value),
            'timestamp': time.time(),
            'units': 'micrometers'
        }
        return data

    def describe(self):
        """Describe piezo device - required for Bluesky"""
        data = OrderedDict()
        data[self.name] = {
            'source': self.name,
            'dtype': 'number',
            'shape': [],
            'units': 'micrometers'
        }
        return data

    def read_configuration(self):
        """Required for Bluesky"""
        return OrderedDict()

    def describe_configuration(self):
        """Required for Bluesky"""
        return OrderedDict()

    # Hardware configuration methods for SPIM
    def set_as_focus_device(self):
        """Set this piezo as the Micro-Manager focus device."""
        self.core.setFocusDevice(self.name)

    def configure_amplitude_offset(self,
                                    amplitude_um: float,
                                    offset_um: float,
                                    pattern: str = "1 - Triangle"):
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

    def configure_for_volume_acquisition(self,
                                          amplitude_um: float,
                                          offset_um: float,
                                          num_slices: int):
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
