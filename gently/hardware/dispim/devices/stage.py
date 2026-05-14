"""
DiSPIM stage positioner devices (Z-stage and XY-stage).
"""

import time
import logging
from collections import OrderedDict
from typing import Tuple

import numpy as np

from ophyd.status import Status
import pymmcore

from gently.exceptions import HardwareError, StageMovementError

logger = logging.getLogger(__name__)


# =========================================================================
# XY-STAGE HARDWARE SAFETY LIMITS — absolute MMCore micrometres.
#
# Layer 0 of the motion-safety stack. Every XY move planned by any layer
# above (Bluesky plans, agent orchestrators, UI tools) is bounded here.
# No layer above can widen these — they are not constructor kwargs and
# DiSPIMXYStage exposes no setter for them.
#
# Update only after physically verifying the new bounds on the rig:
#   1. Drive the stage manually to each corner using the joystick.
#   2. Confirm no collisions with the SPIM head, optics, or sample holder.
#   3. Read the absolute MMCore X / Y values from the live device-state
#      stream (or the XY Stage readout in the Devices > Map view).
#   4. Edit the four constants below; restart the device-layer process.
# =========================================================================
XY_STAGE_X_MIN_UM: float = -3242.55
XY_STAGE_X_MAX_UM: float = 1676.06
XY_STAGE_Y_MIN_UM: float = -2531.39
XY_STAGE_Y_MAX_UM: float = 1461.16


class DiSPIMZstage:
    """
    DiSPIM Z Stage positioner - works with bps.mv(z_stage, position)

    Device-agnostic: any plan that moves a positioner will work with this device
    """

    def __init__(self, name: str, core: pymmcore.CMMCore,
                 limits: Tuple[float, float] = (50.0, 250.0)):
        self.name = name
        self.core = core
        self.parent = None  # Required for Bluesky
        self._limits = limits
        self.tolerance = 0.1  # µm

    @property
    def limits(self):
        return self._limits

    def set(self, position):
        """Move Z stage to position - called by bps.mv()"""
        position = float(position)

        # Round to avoid floating point precision issues
        position = round(position, 2)  # Round to 0.01 μm precision

        # Safety check
        if not (self._limits[0] <= position <= self._limits[1]):
            raise ValueError(f"Position {position} outside limits {self._limits}")

        # Direct MM core implementation like deepthought
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
        """Read current Z stage position - required for Bluesky"""
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
        """Describe Z stage device - required for Bluesky"""
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


class DiSPIMXYStage:
    """
    DiSPIM XY stage - works with bps.mv(xy_stage, [x, y])

    Device-agnostic: any plan that moves XY positions will work with this device
    Based on deepthought XYStage implementation
    """

    def __init__(self, name: str, core: pymmcore.CMMCore):
        self.name = name
        self.core = core
        self.parent = None  # Required for Bluesky

    @property
    def x_limits(self) -> Tuple[float, float]:
        """Read-only view of the hardware safety limits (module constants)."""
        return (XY_STAGE_X_MIN_UM, XY_STAGE_X_MAX_UM)

    @property
    def y_limits(self) -> Tuple[float, float]:
        """Read-only view of the hardware safety limits (module constants)."""
        return (XY_STAGE_Y_MIN_UM, XY_STAGE_Y_MAX_UM)

    def set(self, position):
        """Move XY stage to position [x, y] - called by bps.mv(xy_stage, [x, y])"""
        try:
            x, y = position  # Unpack [x, y] coordinates
            x = float(x)
            y = float(y)

            # Hardware safety check — values pinned to the module-level
            # XY_STAGE_*_UM constants; nothing above this layer can widen them.
            if not (XY_STAGE_X_MIN_UM <= x <= XY_STAGE_X_MAX_UM):
                raise ValueError(
                    f"X position {x} outside hardware limits "
                    f"[{XY_STAGE_X_MIN_UM}, {XY_STAGE_X_MAX_UM}]"
                )
            if not (XY_STAGE_Y_MIN_UM <= y <= XY_STAGE_Y_MAX_UM):
                raise ValueError(
                    f"Y position {y} outside hardware limits "
                    f"[{XY_STAGE_Y_MIN_UM}, {XY_STAGE_Y_MAX_UM}]"
                )

            status = Status(obj=self, timeout=30)

            def wait():
                try:
                    # Set XY position using MM core
                    self.core.setXYPosition(x, y)
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
        """Read current XY stage positions - required for Bluesky"""
        xy_pos = np.array(self.core.getXYPosition())

        data = OrderedDict()
        data[self.name] = {
            'value': xy_pos,
            'timestamp': time.time(),
            'units': 'micrometers'
        }
        return data

    def describe(self):
        """Describe XY stage device - required for Bluesky"""
        data = OrderedDict()
        data[self.name] = {
            'source': self.name,
            'dtype': 'array',
            'shape': [2],
            'units': 'micrometers'
        }
        return data

    def read_configuration(self):
        """Required for Bluesky"""
        return OrderedDict()

    def describe_configuration(self):
        """Required for Bluesky"""
        return OrderedDict()

    # Synchronous convenience methods (usable outside RunEngine)
    def get_position(self) -> np.ndarray:
        """
        Get current XY stage position as numpy array.

        Returns
        -------
        np.ndarray
            Current position as [x, y] in micrometers

        Notes
        -----
        This is a synchronous convenience method that can be used outside
        the RunEngine for interactive use, setup, and debugging. For use
        within plans, prefer yield from bps.rd(xy_stage).
        """
        return self.read()[self.name]['value']

    def get_x(self) -> float:
        """
        Get current X stage position.

        Returns
        -------
        float
            X position in micrometers
        """
        return self.get_position()[0]

    def get_y(self) -> float:
        """
        Get current Y stage position.

        Returns
        -------
        float
            Y position in micrometers
        """
        return self.get_position()[1]

    # Coordinate conversion utilities for embryo centering
    @staticmethod
    def pixel_to_stage_offset(pixel_offset_x: float,
                               pixel_offset_y: float,
                               pixel_size_um: float) -> Tuple[float, float]:
        """
        Convert pixel offsets to stage movement in micrometers.

        IMPORTANT: X-axis is INVERTED - stage +X moves features LEFT in camera view.
        This is a hardware characteristic of the diSPIM coordinate system.

        Parameters
        ----------
        pixel_offset_x : float
            Horizontal pixel displacement (positive = right in image)
        pixel_offset_y : float
            Vertical pixel displacement (positive = down in image)
        pixel_size_um : float
            Effective pixel size in micrometers (physical pixel size / magnification)

        Returns
        -------
        Tuple[float, float]
            Stage movement required (dx_um, dy_um)

        Notes
        -----
        This method delegates to gently.coordinates for the actual calculation.
        """
        from gently.core.coordinates import pixel_displacement_to_stage_movement
        return pixel_displacement_to_stage_movement(pixel_offset_x, pixel_offset_y, pixel_size_um)
