"""
DiSPIM stage positioner devices (Z-stage and XY-stage).
"""

import logging
import time
from collections import OrderedDict

import numpy as np
import pymmcore
from ophyd.status import Status

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
#
# Current envelope is INSET from the operator-measured outer corners by
# ~840–860 µm. The inset absorbs the joystick's deceleration-overshoot
# (we measured ~13 µm at slow joystick, up to ~683 µm at fast). With this
# inset, even a fast-joystick overshoot still lands inside the true safe
# travel envelope the operator measured by hand.
XY_STAGE_X_MIN_UM: float = -2252.1
XY_STAGE_X_MAX_UM: float = 983.0
XY_STAGE_Y_MIN_UM: float = -1677.0
XY_STAGE_Y_MAX_UM: float = 586.6


class DiSPIMZstage:
    """
    DiSPIM Z Stage positioner - works with bps.mv(z_stage, position)

    Device-agnostic: any plan that moves a positioner will work with this device
    """

    def __init__(
        self,
        name: str,
        core: pymmcore.CMMCore,
        limits: tuple[float, float] = (50.0, 250.0),
    ):
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
            "value": float(value),
            "timestamp": time.time(),
            "units": "micrometers",
        }
        return data

    def describe(self):
        """Describe Z stage device - required for Bluesky"""
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
    def x_limits(self) -> tuple[float, float]:
        """Read-only view of the hardware safety limits (module constants)."""
        return (XY_STAGE_X_MIN_UM, XY_STAGE_X_MAX_UM)

    @property
    def y_limits(self) -> tuple[float, float]:
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
            "value": xy_pos,
            "timestamp": time.time(),
            "units": "micrometers",
        }
        return data

    def describe(self):
        """Describe XY stage device - required for Bluesky"""
        data = OrderedDict()
        data[self.name] = {
            "source": self.name,
            "dtype": "array",
            "shape": [2],
            "units": "micrometers",
        }
        return data

    def read_configuration(self):
        """Required for Bluesky"""
        return OrderedDict()

    def describe_configuration(self):
        """Required for Bluesky"""
        return OrderedDict()

    # ASI Tiger firmware soft-limit names. The controller enforces these for
    # ALL motion sources — joystick, MMCore, scripting — so writing them is
    # the right way to plug the joystick-bypass hole. Property values are in
    # millimetres on the ASI adapter; we feed it from the *_UM constants by
    # dividing by 1000.
    _ASI_LIMIT_PROPS = {
        "x_min": "LowerLimX(mm)",
        "x_max": "UpperLimX(mm)",
        "y_min": "LowerLimY(mm)",
        "y_max": "UpperLimY(mm)",
    }

    def set_firmware_limits(
        self,
        x_min_mm: float,
        x_max_mm: float,
        y_min_mm: float,
        y_max_mm: float,
        *,
        readback_tolerance_mm: float = 0.001,
    ) -> None:
        """Push XY safety bounds down to the ASI Tiger controller firmware.

        The Tiger firmware enforces these against every motion source —
        joystick included — so this closes the bypass where the joystick
        could otherwise drive the stage past Layer-1 software limits.

        Refuses to write if the current position is outside the requested
        envelope (controller behaviour is undefined when limits exclude the
        live position). Operator should drive into bounds first.

        Read-back is verified after every write — if the controller silently
        clamped or rejected a value (e.g. unit mismatch, advanced-properties
        gate, firmware quirk), this raises HardwareError so the device layer
        refuses to start in an unsafe state.

        Parameters
        ----------
        x_min_mm, x_max_mm, y_min_mm, y_max_mm : float
            Soft-limit values in millimetres. The XY_STAGE_*_UM constants in
            this module are the source of truth — pass them divided by 1000.
        readback_tolerance_mm : float
            How close the read-back value must be to the written value to
            count as accepted. Default 1 µm — well below any meaningful
            envelope precision.

        Raises
        ------
        ValueError
            If current XY is outside the requested envelope.
        HardwareError
            If a write didn't take or the read-back differs by more than
            ``readback_tolerance_mm``.
        """
        # 1. Sanity-check the requested values against each other.
        if x_min_mm >= x_max_mm or y_min_mm >= y_max_mm:
            raise ValueError(
                f"Degenerate firmware limit envelope: "
                f"x=[{x_min_mm}, {x_max_mm}] y=[{y_min_mm}, {y_max_mm}]"
            )

        # 2. Refuse if the stage is currently outside the new envelope.
        # Allow a small encoder-noise tolerance — sub-µm differences between
        # the operator's recorded corner and the live encoder reading
        # shouldn't block startup. The slop is far below the deceleration
        # overshoot we're trying to absorb anyway.
        POS_SLOP_MM = 0.001  # 1 µm
        try:
            cur = self.read()[self.name]["value"]
            cur_x_mm = float(cur[0]) / 1000.0
            cur_y_mm = float(cur[1]) / 1000.0
        except Exception as exc:
            raise HardwareError(f"Could not read current XY to validate limits: {exc}") from exc
        if not (
            x_min_mm - POS_SLOP_MM <= cur_x_mm <= x_max_mm + POS_SLOP_MM
            and y_min_mm - POS_SLOP_MM <= cur_y_mm <= y_max_mm + POS_SLOP_MM
        ):
            raise ValueError(
                f"Current stage position ({cur_x_mm * 1000:.2f}, {cur_y_mm * 1000:.2f}) µm "
                f"is outside the requested firmware envelope "
                f"x=[{x_min_mm * 1000:.2f}, {x_max_mm * 1000:.2f}] µm "
                f"y=[{y_min_mm * 1000:.2f}, {y_max_mm * 1000:.2f}] µm — "
                f"drive the stage into bounds before applying firmware limits."
            )

        # 3. Write each limit, read it back, and verify.
        targets = [
            (self._ASI_LIMIT_PROPS["x_min"], x_min_mm),
            (self._ASI_LIMIT_PROPS["x_max"], x_max_mm),
            (self._ASI_LIMIT_PROPS["y_min"], y_min_mm),
            (self._ASI_LIMIT_PROPS["y_max"], y_max_mm),
        ]
        for prop, value_mm in targets:
            try:
                self.core.setProperty(self.name, prop, float(value_mm))
            except RuntimeError as exc:
                raise HardwareError(
                    f"setProperty {prop}={value_mm} failed: {exc}. The ASI adapter "
                    f"may require EnableAdvancedProperties=Yes for this write."
                ) from exc
            try:
                got = float(self.core.getProperty(self.name, prop))
            except RuntimeError as exc:
                raise HardwareError(f"getProperty {prop} read-back failed: {exc}") from exc
            if abs(got - value_mm) > readback_tolerance_mm:
                raise HardwareError(
                    f"Firmware limit read-back mismatch for {prop}: "
                    f"wrote {value_mm} mm, controller reports {got} mm "
                    f"(tolerance {readback_tolerance_mm} mm). "
                    f"The controller may have rejected or rescaled the value."
                )
            logger.info("ASI firmware limit %s = %.4f mm (verified)", prop, got)

    def enable_joystick(self, enabled: bool = True) -> None:
        """Set the ASI Tiger 'JoystickEnabled' property on the XY stage.

        Tiger firmware persists this flag in its non-volatile card settings
        (touched whenever someone calls SaveCardSettings — we don't, but
        previous sessions may have). If it persisted as 'No', the physical
        joystick is dead on boot until something writes 'Yes'. This method
        is the boot-time fix; it's called from device_layer.initialize right
        after the firmware soft limits are applied.

        Read-back verified so a silent rejection by the adapter doesn't
        leave the operator wondering why the controller still doesn't move.
        """
        target = "Yes" if enabled else "No"
        prop = "JoystickEnabled"
        try:
            self.core.setProperty(self.name, prop, target)
        except RuntimeError as exc:
            raise HardwareError(
                f"setProperty {prop}={target} failed on {self.name}: {exc}"
            ) from exc
        try:
            got = self.core.getProperty(self.name, prop)
        except RuntimeError as exc:
            raise HardwareError(
                f"getProperty {prop} read-back failed on {self.name}: {exc}"
            ) from exc
        if str(got).strip() != target:
            raise HardwareError(
                f"{prop} read-back mismatch on {self.name}: "
                f"wrote '{target}', controller reports '{got}'."
            )
        logger.info("ASI %s.%s = %s (verified)", self.name, prop, got)

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
        return self.read()[self.name]["value"]

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
    def pixel_to_stage_offset(
        pixel_offset_x: float, pixel_offset_y: float, pixel_size_um: float
    ) -> tuple[float, float]:
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
