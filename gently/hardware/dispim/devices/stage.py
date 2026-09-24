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
# XY-STAGE SAFETY ENVELOPE — factory defaults, absolute MMCore micrometres.
#
# Layer 0 of the motion-safety stack. Every XY move planned by any layer
# above (Bluesky plans, agent orchestrators, UI tools) is bounded by the
# stage's envelope, and the same numbers are pushed into the Tiger firmware
# so the joystick is bounded too.
#
# The four constants below are the DEFAULT envelope. The live one is set by
# the operator from the Map view's "Edit region" wizard (#107): drive the
# stage to each corner of the region you have verified is clear of the SPIM
# head, optics and holder, capture it, apply. That is the same procedure
# the constants were measured with; the wizard just does not require a
# code edit and a restart. The result persists in config/config.local.yml
# and is re-applied at boot.
# =========================================================================
#
# Default envelope is INSET from the operator-measured outer corners by
# ~840–860 µm. The inset absorbs the joystick's deceleration-overshoot
# (we measured ~13 µm at slow joystick, up to ~683 µm at fast). With this
# inset, even a fast-joystick overshoot still lands inside the true safe
# travel envelope the operator measured by hand. The wizard offers the same
# inset as a field.
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
        # Live envelope, µm. Seeded from the defaults; changed only through
        # set_firmware_limits(), which writes the controller first and
        # verifies the read-back, so software and firmware never disagree.
        self._x_limits: tuple[float, float] = (XY_STAGE_X_MIN_UM, XY_STAGE_X_MAX_UM)
        self._y_limits: tuple[float, float] = (XY_STAGE_Y_MIN_UM, XY_STAGE_Y_MAX_UM)

    @property
    def x_limits(self) -> tuple[float, float]:
        """The live X envelope in µm (see set_firmware_limits)."""
        return self._x_limits

    @property
    def y_limits(self) -> tuple[float, float]:
        """The live Y envelope in µm (see set_firmware_limits)."""
        return self._y_limits

    def set(self, position):
        """Move XY stage to position [x, y] - called by bps.mv(xy_stage, [x, y])"""
        try:
            x, y = position  # Unpack [x, y] coordinates
            x = float(x)
            y = float(y)

            # Software half of the envelope; the firmware half is the same
            # numbers, written by set_firmware_limits().
            x_lo, x_hi = self._x_limits
            y_lo, y_hi = self._y_limits
            if not (x_lo <= x <= x_hi):
                raise ValueError(f"X position {x} outside hardware limits [{x_lo}, {x_hi}]")
            if not (y_lo <= y <= y_hi):
                raise ValueError(f"Y position {y} outside hardware limits [{y_lo}, {y_hi}]")

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
        # Only now — after every write has been read back — does the software
        # envelope follow. A partial failure above leaves it where it was.
        self._x_limits = (x_min_mm * 1000.0, x_max_mm * 1000.0)
        self._y_limits = (y_min_mm * 1000.0, y_max_mm * 1000.0)

    def set_software_limits(
        self,
        x_min_um: float,
        x_max_um: float,
        y_min_um: float,
        y_max_um: float,
        *,
        require_inside: bool = True,
    ) -> None:
        """Bound what THIS process will command, without touching the controller.

        The two fences used to be one. `set_firmware_limits` wrote the Tiger
        and then set this envelope from the same numbers, so the software
        bound was always whatever the controller held.

        They have to come apart, because they protect against different
        things. The software bound catches every move Gently makes — it is
        checked in `set()` before anything reaches the hardware, costs
        nothing, and affects nobody else. The firmware bound exists only to
        stop a hand on the joystick, and it binds every other client of the
        controller, Micro-Manager included.

        On a rig driven by trained operators the second is a choice, not a
        default. So the working region lives here, always, and reaches the
        controller only when someone asks for it.

        Narrower than the firmware bound is fine and is the normal case.
        Wider is not: `set()` would command a move the controller then
        refuses, which reads as a mysterious hardware error rather than a
        limit. The caller is responsible for that ordering — see
        `initialize`, which writes the firmware bound first and this one
        after.
        """
        if x_min_um >= x_max_um or y_min_um >= y_max_um:
            raise ValueError(
                f"Degenerate software envelope: x [{x_min_um}, {x_max_um}], "
                f"y [{y_min_um}, {y_max_um}]"
            )
        # Inside what the stage can actually reach. The firmware write used to
        # catch this for free — the controller rejected or clamped the value
        # and the read-back check raised — so a software-only region needs its
        # own guard, or Gently would command moves the hardware refuses and
        # report them as mysterious errors rather than limits.
        if (
            x_min_um < XY_STAGE_X_MIN_UM
            or x_max_um > XY_STAGE_X_MAX_UM
            or y_min_um < XY_STAGE_Y_MIN_UM
            or y_max_um > XY_STAGE_Y_MAX_UM
        ):
            raise ValueError(
                f"Region x=[{x_min_um:.1f}, {x_max_um:.1f}] y=[{y_min_um:.1f}, "
                f"{y_max_um:.1f}] µm is outside the stage's travel "
                f"x=[{XY_STAGE_X_MIN_UM}, {XY_STAGE_X_MAX_UM}] "
                f"y=[{XY_STAGE_Y_MIN_UM}, {XY_STAGE_Y_MAX_UM}] µm."
            )
        # The same refusal the firmware path makes (#107), for the same
        # operator-facing reason: an envelope that excludes where the stage is
        # standing is almost always a mis-measurement. The firmware docstring
        # justifies it by undefined controller behaviour, which does not apply
        # here — but dropping the check on this path would quietly weaken a
        # rule operators already rely on.
        #
        # `require_inside=False` is for boot, where a stage parked outside a
        # saved region must not stop the device layer from starting. The
        # region still binds every move after that.
        if require_inside:
            POS_SLOP_UM = 1.0
            try:
                cur = self.read()[self.name]["value"]
                cur_x, cur_y = float(cur[0]), float(cur[1])
            except Exception as exc:
                raise HardwareError(
                    f"Could not read current XY to validate the software envelope: {exc}"
                ) from exc
            if not (
                x_min_um - POS_SLOP_UM <= cur_x <= x_max_um + POS_SLOP_UM
                and y_min_um - POS_SLOP_UM <= cur_y <= y_max_um + POS_SLOP_UM
            ):
                raise ValueError(
                    f"Current stage position ({cur_x:.2f}, {cur_y:.2f}) µm is outside "
                    f"the requested region x=[{x_min_um:.2f}, {x_max_um:.2f}] "
                    f"y=[{y_min_um:.2f}, {y_max_um:.2f}] µm — drive the stage into "
                    f"bounds before applying it."
                )
        self._x_limits = (float(x_min_um), float(x_max_um))
        self._y_limits = (float(y_min_um), float(y_max_um))
        logger.info(
            "Software envelope: x [%.1f, %.1f] y [%.1f, %.1f] um",
            x_min_um,
            x_max_um,
            y_min_um,
            y_max_um,
        )

    def joystick_enabled(self) -> bool:
        """Read the ASI 'JoystickEnabled' flag from the controller."""
        return str(self.core.getProperty(self.name, "JoystickEnabled")).strip() == "Yes"

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
