"""
DiSPIM optical control devices (LED and laser).
"""

import logging
import time
from collections import OrderedDict

import pymmcore
from ophyd.status import Status

logger = logging.getLogger(__name__)


class DiSPIMLED:
    """
    DiSPIM LED control - works with bps.mv(led, state)

    ASI Tiger LED (LED:X:31) - LED shutter control via ConfigGroup
    Device-agnostic: any plan that sets device state will work
    """

    def __init__(self, core: pymmcore.CMMCore, name: str = "LED", group_name: str | None = None):
        self.core = core
        self.name = name
        self.group_name = group_name or name
        self.parent = None  # Required for Bluesky bps.mv()

        # Cache available configs (should be 'Open' and 'Closed')
        self._available_configs = self._get_available_configs()

    def _get_available_configs(self):
        """Get available LED configurations"""
        try:
            return list(self.core.getAvailableConfigs(self.group_name))
        except Exception:
            return []

    def set(self, state: str):
        """Set LED state - called by bps.mv(led, 'Open') or bps.mv(led, 'Closed')"""
        if state not in self._available_configs:
            raise ValueError(f"State '{state}' not available. Available: {self._available_configs}")

        status = Status(obj=self, timeout=5)

        def wait():
            try:
                self.core.setConfig(self.group_name, state)
                self.core.waitForConfig(self.group_name, state)
            except Exception as exc:
                status.set_exception(exc)
            else:
                status.set_finished()

        import threading

        threading.Thread(target=wait).start()

        return status

    def read(self):
        """Read current LED configuration - required for Bluesky"""
        try:
            current_config = self.core.getCurrentConfig(self.group_name)
        except Exception:
            current_config = "unknown"

        data = OrderedDict()
        data[self.name] = {"value": current_config, "timestamp": time.time()}
        return data

    def describe(self):
        """Describe LED device - required for Bluesky"""
        data = OrderedDict()
        data[self.name] = {"source": self.name, "dtype": "string", "shape": []}
        return data

    def read_configuration(self):
        """Required for Bluesky"""
        return OrderedDict()

    def describe_configuration(self):
        """Required for Bluesky"""
        return OrderedDict()


class DiSPIMLightSource:
    """
    DiSPIM light source — laser ConfigGroup selector + per-line power %.

    Two control surfaces:

    1. **Preset / channel selection** via ``set(config_name)`` (Bluesky-compatible
       ``bps.mv(light_source, 'config_name')`` — e.g. ``"488 and 561"``,
       ``"488 only"``, ``"ALL OFF"``). The PLogic gates which laser lines
       are routed to the SPIM trigger.

    2. **Per-line power %** via ``set_power_pct(wavelength, pct)``. Writes
       the analog setpoint directly on the Coherent Scientific Remote
       device. Hard-limited per wavelength via :attr:`POWER_LIMITS_PCT` —
       attempts outside the limit raise ``ValueError`` from inside the
       device-layer setter. This is the bottom line of safety; any
       orchestrator-level rule must stay within these bounds.

    Adjust ``POWER_LIMITS_PCT`` here (single source of truth) if a future
    experiment requires a different range. Bounds cannot be bypassed at
    runtime.
    """

    # Per-wavelength MM property labels on the Coherent Scientific Remote
    # device. Wavelengths not listed are not addressable via set_power_pct.
    POWER_DEVICE_LABEL = "Coherent-Scientific Remote"
    POWER_PROPERTY = {
        405: "Laser 405-100C - PowerSetpoint (%)",
        488: "Laser 488-100C - PowerSetpoint (%)",
        561: "Laser OBIS LS 561-100 - PowerSetpoint (%)",
        637: "Laser 637-140C - PowerSetpoint (%)",
    }

    # Hard safety limits (percent). Out-of-range writes raise ValueError.
    # Tighter than the laser's electrical limits — protects samples from
    # accidental overexposure regardless of caller. Tune here, not at
    # the call site.
    POWER_LIMITS_PCT = {
        405: (0.0, 100.0),
        488: (2.0, 6.0),
        561: (0.0, 100.0),
        637: (0.0, 100.0),
    }

    def __init__(self, core: pymmcore.CMMCore, name: str = "Laser", group_name: str | None = None):
        self.core = core
        self.name = name
        self.group_name = group_name or name
        self.parent = None  # Required for Bluesky bps.mv()

        # Cache available configs
        self._available_configs = self._get_available_configs()

    def _get_available_configs(self):
        """Get available laser configurations"""
        try:
            return list(self.core.getAvailableConfigs(self.group_name))
        except Exception:
            return []

    def set(self, config_name: str):
        """Set laser configuration - called by bps.mv(laser, 'config_name')"""
        if config_name not in self._available_configs:
            raise ValueError(
                f"Config '{config_name}' not available. Available: {self._available_configs}"
            )

        status = Status(obj=self, timeout=5)

        def wait():
            try:
                self.core.setConfig(self.group_name, config_name)
                self.core.waitForConfig(self.group_name, config_name)
            except Exception as exc:
                status.set_exception(exc)
            else:
                status.set_finished()

        import threading

        threading.Thread(target=wait).start()

        return status

    def set_power_pct(self, wavelength: int, pct: float) -> None:
        """
        Set per-line laser power in percent. Hard-limited by ``POWER_LIMITS_PCT``.

        Synchronous: writes the MM property directly. Safe to call from inside
        Bluesky plan ``configure()`` steps (no Bluesky messages required).

        Parameters
        ----------
        wavelength : int
            Laser wavelength (nm). Must be a key in :attr:`POWER_PROPERTY`.
        pct : float
            Setpoint in percent. Must be within :attr:`POWER_LIMITS_PCT`
            for this wavelength.

        Raises
        ------
        KeyError
            If ``wavelength`` is unknown.
        ValueError
            If ``pct`` is outside the hard safety range for the wavelength.
        """
        if wavelength not in self.POWER_PROPERTY:
            raise KeyError(
                f"Unknown laser wavelength {wavelength}nm. "
                f"Available: {sorted(self.POWER_PROPERTY.keys())}"
            )

        lo, hi = self.POWER_LIMITS_PCT.get(wavelength, (0.0, 100.0))
        if not (lo <= pct <= hi):
            raise ValueError(
                f"{wavelength}nm power {pct}% outside hard safety limit "
                f"[{lo}, {hi}]%. Adjust DiSPIMLightSource.POWER_LIMITS_PCT "
                f"to change this bound (in source — not at runtime)."
            )

        prop = self.POWER_PROPERTY[wavelength]
        self.core.setProperty(self.POWER_DEVICE_LABEL, prop, float(pct))
        # No waitForDevice — analog setpoint applies on the next exposure.
        logger.debug(
            "Set %dnm power to %.4f%% (%s / %s)",
            wavelength,
            pct,
            self.POWER_DEVICE_LABEL,
            prop,
        )

    def get_power_pct(self, wavelength: int) -> float:
        """Read the current laser power % for ``wavelength``."""
        if wavelength not in self.POWER_PROPERTY:
            raise KeyError(
                f"Unknown laser wavelength {wavelength}nm. "
                f"Available: {sorted(self.POWER_PROPERTY.keys())}"
            )
        return float(
            self.core.getProperty(self.POWER_DEVICE_LABEL, self.POWER_PROPERTY[wavelength])
        )

    def read(self):
        """Read current laser configuration - required for Bluesky"""
        try:
            current_config = self.core.getCurrentConfig(self.group_name)
        except Exception:
            current_config = "unknown"

        data = OrderedDict()
        data[self.name] = {"value": current_config, "timestamp": time.time()}
        return data

    def describe(self):
        """Describe laser control device - required for Bluesky"""
        data = OrderedDict()
        data[self.name] = {"source": self.group_name, "dtype": "string", "shape": []}
        return data

    def read_configuration(self):
        """Required for Bluesky"""
        return OrderedDict()

    def describe_configuration(self):
        """Required for Bluesky"""
        return OrderedDict()


# Backwards-compatible alias — existing imports keep working while the
# codebase migrates to the new name.
DiSPIMLaserControl = DiSPIMLightSource
