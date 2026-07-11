"""
ACUITYnano Precision Thermal Controller as a Bluesky/Ophyd-protocol device.

Wraps the vendor SDK — a Peltier/TEC water-cooled controller, 0.0-99.9 C. Two
transports expose the same core API:
  - USB serial : acuitynano_precision_thermalizer_serial  (vendor-recommended
                 for closed-loop automation; zero-latency)
  - MQTT       : acuitynano_precision_thermalizer_api      (multi-client; adds
                 get_peltier_temp())

The device follows the same duck-typed Bluesky protocol as the diSPIM devices
(see gently.hardware.dispim.devices.optical.DiSPIMLED). A temperature controller
is the textbook bluesky "settable that completes on stabilization":

    yield from bps.mv(temperature, 20.0)   # blocks until the controller LOCKS

set(target) commands the setpoint, enables the TEC, and returns an ophyd Status
that finishes only when the controller reports "[ SYSTEM LOCKED ]" (or raises on
timeout). read() reports the live water temperature (plus setpoint / state, and
peltier temp when the transport provides it). BLE-style work runs in a worker
thread so the Status integrates with the RunEngine.

NOTE: the vendor `acuitynano_precision_thermalizer_*` packages are NOT on PyPI.
gently bundles the SERIAL transport under `gently.hardware.vendor`; the MQTT
transport is NOT bundled (it embeds broker credentials) so install it on the
device-layer machine to use `backend: mqtt`. A system-installed copy of either
name takes precedence over the bundled one. Both transports need the `device`
extra (`uv sync --extra device`) for pyserial / paho-mqtt. Local logic can be
exercised with the built-in mock backend:
`python gently/hardware/temperature.py --mock 20`.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import OrderedDict

logger = logging.getLogger(__name__)

TEMP_MIN_C = 0.0
TEMP_MAX_C = 99.9


def _load_vendor(module: str, cls: str):
    """Resolve a vendor SDK class, preferring a system-installed copy.

    The ACUITYnano packages aren't on PyPI. gently bundles the SERIAL transport
    under ``gently.hardware.vendor``; the MQTT transport is NOT bundled (it
    embeds broker credentials), so it must be installed on the device-layer
    machine. We try the top-level module name first (lets an officially-installed
    vendor build override the bundled one), then fall back to the vendored copy.
    """
    import importlib

    last_err: ImportError | None = None
    for name in (module, f"gently.hardware.vendor.{module}"):
        try:
            return getattr(importlib.import_module(name), cls)
        except ImportError as exc:
            last_err = exc
            continue
    raise ImportError(
        f"could not import {cls}: {module!r} is not installed and there is no "
        f"bundled gently.hardware.vendor copy ({last_err})"
    ) from last_err


def _make_backend(cfg: dict):
    """Construct the vendor SDK transport from a config mapping."""
    backend = str(cfg.get("backend", "serial")).lower()
    if backend == "mock":
        return _MockBackend()
    if backend == "serial":
        AcuityNanoPrecisionThermalizerSerial = _load_vendor(
            "acuitynano_precision_thermalizer_serial",
            "AcuityNanoPrecisionThermalizerSerial",
        )

        return AcuityNanoPrecisionThermalizerSerial(
            cfg["com_port"], baud_rate=cfg.get("baud_rate", 115200)
        )
    if backend == "mqtt":
        AcuityNanoPrecisionThermalizerAPI = _load_vendor(
            "acuitynano_precision_thermalizer_api",
            "AcuityNanoPrecisionThermalizerAPI",
        )

        # The vendor package ships with an embedded HiveMQ Cloud broker + creds,
        # so MQTT can run with no config. Pass only the keys actually provided,
        # to override those embedded defaults (and keep secrets in config, not code).
        kwargs = {k: cfg[k] for k in ("broker", "port", "user", "password") if k in cfg}
        return AcuityNanoPrecisionThermalizerAPI(**kwargs)
    raise ValueError(f"unknown temperature backend {backend!r} (use 'serial', 'mqtt', or 'mock')")


def create_temperature_controller(cfg: dict) -> TemperatureController:
    """Factory used by the device layer: build transport + wrap as a device."""
    backend = _make_backend(cfg)
    if "feedback_peltier" in cfg and hasattr(backend, "set_feedback_sensor"):
        backend.set_feedback_sensor(use_peltier=bool(cfg["feedback_peltier"]))
    return TemperatureController(
        backend,
        name=cfg.get("name", "temperature"),
        stabilize_timeout=cfg.get("stabilize_timeout", 600.0),
    )


class TemperatureController:
    """Bluesky-protocol device for the ACUITYnano thermal controller.

    Parameters
    ----------
    backend : object
        Vendor SDK instance exposing set_temperature / get_water_temp /
        get_system_state / enable_tec / wait_for_target.
    name : str
        Device name; the registry key and primary read() field.
    stabilize_timeout : float
        Seconds to wait for "[ SYSTEM LOCKED ]" before set() fails.
    """

    def __init__(self, backend, name: str = "temperature", *, stabilize_timeout: float = 600.0):
        self._dev = backend
        self.name = name
        self.stabilize_timeout = float(stabilize_timeout)
        self.parent = None  # required for Bluesky bps.mv()
        self._setpoint = None  # last commanded target
        self._lock = threading.Lock()

    # -- Bluesky settable protocol -------------------------------------------
    def set(self, target_c):
        """Command setpoint + enable TEC; Status finishes when the system locks."""
        from ophyd.status import Status

        target = float(target_c)
        if not (TEMP_MIN_C <= target <= TEMP_MAX_C):
            raise ValueError(f"target {target} C outside [{TEMP_MIN_C}, {TEMP_MAX_C}]")

        status = Status(obj=self, timeout=self.stabilize_timeout + 30)

        def worker():
            with self._lock:
                try:
                    self._dev.set_temperature(target)  # vendor also validates range
                    self._dev.enable_tec(True)
                    locked = self._dev.wait_for_target(timeout_seconds=self.stabilize_timeout)
                except Exception as exc:
                    logger.warning("temperature %s set(%.2f) failed: %s", self.name, target, exc)
                    status.set_exception(exc)
                    return
                self._setpoint = target
            if locked:
                logger.info("temperature %s locked at %.2f C", self.name, target)
                status.set_finished()
            else:
                status.set_exception(
                    TimeoutError(
                        f"{self.name} did not stabilize at {target} C"
                        f" within {self.stabilize_timeout}s"
                    )
                )

        threading.Thread(target=worker, name=f"{self.name}-set", daemon=True).start()
        return status

    # -- Explicit controls (outside the bps.mv() path) -----------------------
    def enable(self, on: bool = True):
        self._dev.enable_tec(bool(on))

    def setpoint(self, target_c):
        """Command the setpoint without blocking for stabilization."""
        self._dev.set_temperature(float(target_c))

    # -- Bluesky readable protocol -------------------------------------------
    def read(self):
        now = time.time()
        data = OrderedDict()
        data[self.name] = {
            "value": self._safe(self._dev.get_water_temp),
            "timestamp": now,
        }
        data[f"{self.name}_setpoint"] = {"value": self._setpoint, "timestamp": now}
        data[f"{self.name}_state"] = {
            "value": self._safe(self._dev.get_system_state, default="unknown"),
            "timestamp": now,
        }
        if self._has_peltier():
            data[f"{self.name}_peltier"] = {
                "value": self._safe(self._dev.get_peltier_temp),
                "timestamp": now,
            }
        return data

    def describe(self):
        src = f"acuitynano:{self.name}"
        d = OrderedDict()
        d[self.name] = {"source": src, "dtype": "number", "shape": []}
        d[f"{self.name}_setpoint"] = {"source": src, "dtype": "number", "shape": []}
        d[f"{self.name}_state"] = {"source": src, "dtype": "string", "shape": []}
        if self._has_peltier():
            d[f"{self.name}_peltier"] = {"source": src, "dtype": "number", "shape": []}
        return d

    def read_configuration(self):
        return OrderedDict()

    def describe_configuration(self):
        return OrderedDict()

    def close(self):
        """Release the transport (serial port / MQTT client) on shutdown."""
        for method in ("close", "disconnect"):
            fn = getattr(self._dev, method, None)
            if fn is not None:
                try:
                    fn()
                except Exception:
                    pass
                return

    # -- helpers --------------------------------------------------------------
    def _has_peltier(self) -> bool:
        return getattr(self._dev, "get_peltier_temp", None) is not None

    @staticmethod
    def _safe(fn, default=None):
        try:
            return fn()
        except Exception:
            return default


class _MockBackend:
    """In-memory fake mirroring the vendor API, for local testing without hardware."""

    def __init__(self, *args, **kwargs):
        self._target = 25.0
        self._enabled = False

    def set_temperature(self, t):
        if not (TEMP_MIN_C <= float(t) <= TEMP_MAX_C):
            raise ValueError("Target must be between 0.0 and 99.9 C")
        self._target = float(t)

    def enable_tec(self, on):
        self._enabled = bool(on)

    def set_feedback_sensor(self, use_peltier=False):
        pass

    def wait_for_target(self, timeout_seconds=300):
        time.sleep(0.5)  # pretend to ramp + settle
        return True

    def get_water_temp(self):
        return self._target

    def get_peltier_temp(self):
        return self._target - 1.0

    def get_system_state(self):
        return "[ SYSTEM LOCKED ]" if self._enabled else "[ IDLE ]"

    def close(self):
        pass


if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if "--mock" in sys.argv:
        target = 20.0
        for arg in sys.argv[1:]:
            try:
                target = float(arg)
                break
            except ValueError:
                continue
        dev = TemperatureController(_MockBackend(), name="temperature", stabilize_timeout=10)
        print(f"[mock] set({target}) — blocks until locked ...")
        st = dev.set(target)
        st.wait(15)
        print("[mock] read ->", {k: v["value"] for k, v in dev.read().items()})
        print("OK")
    else:
        print(
            "Real-hardware self-test needs the vendor SDK + a controller. "
            "Run with --mock to exercise the device logic locally."
        )
