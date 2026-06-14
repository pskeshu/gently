"""
DiSPIMSystem — synthetic ophyd-style device that owns MMCore system-level
operations (configuration loading, system-wide property cache, event callback
bridge, device-type lookups).

It isn't strictly a Bluesky ophyd Device — it has no per-device positioner
semantics — but it follows the same convention as :mod:`stage`, :mod:`camera`,
etc.: a Python class that *holds* the ``pymmcore.CMMCore`` handle and exposes
methods for every MMCore call it needs to make. Together with the per-device
ophyd classes, this means nothing outside the ``devices/`` package needs
direct ``self.core.*`` access.

Lifecycle
---------
Construct once at boot. ``DiSPIMSystem.__init__`` builds the underlying
``CMMCore`` handle so even that call lives inside the device-layer
abstraction. The per-device ophyd classes are then constructed using
``system.core`` — they keep their own ``self.core`` reference for their own
MMCore traffic, which is fine: they ARE the ophyd boundary for their devices.

Lifecycle ordering (must happen before per-device ophyds exist):
    1. DiSPIMSystem()
    2. system.enable_stderr_log(True)
    3. system.set_device_adapter_search_paths([mm_dir])
    4. system.load_system_configuration(mm_cfg_path)
    5. system.register_callback(bridge)  (optional)
    6. create per-device ophyd objects with system.core
    7. (anywhere) system.update_system_state_cache(), system.get_device_type(d), …
"""

from __future__ import annotations

import logging
from typing import Any

import pymmcore

logger = logging.getLogger(__name__)


class DiSPIMSystem:
    """MMCore-system facade. The only non-per-device class allowed direct
    ``self.core.*`` access; everything else routes through here.
    """

    def __init__(self) -> None:
        self.name = "mmcore_system"
        self.parent = None  # mirrors the ophyd-style convention
        self.core: pymmcore.CMMCore = pymmcore.CMMCore()

    # ----- boot-time configuration ---------------------------------------

    def enable_stderr_log(self, enabled: bool) -> None:
        self.core.enableStderrLog(bool(enabled))

    def set_device_adapter_search_paths(self, paths: list[str]) -> None:
        self.core.setDeviceAdapterSearchPaths(list(paths))

    def load_system_configuration(self, path: str) -> None:
        self.core.loadSystemConfiguration(str(path))

    def get_loaded_devices(self) -> list[str]:
        return list(self.core.getLoadedDevices())

    # ----- system-wide property cache ------------------------------------

    def update_system_state_cache(self) -> None:
        """Re-poll every loaded device into MMCore's internal state cache."""
        self.core.updateSystemStateCache()

    def get_system_state_cache(self) -> Any:
        """Return the cached system Configuration. Use after
        :meth:`update_system_state_cache` to read property dumps without
        further hardware traffic."""
        return self.core.getSystemStateCache()

    def get_device_type(self, device_name: str) -> int:
        """MMCore DeviceType enum value for a device by name."""
        return int(self.core.getDeviceType(device_name))

    # ----- MMCore event bridge ------------------------------------------

    def register_callback(self, callback: pymmcore.MMEventCallback) -> None:
        """Wire an MMEventCallback into MMCore. Callbacks fire on the
        MMCore worker thread; the caller is responsible for marshalling
        onto an asyncio loop if needed."""
        self.core.registerCallback(callback)
