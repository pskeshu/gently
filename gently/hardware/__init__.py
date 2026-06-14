"""
Hardware module loader.

Provides load_hardware() / get_hardware() for accessing the active hardware
platform's knowledge (description, capabilities, safety limits, etc.).

Usage:
    from gently.hardware import load_hardware, get_hardware

    load_hardware("dispim")          # at startup
    hw = get_hardware()              # anywhere else
    print(hw.HARDWARE_DESCRIPTION)   # hardware prompt text
"""

import importlib
import logging
import pkgutil
from types import ModuleType

logger = logging.getLogger(__name__)

_active_hardware: ModuleType | None = None


def available_hardware() -> list[str]:
    """Names of the hardware plugins shipped under gently.hardware."""
    import gently.hardware as _pkg

    return sorted(
        m.name
        for m in pkgutil.iter_modules(_pkg.__path__)
        if m.ispkg and not m.name.startswith("_")
    )


def load_hardware(name: str) -> ModuleType:
    """
    Load a hardware module by name and set it as active.

    Parameters
    ----------
    name : str
        Hardware module name (e.g., "dispim"). Must be a subpackage
        of gently.hardware with the expected exports.

    Returns
    -------
    ModuleType
        The loaded hardware module.

    Raises
    ------
    ImportError
        If the hardware module cannot be found.
    """
    global _active_hardware
    try:
        module = importlib.import_module(f"gently.hardware.{name}")
    except ModuleNotFoundError as e:
        # Only a missing hardware *package* is a config error; re-raise if a
        # dependency inside the module is what's missing.
        if e.name in (f"gently.hardware.{name}", name):
            avail = ", ".join(available_hardware()) or "(none found)"
            raise ValueError(
                f"Unknown hardware '{name}'. Available: {avail}. "
                f"Set 'hardware:' in config/config.yml."
            ) from e
        raise
    _active_hardware = module
    logger.info("Loaded hardware module: %s", name)
    return module


def get_hardware() -> ModuleType:
    """
    Return the active hardware module.

    Raises
    ------
    RuntimeError
        If no hardware has been loaded yet.
    """
    if _active_hardware is None:
        raise RuntimeError(
            "No hardware loaded. Call load_hardware() at startup, or set 'hardware' in config.yml."
        )
    return _active_hardware
