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
from types import ModuleType
from typing import Optional

logger = logging.getLogger(__name__)

_active_hardware: Optional[ModuleType] = None


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
    module = importlib.import_module(f"gently.hardware.{name}")
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
            "No hardware loaded. Call load_hardware() at startup, "
            "or set 'hardware' in config.yml."
        )
    return _active_hardware
