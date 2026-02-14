"""
Organism module loader.

Provides load_organism() / get_organism() for accessing the active organism's
knowledge (stages, biology, detector presets, timing, etc.).

Usage:
    from gently.organisms import load_organism, get_organism

    load_organism("celegans")       # at startup
    org = get_organism()            # anywhere else
    print(org.ORGANISM_DISPLAY_NAME)  # "C. elegans"
"""

import importlib
import logging
from types import ModuleType
from typing import Optional

logger = logging.getLogger(__name__)

_active_organism: Optional[ModuleType] = None


def load_organism(name: str) -> ModuleType:
    """
    Load an organism module by name and set it as active.

    Parameters
    ----------
    name : str
        Organism module name (e.g., "celegans"). Must be a subpackage
        of gently.organisms with the expected exports.

    Returns
    -------
    ModuleType
        The loaded organism module.

    Raises
    ------
    ImportError
        If the organism module cannot be found.
    """
    global _active_organism
    module = importlib.import_module(f"gently.organisms.{name}")
    _active_organism = module
    logger.info(f"Loaded organism module: {name} ({module.ORGANISM_DISPLAY_NAME})")
    return module


def get_organism() -> ModuleType:
    """
    Return the active organism module.

    Raises
    ------
    RuntimeError
        If no organism has been loaded yet.
    """
    if _active_organism is None:
        raise RuntimeError(
            "No organism loaded. Call load_organism() at startup, "
            "or set 'organism' in config.yml."
        )
    return _active_organism
