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
import pkgutil
from types import ModuleType

logger = logging.getLogger(__name__)

_active_organism: ModuleType | None = None


def available_organisms() -> list[str]:
    """Names of the organism plugins shipped under gently.organisms."""
    import gently.organisms as _pkg

    return sorted(
        m.name
        for m in pkgutil.iter_modules(_pkg.__path__)
        if m.ispkg and not m.name.startswith("_")
    )


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
    try:
        module = importlib.import_module(f"gently.organisms.{name}")
    except ModuleNotFoundError as e:
        # Only treat a missing organism *package* as a config error; if a
        # dependency *inside* the organism module is missing, re-raise so the
        # real ImportError isn't masked.
        if e.name in (f"gently.organisms.{name}", name):
            avail = ", ".join(available_organisms()) or "(none found)"
            raise ValueError(
                f"Unknown organism '{name}'. Available: {avail}. "
                f"Set 'organism:' in config/config.yml."
            ) from e
        raise
    _active_organism = module
    logger.info("Loaded organism module: %s (%s)", name, module.ORGANISM_DISPLAY_NAME)
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
            "No organism loaded. Call load_organism() at startup, or set 'organism' in config.yml."
        )
    return _active_organism
