"""
Re-export shim for organism-specific stage definitions.

Stage definitions are provided by the active organism plugin.
This module re-exports them for backward compatibility with code that imports
from gently.agent.perception.stages or gently.harness.perception.stages.

The canonical definitions live in gently.organisms.<name>.stages.
"""


def __getattr__(name):
    """Lazy import from the active organism module."""
    from gently.organisms import get_organism
    org = get_organism()
    stages_module = __import__(
        f"gently.organisms.{org.ORGANISM_NAME}.stages",
        fromlist=[name],
    )
    return getattr(stages_module, name)
