"""
Plugin contracts for organism and hardware modules.

These Protocols define what organism and hardware plugins must export.
The harness layer programs against these interfaces; plugins implement them.

Usage:
    from gently.harness.protocols import OrganismProtocol, HardwareProtocol
"""

from typing import Protocol, runtime_checkable, Dict, List, Set


@runtime_checkable
class OrganismProtocol(Protocol):
    """What an organism plugin must export.

    Each organism module (e.g., gently.organisms.celegans) should export
    these module-level attributes. The harness layer accesses them via
    get_organism().ATTRIBUTE_NAME.
    """

    ORGANISM_NAME: str
    ORGANISM_DISPLAY_NAME: str
    SAMPLE_TERM: str              # "embryo", "cell", "organoid"
    SAMPLE_TERM_PLURAL: str
    STAGES: list
    TERMINAL_STAGES: set
    BIOLOGY_KNOWLEDGE: str        # Markdown text for LLM context
    PERCEPTION_SYSTEM_PROMPT: str


@runtime_checkable
class HardwareProtocol(Protocol):
    """What a hardware plugin must export.

    Each hardware module (e.g., gently.hardware.dispim) should export
    these module-level attributes.
    """

    HARDWARE_NAME: str
    HARDWARE_DISPLAY_NAME: str
    HARDWARE_DESCRIPTION: str     # Markdown text for LLM context
