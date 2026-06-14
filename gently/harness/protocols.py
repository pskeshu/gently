"""
Plugin contracts for organism, hardware, and microscope client modules.

These Protocols define what organism and hardware plugins must export,
and what microscope clients must implement. The harness layer programs
against these interfaces; plugins implement them.

Usage:
    from gently.harness.protocols import OrganismProtocol, HardwareProtocol
    from gently.harness.protocols import MicroscopeClientProtocol
"""

from typing import Protocol, runtime_checkable


@runtime_checkable
class OrganismProtocol(Protocol):
    """What an organism plugin must export.

    Each organism module (e.g., gently.organisms.celegans) should export
    these module-level attributes. The harness layer accesses them via
    get_organism().ATTRIBUTE_NAME.
    """

    ORGANISM_NAME: str
    ORGANISM_DISPLAY_NAME: str
    SAMPLE_TERM: str  # "embryo", "cell", "organoid"
    SAMPLE_TERM_PLURAL: str
    STAGES: list
    TERMINAL_STAGES: set
    BIOLOGY_KNOWLEDGE: str  # Markdown text for LLM context
    PERCEPTION_SYSTEM_PROMPT: str


@runtime_checkable
class HardwareProtocol(Protocol):
    """What a hardware plugin must export.

    Each hardware module (e.g., gently.hardware.dispim) should export
    these module-level attributes and factory functions.

    CAPABILITIES is a set of strings describing what the hardware supports.
    Standard capability names:
        "xy_stage"      — XY positioning
        "z_control"     — Z-axis focusing (piezo, motor, etc.)
        "volume"        — 3D volume acquisition
        "snap"          — 2D single-plane acquisition
        "z_stack"       — Sequential Z-stack acquisition
        "dual_view"     — Dual-view imaging (e.g., diSPIM)
        "autofocus"     — Hardware or software autofocus
        "detection"     — Sample detection (SAM or similar)
        "fluorescence"  — Fluorescence imaging
        "transmitted"   — Transmitted light imaging
    """

    HARDWARE_NAME: str
    HARDWARE_DISPLAY_NAME: str
    HARDWARE_DESCRIPTION: str  # Markdown text for LLM context
    CAPABILITIES: set  # Set of capability strings


# Backward-compat alias — the Microscope base class in harness/microscope.py
# replaces this Protocol. Import from there for new code.
from .microscope import Microscope as MicroscopeClientProtocol  # noqa: E402, F401
