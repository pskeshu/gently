"""
Plugin contracts for organism, hardware, and microscope client modules.

These Protocols define what organism and hardware plugins must export,
and what microscope clients must implement. The harness layer programs
against these interfaces; plugins implement them.

Usage:
    from gently.harness.protocols import OrganismProtocol, HardwareProtocol
    from gently.harness.protocols import MicroscopeClientProtocol
"""

from typing import Protocol, runtime_checkable, Dict, List, Set, Tuple, Optional


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
    HARDWARE_DESCRIPTION: str     # Markdown text for LLM context
    CAPABILITIES: set             # Set of capability strings


@runtime_checkable
class MicroscopeClientProtocol(Protocol):
    """What every microscope client must support.

    This protocol defines the generic interface that the harness and
    generic tools program against. Hardware-specific operations (e.g.,
    piezo-galvo calibration for diSPIM, Z-stack for 2P) live on the
    concrete client class and are used by hardware-specific tools.

    The client is the boundary between the agent/tools and the device
    layer. Each hardware module provides its own client implementation
    via create_client().
    """

    # --- Connection ---

    @property
    def is_connected(self) -> bool:
        """Whether the client is connected to the device layer."""
        ...

    async def connect(self) -> bool:
        """Connect to the device layer. Returns True on success."""
        ...

    async def disconnect(self) -> None:
        """Disconnect from the device layer."""
        ...

    async def get_status(self) -> dict:
        """Get device layer status (devices, queue, etc.)."""
        ...

    # --- Stage (XY) ---

    async def move_to_position(self, x: float, y: float) -> dict:
        """Move XY stage to absolute position in µm."""
        ...

    async def get_stage_position(self) -> Tuple[float, float]:
        """Get current XY stage position in µm."""
        ...

    # --- Z-axis (generic — mechanism varies by hardware) ---

    async def get_z_position(self) -> float:
        """Get current Z position in µm."""
        ...

    # --- Acquisition ---

    async def acquire(self, **params) -> dict:
        """Acquire image data. Parameters are hardware-specific.

        Returns a dict with at least:
            'success': bool
            'volume' or 'image': numpy array (or file ref)
        """
        ...

    async def snap(self, **params) -> dict:
        """Acquire a single 2D image. Parameters are hardware-specific.

        Returns a dict with at least:
            'success': bool
            'image': numpy array (or file ref)
        """
        ...

    # --- Sample detection ---

    @property
    def has_sam(self) -> bool:
        """Whether SAM-based sample detection is available."""
        ...

    async def detect_samples(self, **kwargs) -> dict:
        """Detect samples in the current field of view.

        Returns a dict with at least:
            'success': bool
            'embryos' (or 'samples'): list of detected samples
        """
        ...
