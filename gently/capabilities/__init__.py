"""
Capabilities — What the agent can do.

Capabilities wrap existing systems to provide a clean interface for the agent:
- HardwareCapability: Move stage, acquire images, configure microscope
- PerceptionCapability: Classify stages, detect features
- InteractionCapability: Speak, ask, notify
- IngestionCapability: Ingest papers, protocols, URLs into context

These are the agent's "hands" — how it acts in the world.
"""

from .hardware import HardwareCapability
from .perception import PerceptionCapability
from .interaction import InteractionCapability
from .ingestion import IngestionCapability
from .capabilities import Capabilities

__all__ = [
    "HardwareCapability",
    "PerceptionCapability",
    "InteractionCapability",
    "IngestionCapability",
    "Capabilities",
]
