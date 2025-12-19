"""
Perception System for C. elegans Embryo Microscopy

Simple VLM-based perception: show reference examples, show current image, ask what stage.
"""

from .session import (
    Observation,
    PerceptionSession,
    PerceptionResult,
)
from .example_store import ExampleStore
from .engine import PerceptionEngine
from .manager import PerceptionManager, process_volume

__all__ = [
    "Observation",
    "PerceptionSession",
    "PerceptionResult",
    "ExampleStore",
    "PerceptionEngine",
    "PerceptionManager",
    "process_volume",
]
