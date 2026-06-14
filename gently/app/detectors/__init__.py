"""
Detector framework — uniform interface for per-volume observation pipelines.

Detectors observe an acquired volume and emit a structured ``DetectorResult``
(stage classification, intensity level, hatching status, etc.). The
orchestrator routes per-embryo acquisitions to the detector(s) declared by
their ``EmbryoRole`` — Calibration embryos run the standard perception
proxy; Test embryos run the ad-hoc Claude detector.

Built-in detectors:
- ``DopaminergicSignalDetector`` — Claude vision call for the
  dopaminergic-reporter experiment (intensity_level + structure_quality +
  has_hatched).
- ``HatchingDetector`` — Claude vision call for hatching state.
- ``BlankImageDetector`` — refactored from ``AgentMicroscope.check_blank_image``.
- ``PerceptionProxy`` — wraps ``gently_perception.Perceiver`` so the
  existing nuclear-marker perception speaks the same interface.
"""

from .base import Detector, DetectorResult
from .blank_image import BlankImageDetector
from .dopaminergic_signal import DopaminergicSignalDetector
from .hatching import HatchingDetector
from .perception_proxy import PerceptionProxy
from .registry import DETECTOR_REGISTRY, get_detector

__all__ = [
    "Detector",
    "DetectorResult",
    "DopaminergicSignalDetector",
    "HatchingDetector",
    "BlankImageDetector",
    "PerceptionProxy",
    "DETECTOR_REGISTRY",
    "get_detector",
]
