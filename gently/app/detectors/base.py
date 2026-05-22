"""
Detector base class + DetectorResult dataclass.

Detectors are stateless observation pipelines: given a volume and a
context dict, return a structured result. Concrete implementations may be
Claude-vision-based (DopaminergicSignalDetector, HatchingDetector,
BlankImageDetector) or thin wrappers around the existing
``gently_perception.Perceiver`` (PerceptionProxy).
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np


@dataclass
class DetectorResult:
    """Structured output from a Detector.run() call.

    Fields chosen for parallel persistence into ``predictions.jsonl`` and
    per-timepoint trace JSON files. ``findings`` is a free-form dict of
    detector-specific observations (e.g. ``{intensity_level: "STRONG",
    structure_quality: "GOOD"}`` for the dopaminergic detector,
    ``{stage: "pretzel"}`` for the perception proxy).
    """
    detector_name: str
    embryo_id: str
    timepoint: int
    findings: Dict[str, Any] = field(default_factory=dict)
    confidence: Optional[float] = None
    reasoning: Optional[str] = None
    raw_response: Optional[str] = None
    timestamp: datetime = field(default_factory=datetime.now)
    elapsed_ms: Optional[float] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "detector_name": self.detector_name,
            "embryo_id": self.embryo_id,
            "timepoint": self.timepoint,
            "findings": self.findings,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "raw_response": self.raw_response,
            "timestamp": self.timestamp.isoformat(),
            "elapsed_ms": self.elapsed_ms,
            "error": self.error,
        }


class Detector(ABC):
    """Base class for per-volume observation detectors.

    Subclasses implement ``async run(volume, context)``. ``context`` is a
    dict carrying things the detector might need:

    - ``embryo_id``: str
    - ``timepoint``: int
    - ``claude``: an Anthropic SDK client (for vision-based detectors)
    - ``calibration``: optional dict with dark/flat/edge_roi (Phase 6)
    - ``prior_results``: list of recent DetectorResults for temporal context

    Detectors should be defensive: API failures, malformed JSON, etc.
    should produce a DetectorResult with an ``error`` field rather than
    raising — the orchestrator should keep running on partial information.
    """

    #: Stable identifier for serialization / persistence.
    name: str = "detector"

    @abstractmethod
    async def run(
        self,
        volume: np.ndarray,
        context: Dict[str, Any],
    ) -> DetectorResult:
        """Observe the volume and return a structured result."""
        ...
