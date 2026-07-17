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
from typing import Any

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
    findings: dict[str, Any] = field(default_factory=dict)
    confidence: float | None = None
    reasoning: str | None = None
    raw_response: str | dict | None = None
    timestamp: datetime = field(default_factory=datetime.now)
    elapsed_ms: float | None = None
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
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
        context: dict[str, Any],
    ) -> DetectorResult:
        """Observe the volume and return a structured result."""
        ...
