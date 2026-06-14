"""
PerceptionProxy — adapt the external ``gently_perception.Perceiver`` to
the Detector interface so role-based dispatch can route Calibration
embryos through the standard nuclear-marker pipeline uniformly.

The Perceiver maintains its own per-embryo session state (stability,
observations, current_stage). We delegate to it and translate its result
into a ``DetectorResult`` with stage-related findings.
"""

import logging
import time
from typing import Any

import numpy as np

from .base import Detector, DetectorResult
from .dopaminergic_signal import _volume_to_b64

logger = logging.getLogger(__name__)


class PerceptionProxy(Detector):
    """Thin Detector wrapper around ``gently_perception.Perceiver``."""

    name = "perception"

    def __init__(self, perceiver=None):
        self._perceiver = perceiver

    @property
    def perceiver(self):
        return self._perceiver

    async def run(
        self,
        volume: np.ndarray,
        context: dict[str, Any],
    ) -> DetectorResult:
        embryo_id = context.get("embryo_id", "?")
        timepoint = int(context.get("timepoint", 0))
        start = time.time()

        perceiver = self._perceiver or context.get("perceiver")
        if perceiver is None:
            return DetectorResult(
                detector_name=self.name,
                embryo_id=embryo_id,
                timepoint=timepoint,
                error="No Perceiver instance available",
                elapsed_ms=(time.time() - start) * 1000,
            )

        # Perceiver expects a base64 image of a 2D view (typically View A
        # of a multi-view volume). Reuse the same projection logic the
        # Claude detectors use so the upstream input is consistent.
        b64_image = _volume_to_b64(volume, context.get("calibration"))
        if b64_image is None:
            return DetectorResult(
                detector_name=self.name,
                embryo_id=embryo_id,
                timepoint=timepoint,
                error="Empty / unreadable volume",
                elapsed_ms=(time.time() - start) * 1000,
            )

        try:
            from datetime import datetime

            result = await perceiver(
                embryo_id,
                timepoint,
                b64_image,
                datetime.now().isoformat(),
            )
        except Exception as e:
            logger.exception("[%s] perceiver error for %s", self.name, embryo_id)
            return DetectorResult(
                detector_name=self.name,
                embryo_id=embryo_id,
                timepoint=timepoint,
                error=str(e),
                elapsed_ms=(time.time() - start) * 1000,
            )

        findings = {
            "stage": getattr(result, "stage", None),
        }
        # Pull extra fields if the Perceiver result carries them.
        for key in ("is_transitional", "transition_between", "observed_features"):
            v = getattr(result, key, None)
            if v is not None:
                findings[key] = v

        return DetectorResult(
            detector_name=self.name,
            embryo_id=embryo_id,
            timepoint=timepoint,
            findings=findings,
            confidence=getattr(result, "confidence", None),
            reasoning=getattr(result, "reasoning", None),
            raw_response=getattr(result, "raw_response", None),
            elapsed_ms=(time.time() - start) * 1000,
        )
