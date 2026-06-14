"""
CalibrationPipeline base + CalibrationData container.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np


@dataclass
class CalibrationData:
    """Output of one calibration capture.

    ``payload`` is a free-form dict per-pipeline. The detector pipeline
    pulls expected keys out (e.g. ``dark``, ``flat`` for two-point;
    ``edge_bbox`` for edge ROI). Merged ``CalibrationData`` from multiple
    pipelines is what gets passed to detector context as ``calibration``.
    """

    pipeline_name: str
    captured_at: datetime = field(default_factory=datetime.now)
    source_embryo_ids: list[str] = field(default_factory=list)
    payload: dict[str, Any] = field(default_factory=dict)
    notes: str | None = None


class CalibrationPipeline(ABC):
    """Base class for "capture on reference, apply to subject" pipelines.

    Pipelines are pure-ish: ``capture`` takes a per-embryo dict of
    ``{embryo_id: volume_ndarray}`` plus a metadata dict and returns a
    CalibrationData. ``apply`` is an OPTIONAL preprocessing step the
    detector pipeline can call — most pipelines just deposit data the
    detector reads directly out of ``calibration.payload``.
    """

    name: str = "calibration"

    @abstractmethod
    def capture(
        self,
        source_volumes: dict[str, np.ndarray],
        context: dict[str, Any],
    ) -> CalibrationData:
        """Compute calibration data from one or more source embryo volumes."""
        ...

    def apply(
        self,
        data: CalibrationData,
        target_volume: np.ndarray,
    ) -> np.ndarray:
        """Default: no-op (pipelines that just publish data override this only if needed)."""
        return target_volume


def aggregate_calibrations(
    pipelines_data: list[CalibrationData],
) -> dict[str, Any]:
    """Merge multiple CalibrationData payloads into the single dict the
    detector context expects.

    Each pipeline contributes its own keys to the merged dict (dark, flat,
    edge_bbox, etc.). Aggregation across multiple source embryos for a
    single pipeline is the pipeline's responsibility (typically via
    pixel-wise median, see TwoPointCalibration).
    """
    merged: dict[str, Any] = {}
    for cal in pipelines_data:
        merged.update(cal.payload)
    return merged
