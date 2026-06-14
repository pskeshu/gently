"""
Calibration pipelines — "capture on a reference population, apply to subjects".

Pipelines run on CalibrationEmbryos at session start (or on demand) and
produce a ``CalibrationData`` blob that's passed to detectors as
preprocessing context. The dopaminergic detector for TestEmbryos picks
the dark/flat/edge_bbox out of this blob to clean its input before
sending to Claude.

Built-in pipelines:
- ``TwoPointCalibration`` — per-pixel dark + flat reference from
  CalibrationEmbryos. Background subtraction + flat-field correction.
- ``EdgeRoiCalibration`` — embryo boundary bbox from SAM detections.
  Crops the projection to the embryo body before downstream detection.

Aggregation across multiple CalibrationEmbryos uses the **median**
(robust to a single bad embryo). Single-embryo fallback if only one
Calibration is present.
"""

from .base import CalibrationData, CalibrationPipeline, aggregate_calibrations
from .edge_roi import EdgeRoiCalibration
from .registry import CALIBRATION_REGISTRY, get_calibration_pipeline
from .two_point import TwoPointCalibration

__all__ = [
    "CalibrationData",
    "CalibrationPipeline",
    "TwoPointCalibration",
    "EdgeRoiCalibration",
    "CALIBRATION_REGISTRY",
    "get_calibration_pipeline",
    "aggregate_calibrations",
]
