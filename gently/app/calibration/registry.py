"""Calibration pipeline registry — name → factory."""

from typing import Callable, Dict, Optional

from .base import CalibrationPipeline


CalibrationFactory = Callable[..., CalibrationPipeline]


def _make_two_point(**kw) -> CalibrationPipeline:
    from .two_point import TwoPointCalibration
    return TwoPointCalibration(**kw)


def _make_edge_roi(**kw) -> CalibrationPipeline:
    from .edge_roi import EdgeRoiCalibration
    return EdgeRoiCalibration(**kw)


CALIBRATION_REGISTRY: Dict[str, CalibrationFactory] = {
    "two_point": _make_two_point,
    "edge_roi": _make_edge_roi,
}


def get_calibration_pipeline(name: str, **kw) -> Optional[CalibrationPipeline]:
    factory = CALIBRATION_REGISTRY.get(name)
    if factory is None:
        return None
    return factory(**kw)
