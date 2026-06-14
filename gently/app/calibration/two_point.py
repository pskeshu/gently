"""
Two-point calibration — per-pixel dark + flat reference frames.

Computes:
- ``dark``: per-pixel offset / autofluorescence baseline (from volumes
  acquired with the laser off / at very low power).
- ``flat``: per-pixel response to uniform illumination (from peak-signal
  volumes on calibration embryos at the operating laser power).

Both are stored as max-projections (2D arrays) matching the projection
geometry the dopaminergic detector uses. Aggregation across multiple
calibration embryos uses pixel-wise **median** for robustness.
"""

from typing import Any

import numpy as np

from .base import CalibrationData, CalibrationPipeline


class TwoPointCalibration(CalibrationPipeline):
    """Dark + flat per-pixel calibration from CalibrationEmbryos."""

    name = "two_point"

    def __init__(self, dark_source: str = "dark", flat_source: str = "flat"):
        """``dark_source`` / ``flat_source`` are keys in the source_volumes
        dict that distinguish dark-frame volumes from flat-field volumes.

        Concrete usage:

            cal = TwoPointCalibration()
            data = cal.capture(
                source_volumes={
                    "dark": {"emb_c1": vol_dark_c1, "emb_c2": vol_dark_c2},
                    "flat": {"emb_c1": vol_flat_c1, "emb_c2": vol_flat_c2},
                },
                context={...},
            )
        """
        self.dark_source = dark_source
        self.flat_source = flat_source

    def capture(
        self,
        source_volumes: dict[str, Any],
        context: dict[str, Any],
    ) -> CalibrationData:
        darks = source_volumes.get(self.dark_source, {}) or {}
        flats = source_volumes.get(self.flat_source, {}) or {}

        dark_proj = _aggregate_projections(list(darks.values()))
        flat_proj = _aggregate_projections(list(flats.values()))

        payload: dict[str, Any] = {}
        if dark_proj is not None:
            payload["dark"] = dark_proj
        if flat_proj is not None:
            payload["flat"] = flat_proj
        if not payload:
            payload["notes"] = "No dark or flat volumes provided."

        return CalibrationData(
            pipeline_name=self.name,
            source_embryo_ids=sorted(set(list(darks.keys()) + list(flats.keys()))),
            payload=payload,
            notes=(
                f"dark={'yes' if dark_proj is not None else 'no'} "
                f"flat={'yes' if flat_proj is not None else 'no'}"
            ),
        )


def _aggregate_projections(volumes: list[np.ndarray]):
    """Max-project each volume to 2D, then median across embryos."""
    if not volumes:
        return None
    projs = []
    for vol in volumes:
        if vol is None:
            continue
        v = np.squeeze(vol)
        if v.ndim == 4:
            v = v[0]
        if v.ndim == 3:
            projs.append(np.max(v, axis=0))
        elif v.ndim == 2:
            projs.append(v)
    if not projs:
        return None
    # Median across embryos — robust to per-embryo outliers (dust, mis-mount).
    # If shapes differ, fall back to the median of the smallest common shape.
    shapes = {p.shape for p in projs}
    if len(shapes) > 1:
        h = min(p.shape[0] for p in projs)
        w = min(p.shape[1] for p in projs)
        projs = [p[:h, :w] for p in projs]
    return np.median(np.stack(projs, axis=0), axis=0).astype(np.float32)
