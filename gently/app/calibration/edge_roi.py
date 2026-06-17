"""
Edge ROI calibration — embryo-boundary bbox for detector cropping.

Captures the embryo's body bbox from each calibration embryo's projection
(via the existing SAM detector or a thresholding fallback) and aggregates
to a single representative bbox. Detectors apply this as a crop so the
Claude vision call sees only the embryo body, not surrounding eggshell /
gel artifacts.

The bbox is in projection-pixel coordinates (matching the geometry the
dopaminergic detector uses).
"""

from typing import Any

import numpy as np

from .base import CalibrationData, CalibrationPipeline


class EdgeRoiCalibration(CalibrationPipeline):
    """Embryo-boundary bbox calibration."""

    name = "edge_roi"

    def __init__(self, padding_px: int = 10):
        self.padding_px = padding_px

    def capture(
        self,
        source_volumes: dict[str, Any],
        context: dict[str, Any],
    ) -> CalibrationData:
        """``source_volumes`` here is expected to be a flat
        ``{embryo_id: volume_ndarray}`` dict (no dark/flat split).

        A pre-computed per-embryo bbox can also be supplied directly via
        ``context['embryo_bboxes']`` — useful when SAM has already run
        elsewhere and we just want to reuse its output.
        """
        precomputed = context.get("embryo_bboxes") if context else None
        bboxes: list[tuple[int, int, int, int]] = []

        if precomputed:
            for _eid, bb in precomputed.items():
                if bb is not None and len(bb) == 4:
                    bboxes.append((int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3])))
        else:
            for vol in source_volumes.values():
                bb = _bbox_from_volume(vol, padding=self.padding_px)
                if bb is not None:
                    bboxes.append(bb)

        if not bboxes:
            return CalibrationData(
                pipeline_name=self.name,
                source_embryo_ids=sorted((source_volumes or {}).keys()),
                payload={},
                notes="No bboxes derivable from source volumes.",
            )

        # Aggregate via mean center + median half-extents (robust to outliers).
        xs0 = np.array([b[0] for b in bboxes])
        ys0 = np.array([b[1] for b in bboxes])
        xs1 = np.array([b[2] for b in bboxes])
        ys1 = np.array([b[3] for b in bboxes])
        agg = (
            int(np.median(xs0)),
            int(np.median(ys0)),
            int(np.median(xs1)),
            int(np.median(ys1)),
        )

        return CalibrationData(
            pipeline_name=self.name,
            source_embryo_ids=sorted((source_volumes or {}).keys()),
            payload={"edge_bbox": agg},
            notes=f"{len(bboxes)} source bboxes; padding={self.padding_px}px",
        )


def _bbox_from_volume(vol, padding: int = 10) -> tuple[int, int, int, int] | None:
    """Cheap thresholding-based bbox fallback when SAM hasn't run.

    Max-projects, thresholds at 25th percentile + delta, returns the
    enclosing bbox. Good enough for centering the embryo; not a substitute
    for SAM for high-quality segmentation.
    """
    if vol is None:
        return None
    v = np.squeeze(vol)
    if v.ndim == 4:
        v = v[0]
    if v.ndim == 3:
        proj = np.max(v, axis=0)
    elif v.ndim == 2:
        proj = v
    else:
        return None

    if proj.size == 0 or np.std(proj) < 1.0:
        return None

    p25 = float(np.percentile(proj, 25))
    p99 = float(np.percentile(proj, 99))
    threshold = p25 + 0.3 * max(p99 - p25, 1.0)
    mask = proj > threshold
    if not mask.any():
        return None

    ys, xs = np.where(mask)
    h, w = proj.shape
    x0 = max(0, int(xs.min()) - padding)
    y0 = max(0, int(ys.min()) - padding)
    x1 = min(w, int(xs.max()) + padding)
    y1 = min(h, int(ys.max()) + padding)
    if x1 <= x0 or y1 <= y0:
        return None
    return (x0, y0, x1, y1)
