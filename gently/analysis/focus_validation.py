"""Offline validation of focus algorithms against passively-logged focus traces.

The Operate view logs ``(z, focus_score)`` samples while the operator *manually*
focuses (bottom-cam or SPIM head); the human's resting Z is ground-truth best
focus. This module replays those traces — captured passively, with no autonomous
motion — to answer two safety questions before any autofocus is trusted near the
objective:

1. **Agreement** — would the focus metric's argmax have matched the human's
   chosen Z, and by how much (µm error)?
2. **Peak quality** — is the focus curve peaked at an interior point with enough
   contrast to hill-climb at all? A flat or edge-rising curve means *no* metric
   is safe to drive Z, regardless of agreement on a lucky sweep.

Nothing here moves hardware; it only reads ``focus_traces.jsonl`` (written by the
device layer) and reports. See sub-project C of the bottom-cam operator surface.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class FocusSample:
    """One passively-logged focus reading."""

    z: float
    score: float
    t: float | None = None
    source: str | None = None
    sweep_id: str | None = None


@dataclass
class SweepResult:
    n: int
    human_z: float
    predicted_z: float
    error_um: float
    interior_peak: bool  # argmax strictly inside the sweep (not an endpoint)
    score_contrast: float  # (max-min)/max — low ⇒ unreliable peak


@dataclass
class ValidationReport:
    n_sweeps: int
    median_error_um: float
    p95_error_um: float
    within_margin_frac: float  # fraction with error <= margin_um
    interior_peak_frac: float  # fraction whose metric peak is interior
    sweeps: list[SweepResult] = field(default_factory=list)


def load_focus_traces(path: str | Path) -> list[FocusSample]:
    """Read a ``focus_traces.jsonl`` file into FocusSample rows.

    Each line is a JSON object with ``z`` and ``focus_score`` (or ``score``);
    ``t``, ``source`` and ``sweep_id`` are optional. Malformed lines are skipped.
    """
    samples: list[FocusSample] = []
    p = Path(path)
    if not p.exists():
        return samples
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
            z = row.get("z")
            score = row.get("focus_score", row.get("score"))
            if z is None or score is None:
                continue
            samples.append(
                FocusSample(
                    z=float(z),
                    score=float(score),
                    t=row.get("t"),
                    source=row.get("source"),
                    sweep_id=row.get("sweep_id"),
                )
            )
        except (ValueError, TypeError, json.JSONDecodeError):
            continue
    return samples


def segment_sweeps(samples: list[FocusSample]) -> list[list[FocusSample]]:
    """Group samples into sweeps by ``sweep_id`` when present, else one sweep."""
    if not samples:
        return []
    if any(s.sweep_id is not None for s in samples):
        groups: dict[str, list[FocusSample]] = {}
        for s in samples:
            groups.setdefault(s.sweep_id or "_", []).append(s)
        return list(groups.values())
    return [samples]


def evaluate_sweep(samples: list[FocusSample], human_z: float | None = None) -> SweepResult:
    """Score one sweep: metric argmax vs the human's chosen (resting) Z.

    ``human_z`` defaults to the last sample's Z — where the operator settled.
    """
    if not samples:
        raise ValueError("empty sweep")
    if human_z is None:
        human_z = samples[-1].z
    best = max(samples, key=lambda s: s.score)
    predicted_z = best.z
    scores = [s.score for s in samples]
    smax, smin = max(scores), min(scores)
    contrast = (smax - smin) / smax if smax > 0 else 0.0
    # interior peak: the argmax is not the first/last sample by Z order
    by_z = sorted(samples, key=lambda s: s.z)
    idx = by_z.index(best)
    interior = 0 < idx < len(by_z) - 1
    return SweepResult(
        n=len(samples),
        human_z=float(human_z),
        predicted_z=float(predicted_z),
        error_um=abs(float(predicted_z) - float(human_z)),
        interior_peak=interior,
        score_contrast=float(contrast),
    )


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    k = (len(s) - 1) * pct
    lo = int(k)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def validate(
    sweeps: list[list[FocusSample]],
    margin_um: float = 5.0,
    human_zs: list[float] | None = None,
) -> ValidationReport:
    """Aggregate per-sweep agreement into a validation report.

    ``margin_um`` is the acceptable error for the algorithm to be considered safe
    on a sweep; ``human_zs`` optionally overrides the per-sweep resting-Z default.
    """
    results: list[SweepResult] = []
    for i, sw in enumerate(sweeps):
        if not sw:
            continue
        hz = human_zs[i] if (human_zs is not None and i < len(human_zs)) else None
        results.append(evaluate_sweep(sw, hz))
    if not results:
        return ValidationReport(0, 0.0, 0.0, 0.0, 0.0, [])
    errors = [r.error_um for r in results]
    within = sum(1 for e in errors if e <= margin_um) / len(results)
    interior = sum(1 for r in results if r.interior_peak) / len(results)
    return ValidationReport(
        n_sweeps=len(results),
        median_error_um=_percentile(errors, 0.5),
        p95_error_um=_percentile(errors, 0.95),
        within_margin_frac=within,
        interior_peak_frac=interior,
        sweeps=results,
    )


def validate_file(path: str | Path, margin_um: float = 5.0) -> ValidationReport:
    """Convenience: load a focus_traces.jsonl and validate it end-to-end."""
    return validate(segment_sweeps(load_focus_traces(path)), margin_um=margin_um)
