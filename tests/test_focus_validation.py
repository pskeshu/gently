"""
Tests for analysis.focus_validation — offline replay of passively-logged
manual-focus traces (sub-project C).

Covers:
- evaluate_sweep finds the metric peak and measures error vs the human Z
- interior-peak + contrast flags
- a flat curve is flagged (no interior peak / zero contrast)
- validate aggregates error stats + within-margin fraction
- segment_sweeps groups by sweep_id
- load_focus_traces round-trips a jsonl file (and skips junk)
"""

import json
import math

from gently.analysis.focus_validation import (
    FocusSample,
    evaluate_sweep,
    load_focus_traces,
    segment_sweeps,
    validate,
)


def _gaussian_sweep(z0, lo=80, hi=160, step=2, amp=1000.0, sigma=12.0, base=50.0):
    out = []
    z = lo
    while z <= hi:
        score = base + amp * math.exp(-((z - z0) ** 2) / (2 * sigma**2))
        out.append(FocusSample(z=float(z), score=float(score)))
        z += step
    return out


def test_evaluate_sweep_finds_peak_near_human():
    sw = _gaussian_sweep(z0=120)
    res = evaluate_sweep(sw, human_z=121.0)
    assert abs(res.predicted_z - 120) <= 2  # argmax at/near the true peak
    assert res.error_um <= 3
    assert res.interior_peak is True
    assert res.score_contrast > 0.5


def test_flat_curve_flagged():
    sw = [FocusSample(z=float(z), score=100.0) for z in range(80, 161, 5)]
    res = evaluate_sweep(sw, human_z=120.0)
    assert res.score_contrast == 0.0
    # a flat curve's argmax is the first sample -> not an interior peak
    assert res.interior_peak is False


def test_validate_aggregates():
    # three sweeps; human settles within a couple µm of each metric peak
    sweeps = [_gaussian_sweep(z0=120), _gaussian_sweep(z0=100), _gaussian_sweep(z0=140)]
    human = [121.0, 99.0, 141.0]
    rep = validate(sweeps, margin_um=5.0, human_zs=human)
    assert rep.n_sweeps == 3
    assert rep.median_error_um <= 3
    assert rep.p95_error_um <= 5
    assert rep.within_margin_frac == 1.0
    assert rep.interior_peak_frac == 1.0


def test_validate_flags_out_of_margin():
    sweeps = [_gaussian_sweep(z0=120)]
    rep = validate(sweeps, margin_um=2.0, human_zs=[140.0])  # human far from peak
    assert rep.within_margin_frac == 0.0
    assert rep.median_error_um > 2.0


def test_segment_sweeps_by_id():
    samples = [
        FocusSample(z=1, score=1, sweep_id="a"),
        FocusSample(z=2, score=2, sweep_id="a"),
        FocusSample(z=3, score=3, sweep_id="b"),
    ]
    groups = segment_sweeps(samples)
    assert len(groups) == 2
    assert sorted(len(g) for g in groups) == [1, 2]


def test_segment_sweeps_default_single():
    samples = [FocusSample(z=1, score=1), FocusSample(z=2, score=2)]
    assert len(segment_sweeps(samples)) == 1


def test_load_focus_traces_roundtrip(tmp_path):
    p = tmp_path / "focus_traces.jsonl"
    lines = [
        {"z": 100.0, "focus_score": 0.4, "t": 1.0, "source": "bottom"},
        {"z": 110.0, "score": 0.9, "source": "bottom"},  # 'score' alias
        {"garbage": True},  # missing z/score -> skipped
        "not json",  # junk -> skipped
    ]
    with p.open("w") as f:
        for ln in lines:
            f.write((json.dumps(ln) if isinstance(ln, dict) else ln) + "\n")
    samples = load_focus_traces(p)
    assert len(samples) == 2
    assert samples[0].z == 100.0 and samples[0].score == 0.4
    assert samples[1].score == 0.9


def test_load_missing_file_returns_empty(tmp_path):
    assert load_focus_traces(tmp_path / "nope.jsonl") == []
