"""A run must not start on an uncalibrated embryo.

Nothing checked. `POST /api/devices/timelapse/start` validated
`interval_seconds > 0` and that the ids existed, and started — and the
acquisition layer did not fail either, it invented the scan geometry:

    # Get calibration parameters (use defaults if not calibrated)
    galvo_amplitude = cal.get("galvo_amplitude", 0.5)
    ...
    slope           = cal.get("slope_um_per_deg", 100.0)

So a timelapse could run to completion on embryos that had never been
calibrated, using five literals, and report success. The output is
indistinguishable from real data, which is why a refusal is the kinder answer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from gently.harness import calibration_gate


@dataclass
class _Embryo:
    calibration: dict[str, Any] = field(default_factory=dict)


def _fit(slope: float = 42.0) -> dict[str, Any]:
    """The shape a successful fit writes (calibration_tools.py:891)."""
    return {"slope_um_per_deg": slope, "offset_um": 1.0, "r_squared": 0.85}


# ── is_calibrated ────────────────────────────────────────────────────────────


def test_a_real_fit_counts() -> None:
    assert calibration_gate.is_calibrated(_Embryo(_fit())) is True


def test_an_empty_calibration_does_not() -> None:
    assert calibration_gate.is_calibrated(_Embryo({})) is False


def test_scan_parameters_without_a_fit_do_not_count() -> None:
    """The exact trap: the literals the old code substituted are not a fit.

    An embryo carrying galvo/piezo values but no slope has never been
    calibrated — those are the invented defaults, not a measurement.
    """
    partial = {
        "galvo_amplitude": 0.5,
        "galvo_center": 0.0,
        "piezo_amplitude": 25.0,
        "piezo_center": 50.0,
    }
    assert calibration_gate.is_calibrated(_Embryo(partial)) is False


def test_a_zero_slope_is_not_a_calibration() -> None:
    """Zero slope means the scan has no extent — a fit that failed, not one."""
    assert calibration_gate.is_calibrated(_Embryo(_fit(0.0))) is False


def test_junk_in_the_slope_is_not_a_calibration() -> None:
    junk: list[Any] = [None, "", "n/a", float("nan"), float("inf"), [], {}]
    for bad in junk:
        assert calibration_gate.is_calibrated(_Embryo({"slope_um_per_deg": bad})) is False, bad


def test_a_missing_calibration_attribute_does_not_raise() -> None:
    class Bare:
        pass

    assert calibration_gate.is_calibrated(Bare()) is False


# ── uncalibrated() ───────────────────────────────────────────────────────────


def _roster() -> dict[str, _Embryo]:
    return {
        "e1": _Embryo(_fit()),
        "e2": _Embryo({}),
        "e3": _Embryo(_fit()),
        "e4": _Embryo({"galvo_center": 0.0}),
    }


def test_reports_only_the_uncalibrated_ones() -> None:
    assert calibration_gate.uncalibrated(_roster(), ["e1", "e2", "e3", "e4"]) == ["e2", "e4"]


def test_none_means_every_known_embryo() -> None:
    """`embryo_ids: null` on the route means "all active embryos"."""
    assert calibration_gate.uncalibrated(_roster(), None) == ["e2", "e4"]


def test_a_fully_calibrated_subset_is_allowed_through() -> None:
    assert calibration_gate.uncalibrated(_roster(), ["e1", "e3"]) == []


def test_unknown_ids_are_not_reported_as_uncalibrated() -> None:
    """ "Does not exist" and "not calibrated" send an operator to different fixes.

    The routes already reject unknown ids; conflating the two here would tell
    someone to calibrate an embryo that is not there.
    """
    assert calibration_gate.uncalibrated(_roster(), ["e1", "ghost"]) == []


def test_an_empty_roster_reports_nothing() -> None:
    assert calibration_gate.uncalibrated({}, None) == []
    assert calibration_gate.uncalibrated({}, ["e1"]) == []


# ── the message ──────────────────────────────────────────────────────────────


def test_the_refusal_names_the_embryos_and_the_way_past_it() -> None:
    msg = calibration_gate.refusal_detail(["e2", "e4"])
    assert "e2" in msg and "e4" in msg
    # An operator at a microscope needs the remedy in the message, not in a
    # dependency file or a route signature.
    assert "allow_uncalibrated" in msg
    assert "invented" in msg or "guess" in msg
