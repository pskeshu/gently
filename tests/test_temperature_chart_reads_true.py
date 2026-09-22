"""The water chart must not lie about a steady bath, or hide a drifting one.

Two defects were baked into the old render and neither would fail a test that
only asked "did it draw something":

1. The y-axis fitted min/max of the data, so ±0.05 °C of sensor jitter filled
   the plot. A bath holding 20.0 °C all night looked like a rollercoaster, and
   an operator who learned to ignore that shape would ignore a real excursion
   drawn identically.
2. Points were spaced by index, not time. A gap in sampling — a device-layer
   restart, a stalled controller — compressed into a normal-looking step, so
   the one picture that could show "we stopped hearing from the bath" didn't.

CI runs no JavaScript, so these are pinned in source.
"""

from __future__ import annotations

import re
from pathlib import Path

CHART = (
    Path(__file__).resolve().parents[1]
    / "gently"
    / "ui"
    / "web"
    / "static"
    / "js"
    / "temperature-graph.js"
)


def _src() -> str:
    return CHART.read_text(encoding="utf-8")


def test_the_scale_never_magnifies_noise() -> None:
    """A minimum window, centred on the setpoint, so flat reads as flat."""
    src = _src()
    assert re.search(r"MIN_SPAN_C\s*=\s*[\d.]+", src), "the chart lost its minimum y-window"
    assert "Math.max(MIN_SPAN_C / 2" in src, (
        "the y-window no longer enforces MIN_SPAN_C — sensor jitter will fill the plot again"
    )
    assert re.search(r"centre\s*=\s*sp\s*!=\s*null\s*\?\s*sp", src), (
        "the window is no longer centred on the setpoint — high and low stop reading symmetrically"
    )


def test_x_is_time_not_sample_index() -> None:
    """A gap in sampling must look like a gap."""
    src = _src()
    assert re.search(r"const sx\s*=\s*t\s*=>.*\(t - t0\)\s*/\s*tSpan", src), (
        "x position no longer comes from the timestamp — sampling gaps will compress silently"
    )


def test_drift_is_stated_in_words_not_only_in_colour() -> None:
    """Status colour always rides beside text that says how far off."""
    src = _src()
    assert "off ${sp.toFixed(1)}" in src, (
        "the heading no longer says how far off setpoint the bath is"
    )
    assert "is-drifting" in src, "the drift state is no longer marked on the marks"
    assert "vs set" in src, "the tooltip no longer reports the delta against the setpoint"


def test_the_reference_label_cannot_collide_with_the_end_label() -> None:
    """Both say the same number whenever the bath is at setpoint.

    The reference label is anchored to the LEFT end of its line for exactly
    that reason; putting it back at the right edge stacks two identical
    labels on top of each other in the most common state.
    """
    src = _src()
    assert re.search(r'el\("text",\s*\{\s*x:\s*pad\.left \+ 4', src), (
        "the setpoint label moved off the left end of its line — it will collide "
        "with the end-of-series label whenever the water is at setpoint"
    )


def test_hover_and_keyboard_read_the_same_values() -> None:
    """A crosshair a mouse can find, and arrow keys that walk it."""
    src = _src()
    for key in ("ArrowLeft", "ArrowRight", "Home", "End"):
        assert key in src, f"the chart's crosshair lost {key}"
    assert 'tabindex: "0"' in src, "the plot is no longer focusable"
    assert "pointermove" in src, "the crosshair lost its pointer tracking"
