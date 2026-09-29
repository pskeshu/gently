"""A calibration plot's text can be read, where it is shown.

"the matplotlib graph in calibration tab - the text in the graph are not very
readable."

The figures were drawn 600 px wide with 9 to 12 pt type and shown 126 px
wide in the pane, where that type is three pixels tall.
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib
import numpy as np
import pytest

from gently.ui.web import plots

WEB = Path(__file__).resolve().parents[1] / "gently" / "ui" / "web"
SRC = (WEB / "plots.py").read_text(encoding="utf-8")
CSS = (WEB / "static" / "css" / "operate.css").read_text(encoding="utf-8")
OPERATE = (WEB / "static" / "js" / "operate.js").read_text(encoding="utf-8")

# The narrowest a plot is shown: a column of the pane's grid.
SHOWN_PX = 260
# The least a letter may be there, in pixels, and still be read.
LEAST_PX = 7.5


def _focus(**kw):
    x = np.linspace(-17.0, -5.0, 9)
    y = 1.0e6 * np.exp(-((x + 11.0) ** 2) / (2 * 2.4**2)) + 5.0e4
    return plots.generate_focus_curve_plot(
        x, y, -11.0, fit_params=np.array([1.0e6, -11.0, 2.4, 5.0e4]), r_squared=0.94, **kw
    )


def _summary():
    return plots.generate_calibration_summary_plot(
        "embryo_1", -0.09, 0.038, -11.1, 1.4, 97.8, -2.3, 0.94, 0.91
    )


def _edges():
    return plots.generate_edge_detection_plot(
        [-0.2, -0.1, 0.0, 0.1], [False, True, True, False], edge_top=-0.15, edge_bottom=0.05
    )


class TestTheFigure:
    @pytest.mark.parametrize("draw", [_focus, _summary, _edges])
    def test_every_plot_is_drawn_the_same_size(self, draw):
        img = draw()
        width = round(plots.FIGSIZE[0] * plots.DPI)
        height = round(plots.FIGSIZE[1] * plots.DPI)
        assert img.shape == (height, width, 3) and img.dtype == np.uint8

    def test_it_has_the_pixels_to_be_opened_large(self):
        assert plots.FIGSIZE[0] * plots.DPI >= 1000

    def test_a_caller_can_still_ask_for_its_own_size(self):
        assert _focus(figsize=(4, 3), dpi=100).shape == (300, 400, 3)

    @pytest.mark.parametrize(
        "key",
        ["font.size", "axes.labelsize", "xtick.labelsize", "ytick.labelsize", "legend.fontsize"],
    )
    def test_the_type_can_be_read_where_the_plot_is_shown_smallest(self, key):
        points = float(plots.STYLE[key])  # type: ignore[arg-type]
        shown = points / 72.0 / plots.FIGSIZE[0] * SHOWN_PX
        assert shown >= LEAST_PX, f"{key} is {shown:.1f} px tall at {SHOWN_PX} px wide"

    def test_the_least_type_can_be_read_too(self):
        assert plots.SMALL / 72.0 / plots.FIGSIZE[0] * SHOWN_PX >= LEAST_PX

    def test_no_size_is_set_by_hand_below_the_least(self):
        by_hand = [int(n) for n in re.findall(r"fontsize=(\d+)", SRC)]
        assert all(n >= plots.SMALL for n in by_hand), by_hand

    def test_the_whole_figure_is_drawn_in_the_style(self):
        # matplotlib reads a size when the text is made. A title set outside
        # the style is a title in the default size.
        for name in (
            "generate_focus_curve_plot",
            "generate_calibration_summary_plot",
            "generate_edge_detection_plot",
        ):
            assert f"@_styled\ndef {name}(" in SRC, name

    def test_drawing_leaves_the_defaults_as_they_were(self):
        before = matplotlib.rcParams["font.size"]
        _focus()
        assert matplotlib.rcParams["font.size"] == before

    def test_no_figure_is_left_open(self):
        import matplotlib.pyplot as plt

        before = len(plt.get_fignums())
        _focus()
        _summary()
        _edges()
        assert len(plt.get_fignums()) == before


class TestTheMultiplier:
    def test_large_scores_say_their_multiplier_in_the_label(self):
        unit, label = plots._thousands(np.array([5.0e4, 1.05e6]))
        assert unit == 1.0e6 and "10$^{6}$" in label

    def test_small_scores_are_left_as_they_are(self):
        assert plots._thousands(np.array([0.2, 812.0])) == (1.0, "")

    def test_nothing_to_plot_is_not_an_error(self):
        assert plots._thousands(np.array([])) == (1.0, "")
        assert plots._thousands(np.array([np.nan, np.inf])) == (1.0, "")

    def test_the_curve_and_the_points_are_divided_alike(self):
        body = SRC[SRC.index("def generate_focus_curve_plot(") :]
        body = body[: body.index("\n@_styled")]
        assert "scores / unit" in body and "y_fit / unit" in body
        assert 'f"Focus score{unit_label}"' in body


class TestThePane:
    def test_a_plot_is_given_room(self):
        rule = CSS[CSS.index(".op-cal-kept-strip {") :][:260]
        assert "display: grid;" in rule
        assert f"minmax({SHOWN_PX}px, 1fr)" in rule

    def test_it_fills_its_column(self):
        assert ".op-cal-kept-img img { display: block; width: 100%; height: auto;" in CSS
        assert "height: 84px" not in CSS[CSS.index(".op-cal-kept") :][:1500]

    def test_it_is_asked_for_with_the_pixels_to_fill_it(self):
        asked = re.search(r"\?max=(\d+)\" alt=\"\$\{escapeHtml\(calImageTitle", OPERATE)
        assert asked and int(asked.group(1)) >= 2 * SHOWN_PX
