"""Calibration is a step in the workflow, so it is a pane in the workflow.

#108. Calibrate used to be a button in the SPIM head's status bar, and its
result appeared in a different tab entirely. Ryan, 25:30 on the 2026-08-07
walkthrough: "after setting up the bottom camera and SPIM head, I feel like
I'm not quite sure what to do in gently at this point." There was no next step
on screen, because the next step was a button in the corner of the step before.

His own description of the order is the pane order: "you find the embryos on
the bottom camera, you find them with the SPIM head, you calibrate it, and then
you set up your acquisition parameters."
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OPERATE = ROOT / "gently" / "ui" / "web" / "static" / "js" / "operate.js"
INDEX = ROOT / "gently" / "ui" / "web" / "templates" / "index.html"


def test_the_pane_order_is_the_workflow_order() -> None:
    src = OPERATE.read_text(encoding="utf-8")
    m = re.search(r"const PANE_ORDER = \[([^\]]+)\]", src)
    assert m, "PANE_ORDER is gone — the pane list is spelled out in two places again"
    order = [p.strip().strip("'\"") for p in m.group(1).split(",")]
    assert order == ["bottom", "spim", "cal", "acquire"], order


def test_one_pane_list_not_two() -> None:
    """It used to be written out in showPane and again in showPaneInitial."""
    src = OPERATE.read_text(encoding="utf-8")
    assert "['bottom', 'spim', 'acquire']" not in src
    assert src.count("PANE_ORDER.forEach") >= 2, (
        "a caller stopped using PANE_ORDER — adding a pane now means remembering two places again"
    )


def test_the_subtab_sits_between_spim_and_acquisition() -> None:
    html = INDEX.read_text(encoding="utf-8")
    views = re.findall(r'#?operate|data-view="(bottom|spim|cal|acquire)"', html)
    seen = [v for v in views if v]
    # first occurrence of each, which is the switcher
    order: list[str] = []
    for v in seen:
        if v not in order:
            order.append(v)
    assert order[:4] == ["bottom", "spim", "cal", "acquire"], order


def test_calibrate_left_the_spim_bar() -> None:
    """Its result appeared in another tab; the control belonged with the result."""
    html = INDEX.read_text(encoding="utf-8")
    spim = html[html.index('id="op-pane-spim"') : html.index('id="op-pane-cal"')]
    assert 'id="op-calibrate"' not in spim, "Calibrate is back in the SPIM head bar"
    cal = html[html.index('id="op-pane-cal"') : html.index('id="op-pane-acquire"')]
    assert 'id="op-calibrate"' in cal
    assert 'id="op-cal-result"' in cal, "the fit readout should sit with the button"


def test_the_pane_declares_its_method_before_running_it() -> None:
    """Kesavan on the walkthrough: "it should show what is the method it is
    going to use to calibrate". One method on this rig, so it is named rather
    than offered as a choice."""
    html = INDEX.read_text(encoding="utf-8")
    cal = html[html.index('id="op-pane-cal"') : html.index('id="op-pane-acquire"')]
    assert 'id="op-cal-method"' in cal
    # The LED-not-laser confusion is the thing #106 turned on; say it here.
    assert "laser" in cal.lower()


def test_leaving_the_pane_closes_the_led() -> None:
    """The calibrate path never did, which is half of #106."""
    src = OPERATE.read_text(encoding="utf-8")
    block = src[src.index("        cal: {") :]
    block = block[: block.index("\n        }")]
    assert "forceLedOff()" in block


def test_the_rail_shows_which_embryos_lack_a_fit() -> None:
    """So a refusal at Start is visible beforehand, from any pane."""
    src = OPERATE.read_text(encoding="utf-8")
    assert "showFit: true" in src
    panel = (ROOT / "gently" / "ui" / "web" / "static" / "js" / "panels" / "roster.js").read_text(
        encoding="utf-8"
    )
    # Same field the server-side gate checks.
    assert "slope_um_per_deg" in panel
