"""Calibration runs the operator's parameters, not a baked-in recipe.

`calibrate_embryo` takes eight arguments. The Devices-tab pane sent none of
them — `postJSON(..., {})` — so every calibration paid for a Claude-vision
galvo-edge hunt even on a rig whose range is already known, and the two
numbers that place the pair of calibration points inside that range
(`edge_tolerance_deg`, `inset_fraction`) could not be reached at all. The
tool's own docstring warns those two produce noise-amplified slopes on small
embryos when they are wrong, which made them exactly the wrong pair to bake in.

Two properties matter and are easy to lose:

* an untouched pane must behave as it always did, so the defaults in the form
  must equal the tool's own and only *differences* travel; and
* the route must not forward numbers unchecked — these drive a galvo.

CI runs no JavaScript, so the browser side is pinned in source.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path

from gently.app.tools import calibration_tools

ROOT = Path(__file__).resolve().parents[1]
OPERATE_JS = ROOT / "gently" / "ui" / "web" / "static" / "js" / "operate.js"
INDEX_HTML = ROOT / "gently" / "ui" / "web" / "templates" / "index.html"
DATA_PY = ROOT / "gently" / "ui" / "web" / "routes" / "data.py"

# id in the form -> the tool argument it stands for
FIELDS = {
    "cal-zbuf": "z_buffer_um",
    "cal-estep": "edge_step",
    "cal-erange": "edge_max_range",
    "cal-etol": "edge_tolerance_deg",
    "cal-inset": "inset_fraction",
    "cal-gtop": "galvo_top",
    "cal-gbot": "galvo_bottom",
}


def _cal_defaults() -> dict[str, object]:
    js = OPERATE_JS.read_text(encoding="utf-8")
    block = re.search(r"const CAL_DEFAULTS = \{(.*?)\n    \};", js, re.S)
    assert block, "CAL_DEFAULTS is gone — the pane no longer knows the tool's defaults"
    out: dict[str, object] = {}
    for key, raw in re.findall(r"(\w+):\s*([^,\n]+),", block.group(1)):
        val = raw.split("//")[0].strip()
        if val in ("null", "true", "false"):
            out[key] = {"null": None, "true": True, "false": False}[val]
        else:
            out[key] = float(val)
    return out


def test_the_form_offers_every_argument_the_tool_takes() -> None:
    html = INDEX_HTML.read_text(encoding="utf-8")
    sig = inspect.signature(calibration_tools.calibrate_embryo)
    for field_id, arg in FIELDS.items():
        assert arg in sig.parameters, f"calibrate_embryo lost {arg}; the pane still offers it"
        assert f'id="{field_id}"' in html, f"the pane lost its control for {arg}"
    assert 'id="cal-edges"' in html, "the edge-detection choice is gone"


def test_an_untouched_pane_asks_for_exactly_what_it_used_to() -> None:
    """The form's defaults must equal the tool's, or 'change nothing' changes something."""
    defaults = _cal_defaults()
    sig = inspect.signature(calibration_tools.calibrate_embryo)
    for field_key, arg in (
        ("zbuf", "z_buffer_um"),
        ("estep", "edge_step"),
        ("erange", "edge_max_range"),
        ("etol", "edge_tolerance_deg"),
        ("inset", "inset_fraction"),
    ):
        assert defaults[field_key] == sig.parameters[arg].default, (
            f"the pane's default for {arg} drifted from the tool's "
            f"({defaults[field_key]} vs {sig.parameters[arg].default}); an operator who "
            "touches nothing now gets a different calibration than before"
        )
    # Edge detection on by default means skip_edge_detection stays off.
    assert defaults["edges"] is True
    assert sig.parameters["skip_edge_detection"].default is False

    js = OPERATE_JS.read_text(encoding="utf-8")
    send = re.search(r"function calibrationSettings\(\) \{(.*?)\n    \}", js, re.S)
    assert send, "calibrationSettings is gone"
    assert "!== def" in send.group(1), (
        "the pane now sends every field rather than only what differs; an untouched "
        "pane would pin the tool's defaults at today's values forever"
    )


def test_the_route_refuses_numbers_that_would_drive_the_galvo_wrong() -> None:
    src = DATA_PY.read_text(encoding="utf-8")
    block = re.search(r"for key, cast, lo, hi in \((.*?)\n        \):", src, re.S)
    assert block, "the calibrate route no longer range-checks its payload"
    bounded = dict(re.findall(r'\("(\w+)", float, ([-\d.]+), [\d.]+\)', block.group(1)))
    for arg in FIELDS.values():
        assert arg in bounded, f"{arg} reaches the tool unchecked"
    # inset is applied to each side; 0.5 collapses the two points onto one another.
    assert '("inset_fraction", float, 0.0, 0.49)' in block.group(1), (
        "inset_fraction's upper bound moved off 0.49 — at 0.5 the two calibration "
        "points coincide and the slope fit has no baseline"
    )
    assert "galvo_top and galvo_bottom are equal" in src, (
        "the route accepts an empty galvo range again"
    )
