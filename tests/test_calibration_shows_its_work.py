"""A calibration run shows what it is looking at.

Calibration is sixty to eighty exposures behind a disabled button. Every one of
those frames is already captured, already judged by Claude, already pushed to
the viz server and already broadcast to every client — and the Operate pane
showed none of it. "It shows no in place update on what is going on... not even
images preview of calibration or whatever."

The panel that fixes this reads the frames off the existing broadcast, so the
thing most likely to break it is not its own code but DRIFT: a phase in
`calibration_tools.py` that starts pushing under a new `data_type` the panel
does not listen for goes silent with no error anywhere. That is what this test
pins, across the Python/JS boundary CI does not otherwise cross.

The panel's own reading of a frame is tested in `tests/js/calprogress.test.mjs`
(node --test). CI runs no JavaScript, so the wiring is pinned here.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CAL_TOOLS = ROOT / "gently" / "app" / "tools" / "calibration_tools.py"
PANEL_JS = ROOT / "gently" / "ui" / "web" / "static" / "js" / "panels" / "calprogress.js"
OPERATE_JS = ROOT / "gently" / "ui" / "web" / "static" / "js" / "operate.js"
INDEX_HTML = ROOT / "gently" / "ui" / "web" / "templates" / "index.html"


def _pushed_types() -> set[str]:
    """Every data_type calibration pushes to the viz server."""
    src = CAL_TOOLS.read_text(encoding="utf-8")
    return set(re.findall(r'data_type="([a-z_]+)"', src))


def _panel_types() -> set[str]:
    js = PANEL_JS.read_text(encoding="utf-8")
    block = re.search(r"const TYPES = new Set\(\[(.*?)\]\)", js, re.S)
    assert block, "the panel no longer declares which frames it listens for"
    return set(re.findall(r"'([a-z_]+)'", block.group(1)))


def test_the_panel_listens_for_every_frame_calibration_pushes() -> None:
    pushed, watched = _pushed_types(), _panel_types()
    assert pushed, "calibration stopped pushing frames at all"
    missing = pushed - watched
    assert not missing, (
        f"calibration pushes {sorted(missing)} but the progress panel does not listen "
        "for it — that phase of the run will show nothing, silently"
    )


def test_the_panel_does_not_claim_frames_that_are_not_calibration() -> None:
    """It listens on the broadcast every image rides, so it must be narrow."""
    stray = _panel_types() - _pushed_types()
    assert not stray, (
        f"the panel listens for {sorted(stray)}, which calibration does not push; a "
        "snapshot or volume projection would be narrated as a calibration frame"
    )


def test_the_run_opens_and_closes_the_panel() -> None:
    js = OPERATE_JS.read_text(encoding="utf-8")
    run = re.search(r"async function calibrateSelected\(\) \{(.*?)\n    \}", js, re.S)
    assert run, "calibrateSelected is gone"
    body = run.group(1)
    assert "CalProgressPanel.begin(" in body, (
        "the run no longer opens the progress panel; the pane is blank again until "
        "the first exposure comes back"
    )
    # Both exits report. A failure especially: the frames are the evidence of
    # WHERE it went wrong, which a bare 'failed' withholds.
    assert body.count("CalProgressPanel.finish(") >= 2, (
        "the progress panel is not closed on both the success and failure paths — "
        "a finished run would keep pulsing as though it were still going"
    )
    assert "CalProgressPanel.finish(false" in body, "a failed calibration says nothing"
    assert "CalProgressPanel.mount('op-cal-progress')" in js, "the panel is never mounted"


def test_the_panel_is_served() -> None:
    html = INDEX_HTML.read_text(encoding="utf-8")
    assert "panels/calprogress.js" in html, "the panel script is not loaded"
    assert 'id="op-cal-progress"' in html, "the panel has no host in the calibration pane"
