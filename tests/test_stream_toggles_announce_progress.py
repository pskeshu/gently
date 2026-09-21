"""A stream toggle says what it is doing, and can be clicked at all.

Two defects, found by pressing the button rather than reading the code.

1. **"Start camera" spent the whole wait pretending to be forbidden.** The
   handler disabled the button and left the label alone, so while the stream
   came up the control read "Start camera" at 40% opacity with a `not-allowed`
   cursor — the exact rendering of "you may not press this" — and then flipped
   to a green "Stop camera" with nothing in between. Disabled is a statement
   about permission; a request in flight is a statement about time.

2. **It could not be clicked with a mouse at all** below the 900px container
   breakpoint. `.op-main` and `.op-inst` are both `grid-row: 2`, and the
   stacked layout reset only the column, so both landed in one cell; worse,
   `.op-main` kept `min-height: 0`, so its row squashed to ~403px around a
   708px square camera whose `inset: 0` placeholder then painted over the
   instrument rail and swallowed its clicks. A Playwright click timed out
   against "op-cam-ph intercepts pointer events" — which is what an operator's
   finger was doing too.

CI runs no JavaScript, so both are pinned here.
"""

from __future__ import annotations

import re
from pathlib import Path

WEB = Path(__file__).resolve().parents[1] / "gently" / "ui" / "web"
OPERATE_JS = WEB / "static" / "js" / "operate.js"
OPERATE_CSS = WEB / "static" / "css" / "operate.css"


def test_both_stream_toggles_name_the_verb_in_progress() -> None:
    js = OPERATE_JS.read_text(encoding="utf-8")
    assert re.search(r"function pending\(btn, verb\)", js), (
        "the shared pending-state helper is gone; a toggle in flight will look disabled again"
    )
    # Camera and SPIM view are the same shape and must stay in step.
    assert js.count("const done = pending(b,") == 2, (
        "a stream toggle stopped using the pending helper"
    )
    # Quoting of the ellipsis is not the point; naming the verb is.
    assert js.count("'Starting") == 2 and js.count("'Stopping") == 2, (
        "the in-flight labels no longer name which verb is running"
    )
    assert "aria-busy" in js, "the pending state is no longer announced to screen readers"


def test_a_pending_control_does_not_render_as_a_forbidden_one() -> None:
    css = OPERATE_CSS.read_text(encoding="utf-8")
    block = re.search(r"\.op-btn\.is-pending[^{]*\{([^}]*)\}", css, re.S)
    assert block, "the pending button style is gone"
    body = block.group(1)
    assert "opacity: 1" in body, (
        "a pending button is faded again — that is the disabled look, which means 'not allowed'"
    )
    assert "cursor: progress" in body, "a pending button no longer says 'wait' with its cursor"
    assert ".op-btn-spin" in css, "the in-button spinner is gone"


def test_the_stacked_layout_does_not_bury_the_instrument_rail() -> None:
    """The camera must not overflow its column onto the controls beside it."""
    css = OPERATE_CSS.read_text(encoding="utf-8")
    stacked = re.search(r"@container operate \(max-width: 900px\) \{(.*?)\n\}", css, re.S)
    assert stacked, "the stacked-layout container query is gone"
    body = stacked.group(1)
    assert re.search(r"\.op-main\s*\{[^}]*grid-row:\s*2", body), (
        "stacked, .op-main no longer claims its own row"
    )
    assert re.search(r"\.op-inst\s*\{[^}]*grid-row:\s*3", body), (
        "stacked, the instrument rail shares a cell with the camera again — "
        "the camera's placeholder will swallow its clicks"
    )
    assert re.search(r"\.op-main\s*\{\s*min-height:\s*auto", body), (
        "stacked, .op-main can be squashed below its content again, so the camera "
        "overflows its column and covers whatever is beneath it"
    )
