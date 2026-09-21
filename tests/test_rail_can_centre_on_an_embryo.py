"""The embryo list can send the stage to the embryo, from any pane.

The SPIM head already knew when it was pointed at the wrong place — the
caption reads "Selected: embryo 3 — stage is elsewhere" and the frame is
blanked to "Not at this embryo" rather than showing the previous embryo's
pixels. What it did not have was a way to fix it. "Centre the stage on this
embryo" existed only on the Acquisition pane's roster and on the bottom-camera
image, so from the SPIM head the operator had to leave the pane, act
elsewhere, and come back.

The rail is the embryo list beside EVERY pane, so the verb belongs there. It
draws as a crosshair rather than the word: the rail is ~168px and a
word-width button wrapped "Embryo 1  −184, −585" onto three lines.

CI runs no JavaScript, so the wiring is pinned in source.
"""

from __future__ import annotations

import re
from pathlib import Path

WEB = Path(__file__).resolve().parents[1] / "gently" / "ui" / "web"
OPERATE_JS = WEB / "static" / "js" / "operate.js"
ROSTER_JS = WEB / "static" / "js" / "panels" / "roster.js"


def test_the_rail_offers_centre_on_every_pane() -> None:
    js = OPERATE_JS.read_text(encoding="utf-8")
    mount = re.search(r"RosterPanel\.mount\('op-erail-list',\s*\{([^}]*)\}", js, re.S)
    assert mount, "the rail no longer mounts the roster"
    opts = mount.group(1)
    assert "'centre'" in opts, (
        "the rail lost its centre verb — the SPIM head is back to reporting "
        "'stage is elsewhere' with no way to act on it"
    )
    assert "compact: true" in opts, "the rail's verbs are no longer compact; rows will wrap"


def test_centre_moves_through_the_one_chokepoint() -> None:
    """Centring is a stage move and must keep its interlock."""
    js = OPERATE_JS.read_text(encoding="utf-8")
    centre = re.search(r"async function centerOnEmbryo\(emb\) \{(.*?)\n    \}", js, re.S)
    assert centre, "centerOnEmbryo is gone"
    assert "moveStageTo(" in centre.group(1), (
        "centring no longer routes through moveStageTo, the only caller of the "
        "stage-move route and the whole XY/F-drive guard"
    )


def test_a_glyph_verb_still_says_what_it_is() -> None:
    """Icon-only buttons carry their label for screen readers and tooltips."""
    js = ROSTER_JS.read_text(encoding="utf-8")
    assert 'aria-label="${a.title}"' in js, (
        "roster action buttons lost their aria-label; compact ones render as a "
        "bare glyph with no accessible name"
    )
    assert "opts.compact && a.icon" in js, "the compact rendering is gone"
    # The wide mount keeps words: Acquisition is the pre-run review surface.
    acq = re.search(
        r"RosterPanel\.mount\('op-roster',\s*\n?\s*\{([^}]*)\}",
        OPERATE_JS.read_text(encoding="utf-8"),
        re.S,
    )
    assert acq and "compact" not in acq.group(1), (
        "the Acquisition roster went compact; it has the width for words"
    )
