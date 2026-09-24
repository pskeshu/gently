"""What the region editor needs from the page to work at all.

CI runs no JavaScript and no browser, so the things that actually broke here
are checked as source invariants. Each of these is a bug that shipped, not a
hypothetical:

* a stray `}` in main.css, which makes the CSS parser discard the NEXT rule —
  the failure is silent, and it lands on whoever adds a rule after it;
* the "Edit region" button sitting inside `pointer-events: none`, so a real
  mouse click passed straight through it to the map and nothing happened,
  which is exactly how it was reported;
* an unclosed `<div>`, which made the readout card a child of the editor strip
  and painted it over the controls.
"""

from __future__ import annotations

import re
from pathlib import Path

WEB = Path(__file__).resolve().parents[1] / "gently" / "ui" / "web"
CSS = (WEB / "static" / "css" / "main.css").read_text(encoding="utf-8")
HTML = (WEB / "templates" / "index.html").read_text(encoding="utf-8")
EDITOR = (WEB / "static" / "js" / "region-editor.js").read_text(encoding="utf-8")
DEVICES = (WEB / "static" / "js" / "devices.js").read_text(encoding="utf-8")

NO_COMMENTS = re.sub(r"/\*.*?\*/", "", CSS, flags=re.S)


def test_main_css_braces_balance():
    """A stray `}` is not a cosmetic problem.

    Per the CSS grammar a `}` at the top level starts a qualified rule whose
    prelude runs to the next block — so the parser eats the rule that follows
    it and drops both. That is how a correct `pointer-events: auto` rule came
    to have no effect at all, with nothing in the console to say so.
    """
    opens, closes = NO_COMMENTS.count("{"), NO_COMMENTS.count("}")
    assert opens == closes, f"main.css has {opens} '{{' and {closes} '}}'"


def test_the_readouts_controls_can_be_clicked():
    """The readout is a pass-through overlay so the map under it stays live.

    `pointer-events` inherits, so every control put inside it — the region
    button, the XY-limits switch — inherits `none` and cannot be clicked. Note
    that a scripted `.click()` ignores pointer-events entirely, so this cannot
    be caught by clicking it from a test; only by a real press, or by this.
    """
    assert "pointer-events: none" in CSS.split(".devices-map-readout {", 1)[1][:400]
    rule = re.search(r"\.devices-map-readout button[^{]*\{[^}]*pointer-events:\s*auto", CSS)
    assert rule, "controls inside .devices-map-readout never opt back in"


def test_the_editor_strip_is_a_sibling_of_the_map_overlays():
    """Not their parent.

    When its `</div>` went missing the compass and the whole XY-stage rail
    became children of the strip, so the readout card painted on top of the
    controls and the strip's own width collapsed around them.
    """
    body = HTML[HTML.index('id="devices-map-wrap"') : HTML.index("devices-view-details")]
    depth, seen_strip_close = 0, False
    for line in body.split("\n"):
        depth += len(re.findall(r"<div\b", line)) - line.count("</div>")
        if 'id="region-strip"' in line:
            strip_depth = depth
        if "devices-compass" in line:
            seen_strip_close = depth == strip_depth
    assert seen_strip_close, "the compass is inside the region strip"


def test_the_map_view_markup_is_balanced():
    body = HTML[HTML.index('id="devices-view-map"') : HTML.index("devices-view-details")]
    assert len(re.findall(r"<div\b", body)) == body.count("</div>")


def test_hidden_actually_hides_every_panel_the_editor_shows():
    """`[hidden]` is specificity 0,1,0.

    A class rule that sets `display` beats it, so `el.hidden = true` leaves the
    panel on screen. Every one of these sets `display`, so every one needs the
    guard — this is the bug that left a ready device layer still saying
    "2/5 · Initializing Micro-Manager core".
    """
    for cls in (".region-strip", ".region-history", ".region-cam-ph"):
        rule = re.search(re.escape(cls) + r"\s*\{[^}]*display:", CSS)
        if not rule:
            continue
        assert re.search(re.escape(cls) + r"\[hidden\]\s*\{[^}]*display:\s*none", CSS), (
            f"{cls} sets display without a [hidden] guard"
        )


def test_the_editor_refuses_a_stage_it_cannot_read():
    """An editor opened on four undefined numbers would offer to write them."""
    assert "return false" in EDITOR.split("async function open(", 1)[1][:1600]
    assert "if (!await RegionEditor.open(" in DEVICES


def test_the_sheet_frames_the_travel_while_editing():
    """An edge you cannot see is an edge you cannot click.

    The map otherwise fits the region as applied, so pushing a bound outward
    puts it off the sheet — and the whole gesture is clicking the edge.
    """
    block = DEVICES.split("function computeViewBox()", 1)[1][:2000]
    assert "RegionEditor.isOpen()" in block
    assert "RegionEditor.travel()" in block


def test_the_old_wizard_is_gone_from_every_layer():
    """Markup, stylesheet and script — a half-removed panel is still a panel."""
    for name, src in (("index.html", HTML), ("main.css", CSS), ("devices.js", DEVICES)):
        assert "devices-region-wiz" not in src, f"wizard remnants in {name}"
        assert "devices-region-schem" not in src, f"wizard remnants in {name}"
