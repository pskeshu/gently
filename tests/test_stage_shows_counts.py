"""The stage draws a frame as its counts when it has them.

An embryo's projection and an overview frame both have a `.u16` route now
(#314, #318). The stage that shows them large puts a frame that carries `u16`
on the uint16 viewer, so the contrast is set once and held across the series
and the value under the cursor reads out. The JPEG stays for the thumbnails,
the hover preview, Compare and any frame without counts.
"""

from pathlib import Path

WEB = Path(__file__).resolve().parents[1] / "gently" / "ui" / "web"
STAGE = (WEB / "static" / "js" / "panels" / "overview-stage.js").read_text(encoding="utf-8")


def test_the_stage_puts_a_frame_with_counts_on_the_viewer() -> None:
    assert "U16View.create(" in STAGE
    assert "const hasPixels = f => !!(f && f.u16" in STAGE
    show = STAGE.split("    function show(f) {")[1].split("\n    }\n")[0]
    assert "showPixels(f, 4)" in show and "showPixels(f, 1)" in show, (
        "a frame with counts does not go through the viewer at scrub and settle sizes"
    )
    assert "has-pixels" in show, "the JPEG path does not take the box back from the viewer"


def test_the_range_holds_across_the_series_and_the_bar_sits_beside() -> None:
    body = STAGE.split("    function showPixels(f, scale) {")[1].split("\n    }\n")[0]
    assert "view.setRange(keep[0], keep[1], keep[2])" in body
    assert 'id="ov-display"' in STAGE and "appendChild(pix.bar)" in STAGE


def test_compare_and_a_frame_without_counts_keep_the_jpeg() -> None:
    assert "!compare" in STAGE.split("const hasPixels")[1].split("\n")[0]
    css = (WEB / "static" / "css" / "main.css").read_text(encoding="utf-8")
    assert ".ov-stage.has-pixels .ov-img:not(.ov-img-b) { display: none; }" in css


def test_both_kinds_of_frame_carry_their_counts_url() -> None:
    embryos = (WEB / "static" / "js" / "embryos.js").read_text(encoding="utf-8")
    assert "/volumes/${t}/projection.u16" in embryos
    bf = (WEB / "routes" / "bf.py").read_text(encoding="utf-8")
    assert '"u16": f"/api/bf/frames/{stem}.u16"' in bf
