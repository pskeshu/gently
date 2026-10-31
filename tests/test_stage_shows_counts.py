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


def test_the_lightbox_draws_a_counts_item_on_the_viewer_and_keeps_the_jpeg_otherwise() -> None:
    """A calibration exposure opens in the lightbox as its counts; a plot, a
    Home tile or any item without `u16` opens as the picture it always was."""
    lb = (WEB / "static" / "js" / "lightbox.js").read_text(encoding="utf-8")
    assert "paintPixels(img)" in lb and "U16View.create(" in lb
    for fn in ("showImage(index) {", "showImageByUid(index) {"):
        body = lb.split(fn)[1].split("\n    },\n")[0]
        assert "this.paintPixels(img)" in body, f"{fn} skips the viewer"
    assert "this.hidePixels();" in lb.split("    close() {")[1].split("\n    },\n")[0]
    assert "zoomTarget(view.host)" in lb, "zoom would stay on the hidden <img>"
    op = (WEB / "static" / "js" / "operate.js").read_text(encoding="utf-8")
    assert "u16: i.u16," in op.split("function openCalImage(index)")[1].split("\n    }\n")[0]
    routes = (WEB / "routes" / "calibration_records.py").read_text(encoding="utf-8")
    assert "/image/{n}.u16" in routes


def test_every_thumbnail_is_the_counts_with_the_jpeg_behind_it() -> None:
    """Embryo tiles, the session's embryo cards and the Home tiles mount the
    viewer at the size a tile has (?max=256) and keep the JPEG as the fallback;
    the Home lightbox items carry the counts too."""
    v = (WEB / "static" / "js" / "u16view.js").read_text(encoding="utf-8")
    assert "tile(host, url, fallback, alt)" in v and "host.replaceWith(img)" in v
    for name, host in (
        ("embryos.js", "tile-pix"),
        ("review.js", "embryo-pix"),
        ("home.js", "home-pix"),
    ):
        src = (WEB / "static" / "js" / name).read_text(encoding="utf-8")
        assert f'class="{host}"' in src, f"{name} has no viewer host in its tile"
        assert "U16View.tile(" in src and "max=256" in src, (
            f"{name} does not mount the viewer at tile size"
        )
        assert "data-jpg=" in src, f"{name} dropped the JPEG fallback"
    home = (WEB / "static" / "js" / "home.js").read_text(encoding="utf-8")
    assert "u16: u16Of(s)" in home.split("function openRecent(index)")[1].split("\n    }\n")[0]
