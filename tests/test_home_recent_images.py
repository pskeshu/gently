"""Home's recent images are live, and they open.

"are the recent images in the dashboard home view updated? and they are also
thumbnail only like, and not able to click them really."

Neither. The strip refreshed only on entering Home, at most every 15 s, and
never on the event that changes it; and the tiles were divs. The Lightbox
knew images only by store uid, and these are served per session by URL.
"""

from __future__ import annotations

from pathlib import Path

WEB = Path(__file__).resolve().parents[1] / "gently" / "ui" / "web"
HOME = (WEB / "static" / "js" / "home.js").read_text(encoding="utf-8")
LIGHTBOX = (WEB / "static" / "js" / "lightbox.js").read_text(encoding="utf-8")
MAIN_CSS = (WEB / "static" / "css" / "main.css").read_text(encoding="utf-8")


def test_a_volume_landing_redraws_the_strip():
    for ev in ("VOLUME_ACQUIRED", "IMAGE_ACQUIRED", "ACQUISITION_COMPLETED"):
        assert f"'{ev}'" in HOME, f"Home never hears {ev}"
    fn = HOME[HOME.index("function onImagery()") :][:600]
    assert "_imgState.at = 0" in fn, "the throttle would swallow the refresh"
    assert "loadImages(true)" in fn


def test_the_strip_redraws_live_only_while_home_is_on_screen():
    fn = HOME[HOME.index("function onImagery()") :][:600]
    assert "if (!homeIsShowing()) return;" in fn
    assert "getElementById('home-content')" in HOME, "the panel is what the shell marks active"


def test_a_tile_is_a_button_that_opens_the_lightbox():
    tiles = HOME[HOME.index("_recent = recent.map(") :][:3600]
    assert '<button type="button" class="home-image" data-home-image=' in tiles
    assert "openRecent(Number(b.dataset.homeImage))" in tiles
    opener = HOME[HOME.index("function openRecent(index)") :][:700]
    assert "Lightbox.open(" in opener and "url: s.url" in opener


def test_the_lightbox_opens_an_image_by_url():
    """Home's projections are served per session and carry no store uid."""
    assert "img.url || `/api/images/${img.uid}/png`" in LIGHTBOX, "the main image ignores url"
    assert "if (img.uid || img.url)" in LIGHTBOX, "thumbnails ignore url"
    assert "} else if (img.url) {" in LIGHTBOX, "the open() path ignores url"
    assert '<img src="${img.url}" alt="${img.data_type' in LIGHTBOX, "open() thumbs ignore url"


def test_the_lightbox_names_a_url_item_by_its_timepoint():
    show = LIGHTBOX[LIGHTBOX.index("    showImage(index) {") :][:2600]
    assert "const tp = img.metadata?.timepoint;" in show
    assert "`T${tp}`" in show, "title and TIME fall back to '-'/'Image' with no timestamp"


def test_tiles_read_as_buttons():
    assert "button.home-image:hover" in MAIN_CSS
    assert "cursor: pointer" in MAIN_CSS[MAIN_CSS.index("button.home-image {") :][:400]
