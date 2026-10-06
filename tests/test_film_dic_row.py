"""In the film, the DIC overview is a row of the film; and a thumbnail is an
average, not a sample.

"take a look at the embryo tab - say in the film view. the DIC thumbnails or
aesthetics look a bit off"

Two causes. The thumbnails were made by keeping one pixel in eight, which
keeps that pixel's noise at full strength: a dim 2048 px frame became grain.
And the overview sat above the film as its own block, at twice the size of a
film cell, showing the last twelve frames over a film that starts at the first.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from gently.core.imaging import downsample_mean

ROOT = Path(__file__).resolve().parents[1] / "gently"
EMBRYOS = (ROOT / "ui" / "web" / "static" / "js" / "embryos.js").read_text(encoding="utf-8")
CSS = (ROOT / "ui" / "web" / "static" / "css" / "main.css").read_text(encoding="utf-8")
ROUTE = (ROOT / "ui" / "web" / "routes" / "dic.py").read_text(encoding="utf-8")
ORCH = (ROOT / "app" / "orchestration" / "timelapse.py").read_text(encoding="utf-8")


def test_averaging_removes_the_grain_that_sampling_keeps():
    rng = np.random.default_rng(7)
    flat = 200.0 + rng.normal(0, 30.0, size=(2048, 2048))
    sampled = flat[::8, ::8]
    averaged = downsample_mean(flat, 256)
    assert averaged.shape == (256, 256)
    assert averaged.std() < sampled.std() / 6, "an 8x8 mean has an eighth of the noise"
    assert abs(averaged.mean() - flat.mean()) < 1.0, "and the same brightness"


def test_a_small_image_is_left_alone():
    img = np.ones((100, 120), dtype=np.uint16)
    assert downsample_mean(img, 256) is img


def test_the_longer_side_is_what_is_bounded():
    out = downsample_mean(np.zeros((1024, 2048), dtype=np.uint16), 512)
    assert max(out.shape) <= 512 and out.shape == (256, 512)


def test_a_ragged_edge_is_cropped_not_crashed_on():
    out = downsample_mean(np.zeros((2050, 2047), dtype=np.uint16), 256)
    assert max(out.shape) <= 256 and out.ndim == 2


def test_both_thumbnail_paths_average():
    assert "downsample_mean(arr, int(max))" in ROUTE
    assert "[::step, ::step]" not in ROUTE
    assert "normalize_to_uint8(downsample_mean(image, 512))" in ORCH
    assert "image[::step, ::step]" not in ORCH


def test_the_film_draws_the_overview_as_its_first_row():
    film = EMBRYOS[EMBRYOS.index("    renderFilmstripView() {") :][:4000]
    container = film.index("let html = '<div class=\"filmstrip-container\">';")
    row = film.index("html += this._filmDicRow(thumbSize, config);")
    first_embryo = film.index("for (const embryo of embryos) {")
    assert container < row < first_embryo


def test_the_row_is_a_film_row_with_film_cells():
    fn = EMBRYOS[EMBRYOS.index("    _filmDicRow(thumbSize, config) {") :][:2200]
    assert 'class="filmstrip-row filmstrip-dic-row"' in fn
    assert 'class="filmstrip-label"' in fn, "the label column is what lines the rows up"
    assert 'width="${thumbSize}" height="${thumbSize}"' in fn, "the same cell as an embryo's"
    assert "const shown = skip > 1 ?" in fn and "all.slice(-12)" not in fn, "every frame"
    assert "if (!all.length) return '';" in fn


def test_a_dic_cell_opens_the_viewer_and_is_not_an_embryo_timepoint():
    film = EMBRYOS[EMBRYOS.index("    renderFilmstripView() {") :][:6000]
    assert "this.openDicViewer(Number(cell.dataset.dicIndex))" in film
    assert "'.filmstrip-cell:not(.filmstrip-dic-cell)'" in film


def test_the_block_above_gives_way_in_the_film_and_comes_back():
    strip = EMBRYOS[EMBRYOS.index("    renderDicStrip() {") :][:1800]
    assert "strip.hidden = all.length === 0 || inFilm || expanded;" in strip
    assert "if (inFilm) { this.renderFilmstripView(); return; }" in strip
    switch = EMBRYOS[EMBRYOS.index("    switchView(viewName) {") :][:900]
    assert "this.renderDicStrip();" in switch


def test_the_overview_thumbnail_is_centred_not_cropped_left():
    block = CSS[CSS.index(".filmstrip-dic-thumb {") :][:300]
    assert "object-position: center;" in block
