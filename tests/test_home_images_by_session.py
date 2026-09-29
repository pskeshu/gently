"""Home's recent images are under their sessions, embryo by embryo.

"and the recent images feature in the home screen can be grouped by session
id and embryos?"

The strip was the first eight images wherever they fell, each captioned
``embryo_1 · t24``. ``embryo_1`` is a different embryo in every session, and
nothing said which.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from gently.core.file_store import FileStore
from gently.ui.web.routes import sessions as sessions_routes

WEB = Path(__file__).resolve().parents[1] / "gently" / "ui" / "web"
HOME = (WEB / "static" / "js" / "home.js").read_text(encoding="utf-8")
LIGHTBOX = (WEB / "static" / "js" / "lightbox.js").read_text(encoding="utf-8")
CSS = (WEB / "static" / "css" / "main.css").read_text(encoding="utf-8")

IDS = ["aaaa1111", "bbbb2222", "cccc3333", "dddd4444"]


@pytest.fixture
def store(tmp_path, monkeypatch):
    fs = FileStore(root=tmp_path)
    for sid in IDS:
        fs.create_session(sid)
        for n in (1, 2, 3):
            fs.register_embryo(
                sid, f"embryo_{n}", position_coarse={"x": 1.0, "y": 2.0}, role="test"
            )
            proj = fs._embryo_dir(sid, f"embryo_{n}") / "projections"
            proj.mkdir(parents=True, exist_ok=True)
            for tp in range(n + 1):
                (proj / f"t{tp:04d}.jpg").write_bytes(b"jpg")
    # the newest first, whatever minute the fixture made them in
    monkeypatch.setattr(fs, "recent_session_ids", lambda n: list(reversed(IDS))[:n])
    return fs


def _images(store, **params):
    server = MagicMock()
    server.agent_bridge.agent.store = store
    server.agent_bridge.agent.session_id = None
    app = FastAPI()
    app.include_router(sessions_routes.create_router(server))
    r = TestClient(app).get("/api/home/recent-images", params=params)
    assert r.status_code == 200, r.text
    return r.json()["images"]


class TestTheRoute:
    def test_the_last_sessions_whole(self, store):
        got = _images(store, limit=24, sessions=2)
        assert [(i["session_id"], i["embryo_id"]) for i in got] == [
            ("dddd4444", "embryo_1"),
            ("dddd4444", "embryo_2"),
            ("dddd4444", "embryo_3"),
            ("cccc3333", "embryo_1"),
            ("cccc3333", "embryo_2"),
            ("cccc3333", "embryo_3"),
        ]

    def test_an_image_says_its_session_and_how_much_its_embryo_has(self, store):
        first = _images(store, limit=24, sessions=1)[1]
        assert first["session_id"] == "dddd4444" and first["session_created_at"]
        assert first["embryo_id"] == "embryo_2"
        assert first["timepoint"] == 2 and first["timepoints"] == 3

    def test_a_session_without_images_is_not_one_of_them(self, store):
        store.create_session("eeee5555")
        store.recent_session_ids = lambda n: (["eeee5555"] + list(reversed(IDS)))[:n]
        got = _images(store, limit=24, sessions=1)
        assert {i["session_id"] for i in got} == {"dddd4444"}

    def test_without_the_parameter_it_is_as_it_was(self, store):
        got = _images(store, limit=8)
        assert len(got) == 8
        assert [i["session_id"] for i in got][:4] == ["dddd4444"] * 3 + ["cccc3333"]

    def test_the_limit_still_bounds_it(self, store):
        assert len(_images(store, limit=4, sessions=3)) == 4

    def test_the_count_of_sessions_is_bounded(self, store):
        got = _images(store, limit=48, sessions=9999)
        assert len({i["session_id"] for i in got}) == 4


class TestHome:
    def test_it_asks_for_sessions_not_for_eight_images(self):
        assert "const IMAGE_SESSIONS_N = 3;" in HOME
        assert "recent-images?limit=${IMAGES_N}&sessions=${IMAGE_SESSIONS_N}" in HOME

    def test_each_session_has_its_head(self):
        render = HOME[HOME.index("el.innerHTML = groupBySession(_recent)") :][:1500]
        assert 'class="home-image-group"' in render
        assert "escapeHtml(sessionTitle(g))" in render
        assert "escapeHtml(g.session_id)" in render
        assert "g.items.length} embryo" in render

    def test_a_group_keeps_the_images_places_in_the_whole_list(self):
        # The Lightbox walks the whole list: a tile opens at its own place
        # in it, not at its place in its group.
        fn = HOME[HOME.index("    function groupBySession(images) {") :][:1100]
        assert "images.forEach((image, index) => {" in fn
        assert "g.items.push({ image, index });" in fn
        assert "g.items.map(x => tile(x.image, x.index))" in HOME

    def test_embryos_are_in_their_order(self):
        fn = HOME[HOME.index("    function groupBySession(images) {") :][:1100]
        assert "{ numeric: true }" in fn, "embryo_10 would sort before embryo_2"

    def test_a_session_is_named_or_dated(self):
        fn = HOME[HOME.index("    function sessionTitle(g) {") :][:700]
        assert "g.session_name !== 'unnamed'" in fn
        assert "toLocaleString" in fn

    def test_the_session_open_now_is_said(self):
        assert "g.session_id === liveSessionId()" in HOME
        assert "open now" in HOME

    def test_a_tile_says_which_session_its_embryo_is_of(self):
        fn = HOME[HOME.index("            const tile = (s, i) => {") :][:1300]
        assert "`${s.session_id}/${s.embryo_id}${tp}`" in fn
        assert "escapeHtml(title)" in fn and "escapeHtml(label)" in fn
        assert "escapeHtml(embryoName(s.embryo_id))" in fn

    def test_the_sessions_folder_is_one_click_away(self):
        assert "{ what: 'session', session_id: g.session_id }, 'show'," in HOME

    def test_the_groups_are_styled(self):
        for cls in (".home-image-group-head", ".home-image-session-id", ".home-image-live"):
            assert cls + " {" in CSS, cls


class TestTheViewer:
    def test_the_embryo_is_named_with_its_session(self):
        fn = LIGHTBOX[LIGHTBOX.index("    embryoOf(img) {") :][:400]
        assert "md.session_id ? `${md.session_id}/${md.embryo_id}` : md.embryo_id" in fn

    def test_both_ways_of_showing_use_it(self):
        assert LIGHTBOX.count("this.els.infoEmbryo.textContent = this.embryoOf(img);") == 2
