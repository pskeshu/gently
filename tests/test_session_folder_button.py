"""The header's session pill opens the session's folder.

"next to the tag name in the header showing the session ID there must be a
button to open the folder where all the images are stored." — "in windows
explorer or whatever is the file manager on the OS"
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from gently.core.file_store import FileStore
from gently.ui.web import auth
from gently.ui.web.routes import sessions as sessions_routes

WEB = Path(__file__).resolve().parents[1] / "gently" / "ui" / "web"
HEADER = (WEB / "templates" / "_header.html").read_text(encoding="utf-8")
APP_JS = (WEB / "static" / "js" / "app.js").read_text(encoding="utf-8")
MAIN_CSS = (WEB / "static" / "css" / "main.css").read_text(encoding="utf-8")


@pytest.fixture
def store(tmp_path):
    fs = FileStore(root=tmp_path)
    fs.create_session("s1")
    return fs


def _app(store):
    server = MagicMock()
    server.agent_bridge = None
    server.gently_store = store
    app = FastAPI()
    app.include_router(sessions_routes.create_router(server))
    app.dependency_overrides[auth.require_control] = lambda: True
    return TestClient(app)


def test_the_route_opens_the_sessions_own_folder(store, monkeypatch):
    opened = []
    monkeypatch.setattr(sessions_routes, "_open_in_file_manager", lambda p: opened.append(p))
    r = _app(store).post("/api/sessions/s1/open-folder")
    assert r.status_code == 200, r.text
    assert opened == [store._session_dir("s1")]
    assert r.json()["path"] == str(store._session_dir("s1"))


def test_an_unknown_session_is_404_and_opens_nothing(store, monkeypatch):
    opened = []
    monkeypatch.setattr(sessions_routes, "_open_in_file_manager", lambda p: opened.append(p))
    r = _app(store).post("/api/sessions/nope/open-folder")
    assert r.status_code == 404 and opened == []


def test_a_file_manager_that_fails_is_a_502(store, monkeypatch):
    def boom(p):
        raise OSError("no desktop")

    monkeypatch.setattr(sessions_routes, "_open_in_file_manager", boom)
    r = _app(store).post("/api/sessions/s1/open-folder")
    assert r.status_code == 502 and "no desktop" in r.json()["detail"]


def test_no_store_is_503():
    r = _app(None).post("/api/sessions/s1/open-folder")
    assert r.status_code == 503


def test_the_route_needs_control():
    server = MagicMock()
    server.agent_bridge = None
    server.gently_store = None
    app = FastAPI()
    app.include_router(sessions_routes.create_router(server))
    r = TestClient(app).post("/api/sessions/s1/open-folder")
    assert r.status_code == 403, "anyone who can view could pop windows on the rig"


def test_the_opener_speaks_each_os():
    # The opening is done in one place now, for every button that opens.
    from gently.core import reveal

    src = Path(reveal.__file__).read_text(encoding="utf-8")
    fn = src[src.index("def open_folder") :][:500]
    assert "os.startfile" in fn and '"open"' in fn and '"xdg-open"' in fn
    here = Path(sessions_routes.__file__).read_text(encoding="utf-8")
    assert "reveal.open_folder(path)" in here


def test_the_header_has_the_button_beside_copy():
    copy = HEADER.index('id="session-copy-btn"')
    folder = HEADER.index('id="session-open-btn"')
    assert folder > copy, "the folder button sits after the copy button"
    assert 'onclick="openSessionFolder()"' in HEADER
    assert HEADER.index("</div>", folder) > folder, "inside the same container"


def test_the_button_posts_to_the_route_and_says_the_path():
    fn = APP_JS[APP_JS.index("async function openSessionFolder()") :][:1400]
    assert "/open-folder`" in fn and "method: 'POST'" in fn
    assert "say(`Opened ${data.path}`)" in fn
    assert "Could not open the session folder" in fn


def test_the_button_hides_with_no_session():
    assert ".session-id-link:empty ~ .session-open-btn" in MAIN_CSS
