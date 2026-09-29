"""After the launch gate comes the workspace, with nothing in between.

"there are now two steps before we go into the main gently view. we have the
opening thing - with microscope or agent on off toggle. then we have the
landing page - that we have never clicked on except to go to the workspace. so
might as well get rid of that page."
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from fastapi import FastAPI
from fastapi.templating import Jinja2Templates
from fastapi.testclient import TestClient

from gently.ui.web.routes.pages import create_router

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "gently" / "ui" / "web"
INDEX = (WEB / "templates" / "index.html").read_text(encoding="utf-8")
PAGES = (WEB / "routes" / "pages.py").read_text(encoding="utf-8")
SHELL_CSS = (WEB / "static" / "css" / "shell.css").read_text(encoding="utf-8")


def _client(gate_passed: bool):
    server = MagicMock()
    server.gate_passed = gate_passed
    server.templates = Jinja2Templates(directory=str(WEB / "templates"))
    for name in ("gently_version", "gently_build_id", "gently_build_date"):
        server.templates.env.globals[name] = "test"
    server.templates.env.globals["replay_enabled"] = False
    server.agent_bridge.agent.experiment.embryos = {}
    app = FastAPI()
    app.include_router(create_router(server))
    return TestClient(app)


def test_the_page_is_gone():
    assert "v2-landing" not in INDEX
    assert "What are we doing today" not in INDEX
    assert "Skip to workspace" not in INDEX
    assert not (WEB / "static" / "js" / "landing.js").exists()
    assert not (WEB / "static" / "css" / "landing.css").exists()
    assert "landing.js" not in INDEX and "landing.css" not in INDEX


def test_nothing_decides_whether_to_show_it():
    assert "show_landing" not in PAGES and "show_landing" not in INDEX
    assert "_resumed_at" not in PAGES
    sessions = (WEB / "routes" / "sessions.py").read_text(encoding="utf-8")
    assert "_resumed_at" not in sessions


def test_a_fresh_session_with_no_embryos_lands_in_the_workspace():
    """The case that used to get the landing page: past the gate, nothing
    resumed, nothing marked yet."""
    r = _client(gate_passed=True).get("/")
    assert r.status_code == 200
    assert 'id="home-content"' in r.text and 'class="v2-rail' in r.text
    assert "v2-landing" not in r.text


def test_the_gate_still_comes_first():
    r = _client(gate_passed=False).get("/", follow_redirects=False)
    assert r.status_code == 302 and r.headers["location"] == "/launch"


def test_nothing_in_the_app_still_reaches_for_the_page():
    for js in (WEB / "static" / "js").rglob("*.js"):
        text = js.read_text(encoding="utf-8")
        assert "v2-landing" not in text, f"{js.name} still looks for the landing page"
        assert "V2Landing" not in text, f"{js.name} still calls the landing page"


def test_the_home_greeting_stays_hidden_under_the_shell():
    """The one rule in the page's stylesheet that was not about the page."""
    assert "body.ux-v2 .home-hero { display: none; }" in SHELL_CSS
