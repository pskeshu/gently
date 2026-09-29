"""Nobody is an anonymous eagle.

"on the presence indicator, it is not nice to see ourselves as an anonymous
eagle or something like that"

A browser that had given no name was called "Anonymous" and an animal picked
from its id. It said nothing about who was watching, and it was what the
operator at the microscope was called too.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from gently.core import reveal
from gently.ui.web import auth
from gently.ui.web.routes import auth_routes

WEB = Path(__file__).resolve().parents[1] / "gently" / "ui" / "web"
APP_JS = (WEB / "static" / "js" / "app.js").read_text(encoding="utf-8")
CHAT_JS = (WEB / "static" / "js" / "agent-chat.js").read_text(encoding="utf-8")
PRESENCE = APP_JS[
    APP_JS.index("const PresenceManager = {") : APP_JS.index("const ThemeManager = {")
]


class TestWhatTheServerCallsThem:
    def test_the_microscopes_own_computer(self):
        assert auth.unnamed("127.0.0.1") == "At the microscope"
        assert auth.unnamed("::1") == "At the microscope"

    def test_another_computer(self):
        assert auth.unnamed("203.0.113.9") == "Guest"

    def test_nothing_known_is_a_guest(self):
        assert auth.unnamed(None) == "Guest"

    def _me(self, monkeypatch, local, store=None, username=None):
        monkeypatch.setattr(reveal, "is_local", lambda host: local)
        monkeypatch.setattr(auth_routes, "get_account_store", lambda: store)
        monkeypatch.setattr(auth_routes, "current_username", lambda request: username)
        app = FastAPI()
        app.include_router(auth_routes.create_router(MagicMock()))
        return TestClient(app).get("/api/auth/me").json()

    def test_me_says_where_this_browser_is_without_accounts(self, monkeypatch):
        me = self._me(monkeypatch, local=True)
        assert me["accounts"] is False
        assert me["local"] is True and me["unnamed"] == "At the microscope"

    def test_and_for_a_guest(self, monkeypatch):
        me = self._me(monkeypatch, local=False)
        assert me["local"] is False and me["unnamed"] == "Guest"

    def test_and_when_signed_in(self, monkeypatch):
        store = MagicMock()
        store.has_users.return_value = True
        store.get_role.return_value = "operator"
        me = self._me(monkeypatch, local=True, store=store, username="ryan")
        assert me["username"] == "ryan" and me["local"] is True

    def test_the_chat_and_the_connection_use_the_same_names(self):
        ws = (WEB / "routes" / "agent_ws.py").read_text(encoding="utf-8")
        assert "client_label = username or unnamed(" in ws
        cm = (WEB / "connection_manager.py").read_text(encoding="utf-8")
        assert "name = unnamed(" in cm and "Anonymous" not in cm


class TestThePage:
    def test_no_animals(self):
        assert "ANIMALS" not in APP_JS and "getAnonymousName" not in APP_JS
        for animal in ("Koala", "Eagle", "Meerkat"):
            assert animal not in PRESENCE.replace("Anonymous Eagle", "")

    def test_a_name_somebody_gave_wins(self):
        init = PRESENCE[PRESENCE.index("    init() {") :][:1400]
        assert "this.chosen = !!saved && !/^anonymous\\b/i.test(saved);" in init
        assert "if (!this.chosen) this.name = this.unnamedName();" in init

    def test_the_account_then_the_place(self):
        fn = PRESENCE[PRESENCE.index("    unnamedName() {") :][:400]
        assert fn.index("me.username") < fn.index("me.unnamed")

    def test_nobody_is_announced_before_the_server_has_said_who_they_are(self):
        fn = PRESENCE[PRESENCE.index("    sendJoin() {") :][:700]
        assert "(this.ready || Promise.resolve()).then(join, join);" in fn

    def test_blank_goes_back_to_what_the_server_says(self):
        fn = PRESENCE[PRESENCE.index("    showNamePrompt() {") :][:1100]
        assert "localStorage.removeItem('gently-user-name');" in fn
        assert "this.name = unnamed;" in fn
        # and the name it goes back to is not saved as if it had been chosen
        blank = fn[fn.index("if (newName.trim() === '')") : fn.index("} else {")]
        assert "setName(" not in blank and "this.announce();" in blank

    def test_whoever_is_at_the_microscope_is_shown_as_it(self):
        assert "if (client.name === this.AT_THE_RIG) {" in PRESENCE
        assert "avatar.innerHTML = this.RIG_ICON;" in PRESENCE

    def test_a_name_is_never_put_in_as_html(self):
        render = PRESENCE[PRESENCE.index("    render() {") : PRESENCE.index("    RIG_ICON:")]
        assert render.count("innerHTML") == 2, "the container is cleared, the icon is ours"
        assert "avatar.textContent = this.getInitials(client.name);" in render

    def test_the_chat_calls_an_unnamed_author_a_guest(self):
        fn = CHAT_JS[CHAT_JS.index("    function displayAuthor(author) {") :][:500]
        assert "return 'Anonymous'" not in fn
        assert "if (/^anonymous\\b/i.test(author)) return 'Guest';" in fn

    @pytest.mark.parametrize(
        "me,saved,expected",
        [
            ({"local": True, "unnamed": "At the microscope"}, None, "At the microscope"),
            ({"local": False, "unnamed": "Guest"}, None, "Guest"),
            ({"authenticated": True, "username": "ryan", "local": True}, None, "ryan"),
            ({"local": True, "unnamed": "At the microscope"}, "Kesavan", "Kesavan"),
            (
                {"local": True, "unnamed": "At the microscope"},
                "Anonymous Eagle",
                "At the microscope",
            ),
            (None, None, "Guest"),
        ],
    )
    def test_what_a_browser_is_called(self, me, saved, expected, tmp_path):
        node = shutil.which("node")
        if node is None:
            pytest.skip("node is not installed")
        script = tmp_path / "presence.js"
        script.write_text(
            "const store = " + json.dumps({"gently-user-name": saved} if saved else {}) + ";\n"
            "const localStorage = { getItem: k => (k in store ? store[k] : null),"
            " setItem: (k, v) => { store[k] = v; }, removeItem: k => { delete store[k]; } };\n"
            "const ClientEventBus = { on() {} };\n"
            "const state = {};\n"
            "const fetch = () => Promise.resolve({ ok: "
            + ("true" if me is not None else "false")
            + ", json: () => Promise.resolve("
            + json.dumps(me)
            + ") });\n"
            + PRESENCE
            + "\nPresenceManager.init();\n"
            "const say = () => console.log(JSON.stringify(PresenceManager.name));\n"
            "PresenceManager.ready.then(say);\n",
            encoding="utf-8",
        )
        out = subprocess.run([node, str(script)], capture_output=True, text=True, timeout=30)
        assert out.returncode == 0, out.stderr
        assert json.loads(out.stdout.strip()) == expected
