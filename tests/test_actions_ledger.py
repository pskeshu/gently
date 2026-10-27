"""Every control request a person makes is one line in the live session."""

from types import SimpleNamespace

from gently.core.file_store import FileStore
from gently.ui.web import accounts
from gently.ui.web.actions import is_action, read_actions, record_action
from gently.ui.web.auth import OPERATOR_COOKIE


def test_which_requests_are_actions():
    assert is_action("POST", "/api/devices/stage/move")
    assert is_action("PUT", "/api/sessions/s1/tags")
    assert not is_action("GET", "/api/sessions")
    assert not is_action("POST", "/api/auth/operator")  # identity is not an action
    assert not is_action("POST", "/api/settings/history")
    assert not is_action("POST", "/replay/ingest")


def test_an_action_is_written_under_the_live_session_with_who(tmp_path, monkeypatch):
    monkeypatch.setattr(accounts, "_store", None)
    store = FileStore(tmp_path)
    store.create_session("s1", name="s1")
    server = SimpleNamespace(_current_session_id=lambda: "s1", gently_store=store)
    req = SimpleNamespace(
        cookies={OPERATOR_COOKIE: "hari"},
        client=SimpleNamespace(host="127.0.0.1"),
        method="POST",
        url=SimpleNamespace(path="/api/devices/stage/move", query="axis=x"),
    )
    record_action(server, req, 200)
    record_action(server, req, 403)  # refused: not an action taken
    rows = read_actions(store, "s1")
    assert len(rows) == 1
    assert rows[0]["by"] == "hari" and rows[0]["path"] == "/api/devices/stage/move"
    assert rows[0]["query"] == "axis=x" and rows[0]["at"]
    assert read_actions(store, "nope") == []
