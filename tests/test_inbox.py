"""An @ in a line is an ask, and the person asked finds it in their inbox.

"i want to be able to tag into session to ask for annotation from another
user, and the other user should get a notification system with all that
tasks." An ask is a notebook note with the person in ``mentions``; the inbox
is derived from the notes; done is what the person then wrote on the same
session, a line or the tool note that saving marks writes.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from gently.core.file_store import FileStore
from gently.harness.memory.file_store import FileContextStore
from gently.harness.memory.notebook import Author, Note, NoteKind, mentions_in
from gently.ui.web import auth
from gently.ui.web.routes import inbox as inbox_routes
from gently.ui.web.routes import notebook as notebook_routes


def test_mentions_are_the_people_the_rig_knows():
    people = ["ryan", "hari-shroff", "kesavan"]
    assert mentions_in("@ryan can you mark e02? cc @Hari-Shroff", people) == ["ryan", "hari-shroff"]
    assert mentions_in("mail me at x@ryan.org, and @nobody", people) == []
    assert mentions_in("@ryan, then @ryan again", people) == ["ryan"]
    assert mentions_in("", people) == []


def _client(tmp_path, monkeypatch, me):
    store = FileStore(tmp_path)
    cs = FileContextStore(tmp_path / "agent")
    server = MagicMock()
    server.context_store = cs
    server.gently_store = store
    server.agent_bridge.agent.store = store
    monkeypatch.setattr(inbox_routes, "known_people", lambda _server: ["ryan", "kesavan"])
    app = FastAPI()
    app.include_router(notebook_routes.create_router(server))
    app.include_router(inbox_routes.create_router(server))
    app.dependency_overrides[auth.require_control] = lambda: True
    c = TestClient(app)
    c.cookies.set("gently_operator", me)
    return c, store, cs


def test_an_ask_reaches_the_inbox_and_is_done_once_answered_on_the_session(tmp_path, monkeypatch):
    c, store, cs = _client(tmp_path, monkeypatch, "kesavan")
    store.create_session("s1", name="monday")
    r = c.post(
        "/api/notebook/notes",
        json={"body": "@ryan can you mark twitch to hatch on fields 1-3?", "sessions": ["s1"]},
    )
    assert r.status_code == 200 and r.json()["note"]["mentions"] == ["ryan"]
    ask = r.json()["note"]["id"]
    # kesavan asked; nothing is for kesavan
    assert c.get("/api/inbox").json()["new"] == []

    # ryan: new, with the session it was written on
    c.cookies.set("gently_operator", "ryan")
    box = c.get("/api/inbox").json()
    assert box["me"] == "ryan" and [i["note"]["id"] for i in box["new"]] == [ask]
    assert box["new"][0]["session"] == {"id": "s1", "name": "monday"}

    # opening it: seen, so open rather than new, and the seen file is on disk
    box = c.post("/api/inbox/seen", json={"ids": [ask]}).json()
    assert [i["note"]["id"] for i in box["open"]] == [ask] and box["new"] == []
    assert (tmp_path / "agent" / "people" / "ryan" / "seen.yaml").is_file()

    # a fresh read, as after a restart, says the same
    assert [i["note"]["id"] for i in c.get("/api/inbox").json()["open"]] == [ask]

    # ryan writes on that session (what saving marks writes): done, derived
    cs.notebook.write_note(
        Note(
            id="",
            kind=NoteKind.OBSERVATION,
            body="Marked on 3 embryos: 350 to 372 min, median 361.",
            author=Author.HUMAN,
            by="ryan",
            title="Twitching to hatching",
            sessions=["s1"],
        )
    )
    box = c.get("/api/inbox").json()
    assert box["open"] == [] and [i["note"]["id"] for i in box["done"]] == [ask]

    # the session's stream carries the ask with its mention
    notes = c.get("/api/notebook/notes?session=s1").json()["notes"]
    assert any(n["id"] == ask and n["mentions"] == ["ryan"] for n in notes)


def test_a_line_that_names_nobody_is_for_nobody(tmp_path, monkeypatch):
    c, _store, _cs = _client(tmp_path, monkeypatch, "kesavan")
    assert (
        c.post("/api/notebook/notes", json={"body": "embryo 3 late"}).json()["note"]["mentions"]
        == []
    )
    c.cookies.set("gently_operator", "ryan")
    assert c.get("/api/inbox").json() == {"me": "ryan", "new": [], "open": [], "done": []}


def test_an_ask_with_no_place_is_done_once_opened(tmp_path, monkeypatch):
    c, _store, _cs = _client(tmp_path, monkeypatch, "kesavan")
    ask = c.post("/api/notebook/notes", json={"body": "@ryan see the new protocol"}).json()["note"][
        "id"
    ]
    c.cookies.set("gently_operator", "ryan")
    assert [i["note"]["id"] for i in c.get("/api/inbox").json()["new"]] == [ask]
    box = c.post("/api/inbox/seen", json={"ids": [ask]}).json()
    assert (
        box["new"] == [] and box["open"] == [] and [i["note"]["id"] for i in box["done"]] == [ask]
    )
