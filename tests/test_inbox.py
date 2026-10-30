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
from gently.harness.memory.notebook import Author, Note, NoteKind, NoteStatus, mentions_in
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
    assert c.get("/api/inbox").json() == {
        "me": "ryan",
        "asks": [],
        "new": [],
        "open": [],
        "done": [],
        "mine": [],
    }


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


def test_your_own_lines_are_one_list_with_where_and_whether_answered(tmp_path, monkeypatch):
    """ "if I have left comments on different places, i should have a unified
    interface to see where I have made my recent social activities." """
    c, store, cs = _client(tmp_path, monkeypatch, "kesavan")
    store.create_session("s1", name="monday")
    proj = cs.create_campaign("Does LED dose change time to hatching?", shorthand="dose")
    ask = c.post(
        "/api/notebook/notes",
        json={"body": "@ryan can you mark fields 1 to 3?", "sessions": ["s1"]},
    ).json()["note"]["id"]
    plain = c.post(
        "/api/notebook/notes", json={"body": "dim at dose 2", "projects": [proj]}
    ).json()["note"]["id"]
    mine = c.get("/api/inbox").json()["mine"]
    assert [i["note"]["id"] for i in mine] == [plain, ask]  # newest first
    assert mine[1]["session"] == {"id": "s1", "name": "monday"} and mine[1]["answered_by"] == []
    assert mine[0]["project"] == {"id": proj, "name": "dose"} and "answered_by" not in mine[0]
    # ryan answers on the session: the ask says so
    c.cookies.set("gently_operator", "ryan")
    c.post("/api/notebook/notes", json={"body": "done, all three", "sessions": ["s1"]})
    c.cookies.set("gently_operator", "kesavan")
    mine = c.get("/api/inbox").json()["mine"]
    assert next(i for i in mine if i["note"]["id"] == ask)["answered_by"] == ["ryan"]
    # the notebook route lists a person's lines too
    by = c.get("/api/notebook/notes?by=kesavan").json()["notes"]
    assert sorted(n["id"] for n in by) == sorted([ask, plain])
    assert c.get("/api/notebook/notes?by=ryan").json()["notes"][0]["body"] == "done, all three"


def test_an_older_line_touched_later_does_not_answer_a_newer_ask(tmp_path, monkeypatch):
    """A plain line answers only if it was written after the ask; a status
    change on an old line is not an answer. A tool note is rewritten in place
    on every save, so its latest save is what counts."""
    from datetime import datetime, timedelta

    c, store, cs = _client(tmp_path, monkeypatch, "kesavan")
    store.create_session("s1", name="monday")
    nb = cs.notebook
    old = nb.write_note(
        Note(
            id="",
            kind=NoteKind.OBSERVATION,
            body="an old line of ryan's",
            author=Author.HUMAN,
            by="ryan",
            sessions=["s1"],
            created_at=datetime.now() - timedelta(days=2),
        )
    )
    ask = c.post(
        "/api/notebook/notes", json={"body": "@ryan can you look?", "sessions": ["s1"]}
    ).json()["note"]["id"]
    n = nb.get_note(old)
    n.status = NoteStatus.CONFIRMED
    nb.write_note(n)  # touched now: updated_at moves, created_at does not
    c.cookies.set("gently_operator", "ryan")
    assert [i["note"]["id"] for i in c.get("/api/inbox").json()["new"]] == [ask]
    # a tool note from before the ask, saved again after it, does answer
    tool = nb.write_note(
        Note(
            id="",
            kind=NoteKind.OBSERVATION,
            body="Marked on 1 embryo: 60 min.",
            author=Author.HUMAN,
            by="ryan",
            title="Twitching to hatching",
            sessions=["s1"],
            created_at=datetime.now() - timedelta(days=1),
        )
    )
    nb.write_note(nb.get_note(tool))
    assert [i["note"]["id"] for i in c.get("/api/inbox").json()["done"]] == [ask]


def test_the_asks_are_one_list_in_time_order_with_seen_and_answered_flags(tmp_path, monkeypatch):
    """ "i do not like this open/new model rather the notification is listed in
    order, and the ones that were opened appear different than the new one"."""
    c, store, cs = _client(tmp_path, monkeypatch, "kesavan")
    store.create_session("s1", name="monday")
    first = c.post(
        "/api/notebook/notes", json={"body": "@ryan first ask", "sessions": ["s1"]}
    ).json()["note"]["id"]
    second = c.post(
        "/api/notebook/notes", json={"body": "@ryan second ask", "sessions": ["s1"]}
    ).json()["note"]["id"]
    c.cookies.set("gently_operator", "ryan")
    asks = c.get("/api/inbox").json()["asks"]
    assert [a["note"]["id"] for a in asks] == [second, first]  # newest first, both unopened
    assert [a["seen"] for a in asks] == [False, False] and [a["answered"] for a in asks] == [
        False,
        False,
    ]
    asks = c.post("/api/inbox/seen", json={"ids": [first]}).json()["asks"]
    assert [a["note"]["id"] for a in asks] == [second, first]  # opened, and still in its place
    assert [a["seen"] for a in asks] == [False, True]
    c.post("/api/notebook/notes", json={"body": "done both", "sessions": ["s1"]})
    asks = c.get("/api/inbox").json()["asks"]
    assert [a["answered"] for a in asks] == [True, True] and [a["seen"] for a in asks] == [
        False,
        True,
    ]
