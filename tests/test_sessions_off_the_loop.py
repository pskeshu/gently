"""The Sessions tab's requests stay off the event loop and walk once.

From the Sessions tab audit (2026-10-30): the campaigns list walked every
session once per project, on the loop, beside the Sessions list; the notebook
list parsed every note on the loop and filtered by session afterwards; Home
parsed every unfiled session's logs to count them; listing people walked the
sessions on every request.
"""

from __future__ import annotations

import builtins
from unittest.mock import MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from gently.core.file_store import FileStore
from gently.harness.memory.file_store import FileContextStore
from gently.harness.memory.notebook import Author, Note, NoteKind
from gently.ui.web import auth, operators
from gently.ui.web.routes import campaigns as campaigns_routes
from gently.ui.web.routes import notebook as notebook_routes
from gently.ui.web.routes import sessions as sessions_routes


class CountingStore:
    """A FileStore that counts its walks over the sessions."""

    def __init__(self, store):
        self._s = store
        self.walks = 0

    def list_sessions(self, *a, **k):
        self.walks += 1
        return self._s.list_sessions(*a, **k)

    def __getattr__(self, name):
        return getattr(self._s, name)


def _app(store, cs):
    server = MagicMock()
    server.gently_store = store
    server.context_store = cs
    server.agent_bridge.agent.store = store
    server.agent_bridge.agent.session_id = None
    app = FastAPI()
    for mod in (campaigns_routes, notebook_routes, sessions_routes):
        app.include_router(mod.create_router(server))
    app.dependency_overrides[auth.require_control] = lambda: True
    return TestClient(app)


def test_the_projects_list_walks_the_sessions_once_for_all_projects(tmp_path):
    store = CountingStore(FileStore(tmp_path))
    cs = FileContextStore(tmp_path / "agent")
    roots = [cs.create_campaign(f"Question {i}?", shorthand=f"q{i}") for i in range(3)]
    for i, sid in enumerate(("s1", "s2", "s3", "s4")):
        store.create_session(sid)
        store.link_session(sid, roots[i % 3], home=True)
    c = _app(store, cs)
    store.walks = 0
    trees = c.get("/api/campaigns").json()["campaigns"]
    assert store.walks == 1
    assert sorted(t["project"]["sessions"] for t in trees) == [1, 1, 2]


def test_notes_filed_to_a_session_come_from_the_index(tmp_path):
    cs = FileContextStore(tmp_path / "agent")
    nb = cs.notebook
    a = nb.write_note(
        Note(id="", kind=NoteKind.OBSERVATION, body="on s1", author=Author.HUMAN, sessions=["s1"])
    )
    nb.write_note(
        Note(id="", kind=NoteKind.OBSERVATION, body="on s2", author=Author.HUMAN, sessions=["s2"])
    )
    assert [n.id for n in nb.query_notes(session="s1")] == [a]
    assert nb.ids_for_session("s2") and not nb.ids_for_session("s9")
    again = FileContextStore(tmp_path / "agent").notebook  # a fresh index says the same
    assert [n.id for n in again.query_notes(session="s1")] == [a]
    c = _app(FileStore(tmp_path), cs)
    assert [n["id"] for n in c.get("/api/notebook/notes?session=s1").json()["notes"]] == [a]


def test_people_on_links_are_remembered_at_start_and_listing_never_walks(tmp_path, monkeypatch):
    monkeypatch.setattr(operators, "operators_path", lambda: tmp_path / "config" / "operators.yaml")
    store = FileStore(tmp_path)
    store.create_session("s1")
    store.link_session("s1", "p1", by="ryan", home=True)
    store.create_session("s2")
    store.link_session("s2", "p1", by="backfill", home=True)  # not a person
    assert operators.remember_people_on_links(store) == 1
    assert operators.remember_people_on_links(store) == 0
    walking = CountingStore(store)
    assert [u["id"] for u in operators.list_operators(walking)] == ["ryan"]
    assert walking.walks == 0


def test_the_list_skips_the_logs_when_asked_only_to_count(tmp_path, monkeypatch):
    store = FileStore(tmp_path)
    cs = FileContextStore(tmp_path / "agent")
    store.create_session("s1")
    (store._session_dir("s1") / "events.jsonl").write_text(
        '{"event_type": "WARNING_ISSUED", "timestamp": "2026-10-04T21:00:00", "data": {}}\n',
        encoding="utf-8",
    )
    c = _app(store, cs)
    opened = {"n": 0}
    real = builtins.open

    def counting(file, *a, **k):
        if str(file).endswith("events.jsonl"):
            opened["n"] += 1
        return real(file, *a, **k)

    monkeypatch.setattr(builtins, "open", counting)
    full = c.get("/api/sessions?unfiled=1").json()["sessions"]
    assert len(full) == 1 and full[0]["warnings"] == 1 and opened["n"] >= 1
    opened["n"] = 0
    light = c.get("/api/sessions?unfiled=1&detail=0").json()["sessions"]
    assert len(light) == 1 and "warnings" not in light[0] and opened["n"] == 0


def test_the_detail_carries_what_the_page_draws_and_answers_404(tmp_path):
    store = FileStore(tmp_path)
    cs = FileContextStore(tmp_path / "agent")
    store.create_session("s1", name="monday")
    c = _app(store, cs)
    d = c.get("/api/sessions/s1").json()
    assert d["name"] == "monday" and d["removed_embryos"] == [] and d["embryos"] == []
    assert not {"events", "temperature", "actions"} & set(d)
    assert c.get("/api/sessions/nope").status_code == 404
