"""Tests for the notebook read API."""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from gently.harness.memory.notebook import Note, NoteKind, NoteStatus


def _make_app(context_store):
    from gently.ui.web.routes.notebook import create_router

    app = FastAPI()

    class _Server:
        pass

    server = _Server()
    server.context_store = context_store
    app.include_router(create_router(server))
    return app


def _seed(cs):
    nb = cs.notebook
    nb.write_note(Note(id="o1", kind=NoteKind.OBSERVATION, body="ring formed", strains=["N2"]))
    nb.write_note(Note(id="f1", kind=NoteKind.FINDING, body="rings by comma",
                       status=NoteStatus.PROPOSED, strains=["N2"], threads=["t1"]))
    return cs


class TestListNotes:
    def test_no_store_available_false(self):
        client = TestClient(_make_app(None))
        data = client.get("/api/notebook/notes").json()
        assert data == {"available": False, "notes": []}

    def test_list_all(self, file_context_store):
        client = TestClient(_make_app(_seed(file_context_store)))
        data = client.get("/api/notebook/notes").json()
        assert data["available"] is True
        assert {n["id"] for n in data["notes"]} == {"o1", "f1"}

    def test_filter_by_kind(self, file_context_store):
        client = TestClient(_make_app(_seed(file_context_store)))
        data = client.get("/api/notebook/notes?kind=finding").json()
        assert {n["id"] for n in data["notes"]} == {"f1"}

    def test_filter_by_strain(self, file_context_store):
        client = TestClient(_make_app(_seed(file_context_store)))
        data = client.get("/api/notebook/notes?strain=N2").json()
        assert {n["id"] for n in data["notes"]} == {"o1", "f1"}

    def test_invalid_kind_is_ignored(self, file_context_store):
        client = TestClient(_make_app(_seed(file_context_store)))
        data = client.get("/api/notebook/notes?kind=bogus").json()
        assert {n["id"] for n in data["notes"]} == {"o1", "f1"}


class TestGetNote:
    def test_get_existing(self, file_context_store):
        client = TestClient(_make_app(_seed(file_context_store)))
        data = client.get("/api/notebook/notes/o1").json()
        assert data["id"] == "o1"
        assert data["body"] == "ring formed"

    def test_get_missing_404(self, file_context_store):
        client = TestClient(_make_app(_seed(file_context_store)))
        resp = client.get("/api/notebook/notes/nope")
        assert resp.status_code == 404


class TestThreads:
    def test_no_store(self):
        client = TestClient(_make_app(None))
        assert client.get("/api/notebook/threads").json() == {"available": False, "threads": []}

    def test_thread_counts(self, file_context_store):
        cs = file_context_store
        nb = cs.notebook
        nb.write_note(Note(id="a", kind=NoteKind.QUESTION, body="q", threads=["t1"]))
        nb.write_note(Note(id="b", kind=NoteKind.FINDING, body="f", threads=["t1", "t2"]))
        client = TestClient(_make_app(cs))
        data = client.get("/api/notebook/threads").json()
        assert data["available"] is True
        assert data["threads"] == [{"id": "t1", "count": 2}, {"id": "t2", "count": 1}]


class TestLimit:
    def test_limit_returns_newest(self, file_context_store):
        client = TestClient(_make_app(_seed(file_context_store)))
        data = client.get("/api/notebook/notes?limit=1").json()
        assert len(data["notes"]) == 1  # newest-first, capped
