"""Tests for the notebook read API."""

from fastapi import FastAPI
from fastapi.testclient import TestClient

from gently.harness.memory.notebook import Note, NoteKind, NoteStatus
from gently.ui.web import auth


def _make_app(context_store):
    from gently.ui.web.routes.notebook import create_router

    app = FastAPI()

    class _Server:
        pass

    server = _Server()
    server.context_store = context_store
    app.include_router(create_router(server))
    app.dependency_overrides[auth.require_control] = lambda: True
    return app


def _seed(cs):
    nb = cs.notebook
    nb.write_note(Note(id="o1", kind=NoteKind.OBSERVATION, body="ring formed", strains=["N2"]))
    nb.write_note(
        Note(
            id="f1",
            kind=NoteKind.FINDING,
            body="rings by comma",
            status=NoteStatus.PROPOSED,
            strains=["N2"],
            threads=["t1"],
        )
    )
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


class _AskBlock:
    def __init__(self, inp):
        self.type = "tool_use"
        self.input = inp


class _AskResp:
    def __init__(self, inp):
        self.content = [_AskBlock(inp)]
        self.stop_reason = "tool_use"


class _AskMessages:
    def __init__(self, inp):
        self._inp = inp

    async def create(self, **kwargs):
        return _AskResp(self._inp)


class _AskClient:
    def __init__(self, inp):
        self.messages = _AskMessages(inp)


def _make_app_with_client(context_store, client):
    from gently.ui.web.routes.notebook import create_router

    app = FastAPI()

    class _Server:
        pass

    server = _Server()
    server.context_store = context_store
    server.claude_async = client
    app.include_router(create_router(server))
    app.dependency_overrides[auth.require_control] = lambda: True
    return app


class TestAsk:
    def test_ask_returns_grounded_answer(self, file_context_store):
        cs = _seed(file_context_store)
        canned = {
            "answer": "A ring formed.",
            "points": [{"text": "ring", "note_ids": ["o1"]}],
            "suggested_next": [],
            "coverage": "covered",
        }
        client = TestClient(_make_app_with_client(cs, _AskClient(canned)))
        resp = client.post("/api/notebook/ask", json={"question": "what happened?"})
        assert resp.status_code == 200
        assert resp.json()["coverage"] == "covered"

    def test_ask_no_store(self):
        client = TestClient(_make_app(None))
        resp = client.post("/api/notebook/ask", json={"question": "x"})
        assert resp.json() == {"available": False}

    def test_ask_api_failure_says_why_instead_of_500(self, file_context_store):
        cs = _seed(file_context_store)

        class _Broken:
            class messages:
                @staticmethod
                async def create(**kwargs):
                    raise RuntimeError("no API key")

        client = TestClient(_make_app_with_client(cs, _Broken()))
        resp = client.post("/api/notebook/ask", json={"question": "what happened?"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["available"] is False
        assert "no API key" in body["error"]


class TestDeleteAndRestore:
    def test_delete_takes_the_note_off_and_says_it_is_restorable(self, file_context_store):
        client = TestClient(_make_app(_seed(file_context_store)))
        resp = client.delete("/api/notebook/notes/o1")
        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is True and body["restorable"] is True
        assert body["note"]["id"] == "o1"
        ids = {n["id"] for n in client.get("/api/notebook/notes").json()["notes"]}
        assert ids == {"f1"}
        assert client.get("/api/notebook/notes/o1").status_code == 404

    def test_restore_brings_the_note_back_as_it_was(self, file_context_store):
        client = TestClient(_make_app(_seed(file_context_store)))
        client.delete("/api/notebook/notes/f1")
        resp = client.post("/api/notebook/notes/f1/restore")
        assert resp.status_code == 200
        note = resp.json()["note"]
        assert note["id"] == "f1" and note["status"] == "proposed" and note["threads"] == ["t1"]
        # Back on the page, and back in the thread it was filed under.
        ids = {n["id"] for n in client.get("/api/notebook/notes?thread=t1").json()["notes"]}
        assert ids == {"f1"}

    def test_unknown_ids_are_404(self, file_context_store):
        client = TestClient(_make_app(_seed(file_context_store)))
        assert client.delete("/api/notebook/notes/nope").status_code == 404
        assert client.post("/api/notebook/notes/nope/restore").status_code == 404

    def test_no_store_is_503(self):
        client = TestClient(_make_app(None))
        assert client.delete("/api/notebook/notes/o1").status_code == 503
        assert client.post("/api/notebook/notes/o1/restore").status_code == 503


class TestSetAside:
    def test_aside_leaves_the_record_and_the_notebook_view_drops_it(self, file_context_store):
        client = TestClient(_make_app(_seed(file_context_store)))
        resp = client.post("/api/notebook/notes/o1/in_notebook", json={"value": False})
        assert resp.status_code == 200
        assert resp.json()["note"]["in_notebook"] is False

        def ids(q=""):
            return {n["id"] for n in client.get(f"/api/notebook/notes{q}").json()["notes"]}

        assert ids() == {"o1", "f1"}  # the record: the stream and inbox read this
        assert ids("?shelf=notebook") == {"f1"}  # what the agent reads
        assert ids("?shelf=aside") == {"o1"}  # what was set aside
        # Threads count the shelf asked for.
        client.post("/api/notebook/notes/f1/in_notebook", json={"value": False})
        assert client.get("/api/notebook/threads?shelf=notebook").json()["threads"] == []
        assert client.get("/api/notebook/threads").json()["threads"] == [{"id": "t1", "count": 1}]

    def test_put_back(self, file_context_store):
        client = TestClient(_make_app(_seed(file_context_store)))
        client.post("/api/notebook/notes/o1/in_notebook", json={"value": False})
        resp = client.post("/api/notebook/notes/o1/in_notebook", json={"value": True})
        assert resp.json()["note"]["in_notebook"] is True
        ids = {n["id"] for n in client.get("/api/notebook/notes?shelf=notebook").json()["notes"]}
        assert ids == {"o1", "f1"}

    def test_bad_value_and_unknown_note(self, file_context_store):
        client = TestClient(_make_app(_seed(file_context_store)))
        assert (
            client.post("/api/notebook/notes/o1/in_notebook", json={"value": "no"}).status_code
            == 400
        )
        assert (
            client.post("/api/notebook/notes/zz/in_notebook", json={"value": False}).status_code
            == 404
        )

    def test_ask_reads_only_the_notebook(self, file_context_store):
        cs = _seed(file_context_store)
        cs.notebook.set_in_notebook("o1", False)
        canned = {"answer": "x", "points": [], "suggested_next": [], "coverage": "covered"}
        client = TestClient(_make_app_with_client(cs, _AskClient(canned)))
        body = client.post("/api/notebook/ask", json={"question": "what happened?"}).json()
        assert body["note_ids"] == ["f1"]
