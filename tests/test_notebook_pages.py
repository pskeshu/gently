"""Pages of the notebook: a saved query with a conversation on it.

The notes stay one flat pool; a page holds a query and its talk. Built-ins
need no setup and get a file only once talked on. Asking on a page keeps
both the question and the reply; a reply can be kept as a proposed finding
resting on the notes it cited; a line written from a page is filed where
the page is.
"""

from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from gently.harness.memory.notebook import Author, Note, NoteKind, NoteStatus
from gently.harness.memory.notebook_ask import build_ask_messages
from gently.harness.memory.notebook_pages import (
    Page,
    PageStore,
    Turn,
    builtin_pages,
    clean_query,
    list_pages,
    page_notes,
    pages_of,
    resolve_page,
)
from gently.ui.web import auth


class _CS:
    """A context store as the pages see it: a notebook, and root campaigns."""

    def __init__(self, nb, roots=()):
        self.notebook = nb
        self._roots = list(roots)

    def get_root_campaigns(self, status="active"):
        return self._roots


def _seed(nb):
    nb.write_note(Note(id="o1", kind=NoteKind.OBSERVATION, body="ring formed", threads=["t1"]))
    nb.write_note(
        Note(
            id="f1",
            kind=NoteKind.FINDING,
            body="rings by comma",
            author=Author.AGENT,
            status=NoteStatus.PROPOSED,
            projects=["p1"],
        )
    )
    nb.write_note(
        Note(id="a1", kind=NoteKind.OBSERVATION, body="@ryan look", mentions=["ryan"], by="kes")
    )
    nb.write_note(
        Note(
            id="s1",
            kind=NoteKind.OBSERVATION,
            body="4 embryos…",
            author=Author.AGENT,
            title="Session summary",
            sessions=["sess1"],
        )
    )
    return nb


class TestQueryAndBuiltins:
    def test_clean_query_keeps_only_facets(self):
        q = clean_query({"kind": " finding ", "asks": 1, "bogus": "x", "thread": "", "title": None})
        assert q == {"kind": "finding", "asks": True}
        assert clean_query("nope") == {}

    def test_builtins_are_fixed_pages_plus_one_per_project(self):
        cs = _CS(None, roots=[SimpleNamespace(id="p1", shorthand="hatch-50", description="…")])
        ids = [p.id for p in builtin_pages(cs)]
        assert ids[:4] == ["all", "agent-findings", "asks", "session-summaries"]
        assert "project:p1" in ids
        assert all(p.builtin for p in builtin_pages(cs))

    def test_page_notes_answers_each_builtin(self, file_context_store):
        nb = _seed(file_context_store.notebook)
        cs = _CS(nb, roots=[SimpleNamespace(id="p1", shorthand=None, description="Proj")])
        by = {p.id: {n.id for n in page_notes(nb, p)} for p in builtin_pages(cs)}
        assert by["all"] == {"o1", "f1", "a1", "s1"}
        assert by["agent-findings"] == {"f1"}
        assert by["asks"] == {"a1"}
        assert by["session-summaries"] == {"s1"}
        assert by["project:p1"] == {"f1"}

    def test_a_thread_page_is_built_from_its_id(self, file_context_store):
        nb = _seed(file_context_store.notebook)
        page = resolve_page(_CS(nb), nb, "thread:t1")
        assert page is not None and page.builtin and page.query == {"thread": "t1"}
        assert {n.id for n in page_notes(nb, page)} == {"o1"}
        assert resolve_page(_CS(nb), nb, "nope") is None

    def test_a_note_set_aside_leaves_every_page(self, file_context_store):
        nb = _seed(file_context_store.notebook)
        nb.set_in_notebook("o1", False)
        page = resolve_page(_CS(nb), nb, "all")
        assert "o1" not in {n.id for n in page_notes(nb, page)}


class TestPageStore:
    def test_round_trip_with_turns(self, tmp_path):
        store = PageStore(tmp_path / "pages")
        page = Page(id="", name="Hatching", query={"thread": "t1"}, by="kes")
        page.turns.append(Turn(role="person", text="what formed?", by="kes"))
        page.turns.append(
            Turn(
                role="agent",
                text="A ring.",
                note_ids=["o1"],
                points=[{"text": "ring", "note_ids": ["o1"]}],
                coverage="covered",
            )
        )
        pid = store.save(page)
        back = store.get(pid)
        assert back is not None and back.name == "Hatching" and back.query == {"thread": "t1"}
        assert [t.role for t in back.turns] == ["person", "agent"]
        assert back.turns[1].note_ids == ["o1"] and back.turns[1].coverage == "covered"
        assert store.delete(pid) is True and store.get(pid) is None

    def test_a_builtin_id_with_a_colon_has_a_file(self, tmp_path):
        store = PageStore(tmp_path / "pages")
        store.save(Page(id="project:p1", name="Proj", query={"project": "p1"}, builtin=True))
        assert store.get("project:p1") is not None
        assert list((tmp_path / "pages").glob("project__p1_*.yaml"))

    def test_list_pages_merges_saved_conversations_into_builtins(self, file_context_store):
        nb = _seed(file_context_store.notebook)
        cs = _CS(nb)
        talked = Page(id="asks", name="Asks", query={"asks": True}, builtin=True)
        talked.turns.append(Turn(role="person", text="hi"))
        pages_of(nb).save(talked)
        pages_of(nb).save(Page(id="", name="Mine", query={"kind": "finding"}))
        pages_of(nb).save(Page(id="thread:t1", name="# t1", query={"thread": "t1"}, builtin=True))
        out = list_pages(cs, nb)
        ids = [p.id for p in out]
        assert ids[:4] == ["all", "agent-findings", "asks", "session-summaries"]
        assert len(next(p for p in out if p.id == "asks").turns) == 1
        assert "Mine" in [p.name for p in out]
        assert not any(i.startswith("thread:") for i in ids)  # threads list from the notes


class TestHistoryInTheAsk:
    def test_turns_precede_the_notes_and_alternate(self):
        notes = [Note(id="o1", kind=NoteKind.OBSERVATION, body="ring formed")]
        history = [
            {"role": "person", "text": "what formed?"},
            {"role": "agent", "text": "A ring."},
            {"role": "person", "text": "when?"},
            {"role": "person", "text": "roughly"},
        ]
        msgs = build_ask_messages("and then?", notes, history)
        assert [m["role"] for m in msgs] == ["user", "assistant", "user"]
        assert msgs[0]["content"] == "what formed?"
        assert msgs[1]["content"] == "A ring."
        assert msgs[2]["content"].startswith("when?\n\nroughly\n\nNotebook entries:")
        assert "Question: and then?" in msgs[2]["content"]

    def test_no_history_is_the_plain_ask(self):
        notes = [Note(id="o1", kind=NoteKind.OBSERVATION, body="ring formed")]
        msgs = build_ask_messages("what formed?", notes)
        assert len(msgs) == 1 and msgs[0]["role"] == "user"


# ---- the API ----------------------------------------------------------------


class _Block:
    def __init__(self, inp):
        self.type, self.input = "tool_use", inp


class _Resp:
    def __init__(self, inp):
        self.content, self.stop_reason = [_Block(inp)], "tool_use"


class _AskClient:
    def __init__(self, inp):
        self.calls = []
        self._inp = inp
        client = self

        class _M:
            async def create(self, **kw):
                client.calls.append(kw)
                return _Resp(client._inp)

        self.messages = _M()


def _app(cs, client=None):
    from gently.ui.web.routes.notebook import create_router as notes_router
    from gently.ui.web.routes.notebook_pages import create_router as pages_router

    app = FastAPI()
    server = SimpleNamespace(context_store=cs, claude_async=client)
    app.include_router(notes_router(server))
    app.include_router(pages_router(server))
    app.dependency_overrides[auth.require_control] = lambda: True
    return TestClient(app)


REPLY = {
    "answer": "A ring formed, then rings by comma.",
    "points": [{"text": "ring formed", "note_ids": ["o1"]}],
    "suggested_next": [],
    "coverage": "covered",
}


class TestPagesApi:
    def test_list_pages_with_counts(self, file_context_store):
        nb = _seed(file_context_store.notebook)
        c = _app(_CS(nb))
        data = c.get("/api/notebook/pages").json()
        assert data["available"] is True
        by = {p["id"]: p for p in data["pages"]}
        assert by["asks"]["count"] == 1 and by["asks"]["turn_count"] == 0
        assert "turns" not in by["all"]

    def test_save_get_and_delete_a_page(self, file_context_store):
        nb = _seed(file_context_store.notebook)
        c = _app(_CS(nb))
        r = c.post("/api/notebook/pages", json={"name": "Rings", "query": {"thread": "t1", "x": 1}})
        assert r.status_code == 200
        page = r.json()["page"]
        assert page["query"] == {"thread": "t1"} and page["builtin"] is False
        got = c.get(f"/api/notebook/pages/{page['id']}").json()
        assert {n["id"] for n in got["notes"]} == {"o1"}
        assert c.delete(f"/api/notebook/pages/{page['id']}").json()["ok"] is True
        assert c.get(f"/api/notebook/pages/{page['id']}").status_code == 404
        assert c.post("/api/notebook/pages", json={"name": " "}).status_code == 400

    def test_ask_on_a_page_keeps_both_turns_and_sends_history(self, file_context_store):
        nb = _seed(file_context_store.notebook)
        client = _AskClient(REPLY)
        c = _app(_CS(nb), client)
        c.cookies.set("gently_operator", "kes")
        r = c.post("/api/notebook/pages/thread:t1/ask", json={"question": "what formed?"})
        assert r.status_code == 200 and r.json()["ok"] is True
        turns = r.json()["page"]["turns"]
        assert [t["role"] for t in turns] == ["person", "agent"]
        assert turns[0]["text"] == "what formed?" and turns[0]["by"] == "kes"
        assert turns[1]["note_ids"] == ["o1"] and turns[1]["coverage"] == "covered"
        # The second question carries the first exchange.
        c.post("/api/notebook/pages/thread:t1/ask", json={"question": "and then?"})
        msgs = client.calls[-1]["messages"]
        assert [m["role"] for m in msgs] == ["user", "assistant", "user"]
        # The built-in thread page now has a file, and the rail knows its talk.
        assert pages_of(nb).get("thread:t1") is not None
        assert len(c.get("/api/notebook/pages/thread:t1").json()["page"]["turns"]) == 4

    def test_ask_failure_says_why_and_keeps_no_turn(self, file_context_store):
        nb = _seed(file_context_store.notebook)

        class _Broken:
            class messages:
                @staticmethod
                async def create(**kw):
                    raise RuntimeError("no API key")

        c = _app(_CS(nb), _Broken())
        r = c.post("/api/notebook/pages/all/ask", json={"question": "hi"})
        assert r.status_code == 200 and r.json()["ok"] is False
        assert "no API key" in r.json()["error"]
        assert r.json()["page"]["turns"] == []

    def test_keep_files_the_reply_as_a_proposed_finding_on_the_page(self, file_context_store):
        nb = _seed(file_context_store.notebook)
        cs = _CS(nb, roots=[SimpleNamespace(id="p1", shorthand="proj", description="")])
        c = _app(cs, _AskClient(REPLY))
        c.post("/api/notebook/pages/project:p1/ask", json={"question": "what formed?"})
        r = c.post("/api/notebook/pages/project:p1/turns/1/keep")
        assert r.status_code == 200
        note = r.json()["note"]
        assert (
            note["kind"] == "finding" and note["status"] == "proposed" and note["author"] == "agent"
        )
        assert note["basis"] == ["o1"] and note["projects"] == ["p1"]
        assert r.json()["page"]["turns"][1]["kept_as"] == note["id"]
        # It is on the page now, and keeping again is the same note.
        assert note["id"] in {n["id"] for n in r.json()["notes"]}
        again = c.post("/api/notebook/pages/project:p1/turns/1/keep").json()
        assert again["note"]["id"] == note["id"]
        assert c.post("/api/notebook/pages/project:p1/turns/0/keep").status_code == 404

    def test_a_line_written_from_a_page_is_filed_where_the_page_is(self, file_context_store):
        nb = _seed(file_context_store.notebook)
        c = _app(_CS(nb))
        c.cookies.set("gently_operator", "kes")
        r = c.post("/api/notebook/pages/thread:t1/notes", json={"body": "ring at 10 ms"})
        assert r.status_code == 200
        note = r.json()["note"]
        assert note["threads"] == ["t1"] and note["by"] == "kes" and note["author"] == "human"
        assert note["id"] in {n["id"] for n in r.json()["notes"]}
        assert c.post("/api/notebook/pages/thread:t1/notes", json={"body": " "}).status_code == 400

    def test_clear_and_delete_on_a_builtin_only_drop_the_talk(self, file_context_store):
        nb = _seed(file_context_store.notebook)
        c = _app(_CS(nb), _AskClient(REPLY))
        c.post("/api/notebook/pages/asks/ask", json={"question": "who is asked?"})
        assert pages_of(nb).get("asks") is not None
        r = c.post("/api/notebook/pages/asks/turns/clear")
        assert r.json()["page"]["turns"] == [] and pages_of(nb).get("asks") is None
        c.post("/api/notebook/pages/asks/ask", json={"question": "again"})
        assert c.delete("/api/notebook/pages/asks").json() == {"ok": True, "builtin": True}
        assert c.get("/api/notebook/pages/asks").status_code == 200  # the page is still there

    def test_unknown_page_is_404_and_no_store_is_503(self, file_context_store):
        c = _app(_CS(_seed(file_context_store.notebook)))
        assert c.get("/api/notebook/pages/nope").status_code == 404
        assert c.post("/api/notebook/pages/nope/ask", json={"question": "x"}).status_code == 404
        none = _app(SimpleNamespace(notebook=None))
        assert none.get("/api/notebook/pages").json() == {"available": False, "pages": []}
        assert none.get("/api/notebook/pages/all").status_code == 503
