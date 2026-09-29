"""A note's tags are tags, and an embryo is named with its session.

A note was saved with its embryos as ``e, m, b, r, y, o, _, 1, ",", " ", …``:
the assistant had passed ``"embryo_1, embryo_2, embryo_4"`` as one string, and
``list()`` of a string is its characters. The Notebook drew twenty-eight tags.

And ``embryo_1`` on its own says nothing: it is a different embryo in every
session.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import yaml
from fastapi import FastAPI
from fastapi.testclient import TestClient

from gently.harness.memory.notebook import (
    Note,
    NotebookStore,
    NoteKind,
    as_tags,
    embryo_refs,
    note_from_dict,
)

JS = (
    Path(__file__).resolve().parents[1] / "gently" / "ui" / "web" / "static" / "js" / "notebook.js"
).read_text(encoding="utf-8")

AS_SAVED = list("embryo_1, embryo_2, embryo_4")


class TestTags:
    def test_a_string_is_split_not_spelled(self):
        assert as_tags("embryo_1, embryo_2, embryo_4") == ["embryo_1", "embryo_2", "embryo_4"]

    def test_one_tag_as_a_string_is_one_tag(self):
        assert as_tags("embryo_3") == ["embryo_3"]

    def test_a_list_is_left_as_it_is(self):
        assert as_tags(["embryo_1", "embryo_2"]) == ["embryo_1", "embryo_2"]

    def test_a_list_holding_a_joined_string_is_split(self):
        assert as_tags(["embryo_1, embryo_2"]) == ["embryo_1", "embryo_2"]

    def test_the_note_already_on_disk_is_mended_when_read(self):
        assert as_tags(AS_SAVED) == ["embryo_1", "embryo_2", "embryo_4"]

    def test_nothing_is_nothing(self):
        assert as_tags(None) == []
        assert as_tags("") == []
        assert as_tags([]) == []

    def test_a_tag_is_not_repeated(self):
        assert as_tags("N2, N2 ,OH904") == ["N2", "OH904"]

    def test_two_short_strains_are_two_strains(self):
        # Not mistaken for a string taken apart.
        assert as_tags(["A", "B"]) == ["A", "B"]


class TestNote:
    def test_a_note_built_with_a_string_holds_tags(self):
        n = Note(id="a", kind=NoteKind.OBSERVATION, body="x", embryos="embryo_1, embryo_2")  # type: ignore[arg-type]
        assert n.embryos == ["embryo_1", "embryo_2"]

    def test_the_saved_shape_reads_back_whole(self):
        n = note_from_dict(
            {
                "id": "b3b2215a",
                "kind": "observation",
                "body": "drift",
                "author": "human",
                "embryos": AS_SAVED,
                "sessions": ["6a4a3d9b"],
                "created_at": "2026-07-03T00:11:50.326763",
                "updated_at": "2026-07-03T00:11:50.326763",
            }
        )
        assert n.embryos == ["embryo_1", "embryo_2", "embryo_4"]

    def test_it_is_written_as_tags(self, tmp_path):
        nb = NotebookStore(tmp_path)
        nb.write_note(
            Note(id="a1", kind=NoteKind.OBSERVATION, body="x", embryos="embryo_1,embryo_2")
        )  # type: ignore[arg-type]
        saved = yaml.safe_load(next((tmp_path / "notes").glob("a1_*.yaml")).read_text("utf-8"))
        assert saved["embryos"] == ["embryo_1", "embryo_2"]
        assert nb.ids_for_embryo("embryo_1") == ["a1"]
        assert nb.ids_for_embryo("e") == []


class TestAnEmbryoIsNamedWithItsSession:
    def _note(self, **kw):
        return Note(id="a", kind=NoteKind.OBSERVATION, body="x", **kw)

    def test_one_session_qualifies_its_embryos(self):
        refs = embryo_refs(self._note(embryos=["embryo_1", "embryo_4"], sessions=["6a4a3d9b"]))
        assert [r["label"] for r in refs] == ["6a4a3d9b/embryo_1", "6a4a3d9b/embryo_4"]
        assert refs[0] == {
            "session": "6a4a3d9b",
            "embryo": "embryo_1",
            "label": "6a4a3d9b/embryo_1",
        }

    def test_a_tag_that_carries_its_session_keeps_it(self):
        refs = embryo_refs(self._note(embryos=["851ac998/embryo_2"], sessions=["6a4a3d9b"]))
        assert refs[0]["label"] == "851ac998/embryo_2"
        assert refs[0]["session"] == "851ac998"

    def test_a_note_across_sessions_does_not_guess(self):
        refs = embryo_refs(self._note(embryos=["embryo_1"], sessions=["a", "b"]))
        assert refs[0] == {"session": None, "embryo": "embryo_1", "label": "embryo_1"}

    def test_no_session_no_qualifier(self):
        assert embryo_refs(self._note(embryos=["embryo_1"]))[0]["label"] == "embryo_1"


class TestTheRoute:
    def _client(self, tmp_path):
        from gently.ui.web.routes.notebook import create_router

        nb = NotebookStore(tmp_path)
        server = SimpleNamespace(context_store=SimpleNamespace(notebook=nb))
        app = FastAPI()
        app.include_router(create_router(server))
        return nb, TestClient(app)

    def test_the_list_carries_the_labels(self, tmp_path):
        nb, client = self._client(tmp_path)
        nb.write_note(
            Note(
                id="a1",
                kind=NoteKind.OBSERVATION,
                body="x",
                embryos=["embryo_1"],
                sessions=["6a4a3d9b"],
            )
        )
        note = client.get("/api/notebook/notes").json()["notes"][0]
        assert note["embryos"] == ["embryo_1"]
        assert note["embryo_refs"][0]["label"] == "6a4a3d9b/embryo_1"
        one = client.get("/api/notebook/notes/a1").json()
        assert one["embryo_refs"][0]["label"] == "6a4a3d9b/embryo_1"

    def test_a_note_saved_in_pieces_is_served_whole(self, tmp_path):
        nb, client = self._client(tmp_path)
        (tmp_path / "notes" / "b3b2215a_drift.yaml").write_text(
            yaml.safe_dump(
                {
                    "id": "b3b2215a",
                    "kind": "observation",
                    "body": "drift",
                    "author": "human",
                    "status": "confirmed",
                    "embryos": AS_SAVED,
                    "sessions": ["6a4a3d9b"],
                    "created_at": "2026-07-03T00:11:50.326763",
                    "updated_at": "2026-07-03T00:11:50.326763",
                }
            ),
            encoding="utf-8",
        )
        note = client.get("/api/notebook/notes/b3b2215a").json()
        assert [r["label"] for r in note["embryo_refs"]] == [
            "6a4a3d9b/embryo_1",
            "6a4a3d9b/embryo_2",
            "6a4a3d9b/embryo_4",
        ]


class TestTheTool:
    def test_record_note_takes_a_string_for_what_it(self, tmp_path):
        from gently.app.tools.memory_tools import record_note

        nb = NotebookStore(tmp_path)
        agent = SimpleNamespace(context_store=SimpleNamespace(notebook=nb), session_id="6a4a3d9b")
        fn = getattr(record_note, "__wrapped__", record_note)
        asyncio.run(
            fn(text="drift", embryos="embryo_1, embryo_2, embryo_4", context={"agent": agent})
        )
        notes = nb.query_notes()
        assert len(notes) == 1
        assert notes[0].embryos == ["embryo_1", "embryo_2", "embryo_4"]
        assert notes[0].sessions == ["6a4a3d9b"]


class TestThePage:
    def test_the_tag_is_the_label_the_server_built(self):
        assert "n.embryo_refs.map(r => r.label)" in JS

    def test_a_string_is_never_walked_as_a_list(self):
        assert "(n.embryos || []).map" not in JS
        assert "Array.isArray(n.embryos) ? n.embryos : []" in JS

    def test_a_note_without_embryos_still_says_its_session(self):
        assert "const sessions = embryos.length ? [] :" in JS
        assert "sessions.map(s => 'session ' + s)" in JS
