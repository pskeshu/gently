"""Tests for the shared lab notebook: Note model + NotebookStore."""

from datetime import datetime

from gently.harness.memory.model import Confidence, Learning, Observation
from gently.harness.memory.notebook import (
    Author,
    Note,
    NotebookStore,
    NoteKind,
    NoteStatus,
    learning_to_note,
    note_from_dict,
    note_to_dict,
    observation_to_note,
)


class TestNoteModel:
    def test_round_trip_minimal(self):
        n = Note(id="abc123", kind=NoteKind.OBSERVATION, body="dim rings at 10 ms")
        d = note_to_dict(n)
        assert d["kind"] == "observation"
        assert d["author"] == "agent"  # default
        assert d["status"] == "confirmed"  # default
        back = note_from_dict(d)
        assert back == n

    def test_round_trip_full(self):
        n = Note(
            id="def456",
            kind=NoteKind.FINDING,
            body="temperature shifts timing ~12 min/degC",
            author=Author.AGENT,
            title="Temp shifts timing",
            status=NoteStatus.PROPOSED,
            confidence=Confidence.MEDIUM,
            strains=["N2", "OH904"],
            embryos=["emb_0007"],
            sessions=["20260615_1432_x"],
            threads=["q_division_temp"],
            basis=["obs_1", "obs_2"],
            links=[{"rel": "supports", "to": "q_division_temp"}],
            artifacts=[{"kind": "projection", "session": "s1", "embryo": "emb_0007", "t": 42}],
            created_at=datetime(2026, 6, 16, 11, 0, 0),
            updated_at=datetime(2026, 6, 16, 11, 0, 0),
        )
        back = note_from_dict(note_to_dict(n))
        assert back == n
        assert note_to_dict(n)["confidence"] == "medium"


class TestNotebookStoreReadWrite:
    def test_write_assigns_id_and_persists(self, tmp_path):
        store = NotebookStore(tmp_path / "notebook")
        n = Note(id="", kind=NoteKind.OBSERVATION, body="bean stage at t40")
        note_id = store.write_note(n)
        assert note_id  # non-empty id assigned
        files = list((tmp_path / "notebook" / "notes").glob("*.yaml"))
        assert len(files) == 1
        assert files[0].name.startswith(note_id + "_")

    def test_get_note_round_trip(self, tmp_path):
        store = NotebookStore(tmp_path / "notebook")
        n = Note(id="", kind=NoteKind.FINDING, body="x", strains=["N2"], threads=["t1"])
        note_id = store.write_note(n)
        got = store.get_note(note_id)
        assert got is not None
        assert got.id == note_id
        assert got.kind == NoteKind.FINDING
        assert got.strains == ["N2"]

    def test_get_missing_returns_none(self, tmp_path):
        store = NotebookStore(tmp_path / "notebook")
        assert store.get_note("nope") is None


class TestNotebookIndex:
    def test_index_updated_on_write(self, tmp_path):
        store = NotebookStore(tmp_path / "notebook")
        a = store.write_note(Note(id="", kind=NoteKind.OBSERVATION, body="a", strains=["N2"]))
        b = store.write_note(
            Note(id="", kind=NoteKind.OBSERVATION, body="b", strains=["N2", "OH904"])
        )
        assert set(store.ids_for_strain("N2")) == {a, b}
        assert store.ids_for_strain("OH904") == [b]
        assert store.ids_for_strain("missing") == []

    def test_index_by_embryo_and_thread(self, tmp_path):
        store = NotebookStore(tmp_path / "notebook")
        a = store.write_note(
            Note(id="", kind=NoteKind.FINDING, body="a", embryos=["e1"], threads=["t1"])
        )
        assert store.ids_for_embryo("e1") == [a]
        assert store.ids_for_thread("t1") == [a]

    def test_rebuild_index_from_disk(self, tmp_path):
        store = NotebookStore(tmp_path / "notebook")
        a = store.write_note(Note(id="", kind=NoteKind.OBSERVATION, body="a", strains=["N2"]))
        # a fresh store over the same dir must rebuild the index by scanning notes/
        store2 = NotebookStore(tmp_path / "notebook")
        assert store2.ids_for_strain("N2") == [a]


class TestNotebookQuery:
    def _seed(self, tmp_path):
        store = NotebookStore(tmp_path / "notebook")
        store.write_note(Note(id="o1", kind=NoteKind.OBSERVATION, body="o", strains=["N2"]))
        store.write_note(
            Note(
                id="f1",
                kind=NoteKind.FINDING,
                body="f",
                status=NoteStatus.PROPOSED,
                strains=["N2"],
                threads=["t1"],
            )
        )
        store.write_note(
            Note(id="q1", kind=NoteKind.QUESTION, body="q", status=NoteStatus.OPEN, threads=["t1"])
        )
        return store

    def test_query_by_kind(self, tmp_path):
        store = self._seed(tmp_path)
        ids = {n.id for n in store.query_notes(kind=NoteKind.FINDING)}
        assert ids == {"f1"}

    def test_query_by_thread_scope(self, tmp_path):
        store = self._seed(tmp_path)
        ids = {n.id for n in store.query_notes(thread="t1")}
        assert ids == {"f1", "q1"}

    def test_query_by_thread_and_kind(self, tmp_path):
        store = self._seed(tmp_path)
        ids = {n.id for n in store.query_notes(thread="t1", kind=NoteKind.QUESTION)}
        assert ids == {"q1"}

    def test_query_by_status(self, tmp_path):
        store = self._seed(tmp_path)
        ids = {n.id for n in store.query_notes(status=NoteStatus.OPEN)}
        assert ids == {"q1"}

    def test_query_all_sorted_newest_first(self, tmp_path):
        store = self._seed(tmp_path)
        notes = store.query_notes()
        assert len(notes) == 3
        ts = [n.created_at for n in notes]
        assert ts == sorted(ts, reverse=True)


class TestNotebookLinkSupersede:
    def test_link_notes_adds_typed_edge(self, tmp_path):
        store = NotebookStore(tmp_path / "notebook")
        a = store.write_note(Note(id="", kind=NoteKind.FINDING, body="a"))
        b = store.write_note(Note(id="", kind=NoteKind.QUESTION, body="b"))
        store.link_notes(a, "supports", b)
        got = store.get_note(a)
        assert {"rel": "supports", "to": b} in got.links

    def test_supersede_marks_old_and_points_new(self, tmp_path):
        store = NotebookStore(tmp_path / "notebook")
        old = store.write_note(Note(id="", kind=NoteKind.FINDING, body="old claim"))
        new = store.write_note(Note(id="", kind=NoteKind.FINDING, body="better claim"))
        store.supersede_note(old, new)
        old_n = store.get_note(old)
        new_n = store.get_note(new)
        assert old_n.status == NoteStatus.SUPERSEDED
        assert old_n.superseded_by == new
        assert {"rel": "refines", "to": old} in new_n.links


class TestConverters:
    def test_observation_to_note(self):
        obs = Observation(
            id="o1",
            timestamp=datetime(2026, 6, 16, 9, 0, 0),
            type="milestone",
            content="nerve ring formed",
            embryo_id="e1",
            session_id="s1",
            relates_to=["o0"],
            gently_refs={"kind": "projection", "t": 42},
        )
        n = observation_to_note(obs)
        assert n.id == "o1"
        assert n.kind == NoteKind.OBSERVATION
        assert n.body == "nerve ring formed"
        assert n.author == Author.AGENT
        assert n.embryos == ["e1"]
        assert n.sessions == ["s1"]
        assert {"rel": "relates_to", "to": "o0"} in n.links
        assert n.artifacts == [{"kind": "projection", "t": 42}]
        assert n.created_at == datetime(2026, 6, 16, 9, 0, 0)

    def test_learning_to_note(self):
        lrn = Learning(id="l1", content="rings form by comma", confidence=Confidence.HIGH)
        n = learning_to_note(lrn)
        assert n.id == "l1"
        assert n.kind == NoteKind.FINDING
        assert n.body == "rings form by comma"
        assert n.status == NoteStatus.PROPOSED  # agent-drafted, awaits confirm
        assert n.confidence == Confidence.HIGH


class TestContextStoreNotebook:
    def test_notebook_property_rooted_under_agent_dir(self, file_context_store):
        nb = file_context_store.notebook
        assert nb.root == file_context_store.agent_dir / "notebook"

    def test_notebook_property_is_cached(self, file_context_store):
        assert file_context_store.notebook is file_context_store.notebook


class TestApplyUpdatesMirror:
    def test_apply_updates_mirrors_observations_and_learnings(self, file_context_store):
        from gently.harness.memory.model import ContextUpdates

        cs = file_context_store
        obs = Observation(
            id="o1",
            timestamp=datetime(2026, 6, 16, 9, 0, 0),
            type="milestone",
            content="ring formed",
            embryo_id="e1",
        )
        lrn = Learning(id="l1", content="rings form by comma", confidence=Confidence.HIGH)
        cs.apply_updates(ContextUpdates(new_observations=[obs], new_learnings=[lrn]))

        bodies = {n.body for n in cs.notebook.query_notes()}
        assert "ring formed" in bodies
        assert "rings form by comma" in bodies
        assert cs.notebook.ids_for_embryo("e1") == ["o1"]

    def test_apply_updates_empty_is_noop_for_notebook(self, file_context_store):
        from gently.harness.memory.model import ContextUpdates

        cs = file_context_store
        cs.apply_updates(ContextUpdates())
        assert cs.notebook.query_notes() == []
