"""Tests for the shared lab notebook: Note model + NotebookStore."""

from datetime import datetime

from gently.harness.memory.model import Confidence
from gently.harness.memory.notebook import (
    Author,
    Note,
    NoteKind,
    NoteStatus,
    NotebookStore,
    note_from_dict,
    note_to_dict,
)


class TestNoteModel:
    def test_round_trip_minimal(self):
        n = Note(id="abc123", kind=NoteKind.OBSERVATION, body="dim rings at 10 ms")
        d = note_to_dict(n)
        assert d["kind"] == "observation"
        assert d["author"] == "agent"            # default
        assert d["status"] == "confirmed"        # default
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
