from types import SimpleNamespace

from gently.core.file_store import FileStore, _write_yaml
from gently.harness.memory.file_store import FileContextStore
from gently.harness.memory.notebook import Author, Note, NoteKind
from gently.tools import reading


def _project(tmp_path):
    store = FileStore(tmp_path)
    cs = FileContextStore(tmp_path / "agent")
    proj = cs.create_campaign("Does LED dose change time to hatching?", shorthand="dose")
    for sid, name in (("s1", "monday"), ("s2", "tuesday")):
        store.create_session(sid, name=name, project_id=proj)
        store.register_embryo(sid, "e01", position_x=0, position_y=0)
        store.store_prediction(1, sid, "e01", 1, "bean", 0.9)
        store.store_prediction(1, sid, "e01", 2, "comma", 0.9)
        for n in (1, 2):
            meta = {"frame": n, "field": 0, "captured_at": f"2026-10-27T0{n}:00:00"}
            store.put_snapshot(sid, "bf", [[0]], metadata=meta, stem=f"f{n}")
        _write_yaml(
            store._session_dir(sid) / "annotations" / "twitch_hatch.yaml",
            [{"field": 0, "x": 0.1, "y": 0.1, "twitch_frame": 1, "hatch_frame": 2}],
        )
    cs.notebook.write_note(
        Note(
            id="",
            kind=NoteKind.OBSERVATION,
            body="embryo 3 late",
            author=Author.HUMAN,
            projects=[proj],
        )
    )
    cs.notebook.write_note(
        Note(
            id="",
            kind=NoteKind.OBSERVATION,
            body="dim at dose 2",
            author=Author.HUMAN,
            sessions=["s2"],
        )
    )
    return store, cs, proj


def test_the_pot_holds_question_sessions_rows_and_stream(tmp_path):
    store, cs, proj = _project(tmp_path)
    p = reading.pot(store, cs, proj)
    text = p["text"]
    assert "Does LED dose change time to hatching?" in text
    assert "SESSION monday" in text and "SESSION tuesday" in text
    assert "bean t1, comma t2" in text
    assert "POOLED twitch_hatch" in text and "median 60" in text and "Sulston 370" in text
    assert "embryo 3 late" in text and "dim at dose 2" in text  # by project, and by session
    assert p["sessions"] == ["s1", "s2"] and len(p["notes"]) == 2
    assert reading.load_reading(store, cs, proj)["text"] is None
    bodies = [n.body for n in reading.stream_notes(cs, proj, {"s1", "s2"})]
    assert bodies == ["embryo 3 late", "dim at dose 2"]


def test_the_reading_is_filed_and_goes_stale_when_a_session_lands(tmp_path, monkeypatch):
    store, cs, proj = _project(tmp_path)
    sent = {}

    class Client:
        class messages:
            @staticmethod
            def create(**kw):
                sent.update(kw)
                return SimpleNamespace(
                    stop_reason="end_turn",
                    content=[SimpleNamespace(type="text", text="KNOWN. Both sessions: 60 min.")],
                )

    monkeypatch.setattr("anthropic.Anthropic", lambda: Client())
    out = reading.read_project(store, cs, proj, model="test-model")
    assert out["text"].startswith("KNOWN.") and sent["model"] == "test-model"
    assert "SESSION monday" in sent["messages"][0]["content"]
    assert (cs.campaign_dir(proj) / "reading.md").read_text().startswith("KNOWN.")
    got = reading.load_reading(store, cs, proj)
    assert got["text"] == out["text"] and got["sessions_since"] == 0
    store.create_session("s3", name="wednesday", project_id=proj)
    assert reading.load_reading(store, cs, proj)["sessions_since"] == 1
