from gently.tools.twitch_hatch import SULSTON_TWITCH_TO_HATCH_MIN, report


def test_minutes_and_temperature_from_frame_clocks():
    frames = [
        {"field": 0, "frame": 1, "captured_at": "2026-10-27T20:00:00"},
        {"field": 0, "frame": 2, "captured_at": "2026-10-27T20:05:00"},  # interval changed
        {"field": 0, "frame": 3, "captured_at": "2026-10-27T20:15:00"},
        {"field": 1, "frame": 3, "captured_at": "2026-10-27T20:16:00"},
    ]
    temps = [
        {"t": "2026-10-27T19:59:00", "water_c": 30.0},  # before: ignored
        {"t": "2026-10-27T20:03:00", "water_c": 20.0},
        {"t": "2026-10-27T20:10:00", "water_c": 21.0},
    ]
    rows = [
        {"field": 0, "x": 1, "y": 2, "twitch_frame": 1, "hatch_frame": 3},
        {"field": 1, "x": 1, "y": 2, "twitch_frame": 1, "hatch_frame": 3},  # no frame 1 in field 1
        {"field": 0, "x": 1, "y": 2, "twitch_frame": 2, "hatch_frame": None},
    ]
    out = report(rows, frames, temps)
    assert out[0]["minutes"] == 15.0
    assert out[0]["temp_c"] == 20.5
    assert out[0]["vs_sulston_min"] == 15.0 - SULSTON_TWITCH_TO_HATCH_MIN
    assert out[1]["minutes"] is None
    assert out[2]["minutes"] is None and out[2]["temp_c"] is None


def test_marks_become_one_line_in_the_session_stream(tmp_path):
    from gently.core.file_store import FileStore, _write_yaml
    from gently.harness.memory.file_store import FileContextStore
    from gently.tools import reading, twitch_hatch

    store = FileStore(tmp_path)
    cs = FileContextStore(tmp_path / "agent")
    proj = cs.create_campaign("Does LED dose change time to hatching?", shorthand="dose")
    store.create_session("s1", name="monday")  # not in any project yet
    for n, t in ((1, "20:00"), (2, "21:00"), (3, "22:30")):
        meta = {"frame": n, "field": 0, "captured_at": f"2026-10-27T{t}:00"}
        store.put_snapshot("s1", "bf", [[0]], metadata=meta, stem=f"f{n}")
    marks = store._session_dir("s1") / "annotations" / "twitch_hatch.yaml"
    rows = [
        {"field": 0, "x": 0.1, "y": 0.1, "twitch_frame": 1, "hatch_frame": 2},
        {"field": 0, "x": 0.5, "y": 0.5, "twitch_frame": 1, "hatch_frame": 3},
        {"field": 0, "x": 0.9, "y": 0.9, "twitch_frame": 2, "hatch_frame": None},
    ]
    _write_yaml(marks, rows)
    nid = twitch_hatch.note_marks(store, cs, "s1", who="kesavan")
    note = cs.notebook.get_note(nid)
    assert note.title == "Twitching to hatching" and note.sessions == ["s1"]
    assert note.by == "kesavan" and note.author.value == "human"
    assert (
        note.body
        == "Marked on 2 embryos: 60 to 150 min, median 105. 1 more without a hatching time yet."
    )

    # filed into a project afterwards, the line is in that project's stream and its pot
    store.link_session("s1", proj, home=True)
    assert [n.id for n in reading.stream_notes(cs, proj, {"s1"})] == [nid]
    assert "Twitching to hatching: Marked on 2 embryos" in reading.pot(store, cs, proj)["text"]

    # the next save rewrites the same line, signed by whoever saved
    _write_yaml(marks, rows[:1])
    assert twitch_hatch.note_marks(store, cs, "s1", who="ryan") == nid
    assert cs.notebook.get_note(nid).body == "Marked on 1 embryo: 60 min."
    assert cs.notebook.get_note(nid).by == "ryan"
    assert len(cs.notebook.query_notes()) == 1

    # taking the last mark off takes the line off
    _write_yaml(marks, [])
    assert twitch_hatch.note_marks(store, cs, "s1", who="ryan") is None
    assert cs.notebook.query_notes() == []


def test_marks_saved_before_the_line_existed_get_theirs_once(tmp_path):
    """Session e3afe0b6 was marked on Oct 27 and filed under t2h; the stream
    line came after and is written on save, so t2h's stream stayed empty:
    "i added a session with data that is not showing up in t2h"."""
    from gently.core.file_store import FileStore, _write_yaml
    from gently.harness.memory.file_store import FileContextStore
    from gently.tools import reading, twitch_hatch

    store = FileStore(tmp_path)
    cs = FileContextStore(tmp_path / "agent")
    proj = cs.create_campaign("time to hatching", shorthand="t2h")
    for sid in ("marked", "noted", "bare"):
        store.create_session(sid, name=sid)
        for n, t in ((1, "20:00"), (2, "21:00"), (3, "22:30")):
            meta = {"frame": n, "field": 0, "captured_at": f"2026-10-27T{t}:00"}
            store.put_snapshot(sid, "bf", [[0]], metadata=meta, stem=f"f{n}")
    store.link_session("marked", proj, role="control", home=True)
    rows = [
        {"field": 0, "x": 0.1, "y": 0.1, "twitch_frame": 1, "hatch_frame": 2},
        {"field": 0, "x": 0.5, "y": 0.5, "twitch_frame": 1, "hatch_frame": 3},
    ]
    marked = [
        {**rows[0], "by": "kesavan", "at": "2026-10-27T14:01:13"},
        {**rows[1], "by": "ryan", "at": "2026-10-27T14:03:45"},
    ]
    _write_yaml(store._session_dir("marked") / "annotations" / "twitch_hatch.yaml", marked)
    _write_yaml(store._session_dir("noted") / "annotations" / "twitch_hatch.yaml", rows[:1])
    nid = twitch_hatch.note_marks(store, cs, "noted", who="kesavan")  # saved after the line came
    assert reading.stream_notes(cs, proj, {"marked"}) == []

    assert twitch_hatch.backfill_notes(store, cs) == 1
    [note] = reading.stream_notes(cs, proj, {"marked"})
    assert note.sessions == ["marked"] and note.by == "ryan"  # whoever marked last
    assert note.body == "Marked on 2 embryos: 60 to 150 min, median 105."

    # once: the next start writes nothing, the note already there keeps its
    # signature, and a session with no marks gets no line
    assert twitch_hatch.backfill_notes(store, cs) == 0
    assert cs.notebook.get_note(nid).by == "kesavan"
    assert len(cs.notebook.query_notes()) == 2
