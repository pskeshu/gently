from gently.core.file_store import FileStore
from gently.harness.memory.file_store import FileContextStore
from gently.tools.select import select_sessions


def test_project_phase_and_tag_select_and_intersect(tmp_path):
    store = FileStore(tmp_path)
    cs = FileContextStore(tmp_path / "agent")
    proj = cs.create_campaign("Twitch to hatch in control", shorthand="t2h")
    phase = cs.create_campaign("25 °C nights", shorthand="warm", parent_id=proj)
    for sid, tags in (("s1", ["control"]), ("s2", ["control", "warm"]), ("s3", [])):
        store.create_session(sid, name=sid, project_id=proj)
        store.set_session_tags(sid, tags)
    store.create_session("other", name="other")  # unfiled, untagged
    cs.link_session_campaign("s2", phase)

    ids = lambda out: sorted(s["session_id"] for s in out["sessions"])  # noqa: E731
    assert ids(select_sessions(store, cs, project="t2h")) == ["s1", "s2", "s3"]
    assert select_sessions(store, cs, project="t2h")["label"] == "t2h"
    assert ids(select_sessions(store, cs, phase=phase)) == ["s2"]
    assert ids(select_sessions(store, cs, project="t2h", phase="warm")) == ["s2"]
    assert ids(select_sessions(store, cs, phase="warm")) == []  # shorthand needs its project
    assert ids(select_sessions(store, cs, tag="control")) == ["s1", "s2"]
    both = select_sessions(store, cs, project="t2h", tag="control")
    assert ids(both) == ["s1", "s2"] and both["label"] == "t2h · control"
    assert ids(select_sessions(store, cs, project="nope")) == []
    assert ids(select_sessions(store, cs)) == []
