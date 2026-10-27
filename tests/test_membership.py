from gently.app.filing import backfill_session_projects, file_session_under, unfile_session
from gently.core.file_store import FileStore
from gently.harness.memory.file_store import FileContextStore
from gently.tools.pool import pool
from gently.tools.select import select_sessions


def _two_projects(tmp_path):
    store = FileStore(tmp_path)
    cs = FileContextStore(tmp_path / "agent")
    dose = cs.create_campaign("Does dose change timing?", shorthand="dose")
    warm = cs.create_campaign("25 °C", shorthand="warm", parent_id=dose)
    timing = cs.create_campaign("Timing variance", shorthand="timing")
    store.create_session("s1", name="s1")
    return store, cs, dose, warm, timing


def test_a_session_informs_several_projects_in_different_roles(tmp_path):
    store, cs, dose, warm, timing = _two_projects(tmp_path)
    first = file_session_under(store, cs, "s1", warm, role="control", by="pskeshu")
    assert first["home_set"] and first["project_id"] == dose
    assert store.session_links("s1") == [
        {
            "id": dose,
            "phase": warm,
            "role": "control",
            "by": "pskeshu",
            "at": first["links"][0]["at"],
        }
    ]
    second = file_session_under(store, cs, "s1", timing, role="reference")
    assert second["home_kept"] == dose and not second["home_set"]
    ids = [x["id"] for x in store.session_links("s1")]
    assert ids == [dose, timing]
    # both projects list it; the role is per link
    assert [s["session_id"] for s in store.list_sessions(project_id=dose)] == ["s1"]
    assert [s["session_id"] for s in store.list_sessions(project_id=timing)] == ["s1"]
    assert store.list_sessions(project_id=dose)[0]["projects"][1]["role"] == "reference"
    # a second filing to the same project updates the role, never duplicates
    file_session_under(store, cs, "s1", dose, role="test")
    links = store.session_links("s1")
    assert len(links) == 2 and links[0]["role"] == "test" and links[0]["phase"] == warm
    # move makes the other one home
    moved = file_session_under(store, cs, "s1", timing, move=True)
    assert moved["home_set"] and store.get_session("s1")["project_id"] == timing


def test_unfiling_one_project_keeps_the_other_and_survives_a_restart(tmp_path):
    store, cs, dose, warm, timing = _two_projects(tmp_path)
    file_session_under(store, cs, "s1", warm)
    file_session_under(store, cs, "s1", timing)
    assert unfile_session(store, cs, "s1", dose)
    assert [x["id"] for x in store.session_links("s1")] == [timing]
    assert cs.get_campaign_ids_for_session("s1") == [timing]  # the phase's intent went too
    assert unfile_session(store, cs, "s1")
    assert store.session_links("s1") == [] and store.get_session("s1")["project_id"] is None
    assert "projects" in store.get_session("s1")  # deliberately unfiled, on record
    assert backfill_session_projects(store, cs) == 0  # and the backfill leaves it alone


def test_the_old_scalar_reads_as_one_link_and_the_backfill_files_only_the_never_filed(tmp_path):
    store, cs, dose, warm, timing = _two_projects(tmp_path)
    store.create_session("old", name="old", project_id=dose)  # written the old way
    assert (
        store.session_links("old")[0]["id"] == dose
        and store.session_links("old")[0]["role"] is None
    )
    store.create_session("intent-only", name="intent-only")
    cs.link_session_campaign("intent-only", warm)
    assert backfill_session_projects(store, cs) == 1
    assert store.session_links("intent-only")[0] | {"at": None} == {
        "id": dose,
        "phase": None,
        "role": None,
        "by": "backfill",
        "at": None,
    }


def test_the_selector_and_the_pool_see_links(tmp_path):
    store, cs, dose, warm, timing = _two_projects(tmp_path)
    store.create_session("s2", name="s2")
    file_session_under(store, cs, "s1", warm, role="control")
    file_session_under(store, cs, "s2", dose, role="test")
    assert {s["session_id"] for s in select_sessions(store, cs, project="dose")["sessions"]} == {
        "s1",
        "s2",
    }
    assert [
        s["session_id"]
        for s in select_sessions(store, cs, project="dose", phase="warm")["sessions"]
    ] == ["s1"]
    roles = {
        s["session_id"]: s["role"]
        for s in pool(store, cs, "twitch_hatch", project="dose")["sessions"]
    }
    assert roles == {"s1": "control", "s2": "test"}
