from gently.core.file_store import FileStore
from gently.harness.memory.file_store import FileContextStore
from gently.tools import (
    project_tools,
    session_tools,
    set_project_tools,
    set_session_tools,
    tools_of,
)


def test_tools_have_a_scope_and_each_side_imports_only_its_own(tmp_path):
    assert tools_of("session") == ["twitch_hatch"]
    assert tools_of("project") == ["figures", "reading"]
    store = FileStore(tmp_path)
    cs = FileContextStore(tmp_path / "agent")
    proj = cs.create_campaign("A question", shorthand="q")
    store.create_session("s1", name="s1", project_id=proj)
    assert set_session_tools(store, "s1", ["twitch_hatch", "figures", "nope"]) == ["twitch_hatch"]
    assert session_tools(store, "s1") == ["twitch_hatch"]
    assert project_tools(cs, proj) == []
    assert set_project_tools(cs, proj, ["reading", "twitch_hatch", "figures", "reading"]) == [
        "reading",
        "figures",
    ]
    assert project_tools(cs, proj) == ["reading", "figures"]
    assert cs.get_campaign(proj).shorthand == "q"  # the rest of campaign.yaml is untouched
