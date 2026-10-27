"""The names a rig has seen, for the landing page's who's here."""

from gently.core.file_store import FileStore
from gently.ui.web import operators


def test_names_are_remembered_and_gathered_from_records(tmp_path, monkeypatch):
    monkeypatch.setattr(operators, "operators_path", lambda: tmp_path / "config" / "operators.yaml")
    assert operators.list_operators() == []
    assert operators.remember_operator("  Hari ") == "Hari"
    operators.remember_operator("pskeshu")
    operators.remember_operator("hari")  # same person, case aside: seen again, not twice
    assert operators.list_operators() == ["Hari", "pskeshu"]
    store = FileStore(tmp_path)
    store.create_session("s1", name="s1")
    store.link_session("s1", "proj1", by="ryan")
    store.link_session("s1", "proj2", by="agent")
    assert operators.list_operators(store) == ["Hari", "pskeshu", "ryan"]
    try:
        operators.remember_operator("   ")
    except ValueError:
        pass
    else:
        raise AssertionError("an empty name must not be remembered")
