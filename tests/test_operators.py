"""Who works at this rig: an id that stays, a name that can change."""

from gently.core.file_store import FileStore
from gently.ui.web import operators


def test_a_person_is_an_id_and_a_name(tmp_path, monkeypatch):
    monkeypatch.setattr(operators, "operators_path", lambda: tmp_path / "config" / "operators.yaml")
    assert operators.list_operators() == []
    assert operators.remember_operator("  Hari ") == {"id": "hari", "name": "Hari"}
    operators.remember_operator("pskeshu")
    assert operators.remember_operator("hari") == {"id": "hari", "name": "Hari"}  # the same person
    assert [o["id"] for o in operators.list_operators()] == ["hari", "pskeshu"]
    # a typo is fixed in the name; the id and every record stay
    assert operators.rename_operator("hari", "Hari Shroff") == {"id": "hari", "name": "Hari Shroff"}
    assert operators.names_of()["hari"] == "Hari Shroff"
    assert operators.touch_operator("hari") == {"id": "hari", "name": "Hari Shroff"}
    try:
        operators.touch_operator("nobody")
    except KeyError:
        pass
    else:
        raise AssertionError("an unknown id is not a person")
    # an id on a record the file does not know shows as itself
    store = FileStore(tmp_path)
    store.create_session("s1", name="s1")
    store.link_session("s1", "proj1", by="ryan")
    store.link_session("s1", "proj2", by="agent")
    assert operators.list_operators(store)[-1] == {"id": "ryan", "name": "ryan"}
    for bad in ("   ", "@@@"):
        try:
            operators.remember_operator(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{bad!r} must not be remembered")


def test_a_session_remembers_who_worked_it(tmp_path):
    store = FileStore(tmp_path)
    store.create_session("s1", name="s1")
    assert store.session_operators("s1") == []
    assert store.add_session_operator("s1", "hari") == ["hari"]
    assert store.add_session_operator("s1", "pskeshu") == ["hari", "pskeshu"]
    assert store.add_session_operator("s1", "hari") == ["hari", "pskeshu"]  # once each
    assert store.add_session_operator("s1", None) == ["hari", "pskeshu"]
    assert store.list_sessions()[0]["operators"] == ["hari", "pskeshu"]
