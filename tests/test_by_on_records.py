"""Every record a person writes says who, and when.

Notes, a session's own edits, annotation rows, figure recipes, the reading
request, project links and references: each carries the person's id as
``by`` (or ``asked_by``, ``added_by``), from the request, never typed.
"""

from types import SimpleNamespace

from gently.core.file_store import FileStore
from gently.harness.memory.notebook import Author, Note, NoteKind, note_from_dict, note_to_dict
from gently.ui.web import accounts
from gently.ui.web.auth import OPERATOR_COOKIE, chat_label


def test_a_note_carries_the_person_and_survives_the_disk():
    n = Note(
        id="n1", kind=NoteKind.OBSERVATION, body="embryo 3 late", author=Author.HUMAN, by="hari"
    )
    d = note_to_dict(n)
    assert d["author"] == "human" and d["by"] == "hari"
    assert note_from_dict(d).by == "hari"
    assert note_from_dict({**d, "by": None}).by is None  # the agent's notes have no person
    old = {k: v for k, v in d.items() if k != "by"}  # a note from before the field
    assert note_from_dict(old).by is None


def test_a_session_remembers_its_last_edit(tmp_path):
    store = FileStore(tmp_path)
    store.create_session("s1", name="s1")
    store.note_session_edit("s1", "hari", "name")
    e = store.get_session("s1")["edited"]
    assert e["by"] == "hari" and e["what"] == "name" and e["at"]
    store.note_session_edit("s1", None, "tags")  # nobody named: still when and what
    assert store.get_session("s1")["edited"]["by"] is None


def test_the_chat_signs_with_the_person_the_cookie_names(monkeypatch):
    monkeypatch.setattr(accounts, "_store", None)
    from gently.ui.web import operators

    monkeypatch.setattr(operators, "names_of", lambda store=None: {"hari": "Hari Shroff"})

    def ws(**cookies):
        return SimpleNamespace(cookies=cookies, client=SimpleNamespace(host="127.0.0.1"))

    assert chat_label("account", ws()) == "account"  # an account wins
    assert chat_label(None, ws(**{OPERATOR_COOKIE: "hari"})) == "Hari Shroff"
    assert chat_label(None, ws(**{OPERATOR_COOKIE: "ryan"})) == "ryan"  # unknown id shows as itself
    assert chat_label(None, ws()) == "At the microscope"
