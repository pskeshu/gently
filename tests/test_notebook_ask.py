"""Tests for 'Ask the notebook' — retrieval + grounded synthesis."""

import asyncio

from gently.harness.memory.notebook import Note, NoteKind
from gently.harness.memory.notebook_ask import (
    ASK_TOOL,
    answer_question,
    build_ask_messages,
    normalize_answer,
    select_notes,
)


def _seed(cs):
    nb = cs.notebook
    nb.write_note(
        Note(id="o1", kind=NoteKind.OBSERVATION, body="ring formed", strains=["N2"], threads=["t1"])
    )
    nb.write_note(
        Note(id="f1", kind=NoteKind.FINDING, body="12 min/degC", strains=["N2"], threads=["t1"])
    )
    nb.write_note(Note(id="x1", kind=NoteKind.OBSERVATION, body="unrelated", strains=["OH904"]))
    return nb


class TestSelectNotes:
    def test_scope_by_thread(self, file_context_store):
        nb = _seed(file_context_store)
        ids = {n.id for n in select_notes(nb, thread="t1")}
        assert ids == {"o1", "f1"}

    def test_scope_by_strain(self, file_context_store):
        nb = _seed(file_context_store)
        ids = {n.id for n in select_notes(nb, strain="OH904")}
        assert ids == {"x1"}

    def test_no_scope_returns_recent_capped(self, file_context_store):
        nb = _seed(file_context_store)
        notes = select_notes(nb, limit=2)
        assert len(notes) == 2  # newest-first, capped


class _FakeBlock:
    def __init__(self, inp):
        self.type = "tool_use"
        self.input = inp


class _FakeResp:
    def __init__(self, inp):
        self.content = [_FakeBlock(inp)]
        self.stop_reason = "tool_use"


class _FakeMessages:
    def __init__(self, captured, inp):
        self._captured, self._inp = captured, inp

    async def create(self, **kwargs):
        self._captured.update(kwargs)
        return _FakeResp(self._inp)


class _FakeClient:
    def __init__(self, inp):
        self.captured = {}
        self.messages = _FakeMessages(self.captured, inp)


class TestAnswerQuestion:
    def test_build_messages_embeds_note_ids(self):
        notes = [Note(id="o1", kind=NoteKind.OBSERVATION, body="ring formed")]
        msgs = build_ask_messages("what formed?", notes)
        text = msgs[0]["content"]
        assert "o1" in text and "ring formed" in text and "what formed?" in text

    def test_answer_returns_structured_and_forces_tool(self):
        canned = {
            "answer": "A ring formed.",
            "points": [{"text": "ring formed", "note_ids": ["o1"]}],
            "suggested_next": [],
            "coverage": "covered",
        }
        client = _FakeClient(canned)
        notes = [Note(id="o1", kind=NoteKind.OBSERVATION, body="ring formed")]
        out = asyncio.run(answer_question(client, "m", "what formed?", notes))
        assert out == canned
        assert client.captured["tool_choice"] == {"type": "tool", "name": ASK_TOOL["name"]}
        assert client.captured["model"] == "m"

    def test_answer_no_notes_short_circuits_without_api(self):
        client = _FakeClient({"should": "not be used"})
        out = asyncio.run(answer_question(client, "m", "anything?", []))
        assert out["coverage"] == "not_in_notebook"
        assert client.captured == {}  # no API call when nothing to ground on

    def test_garbled_tool_call_is_salvaged_not_crashed(self):
        # Seen in the wild for "sup": the model leaked tool-call markup into
        # its own values — `points` a string with a <parameter> tag, and the
        # point's text/note_ids spilled to the top level. The page's renderer
        # did `points.map` on a string and showed "Something went wrong".
        garbled = {
            "answer": "Hi! Nothing to answer here.",
            "points": '\n<parameter name="text">Multiple entries ask Ryan to annotate.',
            "note_ids": ["b2f81c48", "e8194702"],
            "text": "There are observations from hatching runs.",
            "coverage": "not_in_notebook",
        }
        client = _FakeClient(garbled)
        notes = [Note(id="b2f81c48", kind=NoteKind.OBSERVATION, body="annotate please")]
        out = asyncio.run(answer_question(client, "m", "sup", notes))
        assert out["answer"] == "Hi! Nothing to answer here."
        assert out["points"] == [
            {"text": "Multiple entries ask Ryan to annotate.", "note_ids": []},
            {
                "text": "There are observations from hatching runs.",
                "note_ids": ["b2f81c48", "e8194702"],
            },
        ]
        assert out["suggested_next"] == []
        assert out["coverage"] == "not_in_notebook"


class TestNormalizeAnswer:
    def test_well_formed_passes_through(self):
        canned = {
            "answer": "A ring formed.",
            "points": [{"text": "ring formed", "note_ids": ["o1"]}],
            "suggested_next": ["try 25C"],
            "coverage": "covered",
        }
        assert normalize_answer(canned) == canned

    def test_wrong_types_are_coerced_never_raised(self):
        out = normalize_answer(
            {
                "answer": None,
                "points": [
                    None,
                    {"text": "", "note_ids": ["o1"]},
                    {"text": "kept", "note_ids": "o2"},
                ],
                "suggested_next": "one thing",
                "coverage": "sure",
            }
        )
        assert out == {
            "answer": "",
            "points": [{"text": "kept", "note_ids": ["o2"]}],
            "suggested_next": ["one thing"],
            "coverage": "partial",  # unknown coverage: partial when any point survived
        }

    def test_not_a_dict_is_empty(self):
        assert normalize_answer("nope") == {
            "answer": "",
            "points": [],
            "suggested_next": [],
            "coverage": "not_in_notebook",
        }


class TestSelectNotesReadsTheNotebookOnly:
    def test_a_note_set_aside_is_out_of_the_ask(self, file_context_store):
        nb = _seed(file_context_store)
        nb.set_in_notebook("o1", False)
        assert {n.id for n in select_notes(nb, thread="t1")} == {"f1"}
        assert {n.id for n in select_notes(nb)} == {"f1", "x1"}
