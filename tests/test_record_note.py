"""Tests for the record_note tool — human notes into the shared notebook."""

import asyncio
from types import SimpleNamespace

import gently.app.tools.memory_tools  # noqa: F401 — registers record_note
from gently.harness.tools.registry import get_tool_registry


def _handler():
    return get_tool_registry().get("record_note").handler


class TestRecordNote:
    def test_writes_human_note_tagged_to_session(self, file_context_store):
        agent = SimpleNamespace(context_store=file_context_store, session_id="sess_1", memory=None)
        out = asyncio.run(
            _handler()(text="ring formed cleanly", embryos=["e1"], context={"agent": agent})
        )
        assert "Noted" in out
        notes = file_context_store.notebook.query_notes()
        assert len(notes) == 1
        n = notes[0]
        assert n.author.value == "human"
        assert n.body == "ring formed cleanly"
        assert n.sessions == ["sess_1"]
        assert n.embryos == ["e1"]

    def test_no_session_still_records(self, file_context_store):
        agent = SimpleNamespace(context_store=file_context_store, session_id=None, memory=None)
        asyncio.run(_handler()(text="general observation", context={"agent": agent}))
        notes = file_context_store.notebook.query_notes()
        assert len(notes) == 1
        assert notes[0].sessions == []

    def test_no_store_returns_message(self):
        agent = SimpleNamespace(context_store=None, session_id=None, memory=None)
        out = asyncio.run(_handler()(text="x", context={"agent": agent}))
        assert "No notebook available" in out
