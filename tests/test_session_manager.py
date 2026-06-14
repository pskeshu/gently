"""
Tests for SessionManager: message serialization, sanitization, and session lifecycle.
"""

from unittest.mock import MagicMock

from gently.harness.session.manager import SessionManager

# ===========================================================================
# Message Sanitization
# ===========================================================================


class TestSanitizeMessages:
    """sanitize_loaded_messages cleans conversation history from JSON snapshots."""

    def test_valid_string_content(self):
        msgs = [{"role": "user", "content": "hello"}]
        result = SessionManager.sanitize_loaded_messages(msgs)
        assert len(result) == 1
        assert result[0]["content"] == "hello"

    def test_valid_dict_blocks(self):
        msgs = [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "hi"},
                ],
            }
        ]
        result = SessionManager.sanitize_loaded_messages(msgs)
        assert len(result) == 1
        assert result[0]["content"][0]["type"] == "text"

    def test_filters_textblock_repr_strings(self):
        """Old snapshots may have serialized TextBlock repr strings."""
        msgs = [
            {
                "role": "assistant",
                "content": [
                    "TextBlock(text='hello', type='text')",
                    {"type": "text", "text": "valid"},
                ],
            }
        ]
        result = SessionManager.sanitize_loaded_messages(msgs)
        assert len(result) == 1
        assert len(result[0]["content"]) == 1
        assert result[0]["content"][0]["type"] == "text"

    def test_filters_tooluseblock_repr(self):
        msgs = [
            {
                "role": "assistant",
                "content": [
                    "ToolUseBlock(id='x', name='y', input={})",
                ],
            }
        ]
        result = SessionManager.sanitize_loaded_messages(msgs)
        assert len(result) == 0  # No valid blocks left

    def test_none_content_skipped(self):
        msgs = [{"role": "user", "content": None}]
        result = SessionManager.sanitize_loaded_messages(msgs)
        assert len(result) == 0

    def test_empty_list_content(self):
        msgs = [{"role": "user", "content": []}]
        result = SessionManager.sanitize_loaded_messages(msgs)
        assert len(result) == 0  # No valid blocks


# ===========================================================================
# Message Serialization
# ===========================================================================


class TestSerializeMessages:
    """serialize_messages converts SDK objects to plain dicts."""

    def test_dict_passthrough(self):
        msgs = [{"role": "user", "content": "hello"}]
        result = SessionManager.serialize_messages(msgs)
        assert result == msgs

    def test_list_of_dict_blocks(self):
        msgs = [
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "response"},
                ],
            }
        ]
        result = SessionManager.serialize_messages(msgs)
        assert result[0]["content"][0] == {"type": "text", "text": "response"}

    def test_model_dump_objects(self):
        """Objects with model_dump() (Pydantic) should be converted."""

        class FakeBlock:
            def model_dump(self):
                return {"type": "text", "text": "from pydantic"}

        msgs = [{"role": "assistant", "content": [FakeBlock()]}]
        result = SessionManager.serialize_messages(msgs)
        assert result[0]["content"][0] == {"type": "text", "text": "from pydantic"}

    def test_typed_objects_without_model_dump(self):
        """Objects with .type/.text attrs should be converted."""

        class FakeTextBlock:
            type = "text"
            text = "hello"

        msgs = [{"role": "assistant", "content": [FakeTextBlock()]}]
        result = SessionManager.serialize_messages(msgs)
        assert result[0]["content"][0] == {"type": "text", "text": "hello"}

    def test_tool_use_block(self):
        class FakeToolUse:
            type = "tool_use"
            id = "tool_123"
            name = "acquire_volume"
            input = {"embryo_id": "e1"}

        msgs = [{"role": "assistant", "content": [FakeToolUse()]}]
        result = SessionManager.serialize_messages(msgs)
        block = result[0]["content"][0]
        assert block["type"] == "tool_use"
        assert block["name"] == "acquire_volume"


# ===========================================================================
# Session Lifecycle
# ===========================================================================


class TestSessionLifecycle:
    """SessionManager create/save/list operations."""

    def _make_manager(self):
        store = MagicMock()
        store.list_sessions.return_value = [
            {"session_id": "abc", "last_active": "2025-01-01T00:00:00"},
            {"session_id": "def", "last_active": "2025-01-02T00:00:00"},
        ]
        return SessionManager(store=store, storage_path="/tmp/test")

    def test_create_session(self):
        mgr = self._make_manager()
        mgr.create_session()
        assert mgr.session_id is not None
        assert len(mgr.session_id) == 8
        mgr.store.create_session.assert_called_once()

    def test_list_sessions(self):
        mgr = self._make_manager()
        sessions = mgr.list_sessions()
        assert len(sessions) == 2

    def test_save_session_no_id(self):
        mgr = self._make_manager()
        # No session created yet
        result = mgr.save_session(MagicMock(), [], "prompt")
        assert result is False

    def test_save_session_success(self):
        mgr = self._make_manager()
        mgr.create_session()
        experiment = MagicMock()
        experiment.to_dict.return_value = {"embryos": {}}
        result = mgr.save_session(experiment, [], "prompt")
        assert result is True
        mgr.store.save_session_snapshot.assert_called_once()

    def test_auto_save_silent_on_error(self):
        mgr = self._make_manager()
        mgr.create_session()
        mgr.store.save_session_snapshot.side_effect = Exception("DB error")
        # Should not raise
        mgr.auto_save(MagicMock(), [], "prompt")
