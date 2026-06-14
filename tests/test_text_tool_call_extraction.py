"""Tests for _extract_text_tool_calls fallback parser."""

import json

from gently.harness.conversation import _extract_text_tool_calls


class TestExtractTextToolCalls:
    def test_no_tool_calls(self):
        text = "Hello! How can I help you today?"
        cleaned, calls = _extract_text_tool_calls(text)
        assert cleaned == text
        assert calls == []

    def test_single_tool_call_with_arguments(self):
        tc = json.dumps(
            {
                "name": "ask_user_choice",
                "arguments": {
                    "question": "What next?",
                    "options": [{"id": "a", "label": "Option A"}],
                },
            }
        )
        text = f"Hey there!\n\n<tool_call>\n{tc}\n</tool_call>"
        cleaned, calls = _extract_text_tool_calls(text)
        assert "<tool_call>" not in cleaned
        assert len(calls) == 1
        assert calls[0]["name"] == "ask_user_choice"
        assert calls[0]["input"]["question"] == "What next?"

    def test_single_tool_call_with_input_key(self):
        tc = json.dumps(
            {
                "name": "get_status",
                "input": {"embryo_id": "embryo_1"},
            }
        )
        text = f"Checking...\n<tool_call>{tc}</tool_call>"
        cleaned, calls = _extract_text_tool_calls(text)
        assert len(calls) == 1
        assert calls[0]["input"]["embryo_id"] == "embryo_1"

    def test_preserves_surrounding_text(self):
        tc = json.dumps({"name": "foo", "arguments": {}})
        text = f"Before text.\n\n<tool_call>{tc}</tool_call>\n\nAfter text."
        cleaned, calls = _extract_text_tool_calls(text)
        assert "Before text." in cleaned
        assert "After text." in cleaned
        assert "<tool_call>" not in cleaned
        assert len(calls) == 1

    def test_multiple_tool_calls(self):
        tc1 = json.dumps({"name": "tool_a", "arguments": {"x": 1}})
        tc2 = json.dumps({"name": "tool_b", "arguments": {"y": 2}})
        text = f"Text\n<tool_call>{tc1}</tool_call>\nMore\n<tool_call>{tc2}</tool_call>"
        cleaned, calls = _extract_text_tool_calls(text)
        assert len(calls) == 2
        assert calls[0]["name"] == "tool_a"
        assert calls[1]["name"] == "tool_b"

    def test_invalid_json_skipped(self):
        text = "Hello\n<tool_call>not valid json</tool_call>"
        cleaned, calls = _extract_text_tool_calls(text)
        assert calls == []

    def test_missing_name_skipped(self):
        tc = json.dumps({"arguments": {"x": 1}})
        text = f"<tool_call>{tc}</tool_call>"
        cleaned, calls = _extract_text_tool_calls(text)
        assert calls == []

    def test_empty_text(self):
        cleaned, calls = _extract_text_tool_calls("")
        assert cleaned == ""
        assert calls == []
