"""
Tests for PromptManager: tool selection, caching, and context data gathering.
"""

from datetime import datetime
from unittest.mock import MagicMock

import pytest

from gently.harness.prompts.manager import PromptManager

# ===========================================================================
# Tool Selection
# ===========================================================================


class TestToolSelection:
    """PromptManager returns correct tools per mode."""

    @pytest.fixture
    def mgr(self):
        return PromptManager(claude_client=MagicMock(), model="test-model")

    def test_run_mode_returns_list(self, mgr):
        tools = mgr.get_tools_for_mode("run", has_microscope=True)
        assert isinstance(tools, list)

    def test_plan_mode_tools_subset(self, mgr):
        run_tools = mgr.get_tools_for_mode("run", has_microscope=False)
        plan_tools = mgr.get_tools_for_mode("plan", has_microscope=False)
        # Plan mode should have fewer tools
        assert len(plan_tools) <= len(run_tools)

    def test_plan_mode_only_plan_tools(self, mgr):
        plan_tools = mgr.get_tools_for_mode("plan", has_microscope=False)
        plan_tool_names = {t["name"] for t in plan_tools}
        allowed = {
            "create_campaign",
            "create_plan_item",
            "update_plan_item",
            "link_plan_items",
            "propose_plan",
            "get_plan_status",
            "get_plan_item",
            "move_plan_item",
            "delete_plan_item",
            "reorder_plan_items",
            "update_phase",
            "delete_phase",
            "export_plan",
            "query_lab_history",
            "check_hardware_capability",
            "search_literature",
            "search_strains",
            "validate_plan",
            "batch_update_status",
            "batch_update_spec",
            "save_plan_template",
            "list_templates",
            "apply_template",
            "ask_user_choice",
        }
        assert plan_tool_names.issubset(allowed)


# ===========================================================================
# Memory Awareness Caching
# ===========================================================================


class TestMemoryCaching:
    """PromptManager caches memory awareness with TTL."""

    def test_no_memory_returns_empty(self):
        mgr = PromptManager(claude_client=MagicMock(), model="test")
        assert mgr.get_cached_memory_awareness() == ""

    def test_caches_result(self):
        mgr = PromptManager(claude_client=MagicMock(), model="test")
        memory = MagicMock()
        memory.get_awareness_summary.return_value = "I know things"
        mgr.memory = memory

        result1 = mgr.get_cached_memory_awareness()
        mgr.get_cached_memory_awareness()
        assert result1 == "I know things"
        # Should only call once (cached)
        assert memory.get_awareness_summary.call_count == 1

    def test_invalidate_cache(self):
        mgr = PromptManager(claude_client=MagicMock(), model="test")
        memory = MagicMock()
        memory.get_awareness_summary.return_value = "cached"
        mgr.memory = memory

        mgr.get_cached_memory_awareness()
        mgr.invalidate_context_cache()
        mgr.get_cached_memory_awareness()
        assert memory.get_awareness_summary.call_count == 2


# ===========================================================================
# Context Data Gathering
# ===========================================================================


class TestContextDataGathering:
    """PromptManager gathers raw context for summarization."""

    def test_empty_context(self):
        mgr = PromptManager(claude_client=MagicMock(), model="test")
        from gently.harness.state import ExperimentState

        exp = ExperimentState()
        data = mgr.gather_context_data(exp, None, None)
        assert data["timelapse_status"] is None
        assert data["recent_events"] == []
        assert data["recent_detections"] == []

    def test_with_detections(self):
        mgr = PromptManager(claude_client=MagicMock(), model="test")
        from gently.harness.state import ExperimentState

        exp = ExperimentState()
        exp.add_embryo("e1")
        exp.embryos["e1"].add_detection_result(
            "comma",
            {
                "timepoint": 50,
                "detected": True,
                "confidence": "HIGH",
                "reasoning": "Clear comma shape",
            },
        )
        data = mgr.gather_context_data(exp, None, None)
        assert len(data["recent_detections"]) == 1
        assert data["recent_detections"][0]["detector"] == "comma"

    def test_with_timelapse(self):
        mgr = PromptManager(claude_client=MagicMock(), model="test")
        from gently.harness.state import ExperimentState

        exp = ExperimentState()

        # Mock timelapse orchestrator
        mock_orch = MagicMock()
        status = MagicMock()
        status.status.value = "running"
        status.total_timepoints = 50
        status.started_at = datetime.now()
        status.embryos = ["e1"]
        mock_orch.get_status.return_value = status

        data = mgr.gather_context_data(exp, mock_orch, None)
        assert data["timelapse_status"] is not None
        assert data["timelapse_status"]["state"] == "running"


# ===========================================================================
# Prompt Caching Format
# ===========================================================================


class TestPromptCaching:
    """get_cached_system_prompt formats for Anthropic prompt caching."""

    def test_returns_list_with_cache_control(self):
        mgr = PromptManager(claude_client=MagicMock(), model="test")
        result = mgr.get_cached_system_prompt("You are a microscopy agent.")
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0]["type"] == "text"
        assert "cache_control" in result[0]
        assert result[0]["text"] == "You are a microscopy agent."
