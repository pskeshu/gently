"""Regression test for A003 — autonomous (event-driven) turns self-document.

Before the fix, only user-message turns wrote a Decision row (call_claude
hardcoded trigger=USER_MESSAGE), so a fully autonomous run left
decisions.jsonl empty — exactly what happened in session 6a4a3d9b, forcing
the post-mortem to reconstruct every autonomous decision from conversation.json.

These tests exercise the generalized ConversationManager._write_production_decision
that run_wake_turn now calls with trigger=EVENT + the wake reason.
"""

from __future__ import annotations

from pathlib import Path

from gently.eval import DecisionLog, DecisionTrigger
from gently.harness.conversation import ConversationManager


def _cm_with_log(tmp_path: Path) -> tuple[ConversationManager, Path]:
    # ConversationManager is lightweight (client/model/registry may be dummies);
    # the decision_log is normally attached by the agent once the session folder
    # is known.
    cm = ConversationManager(client=None, model="test-model", tool_registry=None)
    log_path = tmp_path / "decisions.jsonl"
    cm.decision_log = DecisionLog(log_path)
    cm.decision_log.open()
    return cm, log_path


def test_default_trigger_is_user_message(tmp_path: Path):
    """Existing chat-turn behaviour is unchanged (backward compatible)."""
    cm, log_path = _cm_with_log(tmp_path)
    cm._write_production_decision(
        user_message="what is the status?",
        tool_calls=[],
        response_text="idle",
        duration_ms=1.0,
        prompt_hash_value=None,
        error=None,
    )
    cm.decision_log.close()

    rows = DecisionLog(log_path).read()
    assert len(rows) == 1
    assert rows[0].trigger == DecisionTrigger.USER_MESSAGE
    assert rows[0].trigger_detail == "what is the status?"


def test_event_trigger_captures_wake_reason_and_tools(tmp_path: Path):
    """The wake path passes trigger=EVENT, the wake reason, and its tool calls."""
    cm, log_path = _cm_with_log(tmp_path)
    cm._write_production_decision(
        user_message="[wake] embryo_3 potential arrest at pretzel",
        tool_calls=[{"name": "get_photodose_status", "input": {}, "id": None, "is_error": False}],
        response_text="No action - holding.",
        duration_ms=42.0,
        prompt_hash_value=None,
        error=None,
        trigger=DecisionTrigger.EVENT,
        trigger_detail="embryo_3: potential arrest at stage pretzel",
    )
    cm.decision_log.close()

    rows = DecisionLog(log_path).read()
    assert len(rows) == 1
    d = rows[0]
    assert d.trigger == DecisionTrigger.EVENT
    assert d.trigger_detail == "embryo_3: potential arrest at stage pretzel"
    assert d.tool_calls and d.tool_calls[0]["name"] == "get_photodose_status"
    assert d.response_text == "No action - holding."


def test_autonomous_run_no_longer_yields_empty_log(tmp_path: Path):
    """Several event-driven turns => a non-empty, EVENT-tagged decision log."""
    cm, log_path = _cm_with_log(tmp_path)
    for i in range(3):
        cm._write_production_decision(
            user_message=f"[wake {i}]",
            tool_calls=[],
            response_text="held",
            duration_ms=1.0,
            prompt_hash_value=None,
            error=None,
            trigger=DecisionTrigger.EVENT,
            trigger_detail=f"embryo_3 arrest re-fire {i}",
        )
    cm.decision_log.close()

    rows = DecisionLog(log_path).read()
    assert len(rows) == 3
    assert all(r.trigger == DecisionTrigger.EVENT for r in rows)
    assert [r.trigger_detail for r in rows] == [f"embryo_3 arrest re-fire {i}" for i in range(3)]


def test_capture_is_a_noop_without_a_decision_log(tmp_path: Path):
    """No decision_log attached (e.g. no session) must not raise."""
    cm = ConversationManager(client=None, model="test-model", tool_registry=None)
    assert cm.decision_log is None
    cm._write_production_decision(
        user_message="[wake]",
        tool_calls=[],
        response_text="",
        duration_ms=0.0,
        prompt_hash_value=None,
        error=None,
        trigger=DecisionTrigger.EVENT,
        trigger_detail="noop",
    )  # must simply return
