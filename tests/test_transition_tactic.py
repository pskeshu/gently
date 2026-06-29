"""
FileContextStore.transition_tactic — TDD tests.

Verifies: state flip, live bind merge, updated_at stamp,
True/False return values, no-op on missing plan or tactic.
"""

from gently.core.event_bus import EventType, on

# ---------------------------------------------------------------------------
# Plan fixture shared across tests
# ---------------------------------------------------------------------------

_PLAN = {
    "session_id": "sess_tt",
    "title": "Transition-tactic test plan",
    "goal": "Verify atomic tactic transitions",
    "tactics": [
        {
            "id": "t1",
            "name": "Baseline timelapse",
            "kind": "standing_timelapse",
            "state": "planned",
            "scope": {"mode": "global"},
            "rationale": "Establish pre-ramp cadence",
            "structure": {"cadence_s": 120},
            "live_bind": ["cadence"],
            "relations": {},
        },
        {
            "id": "t2",
            "name": "Temp ramp monitor",
            "kind": "reactive_monitor",
            "state": "planned",
            "scope": {"mode": "global"},
            "rationale": "Detect onset event",
            "structure": {"watch": "temperature > 24.5"},
            "live_bind": ["temperature"],
            "relations": {"after": ["t1"]},
        },
    ],
    "updated_at": "2026-06-28T00:00:00",
    "updated_reason": "initial plan",
}


def _fresh_plan():
    """Deep-copy the fixture so mutations don't bleed between tests."""
    import copy

    return copy.deepcopy(_PLAN)


# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------


class TestTransitionTacticHappyPath:
    def test_returns_true_on_success(self, file_context_store):
        cs = file_context_store
        cs.set_operation_plan("sess_tt", _fresh_plan())
        result = cs.transition_tactic("sess_tt", "t1", state="active")
        assert result is True

    def test_state_updated(self, file_context_store):
        """State flips from planned to active after transition."""
        cs = file_context_store
        cs.set_operation_plan("sess_tt", _fresh_plan())
        cs.transition_tactic("sess_tt", "t1", state="active")
        plan = cs.get_operation_plan("sess_tt")
        t1 = next(t for t in plan["tactics"] if t["id"] == "t1")
        assert t1["state"] == "active"

    def test_bind_written_to_live(self, file_context_store):
        """Bind kwargs land in tactic['live']."""
        cs = file_context_store
        cs.set_operation_plan("sess_tt", _fresh_plan())
        cs.transition_tactic("sess_tt", "t1", state="active", request_id="b1")
        plan = cs.get_operation_plan("sess_tt")
        t1 = next(t for t in plan["tactics"] if t["id"] == "t1")
        assert t1["live"]["request_id"] == "b1"

    def test_state_and_bind_together(self, file_context_store):
        """State flip and live bind work together in one call."""
        cs = file_context_store
        cs.set_operation_plan("sess_tt", _fresh_plan())
        cs.transition_tactic("sess_tt", "t1", state="active", request_id="b1", cadence=90)
        plan = cs.get_operation_plan("sess_tt")
        t1 = next(t for t in plan["tactics"] if t["id"] == "t1")
        assert t1["state"] == "active"
        assert t1["live"]["request_id"] == "b1"
        assert t1["live"]["cadence"] == 90

    def test_bind_merges_into_existing_live(self, file_context_store):
        """Subsequent bind calls merge, not replace, existing live values."""
        cs = file_context_store
        cs.set_operation_plan("sess_tt", _fresh_plan())
        cs.transition_tactic("sess_tt", "t1", request_id="b1")
        cs.transition_tactic("sess_tt", "t1", mp4_path="/data/t1.mp4")
        plan = cs.get_operation_plan("sess_tt")
        t1 = next(t for t in plan["tactics"] if t["id"] == "t1")
        assert t1["live"]["request_id"] == "b1"
        assert t1["live"]["mp4_path"] == "/data/t1.mp4"

    def test_bind_creates_live_dict_when_absent(self, file_context_store):
        """live dict is created if the tactic didn't have one."""
        cs = file_context_store
        plan = _fresh_plan()
        # Ensure no live key exists
        assert "live" not in plan["tactics"][0]
        cs.set_operation_plan("sess_tt", plan)
        cs.transition_tactic("sess_tt", "t1", setpoint=25.0)
        t1 = cs.get_operation_plan("sess_tt")["tactics"][0]
        assert t1["live"]["setpoint"] == 25.0

    def test_state_only_no_bind(self, file_context_store):
        """state-only call with no bind kwargs works and adds no live key."""
        cs = file_context_store
        cs.set_operation_plan("sess_tt", _fresh_plan())
        cs.transition_tactic("sess_tt", "t1", state="done")
        plan = cs.get_operation_plan("sess_tt")
        t1 = next(t for t in plan["tactics"] if t["id"] == "t1")
        assert t1["state"] == "done"
        assert "live" not in t1

    def test_bind_only_no_state_change(self, file_context_store):
        """bind-only call (no state arg) preserves existing state."""
        cs = file_context_store
        cs.set_operation_plan("sess_tt", _fresh_plan())
        cs.transition_tactic("sess_tt", "t1", cadence=60)
        plan = cs.get_operation_plan("sess_tt")
        t1 = next(t for t in plan["tactics"] if t["id"] == "t1")
        assert t1["state"] == "planned"  # unchanged
        assert t1["live"]["cadence"] == 60

    def test_other_tactic_unaffected(self, file_context_store):
        """Transitioning t1 does not alter t2."""
        cs = file_context_store
        cs.set_operation_plan("sess_tt", _fresh_plan())
        cs.transition_tactic("sess_tt", "t1", state="active", request_id="b1")
        plan = cs.get_operation_plan("sess_tt")
        t2 = next(t for t in plan["tactics"] if t["id"] == "t2")
        assert t2["state"] == "planned"
        assert "live" not in t2

    def test_updated_at_stamped(self, file_context_store):
        """updated_at is refreshed after a successful transition."""
        cs = file_context_store
        cs.set_operation_plan("sess_tt", _fresh_plan())
        cs.transition_tactic("sess_tt", "t1", state="active")
        plan = cs.get_operation_plan("sess_tt")
        assert plan["updated_at"] != "2026-06-28T00:00:00"

    def test_updated_reason_stamped(self, file_context_store):
        """updated_reason reflects which tactic was transitioned."""
        cs = file_context_store
        cs.set_operation_plan("sess_tt", _fresh_plan())
        cs.transition_tactic("sess_tt", "t1", state="active")
        plan = cs.get_operation_plan("sess_tt")
        assert "t1" in plan["updated_reason"]

    def test_fires_context_updated_event(self, file_context_store):
        """transition_tactic fires CONTEXT_UPDATED (via set_operation_plan)."""
        cs = file_context_store
        cs.set_operation_plan("sess_tt", _fresh_plan())
        seen = []
        unsub = on(EventType.CONTEXT_UPDATED, lambda e: seen.append(e))
        try:
            cs.transition_tactic("sess_tt", "t1", state="active")
        finally:
            unsub()
        assert len(seen) >= 1


# ---------------------------------------------------------------------------
# No-op / guard tests
# ---------------------------------------------------------------------------


class TestTransitionTacticNoOp:
    def test_absent_plan_returns_false(self, file_context_store):
        """Returns False and does not crash when no plan exists."""
        result = file_context_store.transition_tactic("no_such_session", "t1", state="active")
        assert result is False

    def test_absent_plan_no_side_effects(self, file_context_store):
        """Absent-plan call creates no operation_plan file."""
        cs = file_context_store
        cs.transition_tactic("ghost_session", "t1", state="active")
        assert cs.get_operation_plan("ghost_session") is None

    def test_unknown_tactic_id_returns_false(self, file_context_store):
        """Returns False when the tactic id is not present in the plan."""
        cs = file_context_store
        cs.set_operation_plan("sess_tt", _fresh_plan())
        result = cs.transition_tactic("sess_tt", "no_such_tactic", state="active")
        assert result is False

    def test_unknown_tactic_plan_unchanged(self, file_context_store):
        """Plan is not mutated when the tactic id is not found."""
        cs = file_context_store
        cs.set_operation_plan("sess_tt", _fresh_plan())
        original_updated_at = cs.get_operation_plan("sess_tt")["updated_at"]
        cs.transition_tactic("sess_tt", "no_such_tactic", state="active")
        plan = cs.get_operation_plan("sess_tt")
        assert plan["updated_at"] == original_updated_at

    def test_no_event_on_absent_plan(self, file_context_store):
        """CONTEXT_UPDATED must not fire when the plan is absent."""
        seen = []
        unsub = on(EventType.CONTEXT_UPDATED, lambda e: seen.append(e))
        try:
            file_context_store.transition_tactic("ghost", "t1", state="active")
        finally:
            unsub()
        assert len(seen) == 0

    def test_no_event_on_unknown_tactic(self, file_context_store):
        """CONTEXT_UPDATED must not fire when the tactic id is not found."""
        cs = file_context_store
        cs.set_operation_plan("sess_tt", _fresh_plan())
        seen = []
        unsub = on(EventType.CONTEXT_UPDATED, lambda e: seen.append(e))
        try:
            cs.transition_tactic("sess_tt", "no_such", state="active")
        finally:
            unsub()
        assert len(seen) == 0
