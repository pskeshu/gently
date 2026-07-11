"""
FileContextStore: OperationPlan domain — set/get round-trip + CONTEXT_UPDATED event.

Domain: agent/operation_plans/{session_id}.yaml
Methods: set_operation_plan / get_operation_plan
"""

from gently.core.event_bus import EventType, on

# ---------------------------------------------------------------------------
# Fixture helpers
# ---------------------------------------------------------------------------

_PLAN = {
    "session_id": "sess_001",
    "title": "Temperature-strain protocol",
    "goal": "Characterise onset timing under 25 °C ramp",
    "tactics": [
        {
            "id": "t1",
            "name": "Baseline timelapse",
            "kind": "standing_timelapse",
            "state": "done",
            "scope": {"mode": "global"},
            "rationale": "Establish pre-ramp cadence",
            "structure": {
                "cadence_s": 120,
                "per_embryo": [{"embryo_id": "e1", "cadence_phase": "normal", "interval_s": 120}],
            },
            "live_bind": ["cadence"],
            "relations": {},
        },
        {
            "id": "t2",
            "name": "Temp ramp monitor",
            "kind": "reactive_monitor",
            "state": "active",
            "scope": {"mode": "global"},
            "rationale": "Detect onset event triggered by ramp",
            "structure": {
                "watch": "temperature > 24.5",
                "reaction": "burst_capture",
                "status": "armed",
            },
            "live_bind": ["temperature", "signal"],
            "relations": {"after": ["t1"]},
        },
        {
            "id": "t3",
            "name": "Post-ramp survey",
            "kind": "oneshot",
            "state": "planned",
            "scope": {"mode": "embryos", "embryo_ids": ["e1", "e2"]},
            "rationale": "Confirm recovery after ramp completion",
            "structure": {"note": "single high-res z-stack per embryo"},
            "live_bind": [],
            "relations": {"after": ["t2"]},
        },
    ],
    "updated_at": "2026-06-28T10:00:00",
    "updated_reason": "initial plan",
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestOperationPlanRoundTrip:
    def test_set_then_get_returns_plan(self, file_context_store):
        """set_operation_plan followed by get_operation_plan returns the full dict."""
        cs = file_context_store
        cs.set_operation_plan("sess_001", _PLAN)
        result = cs.get_operation_plan("sess_001")
        assert result is not None
        assert result["session_id"] == "sess_001"
        assert result["title"] == "Temperature-strain protocol"

    def test_tactics_list_preserved(self, file_context_store):
        """All three tactics are returned with their ids and kinds intact."""
        cs = file_context_store
        cs.set_operation_plan("sess_001", _PLAN)
        result = cs.get_operation_plan("sess_001")
        tactics = result["tactics"]
        assert len(tactics) == 3
        ids = [t["id"] for t in tactics]
        assert ids == ["t1", "t2", "t3"]
        kinds = [t["kind"] for t in tactics]
        assert kinds == ["standing_timelapse", "reactive_monitor", "oneshot"]

    def test_tactic_states_preserved(self, file_context_store):
        """State values (done / active / planned) survive the YAML round-trip."""
        cs = file_context_store
        cs.set_operation_plan("sess_001", _PLAN)
        result = cs.get_operation_plan("sess_001")
        states = {t["id"]: t["state"] for t in result["tactics"]}
        assert states == {"t1": "done", "t2": "active", "t3": "planned"}

    def test_nested_structure_preserved(self, file_context_store):
        """Nested fields (scope, structure, live_bind, relations) survive intact."""
        cs = file_context_store
        cs.set_operation_plan("sess_001", _PLAN)
        result = cs.get_operation_plan("sess_001")
        t2 = next(t for t in result["tactics"] if t["id"] == "t2")
        assert t2["structure"]["status"] == "armed"
        assert "temperature" in t2["live_bind"]
        assert t2["relations"]["after"] == ["t1"]

    def test_get_missing_returns_none(self, file_context_store):
        """get_operation_plan returns None when no plan exists for the session."""
        result = file_context_store.get_operation_plan("no_such_session")
        assert result is None

    def test_overwrite_replaces_plan(self, file_context_store):
        """A second set_operation_plan replaces the first."""
        cs = file_context_store
        cs.set_operation_plan("sess_001", _PLAN)
        updated = dict(_PLAN, title="Revised plan", updated_reason="tactic t2 fired")
        cs.set_operation_plan("sess_001", updated)
        result = cs.get_operation_plan("sess_001")
        assert result["title"] == "Revised plan"
        assert result["updated_reason"] == "tactic t2 fired"

    def test_independent_sessions(self, file_context_store):
        """Plans for different session_ids are stored independently."""
        cs = file_context_store
        plan_a = dict(_PLAN, session_id="sess_A", title="Plan A")
        plan_b = dict(_PLAN, session_id="sess_B", title="Plan B")
        cs.set_operation_plan("sess_A", plan_a)
        cs.set_operation_plan("sess_B", plan_b)
        assert cs.get_operation_plan("sess_A")["title"] == "Plan A"
        assert cs.get_operation_plan("sess_B")["title"] == "Plan B"


class TestOperationPlanContextUpdatedEvent:
    def test_set_fires_context_updated(self, file_context_store):
        """set_operation_plan emits CONTEXT_UPDATED on the global event bus."""
        cs = file_context_store
        seen = []
        unsub = on(EventType.CONTEXT_UPDATED, lambda e: seen.append(e))
        try:
            cs.set_operation_plan("sess_001", _PLAN)
        finally:
            unsub()
        assert len(seen) >= 1

    def test_context_updated_kind_is_operation_plan(self, file_context_store):
        """The CONTEXT_UPDATED event carries kind='operation_plan'."""
        cs = file_context_store
        seen = []
        unsub = on(EventType.CONTEXT_UPDATED, lambda e: seen.append(e))
        try:
            cs.set_operation_plan("sess_001", _PLAN)
        finally:
            unsub()
        kinds = [(e.data or {}).get("kind") for e in seen]
        assert "operation_plan" in kinds

    def test_get_does_not_fire_event(self, file_context_store):
        """get_operation_plan is read-only and must not emit CONTEXT_UPDATED."""
        cs = file_context_store
        cs.set_operation_plan("sess_001", _PLAN)
        seen = []
        unsub = on(EventType.CONTEXT_UPDATED, lambda e: seen.append(e))
        try:
            cs.get_operation_plan("sess_001")
        finally:
            unsub()
        assert len(seen) == 0
