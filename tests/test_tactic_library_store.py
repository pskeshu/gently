"""
FileContextStore — Tactic Library store domain TDD tests.

Tests: save_tactic / list_tactics / get_tactic / apply_tactic

Round-trip: save → list → get → apply returns a fresh planned tactic
(new id != template id, state="planned", no "live" key).
get/apply unknown → None.
"""

import copy

# ---------------------------------------------------------------------------
# Shared tactic fixture
# ---------------------------------------------------------------------------

_TACTIC = {
    "id": "t_original",
    "name": "Baseline timelapse",
    "kind": "standing_timelapse",
    "state": "active",
    "scope": {"mode": "global"},
    "rationale": "Establish pre-ramp cadence",
    "structure": {"cadence_s": 120},
    "live_bind": ["cadence"],
    "relations": {},
    "live": {"request_id": "req_abc", "started_at": "2026-06-28T10:00:00"},
}


def _fresh_tactic():
    return copy.deepcopy(_TACTIC)


# ---------------------------------------------------------------------------
# save_tactic
# ---------------------------------------------------------------------------


class TestSaveTactic:
    def test_returns_string_id(self, file_context_store):
        tid = file_context_store.save_tactic(_fresh_tactic())
        assert isinstance(tid, str)
        assert len(tid) > 0

    def test_id_differs_from_original(self, file_context_store):
        """Template id must not equal the source tactic's original id."""
        tid = file_context_store.save_tactic(_fresh_tactic())
        assert tid != "t_original"

    def test_name_param_overrides_tactic_name(self, file_context_store):
        tid = file_context_store.save_tactic(_fresh_tactic(), name="Custom Name")
        tmpl = file_context_store.get_tactic(tid)
        assert tmpl["name"] == "Custom Name"

    def test_name_falls_back_to_tactic_name(self, file_context_store):
        tid = file_context_store.save_tactic(_fresh_tactic())
        tmpl = file_context_store.get_tactic(tid)
        assert tmpl["name"] == "Baseline timelapse"

    def test_live_state_stripped(self, file_context_store):
        """The 'live' key must not appear in the saved template."""
        tid = file_context_store.save_tactic(_fresh_tactic())
        tmpl = file_context_store.get_tactic(tid)
        assert "live" not in tmpl

    def test_run_state_not_preserved(self, file_context_store):
        """'state' from the source tactic (active) is not stored as the template state."""
        tid = file_context_store.save_tactic(_fresh_tactic())
        tmpl = file_context_store.get_tactic(tid)
        # Template itself doesn't carry 'state'; it's restored on apply
        assert tmpl.get("state") is None or tmpl.get("state") == "planned"

    def test_structure_preserved(self, file_context_store):
        tid = file_context_store.save_tactic(_fresh_tactic())
        tmpl = file_context_store.get_tactic(tid)
        assert tmpl["structure"] == {"cadence_s": 120}

    def test_kind_preserved(self, file_context_store):
        tid = file_context_store.save_tactic(_fresh_tactic())
        tmpl = file_context_store.get_tactic(tid)
        assert tmpl["kind"] == "standing_timelapse"

    def test_file_created(self, file_context_store):
        """A YAML file must exist in tactic_library/ after saving."""
        tid = file_context_store.save_tactic(_fresh_tactic())
        tl_dir = file_context_store.agent_dir / "tactic_library"
        files = list(tl_dir.iterdir())
        assert any(f.stem.startswith(tid) for f in files)

    def test_fires_context_updated(self, file_context_store):
        """save_tactic must fire CONTEXT_UPDATED."""
        from gently.core.event_bus import EventType, on

        seen = []
        unsub = on(EventType.CONTEXT_UPDATED, lambda e: seen.append(e))
        try:
            file_context_store.save_tactic(_fresh_tactic())
        finally:
            unsub()
        assert len(seen) >= 1


# ---------------------------------------------------------------------------
# list_tactics
# ---------------------------------------------------------------------------


class TestListTactics:
    def test_empty_when_none_saved(self, file_context_store):
        assert file_context_store.list_tactics() == []

    def test_one_entry_after_save(self, file_context_store):
        file_context_store.save_tactic(_fresh_tactic())
        result = file_context_store.list_tactics()
        assert len(result) == 1

    def test_two_entries_after_two_saves(self, file_context_store):
        file_context_store.save_tactic(_fresh_tactic(), name="Alpha")
        file_context_store.save_tactic(_fresh_tactic(), name="Beta")
        result = file_context_store.list_tactics()
        assert len(result) == 2

    def test_entries_contain_id_and_name(self, file_context_store):
        tid = file_context_store.save_tactic(_fresh_tactic())
        result = file_context_store.list_tactics()
        assert result[0]["id"] == tid
        assert result[0]["name"] == "Baseline timelapse"

    def test_ordered_newest_first(self, file_context_store):
        file_context_store.save_tactic(_fresh_tactic(), name="First")
        file_context_store.save_tactic(_fresh_tactic(), name="Second")
        result = file_context_store.list_tactics()
        # Newest should appear first (sorted by created_at desc)
        names = [r["name"] for r in result]
        assert names[0] == "Second"


# ---------------------------------------------------------------------------
# get_tactic
# ---------------------------------------------------------------------------


class TestGetTactic:
    def test_get_by_id(self, file_context_store):
        tid = file_context_store.save_tactic(_fresh_tactic())
        result = file_context_store.get_tactic(tid)
        assert result is not None
        assert result["id"] == tid

    def test_get_by_name(self, file_context_store):
        file_context_store.save_tactic(_fresh_tactic())
        result = file_context_store.get_tactic("Baseline timelapse")
        assert result is not None
        assert result["kind"] == "standing_timelapse"

    def test_get_unknown_returns_none(self, file_context_store):
        result = file_context_store.get_tactic("no_such_tactic")
        assert result is None

    def test_get_unknown_id_returns_none(self, file_context_store):
        file_context_store.save_tactic(_fresh_tactic())
        result = file_context_store.get_tactic("deadbeef")
        assert result is None


# ---------------------------------------------------------------------------
# apply_tactic
# ---------------------------------------------------------------------------


class TestApplyTactic:
    def test_apply_by_id_returns_dict(self, file_context_store):
        tid = file_context_store.save_tactic(_fresh_tactic())
        applied = file_context_store.apply_tactic(tid)
        assert isinstance(applied, dict)

    def test_apply_by_name_returns_dict(self, file_context_store):
        file_context_store.save_tactic(_fresh_tactic())
        applied = file_context_store.apply_tactic("Baseline timelapse")
        assert isinstance(applied, dict)

    def test_apply_unknown_returns_none(self, file_context_store):
        result = file_context_store.apply_tactic("no_such_tactic")
        assert result is None

    def test_apply_unknown_id_returns_none(self, file_context_store):
        file_context_store.save_tactic(_fresh_tactic())
        result = file_context_store.apply_tactic("deadbeef")
        assert result is None

    def test_applied_id_differs_from_template_id(self, file_context_store):
        """apply_tactic must return a FRESH id, distinct from the template id."""
        tid = file_context_store.save_tactic(_fresh_tactic())
        applied = file_context_store.apply_tactic(tid)
        assert applied["id"] != tid

    def test_applied_id_differs_from_source_tactic_id(self, file_context_store):
        """Applied id must not equal the original source tactic's id either."""
        tid = file_context_store.save_tactic(_fresh_tactic())
        applied = file_context_store.apply_tactic(tid)
        assert applied["id"] != "t_original"

    def test_applied_state_is_planned(self, file_context_store):
        """Returned tactic must have state='planned' regardless of template source state."""
        tid = file_context_store.save_tactic(_fresh_tactic())
        applied = file_context_store.apply_tactic(tid)
        assert applied["state"] == "planned"

    def test_applied_has_no_live_key(self, file_context_store):
        """Returned tactic must not carry any 'live' runtime state."""
        tid = file_context_store.save_tactic(_fresh_tactic())
        applied = file_context_store.apply_tactic(tid)
        assert "live" not in applied

    def test_applied_structure_matches_template(self, file_context_store):
        tid = file_context_store.save_tactic(_fresh_tactic())
        applied = file_context_store.apply_tactic(tid)
        assert applied["structure"] == {"cadence_s": 120}

    def test_applied_kind_matches_template(self, file_context_store):
        tid = file_context_store.save_tactic(_fresh_tactic())
        applied = file_context_store.apply_tactic(tid)
        assert applied["kind"] == "standing_timelapse"

    def test_applied_is_independent_copy(self, file_context_store):
        """Mutating the applied tactic must not affect a second apply call."""
        tid = file_context_store.save_tactic(_fresh_tactic())
        first = file_context_store.apply_tactic(tid)
        first["structure"]["cadence_s"] = 999
        second = file_context_store.apply_tactic(tid)
        assert second["structure"]["cadence_s"] == 120

    def test_two_applies_have_different_ids(self, file_context_store):
        """Each apply call returns a fresh id."""
        tid = file_context_store.save_tactic(_fresh_tactic())
        a1 = file_context_store.apply_tactic(tid)
        a2 = file_context_store.apply_tactic(tid)
        assert a1["id"] != a2["id"]

    def test_applied_no_slug_metadata(self, file_context_store):
        """Template-internal 'slug' must not appear in the applied tactic."""
        tid = file_context_store.save_tactic(_fresh_tactic())
        applied = file_context_store.apply_tactic(tid)
        assert "slug" not in applied

    def test_applied_no_created_at(self, file_context_store):
        """Template 'created_at' must not bleed into the applied tactic."""
        tid = file_context_store.save_tactic(_fresh_tactic())
        applied = file_context_store.apply_tactic(tid)
        assert "created_at" not in applied


# ---------------------------------------------------------------------------
# Fix 1 — scope round-trip
# ---------------------------------------------------------------------------


class TestScopeRoundTrip:
    def test_applied_scope_matches_source(self, file_context_store):
        """scope from source tactic must survive the save→apply round-trip."""
        tid = file_context_store.save_tactic(_fresh_tactic())
        applied = file_context_store.apply_tactic(tid)
        assert applied["scope"] == {"mode": "global"}

    def test_applied_no_scope_when_source_has_none(self, file_context_store):
        """If source tactic has no scope, applied tactic must not gain a scope key."""
        tactic = _fresh_tactic()
        tactic.pop("scope")
        tid = file_context_store.save_tactic(tactic)
        applied = file_context_store.apply_tactic(tid)
        assert "scope" not in applied


# ---------------------------------------------------------------------------
# Fix 2 — rationale preservation
# ---------------------------------------------------------------------------


class TestRationalePreservation:
    def test_rationale_survives_save_apply(self, file_context_store):
        """rationale from source tactic must survive the save→apply round-trip."""
        tid = file_context_store.save_tactic(_fresh_tactic())
        applied = file_context_store.apply_tactic(tid)
        assert applied.get("rationale") == "Establish pre-ramp cadence"

    def test_rationale_in_template(self, file_context_store):
        """rationale must be stored in the saved template, not only in description."""
        tid = file_context_store.save_tactic(_fresh_tactic())
        tmpl = file_context_store.get_tactic(tid)
        assert tmpl.get("rationale") == "Establish pre-ramp cadence"


# ---------------------------------------------------------------------------
# Fix 3 — deterministic name lookup (newest-wins on collision)
# ---------------------------------------------------------------------------


class TestGetTacticDeterministic:
    def test_name_collision_returns_newest(self, file_context_store):
        """When two tactics share a name, get_tactic(name) must return the newest."""
        import time

        t1 = _fresh_tactic()
        t1["structure"] = {"cadence_s": 60}
        file_context_store.save_tactic(t1, name="Shared Name")
        # Small pause to guarantee distinct created_at timestamps
        time.sleep(0.002)
        t2 = _fresh_tactic()
        t2["structure"] = {"cadence_s": 120}
        file_context_store.save_tactic(t2, name="Shared Name")

        result = file_context_store.get_tactic("Shared Name")
        assert result is not None
        # Newest entry (cadence_s=120) must win
        assert result["structure"]["cadence_s"] == 120
