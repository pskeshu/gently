"""
TDD: seed_operation_plan_from_plan_item

Resolution path under test:
  session_id
  → get_campaign_ids_for_session  (reads session_intents/{session_id}.yaml)
  → get_plan_items(campaign_id=cid)
  → find PlanItem where session_id ∈ plan_item.session_ids
        AND plan_item.imaging_spec.tactics is non-empty
  → get_campaign(plan_item.campaign_id)  for goal text
  → build + write Operation Plan via set_operation_plan

Scenarios
---------
1. Two outline tactics  → seeded plan with 2 ``planned`` tactics
2. Plan-level fields (goal, plan_item_id, campaign_id) populated from item/campaign
3. Idempotent — existing plan with active tactic: not clobbered, returns None
4. Idempotent — existing plan with done tactic: not clobbered, returns None
5. No campaign linkage → returns None, no write
6. Plan item linked but imaging_spec.tactics is empty → returns None, no write
"""

from gently.app.tools.operation_plan_seed import seed_operation_plan_from_plan_item

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_campaign_and_item_with_tactics(
    cs,
    session_id: str,
    *,
    target: str = "50 hatching events",
):
    """Create a campaign + imaging plan item with 2 tactics, linked to *session_id*."""
    cid = cs.create_campaign(
        description="Capture hatching events in WT",
        shorthand="hatching-wt",
        target=target,
    )
    spec = {
        "strain": "N2",
        "num_embryos": 4,
        "tactics": [
            {"kind": "standing_timelapse", "name": "Baseline timelapse"},
            {
                "kind": "reactive_monitor",
                "name": "Hatching monitor",
                "target": "embryo_hatching",
                "scope": {"mode": "global"},
            },
        ],
    }
    item_id = cs.create_plan_item(
        campaign_id=cid,
        type="imaging",
        title="WT hatching pilot",
        spec=spec,
    )
    cs.link_session_campaign(session_id, cid)
    cs.link_plan_item_session(item_id, session_id)
    return cid, item_id


# ---------------------------------------------------------------------------
# Basic seeding
# ---------------------------------------------------------------------------


class TestSeedBasic:
    def test_returns_plan_dict(self, file_context_store):
        """seed returns a non-None dict when a linked plan item with tactics exists."""
        cs = file_context_store
        sid = "seed_basic_01"
        cs.create_session_intent(sid, planned_intent="hatching pilot")
        _make_campaign_and_item_with_tactics(cs, sid)

        plan = seed_operation_plan_from_plan_item(cs, sid)

        assert plan is not None
        assert isinstance(plan, dict)

    def test_two_outline_entries_produce_two_tactics(self, file_context_store):
        """An outline with 2 entries produces exactly 2 tactics."""
        cs = file_context_store
        sid = "seed_basic_02"
        cs.create_session_intent(sid)
        _make_campaign_and_item_with_tactics(cs, sid)

        plan = seed_operation_plan_from_plan_item(cs, sid)

        assert len(plan["tactics"]) == 2

    def test_all_seeded_tactics_are_planned(self, file_context_store):
        """Every seeded tactic starts in state='planned'."""
        cs = file_context_store
        sid = "seed_basic_03"
        cs.create_session_intent(sid)
        _make_campaign_and_item_with_tactics(cs, sid)

        plan = seed_operation_plan_from_plan_item(cs, sid)

        assert all(t["state"] == "planned" for t in plan["tactics"])

    def test_tactic_kind_and_name_from_outline(self, file_context_store):
        """Tactics carry the kind and name from the outline entries."""
        cs = file_context_store
        sid = "seed_basic_04"
        cs.create_session_intent(sid)
        _make_campaign_and_item_with_tactics(cs, sid)

        plan = seed_operation_plan_from_plan_item(cs, sid)

        names = [t["name"] for t in plan["tactics"]]
        kinds = [t["kind"] for t in plan["tactics"]]
        assert "Baseline timelapse" in names
        assert "Hatching monitor" in names
        assert "standing_timelapse" in kinds
        assert "reactive_monitor" in kinds

    def test_tactic_ids_generated_and_distinct(self, file_context_store):
        """Each tactic has a non-empty unique id."""
        cs = file_context_store
        sid = "seed_basic_05"
        cs.create_session_intent(sid)
        _make_campaign_and_item_with_tactics(cs, sid)

        plan = seed_operation_plan_from_plan_item(cs, sid)

        ids = [t["id"] for t in plan["tactics"]]
        assert all(ids)
        assert len(set(ids)) == len(ids)

    def test_plan_persisted_in_store(self, file_context_store):
        """The seeded plan is retrievable via get_operation_plan."""
        cs = file_context_store
        sid = "seed_basic_06"
        cs.create_session_intent(sid)
        _make_campaign_and_item_with_tactics(cs, sid)

        seed_operation_plan_from_plan_item(cs, sid)

        stored = cs.get_operation_plan(sid)
        assert stored is not None
        assert len(stored["tactics"]) == 2


# ---------------------------------------------------------------------------
# Plan-level metadata
# ---------------------------------------------------------------------------


class TestSeedMetadata:
    def test_goal_from_campaign_target(self, file_context_store):
        """goal field is taken from campaign.target."""
        cs = file_context_store
        sid = "seed_meta_01"
        cs.create_session_intent(sid)
        _make_campaign_and_item_with_tactics(cs, sid, target="50 hatching events")

        plan = seed_operation_plan_from_plan_item(cs, sid)

        assert plan["goal"] == "50 hatching events"

    def test_goal_falls_back_to_campaign_description(self, file_context_store):
        """When campaign.target is None, goal falls back to campaign.description."""
        cs = file_context_store
        sid = "seed_meta_02"
        cs.create_session_intent(sid)
        cid = cs.create_campaign(description="Characterise WT hatching")
        spec = {"tactics": [{"kind": "standing_timelapse", "name": "Timelapse"}]}
        item_id = cs.create_plan_item(campaign_id=cid, type="imaging", title="Pilot", spec=spec)
        cs.link_session_campaign(sid, cid)
        cs.link_plan_item_session(item_id, sid)

        plan = seed_operation_plan_from_plan_item(cs, sid)

        assert plan["goal"] == "Characterise WT hatching"

    def test_plan_item_id_set(self, file_context_store):
        """plan_item_id on the plan matches the linked plan item's id."""
        cs = file_context_store
        sid = "seed_meta_03"
        cs.create_session_intent(sid)
        cid, item_id = _make_campaign_and_item_with_tactics(cs, sid)

        plan = seed_operation_plan_from_plan_item(cs, sid)

        assert plan["plan_item_id"] == item_id

    def test_campaign_id_set(self, file_context_store):
        """campaign_id on the plan matches the campaign."""
        cs = file_context_store
        sid = "seed_meta_04"
        cs.create_session_intent(sid)
        cid, _ = _make_campaign_and_item_with_tactics(cs, sid)

        plan = seed_operation_plan_from_plan_item(cs, sid)

        assert plan["campaign_id"] == cid

    def test_title_from_plan_item(self, file_context_store):
        """Plan title comes from the plan item's title."""
        cs = file_context_store
        sid = "seed_meta_05"
        cs.create_session_intent(sid)
        _make_campaign_and_item_with_tactics(cs, sid)

        plan = seed_operation_plan_from_plan_item(cs, sid)

        assert plan["title"] == "WT hatching pilot"


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------


class TestIdempotency:
    def test_active_tactic_not_clobbered(self, file_context_store):
        """If the plan has an active tactic, seed returns None and doesn't overwrite."""
        cs = file_context_store
        sid = "seed_idem_01"
        cs.create_session_intent(sid)
        _make_campaign_and_item_with_tactics(cs, sid)

        existing = {
            "session_id": sid,
            "title": "Agent-declared plan",
            "goal": "live goal",
            "tactics": [
                {
                    "id": "t_live",
                    "name": "Live tactic",
                    "kind": "standing_timelapse",
                    "state": "active",
                }
            ],
            "updated_at": "2026-06-28T10:00:00Z",
            "updated_reason": "agent declared",
        }
        cs.set_operation_plan(sid, existing)

        result = seed_operation_plan_from_plan_item(cs, sid)

        assert result is None
        stored = cs.get_operation_plan(sid)
        assert stored["title"] == "Agent-declared plan"
        assert stored["goal"] == "live goal"

    def test_done_tactic_not_clobbered(self, file_context_store):
        """If the plan has a done tactic, seed returns None and doesn't overwrite."""
        cs = file_context_store
        sid = "seed_idem_02"
        cs.create_session_intent(sid)
        _make_campaign_and_item_with_tactics(cs, sid)

        existing = {
            "session_id": sid,
            "title": "Completed plan",
            "goal": "done goal",
            "tactics": [
                {
                    "id": "t_done",
                    "name": "Done tactic",
                    "kind": "oneshot",
                    "state": "done",
                }
            ],
            "updated_at": "2026-06-28T10:00:00Z",
            "updated_reason": "agent declared",
        }
        cs.set_operation_plan(sid, existing)

        result = seed_operation_plan_from_plan_item(cs, sid)

        assert result is None
        stored = cs.get_operation_plan(sid)
        assert stored["title"] == "Completed plan"

    def test_all_planned_existing_plan_adds_missing_tactics(self, file_context_store):
        """If existing plan has only planned tactics, new outline entries are added."""
        cs = file_context_store
        sid = "seed_idem_03"
        cs.create_session_intent(sid)
        _make_campaign_and_item_with_tactics(cs, sid)

        # Pre-seed with only the first tactic (by name)
        existing = {
            "session_id": sid,
            "title": "Partial plan",
            "goal": "partial goal",
            "tactics": [
                {
                    "id": "t_existing",
                    "name": "Baseline timelapse",
                    "kind": "standing_timelapse",
                    "state": "planned",
                }
            ],
            "updated_at": "2026-06-28T09:00:00Z",
            "updated_reason": "partial seed",
        }
        cs.set_operation_plan(sid, existing)

        result = seed_operation_plan_from_plan_item(cs, sid)

        # The second outline entry ("Hatching monitor") should have been added
        assert result is not None
        assert len(result["tactics"]) == 2
        names = [t["name"] for t in result["tactics"]]
        assert "Hatching monitor" in names

    def test_all_tactics_already_present_returns_none(self, file_context_store):
        """If all outline tactics are already in the plan (by name), returns None."""
        cs = file_context_store
        sid = "seed_idem_04"
        cs.create_session_intent(sid)
        _make_campaign_and_item_with_tactics(cs, sid)

        # Pre-seed with both outline names already present
        existing = {
            "session_id": sid,
            "title": "Full plan",
            "goal": "full goal",
            "tactics": [
                {
                    "id": "t1",
                    "name": "Baseline timelapse",
                    "kind": "standing_timelapse",
                    "state": "planned",
                },
                {
                    "id": "t2",
                    "name": "Hatching monitor",
                    "kind": "reactive_monitor",
                    "state": "planned",
                },
            ],
            "updated_at": "2026-06-28T09:00:00Z",
            "updated_reason": "already seeded",
        }
        cs.set_operation_plan(sid, existing)

        result = seed_operation_plan_from_plan_item(cs, sid)

        assert result is None
        # Plan unchanged
        stored = cs.get_operation_plan(sid)
        assert len(stored["tactics"]) == 2


# ---------------------------------------------------------------------------
# No linkage
# ---------------------------------------------------------------------------


class TestNoLinkage:
    def test_no_campaign_linkage_returns_none(self, file_context_store):
        """Session with no campaign linkage → None, nothing written."""
        cs = file_context_store
        sid = "seed_nolink_01"
        cs.create_session_intent(sid, planned_intent="standalone")

        result = seed_operation_plan_from_plan_item(cs, sid)

        assert result is None
        assert cs.get_operation_plan(sid) is None

    def test_no_session_intent_returns_none(self, file_context_store):
        """Session with no session_intent YAML → None."""
        cs = file_context_store

        result = seed_operation_plan_from_plan_item(cs, "ghost_session")

        assert result is None

    def test_plan_item_without_tactics_returns_none(self, file_context_store):
        """Plan item linked to session but imaging_spec.tactics is empty → None."""
        cs = file_context_store
        sid = "seed_notactics_01"
        cs.create_session_intent(sid)

        cid = cs.create_campaign(description="Campaign — no tactics in spec")
        item_id = cs.create_plan_item(
            campaign_id=cid,
            type="imaging",
            title="Item without tactics",
            spec={"strain": "N2"},  # no "tactics" key → empty list on ImagingSpec
        )
        cs.link_session_campaign(sid, cid)
        cs.link_plan_item_session(item_id, sid)

        result = seed_operation_plan_from_plan_item(cs, sid)

        assert result is None
        assert cs.get_operation_plan(sid) is None

    def test_non_imaging_plan_item_returns_none(self, file_context_store):
        """Non-imaging plan item (no imaging_spec) → None."""
        cs = file_context_store
        sid = "seed_bench_01"
        cs.create_session_intent(sid)

        cid = cs.create_campaign(description="Bench work campaign")
        item_id = cs.create_plan_item(
            campaign_id=cid,
            type="bench",
            title="Prep samples",
        )
        cs.link_session_campaign(sid, cid)
        cs.link_plan_item_session(item_id, sid)

        result = seed_operation_plan_from_plan_item(cs, sid)

        assert result is None
