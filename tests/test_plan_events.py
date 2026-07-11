"""Plan writes emit PLAN_UPDATED so the Plans UI can refresh live."""

from gently.core.event_bus import EventType, on
from gently.harness.memory.model import PlanItemStatus


class TestPlanUpdatedEvent:
    def test_plan_writes_emit_plan_updated(self, file_context_store):
        cs = file_context_store
        seen = []
        unsub = on(EventType.PLAN_UPDATED, lambda e: seen.append(e))
        try:
            cid = cs.create_campaign(description="test campaign")
            iid = cs.create_plan_item(campaign_id=cid, type="imaging", title="pilot")
            cs.update_plan_item(iid, status=PlanItemStatus.IN_PROGRESS, session_id="sess_1")
            cs.complete_plan_item(iid, outcome="done")
        finally:
            unsub()
        # create_plan_item + update_plan_item + complete_plan_item(→update) each fire it
        assert len(seen) >= 3
        assert any((e.data or {}).get("campaign_id") == cid for e in seen)

    def test_session_campaign_link_emits(self, file_context_store):
        cs = file_context_store
        seen = []
        unsub = on(EventType.PLAN_UPDATED, lambda e: seen.append(e))
        try:
            cid = cs.create_campaign(description="c")
            cs.link_session_campaign("sess_9", cid)
        finally:
            unsub()
        assert any((e.data or {}).get("campaign_id") == cid for e in seen)


class TestLinkPlanItemSession:
    def test_append_many_sessions(self, file_context_store):
        cs = file_context_store
        cid = cs.create_campaign(description="c")
        iid = cs.create_plan_item(campaign_id=cid, type="imaging", title="x")
        assert cs.link_plan_item_session(iid, "s1") is True
        assert cs.link_plan_item_session(iid, "s2") is True
        cs.link_plan_item_session(iid, "s1")  # duplicate — no-op
        item = cs.get_plan_item(iid)
        assert item.session_ids == ["s1", "s2"]  # appended, deduped
        assert item.session_id == "s2"  # latest, for back-compat readers
        assert item.status == PlanItemStatus.IN_PROGRESS  # PLANNED → IN_PROGRESS on first link

    def test_missing_item_returns_false(self, file_context_store):
        assert file_context_store.link_plan_item_session("nope", "s1") is False
