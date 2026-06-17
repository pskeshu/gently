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
