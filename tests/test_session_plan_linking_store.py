"""
FileContextStore: session ↔ plan-item link/delink (Task 1 — data layer).

Tests:
- link a session to items in 2 different campaigns
- get_plan_items_for_session returns both items
- unlink_plan_item_session removes one (the other remains)
- get_plan_items_for_session returns only the remaining item after unlink
- unlinking a session that isn't linked → False, no side-effects
- back-compat scalar session_id is cleared when the unlinked session matched it
"""

import pytest

from gently.harness.memory.model import PlanItemType

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_item(store, campaign_id: str, title: str) -> str:
    """Create a bench plan item and return its id."""
    return store.create_plan_item(
        campaign_id=campaign_id,
        type=PlanItemType.BENCH.value,
        title=title,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def two_campaigns(file_context_store):
    """Two active campaigns, each with one plan item. Returns (store, cid1, cid2, item1, item2)."""
    store = file_context_store
    cid1 = store.create_campaign(description="Campaign Alpha")
    cid2 = store.create_campaign(description="Campaign Beta")
    item1 = _make_item(store, cid1, "Step A — bench prep")
    item2 = _make_item(store, cid2, "Step B — gel run")
    return store, cid1, cid2, item1, item2


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLinkAndReverseQuery:
    def test_link_and_get_plan_items_for_session(self, two_campaigns):
        """Linking a session to 2 items across 2 campaigns; reverse query returns both."""
        store, _cid1, _cid2, item1, item2 = two_campaigns
        session_id = "sess_abc"

        store.link_plan_item_session(item1, session_id, set_in_progress=False)
        store.link_plan_item_session(item2, session_id, set_in_progress=False)

        items = store.get_plan_items_for_session(session_id)
        ids = {it.id for it in items}
        assert item1 in ids
        assert item2 in ids
        assert len(ids) == 2

    def test_session_not_in_unrelated_item(self, two_campaigns):
        """Items not linked to the session don't appear in the reverse query."""
        store, _cid1, _cid2, item1, _item2 = two_campaigns
        session_id = "sess_xyz"

        store.link_plan_item_session(item1, session_id, set_in_progress=False)
        items = store.get_plan_items_for_session(session_id)
        assert len(items) == 1
        assert items[0].id == item1

    def test_unknown_session_returns_empty(self, two_campaigns):
        """A session with no links → empty list, no error."""
        store, *_ = two_campaigns
        result = store.get_plan_items_for_session("sess_ghost")
        assert result == []


class TestUnlink:
    def test_unlink_removes_from_one_campaign(self, two_campaigns):
        """Unlinking a session from one item leaves the other item still linked."""
        store, _cid1, _cid2, item1, item2 = two_campaigns
        session_id = "sess_def"

        store.link_plan_item_session(item1, session_id, set_in_progress=False)
        store.link_plan_item_session(item2, session_id, set_in_progress=False)

        result = store.unlink_plan_item_session(item1, session_id)
        assert result is True

        remaining = store.get_plan_items_for_session(session_id)
        ids = {it.id for it in remaining}
        assert item1 not in ids
        assert item2 in ids

    def test_unlink_idempotent_returns_false(self, two_campaigns):
        """Unlinking a session that isn't linked returns False without crashing."""
        store, _cid1, _cid2, item1, _item2 = two_campaigns
        result = store.unlink_plan_item_session(item1, "sess_nothere")
        assert result is False

    def test_unlink_unknown_item_returns_false(self, two_campaigns):
        """Unlinking from an item that doesn't exist returns False."""
        store, *_ = two_campaigns
        result = store.unlink_plan_item_session("item_ghost", "sess_x")
        assert result is False

    def test_unlink_no_side_effect_when_not_linked(self, two_campaigns):
        """After a no-op unlink, the item's session_ids are unchanged."""
        store, _cid1, _cid2, item1, _item2 = two_campaigns
        session_id = "sess_linked"
        store.link_plan_item_session(item1, session_id, set_in_progress=False)

        store.unlink_plan_item_session(item1, "sess_other")  # different session — no-op

        item = store.get_plan_item(item1)
        assert session_id in item.session_ids

    def test_full_unlink_clears_all_sessions(self, two_campaigns):
        """After unlinking all sessions, get_plan_items_for_session returns nothing."""
        store, _cid1, _cid2, item1, _item2 = two_campaigns
        session_id = "sess_ghi"

        store.link_plan_item_session(item1, session_id, set_in_progress=False)
        store.unlink_plan_item_session(item1, session_id)

        remaining = store.get_plan_items_for_session(session_id)
        assert remaining == []


class TestBackCompatSessionId:
    def test_back_compat_cleared_on_unlink_when_matched(self, two_campaigns):
        """Back-compat scalar session_id is set to None when the last session is unlinked."""
        store, _cid1, _cid2, item1, _item2 = two_campaigns
        session_id = "sess_jkl"

        store.link_plan_item_session(item1, session_id, set_in_progress=False)
        item_before = store.get_plan_item(item1)
        assert item_before.session_id == session_id  # back-compat set by link

        store.unlink_plan_item_session(item1, session_id)

        item_after = store.get_plan_item(item1)
        assert item_after.session_id is None
        assert item_after.session_ids == []

    def test_back_compat_set_to_most_recent_remaining(self, two_campaigns):
        """When one of two sessions is unlinked, back-compat session_id = remaining session."""
        store, _cid1, _cid2, item1, _item2 = two_campaigns
        sid1 = "sess_first"
        sid2 = "sess_second"

        store.link_plan_item_session(item1, sid1, set_in_progress=False)
        store.link_plan_item_session(item1, sid2, set_in_progress=False)

        store.unlink_plan_item_session(item1, sid1)

        item = store.get_plan_item(item1)
        assert sid2 in item.session_ids
        assert item.session_id == sid2  # most recent remaining
        assert sid1 not in item.session_ids


class TestBackCompatLegacyScalar:
    """Back-compat path: items written before session_ids list existed (scalar only)."""

    def _inject_legacy_item(self, store, campaign_id: str, item_id: str, session_id: str):
        """Overwrite a plan item's YAML record to simulate legacy format:
        session_id set but session_ids absent/empty — as written by pre-list code."""
        loc = store._find_plan_item_location(item_id)
        assert loc is not None, "item must exist before injecting legacy format"
        cid, items, idx = loc
        items[idx]["session_id"] = session_id
        items[idx]["session_ids"] = []  # empty list = pre-list legacy state
        store._write_plan_items(cid, items)

    def test_legacy_scalar_found_by_reverse_query(self, two_campaigns):
        """get_plan_items_for_session finds an item with only scalar session_id set."""
        store, cid1, _cid2, item1, _item2 = two_campaigns
        session_id = "sess_legacy"

        # Inject legacy-format record (scalar session_id, empty session_ids)
        self._inject_legacy_item(store, cid1, item1, session_id)

        items = store.get_plan_items_for_session(session_id)
        assert any(it.id == item1 for it in items), (
            "expected item1 in results for legacy scalar session_id"
        )

    def test_legacy_scalar_unlinked_by_unlink(self, two_campaigns):
        """unlink_plan_item_session removes a legacy-scalar item and returns True."""
        store, cid1, _cid2, item1, _item2 = two_campaigns
        session_id = "sess_legacy2"

        self._inject_legacy_item(store, cid1, item1, session_id)

        result = store.unlink_plan_item_session(item1, session_id)
        assert result is True

        # Must no longer appear in reverse query
        remaining = store.get_plan_items_for_session(session_id)
        assert not any(it.id == item1 for it in remaining), (
            "item1 should be removed after unlinking legacy scalar session"
        )
