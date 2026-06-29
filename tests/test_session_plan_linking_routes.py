"""Route tests: session ↔ plan link/delink HTTP endpoints (Task 2 of F).

Endpoints under test:
  POST   /api/campaigns/{cid}/items/{item_id}/sessions        → link session to item
  DELETE /api/campaigns/{cid}/items/{item_id}/sessions/{sid}  → delink session from item
  GET    /api/sessions/{session_id}/plans                     → plan items for a session
"""

from datetime import datetime
from unittest.mock import MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from gently.harness.memory.model import PlanItem, PlanItemStatus, PlanItemType, SessionIntent

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_item(
    item_id: str,
    campaign_id: str,
    title: str,
    status: PlanItemStatus = PlanItemStatus.PLANNED,
) -> PlanItem:
    return PlanItem(
        id=item_id,
        campaign_id=campaign_id,
        type=PlanItemType.BENCH,
        title=title,
        status=status,
    )


def _make_app(mock_cs):
    """FastAPI test app with campaign + session routers wired to mock_cs."""
    from gently.ui.web.routes.campaigns import create_router as campaign_router
    from gently.ui.web.routes.sessions import create_router as session_router

    app = FastAPI()

    class _Server:
        context_store = mock_cs
        agent_bridge = None
        gently_store = None

    server = _Server()
    app.include_router(campaign_router(server))
    app.include_router(session_router(server))
    return app


def _mock_cs(item=None, sessions=None):
    """Build a mock context store with sensible defaults."""
    cs = MagicMock()
    cs.resolve_campaign.return_value = MagicMock(id="c1")
    cs.get_plan_item.return_value = item
    cs.get_sessions_for_campaign.return_value = sessions if sessions is not None else []
    cs.unlink_plan_item_session.return_value = True
    cs.get_plan_items_for_session.return_value = []
    return cs


# ---------------------------------------------------------------------------
# POST /api/campaigns/{cid}/items/{item_id}/sessions
# ---------------------------------------------------------------------------


class TestPostLinkSession:
    def test_calls_link_plan_item_session(self):
        item = _make_item("i1", "c1", "Step A")
        cs = _mock_cs(item=item)
        client = TestClient(_make_app(cs))

        r = client.post("/api/campaigns/c1/items/i1/sessions", json={"session_id": "s1"})

        assert r.status_code == 200
        cs.link_plan_item_session.assert_called_once_with("i1", "s1")

    def test_calls_link_session_campaign_with_resolved_id(self):
        item = _make_item("i1", "c1", "Step A")
        cs = _mock_cs(item=item)
        cs.resolve_campaign.return_value = MagicMock(id="c1")
        client = TestClient(_make_app(cs))

        client.post("/api/campaigns/c1/items/i1/sessions", json={"session_id": "s1"})

        cs.link_session_campaign.assert_called_once_with("s1", "c1")

    def test_returns_sessions_list(self):
        item = _make_item("i1", "c1", "Step A")
        cs = _mock_cs(item=item, sessions=[])
        client = TestClient(_make_app(cs))

        r = client.post("/api/campaigns/c1/items/i1/sessions", json={"session_id": "s1"})

        assert r.status_code == 200
        body = r.json()
        assert "sessions" in body
        assert isinstance(body["sessions"], list)

    def test_returns_serialized_session_intents(self):
        # Item must have session_ids=["s1"] to simulate the post-link state
        # (mock link_plan_item_session doesn't mutate the item; we configure
        # get_plan_item to return the post-link view so the item-scoped filter works).
        item = PlanItem(
            id="i1",
            campaign_id="c1",
            type=PlanItemType.BENCH,
            title="Step A",
            status=PlanItemStatus.PLANNED,
            session_ids=["s1"],
        )
        si = SessionIntent(
            session_id="s1",
            campaign_ids=["c1"],
            created_at=datetime(2026, 6, 1),
        )
        cs = _mock_cs(item=item, sessions=[si])
        client = TestClient(_make_app(cs))

        r = client.post("/api/campaigns/c1/items/i1/sessions", json={"session_id": "s1"})

        assert r.status_code == 200
        sessions = r.json()["sessions"]
        assert len(sessions) == 1
        assert sessions[0]["session_id"] == "s1"
        assert sessions[0]["campaign_ids"] == ["c1"]

    def test_missing_campaign_is_404(self):
        cs = _mock_cs()
        cs.resolve_campaign.return_value = None
        client = TestClient(_make_app(cs))

        r = client.post("/api/campaigns/nope/items/i1/sessions", json={"session_id": "s1"})

        assert r.status_code == 404

    def test_missing_item_is_404(self):
        cs = _mock_cs(item=None)
        client = TestClient(_make_app(cs))

        r = client.post("/api/campaigns/c1/items/nope/sessions", json={"session_id": "s1"})

        assert r.status_code == 404

    def test_missing_session_id_body_is_400(self):
        item = _make_item("i1", "c1", "Step A")
        cs = _mock_cs(item=item)
        client = TestClient(_make_app(cs))

        r = client.post("/api/campaigns/c1/items/i1/sessions", json={})

        assert r.status_code == 400


# ---------------------------------------------------------------------------
# DELETE /api/campaigns/{cid}/items/{item_id}/sessions/{session_id}
# ---------------------------------------------------------------------------


class TestDeleteUnlinkSession:
    def test_calls_unlink_plan_item_session(self):
        item = _make_item("i1", "c1", "Step A")
        cs = _mock_cs(item=item)
        cs.unlink_plan_item_session.return_value = True
        client = TestClient(_make_app(cs))

        r = client.delete("/api/campaigns/c1/items/i1/sessions/s1")

        assert r.status_code == 200
        cs.unlink_plan_item_session.assert_called_once_with("i1", "s1")

    def test_returns_unlinked_true(self):
        item = _make_item("i1", "c1", "Step A")
        cs = _mock_cs(item=item)
        cs.unlink_plan_item_session.return_value = True
        client = TestClient(_make_app(cs))

        r = client.delete("/api/campaigns/c1/items/i1/sessions/s1")

        assert r.status_code == 200
        assert r.json()["unlinked"] is True

    def test_returns_unlinked_false_when_not_linked(self):
        item = _make_item("i1", "c1", "Step A")
        cs = _mock_cs(item=item)
        cs.unlink_plan_item_session.return_value = False
        client = TestClient(_make_app(cs))

        r = client.delete("/api/campaigns/c1/items/i1/sessions/s99")

        assert r.status_code == 200
        assert r.json()["unlinked"] is False

    def test_missing_item_is_404(self):
        cs = _mock_cs(item=None)
        client = TestClient(_make_app(cs))

        r = client.delete("/api/campaigns/c1/items/nope/sessions/s1")

        assert r.status_code == 404

    def test_missing_campaign_is_404(self):
        cs = _mock_cs()
        cs.resolve_campaign.return_value = None
        client = TestClient(_make_app(cs))

        r = client.delete("/api/campaigns/nope/items/i1/sessions/s1")

        assert r.status_code == 404


# ---------------------------------------------------------------------------
# GET /api/sessions/{session_id}/plans
# ---------------------------------------------------------------------------


class TestGetSessionPlans:
    def test_returns_mapped_plan_items(self):
        cs = _mock_cs()
        cs.get_plan_items_for_session.return_value = [
            _make_item("i1", "c1", "Step A", PlanItemStatus.PLANNED),
            _make_item("i2", "c2", "Step B", PlanItemStatus.IN_PROGRESS),
        ]
        client = TestClient(_make_app(cs))

        r = client.get("/api/sessions/s1/plans")

        assert r.status_code == 200
        plans = r.json()["plans"]
        assert len(plans) == 2
        assert plans[0] == {
            "id": "i1",
            "title": "Step A",
            "campaign_id": "c1",
            "status": "planned",
        }
        assert plans[1] == {
            "id": "i2",
            "title": "Step B",
            "campaign_id": "c2",
            "status": "in_progress",
        }

    def test_empty_list_for_unknown_session(self):
        cs = _mock_cs()
        cs.get_plan_items_for_session.return_value = []
        client = TestClient(_make_app(cs))

        r = client.get("/api/sessions/ghost/plans")

        assert r.status_code == 200
        assert r.json()["plans"] == []

    def test_calls_store_with_session_id(self):
        cs = _mock_cs()
        client = TestClient(_make_app(cs))

        client.get("/api/sessions/sess_abc/plans")

        cs.get_plan_items_for_session.assert_called_once_with("sess_abc")

    def test_graceful_when_no_context_store(self):
        from gently.ui.web.routes.sessions import create_router as session_router

        app = FastAPI()

        class _ServerNoCS:
            context_store = None
            agent_bridge = None
            gently_store = None

        server = _ServerNoCS()
        app.include_router(session_router(server))
        client = TestClient(app)

        r = client.get("/api/sessions/s1/plans")

        assert r.status_code == 200
        assert r.json()["plans"] == []
