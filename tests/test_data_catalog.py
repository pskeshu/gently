"""
Tests for data catalog API routes.
"""

from unittest.mock import MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient


def _make_mock_store(sessions=None, embryos_per_session=None, gt_per_embryo=None):
    """Build a mock GentlyStore with controlled data."""
    store = MagicMock()
    sessions = sessions or []
    embryos_per_session = embryos_per_session or {}
    gt_per_embryo = gt_per_embryo or {}

    mock_sessions = []
    for sid, name in sessions:
        s = MagicMock()
        s.session_id = sid
        s.name = name
        s.created_at = "2024-01-01"
        s.last_active = "2024-01-01"
        mock_sessions.append(s)
    store.list_sessions.return_value = mock_sessions

    def list_embryos(sid):
        emb_ids = embryos_per_session.get(sid, [])
        result = []
        for eid in emb_ids:
            e = MagicMock()
            e.embryo_id = eid
            e.nickname = eid
            result.append(e)
        return result

    store.list_embryos.side_effect = list_embryos

    def list_volumes(sid, eid):
        return [MagicMock(timepoint=i) for i in range(3)]

    store.list_volumes.side_effect = list_volumes

    def get_ground_truth(sid, eid):
        stages = gt_per_embryo.get((sid, eid), [])
        result = []
        for stage in stages:
            gt = MagicMock()
            gt.stage = stage
            gt.start_tp = 0
            gt.end_tp = 10
            result.append(gt)
        return result

    store.get_ground_truth.side_effect = get_ground_truth

    def get_session(sid):
        for s in mock_sessions:
            if s.session_id == sid:
                return s
        return None

    store.get_session.side_effect = get_session

    return store


def _make_test_app(gently_store=None, context_store=None):
    """Create a FastAPI app with mesh routes registered."""
    from gently.mesh.routes import register_mesh_routes

    app = FastAPI()

    # Mock viz server
    viz_server = MagicMock()
    viz_server.app = app
    viz_server.gently_store = gently_store
    viz_server.context_store = context_store
    viz_server.event_bus = None

    # Mock mesh service (minimal)
    mesh_service = MagicMock()
    mesh_service.instance_id = "test-node"
    mesh_service._hostname = "test-host"
    mesh_service.get_local_info.return_value = {
        "instance_id": "test-node",
        "hostname": "test-host",
        "capabilities": {},
        "status": {},
    }
    mesh_service.get_peers.return_value = []
    mesh_service.get_all_peers.return_value = []
    mesh_service.pairing_manager = None  # no auth for testing
    mesh_service.verse_map = MagicMock()
    mesh_service.verse_map.get_all_peers.return_value = []
    mesh_service.verse_map.get_online_peers.return_value = []
    mesh_service.verse_map.get_offline_peers.return_value = []
    mesh_service.verse_map.find_resource.return_value = []

    register_mesh_routes(viz_server, mesh_service)
    return app


class TestSessionsEndpoint:
    def test_empty_store(self):
        store = _make_mock_store()
        app = _make_test_app(gently_store=store)
        client = TestClient(app)
        resp = client.get("/api/data/sessions")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 0
        assert data["sessions"] == []

    def test_sessions_with_data(self):
        store = _make_mock_store(
            sessions=[("s1", "Session 1"), ("s2", "Session 2")],
            embryos_per_session={"s1": ["e1", "e2"], "s2": ["e3"]},
        )
        app = _make_test_app(gently_store=store)
        client = TestClient(app)
        resp = client.get("/api/data/sessions")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 2
        assert data["sessions"][0]["session_id"] == "s1"
        assert data["sessions"][0]["embryo_count"] == 2
        assert data["sessions"][1]["embryo_count"] == 1

    def test_no_store(self):
        app = _make_test_app(gently_store=None)
        client = TestClient(app)
        resp = client.get("/api/data/sessions")
        assert resp.status_code == 200
        assert resp.json()["count"] == 0


class TestSessionDetailEndpoint:
    def test_session_detail(self):
        store = _make_mock_store(
            sessions=[("s1", "Session 1")],
            embryos_per_session={"s1": ["e1", "e2"]},
            gt_per_embryo={("s1", "e1"): ["early"]},
        )
        app = _make_test_app(gently_store=store)
        client = TestClient(app)
        resp = client.get("/api/data/sessions/s1")
        assert resp.status_code == 200
        data = resp.json()
        assert data["session_id"] == "s1"
        assert len(data["embryos"]) == 2
        # e1 has ground truth, e2 does not
        e1 = next(e for e in data["embryos"] if e["embryo_id"] == "e1")
        e2 = next(e for e in data["embryos"] if e["embryo_id"] == "e2")
        assert e1["has_ground_truth"] is True
        assert e2["has_ground_truth"] is False

    def test_session_not_found(self):
        store = _make_mock_store(sessions=[])
        store.get_session.side_effect = lambda sid: None
        app = _make_test_app(gently_store=store)
        client = TestClient(app)
        resp = client.get("/api/data/sessions/nonexistent")
        assert resp.status_code == 404


class TestCoverageEndpoint:
    def test_coverage_with_data(self):
        store = _make_mock_store(
            sessions=[("s1", "Session 1")],
            embryos_per_session={"s1": ["e1", "e2", "e3"]},
            gt_per_embryo={
                ("s1", "e1"): ["early"],
                ("s1", "e2"): ["comma"],
            },
        )
        app = _make_test_app(gently_store=store)
        client = TestClient(app)
        resp = client.get("/api/data/coverage")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_embryos"] == 3
        assert data["annotated_embryos"] == 2
        assert "early" in data["stage_counts"]
        assert "comma" in data["stage_counts"]

    def test_coverage_no_store(self):
        app = _make_test_app(gently_store=None)
        client = TestClient(app)
        resp = client.get("/api/data/coverage")
        assert resp.status_code == 200
        assert resp.json()["total_embryos"] == 0


class TestStagesEndpoint:
    def test_stages_distribution(self):
        store = _make_mock_store(
            sessions=[("s1", "S1"), ("s2", "S2")],
            embryos_per_session={"s1": ["e1"], "s2": ["e2"]},
            gt_per_embryo={
                ("s1", "e1"): ["early", "comma"],
                ("s2", "e2"): ["early"],
            },
        )
        app = _make_test_app(gently_store=store)
        client = TestClient(app)
        resp = client.get("/api/data/stages")
        assert resp.status_code == 200
        data = resp.json()
        assert data["stage_distribution"]["early"] == 2
        assert data["stage_distribution"]["comma"] == 1
        assert "s1" in data["by_session"]

    def test_stages_no_store(self):
        app = _make_test_app(gently_store=None)
        client = TestClient(app)
        resp = client.get("/api/data/stages")
        assert resp.status_code == 200
        assert resp.json()["stage_distribution"] == {}


class TestVerseMapRoutes:
    def test_verse_map_endpoint(self):
        app = _make_test_app()
        client = TestClient(app)
        resp = client.get("/api/mesh/verse-map")
        assert resp.status_code == 200
        data = resp.json()
        assert "peers" in data
        assert "online_count" in data

    def test_verse_map_resources(self):
        app = _make_test_app()
        client = TestClient(app)
        resp = client.get("/api/mesh/verse-map/resources/has_gpu")
        assert resp.status_code == 200
        data = resp.json()
        assert data["capability"] == "has_gpu"
