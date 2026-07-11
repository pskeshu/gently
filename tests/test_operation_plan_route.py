from unittest.mock import MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from gently.ui.web.routes.operation_plan import create_router

_SAMPLE_PLAN = {
    "tactics": [
        {"id": "t1", "description": "Start acquisition", "status": "done"},
        {"id": "t2", "description": "Monitor drift", "status": "active"},
    ]
}


def _server(plan, sessions=(("sess-1",),), context_store_available=True):
    gently_store = MagicMock()
    gently_store.list_sessions.return_value = [{"session_id": sid} for (sid,) in sessions]

    context_store = MagicMock() if context_store_available else None
    if context_store is not None:
        context_store.get_operation_plan.return_value = plan

    srv = MagicMock()
    srv.gently_store = gently_store
    srv.context_store = context_store
    return srv, gently_store, context_store


def _client(server):
    app = FastAPI()
    app.include_router(create_router(server))
    return TestClient(app)


def test_returns_stored_plan():
    srv, _, cs = _server(_SAMPLE_PLAN)
    r = _client(srv).get("/api/operation_plan/sess-1")
    assert r.status_code == 200
    body = r.json()
    assert body["available"] is True
    assert body["session_id"] == "sess-1"
    assert body["plan"]["tactics"][0]["id"] == "t1"
    cs.get_operation_plan.assert_called_with("sess-1")


def test_returns_unavailable_when_no_plan():
    srv, _, cs = _server(None)
    r = _client(srv).get("/api/operation_plan/sess-1")
    assert r.status_code == 200
    body = r.json()
    assert body["available"] is False
    assert body["plan"] is None


def test_current_resolves_newest():
    srv, gently_store, cs = _server(_SAMPLE_PLAN, sessions=(("newest-session",),))
    r = _client(srv).get("/api/operation_plan/current")
    assert r.status_code == 200
    body = r.json()
    assert body["session_id"] == "newest-session"
    cs.get_operation_plan.assert_called_with("newest-session")


def test_current_no_sessions_returns_404():
    srv, gently_store, cs = _server(_SAMPLE_PLAN, sessions=())
    r = _client(srv).get("/api/operation_plan/current")
    assert r.status_code == 404


def test_context_store_unavailable_returns_unavailable():
    srv, _, _ = _server(_SAMPLE_PLAN, context_store_available=False)
    r = _client(srv).get("/api/operation_plan/sess-1")
    assert r.status_code == 200
    body = r.json()
    assert body["available"] is False
    assert body["plan"] is None


def test_context_store_exception_returns_unavailable():
    srv, _, cs = _server(_SAMPLE_PLAN)
    cs.get_operation_plan.side_effect = RuntimeError("disk error")
    r = _client(srv).get("/api/operation_plan/sess-1")
    assert r.status_code == 200
    body = r.json()
    assert body["available"] is False
