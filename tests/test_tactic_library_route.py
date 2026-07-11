from unittest.mock import MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from gently.ui.web.routes.tactic_library import create_router

_SAMPLE_TACTICS = [
    {"id": "tac-1", "name": "Monitor drift", "description": "Watch Z drift every 5 min"},
    {"id": "tac-2", "name": "Adjust laser", "description": "Drop 488 nm by 10%"},
]


def _server(tactics, context_store_available=True):
    context_store = MagicMock() if context_store_available else None
    if context_store is not None:
        context_store.list_tactics.return_value = tactics

    srv = MagicMock()
    srv.context_store = context_store
    return srv, context_store


def _client(server):
    app = FastAPI()
    app.include_router(create_router(server))
    return TestClient(app)


def test_returns_tactic_library():
    srv, cs = _server(_SAMPLE_TACTICS)
    r = _client(srv).get("/api/tactic_library")
    assert r.status_code == 200
    body = r.json()
    assert body["tactics"] == _SAMPLE_TACTICS
    cs.list_tactics.assert_called_once()


def test_returns_empty_when_no_tactics():
    srv, cs = _server([])
    r = _client(srv).get("/api/tactic_library")
    assert r.status_code == 200
    assert r.json() == {"tactics": []}


def test_returns_empty_when_context_store_absent():
    srv, _ = _server(_SAMPLE_TACTICS, context_store_available=False)
    r = _client(srv).get("/api/tactic_library")
    assert r.status_code == 200
    assert r.json() == {"tactics": []}


def test_returns_empty_when_list_tactics_raises():
    srv, cs = _server(_SAMPLE_TACTICS)
    cs.list_tactics.side_effect = RuntimeError("disk error")
    r = _client(srv).get("/api/tactic_library")
    assert r.status_code == 200
    assert r.json() == {"tactics": []}
