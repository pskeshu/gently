from pathlib import Path
from unittest.mock import MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

from gently.ui.web.routes.temperature import create_router


def _server(samples, sessions=(("sess-1", True),)):
    store = MagicMock()
    store.list_sessions.return_value = [{"session_id": sid} for sid, _ in sessions]
    store._session_dir.side_effect = lambda sid: (
        Path("/x") if any(sid == s for s, _ in sessions) else None
    )
    store.read_temperature_log.return_value = samples
    srv = MagicMock()
    srv.gently_store = store
    return srv, store


def _client(server):
    app = FastAPI()
    app.include_router(create_router(server))
    return TestClient(app)


def test_history_returns_samples():
    srv, store = _server(
        [
            {
                "t": "2026-06-27T10:00:00+00:00",
                "water_c": 28.0,
                "setpoint_c": 32.0,
                "state": "heating",
            }
        ]
    )
    r = _client(srv).get("/api/temperature/sess-1/history")
    assert r.status_code == 200
    body = r.json()
    assert body["session_id"] == "sess-1"
    assert body["samples"][0]["water_c"] == 28.0


def test_history_passes_since_through():
    srv, store = _server([])
    _client(srv).get("/api/temperature/sess-1/history?since=2026-06-27T10:00:01+00:00")
    store.read_temperature_log.assert_called_with("sess-1", since="2026-06-27T10:00:01+00:00")


def test_history_current_resolves_newest():
    srv, store = _server([], sessions=(("newest", True),))
    r = _client(srv).get("/api/temperature/current/history")
    assert r.status_code == 200
    assert r.json()["session_id"] == "newest"


def test_history_unknown_session_404():
    srv, store = _server([], sessions=(("sess-1", True),))
    # _session_dir returns None for unknown -> 404
    store._session_dir.side_effect = lambda sid: None
    r = _client(srv).get("/api/temperature/ghost/history")
    assert r.status_code == 404
