"""Two intents, said apart: looking at a session, and making one live.

"we restore a session - but not necessarily to continue a run - and often
to view the data - but at times to restore a acquisition run... so does the
system make any assumptions that make the things awkward?"

Looking is the Sessions tab, which reads the folder. Making a session live
is one path in the agent, ``switch_session``, which also opens a fresh
session from inside the app — "instead of the current method that is
closing gently and opening again."
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from gently.app.agent import MicroscopyAgent
from gently.core.file_store import FileStore
from gently.harness.session.manager import SessionManager
from gently.harness.state import ExperimentState
from gently.ui.web import auth
from gently.ui.web.routes import sessions as sessions_routes

WEB = Path(__file__).resolve().parents[1] / "gently" / "ui" / "web"
REVIEW_JS = (WEB / "static" / "js" / "review.js").read_text(encoding="utf-8")


# ── the agent's one switching path ──────────────────────────────────────────


def _agent(tmp_path, running=False):
    """A stand-in with the parts switch_session touches: a real store and
    session manager and experiment, mocks for the per-session plumbing."""
    store = FileStore(root=tmp_path)
    fake = SimpleNamespace()
    fake.store = store
    fake.sessions = SessionManager(store=store, storage_path=tmp_path)
    fake.experiment = ExperimentState()
    fake.conversation = SimpleNamespace(conversation_history=[], interaction_logger=None)
    fake.timelapse_orchestrator = (
        SimpleNamespace(_status=SimpleNamespace(value="running")) if running else None
    )
    fake.timeline_manager = MagicMock()
    fake.interaction_logger = None
    for name in (
        "_emit_event",
        "_update_system_prompt",
        "stop_event_capture",
        "stop_decision_log",
        "_init_interaction_logger",
        "_init_event_capture",
        "_init_decision_log",
        "_init_timeline_manager",
        "_restore_acquisition_state",
    ):
        setattr(fake, name, MagicMock())
    fake.switch_session = lambda sid: MicroscopyAgent.switch_session(fake, sid)
    return fake, store


def _saved_session(store, sid, *embryo_ids):
    store.create_session(sid)
    for eid in embryo_ids:
        store.register_embryo(sid, eid, position_x=1.0, position_y=2.0, role="test")
    return sid


class TestMakingASessionLive:
    def test_the_live_embryo_list_is_replaced_not_merged(self, tmp_path):
        fake, store = _agent(tmp_path)
        fake.sessions.create_session()
        fake.experiment.add_embryo("embryo_1", position={"x": 9.0, "y": 9.0})
        _saved_session(store, "old00001", "embryo_7", "embryo_8")

        got = fake.switch_session("old00001")

        assert got["ok"] and got["session_id"] == "old00001"
        assert sorted(fake.experiment.embryos) == ["embryo_7", "embryo_8"]  # not embryo_1 too

    def test_the_previous_session_is_saved_first(self, tmp_path):
        fake, store = _agent(tmp_path)
        fake.sessions.create_session()
        first = fake.sessions.session_id
        fake.experiment.add_embryo("embryo_1", position={"x": 9.0, "y": 9.0})
        fake.conversation.conversation_history = [{"role": "user", "content": "hello"}]
        _saved_session(store, "old00001")

        fake.switch_session("old00001")

        snap = store.load_session_snapshot(first)
        assert snap["conversation_history"] == [{"role": "user", "content": "hello"}]
        assert "embryo_1" in snap["experiment_data"]["embryos"]

    def test_restored_positions_are_marked_as_from_another_sample(self, tmp_path):
        fake, store = _agent(tmp_path)
        fake.sessions.create_session()
        _saved_session(store, "old00001", "embryo_7")
        fake.switch_session("old00001")
        marked = fake.experiment.metadata["restored_from"]
        assert marked["session_id"] == "old00001" and marked["embryos"] == ["embryo_7"]

    def test_the_per_session_logs_follow_the_session(self, tmp_path):
        fake, store = _agent(tmp_path)
        fake.sessions.create_session()
        _saved_session(store, "old00001")
        fake.switch_session("old00001")
        fake.stop_event_capture.assert_called_once()
        fake.stop_decision_log.assert_called_once()
        fake.timeline_manager.stop.assert_called_once()
        for name in (
            "_init_interaction_logger",
            "_init_event_capture",
            "_init_decision_log",
            "_init_timeline_manager",
            "_restore_acquisition_state",
        ):
            getattr(fake, name).assert_called_once()
        fake._update_system_prompt.assert_called_once()

    def test_a_live_run_is_never_switched_out_from_under(self, tmp_path):
        fake, store = _agent(tmp_path, running=True)
        fake.sessions.create_session()
        _saved_session(store, "old00001")
        with pytest.raises(RuntimeError, match="a run is running"):
            fake.switch_session("old00001")
        fake.stop_event_capture.assert_not_called()


class TestANewSessionFromInside:
    def test_a_fresh_session_starts_empty_and_marked_as_nothing(self, tmp_path):
        fake, store = _agent(tmp_path)
        fake.sessions.create_session()
        first = fake.sessions.session_id
        fake.experiment.add_embryo("embryo_1", position={"x": 9.0, "y": 9.0})
        fake.experiment.metadata["restored_from"] = {"session_id": "x"}
        fake.conversation.conversation_history = [{"role": "user", "content": "hello"}]

        got = fake.switch_session(None)

        assert got["ok"] and got["previous"] == first and got["session_id"] != first
        assert store.get_session(got["session_id"]) is not None
        assert fake.experiment.embryos == {} and "restored_from" not in fake.experiment.metadata
        assert fake.conversation.conversation_history == []
        assert store.load_session_snapshot(first)["experiment_data"]["embryos"]  # saved first


# ── the routes ──────────────────────────────────────────────────────────────


def _client(store, switch=None, active="live0001"):
    server = MagicMock()
    agent = server.agent_bridge.agent
    agent.store = store
    agent.session_id = active
    agent.switch_session = switch or MagicMock(
        side_effect=lambda sid: {"ok": True, "session_id": sid or "new00001", "previous": active}
    )
    server.manager.broadcast = AsyncMock()
    server.rehydrate_session = MagicMock(return_value=0)
    app = FastAPI()
    app.include_router(sessions_routes.create_router(server))
    app.dependency_overrides[auth.require_control] = lambda: True
    return TestClient(app), server


class TestTheRoutes:
    def test_resume_goes_through_the_switch_and_reloads_every_browser(self, tmp_path):
        store = FileStore(root=tmp_path)
        store.create_session("old00001")
        c, server = _client(store)
        r = c.post("/api/sessions/old00001/resume")
        assert r.status_code == 200, r.text
        server.agent_bridge.agent.switch_session.assert_called_once_with("old00001")
        server.manager.broadcast.assert_awaited_once_with(
            {"type": "session_changed", "session_id": "old00001"}
        )

    def test_a_live_run_answers_409(self, tmp_path):
        store = FileStore(root=tmp_path)
        store.create_session("old00001")
        c, _ = _client(store, switch=MagicMock(side_effect=RuntimeError("a run is running")))
        r = c.post("/api/sessions/old00001/resume")
        assert r.status_code == 409 and "a run is running" in r.json()["detail"]
        assert c.post("/api/sessions/new").status_code == 409

    def test_new_opens_a_fresh_session_and_reloads_every_browser(self, tmp_path):
        store = FileStore(root=tmp_path)
        c, server = _client(store)
        r = c.post("/api/sessions/new")
        assert r.status_code == 200, r.text
        assert r.json() == {"ok": True, "session_id": "new00001", "previous": "live0001"}
        server.agent_bridge.agent.switch_session.assert_called_once_with(None)
        server.rehydrate_session.assert_called_once_with("new00001")
        server.manager.broadcast.assert_awaited_once_with(
            {"type": "session_changed", "session_id": "new00001"}
        )
        assert server.gate_passed is True


class TestWhatTheButtonsSay:
    def test_the_button_says_make_live_and_what_that_means(self):
        assert "Resume in agent" not in REVIEW_JS
        assert ">Make live<" in REVIEW_JS
        assert "new images, the chat and the stage targets belong to it" in REVIEW_JS
        assert "re-centre before trusting them" in REVIEW_JS
        assert "newSession()" in REVIEW_JS and "/api/sessions/new" in REVIEW_JS
        panel = (WEB / "templates" / "_sessions_panel.html").read_text(encoding="utf-8")
        header = (WEB / "templates" / "_header.html").read_text(encoding="utf-8")
        assert "ReviewApp.newSession()" in panel and "ReviewApp.newSession()" in header
