"""A run's tactic says when it started and how it ended.

Ryan's first night on the rig: Start at 16:45, Stop twenty seconds later to
change the laser preset, Start again at 16:46, and eight hours of volumes.
Two Starts, two tactics, both honest — and the Operations tab showed them as
"Adaptive timelapse · done" twice over, with nothing to tell a false start
from the run. "I see the adaptive timelapse thing duplicated."

The store was right; the card said too little. So the tactic is told when
it started, as it starts, and how it ended, as it ends: "stopped by operator
after 20 s · 4 volumes" beside "completed after 8 h 15 min · 337 volumes".
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from gently.app.orchestration import tactic_executor
from gently.app.orchestration.tactic_executor import run_facts, span, when
from gently.harness.state import ExperimentState
from gently.ui.web import auth
from gently.ui.web.routes.data import create_router

ROOT = Path(__file__).resolve().parents[1]
OPERATE = (ROOT / "gently" / "ui" / "web" / "static" / "js" / "operate.js").read_text("utf-8")
OVERVIEW = (ROOT / "gently" / "ui" / "web" / "static" / "js" / "experiment-overview.js").read_text(
    "utf-8"
)
AGENT = (ROOT / "gently" / "app" / "agent.py").read_text("utf-8")


class TestTheWords:
    def test_a_moment(self):
        assert when(datetime(2026, 9, 30, 16, 45, 3)) == "30 Sep 16:45"
        assert when(datetime(2026, 10, 1, 1, 1, 56)) == "1 Oct 01:01"

    @pytest.mark.parametrize(
        "seconds, said",
        [
            (19.6, "20 s"),
            (59, "59 s"),
            (60, "1 min"),
            (754, "12 min"),
            (3600, "1 h"),
            (29730, "8 h 15 min"),
        ],
    )
    def test_a_duration(self, seconds, said):
        assert span(seconds) == said


def _orch(**kw):
    base = {
        "_started_at": datetime.now() - timedelta(seconds=20),
        "_ended": "stopped",
        "_total_timepoints": 4,
        "_volumes": True,
        "_dic_frames": 0,
        "_error_message": None,
        "_operate_tactic_ids": [],
    }
    base.update(kw)
    return SimpleNamespace(**base)


class TestTheFacts:
    def test_a_false_start(self):
        facts = run_facts(_orch(), reason="operator")
        assert facts["ended"] == "stopped by operator after 20 s"
        assert facts["acquired"] == "4 volumes"
        assert facts["started"] == when(datetime.now() - timedelta(seconds=20))

    def test_the_night(self):
        orch = _orch(
            _started_at=datetime.now() - timedelta(hours=8, minutes=15),
            _ended="completed",
            _total_timepoints=337,
        )
        facts = run_facts(orch)
        assert facts["ended"] == "completed after 8 h 15 min"
        assert facts["acquired"] == "337 volumes"

    def test_a_failure_says_why(self):
        orch = _orch(_ended="failed", _error_message="3 brightfield frames in a row failed")
        assert run_facts(orch)["ended"].startswith(
            "failed: 3 brightfield frames in a row failed after"
        )

    def test_a_brightfield_run_counts_frames(self):
        orch = _orch(_volumes=False, _dic_frames=12, _ended="completed")
        assert run_facts(orch)["acquired"] == "12 brightfield frames"

    def test_a_stop_with_no_reason_is_still_a_stop(self):
        assert run_facts(_orch(_ended="stopped"))["ended"] == "stopped after 20 s"

    def test_an_orchestrator_that_knows_nothing_binds_nothing(self):
        assert run_facts(SimpleNamespace()) == {}
        assert (
            run_facts(
                MagicMock(_started_at=None, _ended=None, _volumes=True, _total_timepoints=None)
            )
            == {}
        )


def _plan(*tactics):
    return {"session_id": "s1", "title": "t", "tactics": [dict(t) for t in tactics]}


TL = {"id": "tl_1", "name": "Adaptive timelapse", "kind": "standing_timelapse", "state": "active"}


class TestOnTheCard:
    def test_closing_binds_the_facts(self, file_context_store):
        file_context_store.set_operation_plan("s1", _plan(TL))
        agent = SimpleNamespace(context_store=file_context_store, session_id="s1", store=None)
        orch = _orch()
        assert tactic_executor.close_timelapse_tactics(agent, orch, reason="operator") == ["tl_1"]
        (t,) = file_context_store.get_operation_plan("s1")["tactics"]
        assert t["state"] == "done"
        assert t["live"]["ended"] == "stopped by operator after 20 s"
        assert t["live"]["acquired"] == "4 volumes"
        assert t["live"]["started"]

    def test_closing_without_an_orchestrator_still_closes(self, file_context_store):
        file_context_store.set_operation_plan("s1", _plan(TL))
        agent = SimpleNamespace(context_store=file_context_store, session_id="s1", store=None)
        assert tactic_executor.close_timelapse_tactics(agent) == ["tl_1"]
        (t,) = file_context_store.get_operation_plan("s1")["tactics"]
        assert t["state"] == "done" and not t.get("live")

    def test_the_stops_reason_reaches_the_card(self):
        fn = AGENT[AGENT.index("            def on_run_ended(event):") :][:1500]
        assert 'reason=data.get("reason")' in fn

    def test_two_starts_are_two_cards_that_read_differently(self, file_context_store):
        """The night, replayed on the store: the first tactic's card and the
        second's no longer say the same thing."""
        first = {**TL, "id": "op_1", "live": {"started": "30 Sep 16:45"}}
        file_context_store.set_operation_plan("s1", _plan(first))
        agent = SimpleNamespace(context_store=file_context_store, session_id="s1", store=None)
        tactic_executor.close_timelapse_tactics(agent, _orch(), reason="operator")
        plan = file_context_store.get_operation_plan("s1")
        plan["tactics"].append({**TL, "id": "op_2", "live": {"started": "30 Sep 16:46"}})
        file_context_store.set_operation_plan("s1", plan)
        tactic_executor.close_timelapse_tactics(
            agent,
            _orch(
                _started_at=datetime.now() - timedelta(hours=8, minutes=15),
                _ended="completed",
                _total_timepoints=337,
            ),
        )
        a, b = file_context_store.get_operation_plan("s1")["tactics"]
        assert a["live"] != b["live"]
        assert a["live"]["ended"].startswith("stopped by operator after 20 s")
        assert b["live"]["ended"] == "completed after 8 h 15 min"


def _experiment():
    ex = ExperimentState()
    ex.add_embryo(
        "embryo_1",
        position={"x": 0.0, "y": 0.0},
        calibration={"galvo_center": 0.0, "slope_um_per_deg": 50.0},
    )
    return ex


class TestFromTheStart:
    def test_a_seeded_tactic_says_when_it_started(self, file_context_store):
        orch = MagicMock()
        orch.start = AsyncMock(return_value="Started timelapse for 1 embryos")
        orch.enable_monitoring_mode = MagicMock(return_value="ok")
        agent = MagicMock()
        agent.session_id = "s1"
        agent.context_store = file_context_store
        agent.timelapse_orchestrator = orch
        agent.experiment = _experiment()
        agent.client.set_laser_config = AsyncMock(return_value={"success": True})
        agent.store = None
        server = MagicMock()
        server.agent_bridge.agent = agent
        app = FastAPI()
        app.include_router(create_router(server))
        app.dependency_overrides[auth.require_control] = lambda: True
        r = TestClient(app).post("/api/devices/timelapse/start", json={"interval_seconds": 300})
        assert r.status_code == 200, r.text
        (t,) = file_context_store.get_operation_plan("s1")["tactics"]
        assert t["live"]["started"] == when(datetime.now())

    async def test_a_saved_tactic_too(self, file_context_store):
        tactic = {
            **TL,
            "id": "lib_1",
            "state": "planned",
            "scope": {"mode": "global"},
            "structure": {"cadence_s": 300, "stop_condition": "manual"},
        }
        file_context_store.set_operation_plan("s1", _plan(tactic))
        agent = MagicMock()
        agent.session_id = "s1"
        agent.context_store = file_context_store
        agent.experiment = _experiment()
        agent.client.set_laser_config = AsyncMock(return_value={"success": True})
        agent.timelapse_orchestrator.start = AsyncMock(return_value="Started timelapse")
        agent.store = None
        out = await tactic_executor.execute_tactic(agent, tactic)
        assert out["ok"], out
        (t,) = file_context_store.get_operation_plan("s1")["tactics"]
        assert t["state"] == "active"
        assert t["live"]["started"] == when(datetime.now())


class TestTheScreen:
    def test_the_run_pane_card_says_it(self):
        body = OPERATE[OPERATE.index("function tacticCard(") :]
        body = body[: body.index("\n    }")]
        for key in ("live.started", "live.ended", "live.acquired", "struct.laser_config"):
            assert key in body, key
        assert "op-tcard-when" in body

    def test_the_operations_tab_shows_a_done_tactics_live_facts(self):
        """It already renders every flat live.* key of a done tactic; the
        facts ride on that, so nothing there had to change."""
        body = OVERVIEW[OVERVIEW.index("_renderOpsTactic(") :]
        assert "t.state === 'active' || t.state === 'done'" in body
        assert "ops-livefacts" in body
