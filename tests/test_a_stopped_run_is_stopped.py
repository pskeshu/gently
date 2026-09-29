"""A run that was stopped says so, everywhere.

"after i stop a run also, i still see the tactic showing an active stop
button, and the "default" tactic in the Running tab ... has a green indicator
and active status"

Three things were wrong, and they showed as one:

- A run started from a saved tactic was never linked to it. Stop ended the
  run and left its tactic "active", with a Stop button that did nothing.
- Only the Stop button closed a tactic at all. A run that reached its
  ending, failed, or was stopped by the assistant left its tactic going.
- Stop did not write the checkpoint, so a run stopped on purpose was read
  as "interrupted" at the next start. And a paused run could not be stopped.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import yaml
from fastapi import FastAPI
from fastapi.testclient import TestClient

from gently.app.orchestration import tactic_executor
from gently.app.orchestration.timelapse import TimelapseOrchestrator, TimelapseStatus
from gently.core.file_store import FileStore
from gently.harness.state import ExperimentState
from gently.ui.web import auth
from gently.ui.web.routes.data import create_router

ROOT = Path(__file__).resolve().parents[1]
OPERATE = (ROOT / "gently" / "ui" / "web" / "static" / "js" / "operate.js").read_text("utf-8")
AGENT = (ROOT / "gently" / "app" / "agent.py").read_text("utf-8")

POSITIONS = {"embryo_1": {"x": -800.0, "y": -600.0}, "embryo_2": {"x": -200.0, "y": -600.0}}


def _rig():
    c = MagicMock()
    c.move_to_position = AsyncMock(return_value={"success": True})
    c.acquire_volume = AsyncMock(return_value={"success": True, "volume": None})
    return c


def _experiment():
    ex = ExperimentState()
    for eid, pos in POSITIONS.items():
        ex.add_embryo(eid, position=dict(pos), calibration={"galvo_center": 0.0})
    return ex


def _orchestrator(store):
    return TimelapseOrchestrator(_rig(), _experiment(), store=store, session_id="s1")


@pytest.fixture
def store(tmp_path):
    fs = FileStore(root=tmp_path / "data")
    fs.create_session("s1")
    return fs


def _checkpoint(store) -> dict:
    return yaml.safe_load((store._session_dir("s1") / "timelapse.yaml").read_text("utf-8"))


async def _run_a_little(orch):
    msg = await orch.start(base_interval_seconds=0.1, stop_condition="manual")
    assert msg.startswith("Started"), msg
    await asyncio.sleep(0.4)


# ── the run ──────────────────────────────────────────────────────────────


class TestTheRun:
    def test_a_run_that_is_going_has_not_ended(self, store):
        async def scenario():
            orch = _orchestrator(store)
            await _run_a_little(orch)
            ended = orch._ended
            await orch.stop("test")
            return ended

        assert asyncio.run(scenario()) is None

    def test_stop_says_it_was_stopped(self, store):
        async def scenario():
            orch = _orchestrator(store)
            await _run_a_little(orch)
            await orch.stop("operator")
            return orch

        orch = asyncio.run(scenario())
        assert orch._ended == "stopped"
        assert orch.get_status().status == TimelapseStatus.IDLE

    def test_stop_writes_the_checkpoint(self, store):
        async def scenario():
            orch = _orchestrator(store)
            await _run_a_little(orch)
            assert _checkpoint(store)["status"] == "running"
            await orch.stop("operator")

        asyncio.run(scenario())
        doc = _checkpoint(store)
        assert doc["status"] == "idle" and doc["ended"] == "stopped"

    def test_a_paused_run_can_be_stopped(self, store):
        async def scenario():
            orch = _orchestrator(store)
            await _run_a_little(orch)
            await orch.pause()
            assert orch.get_status().status == TimelapseStatus.PAUSED
            said = await orch.stop("operator")
            return orch, said

        orch, said = asyncio.run(scenario())
        assert said.startswith("Timelapse stopped"), said
        assert orch.get_status().status == TimelapseStatus.IDLE
        assert orch._ended == "stopped"

    def test_stopping_nothing_changes_nothing(self, store):
        orch = _orchestrator(store)
        assert asyncio.run(orch.stop("operator")) == "No timelapse running"
        assert orch._ended is None

    def test_a_stopped_run_is_read_as_stopped_at_the_next_start(self, store):
        async def first():
            orch = _orchestrator(store)
            await _run_a_little(orch)
            await orch.stop("operator")

        asyncio.run(first())
        again = _orchestrator(store)
        assert again.load_state().startswith("Restored")
        assert again._ended == "stopped"
        assert again.can_continue(), "it can still be carried on"

    def test_a_run_the_backend_left_is_read_as_interrupted(self, store):
        async def first():
            orch = _orchestrator(store)
            await _run_a_little(orch)
            # not stop(): the process went away with the checkpoint on disk
            orch._stop_requested = True
            orch._acquisition_task.cancel()
            try:
                await orch._acquisition_task
            except (asyncio.CancelledError, Exception):
                pass

        asyncio.run(first())
        again = _orchestrator(store)
        again.load_state()
        assert again._ended == "interrupted"

    def test_carrying_it_on_makes_it_a_run_again(self, store):
        async def scenario():
            orch = _orchestrator(store)
            await _run_a_little(orch)
            await orch.stop("operator")
            await orch.continue_run()
            ended = orch._ended
            await orch.stop("test")
            return ended

        assert asyncio.run(scenario()) is None

    def test_the_last_embryo_reaching_its_ending_is_completed(self, store):
        async def scenario():
            orch = _orchestrator(store)
            await orch.start(
                base_interval_seconds=0.05, stop_condition="timepoints", condition_value=2
            )
            for _ in range(80):
                if orch.get_status().status != TimelapseStatus.RUNNING:
                    break
                await asyncio.sleep(0.05)
            return orch

        orch = asyncio.run(scenario())
        assert orch.get_status().status == TimelapseStatus.COMPLETED
        assert orch._ended == "completed"
        assert _checkpoint(store)["ended"] == "completed"


# ── its tactic ───────────────────────────────────────────────────────────


def _plan(*tactics):
    return {"session_id": "s1", "title": "t", "tactics": [dict(t) for t in tactics]}


TL = {"id": "tl_1", "name": "default", "kind": "standing_timelapse", "state": "active"}
OLD = {"id": "tl_0", "name": "earlier", "kind": "standing_timelapse", "state": "done"}
WATCH = {"id": "rm_1", "name": "watch", "kind": "reactive_monitor", "state": "active"}


def _agent(file_context_store, *tactics, orch=None):
    file_context_store.set_operation_plan("s1", _plan(*tactics))
    return SimpleNamespace(
        context_store=file_context_store,
        session_id="s1",
        timelapse_orchestrator=orch,
        store=None,
    )


def _states(agent):
    plan = agent.context_store.get_operation_plan("s1")
    return {t["id"]: t["state"] for t in plan["tactics"]}


class TestItsTactic:
    def test_the_runs_end_closes_its_tactic(self, file_context_store):
        agent = _agent(file_context_store, TL)
        assert tactic_executor.close_timelapse_tactics(agent) == ["tl_1"]
        assert _states(agent) == {"tl_1": "done"}

    def test_a_paused_one_too(self, file_context_store):
        agent = _agent(file_context_store, {**TL, "state": "paused"})
        tactic_executor.close_timelapse_tactics(agent)
        assert _states(agent) == {"tl_1": "done"}

    def test_a_tactic_that_is_not_the_run_is_left_alone(self, file_context_store):
        agent = _agent(file_context_store, TL, WATCH, OLD)
        tactic_executor.close_timelapse_tactics(agent)
        assert _states(agent) == {"tl_1": "done", "rm_1": "active", "tl_0": "done"}

    def test_the_run_forgets_the_tactic_it_was(self, file_context_store):
        orch = SimpleNamespace(_operate_tactic_ids=["tl_1"])
        agent = _agent(file_context_store, TL, orch=orch)
        tactic_executor.close_timelapse_tactics(agent, orch)
        assert orch._operate_tactic_ids == []

    def test_no_plan_is_not_an_error(self, file_context_store):
        agent = SimpleNamespace(context_store=file_context_store, session_id="none", store=None)
        assert tactic_executor.close_timelapse_tactics(agent) == []

    def test_every_way_a_run_ends_is_heard(self):
        fn = AGENT[AGENT.index("            def on_run_ended(event):") :][:1300]
        assert "close_timelapse_tactics(" in fn
        for event in ("ACQUISITION_STOPPED", "ACQUISITION_COMPLETED", "ACQUISITION_FAILED"):
            assert f"EventType.{event}," in fn, event
        assert "self._event_bus.subscribe(ended, on_run_ended)" in fn

    def test_a_run_from_a_saved_tactic_knows_which_it_is(self, file_context_store):
        orch = MagicMock()
        orch.start = AsyncMock(return_value="Started timelapse")
        orch.enable_monitoring_mode = MagicMock(return_value="ok")
        agent = MagicMock()
        agent.session_id = "s1"
        agent.context_store = file_context_store
        agent.timelapse_orchestrator = orch
        agent.experiment = _experiment()
        agent.client.set_laser_config = AsyncMock(return_value={"success": True})
        file_context_store.set_operation_plan("s1", _plan({**TL, "state": "planned"}))
        tactic = {
            **TL,
            "scope": {"mode": "embryos", "embryo_ids": ["embryo_1"]},
            "structure": {"cadence_s": 300},
        }
        result = asyncio.run(tactic_executor.execute_tactic(agent, tactic))
        assert result["ok"], result
        assert orch._operate_tactic_ids == ["tl_1"]


# ── the routes ───────────────────────────────────────────────────────────


def _app(agent):
    server = MagicMock()
    server.agent_bridge.agent = agent
    app = FastAPI()
    app.include_router(create_router(server))
    app.dependency_overrides[auth.require_control] = lambda: True
    return TestClient(app)


def _live_agent(file_context_store, store, *tactics):
    orch = _orchestrator(store)
    agent = MagicMock()
    agent.session_id = "s1"
    agent.context_store = file_context_store
    agent.store = store
    agent.timelapse_orchestrator = orch
    agent.experiment = orch.experiment
    file_context_store.set_operation_plan("s1", _plan(*tactics))
    return agent, orch


class TestTheRoutes:
    def test_stop_closes_a_tactic_the_run_was_never_linked_to(self, file_context_store, store):
        # As it was on the rig: run from a saved tactic, nothing linked.
        agent, orch = _live_agent(file_context_store, store, TL)
        assert not getattr(orch, "_operate_tactic_ids", None)
        r = _app(agent).post("/api/devices/timelapse/stop", json={"reason": "operator"})
        assert r.status_code == 200, r.text
        assert _states(agent) == {"tl_1": "done"}

    def test_stop_on_a_run_already_stopped_still_clears_a_stale_tactic(
        self, file_context_store, store
    ):
        agent, _ = _live_agent(file_context_store, store, TL)
        client = _app(agent)
        client.post("/api/devices/timelapse/stop", json={})
        file_context_store.transition_tactic("s1", "tl_1", "active")  # stale again
        client.post("/api/devices/timelapse/stop", json={})
        assert _states(agent) == {"tl_1": "done"}

    def test_the_status_says_how_the_run_ended(self, file_context_store, store):
        agent, orch = _live_agent(file_context_store, store, TL)
        client = _app(agent)
        assert client.get("/api/devices/timelapse/status").json()["ended"] is None
        orch._ended = "stopped"
        assert client.get("/api/devices/timelapse/status").json()["ended"] == "stopped"

    def test_carrying_on_brings_back_one_tactic_not_every_one(self, file_context_store, store):
        done = {**TL, "state": "done"}
        agent, orch = _live_agent(file_context_store, store, OLD, done)
        store.save_acquisition_plan("s1", {"tactic_id": "tl_1", "interval_seconds": 300})

        async def restored():
            o = _orchestrator(store)
            await _run_a_little(o)
            await o.stop("operator")

        asyncio.run(restored())
        orch.load_state()
        r = _app(agent).post("/api/devices/timelapse/resume")
        assert r.status_code == 200, r.text
        try:
            assert _states(agent) == {"tl_0": "done", "tl_1": "active"}
            assert orch._operate_tactic_ids == ["tl_1"]
        finally:
            orch._stop_requested = True
            if orch._acquisition_task:
                orch._acquisition_task.cancel()

    def test_without_a_named_tactic_it_is_the_last_one(self, file_context_store, store):
        agent, orch = _live_agent(file_context_store, store, OLD, {**TL, "state": "done"})
        src = (ROOT / "gently" / "ui" / "web" / "routes" / "data.py").read_text("utf-8")
        fn = src[src.index("    def _reawaken_operate_tactics(") :][:1900]
        assert "ids = [named] if named in ended else ended[-1:]" in fn


# ── the page ─────────────────────────────────────────────────────────────


class TestThePage:
    def test_a_stopped_run_is_not_called_interrupted(self):
        fn = OPERATE[OPERATE.index("    async function renderRun() {") :][:3200]
        assert "st.ended === 'stopped'" in fn
        assert "'stopped — Resume run carries it on'" in fn
        assert "'interrupted — press Resume run to carry on'" in fn

    def test_a_run_that_ended_says_how(self):
        fn = OPERATE[OPERATE.index("    async function renderRun() {") :][:3200]
        assert "(st.ended && !running ? st.ended : st.status)" in fn

    def test_the_gate_does_not_call_a_stopped_run_interrupted(self, store):
        from gently.ui.web.routes import sessions as sessions_routes

        store.register_embryo("s1", "embryo_1", position_coarse={"x": 1.0, "y": 2.0}, role="test")
        (store._session_dir("s1") / "timelapse.yaml").write_text(
            yaml.safe_dump(
                {
                    "status": "idle",
                    "ended": "stopped",
                    "embryos": {"embryo_1": {"is_complete": False}},
                }
            ),
            encoding="utf-8",
        )
        server = MagicMock()
        server.agent_bridge.agent.store = store
        server.agent_bridge.agent.session_id = None
        app = FastAPI()
        app.include_router(sessions_routes.create_router(server))
        got = TestClient(app).get("/api/sessions/resumable").json()["sessions"]
        assert got[0]["session_id"] == "s1" and got[0]["run"] is None
