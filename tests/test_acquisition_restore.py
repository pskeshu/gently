"""A resumed session gets its acquisition configuration back.

"when i resume a session, i do not have all the acquisition configuration
restoration."

The orchestrator checkpointed to timelapse.yaml every round and nothing read
it back; the pane's plan lived in the form and nowhere else. Now the plan is
kept as acquisition.yaml when a run starts, the checkpoint is applied on
resume, and the pane fills itself from the session's plan.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
import yaml
from fastapi import FastAPI
from fastapi.testclient import TestClient

from gently.app.orchestration.resume import plan_from_session, restore_acquisition_state
from gently.core.file_store import FileStore
from gently.ui.web import auth
from gently.ui.web.routes.data import create_router

WEB = Path(__file__).resolve().parents[1] / "gently" / "ui" / "web"
OPERATE = (WEB / "static" / "js" / "operate.js").read_text(encoding="utf-8")
HTML = (WEB / "templates" / "index.html").read_text(encoding="utf-8")
AGENT = (Path(__file__).resolve().parents[1] / "gently" / "app" / "agent.py").read_text(
    encoding="utf-8"
)

PLAN = {
    "cadence_s": 300.0,
    "interval": 300.0,
    "stop_condition": "stages(hatched,hatching)",
    "condition_value": None,
    "monitoring_mode": "idle",
    "num_slices": 40,
    "exposure_ms": 12.0,
    "laser_config": "488 only",
    "dic": {"enabled": True, "every_seconds": None, "position": None, "exposure_ms": None},
    "stop_conditions": {"embryo_2": "timepoints:12"},
    "embryo_ids": ["embryo_1", "embryo_2"],
}


@pytest.fixture
def store(tmp_path):
    fs = FileStore(root=tmp_path)
    fs.create_session("s1")
    return fs


# ── the plan lives beside the checkpoint ─────────────────────────────────


def test_the_plan_round_trips_through_the_session(store):
    path = store.save_acquisition_plan("s1", PLAN)
    assert path.name == "acquisition.yaml"
    assert path.parent == store._session_dir("s1")
    assert store.get_acquisition_plan("s1") == PLAN


def test_a_session_without_a_plan_says_so(store):
    assert store.get_acquisition_plan("s1") is None
    assert plan_from_session(store, "s1") == (None, None)


def test_a_saved_plan_is_preferred(store):
    store.save_acquisition_plan("s1", PLAN)
    plan, source = plan_from_session(store, "s1")
    assert source == "saved" and plan == PLAN


def test_a_session_that_predates_the_plan_file_reads_its_checkpoint(store):
    """The sessions already on the rig: timelapse.yaml + the last sidecar."""
    sd = store._session_dir("s1")
    checkpoint = {
        "status": "running",
        "base_interval_seconds": 300.0,
        "active_monitoring_modes": [],
        "dic": {"enabled": True, "every_seconds": 600.0, "position": None, "exposure_ms": 20.0},
        "embryos": {
            "embryo_1": {"stop_condition": {"spec": "stages(hatched,hatching)"}},
            "embryo_2": {"stop_condition": {"spec": "timepoints:12"}},
            "embryo_3": {"stop_condition": {"spec": "stages(hatched,hatching)"}},
        },
    }
    (sd / "timelapse.yaml").write_text(yaml.safe_dump(checkpoint), encoding="utf-8")
    store.get_acquisition_params = MagicMock(return_value={"num_slices": 50, "exposure_ms": 10.0})

    plan, source = plan_from_session(store, "s1")
    assert source == "run"
    assert plan["cadence_s"] == 300.0
    assert plan["stop_condition"] == "stages(hatched,hatching)", "the ending most share"
    assert plan["stop_conditions"] == {"embryo_2": "timepoints:12"}, "the one that differs"
    assert plan["dic"]["every_seconds"] == 600.0
    assert plan["num_slices"] == 50 and plan["exposure_ms"] == 10.0
    assert plan["embryo_ids"] == ["embryo_1", "embryo_2", "embryo_3"]


# ── the checkpoint is applied on resume ──────────────────────────────────


def _agent(status="idle", session="s1", load="Restored timelapse state: 2 embryos"):
    agent = MagicMock()
    agent.session_id = session
    orch = MagicMock()
    orch._status = MagicMock(value=status)
    orch._session_id = "old"
    orch.load_state = MagicMock(return_value=load)
    agent.timelapse_orchestrator = orch
    return agent


def test_resume_points_the_orchestrator_at_the_session_and_loads_its_checkpoint():
    agent = _agent()
    out = restore_acquisition_state(agent)
    assert out["restored"] is True
    orch = agent.timelapse_orchestrator
    assert orch._session_id == "s1", "checkpoints would file under the previous session"
    assert orch._trace_dir is None
    orch.load_state.assert_called_once_with()


def test_a_live_run_is_left_alone():
    agent = _agent(status="running")
    out = restore_acquisition_state(agent)
    assert out["restored"] is False
    agent.timelapse_orchestrator.load_state.assert_not_called()


def test_a_missing_checkpoint_is_not_a_failure():
    agent = _agent(load="No timelapse.yaml at …")
    out = restore_acquisition_state(agent)
    assert out["restored"] is False and "No timelapse" in out["message"]


def test_the_agent_restores_on_both_resume_paths():
    boot = AGENT[AGENT.index("self._init_timelapse_orchestrator()") :][:400]
    assert "if session_id:\n            self._restore_acquisition_state()" in boot
    switch = AGENT[AGENT.index("def switch_session(self, session_id: str | None)") :]
    assert "self._restore_acquisition_state()" in switch[: switch.index("return {")]
    assert "return bool(self.switch_session(session_id)" in AGENT  # resume goes through it


# ── the start route keeps the plan; a GET serves it back ─────────────────


def _app(store=None, orchestrator=None, session="s1"):
    server = MagicMock()
    agent = server.agent_bridge.agent
    agent.session_id = session
    agent.store = store
    agent.timelapse_orchestrator = orchestrator
    agent.client.set_laser_config = AsyncMock(return_value={"success": True})
    app = FastAPI()
    app.include_router(create_router(server))
    app.dependency_overrides[auth.require_control] = lambda: True
    return TestClient(app)


def test_the_plan_route_serves_the_saved_plan(store):
    store.save_acquisition_plan("s1", PLAN)
    r = _app(store=store).get("/api/devices/timelapse/plan")
    assert r.status_code == 200
    body = r.json()
    assert body["source"] == "saved" and body["plan"] == PLAN and body["session_id"] == "s1"


def test_the_plan_route_never_errors_without_a_store():
    r = _app(store=None).get("/api/devices/timelapse/plan")
    assert r.status_code == 200 and r.json() == {"plan": None, "source": None, "session_id": "s1"}


def test_starting_a_run_keeps_its_plan(store):
    orch = MagicMock()
    orch.start = AsyncMock(return_value="Timelapse started.")
    orch.enable_monitoring_mode = MagicMock(return_value="ok")
    client = _app(store=store, orchestrator=orch)
    payload = {
        "interval_seconds": 300,
        "stop_condition": "manual",
        "embryo_ids": None,
        "num_slices": 40,
        "exposure_ms": 12,
        "laser_config": "488 only",
        "dic": {"enabled": True},
        "stop_conditions": {"embryo_2": "timepoints:12"},
        "monitoring_mode": "expression_monitoring",
    }
    r = client.post("/api/devices/timelapse/start", json=payload)
    assert r.status_code == 200, r.text
    kept = store.get_acquisition_plan("s1")
    assert kept is not None, "acquisition.yaml was not written"
    assert kept["cadence_s"] == 300.0
    assert kept["num_slices"] == 40 and kept["exposure_ms"] == 12.0
    assert kept["laser_config"] == "488 only"
    assert kept["dic"]["enabled"] is True
    assert kept["stop_conditions"] == {"embryo_2": "timepoints:12"}
    assert kept["monitoring_mode"] == "expression_monitoring"


def test_a_start_that_was_already_running_keeps_nothing(store):
    orch = MagicMock()
    orch.start = AsyncMock(return_value="Timelapse already running. Use stop() first.")
    client = _app(store=store, orchestrator=orch)
    r = client.post("/api/devices/timelapse/start", json={"interval_seconds": 60})
    assert r.status_code == 200
    assert store.get_acquisition_plan("s1") is None


# ── the pane fills itself ────────────────────────────────────────────────


def test_the_pane_restores_the_plan_on_entry():
    enter = OPERATE[OPERATE.index("        acquire: {") :][:700]
    assert "onEnter() { restorePlan();" in enter


def test_the_form_is_filled_from_the_plan_route_once_and_not_over_the_operators_typing():
    fn = OPERATE[OPERATE.index("async function restorePlan()") :][:1200]
    assert "getJSON('/api/devices/timelapse/plan')" in fn
    assert "if (_planRestored) return;" in fn
    assert "_planDirty) return;" in fn
    assert "fillPlan(AcquisitionPlan.fromStructure(d.plan))" in fn
    assert "await loadLaserPresets();" in fn, "the laser select is empty until presets load"


def test_typing_in_the_pane_marks_the_plan_dirty_inside_wire():
    wire = OPERATE[OPERATE.index("if (_wired) return;") :]
    assert "addEventListener('input', () => { _planDirty = true; renderPlan(); });" in wire
    change = wire[wire.index("planPanel.addEventListener('change'") :][:120]
    assert "_planDirty = true;" in change


def test_fill_is_the_inverse_of_read():
    fn = OPERATE[OPERATE.index("function fillPlan(plan)") :][:2600]
    for field in (
        "op-tl-interval",
        "op-plan-unit",
        "op-plan-slices",
        "op-plan-exposure",
        "op-plan-laser",
        "op-plan-dic",
        "op-plan-dic-every",
        "op-plan-dic-pos",
        "op-plan-dic-exposure",
        "op-tl-stop",
        "op-tl-condval",
        "op-tl-monitor",
    ):
        assert f"'{field}'" in fn, f"fillPlan never sets {field}"
    assert "renderOverrideRows();" in fn and "plan.overrides.forEach" in fn
    assert "_dicPin = plan.dic.position === 'here'" in fn


def test_the_pane_says_where_the_plan_came_from():
    assert 'id="op-plan-from"' in HTML
