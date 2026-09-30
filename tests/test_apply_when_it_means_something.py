"""An Apply button that exists only when it means something.

The Acquisition pane's SPIM fields are a draft, applied at Start, and a
reload returns them to what the embryos hold. That is what the operator
asked for: a number typed into a field must not reach the microscope on its
own. But the agent can change an embryo mid-run, in effect at its next
acquisition, and the operator could not — "at Start" meant restarting.

So, in that one scenario — a run is going, and a field holds a value the
targeted embryos do not — a button appears beside the line that says what
they hold, and applies the draft through the same door as the agent's tool.
On a vanilla form, or with no run going, there is no button.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import gently.ui.web.auth as auth
from gently.harness.state import ExperimentState
from gently.ui.web.routes.data import create_router

WEB = Path(__file__).resolve().parents[1] / "gently" / "ui" / "web"
OPERATE = (WEB / "static" / "js" / "operate.js").read_text(encoding="utf-8")
INDEX = (WEB / "templates" / "index.html").read_text(encoding="utf-8")


def _experiment():
    ex = ExperimentState()
    for i in range(3):
        ex.add_embryo(f"embryo_{i + 1}", position={"x": 10.0 * i, "y": 0.0})
    ex.embryos["embryo_3"].should_skip = True
    ex.on_embryos_changed = MagicMock()
    return ex


def _app(experiment):
    server = MagicMock()
    agent = server.agent_bridge.agent
    agent.experiment = experiment
    agent.client = MagicMock()
    agent.client.set_laser_config = AsyncMock(return_value={"success": True})
    agent.lightsheet_monitor = None
    app = FastAPI()
    app.include_router(create_router(server))
    app.dependency_overrides[auth.require_control] = lambda: True
    return TestClient(app)


def _apply(app, **body):
    return app.post("/api/embryos/params", json=body)


class TestTheRoute:
    def test_it_writes_the_named_embryos_as_the_operators_and_announces(self):
        ex = _experiment()
        r = _apply(_app(ex), embryo_ids=["embryo_1", "embryo_2"], changes={"num_slices": 60})
        assert r.status_code == 200, r.text
        for eid in ("embryo_1", "embryo_2"):
            e = ex.embryos[eid]
            assert e.num_slices == 60
            assert e.param_provenance["num_slices"]["by"] == "operator"
        assert ex.embryos["embryo_3"].num_slices == 50
        assert ex.on_embryos_changed.call_count == 2
        assert r.json()["applied"]["embryo_1"] == {"num_slices": [50, 60]}

    def test_no_ids_means_every_active_embryo(self):
        ex = _experiment()
        assert _apply(_app(ex), embryo_ids=None, changes={"exposure_ms": 12}).status_code == 200
        assert ex.embryos["embryo_1"].exposure_ms == 12.0
        assert ex.embryos["embryo_2"].exposure_ms == 12.0
        assert ex.embryos["embryo_3"].exposure_ms == 10.0, "a skipped embryo was reconfigured"

    def test_only_the_keys_sent_are_written(self):
        ex = _experiment()
        ex.embryos["embryo_1"].exposure_ms = 7.0
        assert (
            _apply(_app(ex), embryo_ids=["embryo_1"], changes={"num_slices": 60}).status_code == 200
        )
        assert ex.embryos["embryo_1"].exposure_ms == 7.0

    def test_a_power_is_held_to_the_device_layers_limit(self):
        ex = _experiment()
        app = _app(ex)
        assert (
            _apply(app, embryo_ids=["embryo_1"], changes={"laser_power_488_pct": 4}).status_code
            == 200
        )
        assert ex.embryos["embryo_1"].laser_power_488_pct == 4.0
        r = _apply(app, embryo_ids=["embryo_1"], changes={"laser_power_488_pct": 50})
        assert r.status_code == 400 and "488" in r.json()["detail"]
        assert ex.embryos["embryo_1"].laser_power_488_pct == 4.0

    @pytest.mark.parametrize(
        "changes, says",
        [
            ({}, "changes"),
            ({"num_slices": 0}, "num_slices"),
            ({"num_slices": "many"}, "num_slices"),
            ({"exposure_ms": 0}, "exposure_ms"),
            ({"interval_seconds": 30}, "not parameters this route sets"),
            ({"role": "test"}, "not parameters this route sets"),
        ],
    )
    def test_what_it_refuses(self, changes, says):
        ex = _experiment()
        r = _apply(_app(ex), embryo_ids=["embryo_1"], changes=changes)
        assert r.status_code == 400 and says in r.json()["detail"]
        ex.on_embryos_changed.assert_not_called()

    def test_an_embryo_there_is_not(self):
        ex = _experiment()
        r = _apply(_app(ex), embryo_ids=["embryo_9"], changes={"num_slices": 60})
        assert r.status_code == 404
        ex.on_embryos_changed.assert_not_called()

    def test_a_value_already_held_changes_nothing_and_says_so(self):
        ex = _experiment()
        r = _apply(_app(ex), embryo_ids=["embryo_1"], changes={"num_slices": 50})
        assert r.status_code == 200 and r.json()["applied"] == {"embryo_1": {}}
        ex.on_embryos_changed.assert_not_called()


def _js(name: str) -> str:
    body = OPERATE[OPERATE.index(f"function {name}(") :]
    return body[: body.index("\n    }")]


class TestTheButton:
    def test_it_is_not_in_the_template(self):
        """Rendered, not written: a vanilla form has no such button."""
        assert "data-plan-apply" not in INDEX
        assert "op-plan-apply" not in INDEX

    def test_it_exists_only_while_a_run_is_going_with_a_draft_that_differs(self):
        body = _js("syncPlanFromEmbryos")
        assert "const applicable = _runBusy && Object.keys(pending).length > 0;" in body
        assert "if (applicable) {" in body
        assert "b.dataset.planApply = '1';" in body

    def test_a_blank_field_asks_for_nothing(self):
        body = _js("syncPlanFromEmbryos")
        assert "if (typed != null && Number.isFinite(typed)) pending[f.key] = typed;" in body

    def test_without_a_run_the_line_still_says_at_start(self):
        assert "— the field applies at Start." in _js("syncPlanFromEmbryos")

    def test_it_applies_through_the_same_door_as_the_agent(self):
        body = _js("applyPendingToEmbryos")
        assert "postJSON('/api/embryos/params', { embryo_ids: ids, changes: pending })" in body
        assert "_planDirty = false;" in body, "after applying, the form follows the embryos again"
