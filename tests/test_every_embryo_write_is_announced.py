"""One truth for an embryo's acquisition parameters, and everyone hears of it.

The operator and the agent both change what an embryo will be imaged with —
the Acquisition pane at Start, the agent through `modify_parameters` — and
each has to be able to see what the other did. The truth is the EmbryoState;
what puts a change in front of the other party is the EMBRYOS_UPDATE
broadcast that follows it.

The GUI's writes announced themselves, because the GUI needs the redraw.
The agent's did not, because the agent does not: `modify_parameters` set an
embryo to 30 slices and the pane went on showing 50. Not one bug: an audit
found fifteen silent writers, nearly all on the agent's side.

So there is one door, `ExperimentState.set_params`, that writes, records who
and why, and announces; and a guard here that any function writing an
acquisition field announces the change, whoever it belongs to.
"""

from __future__ import annotations

import re
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from gently.harness.state import EmbryoState, ExperimentState

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "gently" / "ui" / "web"

FIELDS = (
    "num_slices",
    "exposure_ms",
    "interval_seconds",
    "priority",
    "acquisition_mode",
    "laser_power_488_pct",
    "laser_power_561_pct",
    "laser_power_405_pct",
    "laser_power_637_pct",
    "should_skip",
    "role",
    "nickname",
    "stop_condition",
    "calibration",
)

# Functions that write these fields and legitimately do not announce: they
# run before there is anyone to tell (a session being loaded), or they ARE
# the announcing (the state module itself).
ALLOWED_SILENT = {
    "gently/harness/state.py",  # the dataclass and set_params live here
    "gently/harness/session/manager.py::_resume_session",  # a session being loaded
    "gently/app/orchestration/timelapse.py::_apply_runtime_state",  # a checkpoint being read
    "gently/app/agent.py::import_embryos_from_session",  # announces through add_embryo
}

ANNOUNCES = ("notify_embryos_changed", "set_params(", "_publish_embryos_update", "EMBRYOS_UPDATE")
WRITE = re.compile(r"\b\w+\.(" + "|".join(FIELDS) + r")\s*=(?!=)")
FUNC = re.compile(r"(?ms)^(\s*)(?:async\s+)?def\s+(\w+)\(.*?(?=^\1(?:async\s+)?def\s|\Z)")


def _silent_writers() -> list[str]:
    found = []
    for f in sorted((ROOT / "gently").rglob("*.py")):
        rel = f.relative_to(ROOT).as_posix()
        if rel in ALLOWED_SILENT:
            continue
        src = f.read_text(encoding="utf-8", errors="replace")
        for m in FUNC.finditer(src):
            body, name = m.group(0), m.group(2)
            hits = sorted(
                {h.group(1) for h in WRITE.finditer(body) if not h.group(0).startswith("self.")}
            )
            if not hits or f"{rel}::{name}" in ALLOWED_SILENT:
                continue
            if not any(a in body for a in ANNOUNCES):
                found.append(f"{rel}::{name} writes {hits}")
    return found


def test_every_writer_announces():
    silent = _silent_writers()
    assert not silent, (
        "these change an embryo and tell nobody — the pane goes on showing the old "
        "value. Write through ExperimentState.set_params, or call "
        "experiment.notify_embryos_changed() when done:\n  " + "\n  ".join(silent)
    )


def test_the_guard_can_see():
    """A guard that finds nothing might be looking at nothing."""
    src = (ROOT / "gently" / "app" / "tools" / "experiment_tools.py").read_text(encoding="utf-8")
    assert "set_params(" in src, "modify_parameters no longer goes through the one door"
    assert any(
        WRITE.search(f.read_text(encoding="utf-8", errors="replace"))
        for f in (ROOT / "gently" / "app").rglob("*.py")
    )


# ---------------------------------------------------------------------------
# The one door
# ---------------------------------------------------------------------------


def _experiment():
    ex = ExperimentState()
    ex.add_embryo("embryo_1", position={"x": 0.0, "y": 0.0})
    ex.add_embryo("embryo_2", position={"x": 10.0, "y": 0.0})
    ex.on_embryos_changed = MagicMock()
    return ex


class TestSetParams:
    def test_it_writes_records_and_announces_once(self):
        ex = _experiment()
        applied = ex.set_params(
            "embryo_2", {"num_slices": 30, "exposure_ms": 12.0}, by="agent", reason="weak signal"
        )
        e = ex.embryos["embryo_2"]
        assert (e.num_slices, e.exposure_ms) == (30, 12.0)
        assert applied == {"num_slices": (50, 30), "exposure_ms": (10.0, 12.0)}
        assert e.param_provenance["num_slices"]["by"] == "agent"
        assert e.param_provenance["num_slices"]["reason"] == "weak signal"
        assert e.param_provenance["num_slices"]["at"]
        ex.on_embryos_changed.assert_called_once()
        assert ex.embryos["embryo_1"].num_slices == 50, "the other embryo was touched"

    def test_a_value_already_held_is_not_a_change(self):
        ex = _experiment()
        assert ex.set_params("embryo_1", {"num_slices": 50}, by="operator") == {}
        ex.on_embryos_changed.assert_not_called()
        assert "num_slices" not in ex.embryos["embryo_1"].param_provenance

    def test_a_field_that_is_not_a_parameter_is_refused_not_ignored(self):
        ex = _experiment()
        with pytest.raises(ValueError, match="not acquisition parameters"):
            ex.set_params("embryo_1", {"slices": 30}, by="agent")
        assert ex.embryos["embryo_1"].num_slices == 50
        ex.on_embryos_changed.assert_not_called()

    def test_an_unknown_embryo_raises(self):
        with pytest.raises(KeyError):
            _experiment().set_params("embryo_9", {"num_slices": 30}, by="agent")

    def test_the_provenance_travels_with_the_embryo(self):
        ex = _experiment()
        ex.set_params("embryo_1", {"laser_power_488_pct": 3.0}, by="operator", reason="Start")
        d = ex.embryos["embryo_1"].to_dict()
        assert d["param_provenance"]["laser_power_488_pct"]["by"] == "operator"
        assert EmbryoState(id="x").param_provenance == {}


# ---------------------------------------------------------------------------
# The agent's change reaches the operator's screen
# ---------------------------------------------------------------------------


def _tool(name):
    import gently.app.tools.experiment_tools  # noqa: F401
    from gently.harness.tools.registry import get_tool_registry

    return get_tool_registry()._tools[name].handler


def _agent():
    from gently.core.event_bus import EventBus, EventType

    agent = MagicMock()
    agent.experiment = ExperimentState()
    agent.experiment.add_embryo("embryo_2", position={"x": 0.0, "y": 0.0})
    bus = EventBus()
    heard = []
    bus.subscribe(EventType.EMBRYOS_UPDATE, lambda ev: heard.append(ev))
    agent._event_bus = bus

    def publish():
        bus.publish(
            EventType.EMBRYOS_UPDATE,
            {"embryos": [e.to_dict() for e in agent.experiment.embryos.values()]},
            source="test",
        )

    agent.experiment.on_embryos_changed = publish
    return agent, heard


class TestTheAgentsChange:
    def test_modify_parameters_is_heard_with_the_new_value(self):
        agent, heard = _agent()
        out = _tool("modify_parameters")(
            embryo_id="embryo_2",
            changes={"num_slices": 30},
            reason="weak signal",
            context={"agent": agent},
        )
        assert out.startswith("Modified embryo_2")
        assert len(heard) == 1
        (row,) = heard[0].data["embryos"]
        assert row["num_slices"] == 30
        assert row["param_provenance"]["num_slices"] == {
            "by": "agent",
            "reason": "weak signal",
            "at": row["param_provenance"]["num_slices"]["at"],
        }

    def test_it_still_reports_what_changed_from_what(self):
        agent, _ = _agent()
        out = _tool("modify_parameters")(
            embryo_id="embryo_2", changes={"num_slices": 30}, reason="r", context={"agent": agent}
        )
        assert '"num_slices": 30' in out and '"num_slices": 50' in out

    def test_a_typo_is_refused_rather_than_silently_ignored(self):
        agent, heard = _agent()
        out = _tool("modify_parameters")(
            embryo_id="embryo_2", changes={"slices": 30}, reason="r", context={"agent": agent}
        )
        assert out.startswith("Unknown parameter")
        assert heard == []

    @pytest.mark.parametrize("key,pct", [("laser_power_488_pct", 50), ("laser_power_561_pct", 101)])
    def test_a_power_outside_the_limit_is_refused_before_the_write(self, key, pct):
        agent, heard = _agent()
        out = _tool("modify_parameters")(
            embryo_id="embryo_2", changes={key: pct}, reason="r", context={"agent": agent}
        )
        assert "outside hard safety limit" in out
        assert heard == [] and getattr(agent.experiment.embryos["embryo_2"], key) is None

    def test_a_bad_mode_is_refused_before_the_write(self):
        agent, heard = _agent()
        out = _tool("modify_parameters")(
            embryo_id="embryo_2",
            changes={"acquisition_mode": "hover", "num_slices": 30},
            reason="r",
            context={"agent": agent},
        )
        assert out.startswith("Invalid acquisition_mode")
        assert heard == [] and agent.experiment.embryos["embryo_2"].num_slices == 50


# ---------------------------------------------------------------------------
# The operator's screen reads it
# ---------------------------------------------------------------------------

OPERATE = (WEB / "static" / "js" / "operate.js").read_text(encoding="utf-8")
INDEX = (WEB / "templates" / "index.html").read_text(encoding="utf-8")
DATA = (WEB / "routes" / "data.py").read_text(encoding="utf-8")


def _js(name: str) -> str:
    body = OPERATE[OPERATE.index(f"function {name}(") :]
    return body[: body.index("\n    }")]


class TestTheScreen:
    def test_the_plans_fields_are_derived_from_the_targets(self):
        body = _js("syncPlanFromEmbryos")
        assert "heldAcross(targets, f.key)" in body
        assert "el.value = held.value != null ? held.value : ''" in body
        assert "`varies ${held.min}–${held.max}`" in body, "two values are shown as one"

    def test_the_fields_follow_every_embryo_change(self):
        body = _js("onEmbryosUpdate")
        assert "renderPlan(); renderRun();" in body
        assert "syncPlanFromEmbryos();" in _js("renderPlan")

    def test_it_reads_the_keys_the_broadcast_carries(self):
        for key in ("num_slices", "exposure_ms", "laser_power_488_pct"):
            assert f"key: '{key}'" in OPERATE, key
        assert "e.param_provenance" in _js("heldAcross")

    def test_a_pending_edit_is_kept_and_the_difference_said(self):
        body = _js("syncPlanFromEmbryos")
        assert "if (!_planDirty)" in body
        assert "the field applies at Start" in body
        assert 'id="op-plan-spim-now"' in INDEX

    def test_start_hands_the_form_back_to_the_embryos(self):
        branch = OPERATE[
            OPERATE.index("if (_mode === 'adaptive') {") : OPERATE.index(
                "if (_mode === 'library') {"
            )
        ]
        assert "_planDirty = false;" in branch

    def test_each_run_row_shows_its_own_parameters_and_who_set_them(self):
        assert "paramsWords(r)" in _js("runRow") and "paramsTitle(r)" in _js("runRow")
        for key in ('"num_slices"', '"exposure_ms"', '"laser_powers"', '"set_by"'):
            assert key in DATA, key
        assert "r.set_by" in _js("paramsTitle")

    def test_the_pane_writes_through_the_same_door(self):
        start = DATA[DATA.index('"/api/devices/timelapse/start"') :]
        start = start[: start.index("# --- Start timelapse")]
        assert "experiment.set_params(" in start and 'by="operator"' in start
        assert "emb.num_slices =" not in start and "emb.exposure_ms =" not in start
