"""Calibration looks once before spending sixty exposures.

Kesavan, in the Calibration pane: "if we can do a pre calibration check —
where we are checking before edge detection — if there are any object in the
image to begin calibration. this can be done by sending the image to the
claude vlm."

The reason it matters is not just the dose. On an empty field, calibration did
not fail — it *succeeded*, producing a slope, an offset and a scan cuboid
fitted to noise. `calibration_gate` then accepts that as a licence to run a
timelapse, because the gate checks for the presence of a fit, not its meaning.
The rig's own embryo_1 carries R² 0.065, which is what that looks like
afterwards. One frame, asked the question Claude is already asked seventy
times during the sweep, distinguishes the two cases up front.

What must stay true:

* the happy path costs ONE frame — a check that is expensive is a check people
  turn off;
* a refusal is never silent about why, because "empty field", "head not
  focused" and "laser never fired" have different fixes and look identical in
  a status code; and
* it can always be overridden, because a dim embryo Claude misjudges must not
  become uncalibratable.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest

from gently.app.tools.calibration_tools import (
    NO_OBJECT_PREFIX,
    PROBE_DATA_TYPE,
    no_object_refusal,
    observe_at_galvo,
    probe_for_object,
)


def _client(frames: int = 9) -> MagicMock:
    """A camera that returns a real array, so normalisation runs for real."""
    client = MagicMock()
    img = np.linspace(0, 4095, 64 * 64, dtype=np.float32).reshape(64, 64)
    client.capture_lightsheet_image = AsyncMock(
        side_effect=[{"success": True, "image": img} for _ in range(frames)]
    )
    return client


def _vision(*verdicts: tuple[bool, int, str]) -> MagicMock:
    vision = MagicMock()
    vision.detect_embryo_presence = AsyncMock(side_effect=list(verdicts))
    return vision


def _agent() -> MagicMock:
    agent = MagicMock()
    agent.viz_server = object()
    agent.push_viz = MagicMock()
    return agent


# ---------------------------------------------------------------------------
# The probe
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_an_embryo_at_the_expected_focus_costs_one_frame() -> None:
    client = _client()
    vision = _vision((True, 8, "embryo, dense nuclei"))
    found, seen = await probe_for_object(
        client, vision, embryo_id="embryo_1", slope=100.0, offset=0.0
    )
    assert found is True
    assert len(seen) == 1, "the probe kept looking after it had its answer"
    assert client.capture_lightsheet_image.await_count == 1


@pytest.mark.asyncio
async def test_an_embryo_off_centre_in_z_is_not_refused() -> None:
    """The expected focus is a heuristic, so one empty frame proves nothing."""
    client = _client()
    vision = _vision(
        (False, 0, "uniform background"),
        (True, 6, "faint embryo edge"),
    )
    found, seen = await probe_for_object(
        client, vision, embryo_id="embryo_1", slope=100.0, offset=0.0, max_range=0.5
    )
    assert found is True, (
        "an embryo away from the heuristic focus was refused for being somewhere "
        "the guess did not look"
    )
    assert [round(o["galvo"], 3) for o in seen] == [0.0, 0.25]


@pytest.mark.asyncio
async def test_an_empty_field_is_refused_after_three_frames() -> None:
    client = _client()
    vision = _vision(
        (False, 0, "uniform background, no structure"),
        (False, 0, "empty"),
        (False, 0, "empty"),
    )
    found, seen = await probe_for_object(
        client, vision, embryo_id="embryo_1", slope=100.0, offset=0.0, max_range=0.5
    )
    assert found is False
    assert len(seen) == 3, "refused on fewer probes than it promised, or kept going"
    # Both sides of the expected focus, not just one.
    assert sorted(round(o["galvo"], 3) for o in seen) == [-0.25, 0.0, 0.25]


@pytest.mark.asyncio
async def test_the_probe_frames_are_shown_to_the_operator() -> None:
    """The frames behind a refusal are the evidence for overriding it."""
    agent = _agent()
    await probe_for_object(
        _client(),
        _vision((False, 0, "empty"), (False, 0, "empty"), (False, 0, "empty")),
        embryo_id="embryo_1",
        slope=100.0,
        offset=0.0,
        agent=agent,
    )
    pushed = [c.kwargs for c in agent.push_viz.call_args_list]
    assert len(pushed) == 3, "the probe took frames the operator never sees"
    assert {p["data_type"] for p in pushed} == {PROBE_DATA_TYPE}
    assert all(p["metadata"]["description"] for p in pushed), (
        "the pushed frame does not carry Claude's words, so the pane can only "
        "say 'nothing there' without saying what it saw"
    )


@pytest.mark.asyncio
async def test_a_camera_that_returns_nothing_is_not_read_as_an_empty_field() -> None:
    """No frame and an empty frame are different problems."""
    client = MagicMock()
    client.capture_lightsheet_image = AsyncMock(
        return_value={"success": False, "error": "camera busy"}
    )
    vision = _vision()
    found, seen = await probe_for_object(
        client, vision, embryo_id="embryo_1", slope=100.0, offset=0.0
    )
    assert found is False
    assert all(o["exposed"] is False for o in seen)
    assert vision.detect_embryo_presence.await_count == 0, "asked Claude about a frame we never got"
    assert "camera busy" in no_object_refusal("embryo_1", seen)


# ---------------------------------------------------------------------------
# The refusal
# ---------------------------------------------------------------------------


def test_the_refusal_quotes_claude_per_position_and_names_the_way_past() -> None:
    seen = [
        {
            "galvo": 0.0,
            "piezo": 0.0,
            "exposed": True,
            "visible": False,
            "description": "empty field",
        },
        {"galvo": 0.25, "piezo": 25.0, "exposed": True, "visible": False, "description": "blurred"},
    ]
    msg = no_object_refusal("embryo_1", seen)
    assert msg.startswith(NO_OBJECT_PREFIX), (
        "the route keys its 409 off this prefix; changing it turns a refusal "
        "back into an indistinguishable failure"
    )
    assert "empty field" in msg and "blurred" in msg
    assert "require_object=false" in msg, "the refusal does not say how to override it"
    # The three things that produce an empty frame and need different fixes.
    for hint in ("focused", "laser", "stage"):
        assert hint in msg


# ---------------------------------------------------------------------------
# One implementation, shared with the sweep
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_the_sweep_and_the_probe_ask_the_same_way() -> None:
    """Two copies of capture-normalise-ask would drift, and a refusal would
    then disagree with the sweep that follows it."""
    import inspect

    from gently.app.tools.calibration_tools import calibrate_embryo

    src = inspect.getsource(getattr(calibrate_embryo, "__wrapped__", calibrate_embryo))
    assert "observe_at_galvo(" in src, (
        "the edge sweep no longer goes through the shared observer; the probe "
        "and the sweep can now disagree about what 'visible' means"
    )
    assert "detect_embryo_presence" not in src, (
        "the sweep grew its own copy of the Claude call back"
    )


@pytest.mark.asyncio
async def test_the_observation_reports_what_it_saw_not_just_whether() -> None:
    obs = await observe_at_galvo(
        _client(),
        _vision((True, 7, "embryo with visible nuclei")),
        galvo=-0.1,
        piezo=-10.0,
        embryo_id="embryo_1",
    )
    assert obs["visible"] is True
    assert obs["feature_score"] == 7
    assert obs["description"] == "embryo with visible nuclei"
    assert obs["exposed"] is True
    assert obs["galvo"] == -0.1 and obs["piezo"] == -10.0


# ---------------------------------------------------------------------------
# The route, and what the pane can tell apart
# ---------------------------------------------------------------------------


def _route_app(message: str, executed: list, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    import gently.harness.tools.registry as registry_mod
    import gently.ui.web.auth as auth
    from gently.harness.state import EmbryoState
    from gently.ui.web.routes.data import create_router

    emb = EmbryoState(id="embryo_1")
    emb.stage_position = {"x": 1.0, "y": 2.0}
    server = MagicMock()
    agent = server.agent_bridge.agent
    agent.experiment.embryos = {"embryo_1": emb}
    client = MagicMock()
    client.is_connected = True
    agent.client = client
    agent.lightsheet_monitor = None

    async def _execute(name, args, ctx):
        executed.append((name, args))
        return message

    reg = MagicMock()
    reg.execute = _execute
    monkeypatch.setattr(registry_mod, "get_tool_registry", lambda: reg)
    app = FastAPI()
    app.include_router(create_router(server))
    app.dependency_overrides[auth.require_control] = lambda: True
    return TestClient(app)


def test_a_refusal_is_409_so_the_pane_can_offer_the_override(monkeypatch) -> None:
    """A 502 would read as a crash; nothing crashed, the run declined."""
    executed: list = []
    refusal = no_object_refusal(
        "embryo_1",
        [{"galvo": 0.0, "piezo": 0.0, "exposed": True, "visible": False, "description": "empty"}],
    )
    r = _route_app(refusal, executed, monkeypatch).post(
        "/api/devices/embryos/embryo_1/calibrate", json={}
    )
    assert r.status_code == 409, r.text
    assert "empty" in r.json()["detail"]


def test_the_check_is_on_by_default_and_the_route_does_not_ask_for_it(monkeypatch) -> None:
    """Absent means check: the tool's default, not a value the pane invents."""
    executed: list = []
    _route_app("Calibration complete", executed, monkeypatch).post(
        "/api/devices/embryos/embryo_1/calibrate", json={}
    )
    assert "require_object" not in executed[0][1]


def test_calibrate_anyway_reaches_the_tool(monkeypatch) -> None:
    executed: list = []
    _route_app("Calibration complete", executed, monkeypatch).post(
        "/api/devices/embryos/embryo_1/calibrate", json={"require_object": False}
    )
    assert executed[0][1]["require_object"] is False


def test_the_progress_panel_knows_about_probe_frames() -> None:
    """The panel listens on the image broadcast; a type it does not know is a
    phase that goes silently dark."""
    import re
    from pathlib import Path

    js = (
        Path(__file__).resolve().parents[1]
        / "gently"
        / "ui"
        / "web"
        / "static"
        / "js"
        / "panels"
        / "calprogress.js"
    ).read_text(encoding="utf-8")
    types = re.search(r"const TYPES = new Set\(\[(.*?)\]\)", js, re.S)
    assert types and PROBE_DATA_TYPE in types.group(1)
    assert "Checking there is something there" in js, "the probe phase has no words"
