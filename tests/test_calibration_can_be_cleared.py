"""A fit can be thrown away.

"should be a way to reset calibration for each embryo."

There was none. A calibration could be created (calibrate), copied (borrow)
and overwritten (recalibrate), but never removed — so an embryo whose fit was
known to be wrong stayed "calibrated" forever, and the calibration gate, which
refuses to start a run on an *un*calibrated embryo, waved it straight through.
That is the gate's own failure mode inverted: instead of refusing a run that
should happen, it permits one that should not, on numbers nobody trusts.

The property that makes this real rather than cosmetic is **persistence**.
`embryo.calibration` is written into the session's `embryo.yaml`, and
`FileStore.register_embryo` COALESCES — `calibration=None` keeps whatever is
already on disk. A clear that passed None, or that only touched memory, would
look right, survive the afternoon, and come back at the next session restore
with nobody able to say which numbers the data was taken with.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import gently.ui.web.auth as auth
from gently.app.tools.calibration_tools import clear_embryo_calibration
from gently.core.file_store import FileStore
from gently.harness import calibration_gate
from gently.harness.state import EmbryoState
from gently.ui.web.routes.data import create_router

FIT = {
    "slope_um_per_deg": 101.2,
    "offset_um": 5.25,
    "r_squared": 0.96,
    "r_squared_top": 0.97,
    "r_squared_bottom": 0.94,
}


def _emb(eid: str, cal: dict | None = None) -> EmbryoState:
    emb = EmbryoState(id=eid)
    emb.position_coarse = {"x": -499.4, "y": -367.9}
    emb.calibration = dict(cal) if cal else {}
    return emb


class _Experiment:
    """Enough ExperimentState for the tool.

    Not a MagicMock: the tool resolves through `get_embryo_by_any_name`, and a
    MagicMock answers that with another MagicMock — which reads as an embryo
    with an empty calibration, so every assertion passes for the wrong reason.
    """

    def __init__(self, embryos: dict[str, EmbryoState]) -> None:
        self.embryos = embryos
        self.notified = 0

    def get_embryo_by_any_name(self, name: str) -> EmbryoState | None:
        return self.embryos.get(name)

    def notify_embryos_changed(self) -> None:
        self.notified += 1


def _agent(embryos: dict[str, EmbryoState], store: Any = None) -> MagicMock:
    agent = MagicMock()
    agent.experiment = _Experiment(embryos)
    agent.store = store
    agent.session_id = "s1" if store is not None else None
    return agent


def _call(agent: MagicMock, embryo_id: str) -> str:
    """The tool is sync but wrapped by @tool; call the underlying function."""
    fn = getattr(clear_embryo_calibration, "__wrapped__", clear_embryo_calibration)
    out = fn(embryo_id=embryo_id, context={"agent": agent})
    if asyncio.iscoroutine(out):
        out = asyncio.get_event_loop().run_until_complete(out)
    return out


# ---------------------------------------------------------------------------
# The tool
# ---------------------------------------------------------------------------


def test_clearing_makes_the_gate_refuse_the_embryo_again() -> None:
    """The whole point: a fit nobody trusts stops being a licence to run."""
    emb = _emb("embryo_1", FIT)
    assert calibration_gate.is_calibrated(emb)
    _call(_agent({"embryo_1": emb}), "embryo_1")
    assert not calibration_gate.is_calibrated(emb), (
        "the embryo still passes the calibration gate after its fit was cleared"
    )


def test_it_says_what_it_discarded() -> None:
    """The numbers are gone; the record of what they were should not be."""
    msg = _call(_agent({"embryo_1": _emb("embryo_1", FIT)}), "embryo_1")
    assert "101.2" in msg, f"the message does not name the discarded fit: {msg}"


def test_clearing_nothing_is_not_reported_as_a_clear() -> None:
    msg = _call(_agent({"embryo_1": _emb("embryo_1")}), "embryo_1")
    assert "no calibration" in msg.lower()


def test_the_clear_reaches_disk(tmp_path) -> None:
    """A clear that only touched memory comes back at the next restore."""
    store = FileStore(root=tmp_path)
    session_id = store.create_session("clear-test-1", name="clear-test")
    store.register_embryo(session_id, "embryo_1", position_x=1.0, position_y=2.0, calibration=FIT)
    stored = store.get_embryo(session_id, "embryo_1")
    assert stored is not None
    assert (stored.get("calibration") or {}).get("slope_um_per_deg")

    agent = _agent({"embryo_1": _emb("embryo_1", FIT)}, store=store)
    agent.session_id = session_id
    _call(agent, "embryo_1")

    on_disk = store.get_embryo(session_id, "embryo_1")
    assert on_disk is not None
    assert not (on_disk.get("calibration") or {}), (
        "the fit is still on disk — register_embryo COALESCES, so a clear that "
        "passes None keeps the old calibration and it returns on restore"
    )


def test_a_clear_that_could_not_persist_says_so() -> None:
    """Silence here would be the worst outcome: reset today, calibrated tomorrow."""
    store = MagicMock()
    store.register_embryo.side_effect = OSError("disk gone")
    agent = _agent({"embryo_1": _emb("embryo_1", FIT)}, store=store)
    msg = _call(agent, "embryo_1")
    assert "in memory only" in msg


# ---------------------------------------------------------------------------
# The route
# ---------------------------------------------------------------------------


def _app(embryos: dict[str, EmbryoState]) -> TestClient:
    server = MagicMock()
    agent = server.agent_bridge.agent
    agent.experiment = _Experiment(embryos)
    agent.store = None
    agent.session_id = None
    agent.client = MagicMock()
    agent.lightsheet_monitor = None
    app = FastAPI()
    app.include_router(create_router(server))
    app.dependency_overrides[auth.require_control] = lambda: True
    return TestClient(app)


def test_the_route_clears_and_returns_what_it_removed() -> None:
    emb = _emb("embryo_1", FIT)
    r = _app({"embryo_1": emb}).request("DELETE", "/api/devices/embryos/embryo_1/calibration")
    assert r.status_code == 200, r.text
    assert r.json()["cleared"]["slope_um_per_deg"] == 101.2
    assert not emb.calibration


def test_an_unknown_embryo_is_404() -> None:
    r = _app({"embryo_1": _emb("embryo_1", FIT)}).request(
        "DELETE", "/api/devices/embryos/embryo_9/calibration"
    )
    assert r.status_code == 404


@pytest.mark.parametrize("started_with", [FIT, {}])
def test_the_embryo_is_uncalibrated_afterwards_either_way(started_with: dict) -> None:
    """Idempotent: clearing twice is not an error, it is just already clear."""
    emb = _emb("embryo_1", started_with)
    client = _app({"embryo_1": emb})
    client.request("DELETE", "/api/devices/embryos/embryo_1/calibration")
    assert not calibration_gate.is_calibrated(emb)


def test_the_pane_asks_twice_before_discarding() -> None:
    """~70 exposures bought that fit; one stray click should not spend them."""
    import re
    from pathlib import Path

    js = (
        Path(__file__).resolve().parents[1]
        / "gently"
        / "ui"
        / "web"
        / "static"
        / "js"
        / "operate.js"
    ).read_text(encoding="utf-8")
    fn = re.search(r"async function clearFit\(\) \{(.*?)\n    \}", js, re.S)
    assert fn, "clearFit is gone"
    body = fn.group(1)
    assert "if (!_clearArmed)" in body, "the clear fires on a single click"
    # And the arming must expire, or a button left armed yesterday fires today.
    assert "_clearTimer = setTimeout(disarmClear" in body, "an armed clear never disarms"
