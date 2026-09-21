"""An embryo can borrow another's calibration instead of paying for its own.

Calibrating costs sixty to eighty exposures of laser on a live embryo. When
another embryo on the same slide already carries a good fit, the cheapest
calibration is no calibration — and `apply_calibration_to_embryos` has existed
as an agent tool for exactly this the whole time, auto-picking the best source
by R². No surface ever offered it, so it could only be reached by asking the
agent in prose.

"a specific atomic tasks of calibration like ... use calibration of another
embryo ... all of these can be tasks."

What must stay true:

* "best" means one thing. The pane names the source it is about to copy and
  its R²; if the pane's idea of best and the tool's ever diverge, the pane
  would name one embryo and copy another's numbers. Both go through
  `rank_calibration_sources`.
* An embryo is never its own source, and a target that is already calibrated
  is not silently overwritten by something worse.
* A refusal is not a success. The tool reports its refusals in prose, so a
  route that returned 200 with the prose in it would leave the pane claiming
  a calibration that does not exist — the exact false-success the calibration
  gate exists to prevent.
"""

from __future__ import annotations

import re
from pathlib import Path
from unittest.mock import MagicMock

from fastapi import FastAPI
from fastapi.testclient import TestClient

import gently.ui.web.auth as auth
from gently.app.tools.calibration_tools import rank_calibration_sources
from gently.harness.state import EmbryoState
from gently.ui.web.routes.data import create_router

OPERATE_JS = (
    Path(__file__).resolve().parents[1] / "gently" / "ui" / "web" / "static" / "js" / "operate.js"
)


def _emb(eid: str, top: float | None = None, bot: float | None = None, **kw) -> EmbryoState:
    emb = EmbryoState(id=eid)
    emb.stage_position = {"x": 1.0, "y": 2.0}
    if top is not None or bot is not None:
        emb.calibration = {
            "slope_um_per_deg": 100.0,
            "r_squared": 0.9,
            "r_squared_top": top,
            "r_squared_bottom": bot,
        }
    for k, v in kw.items():
        setattr(emb, k, v)
    return emb


def _app(embryos: dict[str, EmbryoState]) -> TestClient:
    server = MagicMock()
    agent = server.agent_bridge.agent
    agent.experiment.embryos = embryos
    agent.client = MagicMock()
    agent.lightsheet_monitor = None
    app = FastAPI()
    app.include_router(create_router(server))
    app.dependency_overrides[auth.require_control] = lambda: True
    return TestClient(app)


# ---------------------------------------------------------------------------
# One definition of "best"
# ---------------------------------------------------------------------------


def test_the_worse_end_of_the_sweep_decides() -> None:
    """A wide-but-lopsided fit loses to an even one — acquisitions span both."""
    embryos = {
        "lopsided": _emb("lopsided", top=0.99, bot=0.55),
        "even": _emb("even", top=0.90, bot=0.89),
    }
    ranked = rank_calibration_sources(embryos)
    assert [eid for eid, _, _ in ranked] == ["even", "lopsided"], (
        "ranked by the better end, or by an average — an acquisition spans both "
        "galvo extremes and the weaker end dominates focus quality"
    )


def test_uncalibrated_and_skipped_embryos_are_not_sources() -> None:
    embryos = {
        "fine": _emb("fine", top=0.9, bot=0.9),
        "empty": _emb("empty"),
        "skipped": _emb("skipped", top=0.99, bot=0.99, should_skip=True),
    }
    assert [eid for eid, _, _ in rank_calibration_sources(embryos)] == ["fine"]


def test_the_pane_and_the_tool_use_the_same_metric() -> None:
    """CI runs no JS, so the pane's copy of the rule is pinned in source."""
    js = OPERATE_JS.read_text(encoding="utf-8")
    fn = re.search(r"const fitScore = e => \{(.*?)\n    \};", js, re.S)
    assert fn, "the pane no longer scores fits"
    body = fn.group(1)
    assert "r_squared_top" in body and "r_squared_bottom" in body
    assert "Math.min" in body, (
        "the pane stopped taking the worse end; it would name one embryo as best "
        "while the server copied another's numbers"
    )


def test_the_pane_only_offers_a_better_fit() -> None:
    js = OPERATE_JS.read_text(encoding="utf-8")
    best = re.search(r"function bestSource\(forEmbryo\) \{(.*?)\n    \}", js, re.S)
    assert best, "bestSource is gone"
    assert "best.score <= fitScore(forEmbryo)" in best.group(1), (
        "the pane offers to copy a fit no better than the one the embryo has — "
        "after one borrow that means offering an embryo a copy of its own fit"
    )


# ---------------------------------------------------------------------------
# The route
# ---------------------------------------------------------------------------


def test_auto_picks_the_best_other_embryo() -> None:
    target = _emb("embryo_1")
    embryos = {
        "embryo_1": target,
        "embryo_2": _emb("embryo_2", top=0.71, bot=0.69),
        "embryo_3": _emb("embryo_3", top=0.97, bot=0.94),
    }
    r = _app(embryos).post("/api/devices/embryos/embryo_1/calibration/borrow", json={})
    assert r.status_code == 200, r.text
    assert r.json()["source_embryo_id"] == "embryo_3"
    assert target.calibration["r_squared_top"] == 0.97


def test_the_copy_does_not_alias_the_source() -> None:
    """Two embryos sharing one dict would drift together on the next fit."""
    source = _emb("embryo_2", top=0.9, bot=0.9)
    target = _emb("embryo_1")
    _app({"embryo_1": target, "embryo_2": source}).post(
        "/api/devices/embryos/embryo_1/calibration/borrow", json={}
    )
    target.calibration["slope_um_per_deg"] = 1.0
    assert source.calibration["slope_um_per_deg"] == 100.0, (
        "the target aliases the source's calibration dict"
    )


def test_nothing_to_borrow_is_a_refusal_not_a_success() -> None:
    r = _app({"embryo_1": _emb("embryo_1")}).post(
        "/api/devices/embryos/embryo_1/calibration/borrow", json={}
    )
    assert r.status_code == 409
    assert "borrow" in r.json()["detail"].lower()


def test_an_embryo_cannot_borrow_from_itself() -> None:
    r = _app({"embryo_1": _emb("embryo_1", top=0.9, bot=0.9)}).post(
        "/api/devices/embryos/embryo_1/calibration/borrow", json={"source": "embryo_1"}
    )
    assert r.status_code == 400


def test_an_unknown_source_is_404_not_a_silent_no_op() -> None:
    r = _app({"embryo_1": _emb("embryo_1")}).post(
        "/api/devices/embryos/embryo_1/calibration/borrow", json={"source": "embryo_9"}
    )
    assert r.status_code == 404


def test_a_source_with_no_fit_does_not_report_success() -> None:
    """The tool refuses in prose; the route must not pass prose off as a fit."""
    target, source = _emb("embryo_1"), _emb("embryo_2")
    r = _app({"embryo_1": target, "embryo_2": source}).post(
        "/api/devices/embryos/embryo_1/calibration/borrow", json={"source": "embryo_2"}
    )
    assert r.status_code == 502
    assert not target.calibration


def test_the_sources_listing_excludes_the_embryo_asking() -> None:
    embryos = {
        "embryo_1": _emb("embryo_1", top=0.99, bot=0.99),
        "embryo_2": _emb("embryo_2", top=0.8, bot=0.8),
    }
    body = _app(embryos).get("/api/devices/embryos/embryo_1/calibration/sources").json()
    assert [s["embryo_id"] for s in body["sources"]] == ["embryo_2"]
    assert body["sources"][0]["quality"] == 0.8
