"""Detection is three stages, and the operator chooses which ones run.

    blobs   flat-field + scale-matched blob finder. Sets RECALL — the later
            stages only remove or refine what it proposes.
    claude  classifies each candidate crop, removing non-embryos. Needs a key.
    sam     segments inside the boxes it is handed, for an outline and an area.
            Needs the checkpoint, and a GPU to be quick.

Only the first is load-bearing for "where are the embryos". SAM was
nevertheless mandatory: the client refused the whole call without a checkpoint
and the route answered 503, so on a machine with no GPU the Detect button could
not run at all — including the blob pass that does the actual finding. And the
web UI sent a fixed body, so none of it was selectable from the surface whose
whole job is detecting things on the image.

Pinned here: the SAM stage is optional end to end, the gate follows the stage
actually requested, and the panel sends the choice. CI runs no JavaScript, so
the last part is a source assertion.
"""

from __future__ import annotations

import re
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import gently.ui.web.auth as auth
from gently.hardware.dispim.sam_detection import SAMEmbryoDetector
from gently.ui.web.routes.data import create_router

WEB = Path(__file__).resolve().parents[1] / "gently" / "ui" / "web"


def _app(client=None):
    server = MagicMock()
    server.agent_bridge.agent.client = client
    server.agent_bridge.agent.lightsheet_monitor = None
    app = FastAPI()
    app.include_router(create_router(server))
    app.dependency_overrides[auth.require_control] = lambda: True
    return TestClient(app)


def _client(has_sam: bool = True):
    client = MagicMock()
    client.has_sam = has_sam
    client.detect_embryos = AsyncMock(
        return_value={"success": True, "embryos": [], "stage_position": (0.0, 0.0)}
    )
    return client


# ── the detector ────────────────────────────────────────────────────────────


def test_candidates_alone_are_a_usable_answer() -> None:
    """Without SAM the blob candidates carry every field the UI consumes."""
    cands = [
        {"bbox": (10, 20, 30, 40), "centroid": (25.0, 40.0), "area": 900, "relative_strength": 0.8},
        {"bbox": (5, 5, 10, 10), "centroid": (10.0, 10.0), "area": 100, "relative_strength": 0.4},
    ]
    out = SAMEmbryoDetector.embryos_from_candidates(cands)
    assert [e["embryo_id"] for e in out] == ["embryo_1", "embryo_2"]
    assert (out[0]["pixel_x"], out[0]["pixel_y"]) == (25.0, 40.0)
    assert out[0]["bbox"] == (10, 20, 30, 40)
    assert out[0]["area_pixels"] == 900
    # The blob's relative strength is the only confidence there is here, and
    # circularity is SAM's measurement — reported as 0.0, never guessed.
    assert out[0]["confidence"] == pytest.approx(0.8)
    assert out[0]["circularity"] == 0.0
    assert out[0]["mask"] is None
    assert len({e["uid"] for e in out}) == 2


# ── the gate ────────────────────────────────────────────────────────────────


def test_a_blobs_only_run_does_not_need_sam() -> None:
    """The stage that is not asked for cannot block the call."""
    client = _client(has_sam=False)
    r = _app(client=client).post("/api/devices/detect_embryos", json={"use_sam": False})
    assert r.status_code == 200, (
        "detection without the SAM stage is refused on a device layer with no "
        "checkpoint — the blob pass that actually finds embryos never runs"
    )
    assert client.detect_embryos.await_args.kwargs["use_sam"] is False


def test_asking_for_sam_without_sam_still_refuses() -> None:
    r = _app(client=_client(has_sam=False)).post(
        "/api/devices/detect_embryos", json={"use_sam": True}
    )
    assert r.status_code == 503


def test_the_route_forwards_every_stage_choice() -> None:
    client = _client()
    body = {
        "use_sam": True,
        "use_claude_review": False,
        "min_relative_peak": 0.6,
        "use_last_frame": True,
    }
    assert _app(client=client).post("/api/devices/detect_embryos", json=body).status_code == 200
    kw = client.detect_embryos.await_args.kwargs
    assert kw["use_sam"] is True
    assert kw["use_claude_review"] is False
    assert kw["min_relative_peak"] == pytest.approx(0.6)
    assert kw["use_last_frame"] is True


# ── the panel ───────────────────────────────────────────────────────────────


def test_the_panel_offers_the_pipeline_and_sends_the_choice() -> None:
    panel = (WEB / "static" / "js" / "panels" / "marking.js").read_text(encoding="utf-8")
    operate = (WEB / "static" / "js" / "operate.js").read_text(encoding="utf-8")

    for stage in ('data-set="claude"', 'data-set="sam"', 'data-set="sensitivity"'):
        assert stage in panel, f"the detect panel no longer offers {stage}"
    assert "use_claude_review" in panel and "use_sam" in panel, (
        "the panel's options no longer name the pipeline stages the route takes"
    )
    # Blobs is not optional: it is what sets recall.
    assert "is-fixed" in panel, "the blob stage is presented as switchable, which it is not"
    assert re.search(r"detect:\s*opts\s*=>\s*runDetect\(opts\)", operate), (
        "the detect verb dropped its options again, so the panel's choice cannot reach the request"
    )
    assert "tune.use_sam = cfg.use_sam" in operate, (
        "runDetect no longer forwards the SAM stage choice"
    )
