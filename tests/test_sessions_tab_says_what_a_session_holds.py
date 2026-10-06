"""The Sessions tab says what a session holds before it is restored.

"can we also go over the sessions tab - and see if that view can be more
informative? maybe it can show some preview images? or some of the
acquisition parameters, etc in the sessions tab before i restore a session?"

And the detail pane was blank inside the app: its container reused the
``tab-content`` class that main.css hides for every app tab but the active
one. "currently i guess it shows the conversations? it might be broken too."
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import yaml
from fastapi import FastAPI
from fastapi.testclient import TestClient

from gently.core.file_store import FileStore
from gently.ui.web import auth
from gently.ui.web.routes import sessions as sessions_routes

pytest.importorskip("tifffile")
pytest.importorskip("PIL")

WEB = Path(__file__).resolve().parents[1] / "gently" / "ui" / "web"
REVIEW_JS = (WEB / "static" / "js" / "review.js").read_text(encoding="utf-8")
REVIEW_CSS = (WEB / "static" / "css" / "review.css").read_text(encoding="utf-8")
MAIN_CSS = (WEB / "static" / "css" / "main.css").read_text(encoding="utf-8")


@pytest.fixture
def store(tmp_path):
    return FileStore(root=tmp_path)


def _client(store, active=None):
    server = MagicMock()
    server.agent_bridge.agent.store = store
    server.agent_bridge.agent.session_id = active
    app = FastAPI()
    app.include_router(sessions_routes.create_router(server))
    app.dependency_overrides[auth.require_control] = lambda: True
    return TestClient(app)


def _session_with_a_run(store, sid="s1"):
    store.create_session(sid)
    store.register_embryo(
        sid, "embryo_1", nickname="A", position_x=1.0, position_y=2.0, role="test"
    )
    proj = store._embryo_dir(sid, "embryo_1") / "projections"
    proj.mkdir(parents=True, exist_ok=True)
    for tp in (1, 2, 3):
        (proj / f"t{tp:04d}.jpg").write_bytes(b"jpg")
        stage = ["bean", "comma", "1_5_fold"][tp - 1]
        store.store_prediction(1, sid, "embryo_1", tp, stage, confidence=0.7)
    store.save_acquisition_plan(
        sid,
        {
            "interval_seconds": 600,
            "num_slices": 60,
            "stop_condition": {"kind": "hatching", "value": None},
            "dic": {
                "enabled": True,
                "every_seconds": 1800,
                "light": "led",
                "led_intensity_pct": 30,
            },
        },
    )
    store.put_snapshot(
        sid, "dic", np.zeros((20, 30), dtype=np.uint16), metadata={"channel": "dic", "frame": 1}
    )
    (store._session_dir(sid) / "timelapse.yaml").write_text(
        yaml.safe_dump(
            {
                "status": "completed",
                "started_at": "2026-10-04T21:00:00",
                "saved_at": "2026-10-05T01:05:00",
                "current_round": 3,
                "total_timepoints": 3,
                "base_interval_seconds": 600,
                "embryos": {"embryo_1": {"is_complete": True, "timepoints_acquired": 3}},
            }
        ),
        encoding="utf-8",
    )
    return sid


class TestTheList:
    def test_each_session_says_how_much_it_holds(self, store):
        _session_with_a_run(store)
        s = _client(store).get("/api/sessions").json()["sessions"][0]
        assert (s["embryo_count"], s["timepoints"], s["dic_frames"]) == (1, 3, 1)
        # A finished run is described, but is not an interrupted one the gate would offer.
        assert s["run"] is None
        assert s["last_run"]["status"] == "completed" and s["last_run"]["total_timepoints"] == 3


class TestTheDetail:
    def test_the_plan_the_pictures_and_the_stage_calls_come_without_restoring(self, store):
        sid = _session_with_a_run(store)
        d = _client(store).get(f"/api/sessions/{sid}").json()
        assert d["acquisition"]["interval_seconds"] == 600
        assert d["acquisition"]["dic"]["every_seconds"] == 1800
        (e,) = d["embryos"]
        assert e["nickname"] == "A" and e["role"] == "test"
        assert e["timepoints"] == 3 and e["latest_timepoint"] == 3
        assert e["thumbnail"] == f"/api/sessions/{sid}/projection?embryo=embryo_1&t=3"
        assert e["stage"] == "1_5_fold"
        assert [p["stage"] for p in e["predictions"]] == ["bean", "comma", "1_5_fold"]
        (f,) = d["dic_frames"]
        assert f["frame"] == 1 and f["url"].startswith(f"/api/sessions/{sid}/snapshot/dic_")

    def test_a_dic_frame_of_any_session_renders_as_png(self, store):
        sid = _session_with_a_run(store)
        c = _client(store)
        url = c.get(f"/api/sessions/{sid}").json()["dic_frames"][0]["url"]
        r = c.get(url, params={"max": 16})
        assert r.status_code == 200 and r.headers["content-type"] == "image/png"
        assert c.get(f"/api/sessions/{sid}/snapshot/not_filed.png").status_code == 404


class TestThePane:
    def test_the_detail_pane_no_longer_wears_the_class_main_css_hides(self):
        assert ".tab-content {\n    display: none;" in MAIN_CSS
        assert 'class="tab-content"' not in REVIEW_JS
        assert "session-tab-content" in REVIEW_JS and ".session-tab-content" in REVIEW_CSS

    def test_the_pane_shows_plan_pictures_frames_and_stage_calls(self):
        needles = ("renderPlan(", "embryo-thumb", "renderDicStrip(", "stage-call", "tool_result")
        for needle in needles:
            assert needle in REVIEW_JS, needle
