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
from gently.ui.web.routes.sessions import derive_session_name

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
        assert s["bytes"] > 0  # what the folder occupies on disk
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


def _with_a_night(store, sid):
    import json

    sd = store._session_dir(sid)
    rows = [
        {
            "event_type": "ACQUISITION_STARTED",
            "data": {"embryo_ids": ["embryo_1"]},
            "timestamp": "2026-10-04T21:00:00",
        },
        {
            "event_type": "STATUS_CHANGED",
            "data": {"service": "mesh"},
            "timestamp": "2026-10-04T21:00:01",
        },
        {
            "event_type": "WARNING_ISSUED",
            "data": {"message": "drift"},
            "timestamp": "2026-10-04T23:00:00",
        },
        {
            "event_type": "ACQUISITION_STOPPED",
            "data": {"reason": "restart"},
            "timestamp": "2026-10-05T01:00:00",
        },
    ]
    (sd / "events.jsonl").write_text(
        "\n".join(json.dumps(r) for r in rows) + "\ntruncated{", encoding="utf-8"
    )
    for k in range(600):
        store.append_temperature_sample(
            sid, {"t": f"2026-10-04T21:{k % 60:02d}:00", "water_c": 20.0, "setpoint_c": 20.0}
        )
    store.register_embryo(sid, "embryo_9", nickname="dud", role="test")
    store.set_aside_embryo(sid, "embryo_9", by="operator: nothing there")
    doc = yaml.safe_load((sd / "timelapse.yaml").read_text(encoding="utf-8"))
    doc["dose_budget_base_ms"] = 1000.0
    doc["embryos"]["embryo_1"]["total_exposure_ms"] = 750.0
    (sd / "timelapse.yaml").write_text(yaml.safe_dump(doc), encoding="utf-8")


class TestWhatHappened:
    def test_notable_events_temperature_dose_and_set_aside_come_along(self, store):
        sid = _session_with_a_run(store)
        _with_a_night(store, sid)
        d = _client(store).get(f"/api/sessions/{sid}").json()
        kinds = [e["type"] for e in d["events"]]
        assert kinds == ["ACQUISITION_STARTED", "WARNING_ISSUED", "ACQUISITION_STOPPED"], kinds
        assert d["events"][1]["level"] == "warn" and "drift" in d["events"][1]["text"]
        assert 2 <= len(d["temperature"]) <= 240 and d["temperature"][0]["water_c"] == 20.0
        (e,) = d["embryos"]
        assert e["projection_timepoints"] == [1, 2, 3]
        assert e["dose_ms"] == 750.0 and e["dose_budget_ms"] == 1000.0  # test role: ×1
        (gone,) = d["removed_embryos"]
        assert gone["embryo_id"] == "embryo_9" and "nothing there" in gone["reason"]


class TestNaming:
    def test_an_unnamed_session_is_called_by_its_facts(self, store):
        sid = _session_with_a_run(store)
        c = _client(store)
        row = c.get("/api/sessions").json()["sessions"][0]
        assert row["name"] == sid  # the list keeps the id as the fallback name
        assert row["suggested_name"] == "1 embryo · every 10 min · Oct 4 overnight · complete"
        d = c.get(f"/api/sessions/{sid}").json()
        assert d["name"] is None and d["suggested_name"] == row["suggested_name"]

    def test_the_derived_name_reads_well_with_little_to_go_on(self):
        assert derive_session_name(created_at="2026-10-06T09:38:00") == "Oct 6"
        assert derive_session_name(created_at=None) == "Unnamed session"
        assert (
            derive_session_name(
                created_at="2026-10-04T21:00:00",
                embryo_count=3,
                last_run={"status": "paused", "embryos_going": 2, "interval_seconds": 600},
            )
            == "3 embryos · every 10 min · Oct 4 overnight · cut short"
        )

    def test_renaming_writes_the_session_yaml(self, store):
        sid = _session_with_a_run(store)
        c = _client(store)
        r = c.patch(
            f"/api/sessions/{sid}", json={"name": "  N2 overnight ", "description": "went fine"}
        )
        assert r.status_code == 200 and r.json()["name"] == "N2 overnight"
        assert store.get_session(sid)["name"] == "N2 overnight"
        assert store.get_session(sid)["description"] == "went fine"
        assert c.get("/api/sessions").json()["sessions"][0]["name"] == "N2 overnight"
        assert c.patch(f"/api/sessions/{sid}", json={"name": "x" * 121}).status_code == 400
        assert c.patch("/api/sessions/nope", json={"name": "x"}).status_code == 404

    def test_without_an_api_key_the_suggestion_is_the_derived_name(self, store):
        sid = _session_with_a_run(store)
        server = MagicMock()
        server.agent_bridge.agent.store = store
        server.agent_bridge.agent.session_id = None
        server.agent_bridge.agent.api_enabled = False
        app = FastAPI()
        app.include_router(sessions_routes.create_router(server))
        app.dependency_overrides[auth.require_control] = lambda: True
        got = TestClient(app).post(f"/api/sessions/{sid}/suggest-name").json()
        assert got["source"] == "derived" and got["name"].endswith("complete")
        server.agent_bridge.agent.claude.with_options.assert_not_called()

    def test_with_a_model_the_suggestion_is_what_it_wrote(self, store):
        sid = _session_with_a_run(store)
        server = MagicMock()
        server.agent_bridge.agent.store = store
        server.agent_bridge.agent.session_id = None
        server.agent_bridge.agent.api_enabled = True
        reply = MagicMock()
        reply.content = [
            MagicMock(
                type="text",
                text=(
                    '```json\n{"name": "Hatching watch, one embryo", '
                    '"description": "One embryo imaged every ten minutes until it hatched."}\n```'
                ),
            )
        ]
        server.agent_bridge.agent.claude.with_options.return_value.messages.create.return_value = (
            reply
        )
        app = FastAPI()
        app.include_router(sessions_routes.create_router(server))
        app.dependency_overrides[auth.require_control] = lambda: True
        got = TestClient(app).post(f"/api/sessions/{sid}/suggest-name").json()
        assert got == {
            "name": "Hatching watch, one embryo",
            "description": "One embryo imaged every ten minutes until it hatched.",
            "source": "model",
        }
        create = server.agent_bridge.agent.claude.with_options.return_value.messages.create
        sent = create.call_args.kwargs
        assert "embryo_1" in sent["messages"][0]["content"] and sent["max_tokens"] == 200

    def test_a_model_that_does_not_answer_in_json_falls_back(self, store):
        sid = _session_with_a_run(store)
        server = MagicMock()
        server.agent_bridge.agent.store = store
        server.agent_bridge.agent.session_id = None
        server.agent_bridge.agent.api_enabled = True
        reply = MagicMock()
        reply.content = [MagicMock(type="text", text="I would call it the hatching one.")]
        server.agent_bridge.agent.claude.with_options.return_value.messages.create.return_value = (
            reply
        )
        app = FastAPI()
        app.include_router(sessions_routes.create_router(server))
        app.dependency_overrides[auth.require_control] = lambda: True
        got = TestClient(app).post(f"/api/sessions/{sid}/suggest-name").json()
        assert got["source"] == "derived" and got["name"].endswith("complete")


class TestThePaneAgain:
    def test_the_pane_names_scrubs_bars_doses_and_tells_what_happened(self):
        needles = (
            "editName(",
            "suggestName(",
            "saveName(",
            "wireScrub(",
            "stageBar(",
            "dose(",
            "renderEventsTab(",
            "temperatureSpark(",
            "renderRemoved(",
            "session-search",
        )
        for needle in needles:
            assert needle in REVIEW_JS, needle
        panel = (WEB / "templates" / "_sessions_panel.html").read_text(encoding="utf-8")
        assert 'id="session-search"' in panel
        review_page = (WEB / "templates" / "review.html").read_text(encoding="utf-8")
        assert "stage-colors.js" in review_page, (
            "the stage bar needs the shared ramp on /review too"
        )
