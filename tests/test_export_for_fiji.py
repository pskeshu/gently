"""A session exported the way a biologist finds it again.

"a nice export method that can organize and store the data in a neat manner
that is easily usable in fiji … sorted by timepoint or something instead of
uid in filename … easier to find the original imprints of the experiment
etc, or stored with metadata files"
"""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import yaml
from fastapi import FastAPI
from fastapi.testclient import TestClient

from gently.core.export import export_session, label_for
from gently.core.file_store import FileStore
from gently.ui.web import auth
from gently.ui.web.routes import reveal as reveal_routes
from gently.ui.web.routes import sessions as sessions_routes

pytest.importorskip("tifffile")

WEB = Path(__file__).resolve().parents[1] / "gently" / "ui" / "web"
REVIEW_JS = (WEB / "static" / "js" / "review.js").read_text(encoding="utf-8")


@pytest.fixture
def store(tmp_path):
    return FileStore(root=tmp_path / "data")


def _session(store, sid="s1"):
    store.create_session(sid, name="N2 overnight", description="three embryos until hatching")
    store.register_embryo(
        store_sid := sid,
        "embryo_1",
        nickname="A",
        position_x=-500.0,
        position_y=-400.0,
        role="test",
    )
    store.register_embryo(
        sid, "embryo_2", nickname="ref 2", position_x=-300.0, position_y=-200.0, role="reference"
    )
    store.register_embryo(
        sid, "embryo_3", position_x=0.0, position_y=0.0, role="test"
    )  # no nickname
    for eid, n in (("embryo_1", 3), ("embryo_2", 2)):
        for tp in range(1, n + 1):
            vol = np.full((4, 8, 8), tp, dtype=np.uint16)
            store.put_volume(
                sid,
                eid,
                tp,
                vol,
                metadata={"num_slices": 4, "exposure_ms": 10.0, "laser_power_488_pct": 3.0},
            )
            store.store_prediction(
                1, sid, eid, tp, ["bean", "comma", "1_5_fold"][tp - 1], confidence=0.8
            )
    # DIC frames filed by uuid, out of frame order on disk
    for frame, when in ((2, "2026-10-04T22:00:00"), (1, "2026-10-04T21:30:00")):
        store.put_snapshot(
            sid,
            "dic",
            np.zeros((6, 9), dtype=np.uint16),
            metadata={
                "channel": "dic",
                "frame": frame,
                "captured_at": when,
                "position": {"x": -440.0, "y": -320.0},
            },
        )
    store.save_acquisition_plan(
        sid,
        {
            "interval_seconds": 600,
            "num_slices": 4,
            "stop_condition": {"kind": "hatching"},
            "dic": {"enabled": True, "every_seconds": 1800, "light": "led"},
        },
    )
    sd = store._session_dir(sid)
    (sd / "timelapse.yaml").write_text(
        yaml.safe_dump(
            {
                "status": "completed",
                "started_at": "2026-10-04T21:00:00",
                "embryos": {"embryo_1": {"is_complete": True, "total_exposure_ms": 300.0}},
            }
        ),
        encoding="utf-8",
    )
    (sd / "events.jsonl").write_text(
        "\n".join(
            json.dumps(r)
            for r in (
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
                    "event_type": "HATCHING_DETECTED",
                    "data": {"embryo_id": "embryo_1"},
                    "timestamp": "2026-10-05T00:50:00",
                },
            )
        ),
        encoding="utf-8",
    )
    store.append_temperature_sample(
        sid, {"t": "2026-10-04T21:00:00", "water_c": 20.1, "setpoint_c": 20.0, "state": "locked"}
    )
    return store_sid


def _rows(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


class TestTheLayout:
    def test_one_folder_per_embryo_named_by_what_it_was_called(self, store, tmp_path):
        sid = _session(store)
        out = export_session(store, sid)
        assert out == Path(store.root) / "exports" / store._session_dir(sid).name
        assert sorted(p.name for p in out.iterdir() if p.is_dir()) == [
            "A_embryo_1",
            "dic",
            "embryo_3",
            "metadata",
            "ref-2_embryo_2",
        ]

    def test_volumes_and_frames_sort_in_time_order(self, store):
        sid = _session(store)
        out = export_session(store, sid)
        vols = sorted(p.name for p in (out / "A_embryo_1" / "volumes").iterdir())
        assert vols == ["A_embryo_1_t0001.tif", "A_embryo_1_t0002.tif", "A_embryo_1_t0003.tif"]
        dic = sorted(p.name for p in (out / "dic").iterdir() if p.suffix == ".tif")
        assert dic == ["dic_f0001_20261004-213000.tif", "dic_f0002_20261004-220000.tif"]
        import tifffile

        assert tifffile.imread(out / "A_embryo_1" / "volumes" / "A_embryo_1_t0002.tif").max() == 2

    def test_copies_not_links(self, store):
        sid = _session(store)
        out = export_session(store, sid)
        exported = out / "A_embryo_1" / "volumes" / "A_embryo_1_t0001.tif"
        original = Path(store.get_volume_path(sid, "embryo_1", 1))
        assert (
            exported.stat().st_ino != original.stat().st_ino
            or exported.stat().st_dev != original.stat().st_dev
        )


class TestTheRecord:
    def test_the_metadata_travels_as_kept_and_as_read(self, store):
        sid = _session(store)
        out = export_session(store, sid)
        for name in (
            "session.yaml",
            "acquisition.yaml",
            "timelapse.yaml",
            "events.jsonl",
            "temperature.jsonl",
        ):
            assert (out / "metadata" / name).is_file(), name
        embryos = {r["embryo_id"]: r for r in _rows(out / "embryos.csv")}
        assert embryos["embryo_1"]["label"] == "A_embryo_1"
        assert (
            embryos["embryo_1"]["timepoints"] == "3"
            and embryos["embryo_1"]["last_stage"] == "1_5_fold"
        )
        assert (
            embryos["embryo_1"]["complete"] == "True"
            and embryos["embryo_1"]["total_exposure_ms"] == "300.0"
        )
        assert (
            embryos["embryo_2"]["role"] == "reference" and embryos["embryo_2"]["x_um"] == "-300.0"
        )
        calls = _rows(out / "stage_calls.csv")
        assert [(c["label"], c["timepoint"], c["stage"]) for c in calls][:3] == [
            ("A_embryo_1", "1", "bean"),
            ("A_embryo_1", "2", "comma"),
            ("A_embryo_1", "3", "1_5_fold"),
        ]
        events = _rows(out / "events.csv")
        assert [e["type"] for e in events] == [
            "ACQUISITION_STARTED",
            "HATCHING_DETECTED",
        ]  # no status chatter
        assert _rows(out / "temperature.csv")[0]["water_c"] == "20.1"
        vols = _rows(out / "A_embryo_1" / "volumes.csv")
        assert vols[0]["file"] == "volumes/A_embryo_1_t0001.tif" and vols[0]["z"] == "4"
        assert vols[0]["laser_488_pct"] == "3.0" and vols[0]["exposure_ms"] == "10.0"
        dic = _rows(out / "dic" / "dic.csv")
        assert [d["frame"] for d in dic] == ["1", "2"] and dic[0]["x_um"] == "-440.0"

    def test_the_readme_points_back_at_the_originals(self, store):
        sid = _session(store)
        out = export_session(store, sid)
        text = (out / "README.txt").read_text(encoding="utf-8")
        assert text.startswith("N2 overnight\n")
        assert f"Originals: {store._session_dir(sid)}" in text
        assert "three embryos until hatching" in text
        assert "Interval: every 600 s" in text and "DIC overview: every 1800 s, led" in text
        assert "A_embryo_1/   role test, 3 timepoints, last stage 1_5_fold, complete" in text
        assert "Import > Image Sequence" in text
        assert "copies, not links" in text

    def test_a_destination_of_choice(self, store, tmp_path):
        sid = _session(store)
        out = export_session(store, sid, tmp_path / "usb")
        assert out.parent == tmp_path / "usb" and (out / "README.txt").is_file()

    def test_progress_counts_every_file(self, store):
        sid = _session(store)
        seen = []
        export_session(store, sid, progress=lambda d, t, w: seen.append((d, t)))
        done, total = seen[-1]
        assert (
            done == total and total == 5 + 2 + 5 + 6
        )  # 5 volumes, 2 DIC frames, 5 projections (filed with the volumes), six records


class TestLabels:
    def test_labels_are_safe_and_distinct(self):
        assert label_for({"embryo_id": "embryo_1", "nickname": "A"}) == "A_embryo_1"
        assert (
            label_for({"embryo_id": "embryo_2", "nickname": "the fast one!"})
            == "the-fast-one_embryo_2"
        )
        assert label_for({"embryo_id": "embryo_3", "nickname": None}) == "embryo_3"
        assert label_for({"embryo_id": "embryo_4", "nickname": "embryo_4"}) == "embryo_4"


# ── the routes ──────────────────────────────────────────────────────────────


def _client(store):
    server = MagicMock()
    server.agent_bridge.agent.store = store
    server.agent_bridge.agent.session_id = None
    app = FastAPI()
    app.include_router(sessions_routes.create_router(server))
    app.include_router(reveal_routes.create_router(server))
    app.dependency_overrides[auth.require_control] = lambda: True
    return TestClient(app)


class TestTheRoute:
    def test_export_runs_in_the_background_and_reports_where_it_went(self, store):
        sid = _session(store)
        c = _client(store)
        idle = c.get(f"/api/sessions/{sid}/export").json()
        assert idle["state"] == "idle" and idle["default_dest"] == str(Path(store.root) / "exports")
        started = c.post(f"/api/sessions/{sid}/export", json={})
        assert started.status_code == 200 and started.json()["state"] in ("running", "done")
        for _ in range(100):
            job = c.get(f"/api/sessions/{sid}/export").json()
            if job["state"] != "running":
                break
            time.sleep(0.05)
        assert job["state"] == "done", job
        assert job["done"] == job["total"] and Path(job["path"]).joinpath("README.txt").is_file()
        # and the export folder is a thing the file manager can be pointed at
        r = c.post("/api/reveal", json={"what": "export", "session_id": sid, "action": "path"})
        assert r.status_code == 200, r.text
        assert Path(r.json()["path"]) == Path(job["path"])

    def test_a_relative_destination_is_refused(self, store):
        sid = _session(store)
        r = _client(store).post(f"/api/sessions/{sid}/export", json={"dest": "exports-here"})
        assert r.status_code == 400

    def test_an_unknown_session_is_404(self, store):
        assert _client(store).post("/api/sessions/nope/export", json={}).status_code == 404


class TestThePane:
    def test_the_header_offers_the_export_and_says_what_it_is(self):
        for needle in (
            "Export for Fiji",
            "startExport(",
            "pollExport(",
            "/export",
            "what: 'export'",
        ):
            assert needle in REVIEW_JS, needle
        assert "Copies, not links" in REVIEW_JS
