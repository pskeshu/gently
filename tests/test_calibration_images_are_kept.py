"""What a calibration looked at is kept with the embryo.

A calibration's conclusion was stored: a slope, an offset, two R². Its evidence
was not. Every exposure, curve and montage went to the browser's image store,
in memory, and was gone at the next restart.

"store calibration = yes"
"""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
import tifffile
import yaml
from fastapi import FastAPI
from fastapi.testclient import TestClient

import gently.app.tools.calibration_tools as cal
from gently.core import calibration_record as rec
from gently.core.file_store import FileStore
from gently.ui.web import auth
from gently.ui.web.routes import calibration_records as routes
from gently.ui.web.routes import sessions as sessions_routes

ROOT = Path(__file__).resolve().parents[1]
TOOLS = (ROOT / "gently" / "app" / "tools" / "calibration_tools.py").read_text(encoding="utf-8")
OPERATE = (ROOT / "gently" / "ui" / "web" / "static" / "js" / "operate.js").read_text(
    encoding="utf-8"
)
HTML = (ROOT / "gently" / "ui" / "web" / "templates" / "index.html").read_text(encoding="utf-8")

FRAME = (np.arange(256 * 512, dtype=np.uint16) % 4096).reshape(256, 512)
PLOT = np.full((120, 200, 3), 200, dtype=np.uint8)


@pytest.fixture
def store(tmp_path):
    fs = FileStore(root=tmp_path)
    fs.create_session("s1")
    fs.register_embryo("s1", "embryo_1", position_x=0.0, position_y=0.0)
    return fs


class _Agent:
    """As much of the agent as the recorder touches."""

    def __init__(self, store, viz=False):
        self.store = store
        self.session_id = "s1"
        self.viz_server = MagicMock() if viz else None
        self.pushed: list[str] = []
        emb = MagicMock()
        emb.calibration = {"slope_um_per_deg": 97.8, "offset_um": -2.3, "r_squared_top": 0.94}
        self.experiment = MagicMock()
        self.experiment.embryos = {"embryo_1": emb}

    def push_viz(self, array, uid, data_type="image", metadata=None):
        self.pushed.append(uid)


def _run(agent, body):
    """A calibration whose body shows what `body` tells it to."""

    @cal.records_calibration
    async def calibrate_embryo(embryo_id: str, z_buffer_um: float = 25.0, context=None):
        return await body(agent, embryo_id)

    return asyncio.run(calibrate_embryo("embryo_1", z_buffer_um=30.0, context={"agent": agent}))


async def _a_good_run(agent, embryo_id):
    for i, piezo in enumerate((-3.0, 0.0, 3.0)):
        cal._show(
            agent,
            array=FRAME + i,
            uid=f"focus_dense_{embryo_id}_top_{piezo}",
            data_type="focus_sweep",
            metadata={
                "embryo_id": embryo_id,
                "sweep": "dense",
                "galvo_name": "top",
                "galvo": -0.09,
                "piezo": piezo,
                "score": np.float64(1.5e6 + i),
            },
        )
    cal._show(
        agent,
        array=PLOT,
        uid=f"focus_curve_{embryo_id}_top",
        data_type="focus_plot",
        metadata={"embryo_id": embryo_id, "galvo_name": "top", "r_squared": 0.94},
    )
    return "✓ Calibrated embryo_1\n  Slope: 97.82 µm/deg"


def _only_run(store) -> Path:
    (run,) = [p for p in store.calibration_dir("s1", "embryo_1").iterdir() if p.is_dir()]
    return run


# ── the record ───────────────────────────────────────────────────────────


def test_a_run_leaves_its_exposures_and_its_plots_with_the_embryo(store):
    _run(_Agent(store), _a_good_run)
    run = _only_run(store)
    assert run.parent == store._embryo_dir("s1", "embryo_1") / "calibration"
    assert re.fullmatch(r"\d{8}_\d{6}", run.name)
    frames = sorted(p.name for p in (run / "frames").iterdir())
    assert frames == [
        "001_focus_sweep_dense_top_g-0.090_p-3.0.tif",
        "002_focus_sweep_dense_top_g-0.090_p+0.0.tif",
        "003_focus_sweep_dense_top_g-0.090_p+3.0.tif",
    ]
    assert [p.name for p in (run / "plots").iterdir()] == ["004_focus_plot_top.png"]


def test_an_exposure_is_kept_as_it_came_off_the_camera(store):
    _run(_Agent(store), _a_good_run)
    first = _only_run(store) / "frames" / "001_focus_sweep_dense_top_g-0.090_p-3.0.tif"
    on_disk = tifffile.imread(str(first))
    assert on_disk.dtype == np.uint16 and np.array_equal(on_disk, FRAME), "scaled or altered"


def test_every_image_has_a_line_saying_what_it_is(store):
    _run(_Agent(store), _a_good_run)
    lines = [
        json.loads(x)
        for x in (_only_run(store) / "frames.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [x["n"] for x in lines] == [1, 2, 3, 4]
    assert lines[0]["kind"] == "focus_sweep" and lines[0]["piezo"] == -3.0
    assert lines[0]["score"] == 1.5e6 and lines[0]["shape"] == [256, 512]
    assert lines[3]["kind"] == "focus_plot" and lines[3]["file"] == "plots/004_focus_plot_top.png"


def test_the_head_says_what_was_asked_and_what_came_of_it(store):
    _run(_Agent(store), _a_good_run)
    head = yaml.safe_load((_only_run(store) / "calibration.yaml").read_text(encoding="utf-8"))
    assert head["outcome"] == "calibrated" and head["embryo_id"] == "embryo_1"
    assert head["requested"] == {"embryo_id": "embryo_1", "z_buffer_um": 30.0}
    assert "context" not in head["requested"], "the agent was written into the record"
    assert head["calibration"]["slope_um_per_deg"] == 97.8
    assert (head["frames"], head["plots"]) == (3, 1)
    assert head["finished_at"] is not None


def test_nobody_watching_is_no_reason_not_to_keep_it(store):
    agent = _Agent(store, viz=False)
    _run(agent, _a_good_run)
    assert agent.pushed == [] and len(list((_only_run(store) / "frames").iterdir())) == 3


def test_what_is_kept_is_also_shown(store):
    agent = _Agent(store, viz=True)
    _run(agent, _a_good_run)
    assert len(agent.pushed) == 4


@pytest.mark.parametrize(
    ("message", "outcome"),
    [
        ("No object visible at embryo_1", "refused"),
        ("Error calibrating embryo: camera busy", "failed"),
    ],
)
def test_a_run_that_did_not_work_is_kept_too(store, message, outcome):
    async def body(agent, embryo_id):
        cal._show(
            agent,
            array=FRAME,
            uid="probe",
            data_type="presence_check",
            metadata={"embryo_id": embryo_id, "galvo": 0.0, "piezo": 0.0, "visible": False},
        )
        return message

    _run(_Agent(store), body)
    head = yaml.safe_load((_only_run(store) / "calibration.yaml").read_text(encoding="utf-8"))
    assert head["outcome"] == outcome and head["calibration"] is None and head["frames"] == 1


def test_an_aborted_run_is_kept_and_the_abort_still_aborts(store):
    async def body(agent, embryo_id):
        cal._show(
            agent,
            array=FRAME,
            uid="x",
            data_type="focus_sweep",
            metadata={"embryo_id": embryo_id, "piezo": 1.0},
        )
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        _run(_Agent(store), body)
    head = yaml.safe_load((_only_run(store) / "calibration.yaml").read_text(encoding="utf-8"))
    assert head["outcome"] == "aborted" and head["frames"] == 1


def test_the_record_is_closed_when_the_run_ends(store):
    agent = _Agent(store)
    _run(agent, _a_good_run)
    assert agent._calibration_records == {}
    cal._show(
        agent, array=FRAME, uid="late", data_type="focus_sweep", metadata={"embryo_id": "embryo_1"}
    )
    assert len(list((_only_run(store) / "frames").iterdir())) == 3, "filed after the run closed"


def test_two_runs_are_two_folders(store):
    agent = _Agent(store)
    _run(agent, _a_good_run)
    _run(agent, _a_good_run)
    runs = store.list_calibration_records("s1", "embryo_1")
    assert len(runs) == 2 and runs[0]["run"] != runs[1]["run"]


def test_keeping_never_fails_the_calibration(store, monkeypatch):
    monkeypatch.setattr(tifffile, "imwrite", MagicMock(side_effect=OSError("disk full")))
    result = _run(_Agent(store), _a_good_run)
    assert result.startswith("✓ Calibrated")
    head = yaml.safe_load((_only_run(store) / "calibration.yaml").read_text(encoding="utf-8"))
    assert head["outcome"] == "calibrated" and head["frames"] == 0 and head["plots"] == 1


@pytest.mark.parametrize(
    ("policy", "frames", "plots"), [("all", 3, 1), ("plots", 0, 1), ("none", 0, 0)]
)
def test_the_rig_chooses_how_much_is_kept(store, monkeypatch, policy, frames, plots):
    monkeypatch.setattr(rec, "keep_policy", lambda: policy)
    _run(_Agent(store), _a_good_run)
    head = yaml.safe_load((_only_run(store) / "calibration.yaml").read_text(encoding="utf-8"))
    assert (head["kept"], head["frames"], head["plots"]) == (policy, frames, plots)


def test_the_policy_comes_from_the_rigs_settings():
    from gently.settings import settings

    assert settings.storage.calibration_images == "all"
    assert rec.keep_policy() == "all"


# ── the routine ──────────────────────────────────────────────────────────


def test_everything_calibration_shows_goes_through_show():
    body = TOOLS[TOOLS.index("def records_calibration") :]
    assert "agent.push_viz(" not in body, "an image is shown and not kept"
    assert TOOLS.count("_show(") >= 9


def test_a_frame_is_kept_whether_or_not_a_browser_is_watching():
    assert "if agent.viz_server:\n" not in TOOLS
    assert TOOLS.count("_recording(agent, embryo_id)") >= 5


def test_the_tool_is_recorded_and_still_says_what_it_takes():
    import inspect

    assert "@records_calibration\nasync def calibrate_embryo(" in TOOLS
    params = list(inspect.signature(cal.calibrate_embryo).parameters)
    assert params[0] == "embryo_id" and "context" in params and "z_buffer_um" in params


# ── the routes ───────────────────────────────────────────────────────────


@pytest.fixture
def client(store):
    _run(_Agent(store), _a_good_run)
    server = MagicMock()
    server.agent_bridge.agent.store = store
    server.agent_bridge.agent.session_id = "s1"
    app = FastAPI()
    app.include_router(routes.create_router(server))
    app.dependency_overrides[auth.require_control] = lambda: True
    return TestClient(app), _only_run(store)


def test_the_runs_of_an_embryo_are_listed(client):
    c, run = client
    d = c.get("/api/calibration/records", params={"embryo_id": "embryo_1"}).json()
    (r,) = d["records"]
    assert r["run"] == run.name and r["outcome"] == "calibrated" and r["frames"] == 3


def test_a_run_lists_its_images_with_where_to_get_them(client):
    c, run = client
    d = c.get(f"/api/calibration/records/embryo_1/{run.name}").json()
    assert [i["n"] for i in d["images"]] == [1, 2, 3, 4]
    assert d["images"][0]["url"] == f"/api/calibration/records/embryo_1/{run.name}/image/1.png"


def test_an_exposure_and_a_plot_both_come_back_as_png(client):
    c, run = client
    for n in (1, 4):
        r = c.get(
            f"/api/calibration/records/embryo_1/{run.name}/image/{n}.png", params={"max": 128}
        )
        assert r.status_code == 200 and r.headers["content-type"] == "image/png"
        assert r.content[:8] == b"\x89PNG\r\n\x1a\n"


def test_a_run_or_an_image_that_does_not_exist_is_404(client):
    c, run = client
    assert c.get("/api/calibration/records/embryo_1/nope").status_code == 404
    assert c.get(f"/api/calibration/records/embryo_1/{run.name}/image/99.png").status_code == 404
    assert c.get("/api/calibration/records/embryo_1/..%2F..%2Fsession.yaml").status_code == 404


def test_the_folder_opens_in_the_file_manager(client, monkeypatch):
    c, run = client
    opened = []
    monkeypatch.setattr(sessions_routes, "_open_in_file_manager", lambda p: opened.append(p))
    r = c.post(f"/api/calibration/records/embryo_1/{run.name}/open-folder")
    assert r.status_code == 200 and opened == [run]


# ── the pane ─────────────────────────────────────────────────────────────


def test_the_pane_shows_the_latest_runs_plots_and_opens_the_rest():
    assert 'id="op-cal-kept"' in HTML and 'id="op-cal-kept-open"' in HTML
    fn = OPERATE[OPERATE.index("    async function renderCalKept(emb) {") :][:2600]
    assert "/api/calibration/records?embryo_id=" in fn
    assert "if (_selected !== id) return;" in fn, "a slow answer would land on another embryo"
    assert "(i.file || '').startsWith('plots/')" in fn
    assert "Lightbox.open(_calKept.images.map(" in OPERATE
    wire = OPERATE[OPERATE.index("if (_wired) return;") :]
    assert "keptOpen.addEventListener('click', openCalFolder)" in wire
    assert "openCalImage(Number(b.dataset.calImg))" in wire


# ── a deep data folder ───────────────────────────────────────────────────


def test_a_deep_data_folder_does_not_lose_the_images(tmp_path):
    """A run's images sit seven folders under the storage root, and their
    names say what they are. Under a root deep enough, that passes Windows'
    260 characters while the rest of the session still fits, and every write
    failed with "No such file or directory"."""
    deep = tmp_path
    while len(str(deep)) < 150:
        deep = deep / "deeper"
    fs = FileStore(root=deep)
    fs.create_session("s1")
    fs.register_embryo("s1", "embryo_1", position_x=0.0, position_y=0.0)
    _run(_Agent(fs), _a_good_run)
    (r,) = fs.list_calibration_records("s1", "embryo_1")
    longest = len(r["folder"]) + len("/frames/003_focus_sweep_dense_top_g-0.090_p+3.0.tif")
    assert longest > 260, f"not a deep folder: {longest}"
    assert (r["frames"], r["plots"]) == (3, 1), "images were lost to the length of the path"
    assert not r["folder"].startswith(rec._EXTENDED), "the operator is shown the disk's form"


def test_the_long_form_is_for_the_disk_and_the_short_form_for_people():
    import os

    ordinary = Path("C:/data/run") if os.name == "nt" else Path("/data/run")
    assert rec.short_path(rec.long_path(ordinary)) == Path(os.path.abspath(ordinary))
    assert rec.long_path(rec.long_path(ordinary)) == rec.long_path(ordinary)
    if os.name == "nt":
        assert str(rec.long_path(ordinary)).startswith(rec._EXTENDED)
    else:
        assert rec.long_path(ordinary) == ordinary
