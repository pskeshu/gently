"""A project pools its measurements in a table.

"a pooled data table is also good. all the measurements from a project."
Data is a project tool beside Figures and Reading: every session tool that
can be pooled answers with one row per embryo, each row saying its session
and the session's role in this project; and the pool route gives the same
rows as CSV, for a spreadsheet or a script.
"""

from __future__ import annotations

import csv
import io
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from gently.core.file_store import FileStore, _write_yaml
from gently.harness.memory.file_store import FileContextStore
from gently.tools import TOOLS, tools_of
from gently.tools.pool import create_router, pool, pool_csv

CAMPAIGNS_JS = (
    Path(__file__).resolve().parents[1] / "gently" / "ui" / "web" / "static" / "js" / "campaigns.js"
).read_text(encoding="utf-8")


def _project(tmp_path):
    store = FileStore(tmp_path)
    cs = FileContextStore(tmp_path / "agent")
    proj = cs.create_campaign("time to hatching", shorthand="t2h")
    for sid, role in (("s1", "control"), ("s2", "test")):
        store.create_session(sid, name=sid)
        store.link_session(sid, proj, role=role, home=True)
        for n in (1, 2, 3):
            meta = {"frame": n, "field": 0, "captured_at": f"2026-10-27T0{n}:00:00"}
            store.put_snapshot(sid, "bf", [[0]], metadata=meta, stem=f"f{n}")
    marks = store._session_dir("s1") / "annotations" / "twitch_hatch.yaml"
    _write_yaml(
        marks,
        [
            {"field": 0, "x": 0.1, "y": 0.1, "twitch_frame": 1, "hatch_frame": 2, "by": "ryan"},
            {"field": 0, "x": 0.5, "y": 0.5, "twitch_frame": 1, "hatch_frame": None},
        ],
    )
    _write_yaml(
        store._session_dir("s2") / "annotations" / "twitch_hatch.yaml",
        [{"field": 0, "x": 0.2, "y": 0.2, "twitch_frame": 1, "hatch_frame": 3}],
    )
    return store, cs, proj


def _client(store, cs):
    class _Server:
        context_store = cs
        gently_store = store
        agent_bridge = None

    app = FastAPI()
    app.include_router(create_router(_Server()))
    return TestClient(app)


def test_data_is_a_project_tool_beside_figures_and_reading():
    assert TOOLS["data"]["scope"] == "project"
    assert "data" in tools_of("project")


def test_the_pool_answers_as_csv_with_the_session_and_its_role_first(tmp_path):
    store, cs, proj = _project(tmp_path)
    text = pool_csv(pool(store, cs, "twitch_hatch", project="t2h"))
    header = text.splitlines()[0].split(",")
    assert header[:10] == [
        "session_id",
        "session_name",
        "role",
        "condition",
        "n",
        "field",
        "x",
        "y",
        "twitch_frame",
        "hatch_frame",
    ]
    assert {"minutes", "temp_c", "vs_sulston_min", "by"} <= set(header)
    rows = {(r["session_id"], r["n"]): r for r in csv.DictReader(io.StringIO(text))}
    assert len(rows) == 3
    assert rows[("s1", "1")]["role"] == "control" and rows[("s1", "1")]["minutes"] == "60.0"
    assert rows[("s1", "1")]["by"] == "ryan" and rows[("s1", "1")]["condition"] == "t2h"
    assert rows[("s1", "2")]["hatch_frame"] == "" and rows[("s1", "2")]["minutes"] == ""
    assert rows[("s2", "1")]["role"] == "test" and rows[("s2", "1")]["minutes"] == "120.0"

    client = _client(store, cs)
    r = client.get("/api/tools/twitch_hatch/pool", params={"project": "t2h", "format": "csv"})
    assert r.status_code == 200 and r.headers["content-type"].startswith("text/csv")
    assert r.headers["content-disposition"] == 'attachment; filename="t2h-twitch_hatch.csv"'
    assert r.text == text
    # without the format, the pool as before
    assert (
        len(client.get("/api/tools/twitch_hatch/pool", params={"project": "t2h"}).json()["rows"])
        == 3
    )
    assert client.get("/api/tools/nothing/pool", params={"project": "t2h"}).status_code == 404


def test_the_page_mounts_data_like_the_other_project_tools():
    assert "data: DataTool" in CAMPAIGNS_JS
    assert "`/api/tools/${enc(t.name)}/pool?project=${enc(pid)}`" in CAMPAIGNS_JS
    assert "&format=csv" in CAMPAIGNS_JS
    # a row names its session and role, and opens the session
    for needle in ("<th>Session</th><th>Role</th>", "data-sgo=", "goToSession(tr.dataset.sgo)"):
        assert needle in CAMPAIGNS_JS, needle
    # and opening it reaches the Sessions tab by its global name: ReviewApp is a
    # top-level const there, never a window property, so window.ReviewApp was
    # always undefined and a row never opened
    assert "window.ReviewApp" not in CAMPAIGNS_JS
    assert "if (typeof ReviewApp !== 'undefined') ReviewApp.loadSession(sid)" in CAMPAIGNS_JS
    # the project's own sessions table dropped its signal counts
    assert "<th>Signals</th>" not in CAMPAIGNS_JS
