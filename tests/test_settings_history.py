"""Every change to a setting is kept on disk.

"i want the full history of setting changes to remain on disk"

One append-only file under the storage root. The server writes a line when a
rig setting changes; a browser reports a change to one of its own. A secret is
never written, and a history that cannot be written never stops the change.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from gently.core import settings_history as history
from gently.settings import settings
from gently.ui.web import auth
from gently.ui.web.routes import data as data_routes

JS = Path(__file__).resolve().parents[1] / "gently" / "ui" / "web" / "static" / "js"
SETTINGS_JS = (JS / "settings.js").read_text(encoding="utf-8")


def _lines() -> list[dict]:
    p = history.path()
    if not p.exists():
        return []
    return [json.loads(x) for x in p.read_text(encoding="utf-8").splitlines() if x.strip()]


# ── the file ─────────────────────────────────────────────────────────────


def test_it_lives_under_the_storage_root_with_the_rigs_other_config():
    assert history.path() == Path(settings.storage.base_path) / "config" / "settings_history.jsonl"


def test_a_change_is_one_line_saying_what_from_what_to_what_and_who():
    e = history.record(
        "recording.fidelity",
        "balanced",
        "actions",
        reach="rig",
        via="Settings",
        by="ryan",
        client="127.0.0.1",
        session_id="s1",
        label="Detail",
    )
    (line,) = _lines()
    assert line == e
    assert (line["key"], line["old"], line["new"]) == ("recording.fidelity", "balanced", "actions")
    assert (line["by"], line["client"], line["session_id"]) == ("ryan", "127.0.0.1", "s1")
    assert line["reach"] == "rig" and line["label"] == "Detail"
    assert line["at"][:4].isdigit() and ("+" in line["at"] or "-" in line["at"][10:]), (
        "a time with no offset means a different moment to every reader"
    )


def test_the_history_only_grows():
    for i in range(5):
        history.record("views.film.skipInterval", i, i + 1, reach="browser")
    before = history.path().read_text(encoding="utf-8")
    history.record("views.film.skipInterval", 5, 6, reach="browser")
    after = history.path().read_text(encoding="utf-8")
    assert after.startswith(before), "an earlier line was rewritten"
    assert history.count() == 6


def test_setting_something_to_what_it_already_is_is_not_a_change():
    assert history.record("views.theme", "dark", "dark") is None
    assert _lines() == []


def test_newest_first_and_one_setting_at_a_time():
    history.record("a.one", 1, 2)
    history.record("b.two", 1, 2)
    history.record("a.one", 2, 3)
    assert [e["new"] for e in history.read()] == [3, 2, 2]
    assert [e["new"] for e in history.read(key="a.one")] == [3, 2]
    assert len(history.read(limit=1)) == 1


def test_a_secret_is_never_written():
    history.record_diff(
        "microscope.thermalizer",
        {"broker": "a.example", "password": "hunter2", "user": "lab"},
        {"broker": "b.example", "password": "correct horse", "user": "lab"},
    )
    text = history.path().read_text(encoding="utf-8")
    assert "hunter2" not in text and "correct horse" not in text
    by = {e["key"]: e for e in _lines()}
    assert by["microscope.thermalizer.broker"]["new"] == "b.example"
    assert by["microscope.thermalizer.password"]["new"] == "(not recorded)"
    assert "microscope.thermalizer.user" not in by, "it did not change"


def test_a_history_that_cannot_be_written_does_not_stop_the_change(monkeypatch, caplog):
    def boom(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr("builtins.open", boom)
    assert history.record("views.theme", "light", "dark") is None


def test_a_damaged_line_is_skipped_not_fatal():
    history.record("a.one", 1, 2)
    with open(history.path(), "a", encoding="utf-8") as f:
        f.write("{not json\n")
    history.record("a.one", 2, 3)
    assert [e["new"] for e in history.read()] == [3, 2]


# ── the routes ───────────────────────────────────────────────────────────


@pytest.fixture
def rig(tmp_path, monkeypatch):
    monkeypatch.setattr(data_routes, "_HARDWARE_CONFIG_PATH", tmp_path / "hardware.yaml")
    server = MagicMock()
    agent = server.agent_bridge.agent
    agent.session_id = "s1"
    client = agent.client
    client.is_connected = True
    client.get_joystick = AsyncMock(return_value={"success": True, "enabled": True})
    client.set_joystick = AsyncMock(return_value={"success": True, "enabled": False})
    client.get_temperature_config = AsyncMock(
        return_value={"config": {"backend": "serial", "com_port": "COM8", "baud_rate": 115200}}
    )
    client.set_temperature_config = AsyncMock(return_value={"success": True, "applied": True})
    client.get_stage_envelope = AsyncMock(
        return_value={"envelope": {"x_min": -1, "x_max": 1, "y_min": -1, "y_max": 1}}
    )
    client.set_stage_envelope = AsyncMock(return_value={"success": True})
    client.set_envelope_enforced = AsyncMock(return_value={"success": True})
    app = FastAPI()
    app.include_router(data_routes.create_router(server))
    app.dependency_overrides[auth.require_control] = lambda: True
    return TestClient(app)


def test_a_restart_setting_is_recorded_with_what_it_was(rig):
    rig.put("/api/config/settings-overrides", json={"GENTLY_REPLAY_FIDELITY": "actions"})
    rig.put("/api/config/settings-overrides", json={"GENTLY_REPLAY_FIDELITY": "full"})
    first, second = _lines()
    assert (first["key"], first["old"], first["new"]) == (
        "recording.fidelity",
        "balanced",
        "actions",
    )
    assert (second["old"], second["new"]) == ("actions", "full"), "was: what was on file"
    assert first["label"] == "Detail" and first["session_id"] == "s1" and first["reach"] == "rig"


def test_a_refused_setting_leaves_no_line(rig):
    assert rig.put("/api/config/settings-overrides", json={"GENTLY_VIZ_PORT": 1}).status_code == 400
    assert _lines() == []


def test_the_rigs_defaults_are_recorded_leaf_by_leaf(rig):
    rig.put("/api/config/dashboard-defaults", json={"filmstrip": {"thumbnailSize": 72}})
    rig.put(
        "/api/config/dashboard-defaults",
        json={"filmstrip": {"thumbnailSize": 40}, "defaultView": "board"},
    )
    assert [(e["key"], e["old"], e["new"]) for e in _lines()] == [
        ("rig-defaults.filmstrip.thumbnailSize", None, 72),
        ("rig-defaults.defaultView", None, "board"),
        ("rig-defaults.filmstrip.thumbnailSize", 72, 40),
    ]


def test_the_joystick_lock_is_recorded_as_the_controller_reads_back(rig):
    rig.post("/api/devices/stage/joystick", json={"enabled": False})
    (line,) = _lines()
    assert (line["key"], line["old"], line["new"]) == ("microscope.joystick.enabled", True, False)


def test_the_thermalizer_is_recorded_without_its_password(rig):
    rig.post(
        "/api/devices/temperature/config",
        json={"backend": "mqtt", "broker": "b.example", "password": "hunter2"},
    )
    text = history.path().read_text(encoding="utf-8")
    assert "hunter2" not in text
    by = {e["key"]: e for e in _lines()}
    assert (
        by["microscope.thermalizer.backend"]["old"],
        by["microscope.thermalizer.backend"]["new"],
    ) == (
        "serial",
        "mqtt",
    )


def test_the_region_and_the_limits_are_recorded(rig):
    rig.post("/api/devices/stage/envelope", json={"x_min": -5, "x_max": 5, "y_min": -1, "y_max": 1})
    rig.post("/api/devices/stage/envelope/enforced", json={"enforced": False})
    keys = [e["key"] for e in _lines()]
    assert keys == [
        "microscope.xyRegion.x_max",
        "microscope.xyRegion.x_min",
        "microscope.xyLimits.enforced",
    ]


def test_a_browser_reports_a_change_to_its_own_preference(rig):
    r = rig.post(
        "/api/settings/history",
        json={"key": "views.film.thumbnailSize", "old": 56, "new": 72, "client_id": "abc123"},
    )
    assert r.json() == {"recorded": True}
    (line,) = _lines()
    assert line["reach"] == "browser" and line["label"] == "Frame size"
    assert line["client"].endswith("abc123")


def test_a_browser_cannot_report_a_rig_setting_or_an_invented_one(rig):
    assert (
        rig.post(
            "/api/settings/history", json={"key": "recording.enabled", "new": False}
        ).status_code
        == 400
    )
    assert rig.post("/api/settings/history", json={"key": "made.up", "new": 1}).status_code == 400
    assert _lines() == []


def test_reading_the_history_needs_no_control_and_says_where_it_is():
    app = FastAPI()
    app.include_router(data_routes.create_router(MagicMock()))
    history.record("a.one", 1, 2)
    d = TestClient(app).get("/api/settings/history").json()
    assert d["total"] == 1 and d["changes"][0]["key"] == "a.one"
    assert d["file"].endswith("settings_history.jsonl")


# ── the tab ──────────────────────────────────────────────────────────────


def test_the_tab_reports_before_it_writes_so_the_old_value_is_the_old_value():
    fn = SETTINGS_JS[SETTINGS_JS.index("    async function writeValue(s, value) {") :][:500]
    report = fn.index("report(s.key, readValue(s), value);")
    assert report < fn.index("if (store === 'theme') {")


def test_a_reset_and_an_import_are_changes_too():
    assert "report('views.reset', SettingsStore.local(), {});" in SETTINGS_JS
    assert "report('views.import', SettingsStore.local(), obj);" in SETTINGS_JS


def test_the_tab_shows_the_history():
    assert "fetch('/api/settings/history?limit=200')" in SETTINGS_JS
    assert "SettingsHistory.load();" in SETTINGS_JS
