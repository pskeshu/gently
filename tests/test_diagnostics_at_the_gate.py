"""A run that needs working out can be recorded in full.

"rrweb - we need to take a look at that - perhaps it should be configurable
on its limits? ... replay: rrweb cap (120 MB) hit for tab 6bba95ef — dropping
further frames ... some runs might need that more limit than others ... we
need a diagnostic mode in the startup screen toggle."

The recording stopped partway through the run it was wanted for. Its limits
could be changed, in Settings, with a restart. Which limits a run needs is
known when the run is started, which is at the gate.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from gently import log_config
from gently.core.file_store import FileStore
from gently.ui.web import auth, launch_prefs
from gently.ui.web.routes import replay

WEB = Path(__file__).resolve().parents[1] / "gently" / "ui" / "web"
GATE = (WEB / "templates" / "launch.html").read_text(encoding="utf-8")
INDEX = (WEB / "templates" / "index.html").read_text(encoding="utf-8")
APP_JS = (WEB / "static" / "js" / "app.js").read_text(encoding="utf-8")
REPLAY_SRC = (WEB / "routes" / "replay.py").read_text(encoding="utf-8")

TAB = "6bba95ef"


def _server(diagnostic=None, store=None):
    server = SimpleNamespace(
        templates=SimpleNamespace(env=SimpleNamespace(globals={})),
        gently_store=store,
        _current_session_id=lambda: None,
    )
    if diagnostic is not None:
        server.diagnostic = diagnostic
    return server


class TestTheLimits:
    def test_an_ordinary_night(self):
        got = replay.limits(_server(diagnostic=False))
        assert got == {
            "diagnostic": False,
            "fidelity": "balanced",
            "tab_mb": 120.0,
            "budget_mb": 1024.0,
        }

    def test_with_diagnostics(self):
        got = replay.limits(_server(diagnostic=True))
        assert got["diagnostic"] is True and got["fidelity"] == "full"
        assert got["tab_mb"] == 1024.0 and got["budget_mb"] == 8192.0

    def test_diagnostics_never_lowers_a_limit(self, monkeypatch):
        ui = SimpleNamespace(
            replay=True,
            replay_fidelity="balanced",
            replay_max_tab_mb=4000.0,
            replay_total_budget_mb=50000.0,
            replay_diagnostic_tab_mb=1024.0,
            replay_diagnostic_budget_mb=8192.0,
        )
        monkeypatch.setattr(replay, "settings", SimpleNamespace(ui=ui))
        got = replay.limits(_server(diagnostic=True))
        assert got["tab_mb"] == 4000.0 and got["budget_mb"] == 50000.0

    def test_before_the_gate_it_is_what_was_chosen_last(self, monkeypatch):
        monkeypatch.setattr(launch_prefs, "load_prefs", lambda: {"diagnostic": True})
        assert replay.limits(_server())["diagnostic"] is True
        monkeypatch.setattr(launch_prefs, "load_prefs", lambda: {})
        assert replay.limits(_server())["diagnostic"] is False

    def test_they_are_read_each_time_not_when_gently_was_imported(self):
        server = _server(diagnostic=False)
        assert replay.limits(server)["tab_mb"] == 120.0
        replay.apply_diagnostic(server, True)
        assert replay.limits(server)["tab_mb"] == 1024.0


class TestTurningItOn:
    def test_pages_loaded_from_now_record_in_full(self):
        server = _server(diagnostic=False)
        replay.apply_diagnostic(server, True)
        assert server.templates.env.globals["replay_fidelity"] == "full"
        replay.apply_diagnostic(server, False)
        assert server.templates.env.globals["replay_fidelity"] == "balanced"

    def test_a_tab_that_was_capped_is_recorded_again(self):
        replay._capped_tabs.add(("somewhere", TAB))
        replay.apply_diagnostic(_server(), True)
        assert not replay._capped_tabs

    def test_the_log_files_are_told_everything(self, tmp_path):
        lgr = logging.getLogger("gently.test_diagnostics")
        handler = logging.FileHandler(tmp_path / "x.log", encoding="utf-8")
        handler.setLevel(logging.INFO)
        lgr.addHandler(handler)
        try:
            assert log_config.set_file_detail(True) >= 1
            assert handler.level == logging.DEBUG
            log_config.set_file_detail(False)
            assert handler.level == logging.INFO
        finally:
            lgr.removeHandler(handler)
            handler.close()

    def test_the_console_is_left_as_it_was(self):
        lgr = logging.getLogger("gently.test_diagnostics_console")
        handler = logging.StreamHandler()
        handler.setLevel(logging.WARNING)
        lgr.addHandler(handler)
        try:
            log_config.set_file_detail(True)
            assert handler.level == logging.WARNING
        finally:
            lgr.removeHandler(handler)
            log_config.set_file_detail(False)

    def test_the_gate_applies_it(self):
        src = (WEB / "routes" / "device_layer.py").read_text(encoding="utf-8")
        go = src[src.index("    async def launch_go(") :][:1800]
        assert 'apply_diagnostic(server, bool(prefs.get("diagnostic")))' in go

    def test_it_is_remembered_with_the_other_two(self, tmp_path, monkeypatch):
        monkeypatch.setattr(launch_prefs, "PREFS_PATH", tmp_path / "launch.local.json")
        monkeypatch.setattr(launch_prefs, "_CONFIG_DIR", tmp_path)
        monkeypatch.setattr(launch_prefs, "detect_sam_device", lambda: "cpu")
        assert launch_prefs.load_prefs()["diagnostic"] is False
        launch_prefs.save_prefs({"diagnostic": True})
        assert launch_prefs.load_prefs()["diagnostic"] is True
        assert json.loads((tmp_path / "launch.local.json").read_text("utf-8"))["diagnostic"] is True


class TestTheRecording:
    @pytest.fixture
    def rig(self, tmp_path):
        store = FileStore(root=tmp_path)
        server = _server(diagnostic=False, store=store)
        app = FastAPI()
        app.include_router(replay.create_router(server))
        app.dependency_overrides[auth.require_control] = lambda: True
        replay._capped_tabs.clear()
        return server, TestClient(app), tmp_path

    def _send(self, client, n=1):
        batch = {"tab": TAB, "rrweb": [{"type": 3, "data": "x" * 200}] * n, "actions": []}
        return client.post("/replay/ingest", json=batch)

    def _file(self, root):
        found = list(Path(root).rglob(f"rrweb-{TAB}.jsonl"))
        assert found, "nothing was recorded"
        return found[0]

    def test_past_the_cap_frames_are_dropped(self, rig, monkeypatch):
        server, client, root = rig
        monkeypatch.setattr(
            replay,
            "limits",
            lambda s: {
                "diagnostic": False,
                "fidelity": "balanced",
                "tab_mb": 0.001,
                "budget_mb": 10,
            },
        )
        assert self._send(client, 10).status_code == 200
        size = self._file(root).stat().st_size
        assert size > 1024
        self._send(client, 10)
        assert self._file(root).stat().st_size == size

    def test_with_diagnostics_the_same_tab_goes_on(self, rig, monkeypatch):
        server, client, root = rig
        held = {"diagnostic": False, "fidelity": "balanced", "tab_mb": 0.001, "budget_mb": 10}
        monkeypatch.setattr(replay, "limits", lambda s: held)
        self._send(client, 10)
        self._send(client, 10)
        size = self._file(root).stat().st_size
        held.update(diagnostic=True, tab_mb=1024.0)
        replay._capped_tabs.clear()
        self._send(client, 10)
        assert self._file(root).stat().st_size > size

    def test_the_warning_says_what_to_do(self, rig, monkeypatch, caplog):
        server, client, root = rig
        monkeypatch.setattr(
            replay,
            "limits",
            lambda s: {
                "diagnostic": False,
                "fidelity": "balanced",
                "tab_mb": 0.001,
                "budget_mb": 10,
            },
        )
        self._send(client, 10)
        with caplog.at_level(logging.WARNING, logger=replay.logger.name):
            self._send(client, 10)
        said = " ".join(r.getMessage() for r in caplog.records)
        assert "dropping further frames" in said
        assert "Start Gently with Advanced diagnostics on" in said

    def test_the_page_can_ask_what_it_is_held_to(self, rig):
        server, client, _ = rig
        got = client.get("/replay/limits").json()
        assert got["diagnostic"] is False and got["tab_mb"] == 120.0
        replay.apply_diagnostic(server, True)
        assert client.get("/replay/limits").json()["diagnostic"] is True


class TestTheGate:
    def test_there_is_a_third_switch(self):
        assert 'id="opt-diagnostic" data-key="diagnostic" role="switch"' in GATE
        assert 'for (const id of ["opt-hardware", "opt-agent", "opt-diagnostic"]) {' in GATE

    def test_it_is_off_until_somebody_turns_it_on(self):
        assert "const state = { hardware: true, agent: true, diagnostic: false };" in GATE

    def test_it_says_what_it_costs(self):
        assert "Uses more disk." in GATE

    def test_it_goes_with_the_other_two(self):
        go = GATE[GATE.index('const r = await fetch("/api/launch/go"') :][:300]
        assert "body: JSON.stringify(state)," in go

    def test_it_is_under_the_two_that_are_asked_every_day(self):
        assert (
            GATE.index('id="opt-agent"')
            < GATE.index('id="opt-diagnostic"')
            < GATE.index('id="which"')
        )


class TestTheWorkspace:
    def test_it_says_when_diagnostics_is_on(self):
        assert 'id="diag-badge" hidden' in INDEX
        assert "if (badge) badge.hidden = !(d && d.diagnostic);" in APP_JS

    def test_the_settings_have_it(self):
        from gently.ui.web.settings_registry import SETTINGS

        by_key = {s.key: s for s in SETTINGS}
        assert by_key["recording.diagnostic"].store == "launch:diagnostic"
        assert by_key["recording.diagnosticTabMb"].source == "ui.replay_diagnostic_tab_mb"
        assert by_key["recording.diagnosticBudgetMb"].source == "ui.replay_diagnostic_budget_mb"

    def test_nothing_reads_the_ordinary_cap_past_the_limits(self):
        body = REPLAY_SRC[REPLAY_SRC.index("def create_router(") :]
        assert "settings.ui.replay_max_tab_mb" not in body
        assert (
            "settings.ui.replay_total_budget_mb"
            not in REPLAY_SRC[
                REPLAY_SRC.index("def _prune_recordings(") : REPLAY_SRC.index("def limits(")
            ]
        )
