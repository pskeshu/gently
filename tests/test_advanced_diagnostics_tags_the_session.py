"""Advanced diagnostics is called that wherever the operator reads it, and a
session recorded with it on says so.

The switch was called Diagnostics, which read as if nothing were recorded
without it; the recording is always on, at balanced fidelity. And nothing on
disk said whether a replay's placeholder boxes were the balanced fidelity or
a dropped frame. Now the first recording batch that lands while the switch is
on marks the recording's meta.yaml and the session's metadata, once, with
when it started, and the Home and Sessions lists show a chip.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import yaml
from fastapi import FastAPI
from fastapi.testclient import TestClient

from gently.core.file_store import FileStore
from gently.ui.web import auth
from gently.ui.web.routes import replay

WEB = Path(__file__).resolve().parents[1] / "gently" / "ui" / "web"
GATE = (WEB / "templates" / "launch.html").read_text(encoding="utf-8")
INDEX = (WEB / "templates" / "index.html").read_text(encoding="utf-8")
REGISTRY = (WEB / "settings_registry.py").read_text(encoding="utf-8")
SESSIONS_PY = (WEB / "routes" / "sessions.py").read_text(encoding="utf-8")
HOME_JS = (WEB / "static" / "js" / "home.js").read_text(encoding="utf-8")
REVIEW_JS = (WEB / "static" / "js" / "review.js").read_text(encoding="utf-8")

TAB = "6bba95ef"


def test_the_operator_reads_advanced_diagnostics_everywhere() -> None:
    assert '<div class="t" id="dg-t">Advanced diagnostics</div>' in GATE
    assert ">Advanced diagnostics</span>" in INDEX
    assert "started with Advanced diagnostics on" in INDEX
    assert 'label="Start with Advanced diagnostics on"' in REGISTRY
    assert REGISTRY.count('group="Advanced diagnostics"') == 3
    assert 'group="Diagnostics"' not in REGISTRY
    assert 'id="dg-t">Diagnostics<' not in GATE


def _rig(tmp_path: Path, diagnostic: bool, sid: str):
    store = FileStore(root=tmp_path)
    server = SimpleNamespace(
        templates=SimpleNamespace(env=SimpleNamespace(globals={})),
        gently_store=store,
        _current_session_id=lambda: sid,
        diagnostic=diagnostic,
    )
    app = FastAPI()
    app.include_router(replay.create_router(server))
    app.dependency_overrides[auth.require_control] = lambda: True
    replay._capped_tabs.clear()
    replay._seen_tabs.clear()
    replay._tagged.clear()
    return store, server, TestClient(app)


def _send(client: TestClient, n: int = 1):
    batch = {"tab": TAB, "rrweb": [{"type": 3, "data": "x" * 50}] * n, "actions": []}
    return client.post("/replay/ingest", json=batch)


def _meta_yaml(root: Path) -> dict:
    path = next(
        Path(root).rglob("meta.yaml")
    )  # the session's ui-replay dir, or the unassigned bucket
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


class TestTheTag:
    def test_a_recording_with_it_on_marks_the_session_and_the_recording(self, tmp_path):
        store, _server, client = _rig(tmp_path, True, "s1")
        store.create_session("s1", name="one")
        assert _send(client).status_code == 200
        meta = store.get_session("s1")["metadata"]
        assert meta["advanced_diagnostics"] is True
        assert meta["advanced_diagnostics_since"]
        rec = _meta_yaml(tmp_path)
        assert rec["advanced_diagnostics"] is True
        assert rec["advanced_diagnostics_since"] == meta["advanced_diagnostics_since"]

    def test_the_mark_is_set_once_and_never_cleared(self, tmp_path):
        store, server, client = _rig(tmp_path, True, "s1")
        store.create_session("s1")
        _send(client)
        since = store.get_session("s1")["metadata"]["advanced_diagnostics_since"]
        _send(client)
        replay.apply_diagnostic(server, False)
        _send(client)
        meta = store.get_session("s1")["metadata"]
        assert meta["advanced_diagnostics"] is True
        assert meta["advanced_diagnostics_since"] == since, (
            "the mark moved; it means 'at least from here'"
        )

    def test_an_ordinary_night_leaves_no_mark(self, tmp_path):
        store, _server, client = _rig(tmp_path, False, "s2")
        store.create_session("s2")
        assert _send(client).status_code == 200
        assert "advanced_diagnostics" not in (store.get_session("s2").get("metadata") or {})
        assert "advanced_diagnostics" not in _meta_yaml(tmp_path)

    def test_a_recording_with_no_session_still_marks_itself(self, tmp_path):
        _store, _server, client = _rig(tmp_path, True, "")
        assert _send(client).status_code == 200
        assert _meta_yaml(tmp_path)["advanced_diagnostics"] is True

    def test_the_lists_carry_and_show_it(self) -> None:
        assert '"advanced_diagnostics": bool(' in SESSIONS_PY
        assert "s.advanced_diagnostics" in HOME_JS and "home-tag-diag" in HOME_JS
        assert "s.advanced_diagnostics" in REVIEW_JS and "session-diag-badge" in REVIEW_JS
