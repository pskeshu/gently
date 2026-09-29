"""The launch gate offers the last sessions to carry on from.

"also thinking that resume option should be available when opening gently in
the start screen"

Resuming was a row in the Sessions tab, reached after a new and empty
session had already been opened.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
import yaml
from fastapi import FastAPI
from fastapi.testclient import TestClient

from gently.core.file_store import FileStore
from gently.ui.web import auth
from gently.ui.web.routes import sessions as sessions_routes

WEB = Path(__file__).resolve().parents[1] / "gently" / "ui" / "web"
GATE = (WEB / "templates" / "launch.html").read_text(encoding="utf-8")
ROUTES = (WEB / "routes" / "sessions.py").read_text(encoding="utf-8")


def _session(store, sid, embryos=0, images=0, folder=None):
    store.create_session(sid)
    for n in range(1, embryos + 1):
        store.register_embryo(sid, f"embryo_{n}", position_coarse={"x": 1.0, "y": 2.0}, role="test")
        proj = store._embryo_dir(sid, f"embryo_{n}") / "projections"
        proj.mkdir(parents=True, exist_ok=True)
        for tp in range(images):
            (proj / f"t{tp:04d}.jpg").write_bytes(b"jpg")


def _checkpoint(store, sid, status, complete):
    rows = {eid: {"is_complete": done, "timepoints_acquired": 3} for eid, done in complete.items()}
    (store._session_dir(sid) / "timelapse.yaml").write_text(
        yaml.safe_dump(
            {
                "status": status,
                "saved_at": "2026-09-28T23:49:23",
                "started_at": "2026-09-28T21:52:54",
                "embryos": rows,
            }
        ),
        encoding="utf-8",
    )


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


def _resumable(store, active=None, **params):
    r = _client(store, active).get("/api/sessions/resumable", params=params)
    assert r.status_code == 200, r.text
    return r.json()


class TestTheRoute:
    def test_a_session_with_embryos_is_offered(self, store):
        _session(store, "aaaa1111", embryos=4, images=3)
        got = _resumable(store)["sessions"]
        assert [s["session_id"] for s in got] == ["aaaa1111"]
        assert got[0]["embryo_count"] == 4 and got[0]["timepoints"] == 12
        assert got[0]["last_image_at"] and got[0]["run"] is None

    def test_an_empty_session_is_not(self, store):
        _session(store, "aaaa1111", embryos=2)
        _session(store, "bbbb2222")
        assert [s["session_id"] for s in _resumable(store)["sessions"]] == ["aaaa1111"]

    def test_nothing_to_carry_on_with_is_an_empty_list(self, store):
        _session(store, "bbbb2222")
        assert _resumable(store) == {"sessions": [], "active": None}

    def test_embryos_never_imaged_are_still_something(self, store):
        _session(store, "aaaa1111", embryos=3)
        got = _resumable(store)["sessions"][0]
        assert got["embryo_count"] == 3 and got["timepoints"] == 0
        assert got["last_image_at"] is None

    def test_no_more_than_were_asked_for(self, store, monkeypatch):
        ids = ["aaaa1111", "bbbb2222", "cccc3333", "dddd4444"]
        for sid in ids:
            _session(store, sid, embryos=1)
        monkeypatch.setattr(store, "recent_session_ids", lambda n: list(reversed(ids)))
        got = [s["session_id"] for s in _resumable(store, limit=2)["sessions"]]
        assert got == ["dddd4444", "cccc3333"], "the newest first, and two of them"

    def test_the_limit_is_bounded(self, store, monkeypatch):
        ids = [f"{n:08d}" for n in range(12)]
        for sid in ids:
            _session(store, sid, embryos=1)
        monkeypatch.setattr(store, "recent_session_ids", lambda n: ids)
        assert len(_resumable(store, limit=500)["sessions"]) == 8

    def test_a_run_that_was_going_is_said(self, store):
        _session(store, "aaaa1111", embryos=2, images=3)
        _checkpoint(store, "aaaa1111", "running", {"embryo_1": False, "embryo_2": True})
        run = _resumable(store)["sessions"][0]["run"]
        assert run["status"] == "interrupted" and run["embryos_going"] == 1
        assert run["saved_at"] == "2026-09-28T23:49:23"

    @pytest.mark.parametrize(
        "status,complete",
        [
            ("completed", {"embryo_1": True, "embryo_2": True}),
            ("running", {"embryo_1": True, "embryo_2": True}),
            ("idle", {"embryo_1": False, "embryo_2": False}),
        ],
    )
    def test_a_run_that_ended_is_not(self, store, status, complete):
        _session(store, "aaaa1111", embryos=2, images=1)
        _checkpoint(store, "aaaa1111", status, complete)
        assert _resumable(store)["sessions"][0]["run"] is None

    def test_a_checkpoint_that_cannot_be_read_does_not_hide_the_session(self, store):
        _session(store, "aaaa1111", embryos=2, images=1)
        (store._session_dir("aaaa1111") / "timelapse.yaml").write_text("{{{", encoding="utf-8")
        got = _resumable(store)["sessions"]
        assert got[0]["session_id"] == "aaaa1111" and got[0]["run"] is None

    def test_the_session_open_now_is_marked(self, store):
        _session(store, "aaaa1111", embryos=2)
        got = _resumable(store, active="aaaa1111")
        assert got["active"] == "aaaa1111" and got["sessions"][0]["active"] is True

    def test_resumable_is_not_taken_for_a_session_id(self):
        assert ROUTES.index('"/api/sessions/resumable"') < ROUTES.index(
            '@router.get("/api/sessions/{session_id}")'
        )

    def test_it_opens_no_image(self):
        fn = ROUTES[ROUTES.index("    def _what_a_session_holds(") :]
        fn = fn[: fn.index("    @router.get(")]
        assert "imread" not in fn and "Image.open" not in fn


class TestTheGate:
    def test_a_new_session_is_what_is_chosen(self):
        assert "let resume = null;" in GATE
        assert 'pick("", "A new session"' in GATE

    def test_the_choice_is_not_remembered(self):
        go = GATE[GATE.index('const r = await fetch("/api/launch/go"') :][:400]
        assert "body: JSON.stringify(state)," in go, (
            "the session must not be saved with the toggles"
        )
        assert "resume" not in GATE[GATE.index("const state = {") :][:60]

    def test_nothing_to_carry_on_with_nothing_is_shown(self):
        assert 'id="which" hidden' in GATE
        assert "if (!resumable.length) { box.hidden = true; return; }" in GATE
        assert ".which[hidden]{display:none}" in GATE

    def test_the_session_is_resumed_before_the_microscope_starts(self):
        click = GATE[GATE.index('document.getElementById("go").addEventListener("click"') :]
        resume = click.index('"/resume", { method: "POST" }')
        go = click.index('fetch("/api/launch/go"')
        assert resume < go

    def test_a_resume_that_fails_starts_nothing(self):
        click = GATE[GATE.index('document.getElementById("go").addEventListener("click"') :]
        failed = click[click.index("if (!s.ok) {") : click.index('fetch("/api/launch/go"')]
        assert '"Couldn\'t resume "' in failed and "return;" in failed
        assert "btn.disabled = false;" in failed

    def test_the_button_says_what_it_will_do(self):
        assert '(state.hardware ? "Resume and start " : "Resume ")' in GATE

    def test_a_session_started_with_resume_is_the_one_chosen(self):
        fn = GATE[GATE.index("async function loadResumable()") :][:700]
        assert "const open = resumable.find(s => s.active);" in fn
        assert "if (open) resume = open.session_id;" in fn

    def test_an_interrupted_run_is_said_and_where_to_resume_it(self):
        assert "A run was interrupted" in GATE and "resumed from Acquisition" in GATE

    def test_what_a_session_is_called_is_escaped(self):
        fn = GATE[GATE.index("function paintWhich()") :][:1900]
        assert "esc(s.name)" in fn and "esc(s.session_id)" in fn and "esc(parts.join" in fn

    def test_the_picks_are_a_radio_group(self):
        assert 'role="radiogroup"' in GATE and 'role="radio"' in GATE
        assert 'aria-checked="${String((resume || "") === id)}"' in GATE
