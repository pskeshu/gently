"""One way to a file: the file manager, and Fiji.

"i am curious where else we need to put the open folder button" … "go ahead
with the folder things. and i am curious if we can also have a open in fiji
button or something for images"

No test here opens a window: the two functions that would are replaced.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from gently.core import reveal as os_reveal
from gently.core.file_store import FileStore
from gently.ui.web import auth
from gently.ui.web.routes import reveal as reveal_routes

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "gently" / "ui" / "web"
JS = WEB / "static" / "js"
REVEAL_JS = (JS / "reveal.js").read_text(encoding="utf-8")
INDEX = (WEB / "templates" / "index.html").read_text(encoding="utf-8")
CSS = (WEB / "static" / "css" / "reveal.css").read_text(encoding="utf-8")
ROUTE_SRC = (WEB / "routes" / "reveal.py").read_text(encoding="utf-8")


# ── fixtures ─────────────────────────────────────────────────────────────


@pytest.fixture
def store(tmp_path):
    fs = FileStore(root=tmp_path / "data")
    fs.create_session("s1")
    fs.create_session("s2")
    fs.register_embryo("s1", "embryo_1", position_coarse={"x": 1.0, "y": 2.0}, role="test")
    fs.register_embryo("s1", "embryo_2", position_coarse={"x": 3.0, "y": 4.0}, role="test")
    fs.register_embryo("s2", "embryo_1", position_coarse={"x": 5.0, "y": 6.0}, role="test")
    fs.put_volume("s1", "embryo_1", 3, np.zeros((2, 4, 4), dtype=np.uint16))
    fs.put_volume("s2", "embryo_1", 0, np.zeros((2, 4, 4), dtype=np.uint16))
    (fs.root / "logs").mkdir(parents=True, exist_ok=True)
    return fs


@pytest.fixture
def calls(monkeypatch):
    """What would have been opened. Nothing is."""
    seen: dict[str, list] = {"show": [], "fiji": []}

    def show(path):
        seen["show"].append(Path(path))
        return "folder" if Path(path).is_dir() else "file"

    monkeypatch.setattr(os_reveal, "show", show)
    monkeypatch.setattr(
        os_reveal, "open_in_fiji", lambda path, fiji: seen["fiji"].append((Path(path), Path(fiji)))
    )
    monkeypatch.setattr(os_reveal, "is_local", lambda host: True)
    return seen


@pytest.fixture
def fiji(tmp_path, monkeypatch):
    exe = tmp_path / "Fiji" / "fiji-windows-x64.exe"
    exe.parent.mkdir()
    exe.write_bytes(b"")
    monkeypatch.setattr(reveal_routes, "fiji_path", lambda: exe)
    return exe


def _client(store, session="s1", control=True):
    server = MagicMock()
    server.agent_bridge.agent.store = store
    server.agent_bridge.agent.session_id = session
    server.gently_store = store
    app = FastAPI()
    app.include_router(reveal_routes.create_router(server))
    if control:
        app.dependency_overrides[auth.require_control] = lambda: True
    return TestClient(app)


def _post(client, **body):
    return client.post("/api/reveal", json=body)


# ── what is where ────────────────────────────────────────────────────────


class TestWhatIsWhere:
    def test_the_session_in_hand(self, store, calls):
        r = _post(_client(store), what="session")
        assert r.status_code == 200, r.text
        assert calls["show"] == [store._session_dir("s1")]
        assert r.json()["opened"] is True and r.json()["kind"] == "folder"

    def test_another_session_by_its_id(self, store, calls):
        _post(_client(store), what="session", session_id="s2")
        assert calls["show"] == [store._session_dir("s2")]

    def test_an_embryo(self, store, calls):
        _post(_client(store), what="embryo", embryo_id="embryo_2")
        assert calls["show"] == [store._embryo_dir("s1", "embryo_2")]

    def test_a_timepoint_is_its_volume(self, store, calls):
        r = _post(_client(store), what="timepoint", embryo_id="embryo_1", timepoint=3)
        assert r.status_code == 200, r.text
        assert calls["show"] == [store.get_volume_path("s1", "embryo_1", 3)]
        assert r.json()["kind"] == "file"

    def test_a_timepoint_of_another_session(self, store, calls):
        _post(_client(store), what="timepoint", session_id="s2", embryo_id="embryo_1", timepoint=0)
        assert calls["show"] == [store.get_volume_path("s2", "embryo_1", 0)]

    def test_a_timepoint_without_its_volume_is_its_projection(self, store, calls):
        proj = store._embryo_dir("s1", "embryo_2") / "projections" / "t0007.jpg"
        proj.parent.mkdir(parents=True, exist_ok=True)
        proj.write_bytes(b"jpg")
        _post(_client(store), what="timepoint", embryo_id="embryo_2", timepoint=7)
        assert calls["show"] == [proj]

    def test_a_volume_is_never_its_projection(self, store, calls):
        proj = store._embryo_dir("s1", "embryo_2") / "projections" / "t0007.jpg"
        proj.parent.mkdir(parents=True, exist_ok=True)
        proj.write_bytes(b"jpg")
        r = _post(_client(store), what="volume", embryo_id="embryo_2", timepoint=7)
        assert r.status_code == 404 and calls["show"] == []

    def test_a_dic_frame_by_its_stem(self, store, calls):
        store.put_snapshot("s1", "dic", np.zeros((4, 4), dtype=np.uint16), {"frame": 1})
        rec = store.list_snapshots("s1", "dic")[0]
        stem = Path(rec["file_path"]).stem
        r = _post(_client(store), what="dic", stem=stem)
        assert r.status_code == 200, r.text
        assert calls["show"] == [Path(rec["file_path"])]

    def test_the_logs_and_the_data_folder(self, store, calls):
        client = _client(store)
        _post(client, what="logs")
        _post(client, what="storage")
        assert calls["show"] == [store.root / "logs", store.root]

    def test_the_removed_embryos(self, store, calls):
        store.set_aside_embryo("s1", "embryo_2")
        _post(_client(store), what="removed")
        assert calls["show"] == [store._session_dir("s1") / "removed"]

    def test_a_calibration_run_and_its_image(self, store, calls):
        rec = store.open_calibration_record("s1", "embryo_1", {})
        rec.add(np.zeros((4, 4), dtype=np.uint16), kind="focus_sweep", metadata={})
        rec.finish("calibrated")
        run = store.list_calibration_records("s1", "embryo_1")[0]["run"]
        client = _client(store)
        r = _post(client, what="calibration_run", embryo_id="embryo_1", run=run)
        assert r.status_code == 200, r.text
        r = _post(client, what="calibration_image", embryo_id="embryo_1", run=run, n=1)
        assert r.status_code == 200, r.text
        assert calls["show"][0].name == run
        assert calls["show"][1].suffix == ".tif" and calls["show"][1].is_file()


class TestWhatIsNot:
    @pytest.mark.parametrize(
        "body",
        [
            {"what": "session", "session_id": "nope"},
            {"what": "embryo", "embryo_id": "embryo_9"},
            {"what": "timepoint", "embryo_id": "embryo_1", "timepoint": 99},
            {"what": "dic", "stem": "nope"},
            {"what": "calibration_run", "embryo_id": "embryo_1", "run": "nope"},
            {"what": "removed"},
        ],
    )
    def test_what_is_not_on_disk_is_not_found(self, store, calls, body):
        r = _post(_client(store), **body)
        assert r.status_code == 404, r.text
        assert calls["show"] == []

    @pytest.mark.parametrize(
        "body",
        [
            {"what": "anything"},
            {"what": "embryo"},
            {"what": "timepoint", "embryo_id": "embryo_1"},
            {"what": "session", "action": "delete"},
        ],
    )
    def test_a_request_that_does_not_say_what(self, store, calls, body):
        assert _post(_client(store), **body).status_code == 422
        assert calls["show"] == []

    @pytest.mark.parametrize(
        "embryo", ["..", "../..", "..\\..\\Windows", "embryo_1/../embryo_2", "C:\\Windows", "a/b"]
    )
    def test_an_id_is_not_a_path(self, store, calls, embryo):
        r = _post(_client(store), what="embryo", embryo_id=embryo)
        assert r.status_code in (404, 422), r.text
        assert calls["show"] == []

    @pytest.mark.parametrize("run", ["..", "../../..", "..\\..", ""])
    def test_a_run_is_matched_not_joined(self, store, calls, run):
        r = _post(_client(store), what="calibration_run", embryo_id="embryo_1", run=run)
        assert r.status_code == 404
        assert calls["show"] == []

    def test_no_request_field_is_a_path(self):
        fields = re.findall(
            r"^    (\w+): ", ROUTE_SRC[ROUTE_SRC.index("class RevealRequest") :][:400], re.M
        )
        assert fields == [
            "what",
            "action",
            "session_id",
            "embryo_id",
            "timepoint",
            "run",
            "n",
            "stem",
        ]
        assert "path" not in fields and "file" not in fields and "folder" not in fields


# ── who is asking ────────────────────────────────────────────────────────


class TestWhoIsAsking:
    def test_without_control_nothing_opens(self, store, calls):
        r = TestClient(_client(store, control=False).app, client=("10.0.0.7", 5000)).post(
            "/api/reveal", json={"what": "session"}
        )
        assert r.status_code == 403
        assert calls["show"] == []

    def test_another_computer_is_told_the_path_and_nothing_opens(self, store, calls, monkeypatch):
        monkeypatch.setattr(os_reveal, "is_local", lambda host: False)
        r = _post(_client(store), what="session")
        assert r.status_code == 200
        body = r.json()
        assert body["opened"] is False and body["reason"] == "remote"
        assert Path(body["path"]) == store._session_dir("s1")
        assert calls["show"] == []

    def test_nor_does_fiji_for_another_computer(self, store, calls, fiji, monkeypatch):
        monkeypatch.setattr(os_reveal, "is_local", lambda host: False)
        r = _post(
            _client(store), what="timepoint", embryo_id="embryo_1", timepoint=3, action="fiji"
        )
        assert r.json()["opened"] is False and calls["fiji"] == []

    def test_asking_where_opens_nothing(self, store, calls):
        r = _post(
            _client(store), what="timepoint", embryo_id="embryo_1", timepoint=3, action="path"
        )
        assert r.status_code == 200 and r.json()["opened"] is False
        assert Path(r.json()["path"]) == store.get_volume_path("s1", "embryo_1", 3)
        assert calls["show"] == []

    def test_this_computer_is_local(self):
        assert os_reveal.is_local("127.0.0.1") and os_reveal.is_local("::1")
        assert not os_reveal.is_local("203.0.113.9")
        assert not os_reveal.is_local(None) and not os_reveal.is_local("")

    def test_about_says_what_this_browser_can_ask_for(self, store, calls, fiji):
        body = _client(store).get("/api/reveal/about").json()
        assert body["local"] is True
        assert body["fiji"] == {"found": True, "path": str(fiji)}
        assert body["what"]["timepoint"] == "file" and body["what"]["session"] == "folder"


# ── Fiji ─────────────────────────────────────────────────────────────────


class TestFiji:
    def test_a_volume_opens_in_fiji(self, store, calls, fiji):
        r = _post(
            _client(store), what="timepoint", embryo_id="embryo_1", timepoint=3, action="fiji"
        )
        assert r.status_code == 200, r.text
        assert calls["fiji"] == [(store.get_volume_path("s1", "embryo_1", 3), fiji)]
        assert r.json()["opened"] is True

    def test_a_folder_does_not(self, store, calls, fiji):
        r = _post(_client(store), what="embryo", embryo_id="embryo_1", action="fiji")
        assert r.status_code == 422 and calls["fiji"] == []

    def test_without_fiji_it_says_where_to_say_where_it_is(self, store, calls, monkeypatch):
        monkeypatch.setattr(reveal_routes, "fiji_path", lambda: None)
        r = _post(
            _client(store), what="timepoint", embryo_id="embryo_1", timepoint=3, action="fiji"
        )
        assert r.status_code == 409
        assert "Settings" in r.json()["detail"] and calls["fiji"] == []

    def test_the_configured_path_is_used(self, tmp_path):
        exe = tmp_path / "somewhere" / "ImageJ-win64.exe"
        exe.parent.mkdir()
        exe.write_bytes(b"")
        assert os_reveal.find_fiji(str(exe)) == exe

    def test_the_folder_it_is_in_will_do(self, tmp_path):
        exe = tmp_path / "Fiji.app" / "fiji-windows-x64.exe"
        exe.parent.mkdir()
        exe.write_bytes(b"")
        assert os_reveal.find_fiji(str(exe.parent)) == exe

    def test_a_configured_path_that_is_not_there_is_not_replaced(self, tmp_path, monkeypatch):
        real = tmp_path / "Fiji" / "fiji-windows-x64.exe"
        real.parent.mkdir()
        real.write_bytes(b"")
        monkeypatch.setattr(os_reveal, "_candidates", lambda: [real])
        assert os_reveal.find_fiji(None) == real
        assert os_reveal.find_fiji(str(tmp_path / "gone" / "fiji.exe")) is None

    def test_micro_managers_imagej_is_never_found(self, tmp_path, monkeypatch):
        mm = tmp_path / "Micro-Manager-2.0" / "ImageJ.exe"
        mm.parent.mkdir()
        mm.write_bytes(b"")
        monkeypatch.setattr(os_reveal, "_candidates", lambda: [mm])
        assert os_reveal.find_fiji(None) is None

    def test_nor_used_when_it_is_configured(self, tmp_path):
        mm = tmp_path / "Micro-Manager-1.4" / "ImageJ.exe"
        mm.parent.mkdir()
        mm.write_bytes(b"")
        assert os_reveal.find_fiji(str(mm)) is None
        with pytest.raises(ValueError, match="Micro-Manager"):
            os_reveal.open_in_fiji(tmp_path / "t0000.tif", mm)

    def test_no_candidate_is_micro_managers(self):
        assert not [p for p in os_reveal._candidates() if os_reveal.is_micro_manager(p)]

    def test_it_is_a_setting_of_the_rig(self):
        from gently.ui.web.settings_registry import SETTINGS

        by_key = {s.key: s for s in SETTINGS}
        assert by_key["system.fiji.path"].store == "env:GENTLY_FIJI_PATH"
        assert by_key["system.fiji.path"].source == "ui.fiji_path"
        assert by_key["system.fiji.found"].type == "readonly"


# ── the page ─────────────────────────────────────────────────────────────


class TestThePage:
    def test_the_module_is_loaded_before_the_viewers(self):
        assert INDEX.index("/static/js/reveal.js") < INDEX.index("/static/js/lightbox.js")
        assert '<link rel="stylesheet" href="/static/css/reveal.css">' in INDEX

    def test_it_sends_what_and_never_where(self):
        assert "body: JSON.stringify({ ...what, action })," in REVEAL_JS
        sent = REVEAL_JS[REVEAL_JS.index("async function post(") :][:500]
        assert "path" not in sent

    def test_another_computer_is_handed_the_path(self):
        fn = REVEAL_JS[REVEAL_JS.index("    async function run(") :][:1100]
        assert "data.reason === 'remote'" in fn and "await copy(data.path)" in fn

    def test_a_button_in_a_row_does_not_open_the_row(self):
        tail = REVEAL_JS[REVEAL_JS.index("document.addEventListener('click'") :][:700]
        assert "e.stopPropagation();" in tail
        assert re.search(r"\}, true\);", tail), "the listener is not in the capture phase"

    def test_fijis_button_is_shown_by_a_rule_not_a_pass(self):
        assert ".reveal-fiji { display: none; }" in CSS
        assert 'html[data-reveal-fiji="1"] .reveal-fiji { display: inline-flex; }' in CSS
        assert (
            "root.dataset.revealFiji = _about.local && _about.fiji && _about.fiji.found"
            in REVEAL_JS
        )

    def test_hidden_hides_it(self):
        assert ".reveal-btn[hidden] { display: none; }" in CSS

    def test_a_viewer_offers_only_what_is_on_disk(self):
        fn = REVEAL_JS[REVEAL_JS.index("    async function fill(") :][:900]
        assert "const found = await probe(what);" in fn
        assert "if (host.dataset.revealFor !== key || !found) return;" in fn

    @pytest.mark.parametrize(
        "img,what",
        [
            (
                {"url": "/api/calibration/records/embryo_1/20260929_013218/image/10.png?max=720"},
                {
                    "what": "calibration_image",
                    "embryo_id": "embryo_1",
                    "run": "20260929_013218",
                    "n": 10,
                },
            ),
            ({"url": "/api/dic/frames/dic_f0007.png"}, {"what": "dic", "stem": "dic_f0007"}),
            (
                {
                    "url": "/x.jpg",
                    "metadata": {"embryo_id": "embryo_2", "timepoint": 0, "session_id": "s9"},
                },
                {"what": "timepoint", "embryo_id": "embryo_2", "timepoint": 0, "session_id": "s9"},
            ),
            (
                {"uid": "abc", "metadata": {"embryo_id": "embryo_2", "timepoint": 4}},
                {"what": "timepoint", "embryo_id": "embryo_2", "timepoint": 4},
            ),
            ({"uid": "abc", "data_type": "live"}, None),
            ({"reveal": {"what": "logs"}}, {"what": "logs"}),
        ],
    )
    def test_what_a_viewers_image_is(self, img, what, tmp_path):
        import shutil
        import subprocess

        node = shutil.which("node")
        if node is None:
            pytest.skip("node is not installed")
        script = tmp_path / "describe.js"
        script.write_text(
            "const document = { addEventListener() {}, readyState: 'complete',"
            " documentElement: { dataset: {} } };\n"
            "const fetch = () => Promise.resolve({ ok: false, json: () => ({}) });\n"
            + REVEAL_JS
            + f"\nconsole.log(JSON.stringify(Reveal.describe({json.dumps(img)})));\n",
            encoding="utf-8",
        )
        out = subprocess.run([node, str(script)], capture_output=True, text=True, timeout=30)
        assert out.returncode == 0, out.stderr
        assert json.loads(out.stdout.strip()) == what


class TestWhereTheButtonsAre:
    def _js(self, name):
        return (JS / name).read_text(encoding="utf-8")

    def test_both_viewers(self):
        assert 'id="lightbox-reveal"' in INDEX
        stage = (WEB / "static" / "js" / "panels" / "overview-stage.js").read_text(encoding="utf-8")
        assert "what: 'dic', stem: f.stem" in stage  # the stage reveals the frame it shows
        lightbox = self._js("lightbox.js")
        assert lightbox.count("this.showReveal(img);") == 2, "one of the two ways of showing forgot"

    def test_an_embryo_and_a_timepoint(self):
        embryos = self._js("embryos.js")
        assert "{ what: 'embryo', embryo_id: embryo.embryoId }" in embryos
        assert embryos.count("what: 'timepoint', embryo_id: this.selectedEmbryoId") == 2

    def test_every_session_in_the_list(self):
        assert "{ what: 'session', session_id: s.session_id }" in self._js("review.js")

    def test_the_two_that_were_there_go_the_same_way(self):
        app = self._js("app.js")
        fn = app[app.index("async function openSessionFolder()") :][:700]
        assert "Reveal.run({ what: 'session', session_id: sessionId }, 'show');" in fn
        operate = self._js("operate.js")
        fn = operate[operate.index("    async function openCalFolder() {") :][:500]
        assert "what: 'calibration_run', embryo_id: _calKept.embryoId, run: _calKept.run" in fn

    def test_the_history_file_once_there_is_one(self):
        settings = self._js("settings.js")
        assert "{ what: 'settings_history' }, 'show'," in settings
        assert "if (d.total && typeof Reveal !== 'undefined') {" in settings

    @pytest.mark.parametrize("what", ["logs", "recordings", "storage", "config", "agent"])
    def test_the_folders_that_are_not_a_sessions(self, what):
        assert f'data-reveal=\'{{"what":"{what}"}}\'' in INDEX
        assert what in reveal_routes.WHAT

    def test_every_static_button_asks_for_something_the_route_knows(self):
        asked = re.findall(r"data-reveal='\{\"what\":\"(\w+)\"\}'", INDEX)
        assert asked and all(a in reveal_routes.WHAT for a in asked), asked
