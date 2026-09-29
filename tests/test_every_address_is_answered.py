"""Whatever a page asks for, something answers.

Found by walking the whole app before 1.0.0rc1 and writing down every
request that was refused. Two were refused on every rig, always:

- The Light panel read the LED's state from ``/api/devices/led/status``.
  There was no such route. The LED was never known, so the panel's mode
  (LED, laser, both, off) was "unknown" whatever the hardware was doing.
- The Map's region history has a Restore on every row. There was no route
  for it in the web server, no method on the client, and no route in the
  device layer: only a handler, never registered.

Both had tests. The tests checked the page's source and the handler, and
not that one could reach the other. This checks the chain.
"""

from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from gently.ui.web import auth
from gently.ui.web.routes.data import create_router

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "gently" / "ui" / "web"
CLIENT = (ROOT / "gently" / "hardware" / "dispim" / "client.py").read_text(encoding="utf-8")
DEVICE_LAYER = (ROOT / "gently" / "hardware" / "dispim" / "device_layer.py").read_text(
    encoding="utf-8"
)


# ── the chain, by reading it ─────────────────────────────────────────────


def _web_routes() -> list[tuple[str, re.Pattern]]:
    found = []
    decorator = re.compile(
        r'@router\.(?:get|post|put|delete|patch|websocket|api_route)\(\s*"([^"]+)"'
    )
    for f in sorted((WEB / "routes").glob("*.py")):
        for m in decorator.finditer(f.read_text(encoding="utf-8")):
            path = m.group(1)
            rx = re.escape(path)
            rx = re.sub(r"\\\{[^}]*:path\\\}", ".+", rx)
            rx = re.sub(r"\\\{[^}]*\\\}", "[^/]+", rx)
            found.append((path, re.compile("^" + rx + "/?$")))
    return found


def _addresses_the_pages_call() -> dict[str, list[str]]:
    call = re.compile(r"""(['"`])(/(?:api|replay)/[^'"`\s]*)\1""")
    found: dict[str, list[str]] = {}
    files = list((WEB / "static" / "js").rglob("*.js")) + list((WEB / "templates").glob("*.html"))
    for f in files:
        text = f.read_text(encoding="utf-8", errors="replace")
        for n, line in enumerate(text.splitlines(), 1):
            if line.strip().startswith(("//", "*", "/*", "{#")):
                continue
            for m in call.finditer(line):
                url = m.group(2).split("?")[0]
                found.setdefault(url, []).append(f"{f.relative_to(WEB).as_posix()}:{n}")
    return found


def _is_answered(url: str, routes: list[tuple[str, re.Pattern]]) -> bool:
    # `${...}` stands for one part of the path.
    probe = re.sub(r"\$\{[^}]*\}", "X", url)
    if any(rx.match(probe) for _, rx in routes):
        return True
    # A part that is a variable in the page may be a word in the route:
    # `/api/context/${kind}/${id}/resolve` is three routes, one per kind.
    shape = "[^/]+".join(re.escape(p) for p in re.split(r"\$\{[^}]*\}", url))
    as_written = re.compile("^" + shape + "/?$")
    samples = [re.sub(r"\{[^}]*\}", "X", path) for path, _ in routes]
    if any(as_written.match(s) for s in samples):
        return True
    # A beginning that something is added to: '/api/x/' + id.
    if url.endswith("/"):
        return any(path.startswith(url) for path, _ in routes)
    return False


# Not addresses: a test for whether a request is one of ours.
NOT_A_CALL = {"/api/"}


def test_every_address_a_page_calls_has_a_route():
    routes = _web_routes()
    assert len(routes) > 100, "the routes were not found; this test would pass on nothing"
    called = _addresses_the_pages_call()
    assert len(called) > 80, "the pages' calls were not found"
    unanswered = {
        url: where
        for url, where in called.items()
        if url not in NOT_A_CALL and not _is_answered(url, routes)
    }
    assert not unanswered, (
        "a page calls an address the server has no route for, so it is a 404 on every "
        f"rig: {json.dumps(unanswered, indent=1)}"
    )


def test_the_check_would_have_caught_both():
    routes = [
        (p, rx) for p, rx in _web_routes() if "led/status" not in p and "region/restore" not in p
    ]
    assert not _is_answered("/api/devices/led/status", routes)
    assert not _is_answered("/api/devices/stage/region/restore", routes)
    assert _is_answered("/api/context/${kind}/${encodeURIComponent(id)}/resolve", routes)
    assert _is_answered("/api/embryos/${encodeURIComponent(id)}/restore", routes)


def _device_routes() -> list[tuple[str, str, re.Pattern, str]]:
    found = []
    for method, path, handler in re.findall(
        r'add_(get|post|put|delete)\(\s*"([^"]+)"\s*,\s*self\.(\w+)', DEVICE_LAYER
    ):
        rx = re.sub(r"\\\{[^}]*\\\}", "[^/]+", re.escape(path))
        found.append((method.upper(), path, re.compile("^" + rx + "$"), handler))
    return found


def test_every_address_the_client_calls_has_a_route_in_the_device_layer():
    routes = _device_routes()
    assert len(routes) > 30
    calls = set(re.findall(r'_api_(get|post)\(\s*f?"([^"]+)"', CLIENT))
    assert len(calls) > 30
    unanswered = []
    for method, path in sorted(calls):
        # `{suffix}` is a query string built by the caller, not part of the path.
        probe = re.sub(r"\{suffix\}$", "", path)
        probe = re.sub(r"\{[^}]*\}", "X", probe).split("?")[0]
        if not any(m == method.upper() and rx.match(probe) for m, _, rx, _ in routes):
            unanswered.append(f"{method.upper()} {path}")
    assert not unanswered, f"the client calls what the device layer does not serve: {unanswered}"


def test_every_handler_the_device_layer_has_is_registered():
    handlers = set(
        re.findall(r"^    async def (handle_\w+)\(self, request", DEVICE_LAYER, flags=re.M)
    )
    registered = {h for _, _, _, h in _device_routes()}
    # a handler another handler calls is reached through that one
    called = {h for h in handlers if re.search(r"self\." + h + r"\(", DEVICE_LAYER)}
    forgotten = sorted(handlers - registered - called)
    assert not forgotten, f"written and never registered, so never reachable: {forgotten}"


# ── the LED ──────────────────────────────────────────────────────────────


def _app(client):
    server = MagicMock()
    server.agent_bridge.agent.client = client
    server.agent_bridge.agent.session_id = "s1"
    app = FastAPI()
    app.include_router(create_router(server))
    app.dependency_overrides[auth.require_control] = lambda: True
    return TestClient(app)


def _rig(**methods):
    client = MagicMock()
    client.is_connected = True
    for name, value in methods.items():
        setattr(client, name, value if callable(value) else AsyncMock(return_value=value))
    return client


class TestTheLed:
    def test_the_panel_is_told_the_leds_state(self):
        rig = _rig(get_led_status={"success": True, "current_state": "Open"})
        r = _app(rig).get("/api/devices/led/status")
        assert r.status_code == 200, r.text
        assert r.json()["current_state"] == "Open"

    def test_it_is_the_field_the_panel_reads(self):
        light = (WEB / "static" / "js" / "panels" / "light.js").read_text(encoding="utf-8")
        assert "get('/api/devices/led/status', 'led', d => (d && d.current_state) || null)" in light
        assert '"current_state": led_value' in DEVICE_LAYER

    def test_an_led_that_could_not_be_read_is_not_a_closed_one(self):
        rig = _rig(get_led_status={"success": False, "error": "LED device not found"})
        r = _app(rig).get("/api/devices/led/status")
        assert r.status_code == 502 and "LED device not found" in r.json()["detail"]

    def test_a_read_that_raises_is_a_failure_not_a_state(self):
        rig = _rig(get_led_status=AsyncMock(side_effect=RuntimeError("timed out")))
        assert _app(rig).get("/api/devices/led/status").status_code == 502

    def test_no_microscope_is_said(self):
        rig = _rig()
        rig.is_connected = False
        assert _app(rig).get("/api/devices/led/status").status_code == 503

    def test_reading_needs_no_control(self):
        rig = _rig(get_led_status={"success": True, "current_state": "Closed"})
        server = MagicMock()
        server.agent_bridge.agent.client = rig
        app = FastAPI()
        app.include_router(create_router(server))
        app.dependency_overrides[auth.require_control] = lambda: (_ for _ in ()).throw(
            AssertionError("a read asked for control")
        )
        assert TestClient(app).get("/api/devices/led/status").status_code == 200


# ── the region ───────────────────────────────────────────────────────────

BOX_A = {"x_min": -1000.0, "x_max": 1000.0, "y_min": -800.0, "y_max": 800.0}
BOX_B = {"x_min": -500.0, "x_max": 500.0, "y_min": -400.0, "y_max": 400.0}


class TestTheRegionsRoute:
    def _rig(self, answer):
        return _rig(
            get_stage_envelope={"success": True, "region": dict(BOX_B)},
            restore_stage_region=answer,
        )

    def test_restore_reaches_the_device_layer(self, monkeypatch, tmp_path):
        monkeypatch.setenv("GENTLY_STORAGE_PATH", str(tmp_path))
        rig = self._rig({"success": True, "region": dict(BOX_A), "history": []})
        r = _app(rig).post(
            "/api/devices/stage/region/restore", json={"applied_at": "2026-09-24T10:00:00"}
        )
        assert r.status_code == 200, r.text
        assert r.json()["region"] == BOX_A
        rig.restore_stage_region.assert_awaited_once()
        assert rig.restore_stage_region.await_args.args[0] == "2026-09-24T10:00:00"

    def test_it_answers_what_the_editor_reads(self):
        editor = (WEB / "static" / "js" / "region-editor.js").read_text(encoding="utf-8")
        assert "const RESTORE = '/api/devices/stage/region/restore';" in editor
        fn = editor[editor.index("    async function restore(appliedAt) {") :][:400]
        assert "postJSON(RESTORE, { applied_at: appliedAt })" in fn
        assert "d.region" in fn and "d.history" in fn
        assert '"region": region,' in DEVICE_LAYER and '"history": history,' in DEVICE_LAYER

    def test_a_time_nothing_was_applied_at_is_not_found(self):
        rig = self._rig({"success": False, "error": "no region with that timestamp"})
        r = _app(rig).post("/api/devices/stage/region/restore", json={"applied_at": "never"})
        assert r.status_code == 404

    def test_a_region_the_stage_is_outside_of_is_refused_in_its_words(self):
        rig = self._rig({"success": False, "error": "stage is outside the new region"})
        r = _app(rig).post(
            "/api/devices/stage/region/restore", json={"applied_at": "2026-09-24T10:00:00"}
        )
        assert r.status_code == 409 and "outside" in r.json()["detail"]

    @pytest.mark.parametrize("body", [{}, {"applied_at": ""}, {"applied_at": 7}])
    def test_which_region_has_to_be_said(self, body):
        rig = self._rig({"success": True})
        r = _app(rig).post("/api/devices/stage/region/restore", json=body)
        assert r.status_code == 400
        rig.restore_stage_region.assert_not_awaited()

    def test_it_needs_control(self):
        src = (WEB / "routes" / "data.py").read_text(encoding="utf-8")
        assert (
            '@router.post("/api/devices/stage/region/restore", '
            "dependencies=[Depends(require_control)])" in src
        )

    def test_the_client_asks_the_device_layer(self):
        assert '"/api/stage/region/restore",' in CLIENT
        assert 'add_post("/api/stage/region/restore", self.handle_restore_region)' in DEVICE_LAYER


class _Stage:
    """A stage that records what it was held to, and can refuse."""

    name = "xy"

    def __init__(self, refuse=False):
        self.refuse = refuse
        self.software = None
        self.firmware = None

    def set_software_limits(self, *box):
        if self.refuse:
            raise ValueError("stage is outside the new region")
        self.software = box

    def set_firmware_limits(self, *box):
        if self.refuse:
            raise ValueError("stage is outside the new region")
        self.firmware = box


class _Req:
    def __init__(self, body):
        self._b = body

    async def json(self):
        return self._b


class TestTheDeviceLayer:
    @pytest.fixture
    def region(self, tmp_path, monkeypatch):
        from gently.core import xy_region

        monkeypatch.setattr(xy_region, "region_path", lambda: tmp_path / "xy_region.yaml")
        from datetime import datetime

        xy_region.apply(BOX_A, note="first", now=datetime(2026, 9, 24, 10, 0, 0))
        xy_region.apply(BOX_B, note="second", now=datetime(2026, 9, 24, 11, 0, 0))
        return xy_region

    def _dl(self, stage, enforced=False):
        from gently.hardware.dispim.device_layer import DeviceLayerServer

        dl = DeviceLayerServer.__new__(DeviceLayerServer)
        dl.devices = {"xy_stage": stage}
        dl.config = {"xy_envelope": {**BOX_B, "enforced": enforced}}
        dl._write_sidecar = MagicMock()  # type: ignore[method-assign]
        dl._envelope_payload = lambda st: {"success": True}  # type: ignore[method-assign]

        class _NoPause:
            async def __aenter__(self):
                return None

            async def __aexit__(self, *a):
                return False

        dl.pause_state_updates = lambda: _NoPause()  # type: ignore[method-assign]
        return dl

    def _restore(self, dl, at):
        resp = asyncio.run(dl.handle_restore_region(_Req({"applied_at": at})))
        return resp.status, json.loads(resp.text)

    def _first(self, region):
        return next(h for h in region.load().history if h.box == BOX_A).applied_at

    def test_the_stage_is_held_to_the_region_brought_back(self, region):
        stage = _Stage()
        status, _ = self._restore(self._dl(stage), self._first(region))
        assert status == 200
        assert stage.software == (-1000.0, 1000.0, -800.0, 800.0)
        assert region.load().current.box == BOX_A

    def test_the_controller_hears_only_when_its_limits_are_on(self, region):
        off, on = _Stage(), _Stage()
        self._restore(self._dl(off, enforced=False), self._first(region))
        assert off.firmware is None
        region.apply(BOX_B, note="again")
        self._restore(self._dl(on, enforced=True), self._first(region))
        assert on.firmware == (-1.0, 1.0, -0.8, 0.8), "the controller takes millimetres"

    def test_the_region_replaced_joins_the_history(self, region):
        self._restore(self._dl(_Stage()), self._first(region))
        assert BOX_B in [h.box for h in region.load().history]

    def test_a_refusal_leaves_the_record_as_it_was(self, region):
        # The record used to be changed before the controller was written to.
        before = region.load()
        status, body = self._restore(self._dl(_Stage(refuse=True)), self._first(region))
        assert status == 409 and "outside" in body["error"]
        after = region.load()
        assert after.current.box == BOX_B == before.current.box
        assert len(after.history) == len(before.history)

    def test_a_time_nothing_was_applied_at(self, region):
        stage = _Stage()
        status, _ = self._restore(self._dl(stage), "2001-01-01T00:00:00")
        assert status == 404 and stage.software is None

    def test_looking_changes_nothing(self, region):
        before = region.region_path().read_text(encoding="utf-8")
        assert region.find(self._first(region)).box == BOX_A
        assert region.find("never") is None
        assert region.region_path().read_text(encoding="utf-8") == before
