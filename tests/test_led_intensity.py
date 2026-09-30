"""The LED's brightness is set on the ASI Tiger, and read back from it.

The LED was a shutter to gently: Open or Closed, at whatever brightness the
controller happened to hold. The Tiger adapter has had a brightness property
all along (`LED Intensity(%)`, ASILED.cpp), so this writes that property
directly rather than adding preset levels to the `LED` config group.

Three things are easy to get wrong, and each has a test here:

- 0 is not a brightness. The controller reports 0 for an LED that is off, and
  the adapter's own limits start at 1, so dark is `Closed`.
- Setting the brightness must not open or close the LED. The adapter holds a
  value set while closed and applies it on the next open.
- A status read must survive an LED whose brightness cannot be read. Open or
  Closed is still true, and the Light panel's mode depends on it.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

pytest.importorskip("aiohttp")

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

import gently.ui.web.auth as auth  # noqa: E402
from gently.hardware.dispim.device_layer import DeviceLayerServer  # noqa: E402
from gently.hardware.dispim.devices.optical import DiSPIMLED  # noqa: E402
from gently.ui.web.routes.data import create_router  # noqa: E402

LED_NAME = "LED:X:31"
PROP = "LED Intensity(%)"
LIGHT_JS = (
    Path(__file__).resolve().parents[1]
    / "gently"
    / "ui"
    / "web"
    / "static"
    / "js"
    / "panels"
    / "light.js"
).read_text(encoding="utf-8")


class _Core:
    """The slice of CMMCore the LED touches."""

    def __init__(self, intensity="50", readable=True):
        self.props = {(LED_NAME, PROP): intensity}
        self.readable = readable
        self.config = "Closed"
        self.calls: list[tuple] = []

    def getAvailableConfigs(self, group):  # noqa: N802
        return ["Open", "Closed"]

    def getCurrentConfig(self, group):  # noqa: N802
        return self.config

    def setConfig(self, group, config):  # noqa: N802
        self.calls.append(("setConfig", group, config))
        self.config = config

    def waitForConfig(self, group, config):  # noqa: N802
        pass

    def setProperty(self, dev, prop, val):  # noqa: N802
        self.calls.append(("setProperty", dev, prop, val))
        self.props[(dev, prop)] = str(val)

    def getProperty(self, dev, prop):  # noqa: N802
        if not self.readable:
            raise RuntimeError(f'No property "{prop}" on device "{dev}"')
        return self.props[(dev, prop)]


class _Req:
    def __init__(self, body):
        self._b = body

    async def json(self):
        return self._b


def _led(core=None) -> tuple[DiSPIMLED, _Core]:
    core = core or _Core()
    return DiSPIMLED(core=core, name=LED_NAME, group_name="LED"), core


def _dl(core=None) -> tuple[DeviceLayerServer, _Core]:
    led, core = _led(core)
    dl = DeviceLayerServer.__new__(DeviceLayerServer)
    dl.devices = {"led": led}
    return dl, core


def _body(resp) -> dict:
    return json.loads(resp.text)


# ---------------------------------------------------------------------------
# The device
# ---------------------------------------------------------------------------


class TestTheDevice:
    def test_it_writes_the_tigers_own_property(self):
        led, core = _led()
        led.set_intensity_pct(30)
        assert core.calls == [("setProperty", LED_NAME, PROP, 30)]

    def test_it_is_read_back_from_the_device(self):
        led, _ = _led()
        led.set_intensity_pct(72)
        assert led.get_intensity_pct() == 72

    @pytest.mark.parametrize("pct", [1, 100])
    def test_the_limits_are_inclusive(self, pct):
        led, _ = _led()
        led.set_intensity_pct(pct)
        assert led.get_intensity_pct() == pct

    @pytest.mark.parametrize("pct", [0, -5, 101, 1000])
    def test_out_of_range_is_refused_before_the_hardware(self, pct):
        led, core = _led()
        with pytest.raises(ValueError, match="outside"):
            led.set_intensity_pct(pct)
        assert core.calls == []

    def test_zero_is_told_how_to_turn_the_led_off(self):
        led, _ = _led()
        with pytest.raises(ValueError, match="Closed"):
            led.set_intensity_pct(0)

    @pytest.mark.parametrize("pct", [12.5, True, None, "bright"])
    def test_only_a_whole_percent_is_a_brightness(self, pct):
        led, core = _led()
        with pytest.raises(ValueError):
            led.set_intensity_pct(pct)
        assert core.calls == []

    def test_a_whole_float_is_a_whole_percent(self):
        """A slider hands over 40.0, and JSON does not tell the two apart."""
        led, core = _led()
        led.set_intensity_pct(40.0)
        assert core.calls == [("setProperty", LED_NAME, PROP, 40)]

    def test_it_does_not_open_or_close_the_led(self):
        led, core = _led()
        led.set_intensity_pct(30)
        assert not [c for c in core.calls if c[0] == "setConfig"]
        assert core.config == "Closed"


# ---------------------------------------------------------------------------
# The device layer
# ---------------------------------------------------------------------------


class TestTheDeviceLayer:
    def test_set_answers_with_what_the_device_holds(self):
        dl, core = _dl()
        resp = asyncio.run(dl.handle_set_led_intensity(_Req({"pct": 25})))
        assert resp.status == 200
        assert _body(resp) == {"success": True, "pct": 25, "readback_pct": 25}
        assert core.props[(LED_NAME, PROP)] == "25"

    @pytest.mark.parametrize("body", [{"pct": 0}, {"pct": 101}, {"pct": 12.5}, {}])
    def test_a_refused_value_is_a_400_not_a_500(self, body):
        dl, core = _dl()
        resp = asyncio.run(dl.handle_set_led_intensity(_Req(body)))
        assert resp.status == 400
        assert _body(resp)["success"] is False
        assert core.calls == []

    def test_no_led_is_said(self):
        dl, _ = _dl()
        dl.devices = {}
        resp = asyncio.run(dl.handle_set_led_intensity(_Req({"pct": 25})))
        assert resp.status == 503 and "LED device not found" in _body(resp)["error"]

    def test_status_carries_the_brightness_and_its_bounds(self):
        dl, _ = _dl(_Core(intensity="64"))
        body = _body(asyncio.run(dl.handle_get_led_status(None)))
        assert body["success"] is True
        assert body["current_state"] == "Closed"
        assert body["intensity_pct"] == 64
        assert body["intensity_limits_pct"] == {"min": 1, "max": 100}

    def test_an_unreadable_brightness_does_not_cost_the_state(self):
        dl, _ = _dl(_Core(readable=False))
        body = _body(asyncio.run(dl.handle_get_led_status(None)))
        assert body["success"] is True
        assert body["current_state"] == "Closed"
        assert body["intensity_pct"] is None

    def test_the_route_is_registered(self):
        src = Path(DeviceLayerServer.__module__.replace(".", "/") + ".py")
        text = (Path(__file__).resolve().parents[1] / src).read_text(encoding="utf-8")
        assert 'add_post("/api/led/intensity", self.handle_set_led_intensity)' in text


# ---------------------------------------------------------------------------
# The web route
# ---------------------------------------------------------------------------


def _app(client=None, control=True):
    server = MagicMock()
    server.agent_bridge.agent.client = client
    app = FastAPI()
    app.include_router(create_router(server))
    if control:
        app.dependency_overrides[auth.require_control] = lambda: True
    return TestClient(app)


class TestTheWebRoute:
    def test_it_forwards_to_the_client(self):
        client = MagicMock()
        client.set_led_intensity = AsyncMock(return_value={"success": True, "readback_pct": 30})
        r = _app(client).post("/api/devices/led/intensity", json={"pct": 30})
        assert r.status_code == 200 and r.json()["readback_pct"] == 30
        client.set_led_intensity.assert_awaited_once_with(30)

    @pytest.mark.parametrize("body", [{}, {"pct": "bright"}, {"pct": None}])
    def test_a_body_with_no_number_is_a_400(self, body):
        client = MagicMock()
        client.set_led_intensity = AsyncMock()
        r = _app(client).post("/api/devices/led/intensity", json=body)
        assert r.status_code == 400
        client.set_led_intensity.assert_not_awaited()

    def test_a_client_that_raises_is_a_502(self):
        client = MagicMock()
        client.set_led_intensity = AsyncMock(side_effect=RuntimeError("timed out"))
        r = _app(client).post("/api/devices/led/intensity", json={"pct": 30})
        assert r.status_code == 502


# ---------------------------------------------------------------------------
# The agent's tool
# ---------------------------------------------------------------------------


def _tool(name):
    import gently.app.tools.led_tools  # noqa: F401  (registers on import)
    from gently.harness.tools.registry import get_tool_registry

    return get_tool_registry()._tools[name].handler


class TestTheTool:
    async def test_it_reports_the_readback(self):
        client = MagicMock()
        client.set_led_intensity = AsyncMock(return_value={"success": True, "readback_pct": 20})
        out = await _tool("set_led_intensity")(pct=20, context={"client": client})
        assert out == "LED intensity set to 20% (readback: 20%)"
        client.set_led_intensity.assert_awaited_once_with(20)

    async def test_a_refusal_reaches_the_agent_in_the_devices_words(self):
        client = MagicMock()
        client.set_led_intensity = AsyncMock(
            return_value={"success": False, "error": "LED intensity 0% outside [1, 100]%."}
        )
        out = await _tool("set_led_intensity")(pct=0, context={"client": client})
        assert "outside [1, 100]" in out

    async def test_status_says_the_brightness(self):
        client = MagicMock()
        client.get_led_status = AsyncMock(
            return_value={"success": True, "current_state": "Open", "intensity_pct": 45}
        )
        out = await _tool("get_led_status")(context={"client": client})
        assert "Current state: Open" in out and "Intensity: 45%" in out

    async def test_status_does_not_invent_a_brightness(self):
        client = MagicMock()
        client.get_led_status = AsyncMock(
            return_value={"success": True, "current_state": "Open", "intensity_pct": None}
        )
        out = await _tool("get_led_status")(context={"client": client})
        assert "Intensity: unknown" in out


# ---------------------------------------------------------------------------
# The Light panel
# ---------------------------------------------------------------------------


def _fn(name: str) -> str:
    body = LIGHT_JS[LIGHT_JS.index(f"function {name}(") :]
    return body[: body.index("\n    }")]


def _led_detail() -> str:
    """The rows every LED mount draws, the full panel and the LED card alike."""
    return _fn("ledRows")


class TestThePanel:
    def test_the_slider_writes_to_the_route_that_exists(self):
        assert "send('/api/devices/led/intensity', { pct: Number(ledPct.value) })" in LIGHT_JS

    def test_it_writes_on_release_not_on_drag(self):
        """Every input event would be a serial write to the controller."""
        assert "ledPct.onchange" in LIGHT_JS
        assert "ledPct.oninput" not in LIGHT_JS

    def test_the_brightness_is_read_from_the_status_the_panel_already_asks_for(self):
        """One request answers both: the device layer is slow to read."""
        assert "d.intensity_pct" in LIGHT_JS
        # Once in the full read and once in the LED-only one, and never a
        # second request beside either.
        assert LIGHT_JS.count("'/api/devices/led/status'") == 2
        assert _fn("readAll").count("/api/devices/led/") == 1
        assert _fn("readLed").count("/api/devices/led/") == 1

    def test_an_unread_brightness_is_a_dash_and_a_dead_slider(self):
        body = _led_detail()
        assert "val == null ? 'disabled' : ''" in body
        assert "val == null ? '—' : val" in body

    def test_the_bounds_come_from_the_server(self):
        assert "s.ledLim ||" in _led_detail()

    def test_a_closed_led_says_why_the_image_did_not_change(self):
        assert "s.led === 'Closed'" in _led_detail()

    def test_the_panel_no_longer_says_there_is_no_control(self):
        assert "no intensity control" not in LIGHT_JS


# ---------------------------------------------------------------------------
# The bottom camera's LED card
# ---------------------------------------------------------------------------

WEB = Path(__file__).resolve().parents[1] / "gently" / "ui" / "web"
OPERATE_JS = (WEB / "static" / "js" / "operate.js").read_text(encoding="utf-8")
INDEX = (WEB / "templates" / "index.html").read_text(encoding="utf-8")


def _pane(name: str) -> str:
    start = INDEX.index(f'id="op-pane-{name}"')
    return INDEX[start : INDEX.index("</section>", start)]


class TestTheBottomCameraCard:
    """The bottom camera is lit by the LED, and had no way to dim it.

    It is the Light panel mounted a second time, not a second control: one
    subject, one panel, one state (docs/architecture/PANELS.md rules 1 and 2).
    """

    def test_the_host_is_on_the_bottom_pane_and_nowhere_else(self):
        assert 'id="op-led-host"' in _pane("bottom")
        assert INDEX.count('id="op-led-host"') == 1

    def test_operate_mounts_the_shared_panel_into_it(self):
        assert "LightPanel.mount('op-led-host', { only: 'led' })" in OPERATE_JS

    def test_it_is_mounted_with_the_pane_it_starts_on(self):
        """`mountLightPanel` waits for the SPIM pane; the bottom pane is first."""
        body = OPERATE_JS[OPERATE_JS.index("function mountPanels()") :]
        body = body[: body.index("\n    }")]
        assert "op-led-host" in body

    def test_the_card_draws_the_same_rows_as_the_full_panel(self):
        assert "${ledRows(s)}" in _fn("ledCard")
        assert "${ledRows(s)}" in _fn("ledDetail")

    def test_the_card_offers_nothing_of_the_laser(self):
        card = _fn("ledCard")
        for laser in ("modeRow", "laserBranch", "data-config", "data-beam", "lp-emit"):
            assert laser not in card, f"the LED card draws {laser}"

    def test_the_card_has_a_switch_and_the_full_panel_keeps_its_mode_control(self):
        """On the bottom pane the LED was read-only: it read Closed after any
        visit to the SPIM pane (leaving closes it) and nothing there could
        open it. The switch is the card's; the full panel switches the LED
        through its mode row, and one control per surface is the rule."""
        card = _fn("ledCard")
        assert "data-led-switch=\"${open ? 'Closed' : 'Open'}\"" in card
        assert "data-led-switch" not in _fn("ledRows")
        assert "data-led-switch" not in _fn("ledDetail")
        assert "sw.dataset.ledSwitch === 'Open'" in _fn("wire")

    def test_opening_the_led_gates_the_lasers_first_whatever_they_were_doing(self):
        """This card reads only the LED, so whether a line is routed is
        unknown here — and unknown is what #106 was made of."""
        body = _fn("ledSwitch")
        assert "if (open) await send('/api/devices/laser/config', { config: 'ALL OFF' });" in body
        assert body.index("laser/config") < body.index("led/set")
        assert "state: open ? 'Open' : 'Closed'" in body

    def test_an_led_card_on_screen_reads_only_the_led(self):
        body = _fn("readLed")
        assert "/api/devices/led/status" in body
        for other in ("/api/devices/beam", "/api/devices/laser"):
            assert other not in body

    def test_an_led_read_does_not_refresh_the_full_panels_age(self):
        body = _fn("readLed")
        assert "ledReadAt: Date.now()" in body
        assert " readAt" not in body and "next.readAt" not in body

    def test_a_full_panel_on_screen_still_reads_everything(self):
        assert "if (!opts.only) return 'all'" in _fn("visible")

    def test_other_callers_refresh_still_reads_everything(self):
        """operate.js calls `LightPanel.refresh()` after it closes the LED."""
        assert "refresh: readAll," in LIGHT_JS
