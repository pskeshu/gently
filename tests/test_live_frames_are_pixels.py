"""The live view draws the camera's own numbers.

A frame leaves the device layer as uint16 bytes, the viz server relays it as
it came, and the Operate pane draws it through the uint16 viewer instead of a
JPEG the device layer stretched for it (#149).
"""

import sys
from pathlib import Path
from unittest.mock import MagicMock

for _mod in (
    "bluesky",
    "bluesky.run_engine",
    "ophyd",
    "ophyd.status",
    "pymmcore",
    "gently.hardware.console_ui",
):
    sys.modules.setdefault(_mod, MagicMock())
import bluesky as _bs  # noqa: E402

_bs.RunEngine = MagicMock(name="RunEngine")  # type: ignore[misc]

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from gently.hardware.dispim.device_layer import DeviceLayerServer  # noqa: E402
from gently.ui.web.routes import data as data_routes  # noqa: E402


class _Req:
    def __init__(self, **q: str) -> None:
        self.query = q


def _dl() -> DeviceLayerServer:
    dl = DeviceLayerServer.__new__(DeviceLayerServer)
    dl._last_bottom_frame = None
    dl._last_bottom_frame_xy = None
    dl._last_ls_frame = None
    return dl


@pytest.mark.asyncio
async def test_the_device_layer_hands_out_its_last_frame_as_uint16() -> None:
    dl = _dl()
    assert (await dl.handle_bottom_camera_frame_u16(_Req())).status == 404
    dl._last_bottom_frame = np.arange(64 * 48, dtype=np.uint16).reshape(48, 64)
    dl._last_bottom_frame_xy = [1234.5, -20.0]
    r = await dl.handle_bottom_camera_frame_u16(_Req(scale="2"))
    assert r.status == 200 and r.content_type == "application/octet-stream"
    assert r.headers["X-Image-Width"] == "32" and r.headers["X-Image-Height"] == "24"
    assert r.headers["X-Image-Scale"] == "2" and r.headers["X-Image-Source-Width"] == "64"
    assert r.headers["X-Image-Stage-X"] == "1234.5" and r.headers["X-Image-Stage-Y"] == "-20.0"
    assert r.headers["Cache-Control"] == "no-store"
    assert isinstance(r.body, bytes) and len(r.body) == 32 * 24 * 2
    # the first 2×2 block, averaged and rounded half up
    assert np.frombuffer(r.body, "<u2")[0] == (0 + 1 + 64 + 65 + 2) // 4


@pytest.mark.asyncio
async def test_the_light_sheet_frame_too_and_a_bad_scale_is_one() -> None:
    dl = _dl()
    assert (await dl.handle_lightsheet_frame_u16(_Req())).status == 404
    dl._last_ls_frame = np.full((8, 8), 7, dtype=np.uint16)
    r = await dl.handle_lightsheet_frame_u16(_Req(scale="junk"))
    assert r.status == 200 and r.headers["X-Image-Scale"] == "1"
    assert "X-Image-Stage-X" not in r.headers


def _app(client: object) -> FastAPI:
    server = MagicMock()
    server.agent_bridge.agent.client = client
    app = FastAPI()
    app.include_router(data_routes.create_router(server))
    return app


def test_the_viz_server_relays_the_bytes_and_headers_as_they_came() -> None:
    class Client:
        asked: tuple[str, int] | None = None

        async def get_frame_u16(
            self, path: str, scale: int, max_dim: int | None = None
        ) -> tuple[int, dict[str, str], bytes]:
            self.asked = (path, scale)
            heads = {"X-Image-Width": "4", "X-Image-Height": "1", "X-Image-Scale": str(scale)}
            return 200, heads, b"\x01\x00\x02\x00\x03\x00\x04\x00"

    c = Client()
    with TestClient(_app(c)) as t:
        r = t.get("/api/devices/bottom_camera/frame.u16?scale=4")
    assert r.status_code == 200
    assert r.content == b"\x01\x00\x02\x00\x03\x00\x04\x00"
    assert r.headers["content-type"].startswith("application/octet-stream")
    assert r.headers["x-image-width"] == "4" and r.headers["cache-control"] == "no-store"
    assert c.asked == ("/api/bottom_camera/frame.u16", 4)


def test_no_frame_yet_and_no_microscope_each_say_so() -> None:
    class Client:
        async def get_frame_u16(
            self, path: str, scale: int, max_dim: int | None = None
        ) -> tuple[int, dict[str, str], bytes]:
            return 404, {}, b""

    with TestClient(_app(Client())) as t:
        assert t.get("/api/devices/lightsheet/frame.u16").status_code == 404
    with TestClient(_app(None)) as t:
        assert t.get("/api/devices/bottom_camera/frame.u16").status_code == 503


JS = Path(__file__).resolve().parents[1] / "gently" / "ui" / "web" / "static" / "js"


def test_the_operate_pane_draws_pixels_and_keeps_the_jpeg_as_the_fallback() -> None:
    src = (JS / "operate.js").read_text(encoding="utf-8")
    assert "/api/devices/bottom_camera/frame.u16" in src
    assert "/api/devices/lightsheet/frame.u16" in src
    assert "function showFrame(" in src
    for fn in ("onBottomFrame", "onSpimFrame"):
        body = src.split(f"function {fn}(p)")[1].split("\n    }\n")[0]
        assert "showFrame(" in body, f"{fn} does not go through showFrame"
    assert "setImg(imgId, phId, p)" in src, (
        "the JPEG path is gone: a rig without pixels would show nothing"
    )
    iv = (JS / "panels" / "imageview.js").read_text(encoding="utf-8")
    assert "_u16" in iv and "frameArrived" in iv, "the display panel does not read the uint16 view"
