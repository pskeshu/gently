"""An image the agent pushes to the page keeps its counts.

A pushed image lived on only as the PNG drawn from it. The viz server now
keeps the numbers behind the recent ones within a byte budget, says so on
the record (``has_counts``), and hands them out as uint16 bytes, so the
Gallery and the lightbox draw them with their own contrast (gently.core.u16).
"""

from pathlib import Path

import numpy as np
from fastapi import FastAPI
from fastapi.testclient import TestClient

from gently.core.imaging import (
    apply_crop_bounds,
    compute_crop_bounds,
    projection_three_view,
    pushed_counts,
)
from gently.core.u16 import CountsKeep
from gently.ui.web.models import ImageData
from gently.ui.web.routes import images as images_routes

WEB = Path(__file__).resolve().parents[1] / "gently" / "ui" / "web"


def test_the_keep_holds_the_newest_within_its_budget() -> None:
    k = CountsKeep(budget=3 * 8)  # three 2×2 uint16 frames
    for i in range(4):
        k.put(f"u{i}", np.full((2, 2), i, dtype=np.uint16))
    assert len(k) == 3 and "u0" not in k and "u3" in k
    k.get("u1")  # a read keeps it young
    k.put("u4", np.zeros((2, 2), dtype=np.uint16))
    assert "u1" in k and "u2" not in k
    k.put("big", np.zeros((100, 100), dtype=np.uint16))
    assert "big" not in k  # larger than the whole budget: not kept
    k.put("u3", np.ones((2, 2), dtype=np.uint16))  # the same uid again replaces, not doubles
    assert len(k) == 3
    k.clear()
    assert len(k) == 0 and k.get("u4") is None


def test_a_pushed_volume_gives_the_three_view_its_png_shows_and_a_rendering_gives_none() -> None:
    vol = np.random.default_rng(2).poisson(200, (6, 40, 50)).astype(np.uint16)
    vol[3, 10:30, 15:40] = 3000
    got = pushed_counts(vol)
    assert got is not None
    arr, layout = got
    cropped = apply_crop_bounds(vol, compute_crop_bounds(vol))
    png, _ = projection_three_view(cropped)
    assert arr.shape == png.shape and arr.dtype == np.uint16
    assert layout["views"] == "xy+yz/xz"
    plane, layout = pushed_counts(np.full((8, 9), 7, dtype=np.uint16))  # type: ignore[misc]
    assert plane.shape == (8, 9) and layout["xy"] == [8, 9]
    assert pushed_counts(np.zeros((8, 9, 3), dtype=np.uint8)) is None  # an RGB rendering
    stack = np.zeros((2, 3, 4, 5), dtype=np.uint16)
    stack[0] = 5
    first, _ = pushed_counts(stack)  # type: ignore[misc]
    assert first[:4, :5].max() == 5  # view A


def test_the_route_hands_out_the_kept_counts_and_says_when_they_are_gone() -> None:
    class Server:
        def __init__(self) -> None:
            self.keep = CountsKeep()

        def counts_for(self, uid: str):
            return self.keep.get(uid)

    srv = Server()
    srv.keep.put("abc", np.arange(12, dtype=np.uint16).reshape(3, 4), {"views": "xy", "xy": [3, 4]})
    app = FastAPI()
    app.include_router(images_routes.create_router(srv))
    with TestClient(app) as c:
        r = c.get("/api/images/abc.u16")
        assert r.status_code == 200 and len(r.content) == 24
        assert r.headers["x-image-views"] == "xy" and r.headers["x-image-xy-width"] == "4"
        assert c.get("/api/images/abc.u16?max=2").headers["x-image-scale"] == "2"
        assert c.get("/api/images/nope.u16").status_code == 404


def test_the_record_says_so_and_the_page_asks_for_them() -> None:
    assert ImageData(uid="u", data_type="t", timestamp="now").to_dict()["has_counts"] is False
    gallery = (WEB / "static" / "js" / "gallery.js").read_text(encoding="utf-8")
    assert "function pushedThumb(img, cls" in gallery and gallery.count("mountPushedTiles(") >= 6
    assert 'data-u16="/api/images/${encodeURIComponent(img.uid)}.u16?max=' in gallery
    lightbox = (WEB / "static" / "js" / "lightbox.js").read_text(encoding="utf-8")
    assert "img.has_counts && img.uid" in lightbox
    server = (WEB / "server.py").read_text(encoding="utf-8")
    assert "image_data.has_counts = uid in self._counts" in server
