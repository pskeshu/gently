"""Frames leave the server as the numbers they are.

"i want to take a crack at the intensity of the images problem - as in pixel
wise intensity." Every image used to leave as 8-bit PNG or JPEG after a
percentile stretch. The .u16 routes send little-endian uint16 bytes, averaged
down by an integer factor if asked, with shape and range in headers, so the
browser draws its own contrast and reads exact values.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from gently.core.file_store import FileStore
from gently.core.u16 import block_mean, u16_payload
from gently.ui.web import auth
from gently.ui.web.routes import bf as bf_routes
from gently.ui.web.routes import sessions as sessions_routes

pytest.importorskip("tifffile")


def test_the_payload_is_the_array_little_endian_with_its_shape_and_range():
    arr = np.arange(12, dtype=np.uint16).reshape(3, 4) * 1000
    body, h = u16_payload(arr)
    assert len(body) == 3 * 4 * 2
    assert np.array_equal(np.frombuffer(body, "<u2").reshape(3, 4), arr)
    assert (h["X-Image-Width"], h["X-Image-Height"], h["X-Image-Dtype"]) == ("4", "3", "uint16")
    assert (h["X-Image-Min"], h["X-Image-Max"], h["X-Image-Scale"]) == ("0", "11000", "1")


def test_scale_averages_blocks_and_keeps_the_source_shape_in_headers():
    arr = np.array([[0, 2, 10, 12], [4, 6, 14, 16], [1, 1, 1, 1]], dtype=np.uint16)
    body, h = u16_payload(arr, scale=2)
    assert (h["X-Image-Width"], h["X-Image-Height"]) == ("2", "1")  # the odd row is dropped
    assert (h["X-Image-Source-Width"], h["X-Image-Source-Height"]) == ("4", "3")
    assert np.frombuffer(body, "<u2").tolist() == [3, 13]
    assert block_mean(arr, 1) is arr
    # a float frame (a corrected one) is rounded and clipped into range
    body, _ = u16_payload(np.array([[-5.0, 70000.0, 12.6]]))
    assert np.frombuffer(body, "<u2").tolist() == [0, 65535, 13]


def _client(store, active=None):
    server = MagicMock()
    server.agent_bridge.agent.store = store
    server.agent_bridge.agent.session_id = active
    server.gently_store = store
    app = FastAPI()
    app.include_router(sessions_routes.create_router(server))
    app.include_router(bf_routes.create_router(server))
    app.dependency_overrides[auth.require_control] = lambda: True
    return TestClient(app)


def test_a_brightfield_frame_comes_back_as_its_pixels(tmp_path):
    store = FileStore(tmp_path)
    store.create_session("s1")
    frame = (np.arange(64 * 48, dtype=np.uint16).reshape(48, 64) * 7) % 60000
    store.put_snapshot("s1", "bf", frame, metadata={"channel": "bf", "frame": 1}, stem="f1")
    # the store files a snapshot as {source}_{stem}; the routes take that name
    (rec,) = store.list_snapshots("s1", "bf")
    from pathlib import Path

    stem = Path(rec["file_path"]).stem
    c = _client(store, active="s1")
    for url in (f"/api/sessions/s1/snapshot/{stem}.u16", f"/api/bf/frames/{stem}.u16"):
        r = c.get(url)
        assert r.status_code == 200, url
        assert r.headers["content-type"].startswith("application/octet-stream")
        assert (r.headers["x-image-width"], r.headers["x-image-height"]) == ("64", "48")
        assert np.array_equal(np.frombuffer(r.content, "<u2").reshape(48, 64), frame)
    r = c.get(f"/api/sessions/s1/snapshot/{stem}.u16?scale=4")
    assert (r.headers["x-image-width"], r.headers["x-image-height"]) == ("16", "12")
    assert r.headers["x-image-source-width"] == "64" and r.headers["x-image-scale"] == "4"
    assert np.array_equal(
        np.frombuffer(r.content, "<u2").reshape(12, 16), np.rint(block_mean(frame, 4)).astype("<u2")
    )
    assert c.get("/api/sessions/s1/snapshot/nope.u16").status_code == 404


def test_a_stored_volume_gives_one_plane_or_its_projection(tmp_path):
    import tifffile

    store = FileStore(tmp_path)
    store.create_session("s1")
    store.register_embryo("s1", "e01", position_x=0, position_y=0)
    vdir = store._embryo_dir("s1", "e01") / "volumes"
    vdir.mkdir(parents=True, exist_ok=True)
    vol = np.zeros((5, 8, 6), dtype=np.uint16)
    vol[0] = 100
    vol[1, 2:5, 1:4] = 900
    vol[2] = 300
    vol[4, 0, 0] = 2000
    # written the way the store writes volumes: one page per plane, zlib
    tifffile.imwrite(str(vdir / store._volume_filename(7)), vol, compression="zlib")
    c = _client(store)
    r = c.get("/api/sessions/s1/embryos/e01/volumes/7/slice/1.u16")
    assert (
        r.status_code == 200
        and r.headers["x-image-planes"] == "5"
        and r.headers["x-image-plane"] == "1"
    )
    assert np.array_equal(np.frombuffer(r.content, "<u2").reshape(8, 6), vol[1])
    r = c.get("/api/sessions/s1/embryos/e01/volumes/7/projection.u16")
    assert r.headers["x-image-projection"] == "max"
    assert np.array_equal(np.frombuffer(r.content, "<u2").reshape(8, 6), vol.max(axis=0))
    assert c.get("/api/sessions/s1/embryos/e01/volumes/7/slice/5.u16").status_code == 404
    assert c.get("/api/sessions/s1/embryos/e01/volumes/8/projection.u16").status_code == 404
