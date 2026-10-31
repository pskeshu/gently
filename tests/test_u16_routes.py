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
    # The projection is the three-view layout in counts (gently.core.imaging):
    # XY at the top left, YZ beside it, XZ under. An older session has no
    # stored one, so it is made from the volume and kept for the next time.
    from gently.core.imaging import counts_projection

    want, layout = counts_projection(vol)
    r = c.get("/api/sessions/s1/embryos/e01/volumes/7/projection.u16")
    assert r.headers["x-image-projection"] == "max"
    assert r.headers["x-image-views"] == "xy+yz/xz"
    assert (r.headers["x-image-xy-height"], r.headers["x-image-xy-width"]) == ("8", "6")
    got = np.frombuffer(r.content, "<u2").reshape(want.shape)
    assert np.array_equal(got, want)
    assert np.array_equal(got[:8, :6], vol.max(axis=0))  # the XY view is the plain max
    stored = store.get_projection_u16_path("s1", "e01", 7)
    assert stored is not None and stored.name == "t0007.tif"
    r2 = c.get("/api/sessions/s1/embryos/e01/volumes/7/projection.u16")
    assert r2.content == r.content and r2.headers["x-image-xy-width"] == "6"
    assert c.get("/api/sessions/s1/embryos/e01/volumes/7/slice/5.u16").status_code == 404
    assert c.get("/api/sessions/s1/embryos/e01/volumes/8/projection.u16").status_code == 404


def test_the_store_writes_the_counts_beside_the_jpeg(tmp_path):
    """put_volume leaves t0000.jpg for the thumbnails and t0000.tif, the same
    picture in uint16 with its layout in the TIFF description, for the viewer."""
    from gently.core.imaging import projection_three_view, read_counts_projection

    store = FileStore(tmp_path)
    store.create_session("s1")
    store.register_embryo("s1", "e01", position_x=0, position_y=0)
    vol = (np.random.default_rng(1).poisson(300, (4, 10, 12))).astype(np.uint16)
    vol[2, 3:6, 4:9] = 5000
    store.put_volume("s1", "e01", 0, vol)
    proj_dir = store._embryo_dir("s1", "e01") / "projections"
    assert (proj_dir / "t0000.jpg").exists()
    path = store.get_projection_u16_path("s1", "e01", 0)
    assert path == proj_dir / "t0000.tif"
    arr, layout = read_counts_projection(path)
    assert arr.dtype == np.uint16
    assert layout == {"views": "xy+yz/xz", "xy": [10, 12]}
    # the same layout the JPEG draws, pixel for pixel in size
    jpeg_like, _ = projection_three_view(vol)
    assert arr.shape == jpeg_like.shape
    assert np.array_equal(arr[:10, :12], vol.max(axis=0))
    assert (arr[:10, 12:15] == 0).all()  # the band between the views is black, not a grey
    # the YZ view keeps the counts: nearest-neighbour, so every value is one the camera read
    assert arr[:10, 15:].max() == 5000
    assert set(np.unique(arr[:10, 15:])) <= set(np.unique(vol))


def test_a_four_d_stack_projects_view_a_and_a_plane_stays_a_plane():
    from gently.core.imaging import counts_projection

    stack = np.zeros((2, 3, 4, 5), dtype=np.uint16)
    stack[0, 1] = 7
    stack[1] = 900
    arr, layout = counts_projection(stack)
    assert layout["xy"] == [4, 5] and arr[:4, :5].max() == 7
    plane, layout = counts_projection(np.full((6, 7), 3, dtype=np.int32))
    assert plane.shape == (6, 7) and plane.dtype == np.uint16 and layout["views"] == "xy"


def test_max_asks_for_a_size_and_the_route_picks_the_scale(tmp_path):
    """A thumbnail says the pixels it has, not a factor: ?max=N gives the
    smallest integer scale that brings the longer side within N."""
    body, h = u16_payload(np.zeros((300, 1000), dtype=np.uint16), max_dim=256)
    assert (h["X-Image-Scale"], h["X-Image-Width"], h["X-Image-Height"]) == ("4", "250", "75")
    body, h = u16_payload(np.zeros((300, 1000), dtype=np.uint16), scale=8, max_dim=256)
    assert h["X-Image-Scale"] == "8"  # the larger of the two wins
    store = FileStore(tmp_path)
    store.create_session("s1")
    store.put_snapshot("s1", "bf", np.zeros((480, 640), dtype=np.uint16), stem="f9")
    (rec,) = store.list_snapshots("s1", "bf")
    from pathlib import Path

    stem = Path(rec["file_path"]).stem
    c = _client(store, active="s1")
    r = c.get(f"/api/sessions/s1/snapshot/{stem}.u16?max=200")
    assert (r.headers["x-image-scale"], r.headers["x-image-width"]) == ("4", "160")
    r = c.get(f"/api/bf/frames/{stem}.u16?max=100")
    assert (r.headers["x-image-scale"], r.headers["x-image-width"]) == ("7", "91")
