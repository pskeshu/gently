"""The map is told what a pixel is at the sample, never the camera's own pixel.

2026-10-21, session ce91600d: the agent detected four embryos and handed the
map the image with ``pixel_size_um=6.5``, the camera's physical pixel. The
map scales every mark by that number; the bottom camera's 10× objective makes
a pixel 0.65 µm at the sample. Every registered position was ten times too
far from where the frame was taken, and the stage drove to them.

Recorded in that session's labelled snapshot: a mark at pixel (1035.5, 1023.5)
of a 2048×2048 frame taken at stage (60.22, −210.449) was filed at
(134.97, −207.199). It is at (67.70, −210.12).
"""

from __future__ import annotations

import asyncio
import inspect
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from gently.app.tools import detection_tools
from gently.core.coordinates import DEFAULT_OBJECTIVE_MAG, DEFAULT_PIXEL_SIZE_UM, get_um_per_pixel
from gently.ui.web import server as web_server

JS = Path(__file__).resolve().parents[1] / "gently" / "ui" / "web" / "static" / "js"


def test_the_hand_off_carries_the_sample_plane_pixel():
    sig = inspect.signature(detection_tools._route_to_map_view)
    assert "pixel_size_um" not in sig.parameters, "the camera's pixel has no business here"
    default = sig.parameters["um_per_px"].default
    assert default == pytest.approx(0.65)
    assert default == get_um_per_pixel(DEFAULT_PIXEL_SIZE_UM, DEFAULT_OBJECTIVE_MAG)
    server_default = inspect.signature(web_server.VisualizationServer.start_marking_session)
    assert server_default.parameters["pixel_size_um"].default == pytest.approx(default)


def test_the_hand_off_refuses_a_camera_pixel():
    agent = MagicMock()
    agent.viz_server = object()
    with pytest.raises(ValueError, match="sample-plane"):
        asyncio.run(
            detection_tools._route_to_map_view(
                agent=agent,
                image=np.zeros((4, 4), dtype=np.uint8),
                initial_markers=[],
                stage_position=(0.0, 0.0),
                um_per_px=DEFAULT_PIXEL_SIZE_UM,
            )
        )


def test_the_server_refuses_to_start_a_session_the_stage_would_follow_wrongly():
    with pytest.raises(ValueError, match="sample-plane"):
        asyncio.run(
            web_server.VisualizationServer.start_marking_session(
                MagicMock(), np.zeros((4, 4), dtype=np.uint8), pixel_size_um=6.5
            )
        )


def test_the_recorded_mark_lands_where_the_embryo_is():
    """The pane's conversion (operate-math frameToStage) with the right scale."""
    u = get_um_per_pixel(DEFAULT_PIXEL_SIZE_UM, DEFAULT_OBJECTIVE_MAG)
    cx, cy = 60.22, -210.449
    px, py = 1035.5, 1023.5
    sx, sy = cx + (px - 1024) * u, cy - (py - 1024) * u
    assert (sx, sy) == (pytest.approx(67.70, abs=0.01), pytest.approx(-210.12, abs=0.01))
    # and the wrong scale is exactly what was filed
    wx, wy = cx + (px - 1024) * 6.5, cy - (py - 1024) * 6.5
    assert (wx, wy) == (pytest.approx(134.97, abs=0.01), pytest.approx(-207.199, abs=0.01))


def test_the_pane_ignores_a_scale_no_objective_could_give():
    operate = (JS / "operate.js").read_text(encoding="utf-8")
    assert "scale >= 0.01 && scale <= 5" in operate
    assert "pixelSizeUm: sane ? scale : undefined" in operate
