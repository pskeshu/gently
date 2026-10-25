"""The live light-sheet view, switched from side A to B, drives B's devices.

Three things a side change has to do, none of which the per-value park cache
did on its own:

- park the *new* side's scanner and piezo even when the numbers are the same
  as the old side's (a different device has not been parked yet);
- tag each streamed frame with the side it was grabbed from, so the pane can
  drop a frame of the other side that landed mid-switch;
- carry an armed beam over to the new side and disarm the old one, since two
  sheets on at once put the other sheet across the camera edge-on.
"""

import importlib.util
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

# The side-param test already patches the hardware stack and builds a device
# layer with fakes for both sides; reuse it by file, since tests/ is no package.
_spec = importlib.util.spec_from_file_location(
    "lightsheet_side_param", Path(__file__).with_name("test_lightsheet_side_param.py")
)
assert _spec is not None and _spec.loader is not None
_side_param = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_side_param)
_make_dl = _side_param._make_dl


def _wire_beam(dl):
    """A readable BeamEnabled per side, and enable_beam that writes it."""
    state = {"a": False, "b": False}
    for key, dev_name in dl._BEAM_SIDES.items():
        dev = dl.devices.get(dev_name)
        if dev is None:
            continue
        dev.enable_beam = lambda on, k=key: state.__setitem__(k, bool(on))
    dl._read_beam = lambda: dict(state)
    return state


def test_same_numbers_on_the_other_side_still_park_its_devices():
    dl = _make_dl(with_camera_b=True, with_scanner_b=True, with_piezo_b=True)
    dl._park_lightsheet_sync()  # side A, galvo 0.0 / piezo 50.0
    assert dl.devices["scanner_b"].sa_offset_y.last_pos is None
    assert dl.devices["piezo_b"].last_pos is None

    dl._ls_params["side"] = "B"
    dl._park_lightsheet_sync()  # same numbers, other side

    assert dl.devices["scanner_b"].sa_offset_y.last_pos == 0.0
    assert dl.devices["piezo_b"].last_pos == 50.0
    assert dl._ls_parked["side"] == "B"


@pytest.mark.asyncio
async def test_a_side_change_moves_the_armed_beam_and_leaves_a_dark_scope_dark():
    dl = _make_dl(with_camera_b=True, with_scanner_b=True, with_piezo_b=True)
    state = _wire_beam(dl)

    # Dark scope: a side change arms nothing.
    req = AsyncMock()
    req.json = AsyncMock(return_value={"side": "B"})
    await dl.handle_lightsheet_params(req)
    assert state == {"a": False, "b": False}
    assert dl._ls_params["side"] == "B"

    # A armed, view goes back to A then to B: B ends armed, A not.
    state["a"] = True
    req.json = AsyncMock(return_value={"side": "A"})
    await dl.handle_lightsheet_params(req)
    assert state == {"a": True, "b": False}
    req.json = AsyncMock(return_value={"side": "B"})
    await dl.handle_lightsheet_params(req)
    assert state == {"a": False, "b": True}


@pytest.mark.asyncio
async def test_each_frame_says_which_side_it_came_from(monkeypatch):
    import asyncio

    import numpy as np

    dl = _make_dl(with_camera_b=True, with_scanner_b=True, with_piezo_b=True)
    dl._ls_params["side"] = "B"
    dl._state_pause_counter = 0
    dl._ls_subscribers = [asyncio.Queue(maxsize=4)]
    dl._ls_target_max_dim = 64
    dl._ls_jpeg_quality = 50
    monkeypatch.setattr(dl, "_grab_lightsheet_frame_sync", lambda: np.zeros((8, 8), np.uint16))
    monkeypatch.setattr(dl, "_encode_frame_for_stream", lambda img, d, q: {"jpeg_b64": "x"})
    monkeypatch.setattr(dl, "_stop_lightsheet_sequence_sync", lambda: None)
    monkeypatch.setattr(dl, "_log_focus_trace", lambda *a, **k: None)

    task = asyncio.create_task(dl._lightsheet_streamer())
    payload = await asyncio.wait_for(dl._ls_subscribers[0].get(), timeout=2.0)
    dl._ls_subscribers.clear()
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass

    assert payload["side"] == "B"
