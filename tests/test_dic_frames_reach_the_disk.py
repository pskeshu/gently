"""A DIC overview frame reaches the disk on a real device layer.

A night's run logged 24 DIC frames acquired. None were on disk.

The client swaps each staged file for its pixels and keeps the path on the
result, so the bottom-camera capture reported no path; the run asked
``if image_path`` before filing, and skipped without a word. The staged file
was then cleaned up five minutes later. The tests passed because their fake
camera returned a path, which the real one never did.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pytest
import tifffile

from gently.app.orchestration.timelapse import TimelapseOrchestrator
from gently.app.orchestration.timelapse_models import DicOverview
from gently.core.file_store import FileStore
from gently.hardware.dispim.client import DiSPIMMicroscope
from gently.harness.state import ExperimentState

FRAME = (np.arange(2048 * 2048, dtype=np.uint16) % 4096).reshape(2048, 2048)


@pytest.fixture
def store(tmp_path):
    fs = FileStore(root=tmp_path)
    fs.create_session("s1")
    return fs


def _experiment():
    ex = ExperimentState()
    ex.add_embryo("embryo_1", position={"x": 0.0, "y": 0.0}, calibration={"galvo_center": 0.0})
    return ex


def _rig(capture):
    """A microscope that answers the way the real one does."""
    c = MagicMock()
    c.move_to_position = AsyncMock(return_value={"success": True})
    c.acquire_volume = AsyncMock(return_value={"success": True, "volume": None})
    c.capture_bottom_image = AsyncMock(side_effect=capture)
    return c


async def _one_round(orch):
    msg = await orch.start(
        base_interval_seconds=100, dic=DicOverview(enabled=True, every_seconds=100)
    )
    assert msg.startswith("Started"), msg
    await asyncio.sleep(0.4)
    await orch.stop("test done")


def test_pixels_without_a_path_are_filed(store):
    """What the rig returned every time: an image, and no path."""

    async def capture(use_led=False, exposure_ms=None):
        return {"image": FRAME, "image_path": None}

    orch = TimelapseOrchestrator(_rig(capture), _experiment(), store=store, session_id="s1")
    asyncio.run(_one_round(orch))

    frames = store.list_snapshots("s1", "dic")
    assert len(frames) == 1, "the frame was acquired and never filed"
    rec = frames[0]
    assert rec["metadata"]["frame"] == 1 and rec["metadata"]["channel"] == "dic"
    assert rec["width"] == 2048 and rec["height"] == 2048, "the full frame, not the thumbnail"
    on_disk = tifffile.imread(rec["file_path"])
    assert on_disk.shape == (2048, 2048) and on_disk.dtype == np.uint16
    assert np.array_equal(on_disk, FRAME)


def test_a_staged_file_is_moved_not_copied(store, tmp_path):
    staged = tmp_path / "incoming" / "abc123.tif"
    staged.parent.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(str(staged), FRAME)

    async def capture(use_led=False, exposure_ms=None):
        return {"image": FRAME, "image_path": staged}

    orch = TimelapseOrchestrator(_rig(capture), _experiment(), store=store, session_id="s1")
    asyncio.run(_one_round(orch))
    frames = store.list_snapshots("s1", "dic")
    assert len(frames) == 1 and Path(frames[0]["file_path"]).name == "dic_abc123.tif"
    assert not staged.exists(), "the staged file was left for cleanup to delete"


def test_a_frame_that_cannot_be_filed_says_so(store, caplog):
    """The client's empty-capture placeholder: nothing real to keep."""

    async def capture(use_led=False, exposure_ms=None):
        return {"image": np.zeros((100, 100), dtype=np.uint16), "image_path": None}

    orch = TimelapseOrchestrator(_rig(capture), _experiment(), store=store, session_id="s1")
    with caplog.at_level(logging.WARNING):
        asyncio.run(_one_round(orch))
    assert store.list_snapshots("s1", "dic") == []
    assert any("was NOT filed" in r.message for r in caplog.records), "skipped in silence"


# ── the client ───────────────────────────────────────────────────────────


def _client_after(result):
    c = DiSPIMMicroscope.__new__(DiSPIMMicroscope)
    c.set_bottom_camera_exposure = AsyncMock()
    c.set_camera_led_mode = AsyncMock()
    c._submit_plan_and_wait = AsyncMock(return_value=result)
    return c


def test_the_capture_reports_the_staged_path():
    """_submit_plan_and_wait has already swapped the file ref for its pixels
    and kept the path on the result. That path is the image's."""
    result = {
        "success": True,
        "volume_path": "D:/Gently3/incoming/6624c1aed7ff.tif",
        "documents": {"events": [{"data": {"bottom_camera": FRAME}}]},
    }
    out = asyncio.run(_client_after(result).capture_bottom_image(use_led=True))
    assert out["image"].shape == (2048, 2048)
    assert out["image_path"] == Path("D:/Gently3/incoming/6624c1aed7ff.tif")


def test_an_inline_image_has_no_path():
    result = {"success": True, "documents": {"events": [{"data": {"bottom_camera": FRAME}}]}}
    out = asyncio.run(_client_after(result).capture_bottom_image())
    assert out["image_path"] is None and out["image"].shape == (2048, 2048)


# ── the store ────────────────────────────────────────────────────────────


def test_a_snapshot_filed_from_pixels_reads_like_one_that_was_moved(store):
    path = store.put_snapshot("s1", "dic", FRAME, metadata={"frame": 3})
    assert path.parent.name == "snapshots" and path.name.startswith("dic_")
    rec = store.list_snapshots("s1", "dic")[0]
    assert set(rec) >= {"session_id", "source", "file_path", "metadata", "captured_at"}
    assert rec["width"] == 2048 and rec["height"] == 2048 and rec["metadata"]["frame"] == 3
