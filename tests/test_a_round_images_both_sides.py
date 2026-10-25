"""A round on sides A and B: one timepoint, two volumes, each with its own numbers.

The orchestrator's per-embryo acquisition takes the run's sides in order. Side
A images with the embryo's calibration as it always has; side B with the
embryo's ``calibration["B"]``, the other scanner and piezo's own numbers. An
embryo that has no B calibration is imaged on A only, and the run says so
once. Side A is the round's verdict and the perceiver's input; B is filed
beside it, under ``volumes/B``.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import numpy as np

from gently.app.orchestration.timelapse import TimelapseOrchestrator
from gently.harness.state import ExperimentState

CAL_A = {"galvo_center": 0.3, "galvo_amplitude": 0.4, "piezo_center": 62.0, "piezo_amplitude": 18.0}
CAL_B = {
    "galvo_center": -0.2,
    "galvo_amplitude": 0.4,
    "piezo_center": 41.0,
    "piezo_amplitude": 18.0,
}


def _client():
    c = MagicMock()
    c.move_to_position = AsyncMock(return_value={"success": True})
    c.acquire_volume = AsyncMock(
        return_value={"success": True, "volume": np.zeros((3, 4, 4), np.uint16)}
    )
    return c


def _orch(sides, with_b=True):
    ex = ExperimentState()
    cal = dict(CAL_A)
    if with_b:
        cal["B"] = dict(CAL_B)
    ex.add_embryo("embryo_1", position={"x": -200.0, "y": -600.0}, calibration=cal)
    orch = TimelapseOrchestrator(_client(), ex)
    orch._sides = orch._normalise_sides(sides)
    calls: list[dict] = []

    async def cb(embryo_id, timepoint, data, volume_path=None, **extra):
        calls.append({"embryo_id": embryo_id, "t": timepoint, **extra})
        return {"volume_uid": "v"}

    orch.on_volume_callback = cb
    return orch, ex.embryos["embryo_1"], calls


def test_sides_are_normalised():
    n = TimelapseOrchestrator._normalise_sides
    assert n(None) == ("A",)
    assert n(["A", "B"]) == ("A", "B")
    assert n("b") == ("B",)
    assert n(["B", "A", "B", "x"]) == ("B", "A")


def test_a_round_on_both_sides_is_one_timepoint_with_each_side_s_numbers():
    orch, embryo, calls = _orch(["A", "B"])
    asyncio.run(orch._acquire_embryo(embryo))

    sides = [c.kwargs["side"] for c in orch.client.acquire_volume.await_args_list]
    assert sides == ["A", "B"]
    a, b = (c.kwargs for c in orch.client.acquire_volume.await_args_list)
    assert (a["piezo_center"], a["galvo_center"]) == (62.0, 0.3)
    assert (b["piezo_center"], b["galvo_center"]) == (41.0, -0.2)
    assert embryo.timepoints_acquired == 1
    # The callback saw both, A without a side keyword, B with one.
    assert [c.get("side") for c in calls] == [None, "B"]
    assert all(c["t"] == 1 for c in calls)
    # Two sheets were on: the dose ledger counted both.
    assert embryo.exposure_count == 2


def test_an_embryo_without_a_b_calibration_is_imaged_on_a_only(caplog):
    orch, embryo, calls = _orch(["A", "B"], with_b=False)
    with caplog.at_level("WARNING"):
        asyncio.run(orch._acquire_embryo(embryo))
        asyncio.run(orch._acquire_embryo(embryo))
    sides = [c.kwargs["side"] for c in orch.client.acquire_volume.await_args_list]
    assert sides == ["A", "A"]
    assert sum("no side-B calibration" in r.message for r in caplog.records) == 1
    assert embryo.timepoints_acquired == 2


def test_a_run_on_a_alone_is_unchanged():
    orch, embryo, calls = _orch(["A"])
    asyncio.run(orch._acquire_embryo(embryo))
    assert [c.kwargs["side"] for c in orch.client.acquire_volume.await_args_list] == ["A"]
    assert calls == [{"embryo_id": "embryo_1", "t": 1}]


def test_the_checkpoint_keeps_the_sides():
    orch, _, _ = _orch(["A", "B"])
    assert orch._serialize_runtime_state()["sides"] == ["A", "B"]


def test_restoring_a_checkpoint_restores_the_sides():
    orch, _, _ = _orch(["A"])
    orch._apply_runtime_state({"sides": ["A", "B"]})
    assert orch._sides == ("A", "B")
    orch._apply_runtime_state({})
    assert orch._sides == ("A",)


def test_side_b_volumes_live_beside_a_on_disk(tmp_path):
    from gently.core.file_store import FileStore

    store = FileStore(tmp_path / "g")
    store.create_session("s1")
    vol = np.zeros((2, 4, 4), np.uint16)
    a = store.put_volume("s1", "embryo_1", 1, vol)
    b = store.put_volume("s1", "embryo_1", 1, vol, side="B")
    assert a.parent.name == "volumes" and a.name == "t0001.tif"
    assert b.parent.name == "B" and b.parent.parent.name == "volumes" and b.name == "t0001.tif"
    assert (b.parent / "t0001.meta.yaml").read_text().find("side: B") >= 0
    assert (a.parent / "t0001.meta.yaml").read_text().find("side: A") >= 0
    assert (a.parent.parent / "projections" / "t0001.jpg").exists()
    assert (a.parent.parent / "projections" / "B" / "t0001.jpg").exists()
    # The readers that know one view still find A and nothing else.
    assert store.list_projection_timepoints("s1", "embryo_1") == [1]
    assert store.get_volume_path("s1", "embryo_1", 1) == a
