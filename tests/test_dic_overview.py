"""The DIC overview: one frame of the whole field per round, on its own clock.

The second channel of a Gently timelapse. The bottom camera's field covers
every embryo, so one frame a round records all of them; and because embryo
cadences are independent and adaptive, "a timepoint" is not a global thing
— so the overview is scheduled as a subject of its own rather than hung off
any embryo's acquisition.

Everything here runs against a fake microscope client. The loop is the real
one, driven with intervals short enough to watch several rounds in under a
second.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from gently.app.orchestration.timelapse import TimelapseOrchestrator
from gently.app.orchestration.timelapse_models import DicOverview
from gently.harness.state import ExperimentState


@pytest.fixture(autouse=True, scope="module")
def _organism():
    """A stage-based stop ("hatching") is looked up in the organism's stage
    table, which nothing loads in a bare test process."""
    from gently.organisms import load_organism

    load_organism("celegans")


POSITIONS = {
    "embryo_1": {"x": -800.0, "y": -600.0},
    "embryo_2": {"x": -200.0, "y": -600.0},
    "embryo_3": {"x": -500.0, "y": 0.0},
}
CENTROID = {"x": -500.0, "y": -400.0}


def _client(tmp_path: Path):
    """A microscope that answers every call at once."""
    c = MagicMock()
    c.move_to_position = AsyncMock(return_value={"success": True})
    c.acquire_volume = AsyncMock(return_value={"success": True, "volume": None})
    c.capture_lightsheet_image = AsyncMock(return_value={"success": True, "image": None})
    # The overview's light: already on, so these tests are about the frame and
    # not about switching (tests/test_dic_light.py is about switching).
    c.get_room_light_status = AsyncMock(return_value={"success": True, "state": "on"})
    c.set_room_light = AsyncMock(return_value={"success": True})
    c.set_led = AsyncMock(return_value={"success": True})

    frames = {"n": 0}

    async def capture_bottom_image(use_led=False, exposure_ms=None):
        import numpy as np

        frames["n"] += 1
        p = tmp_path / "incoming" / f"dic{frames['n']}.tif"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"II*\0")  # enough to exist; nothing reads it
        # A frame the size of the real camera's, so the thumbnail path runs.
        return {
            "image": (np.arange(2048 * 2048, dtype=np.uint16) % 4096).reshape(2048, 2048),
            "image_path": p,
        }

    c.capture_bottom_image = AsyncMock(side_effect=capture_bottom_image)
    return c


def _store(tmp_path: Path):
    st = MagicMock()
    st.root = tmp_path
    st._session_dir = MagicMock(return_value=tmp_path / "sessions" / "s1")
    (tmp_path / "sessions" / "s1").mkdir(parents=True, exist_ok=True)
    st.cleanup_incoming = MagicMock()

    def register_snapshot(session_id, source, incoming_path, metadata=None):
        out = (
            tmp_path
            / "sessions"
            / session_id
            / "snapshots"
            / f"{source}_{Path(incoming_path).stem}.tif"
        )
        out.parent.mkdir(parents=True, exist_ok=True)
        Path(incoming_path).rename(out)
        return out

    st.register_snapshot = MagicMock(side_effect=register_snapshot)
    return st


def _experiment(calibrated=True):
    ex = ExperimentState()
    for eid, pos in POSITIONS.items():
        ex.add_embryo(
            eid, position=dict(pos), calibration={"galvo_center": 0.0} if calibrated else None
        )
    return ex


def _orchestrator(tmp_path, *, session_id="s1", store=True):
    return TimelapseOrchestrator(
        _client(tmp_path),
        _experiment(),
        store=_store(tmp_path) if store else None,
        session_id=session_id,
    )


async def _run(orch, seconds, **start_kwargs):
    msg = await orch.start(base_interval_seconds=0.15, **start_kwargs)
    assert msg.startswith("Started"), msg
    await asyncio.sleep(seconds)
    await orch.stop("test done")
    return msg


def _calls(orch, name):
    return [c for c in orch.client.method_calls if c[0] == name]


# ---------------------------------------------------------------------------


def test_without_a_dic_channel_nothing_touches_the_bottom_camera(tmp_path):
    orch = _orchestrator(tmp_path)
    asyncio.run(_run(orch, 0.5))
    assert not _calls(orch, "capture_bottom_image")
    assert len(_calls(orch, "acquire_volume")) >= 3, "the volumes still ran"
    assert orch.get_status().dic is None


def test_the_overview_goes_first_and_is_taken_from_the_centroid(tmp_path):
    # The first frame is due at t0, ahead of the first embryo, so the series
    # starts where the volumes do; and absent a pinned spot it is taken from
    # the centroid of the subjects — the one place that is the same for all.
    orch = _orchestrator(tmp_path)
    asyncio.run(_run(orch, 0.5, dic=DicOverview(enabled=True, every_seconds=100)))
    names = [c[0] for c in orch.client.method_calls]
    first_capture = names.index("capture_bottom_image")
    first_volume = names.index("acquire_volume")
    assert first_capture < first_volume, names[:6]
    move_before = [
        c for c in orch.client.method_calls[:first_capture] if c[0] == "move_to_position"
    ]
    assert move_before and move_before[-1][1] == (CENTROID["x"], CENTROID["y"])


def test_one_frame_per_its_own_interval_not_per_embryo(tmp_path):
    # Three embryos at 0.15 s each; the overview every 100 s. Over half a
    # second the embryos are imaged many times and the field exactly once.
    orch = _orchestrator(tmp_path)
    asyncio.run(_run(orch, 0.6, dic={"enabled": True, "every_seconds": 100}))
    assert len(_calls(orch, "capture_bottom_image")) == 1
    assert len(_calls(orch, "acquire_volume")) >= 4
    assert orch.get_status().dic["frames"] == 1


def test_the_overview_keeps_its_own_clock(tmp_path):
    # Every 0.2 s over ~0.65 s: three or four frames, and never one per volume.
    orch = _orchestrator(tmp_path)
    asyncio.run(_run(orch, 0.65, dic=DicOverview(enabled=True, every_seconds=0.2)))
    frames = len(_calls(orch, "capture_bottom_image"))
    volumes = len(_calls(orch, "acquire_volume"))
    assert 2 <= frames <= 5, frames
    assert volumes > frames, (volumes, frames)


def test_a_bare_timepoints_stop_with_a_count_is_a_count_not_manual(tmp_path):
    # The route documents stop_condition="timepoints" + condition_value; only
    # "timepoints:N" used to parse, and the bare word became "manual".
    orch = _orchestrator(tmp_path)
    asyncio.run(_run(orch, 0.2, stop_condition="timepoints", condition_value=12))
    sc = orch.experiment.embryos["embryo_1"].stop_condition
    assert sc.condition_type.value == "fixed_timepoints" and sc.value == 12


def test_a_pinned_position_wins_over_the_centroid(tmp_path):
    orch = _orchestrator(tmp_path)
    asyncio.run(
        _run(
            orch,
            0.3,
            dic=DicOverview(enabled=True, every_seconds=100, position={"x": 1.0, "y": 2.0}),
        )
    )
    names = [c[0] for c in orch.client.method_calls]
    at = names.index("capture_bottom_image")
    moves = [c for c in orch.client.method_calls[:at] if c[0] == "move_to_position"]
    assert moves[-1][1] == (1.0, 2.0)


def test_frames_are_filed_beside_the_sessions_snapshots(tmp_path):
    orch = _orchestrator(tmp_path)
    asyncio.run(_run(orch, 0.3, dic=DicOverview(enabled=True, every_seconds=100, exposure_ms=8.0)))
    st = orch._store
    assert st.register_snapshot.call_count == 1
    args, kwargs = st.register_snapshot.call_args
    assert args[0] == "s1" and args[1] == "dic"
    meta = kwargs["metadata"]
    assert meta["channel"] == "dic" and meta["frame"] == 1 and meta["exposure_ms"] == 8.0
    assert (tmp_path / "sessions" / "s1" / "snapshots" / "dic_dic1.tif").exists()


def test_a_failed_frame_does_not_take_the_volumes_down(tmp_path):
    orch = _orchestrator(tmp_path)
    orch.client.capture_bottom_image = AsyncMock(side_effect=RuntimeError("camera busy"))
    asyncio.run(_run(orch, 0.5, dic=DicOverview(enabled=True, every_seconds=0.1)))
    assert len(_calls(orch, "acquire_volume")) >= 3, "the experiment carried on"
    assert orch.get_status().dic["frames"] == 0
    assert orch.get_status().status.value in ("stopped", "completed", "idle")


def test_per_embryo_termination_overrides_the_runs_default(tmp_path):
    # "embryo 2 at hatching, the rest after 12 timepoints"
    orch = _orchestrator(tmp_path)
    asyncio.run(
        _run(
            orch,
            0.2,
            stop_condition="timepoints",
            condition_value=12,
            stop_conditions={
                "embryo_2": "hatching",
                "embryo_3": {"stop_condition": "timepoints", "condition_value": 3},
            },
        )
    )
    ex = orch.experiment.embryos
    assert ex["embryo_1"].stop_condition.describe() != ex["embryo_2"].stop_condition.describe()
    assert "hatch" in ex["embryo_2"].stop_condition.describe().lower()
    assert ex["embryo_3"].stop_condition.value == 3


def test_an_override_for_an_embryo_not_in_the_run_is_reported_not_applied(tmp_path, caplog):
    orch = _orchestrator(tmp_path)
    asyncio.run(_run(orch, 0.2, stop_conditions={"embryo_9": "hatching"}))
    assert "embryo_9" in caplog.text


def test_the_overview_survives_a_restart(tmp_path):
    # timelapse.yaml carries the channel, its clock and its frame count, so a
    # resumed run does not start a second series at frame 1.
    orch = _orchestrator(tmp_path)
    asyncio.run(
        _run(
            orch,
            0.3,
            dic=DicOverview(enabled=True, every_seconds=100, position={"x": 1.0, "y": 2.0}),
        )
    )
    doc = orch._serialize_runtime_state()
    assert doc["dic"]["enabled"] is True and doc["dic"]["position"] == {"x": 1.0, "y": 2.0}
    assert doc["dic_frames"] == 1

    fresh = _orchestrator(tmp_path)
    fresh._apply_runtime_state(doc)
    assert fresh._dic is not None and fresh._dic.position == {"x": 1.0, "y": 2.0}
    assert fresh._dic_frames == 1
    assert fresh._dic_next_due_at is not None


def test_the_trace_dir_follows_the_storage_root_at_call_time(tmp_path, monkeypatch):
    # TRACE_BASE_PATH was resolved at import, so it stayed the rig's real
    # traces dir through any redirect of the storage root — the same class
    # of bug that wrote a fixture's region into D:\\Gently3.
    from gently.settings import settings

    orch = _orchestrator(tmp_path)
    asyncio.run(_run(orch, 0.1))
    assert orch._trace_dir is not None
    assert str(orch._trace_dir).startswith(str(settings.storage.base_path)), orch._trace_dir
    assert "Gently3" not in str(orch._trace_dir)


def test_dic_overview_round_trips_through_a_dict():
    d = DicOverview(enabled=True, every_seconds=300.0, position={"x": 1, "y": 2}, exposure_ms=5.0)
    assert DicOverview.from_dict(d.to_dict()) == d
    assert DicOverview.from_dict(None).enabled is False
    assert (
        DicOverview.from_dict({"enabled": True, "position": {"x": None, "y": 3}}).position is None
    )


@pytest.mark.parametrize("bad", [{"enabled": True, "every_seconds": "soon"}])
def test_dic_overview_rejects_a_non_number_interval(bad):
    with pytest.raises((TypeError, ValueError)):
        DicOverview.from_dict(bad)


def test_the_frame_event_carries_a_thumbnail_the_tab_can_show(tmp_path):
    # The Embryos tab shows the frame as it lands. The full frame is filed;
    # what rides on the event is small enough to send every round.
    from gently.core import EventType, get_event_bus

    seen = []
    bus = get_event_bus()
    bus.subscribe(EventType.IMAGE_ACQUIRED, lambda ev: seen.append(ev))
    orch = _orchestrator(tmp_path)
    asyncio.run(_run(orch, 0.3, dic=DicOverview(enabled=True, every_seconds=100)))
    dic = [e for e in seen if (getattr(e, "data", None) or {}).get("source") == "dic"]
    assert dic, "no IMAGE_ACQUIRED for the overview"
    data = dic[0].data
    assert data["frame"] == 1 and data["embryo_id"] is None
    assert isinstance(data["image_b64"], str) and len(data["image_b64"]) > 100
    assert len(data["image_b64"]) < 400_000, "that is not a thumbnail"
