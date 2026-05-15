"""Tests for the gently.eval package: capture / replay / shadow."""

from __future__ import annotations

import json
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from gently.core.event_bus import EventBus, EventType, Event
from gently.eval import (
    Decision,
    DecisionLog,
    DecisionTrigger,
    EventCapture,
    EventReplay,
    NoOpCandidate,
    ShadowRunner,
)


# =============================================================================
# EventCapture
# =============================================================================

def test_capture_writes_meaningful_events_skips_telemetry(tmp_path: Path):
    """Capture skips high-volume telemetry by default but keeps the rest."""
    bus = EventBus()
    log = tmp_path / "events.jsonl"
    cap = EventCapture(log)
    cap.start(bus)

    bus.publish(EventType.EMBRYOS_UPDATE, {"a": 1}, source="t")
    bus.publish(EventType.DEVICE_STATE_UPDATE, {"x": 0}, source="t")  # skipped
    bus.publish(EventType.STAGE_MOVED, {"x": 100.0}, source="t")
    bus.publish(EventType.BOTTOM_CAMERA_FRAME, {"jpeg": ""}, source="t")  # skipped
    cap.stop()

    assert cap.count == 2  # the two non-telemetry events
    lines = log.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2
    parsed = [json.loads(ln) for ln in lines]
    assert {p["event_type"] for p in parsed} == {"EMBRYOS_UPDATE", "STAGE_MOVED"}


def test_capture_handles_non_json_native_payloads(tmp_path: Path):
    """numpy scalars / arrays / Paths / datetimes / sets serialise cleanly."""
    bus = EventBus()
    log = tmp_path / "events.jsonl"
    cap = EventCapture(log)
    cap.start(bus)

    bus.publish(EventType.STAGE_MOVED, {
        "np_scalar": np.float64(1.5),
        "np_array": np.array([1, 2, 3]),
        "path": Path("/tmp/foo.tif"),
        "now": datetime(2026, 5, 15, 12, 0, 0),
        "as_set": {"a", "b"},
    }, source="t")
    cap.stop()

    record = json.loads(log.read_text().strip())
    data = record["data"]
    assert data["np_scalar"] == pytest.approx(1.5)
    assert data["np_array"] == [1, 2, 3]
    assert "tmp" in data["path"] and "foo.tif" in data["path"]
    assert data["now"] == "2026-05-15T12:00:00"
    assert sorted(data["as_set"]) == ["a", "b"]


def test_capture_start_stop_idempotent(tmp_path: Path):
    """Repeated start / stop don't error or duplicate subscribers."""
    bus = EventBus()
    cap = EventCapture(tmp_path / "events.jsonl")
    cap.start(bus)
    cap.start(bus)  # second start = no-op
    bus.publish(EventType.STAGE_MOVED, {}, source="t")
    cap.stop()
    cap.stop()  # second stop = no-op
    # Even with two start() calls the single subscription captures the event
    # exactly once.
    assert cap.count == 1


def test_capture_thread_safety(tmp_path: Path):
    """Concurrent publishers from many threads all land in the log."""
    bus = EventBus()
    cap = EventCapture(tmp_path / "events.jsonl")
    cap.start(bus)

    N_THREADS = 8
    N_EVENTS_PER_THREAD = 50

    def worker(idx: int):
        for i in range(N_EVENTS_PER_THREAD):
            bus.publish(EventType.STAGE_MOVED,
                        {"t": idx, "i": i}, source=f"thread-{idx}")

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(N_THREADS)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    cap.stop()
    # All events visible, jsonl is well-formed (each line a valid JSON object).
    assert cap.count == N_THREADS * N_EVENTS_PER_THREAD
    lines = (tmp_path / "events.jsonl").read_text().strip().split("\n")
    assert len(lines) == N_THREADS * N_EVENTS_PER_THREAD
    for ln in lines:
        json.loads(ln)


# =============================================================================
# EventReplay
# =============================================================================

def test_replay_preserves_event_fields(tmp_path: Path):
    """Round-trip: capture, then replay, then verify Event field identity."""
    src_bus = EventBus()
    cap = EventCapture(tmp_path / "events.jsonl")
    cap.start(src_bus)

    e1 = src_bus.publish(EventType.EMBRYOS_UPDATE,
                          {"embryos": [{"id": "e1"}], "count": 1},
                          source="capture-test", correlation_id="corr-A")
    e2 = src_bus.publish(EventType.ERROR_OCCURRED,
                          {"msg": "bang"}, source="capture-test")
    cap.stop()

    rep = EventReplay(tmp_path / "events.jsonl")
    dest = EventBus()
    received: list[Event] = []
    dest.subscribe("*", lambda ev: received.append(ev))
    rep.replay(dest)

    assert len(received) == 2

    # event_type, source, correlation_id, event_id, timestamp preserved
    by_id = {r.event_id: r for r in received}
    r1 = by_id[e1.event_id]
    r2 = by_id[e2.event_id]
    assert r1.event_type == EventType.EMBRYOS_UPDATE
    assert r1.source == "capture-test"
    assert r1.correlation_id == "corr-A"
    assert r1.data == e1.data
    assert r1.timestamp == e1.timestamp
    assert r2.event_type == EventType.ERROR_OCCURRED
    assert r2.timestamp == e2.timestamp


def test_replay_histogram(tmp_path: Path):
    bus = EventBus()
    cap = EventCapture(tmp_path / "events.jsonl")
    cap.start(bus)
    bus.publish(EventType.STAGE_MOVED, {}, source="t")
    bus.publish(EventType.STAGE_MOVED, {}, source="t")
    bus.publish(EventType.EMBRYOS_UPDATE, {}, source="t")
    cap.stop()

    rep = EventReplay(tmp_path / "events.jsonl")
    hist = rep.event_types()
    assert hist == {"STAGE_MOVED": 2, "EMBRYOS_UPDATE": 1}


def test_replay_real_time_respects_cadence(tmp_path: Path):
    """Two events 200 ms apart replay in ~200 ms in real-time mode."""
    bus = EventBus()
    cap = EventCapture(tmp_path / "events.jsonl")
    cap.start(bus)
    bus.publish(EventType.STAGE_MOVED, {"i": 0}, source="t")
    time.sleep(0.2)
    bus.publish(EventType.STAGE_MOVED, {"i": 1}, source="t")
    cap.stop()

    dest = EventBus()
    dest.subscribe("*", lambda ev: None)

    t0 = time.monotonic()
    EventReplay(tmp_path / "events.jsonl").replay(dest, real_time=True)
    elapsed = time.monotonic() - t0
    assert 0.10 < elapsed < 0.40, f"real-time elapsed={elapsed}"


def test_replay_time_scale_speeds_up(tmp_path: Path):
    """time_scale=4 should approximately quarter the real-time wall delay."""
    bus = EventBus()
    cap = EventCapture(tmp_path / "events.jsonl")
    cap.start(bus)
    bus.publish(EventType.STAGE_MOVED, {"i": 0}, source="t")
    time.sleep(0.4)
    bus.publish(EventType.STAGE_MOVED, {"i": 1}, source="t")
    cap.stop()

    dest = EventBus()
    dest.subscribe("*", lambda ev: None)
    t0 = time.monotonic()
    EventReplay(tmp_path / "events.jsonl").replay(
        dest, real_time=True, time_scale=4.0,
    )
    elapsed = time.monotonic() - t0
    # 0.4s scaled by 4 -> ~0.1s, with generous slack for scheduling.
    assert 0.03 < elapsed < 0.30, f"scaled elapsed={elapsed}"


def test_replay_skips_malformed_lines(tmp_path: Path):
    """A garbage line in the log doesn't abort the whole replay."""
    log = tmp_path / "events.jsonl"
    log.write_text(
        json.dumps({
            "event_type": "STAGE_MOVED", "data": {}, "source": "t",
            "timestamp": "2026-01-01T00:00:00", "event_id": "abc",
            "correlation_id": None,
        }) + "\n"
        "not valid json garbage\n"
        + json.dumps({
            "event_type": "EMBRYOS_UPDATE", "data": {}, "source": "t",
            "timestamp": "2026-01-01T00:00:01", "event_id": "def",
            "correlation_id": None,
        }) + "\n",
        encoding="utf-8",
    )
    rep = EventReplay(log)
    seen = list(rep.events())
    assert [s.event_type.name for s in seen] == ["STAGE_MOVED", "EMBRYOS_UPDATE"]


def test_replay_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        EventReplay(tmp_path / "nope.jsonl")


# =============================================================================
# DecisionLog
# =============================================================================

def test_decision_log_round_trip(tmp_path: Path):
    log_path = tmp_path / "decisions.jsonl"
    dlog = DecisionLog(log_path)
    dlog.open()

    d1 = Decision(
        timestamp=datetime(2026, 5, 15, 12, 0, 0),
        agent="prod",
        trigger=DecisionTrigger.USER_MESSAGE,
        trigger_detail="detect embryos",
        tool_calls=[{"name": "detect_embryos", "input": {}}],
        response_text="Detected 4 embryos.",
        context_summary="2 embryos active",
        recent_event_ids=["abc12345"],
        prompt_hash="deadbeef",
        duration_ms=820.5,
    )
    dlog.append(d1)
    dlog.append(Decision(
        timestamp=datetime(2026, 5, 15, 12, 0, 5),
        agent="prod",
        trigger=DecisionTrigger.EVENT,
        trigger_detail="EMBRYOS_UPDATE",
        error=None,
    ))
    dlog.close()

    back = dlog.read()
    assert len(back) == 2
    assert back[0].agent == "prod"
    assert back[0].trigger is DecisionTrigger.USER_MESSAGE
    assert back[0].tool_calls == [{"name": "detect_embryos", "input": {}}]
    assert back[0].duration_ms == pytest.approx(820.5)
    assert back[1].trigger is DecisionTrigger.EVENT
    assert back[1].trigger_detail == "EMBRYOS_UPDATE"


def test_decision_log_context_manager(tmp_path: Path):
    log_path = tmp_path / "decisions.jsonl"
    with DecisionLog(log_path) as dlog:
        dlog.append(Decision(
            timestamp=datetime.now(),
            agent="t",
            trigger=DecisionTrigger.TICK,
        ))
    assert log_path.exists()
    assert len(log_path.read_text().splitlines()) == 1


# =============================================================================
# ShadowRunner + NoOpCandidate
# =============================================================================

def test_shadow_runner_forwards_to_all_candidates(tmp_path: Path):
    bus = EventBus()
    log_a = DecisionLog(tmp_path / "a.jsonl")
    log_b = DecisionLog(tmp_path / "b.jsonl")
    log_a.open()
    log_b.open()

    cand_a = NoOpCandidate("cand-a", log_a)
    cand_b = NoOpCandidate("cand-b", log_b)
    runner = ShadowRunner(bus)
    runner.add(cand_a)
    runner.add(cand_b)
    runner.start()

    bus.publish(EventType.STAGE_MOVED, {"x": 1}, source="t")
    bus.publish(EventType.EMBRYOS_UPDATE, {"count": 1}, source="t")
    runner.stop()
    log_a.close()
    log_b.close()

    decisions_a = log_a.read()
    decisions_b = log_b.read()
    assert len(decisions_a) == 2
    assert len(decisions_b) == 2
    assert [d.trigger_detail for d in decisions_a] == ["STAGE_MOVED", "EMBRYOS_UPDATE"]
    assert [d.agent for d in decisions_a] == ["cand-a", "cand-a"]
    assert [d.agent for d in decisions_b] == ["cand-b", "cand-b"]


def test_shadow_runner_isolates_candidate_failures(tmp_path: Path):
    """A failing candidate doesn't break delivery to its peers."""
    bus = EventBus()
    log_ok = DecisionLog(tmp_path / "ok.jsonl")
    log_ok.open()

    class BoomCandidate(NoOpCandidate):
        def on_event(self, event):
            raise RuntimeError("intentional")

    runner = ShadowRunner(bus)
    runner.add(BoomCandidate("boom", DecisionLog(tmp_path / "boom.jsonl")))
    runner.add(NoOpCandidate("ok", log_ok))
    runner.start()

    bus.publish(EventType.STAGE_MOVED, {}, source="t")
    runner.stop()
    log_ok.close()

    # Production candidate still received the event.
    assert len(log_ok.read()) == 1


def test_shadow_runner_watch_filter(tmp_path: Path):
    """NoOpCandidate(watch=[...]) only fires for matching event types."""
    bus = EventBus()
    dlog = DecisionLog(tmp_path / "d.jsonl")
    dlog.open()
    runner = ShadowRunner(bus)
    runner.add(NoOpCandidate("only-errors", dlog, watch=["ERROR_OCCURRED"]))
    runner.start()

    bus.publish(EventType.STAGE_MOVED, {}, source="t")
    bus.publish(EventType.EMBRYOS_UPDATE, {}, source="t")
    bus.publish(EventType.ERROR_OCCURRED, {"msg": "x"}, source="t")
    runner.stop()
    dlog.close()

    decs = dlog.read()
    assert len(decs) == 1
    assert decs[0].trigger_detail == "ERROR_OCCURRED"
