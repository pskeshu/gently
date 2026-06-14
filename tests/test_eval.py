"""Tests for the gently.eval package: capture / replay / shadow."""

from __future__ import annotations

import json
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from gently.core.event_bus import Event, EventBus, EventType
from gently.eval import (
    Decision,
    DecisionLog,
    DecisionTrigger,
    EventCapture,
    EventReplay,
    NoOpCandidate,
    ReactiveCandidate,
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

    bus.publish(
        EventType.STAGE_MOVED,
        {
            "np_scalar": np.float64(1.5),
            "np_array": np.array([1, 2, 3]),
            "path": Path("/tmp/foo.tif"),
            "now": datetime(2026, 5, 15, 12, 0, 0),
            "as_set": {"a", "b"},
        },
        source="t",
    )
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
            bus.publish(EventType.STAGE_MOVED, {"t": idx, "i": i}, source=f"thread-{idx}")

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

    e1 = src_bus.publish(
        EventType.EMBRYOS_UPDATE,
        {"embryos": [{"id": "e1"}], "count": 1},
        source="capture-test",
        correlation_id="corr-A",
    )
    e2 = src_bus.publish(EventType.ERROR_OCCURRED, {"msg": "bang"}, source="capture-test")
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
        dest,
        real_time=True,
        time_scale=4.0,
    )
    elapsed = time.monotonic() - t0
    # 0.4s scaled by 4 -> ~0.1s, with generous slack for scheduling.
    assert 0.03 < elapsed < 0.30, f"scaled elapsed={elapsed}"


def test_replay_skips_malformed_lines(tmp_path: Path):
    """A garbage line in the log doesn't abort the whole replay."""
    log = tmp_path / "events.jsonl"
    log.write_text(
        json.dumps(
            {
                "event_type": "STAGE_MOVED",
                "data": {},
                "source": "t",
                "timestamp": "2026-01-01T00:00:00",
                "event_id": "abc",
                "correlation_id": None,
            }
        )
        + "\n"
        "not valid json garbage\n"
        + json.dumps(
            {
                "event_type": "EMBRYOS_UPDATE",
                "data": {},
                "source": "t",
                "timestamp": "2026-01-01T00:00:01",
                "event_id": "def",
                "correlation_id": None,
            }
        )
        + "\n",
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
    dlog.append(
        Decision(
            timestamp=datetime(2026, 5, 15, 12, 0, 5),
            agent="prod",
            trigger=DecisionTrigger.EVENT,
            trigger_detail="EMBRYOS_UPDATE",
            error=None,
        )
    )
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
        dlog.append(
            Decision(
                timestamp=datetime.now(),
                agent="t",
                trigger=DecisionTrigger.TICK,
            )
        )
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


# =============================================================================
# prompt_hash
# =============================================================================


def test_prompt_hash_stable_and_distinguishing():
    """Identical inputs → identical hash; any change → different hash."""
    from gently.eval import prompt_hash

    h1 = prompt_hash("sys-A", [{"role": "user", "content": "hi"}])
    h2 = prompt_hash("sys-A", [{"role": "user", "content": "hi"}])
    h3 = prompt_hash("sys-A", [{"role": "user", "content": "hello"}])
    h4 = prompt_hash("sys-B", [{"role": "user", "content": "hi"}])
    assert h1 == h2
    assert h1 != h3
    assert h1 != h4
    assert len(h1) == 16  # documented short fingerprint length


def test_prompt_hash_accepts_list_system_prompt():
    """Cached system prompts use the list-of-blocks shape; hashing must work."""
    from gently.eval import prompt_hash

    list_prompt = [{"type": "text", "text": "sys", "cache_control": {"type": "ephemeral"}}]
    str_prompt = "sys"
    # Different shapes, different hashes — that's fine, the point is just
    # that the list case doesn't raise.
    h_list = prompt_hash(list_prompt, [])
    h_str = prompt_hash(str_prompt, [])
    assert isinstance(h_list, str) and isinstance(h_str, str)
    assert len(h_list) == 16 and len(h_str) == 16


# =============================================================================
# ConversationManager production-decision capture (success + error paths)
# =============================================================================


def _make_fake_conversation_manager(claude_client):
    """Build a ConversationManager with a fake Claude client and a no-op
    tool registry — enough to exercise call_claude's decision-write path."""
    import asyncio  # noqa: F401  (used by callers)

    from gently.harness.conversation import ConversationManager

    class _NoopReg:
        # call_claude doesn't use this directly; tools list is passed in
        pass

    return ConversationManager(claude_client, "claude-haiku-4-5-20251001", _NoopReg())


class _Usage:
    input_tokens = 10
    output_tokens = 20
    cache_creation_input_tokens = 0
    cache_read_input_tokens = 0


class _ToolBlock:
    type = "tool_use"
    name = "detect_embryos"
    input = {"min_confidence": 0.7}
    id = "t1"


class _TextBlock:
    type = "text"
    text = "Done."


class _R1:
    stop_reason = "tool_use"
    content = [_ToolBlock()]
    usage = _Usage()


class _R2:
    stop_reason = "end_turn"
    content = [_TextBlock()]
    usage = _Usage()


def test_production_decision_capture_success(tmp_path: Path):
    """One success turn through call_claude writes one Decision row."""
    import asyncio

    calls = {"n": 0}

    class _FakeMessages:
        def create(self, **kw):
            calls["n"] += 1
            return _R1() if calls["n"] == 1 else _R2()

    class _FakeClient:
        messages = _FakeMessages()

    cm = _make_fake_conversation_manager(_FakeClient())

    # Bypass actual tool execution
    async def fake_exec(content_blocks, interaction):
        return [{"type": "tool_result", "tool_use_id": "t1", "content": "ok"}]

    cm._execute_tools_with_logging = fake_exec

    dlog = DecisionLog(tmp_path / "decisions.jsonl")
    dlog.open()
    cm.decision_log = dlog

    async def run():
        return await cm.call_claude(
            user_message="find embryos please",
            system_prompt="system",
            tools=[],
            mode="run",
            auto_save_fn=lambda: None,
        )

    out = asyncio.run(run())
    dlog.close()

    assert out == "Done."
    decs = dlog.read()
    assert len(decs) == 1
    d = decs[0]
    assert d.agent == "production"
    assert d.trigger is DecisionTrigger.USER_MESSAGE
    assert d.trigger_detail == "find embryos please"
    assert d.tool_calls == [
        {
            "name": "detect_embryos",
            "input": {"min_confidence": 0.7},
            "id": "t1",
        }
    ]
    assert d.response_text == "Done."
    assert d.error is None
    assert d.prompt_hash is not None and len(d.prompt_hash) == 16
    assert d.duration_ms is not None and d.duration_ms >= 0


def test_production_decision_capture_error(tmp_path: Path):
    """A failing Claude call writes a Decision with error before re-raising."""
    import asyncio

    class _BoomMessages:
        def create(self, **kw):
            raise RuntimeError("simulated outage")

    class _BoomClient:
        messages = _BoomMessages()

    cm = _make_fake_conversation_manager(_BoomClient())
    dlog = DecisionLog(tmp_path / "decisions.jsonl")
    dlog.open()
    cm.decision_log = dlog

    async def run():
        with pytest.raises(RuntimeError, match="simulated outage"):
            await cm.call_claude(
                user_message="do something",
                system_prompt="system",
                tools=[],
                mode="run",
                auto_save_fn=lambda: None,
            )

    asyncio.run(run())
    dlog.close()
    decs = dlog.read()
    assert len(decs) == 1
    assert decs[0].error == "simulated outage"
    assert decs[0].trigger is DecisionTrigger.USER_MESSAGE
    assert decs[0].response_text and "simulated outage" in decs[0].response_text


def test_production_decision_capture_no_log_is_no_op(tmp_path: Path):
    """No DecisionLog attached → call_claude proceeds normally, no errors."""
    import asyncio

    calls = {"n": 0}

    class _M:
        def create(self, **kw):
            calls["n"] += 1
            return _R2()  # immediate end_turn, no tool loop

    class _C:
        messages = _M()

    cm = _make_fake_conversation_manager(_C())
    assert cm.decision_log is None  # default

    async def run():
        return await cm.call_claude(
            user_message="hi",
            system_prompt="sys",
            tools=[],
            mode="run",
            auto_save_fn=lambda: None,
        )

    out = asyncio.run(run())
    assert out == "Done."  # no log to read; we just want no error


# =============================================================================
# ReactiveCandidate
# =============================================================================


def _publish(bus, event_type_name: str, data: dict, source: str = "test"):
    bus.publish(EventType[event_type_name], data, source=source)


def _decisions_for(dlog: DecisionLog):
    return dlog.read()


def test_reactive_ingests_embryos_update_silently(tmp_path: Path):
    """EMBRYOS_UPDATE updates the world model but emits no decision."""
    bus = EventBus()
    dlog = DecisionLog(tmp_path / "d.jsonl")
    dlog.open()
    cand = ReactiveCandidate("reactive-test", dlog)
    runner = ShadowRunner(bus)
    runner.add(cand)
    runner.start()

    _publish(
        bus,
        "EMBRYOS_UPDATE",
        {
            "embryos": [
                {
                    "id": "embryo_1",
                    "position_coarse": {"x": 1.0, "y": 2.0},
                    "position_fine": {},
                    "has_fine_position": False,
                },
                {
                    "id": "embryo_2",
                    "position_coarse": {"x": 3.0, "y": 4.0},
                    "position_fine": {"x": 3.1, "y": 4.1},
                    "has_fine_position": True,
                },
            ]
        },
    )

    runner.stop()
    dlog.close()

    assert _decisions_for(dlog) == []  # silent
    assert set(cand.world.embryos.keys()) == {"embryo_1", "embryo_2"}
    assert cand.world.embryos["embryo_2"]["has_fine"]


def test_reactive_proposes_recalibrate_when_fine_invalidated(tmp_path: Path):
    bus = EventBus()
    dlog = DecisionLog(tmp_path / "d.jsonl")
    dlog.open()
    runner = ShadowRunner(bus)
    runner.add(ReactiveCandidate("reactive", dlog))
    runner.start()

    _publish(
        bus,
        "OPERATOR_EDITED_EMBRYO",
        {
            "embryo_id": "embryo_2",
            "old_position_coarse": {"x": 3, "y": 4},
            "new_position_coarse": {"x": 30, "y": 40},
            "fine_position_invalidated": True,
        },
    )
    runner.stop()
    dlog.close()
    decs = _decisions_for(dlog)
    assert len(decs) == 1
    assert decs[0].tool_calls == [
        {
            "name": "recalibrate_embryo",
            "input": {"embryo_id": "embryo_2"},
            "id": None,
        }
    ]


def test_reactive_skips_recalibrate_when_no_fine_existed(tmp_path: Path):
    bus = EventBus()
    dlog = DecisionLog(tmp_path / "d.jsonl")
    dlog.open()
    runner = ShadowRunner(bus)
    runner.add(ReactiveCandidate("reactive", dlog))
    runner.start()

    _publish(
        bus,
        "OPERATOR_EDITED_EMBRYO",
        {
            "embryo_id": "embryo_1",
            "old_position_coarse": {"x": 1, "y": 2},
            "new_position_coarse": {"x": 10, "y": 20},
            "fine_position_invalidated": False,
        },
    )
    runner.stop()
    dlog.close()
    decs = _decisions_for(dlog)
    assert len(decs) == 1
    assert decs[0].tool_calls == []  # nothing to refresh
    assert "no action" in decs[0].response_text.lower()


def test_reactive_proposes_calibrate_all_on_marked(tmp_path: Path):
    bus = EventBus()
    dlog = DecisionLog(tmp_path / "d.jsonl")
    dlog.open()
    runner = ShadowRunner(bus)
    runner.add(ReactiveCandidate("reactive", dlog))
    runner.start()

    _publish(
        bus,
        "OPERATOR_MARKED_EMBRYOS",
        {
            "embryo_ids": ["embryo_001", "embryo_002", "embryo_003"],
            "count": 3,
        },
    )
    runner.stop()
    dlog.close()
    decs = _decisions_for(dlog)
    assert len(decs) == 1
    assert decs[0].tool_calls == [
        {
            "name": "calibrate_all_embryos",
            "input": {"embryo_ids": ["embryo_001", "embryo_002", "embryo_003"]},
            "id": None,
        }
    ]


def test_reactive_proposes_forget_on_removal(tmp_path: Path):
    bus = EventBus()
    dlog = DecisionLog(tmp_path / "d.jsonl")
    dlog.open()
    runner = ShadowRunner(bus)
    runner.add(ReactiveCandidate("reactive", dlog))
    runner.start()

    _publish(
        bus,
        "OPERATOR_REMOVED_EMBRYO",
        {
            "embryo_id": "embryo_5",
            "last_position": {"coarse": {"x": 1, "y": 2}, "fine": None},
        },
    )
    runner.stop()
    dlog.close()
    decs = _decisions_for(dlog)
    assert len(decs) == 1
    assert decs[0].tool_calls == [
        {
            "name": "forget_embryo",
            "input": {"embryo_id": "embryo_5"},
            "id": None,
        }
    ]


def test_reactive_escalates_first_error_then_suppresses_repeat(tmp_path: Path):
    bus = EventBus()
    dlog = DecisionLog(tmp_path / "d.jsonl")
    dlog.open()
    runner = ShadowRunner(bus)
    runner.add(ReactiveCandidate("reactive", dlog))
    runner.start()

    _publish(bus, "ERROR_OCCURRED", {"msg": "camera lost lock"})
    _publish(bus, "ERROR_OCCURRED", {"msg": "camera lost lock"})  # within 30s
    _publish(bus, "ERROR_OCCURRED", {"msg": "different error"})
    runner.stop()
    dlog.close()
    decs = _decisions_for(dlog)
    assert len(decs) == 3
    # 1st: escalate
    assert decs[0].tool_calls[0]["name"] == "escalate_to_operator"
    # 2nd: suppressed
    assert decs[1].tool_calls == []
    assert "suppressed" in decs[1].response_text.lower()
    # 3rd: different message -> escalate
    assert decs[2].tool_calls[0]["name"] == "escalate_to_operator"


def test_reactive_full_event_stream_through_replay(tmp_path: Path):
    """Capture a realistic operator-driven session and replay through the
    candidate. End-to-end smoke that the recorded jsonl is enough to
    drive a candidate's decision log without any other inputs."""
    src = EventBus()
    cap = EventCapture(tmp_path / "events.jsonl")
    cap.start(src)

    src.publish(
        EventType.EMBRYOS_UPDATE,
        {
            "embryos": [
                {
                    "id": "embryo_1",
                    "position_coarse": {"x": 1.0, "y": 2.0},
                    "position_fine": {"x": 1.05, "y": 2.05},
                    "has_fine_position": True,
                },
            ]
        },
        source="agent",
    )
    src.publish(
        EventType.OPERATOR_EDITED_EMBRYO,
        {
            "embryo_id": "embryo_1",
            "old_position_coarse": {"x": 1.0, "y": 2.0},
            "new_position_coarse": {"x": 5.0, "y": 6.0},
            "fine_position_invalidated": True,
        },
        source="web:map-edit",
    )
    src.publish(
        EventType.OPERATOR_MARKED_EMBRYOS,
        {
            "embryo_ids": ["embryo_2", "embryo_3"],
            "count": 2,
        },
        source="detect_embryos:web-editor",
    )
    src.publish(EventType.ERROR_OCCURRED, {"msg": "lost focus"}, source="device-layer")
    cap.stop()

    dst = EventBus()
    dlog = DecisionLog(tmp_path / "decisions.jsonl")
    dlog.open()
    runner = ShadowRunner(dst)
    runner.add(ReactiveCandidate("replay-cand", dlog))
    runner.start()

    n = EventReplay(tmp_path / "events.jsonl").replay(dst)
    runner.stop()
    dlog.close()

    assert n == 4  # all four events replayed
    decs = _decisions_for(dlog)
    # EMBRYOS_UPDATE is silent ingest; the other 3 each produce a decision.
    triggers = [(d.trigger.value, d.trigger_detail) for d in decs]
    assert triggers == [
        ("event", "OPERATOR_EDITED_EMBRYO"),
        ("event", "OPERATOR_MARKED_EMBRYOS"),
        ("event", "ERROR_OCCURRED"),
    ]
    tool_names = [d.tool_calls[0]["name"] if d.tool_calls else None for d in decs]
    assert tool_names == [
        "recalibrate_embryo",
        "calibrate_all_embryos",
        "escalate_to_operator",
    ]
