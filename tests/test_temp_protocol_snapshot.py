"""Task 6: temp-protocol band + setpoint changes visible in strategy_snapshot.

Feed _replay_timeline (via build_strategy_snapshot over a hand-written
timeline.jsonl) a sequence:
  temp_protocol_started → setpoint_changed(to=25) →
  burst_started → burst_completed → temp_protocol_completed

Assert the returned snapshot exposes:
  embryos[0]["temp_protocol"]  — a band with start < end, correct params
  embryos[0]["setpoint_changes"]  — [{t: ..., to: 25}]
"""

import json
from datetime import datetime, timedelta
from pathlib import Path

import pytest
import yaml

from gently.ui.web.strategy_snapshot import build_strategy_snapshot

SESSION_ID = "sess-tp1"
EMBRYO_ID = "e1"
STARTED_AT = datetime(2025, 1, 1, 10, 0, 0)
TARGET_C = 25.0


def _ts(delta_s: float) -> str:
    return (STARTED_AT + timedelta(seconds=delta_s)).isoformat()


def _write_session(session_dir: Path) -> None:
    session_dir.mkdir(parents=True, exist_ok=True)

    # Minimal session.yaml
    (session_dir / "session.yaml").write_text(
        yaml.dump({"name": "temp-protocol test"}), encoding="utf-8"
    )

    # timelapse.yaml — minimal; one embryo
    timelapse = {
        "started_at": STARTED_AT.isoformat(),
        "base_interval_seconds": 120,
        "embryos": {
            EMBRYO_ID: {"interval_seconds": 120},
        },
    }
    (session_dir / "timelapse.yaml").write_text(yaml.dump(timelapse), encoding="utf-8")

    # timeline.jsonl — the sequence under test
    events = [
        {
            "event_id": "ev-tps",
            "event_type": "tactic",
            "event_subtype": "temp_protocol_started",
            "timestamp": _ts(10),
            "source": "test",
            "session_id": SESSION_ID,
            "embryo_id": EMBRYO_ID,
            "data": {
                "embryo_id": EMBRYO_ID,
                "target_setpoint_c": TARGET_C,
                "frames": 60,
                "bursts_before": 1,
                "bursts_after": 1,
            },
        },
        {
            "event_id": "ev-sc",
            "event_type": "temperature",
            "event_subtype": "setpoint_changed",
            "timestamp": _ts(20),
            "source": "test",
            "session_id": SESSION_ID,
            "embryo_id": EMBRYO_ID,
            "data": {"embryo_id": EMBRYO_ID, "to": TARGET_C},
        },
        {
            "event_id": "ev-bs",
            "event_type": "timelapse",
            "event_subtype": "burst_started",
            "timestamp": _ts(30),
            "source": "test",
            "session_id": SESSION_ID,
            "embryo_id": EMBRYO_ID,
            "data": {"embryo_id": EMBRYO_ID, "mode": "1hz", "frames": 60},
        },
        {
            "event_id": "ev-bc",
            "event_type": "timelapse",
            "event_subtype": "burst_completed",
            "timestamp": _ts(40),
            "source": "test",
            "session_id": SESSION_ID,
            "embryo_id": EMBRYO_ID,
            "data": {"embryo_id": EMBRYO_ID},
        },
        {
            "event_id": "ev-tpc",
            "event_type": "tactic",
            "event_subtype": "temp_protocol_completed",
            "timestamp": _ts(50),
            "source": "test",
            "session_id": SESSION_ID,
            "embryo_id": EMBRYO_ID,
            "data": {"embryo_id": EMBRYO_ID, "locked": True, "cancelled": False, "error": None},
        },
    ]
    timeline_path = session_dir / "timeline.jsonl"
    with open(timeline_path, "w", encoding="utf-8") as f:
        for ev in events:
            f.write(json.dumps(ev) + "\n")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_temp_protocol_band_present(tmp_path: Path) -> None:
    """Snapshot exposes a temp_protocol band with start and end on the embryo."""
    session_dir = tmp_path / "sessions" / "20250101_1000_test"
    _write_session(session_dir)

    snap = build_strategy_snapshot(session_dir, SESSION_ID)

    assert snap["embryos"], "no embryos in snapshot"
    emb = snap["embryos"][0]

    tp = emb.get("temp_protocol")
    assert tp is not None, "temp_protocol key missing from embryo"
    assert tp["start"] == pytest.approx(10.0), f"expected start=10s, got {tp['start']}"
    assert tp["end"] == pytest.approx(50.0), f"expected end=50s, got {tp['end']}"
    assert tp["target_setpoint_c"] == pytest.approx(TARGET_C)


def test_setpoint_changes_recorded(tmp_path: Path) -> None:
    """Snapshot exposes setpoint_changes list with one entry {t, to}."""
    session_dir = tmp_path / "sessions" / "20250101_1000_test"
    _write_session(session_dir)

    snap = build_strategy_snapshot(session_dir, SESSION_ID)

    emb = snap["embryos"][0]
    sc = emb.get("setpoint_changes")
    assert sc is not None, "setpoint_changes key missing from embryo"
    assert len(sc) == 1, f"expected 1 setpoint change, got {len(sc)}"
    assert sc[0]["t"] == pytest.approx(20.0)
    assert sc[0]["to"] == pytest.approx(TARGET_C)


def test_burst_phase_still_present(tmp_path: Path) -> None:
    """Existing burst phase handling is not disrupted by temp-protocol events."""
    session_dir = tmp_path / "sessions" / "20250101_1000_test"
    _write_session(session_dir)

    snap = build_strategy_snapshot(session_dir, SESSION_ID)

    emb = snap["embryos"][0]
    burst_phases = [p for p in emb["phases"] if p.get("mode") == "burst"]
    assert burst_phases, "burst phase disappeared after temp-protocol changes"
    bp = burst_phases[0]
    assert bp["start"] == pytest.approx(30.0)
    # end is extended to now_offset_s by the tail-close sweep when no
    # cadence_changed follows — just confirm it's set and >= burst start
    assert bp["end"] is not None and bp["end"] >= 30.0


def test_temp_protocol_fields_initialized_even_without_events(tmp_path: Path) -> None:
    """Embryo dicts always have temp_protocol and setpoint_changes, even with no events."""
    session_dir = tmp_path / "sessions" / "20250101_1000_empty"
    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "session.yaml").write_text("{}", encoding="utf-8")
    timelapse = {
        "started_at": STARTED_AT.isoformat(),
        "base_interval_seconds": 120,
        "embryos": {EMBRYO_ID: {"interval_seconds": 120}},
    }
    (session_dir / "timelapse.yaml").write_text(yaml.dump(timelapse), encoding="utf-8")
    # No timeline.jsonl

    snap = build_strategy_snapshot(session_dir, SESSION_ID)

    emb = snap["embryos"][0]
    assert "temp_protocol" in emb, "temp_protocol key absent with no events"
    assert "setpoint_changes" in emb, "setpoint_changes key absent with no events"
    assert emb["temp_protocol"] is None
    assert emb["setpoint_changes"] == []
