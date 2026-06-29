"""Task 2: Burst phase field emitted in BURST_START and recorded in snapshot.

Feeds _replay_timeline (via build_strategy_snapshot over a hand-written
timeline.jsonl) a burst_started event that carries ``phase="during"`` and
asserts the resulting snapshot burst phase dict includes ``"phase": "during"``.
"""

import json
from datetime import datetime, timedelta
from pathlib import Path

import yaml

from gently.ui.web.strategy_snapshot import build_strategy_snapshot

SESSION_ID = "sess-bp1"
EMBRYO_ID = "e1"
STARTED_AT = datetime(2026, 6, 28, 10, 0, 0)


def _ts(delta_s: float) -> str:
    return (STARTED_AT + timedelta(seconds=delta_s)).isoformat()


def _write_session(session_dir: Path) -> None:
    session_dir.mkdir(parents=True, exist_ok=True)

    # Minimal session.yaml
    (session_dir / "session.yaml").write_text(
        yaml.dump({"name": "burst-phase test"}), encoding="utf-8"
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

    # timeline.jsonl
    events = [
        {
            "event_id": "ev-bs",
            "event_type": "timelapse",
            "event_subtype": "burst_started",
            "timestamp": _ts(10),
            "source": "test",
            "session_id": SESSION_ID,
            "embryo_id": EMBRYO_ID,
            "data": {"embryo_id": EMBRYO_ID, "mode": "1hz", "frames": 60, "phase": "during"},
        },
        {
            "event_id": "ev-bc",
            "event_type": "timelapse",
            "event_subtype": "burst_completed",
            "timestamp": _ts(70),
            "source": "test",
            "session_id": SESSION_ID,
            "embryo_id": EMBRYO_ID,
            "data": {"embryo_id": EMBRYO_ID},
        },
    ]
    timeline_path = session_dir / "timeline.jsonl"
    with open(timeline_path, "w", encoding="utf-8") as f:
        for ev in events:
            f.write(json.dumps(ev) + "\n")


def test_burst_phase_recorded(tmp_path: Path) -> None:
    """Snapshot burst phase dict carries the phase field from BURST_START data."""
    session_dir = tmp_path / "sessions" / "20260628_1000_test"
    _write_session(session_dir)

    snap = build_strategy_snapshot(session_dir, SESSION_ID)

    assert snap["embryos"], "no embryos in snapshot"
    emb = snap["embryos"][0]
    burst_phases = [p for p in emb["phases"] if p.get("mode") == "burst"]
    assert burst_phases, "burst phase not found in snapshot"
    bp = burst_phases[0]
    assert bp.get("phase") == "during", (
        f"expected phase='during', got {bp.get('phase')!r}; full burst dict: {bp}"
    )


def test_burst_phase_none_when_absent(tmp_path: Path) -> None:
    """A burst_started event without a phase field records phase=None gracefully."""
    session_dir = tmp_path / "sessions" / "20260628_1000_nophase"
    session_dir.mkdir(parents=True, exist_ok=True)

    (session_dir / "session.yaml").write_text(
        yaml.dump({"name": "burst-nophase test"}), encoding="utf-8"
    )
    timelapse = {
        "started_at": STARTED_AT.isoformat(),
        "base_interval_seconds": 120,
        "embryos": {EMBRYO_ID: {"interval_seconds": 120}},
    }
    (session_dir / "timelapse.yaml").write_text(yaml.dump(timelapse), encoding="utf-8")

    events = [
        {
            "event_id": "ev-bs2",
            "event_type": "timelapse",
            "event_subtype": "burst_started",
            "timestamp": _ts(10),
            "source": "test",
            "session_id": SESSION_ID,
            "embryo_id": EMBRYO_ID,
            "data": {"embryo_id": EMBRYO_ID, "mode": "1hz", "frames": 30},
        },
    ]
    with open(session_dir / "timeline.jsonl", "w", encoding="utf-8") as f:
        for ev in events:
            f.write(json.dumps(ev) + "\n")

    snap = build_strategy_snapshot(session_dir, SESSION_ID)
    emb = snap["embryos"][0]
    burst_phases = [p for p in emb["phases"] if p.get("mode") == "burst"]
    assert burst_phases, "burst phase not found"
    bp = burst_phases[0]
    # "phase" key must exist; value is None when not supplied
    assert "phase" in bp, f"'phase' key absent from burst dict: {bp}"
    assert bp["phase"] is None, f"expected None, got {bp['phase']!r}"
