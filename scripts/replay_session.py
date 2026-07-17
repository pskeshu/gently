#!/usr/bin/env python
"""Replay a captured session's events into a fresh EventBus.

Useful for:
  - Diffing what a candidate orchestrator would have decided from the
    same input stream the production agent saw.
  - Inspecting the event histogram of a session before deciding what to
    investigate ("did this session even fire any ERROR_OCCURRED?").
  - Re-running a session offline with a different filter / candidate
    set without touching hardware.

Examples
--------
List the events recorded in a session:
    python scripts/replay_session.py 2e0e0356 --histogram

Replay as fast as possible:
    python scripts/replay_session.py 2e0e0356

Replay with original cadence, 4x speed, and a NoOpCandidate writing a
decision log into the current directory:
    python scripts/replay_session.py 2e0e0356 --real-time --time-scale 4 --candidate noop-test

Custom root (default: $GENTLY_STORAGE_PATH or D:/Gently3):
    python scripts/replay_session.py 2e0e0356 --root /path/to/sessions
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

# Allow `python scripts/replay_session.py …` from the repo root without
# requiring PYTHONPATH=.; the project root is one level up from this file.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "session_id",
        help="Session id (full or prefix) to replay",
    )
    parser.add_argument(
        "--root",
        default=None,
        help="Storage root (default: $GENTLY_STORAGE_PATH or D:/Gently3)",
    )
    parser.add_argument(
        "--real-time",
        action="store_true",
        help="Preserve original cadence between events (default: fast)",
    )
    parser.add_argument(
        "--time-scale",
        type=float,
        default=1.0,
        help="Real-time replay speed multiplier (default: 1.0)",
    )
    parser.add_argument(
        "--candidate",
        default=None,
        help="Attach a NoOpCandidate; decisions written to replay-decisions-<sid>.jsonl",
    )
    parser.add_argument(
        "--histogram",
        action="store_true",
        help="Print event-type histogram, don't replay",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose logging",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    from gently.core.event_bus import EventBus
    from gently.core.file_store import FileStore
    from gently.eval import DecisionLog, EventReplay, NoOpCandidate, ShadowRunner

    root = args.root or os.environ.get("GENTLY_STORAGE_PATH", "D:/Gently3")
    store = FileStore(root=Path(root))

    sessions = store.list_sessions()
    matches = [s for s in sessions if s["session_id"].startswith(args.session_id)]
    if not matches:
        print(f"No session matching '{args.session_id}'", file=sys.stderr)
        return 1
    if len(matches) > 1:
        print(f"Multiple sessions match '{args.session_id}':", file=sys.stderr)
        for s in matches:
            print(f"  {s['session_id']}", file=sys.stderr)
        return 1

    session = matches[0]
    session_dir = store._session_dir(session["session_id"])
    assert session_dir is not None
    log_path = session_dir / "events.jsonl"
    if not log_path.exists():
        print(f"No events.jsonl in {session_dir}", file=sys.stderr)
        return 1

    rep = EventReplay(log_path)

    if args.histogram:
        hist = rep.event_types()
        total = sum(hist.values())
        print(f"{total} events in {log_path}:")
        for ev, n in sorted(hist.items(), key=lambda kv: -kv[1]):
            print(f"  {n:>6}  {ev}")
        return 0

    bus = EventBus()
    runner = None
    dlog = None
    if args.candidate:
        out = Path.cwd() / f"replay-decisions-{session['session_id'][:8]}.jsonl"
        dlog = DecisionLog(out)
        dlog.open()
        runner = ShadowRunner(bus)
        runner.add(NoOpCandidate(args.candidate, dlog))
        runner.start()
        print(f"Candidate '{args.candidate}' attached; decisions -> {out}")

    try:
        emitted = rep.replay(
            bus,
            real_time=args.real_time,
            time_scale=args.time_scale,
        )
        print(f"Replayed {emitted} events from session {session['session_id']}")
    finally:
        if runner is not None:
            runner.stop()
        if dlog is not None:
            dlog.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
