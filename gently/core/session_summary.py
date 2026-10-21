"""What a session holds and what happened in it, read from its folder.

The Sessions tab, the project view and the agent's session-end note all say
the same things about a session, so they come from here: directory names and
one small checkpoint, no image decoded; the event capture for what went wrong;
the interaction log for how much the agent did and how often it was corrected.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# The events worth reading back, in a sentence each. Status chatter and
# per-frame traffic are left out on purpose.
NOTABLE = {
    "SESSION_RESTORED": lambda d: "Session restored",
    "ACQUISITION_STARTED": lambda d: (
        "Acquisition started"
        + (f" · {len(d['embryo_ids'])} embryos" if isinstance(d.get("embryo_ids"), list) else "")
    ),
    "ACQUISITION_COMPLETED": lambda d: "Acquisition completed",
    "ACQUISITION_STOPPED": lambda d: (
        "Acquisition stopped" + (f" · {d['reason']}" if d.get("reason") else "")
    ),
    "ACQUISITION_FAILED": lambda d: (
        "Acquisition failed"
        + (
            f" · {d.get('error') or d.get('error_message')}"
            if d.get("error") or d.get("error_message")
            else ""
        )
    ),
    "TRIGGER_FIRED": lambda d: (
        f"Rule {d.get('rule') or d.get('name') or '?'} fired"
        + (f" on {d['embryo_id']}" if d.get("embryo_id") else "")
    ),
    "BURST_START": lambda d: (
        f"Burst on {d.get('embryo_id', '?')}: {d.get('frames', '?')} frames"
        + (f" ({d['mode']})" if d.get("mode") else "")
    ),
    "BURST_COMPLETE": lambda d: (
        f"Burst done on {d.get('embryo_id', '?')}"
        + (f": {d['frames_captured']} frames" if d.get("frames_captured") is not None else "")
        + (
            f" at {d['sustained_hz']:.1f} Hz"
            if isinstance(d.get("sustained_hz"), (int, float))
            else ""
        )
    ),
    "POWER_RAMP_STEP": lambda d: (
        f"{d.get('wavelength', '?')} nm power {d.get('old_pct')}% → {d.get('new_pct')}%"
        + (f" on {d['embryo_id']}" if d.get("embryo_id") else "")
    ),
    "HATCHING_DETECTED": lambda d: f"Hatching detected on {d.get('embryo_id', '?')}",
    "DETECTION_TRIGGERED": lambda d: (
        f"{d.get('detector_name') or d.get('detector') or 'Detector'} fired"
        + (f" on {d['embryo_id']}" if d.get("embryo_id") else "")
    ),
    "EMBRYO_TERMINATED": lambda d: (
        f"{d.get('embryo_id', '?')} terminated" + (f" · {d['reason']}" if d.get("reason") else "")
    ),
    "EMBRYO_SKIPPED": lambda d: (
        f"{d.get('embryo_id', '?')} skipped" + (f" · {d['reason']}" if d.get("reason") else "")
    ),
    "OPERATOR_REMOVED_EMBRYO": lambda d: f"Operator removed {d.get('embryo_id', '?')}",
    "TEMPERATURE_SETPOINT_CHANGED": lambda d: f"Setpoint → {d.get('to')} °C",
    "TEMP_PROTOCOL_COMPLETED": lambda d: (
        "Temperature protocol "
        + ("locked" if d.get("locked") else "cancelled" if d.get("cancelled") else "ended")
        + (f" · {d['error']}" if d.get("error") else "")
    ),
    "ERROR_OCCURRED": lambda d: f"Error: {d.get('message') or d.get('error') or str(d)[:160]}",
    "WARNING_ISSUED": lambda d: f"Warning: {d.get('message') or str(d)[:160]}",
}

LEVEL = {
    "ACQUISITION_FAILED": "error",
    "ERROR_OCCURRED": "error",
    "WARNING_ISSUED": "warn",
    "EMBRYO_TERMINATED": "warn",
    "EMBRYO_SKIPPED": "warn",
    "ACQUISITION_STOPPED": "warn",
}


def _jsonl(path: Path):
    """Each record of a jsonl file; a truncated line is skipped, not fatal."""
    if not path.is_file():
        return
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue
    except OSError:
        return


def read_checkpoint(store, sid: str) -> dict:
    """The session's timelapse.yaml, or {}."""
    try:
        folder = store._session_dir(sid)
        path = Path(folder) / "timelapse.yaml" if folder is not None else None
        if path is not None and path.is_file():
            import yaml

            doc = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
            return doc if isinstance(doc, dict) else {}
    except Exception:
        logger.debug("checkpoint of %s could not be read", sid, exc_info=True)
    return {}


def what_a_session_holds(store, sid: str) -> dict | None:
    """What there is to carry on with in a session, read from its folder:
    directory names and one small checkpoint, no image decoded. None for
    a session with no embryo in it."""
    try:
        embryo_ids = store.list_embryo_ids(sid)
    except Exception:
        embryo_ids = []
    if not embryo_ids:
        return None
    timepoints = 0
    last_image: float | None = None
    for eid in embryo_ids:
        try:
            tps = store.list_projection_timepoints(sid, eid) or []
        except Exception:
            tps = []
        if not tps:
            continue
        timepoints += len(tps)
        try:
            newest = store.get_projection_path(sid, eid, max(tps))
            if newest is not None:
                taken = newest.stat().st_mtime
                last_image = taken if last_image is None else max(last_image, taken)
        except OSError:
            pass

    # ``run`` is only an interrupted run — the launch gate offers those to
    # carry on. ``last_run`` is whatever the checkpoint says, for the
    # Sessions tab to describe a session before it is restored.
    run = None
    last_run = None
    state = read_checkpoint(store, sid)
    if state:
        rows = state.get("embryos") or {}
        going = [k for k, v in rows.items() if not (v or {}).get("is_complete")]
        last_run = {
            "status": state.get("status"),
            "embryos_going": len(going),
            "saved_at": state.get("saved_at"),
            "started_at": state.get("started_at"),
            "rounds": state.get("current_round"),
            "total_timepoints": state.get("total_timepoints"),
            "interval_seconds": state.get("base_interval_seconds"),
        }
        if state.get("status") in ("running", "paused") and going:
            run = dict(last_run, status="interrupted")

    info = store.get_session(sid) or {}
    return {
        "session_id": sid,
        "name": info.get("name") or sid,
        "created_at": info.get("created_at", ""),
        "embryo_count": len(embryo_ids),
        "timepoints": timepoints,
        "last_image_at": (
            datetime.fromtimestamp(last_image).isoformat(timespec="seconds")
            if last_image is not None
            else None
        ),
        "run": run,
        "last_run": last_run,
    }


def notable_events(path: Path, limit: int | None = 300) -> list[dict]:
    """The run's events worth reading back, oldest first, from events.jsonl."""
    out: list[dict] = []
    for rec in _jsonl(path):
        kind = rec.get("event_type") or rec.get("type")
        say = NOTABLE.get(kind)
        if say is None:
            continue
        data = rec.get("data") if isinstance(rec.get("data"), dict) else {}
        try:
            text = say(data)
        except Exception:
            text = str(kind).replace("_", " ").capitalize()
        out.append(
            {
                "type": kind,
                "at": rec.get("timestamp"),
                "text": text,
                "level": LEVEL.get(kind, "info"),
            }
        )
    return out[-limit:] if limit else out


def signal_counts(path: Path) -> dict[str, int]:
    """How many errors and warnings the event capture holds."""
    errors = warnings = 0
    for rec in _jsonl(path):
        level = LEVEL.get(rec.get("event_type") or rec.get("type"))
        if level == "error":
            errors += 1
        elif level == "warn":
            warnings += 1
    return {"errors": errors, "warnings": warnings}


def interaction_stats(path: Path) -> dict[str, int]:
    """Turns, corrections, tool calls and errored turns from the interaction log."""
    turns = corrections = tool_calls = errors = 0
    for rec in _jsonl(path):
        turns += 1
        if rec.get("was_corrected"):
            corrections += 1
        if rec.get("error"):
            errors += 1
        tool_calls += len(rec.get("tool_calls") or [])
    return {
        "turns": turns,
        "corrections": corrections,
        "tool_calls": tool_calls,
        "agent_errors": errors,
    }


def summarize_session(store, sid: str) -> dict[str, Any]:
    """Everything the project view and the session-end note say about a session."""
    folder = store._session_dir(sid)
    base = Path(folder) if folder is not None else None
    held = what_a_session_holds(store, sid) or {}
    events = notable_events(base / "events.jsonl", limit=None) if base else []
    return {
        "session_id": sid,
        "held": held,
        "events": events,
        "signals": signal_counts(base / "events.jsonl") if base else {"errors": 0, "warnings": 0},
        "agent": interaction_stats(base / "interaction_log.jsonl")
        if base
        else {"turns": 0, "corrections": 0, "tool_calls": 0, "agent_errors": 0},
    }


def _plural(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


def session_end_text(summary: dict[str, Any]) -> str:
    """One deterministic paragraph for the night, from the summary. Empty when
    the session held nothing and nothing happened, so empty sessions leave no note."""
    held = summary.get("held") or {}
    events = summary.get("events") or []
    signals = summary.get("signals") or {}
    agent = summary.get("agent") or {}
    if not held and not events and not agent.get("turns"):
        return ""
    parts: list[str] = []
    if held.get("embryo_count"):
        bit = _plural(int(held["embryo_count"]), "embryo")
        if held.get("timepoints"):
            bit += f", {_plural(int(held['timepoints']), 'timepoint')}"
        last = held.get("last_run") or {}
        if last.get("interval_seconds"):
            bit += f" every {int(last['interval_seconds'])} s"
        parts.append(bit + ".")
        status = last.get("status")
        if status in ("running", "paused") and last.get("embryos_going"):
            parts.append(
                "The run was cut short with "
                f"{_plural(int(last['embryos_going']), 'embryo')} still going."
            )
        elif status in ("completed", "complete", "stopped"):
            parts.append(f"The run ended {status}.")
    errs, warns = int(signals.get("errors") or 0), int(signals.get("warnings") or 0)
    if errs or warns:
        said = []
        if errs:
            said.append(_plural(errs, "error"))
        if warns:
            said.append(_plural(warns, "warning"))
        parts.append("Signals: " + " and ".join(said) + ".")
    worst = [e for e in events if e.get("level") == "error"][-2:]
    for e in worst:
        at = str(e.get("at") or "")[11:16]
        parts.append(f"{at} {e.get('text')}".strip() + ".")
    if agent.get("turns"):
        bit = _plural(int(agent["turns"]), "agent turn")
        if agent.get("corrections"):
            bit += f", {_plural(int(agent['corrections']), 'correction')}"
        parts.append(bit + ".")
    return " ".join(parts)
