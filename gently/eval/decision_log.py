"""DecisionLog — records each "decision moment" the orchestrator (or a
shadow candidate) acts on.

A "decision moment" is whenever the agent wakes up and produces an output:
a Claude tool call, a refusal, a chat reply, or even an explicit no-op
("I see what happened, nothing to do"). Capturing these gives us the diff
substrate for shadow-mode A/B: same input event stream, different
candidates, compare what each decided.

File format: one JSON object per line, written to
D:/Gently3/sessions/{id}/decisions.jsonl (or wherever the caller points
it). Lossless enough to reconstruct what the agent saw + chose, terse
enough to skim across sessions.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from .event_capture import _json_default

logger = logging.getLogger(__name__)


def prompt_hash(system_prompt: Any, messages: Any) -> str:
    """Stable short fingerprint of the input the orchestrator saw.

    Two candidates seeing byte-identical (system_prompt, messages) get
    the same hash; a difference here means they're working from different
    context, so any decision divergence is expected. Used in shadow A/B
    to filter out apples-to-oranges comparisons.

    SHA-256 truncated to 16 hex chars — enough to make accidental
    collisions vanishingly unlikely at the scale of one session's
    decisions, short enough to skim by eye in a log.
    """
    h = hashlib.sha256()
    if isinstance(system_prompt, str):
        h.update(system_prompt.encode("utf-8"))
    else:
        h.update(json.dumps(system_prompt, sort_keys=True, default=_json_default).encode("utf-8"))
    h.update(b"\x1f")  # separator so prompt boundary can't be ambiguous
    h.update(json.dumps(messages, sort_keys=True, default=_json_default).encode("utf-8"))
    return h.hexdigest()[:16]


class DecisionTrigger(str, Enum):
    """What woke the agent up for this decision moment."""

    USER_MESSAGE = "user_message"
    EVENT = "event"  # event-driven (perception, error, etc.)
    TICK = "tick"  # scheduled / periodic checkpoint
    PHASE = "phase"  # plan phase boundary (between embryos / timepoints)
    STARTUP = "startup"  # initial session bring-up
    UNKNOWN = "unknown"


@dataclass
class Decision:
    """A single decision moment.

    The fields try to capture three things:
      WHY the agent woke up: trigger, trigger_detail
      WHAT it saw: context_summary, recent_event_ids
      WHAT it did: tool_calls, response_text

    `prompt_hash` is a stable fingerprint of the actual prompt+context
    sent to Claude so two candidates with byte-identical input but
    different decisions can be told apart by a single field.
    """

    timestamp: datetime
    agent: str  # "production" or candidate name
    trigger: DecisionTrigger
    trigger_detail: str | None = None  # event_id, user message excerpt, tick name

    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    response_text: str | None = None
    prompt_hash: str | None = None

    context_summary: str | None = None  # one-line description of state
    recent_event_ids: list[str] = field(default_factory=list)

    duration_ms: float | None = None  # how long the decision took
    error: str | None = None  # if the decision moment errored

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "agent": self.agent,
            "trigger": self.trigger.value,
            "trigger_detail": self.trigger_detail,
            "tool_calls": self.tool_calls,
            "response_text": self.response_text,
            "prompt_hash": self.prompt_hash,
            "context_summary": self.context_summary,
            "recent_event_ids": self.recent_event_ids,
            "duration_ms": self.duration_ms,
            "error": self.error,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Decision:
        return cls(
            timestamp=datetime.fromisoformat(d["timestamp"]),
            agent=d.get("agent", "unknown"),
            trigger=DecisionTrigger(d.get("trigger", "unknown")),
            trigger_detail=d.get("trigger_detail"),
            tool_calls=d.get("tool_calls") or [],
            response_text=d.get("response_text"),
            prompt_hash=d.get("prompt_hash"),
            context_summary=d.get("context_summary"),
            recent_event_ids=d.get("recent_event_ids") or [],
            duration_ms=d.get("duration_ms"),
            error=d.get("error"),
        )


class DecisionLog:
    """Append-only jsonl sink for Decisions. Thread-safe."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._fp: Any = None
        self._lock = threading.Lock()
        self._count = 0

    def open(self) -> None:
        if self._fp is not None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fp = self.path.open("a", encoding="utf-8")
        logger.info("DecisionLog: writing to %s", self.path)

    def close(self) -> None:
        with self._lock:
            if self._fp is not None:
                try:
                    self._fp.close()
                except Exception:
                    logger.exception("DecisionLog: close failed")
                self._fp = None
        logger.info("DecisionLog: closed (%d decisions written)", self._count)

    def append(self, decision: Decision) -> None:
        try:
            line = json.dumps(decision.to_dict(), default=_json_default)
        except Exception:
            logger.exception("DecisionLog: failed to serialise %s", decision)
            return
        with self._lock:
            if self._fp is None:
                self.open()
            try:
                self._fp.write(line + "\n")
                self._fp.flush()
                self._count += 1
            except Exception:
                logger.exception("DecisionLog: write failed")

    @property
    def count(self) -> int:
        return self._count

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False

    def read(self) -> list[Decision]:
        """Read every decision back from disk. Quick + dirty diff substrate."""
        if not self.path.exists():
            return []
        out: list[Decision] = []
        with self.path.open("r", encoding="utf-8") as f:
            for line_no, raw in enumerate(f, start=1):
                raw = raw.strip()
                if not raw:
                    continue
                try:
                    out.append(Decision.from_dict(json.loads(raw)))
                except Exception:
                    logger.exception("DecisionLog: parse failure on line %d", line_no)
        return out
