"""Shadow orchestrator scaffolding.

A candidate orchestrator runs alongside production: it sees the same
events but its decisions are LOGGED, not enacted. Diff the decision logs
between production and a candidate (or between two candidates) to compare
architectures on identical input streams.

Two entry points:

  OrchestratorCandidate
    Protocol that any candidate must satisfy. Receives events via
    on_event() and ticks via on_tick(); is given a DecisionLog to write
    into. Never gets to call tools that touch hardware — by construction
    its only output is the log.

  ShadowRunner
    Hosts a set of candidates against a single EventBus. Wildcards onto
    the bus and forwards each event to every registered candidate.
    Lifecycle (start / stop) keeps subscriptions tidy.

The simplest candidate is NoOpCandidate, included as a worked example
and as proof-of-life for the wiring (events visible? decision log
writeable? shutdown clean?).
"""

from __future__ import annotations

import logging
import threading
from abc import ABC, abstractmethod
from collections.abc import Callable
from datetime import datetime

from gently.core.event_bus import Event, EventBus

from .decision_log import Decision, DecisionLog, DecisionTrigger

logger = logging.getLogger(__name__)


class OrchestratorCandidate(ABC):
    """Base class for a shadow orchestrator candidate.

    A candidate is given:
      - its name (e.g. "reactive-v1", "haiku-summariser")
      - a DecisionLog to write decisions into

    It receives events synchronously via ``on_event``. If it needs to
    do heavy work (LLM call, long compute), it should hand off to its
    own task / thread and write into the log asynchronously.

    Candidates MUST NOT touch hardware. They have no access to the
    device-layer client, no permission to publish events back onto the
    bus, no MMCore handle. The only side effect they're allowed is
    writing to their decision log.
    """

    def __init__(self, name: str, decisions: DecisionLog):
        self.name = name
        self.decisions = decisions

    @abstractmethod
    def on_event(self, event: Event) -> None:
        """Handle one event from the bus. Synchronous, must not block long."""

    def on_start(self) -> None:  # noqa: B027
        """Called once when the shadow runner attaches this candidate."""

    def on_stop(self) -> None:  # noqa: B027
        """Called once when the shadow runner detaches this candidate."""

    # ---- helpers candidates can use ---------------------------------------

    def log_decision(
        self,
        *,
        trigger: DecisionTrigger,
        trigger_detail: str | None = None,
        tool_calls: list[dict] | None = None,
        response_text: str | None = None,
        context_summary: str | None = None,
        recent_event_ids: list[str] | None = None,
        prompt_hash: str | None = None,
        duration_ms: float | None = None,
        error: str | None = None,
    ) -> None:
        self.decisions.append(
            Decision(
                timestamp=datetime.now(),
                agent=self.name,
                trigger=trigger,
                trigger_detail=trigger_detail,
                tool_calls=tool_calls or [],
                response_text=response_text,
                context_summary=context_summary,
                recent_event_ids=recent_event_ids or [],
                prompt_hash=prompt_hash,
                duration_ms=duration_ms,
                error=error,
            )
        )


class NoOpCandidate(OrchestratorCandidate):
    """Trivial candidate: logs every event it sees as a decision marker.

    Useful as the smoke test for the wiring (events visible? decision
    log writeable? shutdown clean?) and as the template every real
    candidate evolves from.
    """

    def __init__(self, name: str, decisions: DecisionLog, *, watch: list[str] | None = None):
        super().__init__(name, decisions)
        # Optional whitelist of event_type names to react to. None = all.
        self._watch = set(watch) if watch else None
        self._seen = 0

    def on_event(self, event: Event) -> None:
        if self._watch is not None and event.event_type.name not in self._watch:
            return
        self._seen += 1
        self.log_decision(
            trigger=DecisionTrigger.EVENT,
            trigger_detail=event.event_type.name,
            response_text=f"(noop) seen {event.event_type.name} from {event.source}",
            recent_event_ids=[event.event_id],
            context_summary=f"noop candidate; events seen so far: {self._seen}",
        )


class ShadowRunner:
    """Hosts a set of OrchestratorCandidates against an EventBus.

    Wildcards onto the bus, dispatches each event to every registered
    candidate. Candidates' exceptions are caught and logged so one
    bad candidate doesn't take down the others or affect the live bus.

    The runner itself never enacts decisions — it only forwards events
    and lets candidates write their own logs.
    """

    def __init__(self, bus: EventBus):
        self.bus = bus
        self._candidates: list[OrchestratorCandidate] = []
        self._unsub: Callable[[], None] | None = None
        self._lock = threading.RLock()
        self._running = False

    def add(self, candidate: OrchestratorCandidate) -> None:
        with self._lock:
            self._candidates.append(candidate)
            if self._running:
                try:
                    candidate.on_start()
                except Exception:
                    logger.exception("ShadowRunner: on_start failed for %s", candidate.name)

    def remove(self, candidate: OrchestratorCandidate) -> None:
        with self._lock:
            try:
                self._candidates.remove(candidate)
            except ValueError:
                return
            try:
                candidate.on_stop()
            except Exception:
                logger.exception("ShadowRunner: on_stop failed for %s", candidate.name)

    def start(self) -> None:
        """Subscribe to the bus and notify every candidate. Idempotent."""
        with self._lock:
            if self._running:
                return
            self._unsub = self.bus.subscribe("*", self._on_event)
            for c in self._candidates:
                try:
                    c.on_start()
                except Exception:
                    logger.exception("ShadowRunner: on_start failed for %s", c.name)
            self._running = True
        logger.info("ShadowRunner: started with %d candidate(s)", len(self._candidates))

    def stop(self) -> None:
        """Unsubscribe from the bus and notify every candidate. Idempotent."""
        with self._lock:
            if not self._running:
                return
            if self._unsub is not None:
                try:
                    self._unsub()
                except Exception:
                    logger.exception("ShadowRunner: unsubscribe failed")
                self._unsub = None
            for c in self._candidates:
                try:
                    c.on_stop()
                except Exception:
                    logger.exception("ShadowRunner: on_stop failed for %s", c.name)
            self._running = False
        logger.info("ShadowRunner: stopped")

    @property
    def candidates(self) -> list[OrchestratorCandidate]:
        with self._lock:
            return list(self._candidates)

    def _on_event(self, event: Event) -> None:
        # Snapshot under the lock so a remove() mid-dispatch doesn't break us.
        with self._lock:
            candidates = list(self._candidates)
        for c in candidates:
            try:
                c.on_event(event)
            except Exception:
                logger.exception(
                    "ShadowRunner: candidate %s raised on %s",
                    c.name,
                    event,
                )
