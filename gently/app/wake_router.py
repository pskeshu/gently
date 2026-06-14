"""Decision-moment wake-router for autonomous agent turns.

Subscribes to wake-worthy perception/lifecycle events and, when enabled, wakes
the conversational agent between user messages so it can re-decide acquisition
(cadence, power, stop conditions) in response to what perception sees — the
closed loop.

Design (opt-in, default OFF):
  * Triggers: critical events (hatching / arrest / embryo-terminated / errors)
    plus developmental stage transitions. No periodic heartbeat.
  * Debounce: a burst of events inside COALESCE_WINDOW collapses into ONE wake.
  * Throttle: non-critical wakes are rate-limited by MIN_WAKE_INTERVAL; critical
    events bypass the throttle.
  * Serialization: the wake turn runs through the agent's normal streaming
    pipeline, which holds the agent turn-lock, so it never races a user turn.
    A wake therefore waits for any in-progress user turn — including an open
    choice picker — to finish before it runs; "critical bypasses the throttle"
    means it skips MIN_WAKE_INTERVAL, not that it preempts an active user turn
    (preempting would interleave on the shared conversation history).

Nothing fires until ``set_enabled(True)`` (e.g. via the set_autonomy tool).
"""

from __future__ import annotations

import asyncio
import logging

from gently.core.event_bus import EventType

logger = logging.getLogger(__name__)

# Tunables (seconds).
COALESCE_WINDOW = 20.0  # collapse a burst of events into one wake
MIN_WAKE_INTERVAL = 120.0  # throttle non-critical wakes
ASK_TIMEOUT_SEC = 300.0  # ASK mode: how long to wait for operator approval -> Skip

# Events that always wake immediately (bypass MIN_WAKE_INTERVAL).
CRITICAL_EVENTS = frozenset(
    {
        EventType.HATCHING_DETECTED,
        EventType.EMBRYO_TERMINATED,
        EventType.ERROR_OCCURRED,
        EventType.ACQUISITION_FAILED,
        EventType.ANOMALY_DETECTED,
    }
)
# Non-critical events we also inspect (filtered for real transitions / arrest).
WATCH_EVENTS = frozenset({EventType.DETECTOR_EVALUATED})


class WakeRouter:
    """Routes wake-worthy events into coalesced, throttled autonomous agent turns."""

    def __init__(self, agent, bus):
        self.agent = agent
        self.bus = bus
        self.mode = "off"  # 'off' | 'ask' | 'auto'
        self._loop = None
        self._pending = []  # list[(EventType, dict)]
        self._flush_handle = None  # TimerHandle for the coalesce window
        self._last_wake = 0.0  # loop.time() of the last fired wake
        self._last_stage = {}  # embryo_id -> last stage seen (transition detection)
        self._in_flight = False
        self._unsubs = []
        self._subscribe()

    # -- public control -------------------------------------------------
    @property
    def enabled(self) -> bool:
        return self.mode != "off"

    def set_mode(self, mode: str) -> str:
        mode = (mode or "off").strip().lower()
        if mode not in ("off", "ask", "auto"):
            mode = "off"
        self.mode = mode
        if mode == "off":
            self._cancel_flush()
            self._pending.clear()
        logger.info("Wake-router mode -> %s", mode.upper())
        return self.mode

    def set_enabled(self, enabled: bool) -> bool:
        """Back-compat boolean toggle: maps to AUTO / OFF."""
        self.set_mode("auto" if enabled else "off")
        return self.enabled

    def is_enabled(self) -> bool:
        return self.enabled

    def shutdown(self):
        self._cancel_flush()
        for unsub in self._unsubs:
            try:
                unsub()
            except Exception:
                pass
        self._unsubs.clear()

    # -- subscription ---------------------------------------------------
    def _subscribe(self):
        for et in CRITICAL_EVENTS | WATCH_EVENTS:
            try:
                self._unsubs.append(
                    self.bus.subscribe(et, lambda e, _et=et: self._on_event(_et, e))
                )
            except Exception:
                logger.exception("wake-router failed to subscribe %s", et)

    # -- event intake ---------------------------------------------------
    def _on_event(self, event_type, event):
        # Synchronous handler (the bus calls it inline). Cheap-filter, then
        # schedule a coalesced flush on the running loop. Never raise — the bus
        # swallows handler exceptions, so failures would otherwise vanish.
        try:
            if not self.enabled:
                return
            if self._loop is None:
                try:
                    self._loop = asyncio.get_running_loop()
                except RuntimeError:
                    return  # no running loop -> can't schedule a wake; drop
            data = getattr(event, "data", None) or {}
            if not self._is_wake_worthy(event_type, data):
                return
            self._pending.append((event_type, data))
            self._schedule_flush(critical=event_type in CRITICAL_EVENTS)
        except Exception:
            logger.exception("wake-router _on_event error")

    def _is_wake_worthy(self, event_type, data) -> bool:
        if event_type in CRITICAL_EVENTS:
            return True
        if event_type == EventType.DETECTOR_EVALUATED:
            if data.get("skipped"):
                return False
            if data.get("detector_name") != "perception":
                return False  # role=test pseudo-stages are not developmental
            stage = data.get("stage")
            if not stage or stage == "no_object":
                return False  # empty-field sentinel — not a developmental change
            ta = data.get("temporal_analysis") or {}
            if ta.get("is_potentially_arrested"):
                return True
            eid = data.get("embryo_id")
            last = self._last_stage.get(eid)
            self._last_stage[eid] = stage
            return stage != last  # only a real transition wakes
        return False

    # -- coalescing / flush --------------------------------------------
    def _schedule_flush(self, critical: bool):
        loop = self._loop
        if loop is None:
            return
        delay = 0.0 if critical else COALESCE_WINDOW
        if self._flush_handle is None:
            self._flush_handle = loop.call_later(delay, self._fire_flush)
        elif critical:
            # bring a pending window-flush forward
            self._flush_handle.cancel()
            self._flush_handle = loop.call_later(0.0, self._fire_flush)

    def _cancel_flush(self):
        if self._flush_handle is not None:
            try:
                self._flush_handle.cancel()
            except Exception:
                pass
            self._flush_handle = None

    def _fire_flush(self):
        self._flush_handle = None
        loop = self._loop
        if loop is not None:
            asyncio.ensure_future(self._flush(), loop=loop)

    async def _flush(self):
        if not self._pending or not self.enabled:
            self._pending.clear()
            return
        # Evaluate the guards BEFORE draining so a deferral can't lose events.
        critical = any(et in CRITICAL_EVENTS for et, _ in self._pending)
        now = self._loop.time() if self._loop else 0.0
        if self._in_flight or (not critical and (now - self._last_wake) < MIN_WAKE_INTERVAL):
            # A wake is already running, or we're inside the non-critical throttle
            # window. Keep _pending intact and re-arm so these events — including
            # any CRITICAL ones — are retried once the turn finishes / window
            # elapses, rather than being dropped.
            logger.debug("wake deferred (in_flight=%s critical=%s)", self._in_flight, critical)
            # Retry on the coalesce window (not delay 0) so a critical event
            # deferred behind an in-flight turn doesn't busy-spin call_later(0).
            self._schedule_flush(critical=False)
            return
        events = self._pending
        self._pending = []
        self._in_flight = True
        self._last_wake = now
        try:
            ask = self.mode == "ask"
            note, trigger = self._build_wake_note(events, ask=ask)
            logger.info(
                "Wake-router firing %s turn (%d event(s)): %s",
                self.mode.upper(),
                len(events),
                trigger,
            )
            await self.agent.run_wake_turn(note, trigger=trigger, interactive=ask)
        except Exception:
            logger.exception("wake turn failed")
        finally:
            self._in_flight = False
            # Events that arrived while we were busy (including deferred CRITICAL
            # ones) are still in _pending — re-fire promptly rather than waiting
            # out another coalesce window. _in_flight is now False so this flush
            # will proceed instead of deferring (no busy-spin).
            if self._pending and self.enabled:
                self._schedule_flush(critical=any(et in CRITICAL_EVENTS for et, _ in self._pending))

    # -- wake package ---------------------------------------------------
    def _build_wake_note(self, events, ask=False):
        """Return (note, trigger_str). The note is the agent-facing wake prompt;
        trigger_str is the short human-readable reason shown in the chat banner.
        When ask=True the note instructs propose-then-confirm instead of acting."""
        from gently.harness.prompts.templates import build_perception_snapshot

        triggers = []
        for et, data in events:
            name = getattr(et, "name", str(et))
            eid = data.get("embryo_id", "?")
            stage = data.get("stage")
            if et == EventType.HATCHING_DETECTED:
                triggers.append(f"{eid}: hatching detected")
            elif et == EventType.EMBRYO_TERMINATED:
                triggers.append(f"{eid}: terminated ({data.get('completion_reason', '?')})")
            elif et in (
                EventType.ERROR_OCCURRED,
                EventType.ACQUISITION_FAILED,
                EventType.ANOMALY_DETECTED,
            ):
                triggers.append(f"{eid}: {name.lower().replace('_', ' ')}")
            elif et == EventType.DETECTOR_EVALUATED:
                ta = data.get("temporal_analysis") or {}
                if ta.get("is_potentially_arrested"):
                    triggers.append(f"{eid}: potential arrest at stage {stage}")
                else:
                    triggers.append(f"{eid}: stage -> {stage}")
            else:
                triggers.append(f"{eid}: {name.lower()}")
        triggers = list(dict.fromkeys(triggers))  # dedupe, preserve order

        try:
            snap = build_perception_snapshot(
                getattr(self.agent, "perceiver", None),
                getattr(getattr(self.agent, "experiment", None), "embryos", {}) or {},
            )
        except Exception:
            snap = ""
        snap = snap or "(no live perception data)"
        trigger_str = "; ".join(triggers)

        head = (
            "[AUTONOMOUS WAKE] Something changed while no one was typing.\n\n"
            f"What triggered this: {trigger_str}\n\n"
            f"{snap}\n\n"
        )
        if ask:
            tail = (
                "Decide whether any acquisition change is warranted. If so, briefly "
                "state your proposed change and WHY, then call ask_user_choice with "
                "options Approve / Modify / Skip and act ONLY if the operator approves. "
                "If nothing needs doing, say so briefly and take no action (no need to ask)."
            )
        else:
            tail = (
                "If a change helps (adjust interval/power, add a stop condition, queue a "
                "burst, or stop an embryo), do it now using your tools. If nothing needs "
                "doing, say so briefly and take no action."
            )
        return head + tail, trigger_str
