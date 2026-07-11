"""Operation-plan updater — subscribes to execution lifecycle events and
transitions plan tactics to their completed state.

Modeled on `gently/app/temperature_sampler.py` (Service lifecycle) and
`gently/harness/session/timeline.py` TimelineManager.start (bus-subscribe
pattern).

When a tactic's execution completes — burst captured, temperature protocol
finished, trigger fired — this service reads `tactic_id` from the event
payload and calls `context_store.transition_tactic(session_id, tactic_id,
state, **bind)` to stamp live values and advance the tactic state.  Missing
`tactic_id` or session → skip.  Handler exceptions are caught so a bad event
never crashes the bus.
"""

from __future__ import annotations

import logging
from collections.abc import Callable

from gently.core.event_bus import EventType, get_event_bus
from gently.core.service import Service

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Event → (state, bind-keys) mapping
# ---------------------------------------------------------------------------
# state=None means bind-only (no state transition).
_EVENT_ACTIONS: dict[EventType, tuple[str | None, list[str]]] = {
    EventType.BURST_COMPLETE: (
        "done",
        ["request_id", "mp4_path", "sustained_hz", "frames_captured"],
    ),
    EventType.TEMP_PROTOCOL_COMPLETED: (
        "done",
        ["locked", "cancelled", "error"],
    ),
    EventType.EMBRYO_CADENCE_CHANGED: (
        None,  # bind-only; standing tactic stays active
        ["embryo_id", "old_phase", "new_phase", "old_interval_s", "new_interval_s", "next_due_at"],
    ),
    EventType.TRIGGER_FIRED: (
        None,  # bind-only; record last firing time
        ["embryo_id", "rule_name", "rule_kind", "trigger_detector", "trigger_stage"],
    ),
}


class OperationPlanUpdater(Service):
    """Subscribe to execution events; transition plan tactics accordingly."""

    def __init__(self, context_store, session_id_getter: Callable[[], str | None]):
        super().__init__(name="operation-plan-updater", service_type="monitor")
        self._context_store = context_store
        self._session_id_getter = session_id_getter
        self._unsubscribers: list[Callable[[], None]] = []

    # ------------------------------------------------------------------
    # Service lifecycle
    # ------------------------------------------------------------------

    async def on_start(self) -> None:
        bus = get_event_bus()
        for event_type in _EVENT_ACTIONS:
            unsub = bus.subscribe(event_type, self._on_event)
            self._unsubscribers.append(unsub)
        logger.info(
            "OperationPlanUpdater started — subscribed to %d event types",
            len(_EVENT_ACTIONS),
        )

    async def on_stop(self) -> None:
        for unsub in self._unsubscribers:
            try:
                unsub()
            except Exception as exc:
                logger.warning("OperationPlanUpdater unsubscribe error: %s", exc)
        self._unsubscribers.clear()
        logger.info("OperationPlanUpdater stopped")

    # ------------------------------------------------------------------
    # Event handler
    # ------------------------------------------------------------------

    def _on_event(self, event) -> None:
        """Handle an execution lifecycle event — guard everything; never raise."""
        try:
            self._handle(event)
        except Exception as exc:
            logger.warning(
                "OperationPlanUpdater: unhandled error in _on_event (%s): %s",
                event.event_type,
                exc,
            )

    def _handle(self, event) -> None:
        data = event.data or {}
        tactic_id = data.get("tactic_id")
        session_id = self._session_id_getter()

        action = _EVENT_ACTIONS.get(event.event_type)
        if action is None:
            return  # unknown event type — ignore

        state, bind_keys = action

        # For bind-only events with no tactic_id there is nothing to attach to.
        if not tactic_id:
            logger.debug(
                "OperationPlanUpdater: no tactic_id in %s payload — skip",
                event.event_type,
            )
            return

        if not session_id:
            logger.debug(
                "OperationPlanUpdater: no active session for %s — skip",
                event.event_type,
            )
            return

        bind = {k: data[k] for k in bind_keys if k in data}

        # Add a last_fired marker for TRIGGER_FIRED so operators can see it.
        if event.event_type is EventType.TRIGGER_FIRED:
            bind["last_fired"] = event.timestamp

        ok = self._context_store.transition_tactic(session_id, tactic_id, state, **bind)
        if ok:
            logger.info(
                "OperationPlanUpdater: tactic %s → %s (event=%s)",
                tactic_id,
                state or "bind-only",
                event.event_type,
            )
        else:
            logger.debug(
                "OperationPlanUpdater: transition_tactic no-op for tactic=%s session=%s",
                tactic_id,
                session_id,
            )
