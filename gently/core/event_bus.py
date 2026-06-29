"""
Event Bus for async message passing between components

Provides a publish/subscribe pattern for decoupled communication:
- Components publish events without knowing who listens
- Handlers subscribe to event types they care about
- Supports both sync and async handlers
- Event history for debugging and replay
"""

import asyncio
import logging
import threading
import uuid
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any

logger = logging.getLogger(__name__)


class EventType(Enum):
    """Standard event types for microscope operations"""

    # Lifecycle events
    SESSION_STARTED = auto()
    SESSION_ENDED = auto()
    SESSION_SAVED = auto()
    SESSION_RESTORED = auto()

    # Acquisition events
    ACQUISITION_STARTED = auto()
    ACQUISITION_COMPLETED = auto()
    ACQUISITION_STOPPED = auto()  # Manual stop from agent/user
    ACQUISITION_FAILED = auto()
    VOLUME_ACQUIRED = auto()
    IMAGE_ACQUIRED = auto()

    # Embryo events
    EMBRYO_DETECTED = auto()
    EMBRYO_CENTERED = auto()
    EMBRYO_CALIBRATED = auto()
    EMBRYO_SKIPPED = auto()
    # {embryo_id, completion_reason} - emitted when an embryo's imaging stops
    # (any reason: no_object terminal, stop condition met, errors, user removal)
    EMBRYO_TERMINATED = auto()

    # Analysis events
    ANALYSIS_STARTED = auto()
    ANALYSIS_COMPLETED = auto()
    DETECTOR_EVALUATED = auto()  # Emitted for every detector run (all evaluations)
    DETECTION_TRIGGERED = auto()  # Emitted only when detected=True (positive detection)
    HATCHING_DETECTED = auto()

    # Verification events (multi-strategy verification for detections)
    VERIFICATION_STARTED = auto()  # Verification round begins for embryo
    VERIFICATION_STRATEGY = auto()  # Individual strategy result (adversarial, temporal, etc.)
    VERIFICATION_PROGRESS = auto()  # Progress update (e.g., "3/5 strategies complete")
    VERIFICATION_COMPLETED = auto()  # Final verification result with consensus

    # CV Subagent events
    SEGMENTATION_COMPLETED = auto()
    STAGE_DETECTED = auto()
    CELL_DIVISION_DETECTED = auto()
    LINEAGE_UPDATED = auto()
    ANOMALY_DETECTED = auto()
    CV_TASK_QUEUED = auto()
    CV_TASK_COMPLETED = auto()
    CV_TASK_FAILED = auto()
    CV_AGENT_THINKING = auto()  # Streamed thinking blocks from CV agent

    # CV Atomic tool result events (for event-driven communication)
    CV_RESULT_READY = auto()  # Generic result ready event
    CV_NUCLEI_COUNTED = auto()
    CV_STAGE_CLASSIFIED = auto()
    CV_ELONGATION_MEASURED = auto()
    CV_HATCHING_DETECTED = auto()

    # Hardware events
    STAGE_MOVED = auto()
    FOCUS_CHANGED = auto()
    LASER_CHANGED = auto()
    DEVICE_STATE_UPDATE = auto()  # Periodic device-state snapshot from device layer
    TEMPERATURE_UPDATE = auto()  # Temperature reading from device layer
    BOTTOM_CAMERA_FRAME = auto()  # Live JPEG frame from the bottom camera stream
    LIGHTSHEET_FRAME = auto()  # Live JPEG frame from the SPIM lightsheet live stream
    EMBRYOS_UPDATE = auto()  # Full embryo list snapshot from agent.experiment
    SCAN_GEOMETRY_UPDATE = auto()  # Scan cuboid + light-sheet mode for the 3D optical-space view

    # Python logging.LogRecord republished onto the bus so the Events page
    # surfaces what would otherwise only land in the terminal. See
    # gently/core/log_bridge.py — opt-in handler.
    LOG_RECORD = auto()

    # Agent context/mind updates (expectations / watchpoints / questions) —
    # drives the shared-visibility surface in the v2 UI.
    CONTEXT_UPDATED = auto()

    # Plan/campaign mutated (item status, session link, new item, progress) —
    # drives live refresh of the Plans UI.
    PLAN_UPDATED = auto()

    # Operator-action events. Distinct from EMBRYOS_UPDATE because they
    # carry intent ("a human did this") rather than just state delta.
    # Candidate orchestrators can subscribe and reason about what the
    # operator just did without having to type it in chat.
    OPERATOR_EDITED_EMBRYO = auto()  # Map drag/drop -> PUT /api/embryos/{id}/position
    OPERATOR_REMOVED_EMBRYO = auto()  # Map delete  -> DELETE /api/embryos/{id}
    OPERATOR_MARKED_EMBRYOS = auto()  # Marking canvas "Done" — operator confirmed N positions

    # System events
    ERROR_OCCURRED = auto()
    WARNING_ISSUED = auto()
    STATUS_CHANGED = auto()

    # Async timelapse — per-embryo cadence transitions (Phase 4)
    EMBRYO_CADENCE_CHANGED = (
        auto()
    )  # {embryo_id, old_phase, new_phase, old_interval_s, new_interval_s, next_due_at}

    # Burst lifecycle (Phase 7 / 10)
    BURST_QUEUED = auto()  # {embryo_id, request_id, position_in_queue}
    BURST_START = auto()  # {embryo_id, request_id, frames, mode}
    BURST_FRAME = auto()  # {embryo_id, request_id, frame_idx, total_frames}
    BURST_COMPLETE = auto()  # {embryo_id, request_id, mp4_path, sustained_hz, frames_captured}

    # Temperature protocol events (Phase X / 10) — temp-change burst tactic
    TEMPERATURE_SETPOINT_CHANGED = auto()  # {embryo_id, to}
    TEMP_PROTOCOL_STARTED = (
        auto()
    )  # {embryo_id, target_setpoint_c, frames, bursts_before, bursts_after}
    TEMP_PROTOCOL_COMPLETED = auto()  # {embryo_id, locked, cancelled, error}

    # Reactive control telemetry (Phase 5 / 10)
    POWER_RAMP_STEP = auto()  # {embryo_id, rule, wavelength, old_pct, new_pct, direction}

    # Adaptive rule fired (Phase 5 / 10) — discrete trigger event separate
    # from the EMBRYO_CADENCE_CHANGED / POWER_RAMP_STEP it caused, so the
    # strategy / experiment view can replay rule firings independently of
    # whatever side-effect they had on cadence or laser power.
    # {embryo_id, rule_name, rule_kind ("interval"|"power"), trigger_detector,
    #  trigger_stage, trigger_intensity_level, applied: {...}}
    TRIGGER_FIRED = auto()

    # Per-detector findings stream (Phase 2 / 10)
    CLAUDE_DETECTOR_RESULT = auto()  # {embryo_id, timepoint, detector_name, findings}

    # Data events
    DATA_STORED = auto()
    DATA_RETRIEVED = auto()

    # User interaction events
    USER_INPUT = auto()
    USER_COMMAND = auto()

    # Mesh events
    MESH_PEER_DISCOVERED = auto()
    MESH_PEER_LOST = auto()
    MESH_PEER_UPDATED = auto()
    MESH_PAIRING_REQUESTED = auto()
    MESH_PAIRING_COMPLETED = auto()

    # Mesh security events (Phase 3)
    MESH_AUTH_FAILURE = auto()
    MESH_CERT_PIN_FAILURE = auto()
    MESH_SCOPE_DENIED = auto()

    # Mesh topology events
    MESH_PEER_OFFLINE = auto()  # peer marked offline in verse map (kept in map)
    MESH_PEER_RETURNED = auto()  # previously offline peer came back online

    # ML pipeline events
    ML_PIPELINE_CREATED = auto()
    ML_TRAINING_STARTED = auto()
    ML_TRAINING_PROGRESS = auto()  # per-epoch updates
    ML_TRAINING_COMPLETED = auto()
    ML_TRAINING_FAILED = auto()
    ML_EVALUATION_COMPLETED = auto()
    ML_SUBAGENT_STATUS = auto()  # subagent thinking/planning updates

    # Bulk transfer events
    TRANSFER_STARTED = auto()
    TRANSFER_PROGRESS = auto()
    TRANSFER_COMPLETED = auto()
    TRANSFER_FAILED = auto()


# High-volume telemetry events that skip the bounded history deque. These
# fire many times per second and would push out events that humans actually
# want to inspect later (acquisitions, perceptions, errors).
_NO_HISTORY_TYPES = frozenset(
    {
        EventType.DEVICE_STATE_UPDATE,
        EventType.TEMPERATURE_UPDATE,  # High-volume telemetry from temperature controller
        EventType.BOTTOM_CAMERA_FRAME,  # ~2 Hz JPEG frames — would crowd history out
        EventType.LIGHTSHEET_FRAME,  # High-volume live frames — keep out of history
        EventType.LOG_RECORD,  # log lines can hit hundreds/min during
        # calibration; durable copy is in the
        # gently_*.log file already
    }
)


@dataclass
class Event:
    """
    Event message passed through the bus

    Attributes
    ----------
    event_type : EventType
        Type of event
    data : dict
        Event payload
    source : str
        Component that emitted the event
    timestamp : datetime
        When the event occurred
    event_id : str
        Unique identifier for this event
    correlation_id : str, optional
        ID to correlate related events (e.g., request/response)
    """

    event_type: EventType
    data: dict[str, Any] = field(default_factory=dict)
    source: str = "unknown"
    timestamp: datetime = field(default_factory=datetime.now)
    event_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    correlation_id: str | None = None

    def __str__(self) -> str:
        return f"Event({self.event_type.name}, source={self.source}, id={self.event_id})"

    def to_dict(self) -> dict:
        """Serialize for storage/transmission"""
        return {
            "event_type": self.event_type.name,
            "data": self.data,
            "source": self.source,
            "timestamp": self.timestamp.isoformat(),
            "event_id": self.event_id,
            "correlation_id": self.correlation_id,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Event":
        """Deserialize from dict"""
        return cls(
            event_type=EventType[d["event_type"]],
            data=d.get("data", {}),
            source=d.get("source", "unknown"),
            timestamp=datetime.fromisoformat(d["timestamp"])
            if "timestamp" in d
            else datetime.now(),
            event_id=d.get("event_id", str(uuid.uuid4())[:8]),
            correlation_id=d.get("correlation_id"),
        )


# Type alias for event handlers
EventHandler = Callable[[Event], None]
AsyncEventHandler = Callable[[Event], Any]  # Can be coroutine


class EventBus:
    """
    Central event bus for publish/subscribe messaging

    Features:
    - Sync and async handler support
    - Event history for debugging
    - Wildcard subscriptions (subscribe to all events)
    - Thread-safe operations
    """

    def __init__(self, history_size: int = 100):
        """
        Parameters
        ----------
        history_size : int
            Number of recent events to keep in history
        """
        self._handlers: dict[EventType, list[EventHandler]] = {}
        self._async_handlers: dict[EventType, list[AsyncEventHandler]] = {}
        self._wildcard_handlers: list[EventHandler] = []
        self._async_wildcard_handlers: list[AsyncEventHandler] = []
        self._history: deque = deque(maxlen=history_size)
        self._lock = threading.RLock()
        self._event_loop: asyncio.AbstractEventLoop | None = None

    def subscribe(
        self,
        event_type: EventType | str,
        handler: EventHandler,
    ) -> Callable[[], None]:
        """
        Subscribe a handler to an event type

        Parameters
        ----------
        event_type : EventType or str
            Event type to subscribe to, or "*" for all events
        handler : callable
            Function to call when event occurs

        Returns
        -------
        callable
            Unsubscribe function
        """
        if event_type == "*":
            with self._lock:
                self._wildcard_handlers.append(handler)
            return lambda: self._wildcard_handlers.remove(handler)

        if isinstance(event_type, str):
            event_type = EventType[event_type]

        with self._lock:
            if event_type not in self._handlers:
                self._handlers[event_type] = []
            self._handlers[event_type].append(handler)

        def unsubscribe():
            with self._lock:
                if event_type in self._handlers:
                    self._handlers[event_type].remove(handler)

        return unsubscribe

    def subscribe_async(
        self,
        event_type: EventType | str,
        handler: AsyncEventHandler,
    ) -> Callable[[], None]:
        """
        Subscribe an async handler to an event type

        Parameters
        ----------
        event_type : EventType or str
            Event type to subscribe to, or "*" for all events
        handler : coroutine function
            Async function to call when event occurs

        Returns
        -------
        callable
            Unsubscribe function
        """
        if event_type == "*":
            with self._lock:
                self._async_wildcard_handlers.append(handler)
            return lambda: self._async_wildcard_handlers.remove(handler)

        if isinstance(event_type, str):
            event_type = EventType[event_type]

        with self._lock:
            if event_type not in self._async_handlers:
                self._async_handlers[event_type] = []
            self._async_handlers[event_type].append(handler)

        def unsubscribe():
            with self._lock:
                if event_type in self._async_handlers:
                    self._async_handlers[event_type].remove(handler)

        return unsubscribe

    def publish(
        self,
        event_type: EventType,
        data: dict | None = None,
        source: str = "unknown",
        correlation_id: str | None = None,
    ) -> Event:
        """
        Publish an event to all subscribers

        Parameters
        ----------
        event_type : EventType
            Type of event
        data : dict, optional
            Event payload
        source : str
            Component publishing the event
        correlation_id : str, optional
            ID to correlate with related events

        Returns
        -------
        Event
            The published event
        """
        event = Event(
            event_type=event_type,
            data=data or {},
            source=source,
            correlation_id=correlation_id,
        )

        # Add to history. High-volume telemetry skips history so it doesn't
        # crowd out the bounded deque of more meaningful events.
        if event_type not in _NO_HISTORY_TYPES:
            with self._lock:
                self._history.append(event)

        # Call sync handlers
        self._dispatch_sync(event)

        # Schedule async handlers
        self._dispatch_async(event)

        logger.debug("Published: %s", event)
        return event

    def publish_event(self, event: Event) -> Event:
        """
        Publish a pre-constructed event

        Parameters
        ----------
        event : Event
            Event to publish

        Returns
        -------
        Event
            The published event
        """
        if event.event_type not in _NO_HISTORY_TYPES:
            with self._lock:
                self._history.append(event)

        self._dispatch_sync(event)
        self._dispatch_async(event)

        logger.debug("Published: %s", event)
        return event

    def _dispatch_sync(self, event: Event):
        """Dispatch event to sync handlers"""
        handlers = []

        with self._lock:
            # Get type-specific handlers
            if event.event_type in self._handlers:
                handlers.extend(self._handlers[event.event_type])
            # Add wildcard handlers
            handlers.extend(self._wildcard_handlers)

        for handler in handlers:
            try:
                handler(event)
            except Exception as e:
                logger.error("Handler error for %s: %s", event, e)

    def _dispatch_async(self, event: Event):
        """Dispatch event to async handlers"""
        handlers = []

        with self._lock:
            if event.event_type in self._async_handlers:
                handlers.extend(self._async_handlers[event.event_type])
            handlers.extend(self._async_wildcard_handlers)

        if not handlers:
            return

        # Try to get running loop (if we're in async context)
        running_loop = None
        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            pass

        # Use running loop if available, otherwise fall back to cached loop
        loop = running_loop or self._event_loop
        if loop is None or loop.is_closed():
            return  # Can't dispatch async without a loop

        for handler in handlers:
            try:
                coro = handler(event)
                if asyncio.iscoroutine(coro):
                    if running_loop is not None:
                        # We're in an async context, can use ensure_future directly
                        asyncio.ensure_future(coro, loop=loop)
                    else:
                        # We're in sync context, need to schedule thread-safely
                        loop.call_soon_threadsafe(
                            lambda c=coro: asyncio.ensure_future(c, loop=loop)
                        )
            except Exception as e:
                logger.error("Async handler error for %s: %s", event, e)

    def set_event_loop(self, loop: asyncio.AbstractEventLoop):
        """Set the event loop to use for async handlers"""
        self._event_loop = loop

    def get_history(
        self,
        event_type: EventType | None = None,
        source: str | None = None,
        limit: int = 50,
    ) -> list[Event]:
        """
        Get recent event history

        Parameters
        ----------
        event_type : EventType, optional
            Filter by event type
        source : str, optional
            Filter by source
        limit : int
            Maximum events to return

        Returns
        -------
        list of Event
            Recent events (newest first)
        """
        with self._lock:
            events = list(self._history)

        # Filter
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        if source:
            events = [e for e in events if e.source == source]

        # Return newest first, limited
        return list(reversed(events))[:limit]

    def clear_history(self):
        """Clear event history"""
        with self._lock:
            self._history.clear()

    def get_handler_count(self, event_type: EventType | None = None) -> int:
        """Get count of registered handlers"""
        with self._lock:
            if event_type:
                sync = len(self._handlers.get(event_type, []))
                async_ = len(self._async_handlers.get(event_type, []))
                return sync + async_
            else:
                total = sum(len(h) for h in self._handlers.values())
                total += sum(len(h) for h in self._async_handlers.values())
                total += len(self._wildcard_handlers)
                total += len(self._async_wildcard_handlers)
                return total


# Global event bus instance
_global_bus: EventBus | None = None


def get_event_bus() -> EventBus:
    """Get or create the global event bus"""
    global _global_bus
    if _global_bus is None:
        _global_bus = EventBus()
    return _global_bus


def set_event_bus(bus: EventBus):
    """Set the global event bus"""
    global _global_bus
    _global_bus = bus


# Convenience functions for common operations
def emit(
    event_type: EventType,
    data: dict | None = None,
    source: str = "unknown",
) -> Event:
    """Emit an event on the global bus"""
    return get_event_bus().publish(event_type, data, source)


def on(event_type: EventType | str, handler: EventHandler) -> Callable[[], None]:
    """Subscribe to events on the global bus"""
    return get_event_bus().subscribe(event_type, handler)


# Decorator for event handlers
def handles(event_type: EventType | str):
    """
    Decorator to register a function as an event handler

    Usage:
        @handles(EventType.VOLUME_ACQUIRED)
        def on_volume(event):
            print(f"Got volume: {event.data}")
    """

    def decorator(func: EventHandler) -> EventHandler:
        get_event_bus().subscribe(event_type, func)
        return func

    return decorator
