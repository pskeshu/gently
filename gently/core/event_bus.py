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
import time
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional, Set, Union
from collections import deque
import threading
import uuid

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
    ACQUISITION_FAILED = auto()
    VOLUME_ACQUIRED = auto()
    IMAGE_ACQUIRED = auto()

    # Embryo events
    EMBRYO_DETECTED = auto()
    EMBRYO_CENTERED = auto()
    EMBRYO_CALIBRATED = auto()
    EMBRYO_SKIPPED = auto()

    # Analysis events
    ANALYSIS_STARTED = auto()
    ANALYSIS_COMPLETED = auto()
    DETECTION_TRIGGERED = auto()
    HATCHING_DETECTED = auto()

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

    # System events
    ERROR_OCCURRED = auto()
    WARNING_ISSUED = auto()
    STATUS_CHANGED = auto()

    # Data events
    DATA_STORED = auto()
    DATA_RETRIEVED = auto()

    # User interaction events
    USER_INPUT = auto()
    USER_COMMAND = auto()


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
    data: Dict[str, Any] = field(default_factory=dict)
    source: str = "unknown"
    timestamp: datetime = field(default_factory=datetime.now)
    event_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    correlation_id: Optional[str] = None

    def __str__(self) -> str:
        return f"Event({self.event_type.name}, source={self.source}, id={self.event_id})"

    def to_dict(self) -> Dict:
        """Serialize for storage/transmission"""
        return {
            'event_type': self.event_type.name,
            'data': self.data,
            'source': self.source,
            'timestamp': self.timestamp.isoformat(),
            'event_id': self.event_id,
            'correlation_id': self.correlation_id,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> 'Event':
        """Deserialize from dict"""
        return cls(
            event_type=EventType[d['event_type']],
            data=d.get('data', {}),
            source=d.get('source', 'unknown'),
            timestamp=datetime.fromisoformat(d['timestamp']) if 'timestamp' in d else datetime.now(),
            event_id=d.get('event_id', str(uuid.uuid4())[:8]),
            correlation_id=d.get('correlation_id'),
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
        self._handlers: Dict[EventType, List[EventHandler]] = {}
        self._async_handlers: Dict[EventType, List[AsyncEventHandler]] = {}
        self._wildcard_handlers: List[EventHandler] = []
        self._async_wildcard_handlers: List[AsyncEventHandler] = []
        self._history: deque = deque(maxlen=history_size)
        self._lock = threading.RLock()
        self._event_loop: Optional[asyncio.AbstractEventLoop] = None

    def subscribe(
        self,
        event_type: Union[EventType, str],
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
        event_type: Union[EventType, str],
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
        data: Optional[Dict] = None,
        source: str = "unknown",
        correlation_id: Optional[str] = None,
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

        # Add to history
        with self._lock:
            self._history.append(event)

        # Call sync handlers
        self._dispatch_sync(event)

        # Schedule async handlers
        self._dispatch_async(event)

        logger.debug(f"Published: {event}")
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
        with self._lock:
            self._history.append(event)

        self._dispatch_sync(event)
        self._dispatch_async(event)

        logger.debug(f"Published: {event}")
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
                logger.error(f"Handler error for {event}: {e}")

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
                logger.error(f"Async handler error for {event}: {e}")

    def set_event_loop(self, loop: asyncio.AbstractEventLoop):
        """Set the event loop to use for async handlers"""
        self._event_loop = loop

    def get_history(
        self,
        event_type: Optional[EventType] = None,
        source: Optional[str] = None,
        limit: int = 50,
    ) -> List[Event]:
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

    def get_handler_count(self, event_type: Optional[EventType] = None) -> int:
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
_global_bus: Optional[EventBus] = None


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
    data: Optional[Dict] = None,
    source: str = "unknown",
) -> Event:
    """Emit an event on the global bus"""
    return get_event_bus().publish(event_type, data, source)


def on(event_type: Union[EventType, str], handler: EventHandler) -> Callable[[], None]:
    """Subscribe to events on the global bus"""
    return get_event_bus().subscribe(event_type, handler)


# Decorator for event handlers
def handles(event_type: Union[EventType, str]):
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
