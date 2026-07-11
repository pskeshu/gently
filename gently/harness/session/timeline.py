"""
Timeline Manager for tracking timelapse and detection events

Provides:
- Event capture from EventBus
- Persistent storage (JSONL format)
- Filtering and querying by type, embryo, time range
- Horizontal timeline visualization support
"""

import json
import logging
import threading
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from gently.core.event_bus import Event, EventType, get_event_bus

logger = logging.getLogger(__name__)


@dataclass
class TimelineEvent:
    """
    A single event in the timeline

    Attributes
    ----------
    event_id : str
        Unique identifier (uuid[:8])
    event_type : str
        Category: "timelapse" | "detection"
    event_subtype : str
        Specific event: "started" | "volume_acquired" | "triggered" | etc.
    timestamp : datetime
        When the event occurred
    source : str
        Component that emitted the event
    embryo_id : str, optional
        Related embryo ID
    detector_name : str, optional
        Related detector name (for detection events)
    timepoint : int, optional
        Timepoint number
    confidence : str, optional
        Detection confidence level
    data : dict
        Additional event payload
    icon : str
        Display icon character
    severity : str
        Severity level: info | success | warning | error
    """

    event_id: str
    event_type: str
    event_subtype: str
    timestamp: datetime
    source: str
    session_id: str | None = None  # Session this event belongs to
    embryo_id: str | None = None
    detector_name: str | None = None
    timepoint: int | None = None
    confidence: str | None = None
    data: dict[str, Any] = field(default_factory=dict)
    icon: str = ">"
    severity: str = "info"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary"""
        return {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "event_subtype": self.event_subtype,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "session_id": self.session_id,
            "embryo_id": self.embryo_id,
            "detector_name": self.detector_name,
            "timepoint": self.timepoint,
            "confidence": self.confidence,
            "data": self.data,
            "icon": self.icon,
            "severity": self.severity,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TimelineEvent":
        """Deserialize from dictionary"""
        return cls(
            event_id=d["event_id"],
            event_type=d["event_type"],
            event_subtype=d["event_subtype"],
            timestamp=datetime.fromisoformat(d["timestamp"]),
            source=d.get("source", "unknown"),
            session_id=d.get("session_id"),
            embryo_id=d.get("embryo_id"),
            detector_name=d.get("detector_name"),
            timepoint=d.get("timepoint"),
            confidence=d.get("confidence"),
            data=d.get("data", {}),
            icon=d.get("icon", ">"),
            severity=d.get("severity", "info"),
        )

    @property
    def short_label(self) -> str:
        """Short label for timeline display (e.g., 'TL', 'DET')"""
        if self.event_type == "timelapse":
            return "TL"
        elif self.event_type == "detection":
            return "DET"
        else:
            return "SYS"

    @property
    def description(self) -> str:
        """Human-readable description of the event"""
        if self.event_type == "timelapse":
            if self.event_subtype == "started":
                embryos = self.data.get("embryo_ids", [])
                return f"Started timelapse with {len(embryos)} embryo(s)"
            elif self.event_subtype == "volume_acquired":
                return f"{self.embryo_id} @ t={self.timepoint}"
            elif self.event_subtype == "completed":
                total = self.data.get("total_timepoints", "?")
                return f"Completed ({total} timepoints)"
            elif self.event_subtype == "failed":
                return f"Failed: {self.data.get('error', 'unknown error')}"
        elif self.event_type == "detection":
            detected = self.data.get("detected", False)
            status = "Detected" if detected else "Not detected"
            conf = f" ({self.confidence})" if self.confidence else ""
            return f"{self.detector_name} on {self.embryo_id} - {status}{conf}"
        return self.event_subtype


# Mapping from EventBus EventType to TimelineEvent properties
EVENT_MAPPING = {
    EventType.ACQUISITION_STARTED: {
        "event_type": "timelapse",
        "event_subtype": "started",
        "icon": ">",
        "severity": "info",
    },
    EventType.VOLUME_ACQUIRED: {
        "event_type": "timelapse",
        "event_subtype": "volume_acquired",
        "icon": "+",
        "severity": "success",
    },
    EventType.ACQUISITION_COMPLETED: {
        "event_type": "timelapse",
        "event_subtype": "completed",
        "icon": "+",
        "severity": "success",
    },
    EventType.ACQUISITION_STOPPED: {
        "event_type": "timelapse",
        "event_subtype": "stopped",
        "icon": "-",
        "severity": "info",
    },
    EventType.ACQUISITION_FAILED: {
        "event_type": "timelapse",
        "event_subtype": "failed",
        "icon": "x",
        "severity": "error",
    },
    EventType.DETECTOR_EVALUATED: {
        "event_type": "detection",
        "event_subtype": "evaluated",
        "icon": "?",
        "severity": "info",
    },
    EventType.DETECTION_TRIGGERED: {
        "event_type": "detection",
        "event_subtype": "triggered",
        "icon": "!",
        "severity": "success",
    },
    EventType.HATCHING_DETECTED: {
        "event_type": "detection",
        "event_subtype": "hatching",
        "icon": "+",
        "severity": "success",
    },
    # Strategy / experiment view persistence — these were already emitted on
    # the EventBus but weren't being captured to timeline.jsonl, so the
    # swimlane view had no event history to replay.
    EventType.EMBRYO_CADENCE_CHANGED: {
        "event_type": "timelapse",
        "event_subtype": "cadence_changed",
        "icon": "~",
        "severity": "info",
    },
    EventType.POWER_RAMP_STEP: {
        "event_type": "timelapse",
        "event_subtype": "power_changed",
        "icon": "*",
        "severity": "info",
    },
    EventType.TRIGGER_FIRED: {
        "event_type": "timelapse",
        "event_subtype": "trigger_fired",
        "icon": "<>",
        "severity": "info",
    },
    EventType.BURST_QUEUED: {
        "event_type": "timelapse",
        "event_subtype": "burst_queued",
        "icon": "q",
        "severity": "info",
    },
    EventType.BURST_START: {
        "event_type": "timelapse",
        "event_subtype": "burst_started",
        "icon": "^",
        "severity": "info",
    },
    EventType.BURST_COMPLETE: {
        "event_type": "timelapse",
        "event_subtype": "burst_completed",
        "icon": "v",
        "severity": "success",
    },
    EventType.TEMPERATURE_SETPOINT_CHANGED: {
        "event_type": "temperature",
        "event_subtype": "setpoint_changed",
        "icon": "T",
        "severity": "info",
    },
    EventType.TEMP_PROTOCOL_STARTED: {
        "event_type": "tactic",
        "event_subtype": "temp_protocol_started",
        "icon": "~",
        "severity": "info",
    },
    EventType.TEMP_PROTOCOL_COMPLETED: {
        "event_type": "tactic",
        "event_subtype": "temp_protocol_completed",
        "icon": "+",
        "severity": "success",
    },
}


class TimelineManager:
    """
    Manages timeline events with persistence and filtering

    Features:
    - Subscribes to EventBus for real-time event capture
    - Persists events to JSONL file for cross-session history
    - In-memory ring buffer for fast recent queries
    - Filtering by type, embryo, time range, session
    """

    def __init__(
        self,
        storage_path: Path | None = None,
        max_events: int = 1000,
        session_id: str | None = None,
    ):
        """
        Parameters
        ----------
        storage_path : Path, optional
            Directory for timeline.jsonl storage. If None, no persistence.
        max_events : int
            Maximum events to keep in memory
        session_id : str, optional
            Current session ID for tagging events
        """
        self._storage_path = Path(storage_path) if storage_path else None
        self._max_events = max_events
        self._session_id = session_id
        self._events: deque[TimelineEvent] = deque(maxlen=max_events)
        self._lock = threading.RLock()
        self._unsubscribers: list[Callable] = []
        self._started = False

        # Load existing events from storage
        if self._storage_path:
            self._load_from_file()

    def set_session_id(self, session_id: str) -> None:
        """Update the current session ID"""
        self._session_id = session_id

    @property
    def storage_file(self) -> Path | None:
        """Path to the timeline JSONL file"""
        if self._storage_path:
            return self._storage_path / "timeline.jsonl"
        return None

    def start(self) -> None:
        """Subscribe to EventBus events"""
        if self._started:
            return

        bus = get_event_bus()

        # Subscribe to all mapped event types
        for event_type in EVENT_MAPPING.keys():
            unsub = bus.subscribe(event_type, self._on_event)
            self._unsubscribers.append(unsub)

        self._started = True
        logger.info("TimelineManager started - subscribed to EventBus")

    def stop(self) -> None:
        """Unsubscribe from EventBus events"""
        for unsub in self._unsubscribers:
            try:
                unsub()
            except Exception as e:
                logger.warning(f"Error unsubscribing: {e}")
        self._unsubscribers.clear()
        self._started = False
        logger.info("TimelineManager stopped")

    def _on_event(self, event: Event) -> None:
        """Handle incoming EventBus event"""
        mapping = EVENT_MAPPING.get(event.event_type)
        if not mapping:
            return

        # Extract common fields from event data
        data = event.data or {}

        timeline_event = TimelineEvent(
            event_id=event.event_id,
            event_type=mapping["event_type"],
            event_subtype=mapping["event_subtype"],
            timestamp=event.timestamp,
            source=event.source,
            session_id=self._session_id,  # Tag with current session
            embryo_id=data.get("embryo_id"),
            detector_name=data.get("detector_name"),
            timepoint=data.get("timepoint"),
            confidence=data.get("confidence"),
            data=data,
            icon=mapping["icon"],
            severity=mapping["severity"],
        )

        self.add_event(timeline_event)

    def add_event(self, event: TimelineEvent) -> None:
        """
        Add an event to the timeline

        Parameters
        ----------
        event : TimelineEvent
            Event to add
        """
        with self._lock:
            self._events.append(event)

        # Persist to file
        self._persist_event(event)

        logger.debug(f"Timeline event: {event.event_type}/{event.event_subtype}")

    def get_events(
        self,
        event_type: str | None = None,
        embryo_id: str | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        session_id: str | None = "current",
        limit: int = 50,
    ) -> list[TimelineEvent]:
        """
        Get filtered events from timeline

        Parameters
        ----------
        event_type : str, optional
            Filter by type ("timelapse" or "detection")
        embryo_id : str, optional
            Filter by embryo
        since : datetime, optional
            Only events after this time
        until : datetime, optional
            Only events before this time
        session_id : str, optional
            Filter by session. Use "current" (default) for current session,
            None or "all" for all sessions, or a specific session ID.
        limit : int
            Maximum events to return

        Returns
        -------
        list of TimelineEvent
            Events matching filters (oldest first)
        """
        with self._lock:
            events = list(self._events)

        # Apply session filter (default: current session only)
        if session_id == "current":
            session_id = self._session_id
        if session_id and session_id != "all":
            events = [e for e in events if e.session_id == session_id]

        # Apply other filters
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        if embryo_id:
            events = [e for e in events if e.embryo_id == embryo_id]
        if since:
            events = [e for e in events if e.timestamp >= since]
        if until:
            events = [e for e in events if e.timestamp <= until]

        # Return limited, oldest first (chronological)
        return events[-limit:] if len(events) > limit else events

    def get_time_range(self) -> tuple[datetime | None, datetime | None]:
        """
        Get the time range of events in the timeline

        Returns
        -------
        tuple
            (earliest_timestamp, latest_timestamp) or (None, None) if empty
        """
        with self._lock:
            if not self._events:
                return None, None
            events = list(self._events)

        return events[0].timestamp, events[-1].timestamp

    def clear_events(self, before: datetime | None = None) -> int:
        """
        Clear events from timeline

        Parameters
        ----------
        before : datetime, optional
            Only clear events before this time. If None, clear all.

        Returns
        -------
        int
            Number of events cleared
        """
        with self._lock:
            if before is None:
                count = len(self._events)
                self._events.clear()
            else:
                old_count = len(self._events)
                self._events = deque(
                    (e for e in self._events if e.timestamp >= before),
                    maxlen=self._max_events,
                )
                count = old_count - len(self._events)

        # Rewrite storage file if we have a storage path
        if self._storage_path and count > 0:
            self._rewrite_storage()

        logger.info(f"Cleared {count} timeline events")
        return count

    def _load_from_file(self) -> None:
        """Load events from JSONL storage file"""
        if not self.storage_file or not self.storage_file.exists():
            return

        try:
            with open(self.storage_file, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    # Only parse lines that look like JSON objects
                    if line and line.startswith("{"):
                        try:
                            data = json.loads(line)
                            event = TimelineEvent.from_dict(data)
                            self._events.append(event)
                        except (json.JSONDecodeError, KeyError) as e:
                            logger.warning(f"Skipping invalid timeline entry: {e}")
            logger.info(f"Loaded {len(self._events)} events from timeline storage")
        except Exception as e:
            logger.error(f"Error loading timeline: {e}")

    def _persist_event(self, event: TimelineEvent) -> None:
        """Append event to JSONL storage file"""
        if not self._storage_path:
            return

        try:
            # Ensure directory exists
            self._storage_path.mkdir(parents=True, exist_ok=True)

            assert self.storage_file is not None  # implied by _storage_path guard above
            with open(self.storage_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(event.to_dict()) + "\n")
        except Exception as e:
            logger.error(f"Error persisting timeline event: {e}")

    def _rewrite_storage(self) -> None:
        """Rewrite storage file with current events"""
        if not self._storage_path:
            return

        try:
            self._storage_path.mkdir(parents=True, exist_ok=True)

            with self._lock:
                events = list(self._events)

            assert self.storage_file is not None  # implied by _storage_path guard above
            with open(self.storage_file, "w", encoding="utf-8") as f:
                for event in events:
                    f.write(json.dumps(event.to_dict()) + "\n")
        except Exception as e:
            logger.error(f"Error rewriting timeline storage: {e}")

    def __len__(self) -> int:
        """Number of events in timeline"""
        with self._lock:
            return len(self._events)


def parse_time_delta(s: str) -> timedelta | None:
    """
    Parse a time delta string like "1h", "30m", "2d"

    Parameters
    ----------
    s : str
        Time string (e.g., "1h", "30m", "2d", "1w")

    Returns
    -------
    timedelta or None
        Parsed duration, or None if invalid
    """
    s = s.strip().lower()
    if not s:
        return None

    try:
        if s.endswith("m"):
            return timedelta(minutes=int(s[:-1]))
        elif s.endswith("h"):
            return timedelta(hours=int(s[:-1]))
        elif s.endswith("d"):
            return timedelta(days=int(s[:-1]))
        elif s.endswith("w"):
            return timedelta(weeks=int(s[:-1]))
        else:
            # Try parsing as minutes
            return timedelta(minutes=int(s))
    except ValueError:
        return None
