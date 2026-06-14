"""
Core infrastructure for Gently microscope control system

Provides:
- FileStore: Unified file-based data persistence
- EventBus: Async message passing between components
- Service: Base class and registry for services
"""

from .event_bus import (
    Event,
    EventBus,
    EventType,
    emit,
    get_event_bus,
    handles,
    on,
    set_event_bus,
)
from .service import (
    Service,
    ServiceClient,
    ServiceInfo,
    ServiceRegistry,
    ServiceState,
    get_service_registry,
    set_service_registry,
)

__all__ = [
    # Event bus
    "Event",
    "EventType",
    "EventBus",
    "get_event_bus",
    "set_event_bus",
    "emit",
    "on",
    "handles",
    # Service
    "Service",
    "ServiceState",
    "ServiceInfo",
    "ServiceRegistry",
    "ServiceClient",
    "get_service_registry",
    "set_service_registry",
]
