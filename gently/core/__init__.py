"""
Core infrastructure for Gently microscope control system

Provides:
- GentlyStore: Unified data persistence (SQLite + filesystem)
- EventBus: Async message passing between components
- Service: Base class and registry for services
"""

from .event_bus import (
    Event,
    EventType,
    EventBus,
    get_event_bus,
    set_event_bus,
    emit,
    on,
    handles,
)

from .service import (
    Service,
    ServiceState,
    ServiceInfo,
    ServiceRegistry,
    ServiceClient,

    get_service_registry,
    set_service_registry,
)

__all__ = [
    # Event bus
    'Event',
    'EventType',
    'EventBus',
    'get_event_bus',
    'set_event_bus',
    'emit',
    'on',
    'handles',
    # Service
    'Service',
    'ServiceState',
    'ServiceInfo',
    'ServiceRegistry',
    'ServiceClient',

    'get_service_registry',
    'set_service_registry',
]
