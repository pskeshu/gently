"""
Core infrastructure for Gently microscope control system

Provides:
- DataStore: UID-based data persistence through Databroker
- EventBus: Async message passing between components
- Service: Base class and registry for services
"""

from .data_store import (
    DataReference,
    DataStore,
    DatabrokerStore,
    TiledStore,
    get_data_store,
    set_data_store,
)

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
    RPCService,
    HTTPService,
    get_service_registry,
    set_service_registry,
)

__all__ = [
    # Data store
    'DataReference',
    'DataStore',
    'DatabrokerStore',
    'TiledStore',
    'get_data_store',
    'set_data_store',
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
    'RPCService',
    'HTTPService',
    'get_service_registry',
    'set_service_registry',
]
