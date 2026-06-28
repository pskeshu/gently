"""
Service Infrastructure for Gently

Provides a unified framework for service management:
- Service base class with lifecycle hooks
- Service registry for discovery
- Health checking and monitoring
- Event bus integration
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any

import aiohttp

from .event_bus import EventType, get_event_bus

logger = logging.getLogger(__name__)


class ServiceState(Enum):
    """Service lifecycle states"""

    CREATED = auto()
    STARTING = auto()
    RUNNING = auto()
    STOPPING = auto()
    STOPPED = auto()
    ERROR = auto()


@dataclass
class ServiceInfo:
    """Information about a registered service"""

    name: str
    service_type: str
    host: str = "localhost"
    port: int | None = None
    state: ServiceState = ServiceState.CREATED
    started_at: datetime | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    health_check_url: str | None = None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "service_type": self.service_type,
            "host": self.host,
            "port": self.port,
            "state": self.state.name,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "metadata": self.metadata,
            "health_check_url": self.health_check_url,
        }


class Service(ABC):
    """
    Base class for all Gently services

    Provides:
    - Lifecycle management (start, stop, health check)
    - Event bus integration
    - Service registration
    """

    def __init__(
        self,
        name: str,
        service_type: str = "generic",
        host: str = "localhost",
        port: int | None = None,
    ):
        self.name = name
        self.service_type = service_type
        self.host = host
        self.port = port
        self._state = ServiceState.CREATED
        self._started_at: datetime | None = None
        self._event_bus = get_event_bus()
        self._metadata: dict[str, Any] = {}

    @property
    def state(self) -> ServiceState:
        return self._state

    @property
    def info(self) -> ServiceInfo:
        return ServiceInfo(
            name=self.name,
            service_type=self.service_type,
            host=self.host,
            port=self.port,
            state=self._state,
            started_at=self._started_at,
            metadata=self._metadata,
        )

    async def start(self):
        """Start the service"""
        if self._state == ServiceState.RUNNING:
            logger.warning(f"Service {self.name} already running")
            return

        self._state = ServiceState.STARTING
        self._emit_event(EventType.STATUS_CHANGED, {"state": "starting"})

        try:
            await self.on_start()
            self._state = ServiceState.RUNNING
            self._started_at = datetime.now()
            self._emit_event(EventType.STATUS_CHANGED, {"state": "running"})
            logger.info(f"Service {self.name} started")

        except Exception as e:
            self._state = ServiceState.ERROR
            self._emit_event(EventType.ERROR_OCCURRED, {"error": str(e)})
            logger.error(f"Service {self.name} failed to start: {e}")
            raise

    async def stop(self):
        """Stop the service"""
        if self._state in (ServiceState.STOPPED, ServiceState.CREATED):
            return

        self._state = ServiceState.STOPPING
        self._emit_event(EventType.STATUS_CHANGED, {"state": "stopping"})

        try:
            await self.on_stop()
            self._state = ServiceState.STOPPED
            self._emit_event(EventType.STATUS_CHANGED, {"state": "stopped"})
            logger.info(f"Service {self.name} stopped")

        except Exception as e:
            self._state = ServiceState.ERROR
            logger.error(f"Service {self.name} failed to stop cleanly: {e}")

    async def health_check(self) -> dict:
        """Check service health"""
        return {
            "name": self.name,
            "state": self._state.name,
            "healthy": self._state == ServiceState.RUNNING,
            "uptime_seconds": (
                (datetime.now() - self._started_at).total_seconds() if self._started_at else 0
            ),
        }

    @abstractmethod
    async def on_start(self):
        """Called when service starts - implement in subclass"""
        pass

    @abstractmethod
    async def on_stop(self):
        """Called when service stops - implement in subclass"""
        pass

    def _emit_event(self, event_type: EventType, data: dict):
        """Emit event on the bus"""
        self._event_bus.publish(
            event_type=event_type,
            data={"service": self.name, **data},
            source=f"service:{self.name}",
        )


class ServiceRegistry:
    """
    Registry for service discovery

    Features:
    - Register/unregister services
    - Find services by name or type
    - Health monitoring
    """

    def __init__(self):
        self._services: dict[str, Service] = {}
        self._service_info: dict[str, ServiceInfo] = {}

    def register(self, service: Service):
        """Register a service"""
        self._services[service.name] = service
        self._service_info[service.name] = service.info
        logger.info(f"Registered service: {service.name} ({service.service_type})")

    def register_info(self, info: ServiceInfo):
        """Register service info (for remote services)"""
        self._service_info[info.name] = info
        logger.info(f"Registered service info: {info.name} ({info.service_type})")

    def unregister(self, name: str):
        """Unregister a service"""
        if name in self._services:
            del self._services[name]
        if name in self._service_info:
            del self._service_info[name]
        logger.info(f"Unregistered service: {name}")

    def get(self, name: str) -> Service | None:
        """Get service by name"""
        return self._services.get(name)

    def get_info(self, name: str) -> ServiceInfo | None:
        """Get service info by name"""
        if name in self._services:
            return self._services[name].info
        return self._service_info.get(name)

    def find_by_type(self, service_type: str) -> list[ServiceInfo]:
        """Find all services of a given type"""
        results = []
        for _name, service in self._services.items():
            if service.service_type == service_type:
                results.append(service.info)
        for name, info in self._service_info.items():
            if info.service_type == service_type and name not in self._services:
                results.append(info)
        return results

    def list_all(self) -> list[ServiceInfo]:
        """List all registered services"""
        seen = set()
        results = []
        for name, service in self._services.items():
            results.append(service.info)
            seen.add(name)
        for name, info in self._service_info.items():
            if name not in seen:
                results.append(info)
        return results

    async def health_check_all(self) -> dict[str, dict]:
        """Check health of all services"""
        results = {}
        for name, service in self._services.items():
            try:
                results[name] = await service.health_check()
            except Exception as e:
                results[name] = {
                    "name": name,
                    "state": "ERROR",
                    "healthy": False,
                    "error": str(e),
                }
        return results

    async def start_all(self):
        """Start all registered services"""
        for _name, service in self._services.items():
            if service.state == ServiceState.CREATED:
                await service.start()

    async def stop_all(self):
        """Stop all registered services"""
        for _name, service in self._services.items():
            if service.state == ServiceState.RUNNING:
                await service.stop()


# Global registry
_global_registry: ServiceRegistry | None = None


def get_service_registry() -> ServiceRegistry:
    """Get or create the global service registry"""
    global _global_registry
    if _global_registry is None:
        _global_registry = ServiceRegistry()
    return _global_registry


def set_service_registry(registry: ServiceRegistry):
    """Set the global service registry"""
    global _global_registry
    _global_registry = registry


# =============================================================================
# Service Client for Communication
# =============================================================================


class ServiceClient:
    """
    Unified client for communicating with services

    Handles:
    - Service discovery via registry
    - Connection management
    - Protocol abstraction (HTTP)
    """

    def __init__(self, registry: ServiceRegistry | None = None):
        self._registry = registry or get_service_registry()
        self._connections: dict[str, Any] = {}

    async def connect(self, service_name: str) -> Any:
        """
        Connect to a service by name

        Returns a connection object appropriate for the service type.
        """
        if service_name in self._connections:
            return self._connections[service_name]

        info = self._registry.get_info(service_name)
        if not info:
            raise ValueError(f"Service not found: {service_name}")

        if info.service_type == "http":
            conn = await self._connect_http(info)
        else:
            raise ValueError(f"Unknown service type: {info.service_type}")

        self._connections[service_name] = conn
        return conn

    async def _connect_http(self, info: ServiceInfo):
        """Connect to HTTP service"""
        base_url = f"http://{info.host}:{info.port}"
        timeout = aiohttp.ClientTimeout(total=30.0)
        session = aiohttp.ClientSession(timeout=timeout)
        return {"base_url": base_url, "session": session}

    async def disconnect(self, service_name: str):
        """Disconnect from a service"""
        if service_name in self._connections:
            conn = self._connections.pop(service_name)
            if isinstance(conn, dict) and "session" in conn:
                await conn["session"].close()
            elif hasattr(conn, "close"):
                if asyncio.iscoroutinefunction(conn.close):
                    await conn.close()
                else:
                    conn.close()

    async def disconnect_all(self):
        """Disconnect from all services"""
        for name in list(self._connections.keys()):
            await self.disconnect(name)

    async def call(self, service_name: str, method: str, *args, **kwargs) -> Any:
        """
        Call a method on a service

        Parameters
        ----------
        service_name : str
            Name of service to call
        method : str
            Method name to call
        *args, **kwargs
            Arguments to pass to method

        Returns
        -------
        any
            Method result
        """
        conn = await self.connect(service_name)

        info = self._registry.get_info(service_name)
        if info is None:
            raise ValueError(f"Unknown service: {service_name}")

        if info.service_type == "rpc":
            # RPC call
            func = getattr(conn, method)
            return await asyncio.to_thread(func, *args, **kwargs)

        elif info.service_type == "http":
            # HTTP call (assume POST to /{method})
            url = f"{conn['base_url']}/{method}"
            async with conn["session"].post(url, json=kwargs) as resp:
                resp.raise_for_status()
                return await resp.json()

    def __contains__(self, service_name: str) -> bool:
        return service_name in self._connections
