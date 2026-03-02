"""
Gently Mesh — LAN peer discovery and status exchange.

Provides automatic zero-config discovery of Gently instances on the
local network via UDP broadcast heartbeats, with HTTP status fetching.
"""

from .audit import AuditEvent, MeshAuditLog
from .mesh_service import MeshService
from .models import (
    DatasetAdvertisement,
    GpuInfo,
    PeerCapability,
    PeerInfo,
    PeerRole,
    PeerStatus,
    PersistedPeer,
)
from .routes import register_mesh_routes

__all__ = [
    "AuditEvent",
    "DatasetAdvertisement",
    "GpuInfo",
    "MeshAuditLog",
    "MeshService",
    "PeerCapability",
    "PeerInfo",
    "PeerRole",
    "PeerStatus",
    "PersistedPeer",
    "register_mesh_routes",
]
