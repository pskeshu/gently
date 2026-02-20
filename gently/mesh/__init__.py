"""
Gently Mesh — LAN peer discovery and status exchange.

Provides automatic zero-config discovery of Gently instances on the
local network via UDP broadcast heartbeats, with HTTP status fetching.
"""

from .mesh_service import MeshService
from .models import PeerCapability, PeerInfo, PeerStatus
from .routes import register_mesh_routes

__all__ = [
    "MeshService",
    "PeerCapability",
    "PeerInfo",
    "PeerStatus",
    "register_mesh_routes",
]
