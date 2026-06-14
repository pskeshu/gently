"""
Mesh data models for peer discovery and status exchange.

Defines the core dataclasses:
- PeerCapability: what a node can do
- PeerStatus: current runtime state
- PeerInfo: complete peer record
- GpuInfo: GPU device details
- DatasetAdvertisement: what data a node has available
- PeerRole: dynamic roles for a node
- PersistedPeer: verse map entry that survives restarts
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from ..settings import settings


class PeerRole(str, Enum):
    """Dynamic roles a gently node can fill."""

    MICROSCOPE_CONTROLLER = "microscope_controller"
    ML_TRAINER = "ml_trainer"
    DATA_SERVER = "data_server"
    PLANNER = "planner"


@dataclass
class GpuInfo:
    """Details about a single GPU device."""

    device_index: int = 0
    name: str = ""
    vram_gb: float = 0.0
    compute_capability: str = ""
    utilization_pct: float = 0.0
    memory_used_gb: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "device_index": self.device_index,
            "name": self.name,
            "vram_gb": self.vram_gb,
            "compute_capability": self.compute_capability,
            "utilization_pct": self.utilization_pct,
            "memory_used_gb": self.memory_used_gb,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "GpuInfo":
        return cls(
            device_index=d.get("device_index", 0),
            name=d.get("name", ""),
            vram_gb=d.get("vram_gb", 0.0),
            compute_capability=d.get("compute_capability", ""),
            utilization_pct=d.get("utilization_pct", 0.0),
            memory_used_gb=d.get("memory_used_gb", 0.0),
        )


@dataclass
class DatasetAdvertisement:
    """Advertises what data a node has available for training."""

    session_id: str = ""
    session_name: str = ""
    embryo_count: int = 0
    volume_count: int = 0
    has_ground_truth: bool = False
    ground_truth_count: int = 0
    stages_covered: list[str] = field(default_factory=list)
    total_size_gb: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "session_name": self.session_name,
            "embryo_count": self.embryo_count,
            "volume_count": self.volume_count,
            "has_ground_truth": self.has_ground_truth,
            "ground_truth_count": self.ground_truth_count,
            "stages_covered": self.stages_covered,
            "total_size_gb": self.total_size_gb,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "DatasetAdvertisement":
        return cls(
            session_id=d.get("session_id", ""),
            session_name=d.get("session_name", ""),
            embryo_count=d.get("embryo_count", 0),
            volume_count=d.get("volume_count", 0),
            has_ground_truth=d.get("has_ground_truth", False),
            ground_truth_count=d.get("ground_truth_count", 0),
            stages_covered=d.get("stages_covered", []),
            total_size_gb=d.get("total_size_gb", 0.0),
        )


@dataclass
class PeerCapability:
    """What a Gently node can do."""

    has_microscope: bool = False
    has_sam: bool = False
    has_gpu: bool = False
    gpu_name: str = ""
    gpu_vram_gb: float = 0.0
    storage_free_gb: float = 0.0
    tool_categories: list[str] = field(default_factory=list)
    organism: str = ""
    hardware_profile: str = ""
    # Enhanced capability fields (backward-compatible — old peers get defaults)
    gpus: list[GpuInfo] = field(default_factory=list)
    roles: list[str] = field(default_factory=list)
    datasets: list[DatasetAdvertisement] = field(default_factory=list)
    microscope_connected: bool = False
    cpu_cores: int = 0
    ram_gb: float = 0.0
    storage_total_gb: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "has_microscope": self.has_microscope,
            "has_sam": self.has_sam,
            "has_gpu": self.has_gpu,
            "gpu_name": self.gpu_name,
            "gpu_vram_gb": self.gpu_vram_gb,
            "storage_free_gb": self.storage_free_gb,
            "tool_categories": self.tool_categories,
            "organism": self.organism,
            "hardware_profile": self.hardware_profile,
            "gpus": [g.to_dict() for g in self.gpus],
            "roles": self.roles,
            "datasets": [d.to_dict() for d in self.datasets],
            "microscope_connected": self.microscope_connected,
            "cpu_cores": self.cpu_cores,
            "ram_gb": self.ram_gb,
            "storage_total_gb": self.storage_total_gb,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PeerCapability":
        return cls(
            has_microscope=d.get("has_microscope", False),
            has_sam=d.get("has_sam", False),
            has_gpu=d.get("has_gpu", False),
            gpu_name=d.get("gpu_name", ""),
            gpu_vram_gb=d.get("gpu_vram_gb", 0.0),
            storage_free_gb=d.get("storage_free_gb", 0.0),
            tool_categories=d.get("tool_categories", []),
            organism=d.get("organism", ""),
            hardware_profile=d.get("hardware_profile", ""),
            gpus=[GpuInfo.from_dict(g) for g in d.get("gpus", [])],
            roles=d.get("roles", []),
            datasets=[DatasetAdvertisement.from_dict(ds) for ds in d.get("datasets", [])],
            microscope_connected=d.get("microscope_connected", False),
            cpu_cores=d.get("cpu_cores", 0),
            ram_gb=d.get("ram_gb", 0.0),
            storage_total_gb=d.get("storage_total_gb", 0.0),
        )


@dataclass
class PeerStatus:
    """Current runtime state of a Gently node."""

    session_id: str = ""
    acquisition_status: str = "idle"
    embryo_count: int = 0
    total_timepoints: int = 0
    uptime_seconds: float = 0.0
    agent_mode: str = "run"
    active_plan: str = ""
    version: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "acquisition_status": self.acquisition_status,
            "embryo_count": self.embryo_count,
            "total_timepoints": self.total_timepoints,
            "uptime_seconds": self.uptime_seconds,
            "agent_mode": self.agent_mode,
            "active_plan": self.active_plan,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PeerStatus":
        return cls(
            session_id=d.get("session_id", ""),
            acquisition_status=d.get("acquisition_status", "idle"),
            embryo_count=d.get("embryo_count", 0),
            total_timepoints=d.get("total_timepoints", 0),
            uptime_seconds=d.get("uptime_seconds", 0.0),
            agent_mode=d.get("agent_mode", "run"),
            active_plan=d.get("active_plan", ""),
            version=d.get("version", ""),
        )


@dataclass
class PeerInfo:
    """Complete record for a discovered peer."""

    instance_id: str = ""
    hostname: str = ""
    ip_address: str = ""
    viz_port: int = field(default_factory=lambda: settings.network.viz_port)
    capabilities: PeerCapability = field(default_factory=PeerCapability)
    status: PeerStatus = field(default_factory=PeerStatus)
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    is_self: bool = False
    is_trusted: bool = False
    tls_enabled: bool = False
    udp_verified: bool = False

    @property
    def base_url(self) -> str:
        scheme = "https" if self.tls_enabled else "http"
        return f"{scheme}://{self.ip_address}:{self.viz_port}"

    @property
    def is_stale(self) -> bool:
        """True if no heartbeat beyond the stale threshold."""
        return (time.time() - self.last_seen) > settings.mesh.stale_threshold_s

    @property
    def is_dead(self) -> bool:
        """True if no heartbeat beyond the dead threshold."""
        return (time.time() - self.last_seen) > settings.mesh.dead_threshold_s

    def to_dict(self) -> dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "hostname": self.hostname,
            "ip_address": self.ip_address,
            "viz_port": self.viz_port,
            "capabilities": self.capabilities.to_dict(),
            "status": self.status.to_dict(),
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "is_self": self.is_self,
            "is_trusted": self.is_trusted,
            "tls_enabled": self.tls_enabled,
            "udp_verified": self.udp_verified,
            "base_url": self.base_url,
            "is_stale": self.is_stale,
            "is_dead": self.is_dead,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PeerInfo":
        return cls(
            instance_id=d.get("instance_id", ""),
            hostname=d.get("hostname", ""),
            ip_address=d.get("ip_address", ""),
            viz_port=d.get("viz_port", settings.network.viz_port),
            capabilities=PeerCapability.from_dict(d.get("capabilities", {})),
            status=PeerStatus.from_dict(d.get("status", {})),
            first_seen=d.get("first_seen", time.time()),
            last_seen=d.get("last_seen", time.time()),
            is_self=d.get("is_self", False),
            is_trusted=d.get("is_trusted", False),
            tls_enabled=d.get("tls_enabled", False),
            udp_verified=d.get("udp_verified", False),
        )


@dataclass
class PersistedPeer:
    """Verse map entry that survives restarts.

    Wraps PeerInfo with persistence fields for the topology map.
    """

    instance_id: str = ""
    hostname: str = ""
    ip_address: str = ""
    viz_port: int = field(default_factory=lambda: settings.network.viz_port)
    capabilities: PeerCapability = field(default_factory=PeerCapability)
    status: PeerStatus = field(default_factory=PeerStatus)
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    is_trusted: bool = False
    tls_enabled: bool = False
    # Persistence fields
    online: bool = True
    last_online: float = field(default_factory=time.time)
    roles: list[str] = field(default_factory=list)
    datasets: list[DatasetAdvertisement] = field(default_factory=list)

    @property
    def base_url(self) -> str:
        scheme = "https" if self.tls_enabled else "http"
        return f"{scheme}://{self.ip_address}:{self.viz_port}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "hostname": self.hostname,
            "ip_address": self.ip_address,
            "viz_port": self.viz_port,
            "capabilities": self.capabilities.to_dict(),
            "status": self.status.to_dict(),
            "first_seen": self.first_seen,
            "last_seen": self.last_seen,
            "is_trusted": self.is_trusted,
            "tls_enabled": self.tls_enabled,
            "online": self.online,
            "last_online": self.last_online,
            "roles": self.roles,
            "datasets": [d.to_dict() for d in self.datasets],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "PersistedPeer":
        return cls(
            instance_id=d.get("instance_id", ""),
            hostname=d.get("hostname", ""),
            ip_address=d.get("ip_address", ""),
            viz_port=d.get("viz_port", settings.network.viz_port),
            capabilities=PeerCapability.from_dict(d.get("capabilities", {})),
            status=PeerStatus.from_dict(d.get("status", {})),
            first_seen=d.get("first_seen", time.time()),
            last_seen=d.get("last_seen", time.time()),
            is_trusted=d.get("is_trusted", False),
            tls_enabled=d.get("tls_enabled", False),
            online=d.get("online", True),
            last_online=d.get("last_online", d.get("last_seen", time.time())),
            roles=d.get("roles", []),
            datasets=[DatasetAdvertisement.from_dict(ds) for ds in d.get("datasets", [])],
        )

    @classmethod
    def from_peer_info(cls, peer: PeerInfo) -> "PersistedPeer":
        """Create a PersistedPeer from a live PeerInfo."""
        return cls(
            instance_id=peer.instance_id,
            hostname=peer.hostname,
            ip_address=peer.ip_address,
            viz_port=peer.viz_port,
            capabilities=peer.capabilities,
            status=peer.status,
            first_seen=peer.first_seen,
            last_seen=peer.last_seen,
            is_trusted=peer.is_trusted,
            tls_enabled=peer.tls_enabled,
            online=True,
            last_online=peer.last_seen,
            roles=peer.capabilities.roles,
            datasets=peer.capabilities.datasets,
        )
