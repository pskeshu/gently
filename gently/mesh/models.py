"""
Mesh data models for peer discovery and status exchange.

Defines the three core dataclasses:
- PeerCapability: what a node can do
- PeerStatus: current runtime state
- PeerInfo: complete peer record
"""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class PeerCapability:
    """What a Gently node can do."""

    has_microscope: bool = False
    has_sam: bool = False
    has_gpu: bool = False
    gpu_name: str = ""
    gpu_vram_gb: float = 0.0
    storage_free_gb: float = 0.0
    tool_categories: List[str] = field(default_factory=list)
    organism: str = ""
    hardware_profile: str = ""

    def to_dict(self) -> Dict[str, Any]:
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
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PeerCapability":
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
        )


@dataclass
class PeerStatus:
    """Current runtime state of a Gently node."""

    session_id: str = ""
    acquisition_status: str = "idle"
    embryo_count: int = 0
    total_timepoints: int = 0
    uptime_seconds: float = 0.0
    copilot_mode: str = "run"
    active_plan: str = ""
    version: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "acquisition_status": self.acquisition_status,
            "embryo_count": self.embryo_count,
            "total_timepoints": self.total_timepoints,
            "uptime_seconds": self.uptime_seconds,
            "copilot_mode": self.copilot_mode,
            "active_plan": self.active_plan,
            "version": self.version,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PeerStatus":
        return cls(
            session_id=d.get("session_id", ""),
            acquisition_status=d.get("acquisition_status", "idle"),
            embryo_count=d.get("embryo_count", 0),
            total_timepoints=d.get("total_timepoints", 0),
            uptime_seconds=d.get("uptime_seconds", 0.0),
            copilot_mode=d.get("copilot_mode", "run"),
            active_plan=d.get("active_plan", ""),
            version=d.get("version", ""),
        )


@dataclass
class PeerInfo:
    """Complete record for a discovered peer."""

    instance_id: str = ""
    hostname: str = ""
    ip_address: str = ""
    viz_port: int = 8080
    capabilities: PeerCapability = field(default_factory=PeerCapability)
    status: PeerStatus = field(default_factory=PeerStatus)
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)
    is_self: bool = False

    @property
    def base_url(self) -> str:
        return f"http://{self.ip_address}:{self.viz_port}"

    @property
    def is_stale(self) -> bool:
        """True if no heartbeat for >15s."""
        return (time.time() - self.last_seen) > 15.0

    @property
    def is_dead(self) -> bool:
        """True if no heartbeat for >30s."""
        return (time.time() - self.last_seen) > 30.0

    def to_dict(self) -> Dict[str, Any]:
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
            "base_url": self.base_url,
            "is_stale": self.is_stale,
            "is_dead": self.is_dead,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "PeerInfo":
        return cls(
            instance_id=d.get("instance_id", ""),
            hostname=d.get("hostname", ""),
            ip_address=d.get("ip_address", ""),
            viz_port=d.get("viz_port", 8080),
            capabilities=PeerCapability.from_dict(d.get("capabilities", {})),
            status=PeerStatus.from_dict(d.get("status", {})),
            first_seen=d.get("first_seen", time.time()),
            last_seen=d.get("last_seen", time.time()),
            is_self=d.get("is_self", False),
        )
