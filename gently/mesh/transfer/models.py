"""
Transfer data models.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class TransferType(str, Enum):
    """Type of data being transferred."""

    DATASET = "dataset"
    MODEL_WEIGHTS = "model_weights"
    SESSION = "session"


class TransferStatus(str, Enum):
    """Transfer state machine."""

    PENDING = "pending"
    TRANSFERRING = "transferring"
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class TransferFile:
    """A single file in a transfer manifest."""

    relative_path: str = ""
    total_size: int = 0
    sha256: str = ""
    transferred: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "relative_path": self.relative_path,
            "total_size": self.total_size,
            "sha256": self.sha256,
            "transferred": self.transferred,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TransferFile":
        return cls(
            relative_path=d.get("relative_path", ""),
            total_size=d.get("total_size", 0),
            sha256=d.get("sha256", ""),
            transferred=d.get("transferred", 0),
        )


@dataclass
class TransferManifest:
    """List of files to transfer."""

    files: list[TransferFile] = field(default_factory=list)
    total_size: int = 0
    file_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "files": [f.to_dict() for f in self.files],
            "total_size": self.total_size,
            "file_count": self.file_count,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TransferManifest":
        files = [TransferFile.from_dict(f) for f in d.get("files", [])]
        return cls(
            files=files,
            total_size=d.get("total_size", sum(f.total_size for f in files)),
            file_count=d.get("file_count", len(files)),
        )


@dataclass
class TransferJob:
    """State of a single transfer (send or receive)."""

    id: str = ""
    transfer_type: str = TransferType.DATASET.value
    status: str = TransferStatus.PENDING.value
    direction: str = "send"  # "send" or "receive"
    peer_instance_id: str = ""
    peer_hostname: str = ""
    manifest: TransferManifest = field(default_factory=TransferManifest)
    source_path: str = ""
    dest_path: str = ""
    bytes_transferred: int = 0
    total_bytes: int = 0
    started_at: float = 0.0
    completed_at: float = 0.0
    error: str = ""
    # For dataset transfers
    session_id: str = ""
    # For model weight transfers
    pipeline_id: str = ""

    @property
    def progress_pct(self) -> float:
        if self.total_bytes == 0:
            return 0.0
        return (self.bytes_transferred / self.total_bytes) * 100

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "transfer_type": self.transfer_type,
            "status": self.status,
            "direction": self.direction,
            "peer_instance_id": self.peer_instance_id,
            "peer_hostname": self.peer_hostname,
            "manifest": self.manifest.to_dict(),
            "source_path": self.source_path,
            "dest_path": self.dest_path,
            "bytes_transferred": self.bytes_transferred,
            "total_bytes": self.total_bytes,
            "progress_pct": round(self.progress_pct, 1),
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "error": self.error,
            "session_id": self.session_id,
            "pipeline_id": self.pipeline_id,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TransferJob":
        return cls(
            id=d.get("id", ""),
            transfer_type=d.get("transfer_type", TransferType.DATASET.value),
            status=d.get("status", TransferStatus.PENDING.value),
            direction=d.get("direction", "send"),
            peer_instance_id=d.get("peer_instance_id", ""),
            peer_hostname=d.get("peer_hostname", ""),
            manifest=TransferManifest.from_dict(d.get("manifest", {})),
            source_path=d.get("source_path", ""),
            dest_path=d.get("dest_path", ""),
            bytes_transferred=d.get("bytes_transferred", 0),
            total_bytes=d.get("total_bytes", 0),
            started_at=d.get("started_at", 0.0),
            completed_at=d.get("completed_at", 0.0),
            error=d.get("error", ""),
            session_id=d.get("session_id", ""),
            pipeline_id=d.get("pipeline_id", ""),
        )
