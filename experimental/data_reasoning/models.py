"""
Data models for the data reasoning engine.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class SessionSummary:
    """Summary of a single imaging session's data."""

    session_id: str = ""
    session_name: str = ""
    source_peer: str = ""  # instance_id of peer (empty = local)
    embryo_count: int = 0
    volume_count: int = 0
    annotated_embryos: int = 0
    ground_truth_count: int = 0
    stages_covered: list[str] = field(default_factory=list)
    total_size_gb: float = 0.0
    is_remote: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "session_id": self.session_id,
            "session_name": self.session_name,
            "source_peer": self.source_peer,
            "embryo_count": self.embryo_count,
            "volume_count": self.volume_count,
            "annotated_embryos": self.annotated_embryos,
            "ground_truth_count": self.ground_truth_count,
            "stages_covered": self.stages_covered,
            "total_size_gb": self.total_size_gb,
            "is_remote": self.is_remote,
        }


@dataclass
class NetworkDataInventory:
    """Aggregated inventory of all data across the mesh."""

    local_sessions: list[SessionSummary] = field(default_factory=list)
    remote_sessions: list[SessionSummary] = field(default_factory=list)
    total_embryos: int = 0
    total_volumes: int = 0
    total_annotated: int = 0
    total_ground_truth: int = 0
    peers_queried: int = 0
    peers_failed: int = 0

    @property
    def all_sessions(self) -> list[SessionSummary]:
        return self.local_sessions + self.remote_sessions

    def to_dict(self) -> dict[str, Any]:
        return {
            "local_sessions": [s.to_dict() for s in self.local_sessions],
            "remote_sessions": [s.to_dict() for s in self.remote_sessions],
            "total_embryos": self.total_embryos,
            "total_volumes": self.total_volumes,
            "total_annotated": self.total_annotated,
            "total_ground_truth": self.total_ground_truth,
            "peers_queried": self.peers_queried,
            "peers_failed": self.peers_failed,
        }


@dataclass
class CoverageReport:
    """Annotation coverage analysis across network data."""

    total_embryos: int = 0
    annotated_embryos: int = 0
    coverage_pct: float = 0.0
    stage_counts: dict[str, int] = field(default_factory=dict)
    imbalance_ratio: float = 0.0
    gaps: list[str] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_embryos": self.total_embryos,
            "annotated_embryos": self.annotated_embryos,
            "coverage_pct": self.coverage_pct,
            "stage_counts": self.stage_counts,
            "imbalance_ratio": self.imbalance_ratio,
            "gaps": self.gaps,
            "recommendations": self.recommendations,
        }


@dataclass
class DataQualityReport:
    """Data quality validation results."""

    total_volumes_checked: int = 0
    readable_volumes: int = 0
    missing_projections: int = 0
    inconsistent_annotations: int = 0
    issues: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_volumes_checked": self.total_volumes_checked,
            "readable_volumes": self.readable_volumes,
            "missing_projections": self.missing_projections,
            "inconsistent_annotations": self.inconsistent_annotations,
            "issues": self.issues,
        }
