"""
State management for embryos and experiments
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path


@dataclass
class ImageRecord:
    """Record of a single acquired image/volume"""
    embryo_id: str
    timepoint: int
    timestamp: datetime
    volume_path: str  # Path to full TIFF volume on disk
    max_projection_b64: str  # Base64-encoded JPEG for Claude Vision
    size_kb: float
    # UID-based data references (new data layer)
    volume_uid: Optional[str] = None  # UID for volume in DataStore
    projection_uid: Optional[str] = None  # UID for max projection in DataStore


@dataclass
class EmbryoState:
    """Complete state of one C. elegans embryo"""

    # Identity
    id: str  # "embryo_1"
    nickname: Optional[str] = None  # Agent-assigned: "the fast one"
    user_label: Optional[str] = None  # User-provided: "control_1"

    # Position
    stage_position: Dict[str, float] = field(default_factory=dict)  # {'x': 1234.5, 'y': 5678.9}
    calibration: Dict = field(default_factory=dict)  # Galvo/piezo parameters
    detection_confidence: float = 0.0  # SAM/detection confidence score (0-1)

    # Acquisition Parameters (current)
    interval_seconds: float = 120
    num_slices: int = 50
    exposure_ms: float = 10.0
    priority: str = "normal"  # high/normal/low

    # Status
    last_imaged: Optional[datetime] = None
    timepoints_acquired: int = 0
    should_skip: bool = False
    skip_reason: Optional[str] = None

    # Analysis Results (cached)
    hatching_status: Dict = field(default_factory=dict)
    # {hatched: bool, confidence: str, timepoint: int}

    morphology_history: List[Dict] = field(default_factory=list)
    # [{timepoint, size, shape, activity_score}]

    fluorescence_history: List[Dict] = field(default_factory=list)
    # [{timepoint, mean_intensity, photobleaching_estimate}]

    custom_classifications: Dict = field(default_factory=dict)
    # User-defined: {"first_cleavage": {detected: bool, timepoint: 42}}

    # Detection results from detector system
    detection_results: Dict[str, List[Dict]] = field(default_factory=dict)
    # detector_name -> list of detection results
    # e.g., {"comma_stage": [{"timepoint": 120, "detected": False, "confidence": "HIGH"}, ...]}

    # Images (recent for context)
    recent_images: List[ImageRecord] = field(default_factory=list)
    # Keep last 10 for temporal context in Claude Vision calls

    def add_detection_result(self, detector_name: str, result: Dict):
        """
        Add detection result from detector system

        Parameters
        ----------
        detector_name : str
            Name of detector
        result : dict
            Detection result (from DetectionResult.to_dict())
        """
        if detector_name not in self.detection_results:
            self.detection_results[detector_name] = []

        self.detection_results[detector_name].append(result)

    def get_latest_detection(self, detector_name: str) -> Optional[Dict]:
        """Get most recent detection result for a detector"""
        if detector_name not in self.detection_results:
            return None

        results = self.detection_results[detector_name]
        return results[-1] if results else None

    def was_detected(self, detector_name: str) -> bool:
        """Check if detector has ever fired (detected=True) for this embryo"""
        if detector_name not in self.detection_results:
            return False

        return any(r.get('detected', False) for r in self.detection_results[detector_name])

    def update_from_analysis(self, analysis_result: Dict):
        """Update state with new analysis"""
        if 'hatching' in analysis_result:
            self.hatching_status = analysis_result['hatching']

        if 'morphology' in analysis_result:
            self.morphology_history.append({
                'timepoint': self.timepoints_acquired,
                **analysis_result['morphology']
            })

        if 'fluorescence' in analysis_result:
            self.fluorescence_history.append({
                'timepoint': self.timepoints_acquired,
                **analysis_result['fluorescence']
            })

    def to_summary(self) -> str:
        """Format for Claude system prompt"""
        status_parts = []

        # Identity
        name = self.nickname or self.user_label or self.id
        status_parts.append(f"{name} ({self.id})")

        # Timing
        if self.last_imaged:
            mins_ago = (datetime.now() - self.last_imaged).seconds // 60
            status_parts.append(f"last imaged {mins_ago}min ago")
        else:
            status_parts.append("not yet imaged")

        # Status
        if self.hatching_status.get('hatched'):
            status_parts.append(f"hatched at t{self.hatching_status['timepoint']:04d}")
        elif self.should_skip:
            status_parts.append(f"skipped ({self.skip_reason})")
        else:
            status_parts.append("active")

        # Current params
        status_parts.append(f"interval={self.interval_seconds}s")
        status_parts.append(f"slices={self.num_slices}")
        status_parts.append(f"priority={self.priority}")

        return " | ".join(status_parts)

    def to_dict(self) -> Dict:
        """Serialize for API responses"""
        return {
            'id': self.id,
            'nickname': self.nickname,
            'user_label': self.user_label,
            'stage_position': self.stage_position,
            'interval_seconds': self.interval_seconds,
            'num_slices': self.num_slices,
            'exposure_ms': self.exposure_ms,
            'priority': self.priority,
            'last_imaged': self.last_imaged.isoformat() if self.last_imaged else None,
            'timepoints_acquired': self.timepoints_acquired,
            'should_skip': self.should_skip,
            'skip_reason': self.skip_reason,
            'hatching_status': self.hatching_status,
            'recent_analyses': {
                'morphology': self.morphology_history[-5:] if self.morphology_history else [],
                'fluorescence': self.fluorescence_history[-5:] if self.fluorescence_history else [],
                'custom': self.custom_classifications,
            }
        }


class ExperimentState:
    """Global experiment state"""

    def __init__(self):
        self.embryos: Dict[str, EmbryoState] = {}
        self.start_time: Optional[datetime] = None
        self.acquisition_status: str = "idle"  # idle/running/paused/completed
        self.current_plan_name: Optional[str] = None
        self.plan_history: List[Dict] = []
        self.metadata: Dict = {}

    def add_embryo(self, embryo_id: str, position: Dict = None,
                   calibration: Dict = None, user_label: Optional[str] = None,
                   confidence: float = 0.0):
        """Register new embryo"""
        # Auto-start experiment when first embryo is added
        if self.start_time is None:
            self.start_time = datetime.now()

        self.embryos[embryo_id] = EmbryoState(
            id=embryo_id,
            stage_position=position or {},
            calibration=calibration or {},
            user_label=user_label,
            detection_confidence=confidence
        )

    def remove_embryo(self, embryo_id: str) -> bool:
        """Remove embryo from experiment (e.g., false detection)"""
        if embryo_id in self.embryos:
            del self.embryos[embryo_id]
            return True
        return False

    def assign_nickname(self, embryo_id: str, nickname: str):
        """Agent assigns intuitive name"""
        if embryo_id in self.embryos:
            self.embryos[embryo_id].nickname = nickname

    def get_embryo_by_any_name(self, name: str) -> Optional[EmbryoState]:
        """Get embryo by ID, nickname, or user label"""
        # Direct ID match
        if name in self.embryos:
            return self.embryos[name]

        # Search by nickname or label
        for embryo in self.embryos.values():
            if embryo.nickname == name or embryo.user_label == name:
                return embryo

        # Try extracting number from name like "embryo 3" -> "embryo_3"
        import re
        match = re.search(r'(\d+)', name)
        if match:
            num = int(match.group(1))
            # Try simple format first (embryo_3)
            potential_id = f"embryo_{num}"
            if potential_id in self.embryos:
                return self.embryos[potential_id]
            # Also try padded format for backwards compatibility (embryo_003)
            potential_id_padded = f"embryo_{num:03d}"
            if potential_id_padded in self.embryos:
                return self.embryos[potential_id_padded]

        return None

    def get_summary(self) -> str:
        """Full experiment summary for Claude"""
        if not self.start_time:
            return "No active experiment"

        duration = datetime.now() - self.start_time
        hours = duration.seconds // 3600
        minutes = (duration.seconds % 3600) // 60

        lines = [
            f"Experiment Status: {self.acquisition_status}",
            f"Duration: {hours}h {minutes}m",
            f"Embryos: {len(self.embryos)}",
            "",
            "Per-embryo status:"
        ]

        for embryo in sorted(self.embryos.values(), key=lambda e: e.id):
            lines.append(f"  {embryo.to_summary()}")

        if self.current_plan_name:
            lines.append(f"\nCurrent plan: {self.current_plan_name}")

        return "\n".join(lines)

    def to_dict(self) -> Dict:
        """Serialize for API responses"""
        return {
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'acquisition_status': self.acquisition_status,
            'current_plan_name': self.current_plan_name,
            'embryo_count': len(self.embryos),
            'embryos': {eid: e.to_dict() for eid, e in self.embryos.items()},
            'metadata': self.metadata
        }
