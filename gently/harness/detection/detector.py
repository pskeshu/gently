"""
Generic detector system for runtime-configurable event detection
"""

from dataclasses import asdict, dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any


class DetectionMode(str, Enum):
    """Action mode when detector fires"""

    PASSIVE = "passive"  # Just flag, no action
    RECOMMEND = "recommend"  # Suggest actions to user
    AUTO = "auto"  # Execute actions automatically


class ConfidenceLevel(str, Enum):
    """Confidence levels for detections"""

    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


@dataclass
class DetectorConditions:
    """Conditions for when to run a detector"""

    min_timepoint: int | None = None  # Don't run before this timepoint
    max_timepoint: int | None = None  # Don't run after this timepoint
    embryo_ids: list[str] | None = None  # Only run on these embryos (None = all)
    run_if_detected: bool = True  # Continue running after first detection?
    min_interval_timepoints: int = 1  # Minimum timepoints between runs

    def should_run(
        self,
        embryo_id: str,
        timepoint: int,
        last_run_timepoint: int | None,
        already_detected: bool,
    ) -> bool:
        """
        Check if detector should run

        Parameters
        ----------
        embryo_id : str
            Embryo being checked
        timepoint : int
            Current timepoint
        last_run_timepoint : int or None
            Last timepoint this detector ran
        already_detected : bool
            Whether this detector already fired for this embryo

        Returns
        -------
        bool
            True if should run
        """
        # Check timepoint range
        if self.min_timepoint is not None and timepoint < self.min_timepoint:
            return False
        if self.max_timepoint is not None and timepoint > self.max_timepoint:
            return False

        # Check embryo whitelist
        if self.embryo_ids is not None and embryo_id not in self.embryo_ids:
            return False

        # Check if already detected
        if already_detected and not self.run_if_detected:
            return False

        # Check minimum interval
        if last_run_timepoint is not None:
            if (timepoint - last_run_timepoint) < self.min_interval_timepoints:
                return False

        return True


@dataclass
class DetectorActions:
    """Actions to take when detector fires"""

    mode: DetectionMode = DetectionMode.RECOMMEND
    parameter_changes: dict[str, Any] | None = None  # e.g., {"interval_seconds": 60}
    stop_timelapse: bool = False  # Stop timelapse when detected
    custom_message: str | None = None  # Custom notification
    webhook_url: str | None = None  # External notification

    def get_recommendation_message(self, detector_name: str, embryo_id: str) -> str:
        """Generate recommendation message for user"""
        msg = f"Detector '{detector_name}' triggered for {embryo_id}"

        if self.parameter_changes:
            msg += "\n\nRecommended parameter changes:"
            for param, value in self.parameter_changes.items():
                msg += f"\n  - {param}: {value}"

        if self.custom_message:
            msg += f"\n\n{self.custom_message}"

        msg += "\n\nApply these changes?"
        return msg


@dataclass
class DetectionResult:
    """Result of a single detection attempt"""

    detector_name: str
    embryo_id: str
    timepoint: int
    timestamp: datetime
    detected: bool
    confidence: ConfidenceLevel | None = None
    reasoning: str | None = None
    error: bool = False
    error_message: str | None = None
    api_duration: float | None = None  # seconds
    num_images: int = 1
    full_response: str | None = None

    def to_dict(self) -> dict:
        """Convert to dictionary"""
        d = asdict(self)
        d["timestamp"] = self.timestamp.isoformat()
        if self.confidence:
            d["confidence"] = self.confidence.value
        return d


@dataclass
class Detector:
    """
    Generic detector for runtime-configurable event detection

    A detector analyzes volumes using Claude Vision API to detect specific
    events or states (e.g., "comma stage", "hatching", "neural activity").
    """

    name: str  # Unique identifier (e.g., "comma_stage")
    description: str  # Human-readable description
    detection_prompt: str  # Claude Vision API prompt
    enabled: bool = True  # Can be toggled on/off
    conditions: DetectorConditions = field(default_factory=DetectorConditions)
    actions: DetectorActions = field(default_factory=DetectorActions)
    confidence_threshold: ConfidenceLevel = ConfidenceLevel.MEDIUM
    use_temporal_context: bool = True  # Include recent images?
    temporal_context_size: int = 5  # How many recent images
    created_at: datetime = field(default_factory=datetime.now)
    modified_at: datetime = field(default_factory=datetime.now)
    detection_count: int = 0  # Total detections fired
    run_count: int = 0  # Total times run

    # Tracking per-embryo state
    _last_run_timepoint: dict[str, int] = field(default_factory=dict, repr=False)
    _detected_embryos: set = field(default_factory=set, repr=False)

    def should_run(self, embryo_id: str, timepoint: int) -> bool:
        """
        Check if this detector should run for given embryo/timepoint

        Parameters
        ----------
        embryo_id : str
            Embryo ID
        timepoint : int
            Current timepoint

        Returns
        -------
        bool
            True if should run
        """
        if not self.enabled:
            return False

        last_run = self._last_run_timepoint.get(embryo_id)
        already_detected = embryo_id in self._detected_embryos

        return self.conditions.should_run(embryo_id, timepoint, last_run, already_detected)

    def mark_run(self, embryo_id: str, timepoint: int):
        """Mark that detector ran for this embryo/timepoint"""
        self._last_run_timepoint[embryo_id] = timepoint
        self.run_count += 1

    def mark_detected(self, embryo_id: str):
        """Mark that detector fired for this embryo"""
        self._detected_embryos.add(embryo_id)
        self.detection_count += 1

    def was_detected(self, embryo_id: str) -> bool:
        """Check if detector already fired for this embryo"""
        return embryo_id in self._detected_embryos

    def build_detection_content(
        self, images: list[dict], embryo_id: str, timepoint: int
    ) -> list[dict]:
        """
        Build Claude Vision API content array

        Parameters
        ----------
        images : list of dict
            Recent images, each with {'timepoint', 'b64_image', 'size'}
        embryo_id : str
            Embryo being analyzed
        timepoint : int
            Current timepoint

        Returns
        -------
        list of dict
            Content array for Claude API
        """
        content: list[dict[str, Any]] = []

        # Add instruction
        content.append({"type": "text", "text": f"Analyzing {embryo_id} at timepoint {timepoint}"})

        # Add temporal context if enabled
        if self.use_temporal_context and len(images) > 1:
            content.append(
                {
                    "type": "text",
                    "text": f"Recent images (for temporal context, {len(images)} timepoints):",
                }
            )

            # Add older images first
            for img_data in images[:-1]:
                content.append({"type": "text", "text": f"Timepoint {img_data['timepoint']:04d}"})
                content.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": img_data["b64_image"],
                        },
                    }
                )

        # Add current/latest image
        latest = images[-1]
        content.append(
            {
                "type": "text",
                "text": (
                    f"Current image (timepoint {latest['timepoint']:04d})"
                    " - FOCUS YOUR ANALYSIS HERE:"
                ),
            }
        )
        content.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": latest["b64_image"],
                },
            }
        )

        # Add detection prompt
        content.append({"type": "text", "text": self.detection_prompt})

        return content

    def parse_detection_response(self, response_text: str) -> dict:
        """
        Parse Claude's detection response

        Expected format:
        DETECTED: YES/NO
        CONFIDENCE: HIGH/MEDIUM/LOW
        REASONING: explanation

        Parameters
        ----------
        response_text : str
            Claude's response

        Returns
        -------
        dict
            Parsed result with keys: detected, confidence, reasoning
        """
        detected = None
        confidence = None
        reasoning = None

        lines = response_text.strip().split("\n")

        for line in lines:
            line = line.strip()
            if line.startswith("DETECTED:"):
                value = line.split(":", 1)[1].strip().upper()
                detected = value in ["YES", "TRUE", "1"]
            elif line.startswith("CONFIDENCE:"):
                conf_str = line.split(":", 1)[1].strip().upper()
                try:
                    confidence = ConfidenceLevel(conf_str)
                except ValueError:
                    confidence = None
            elif line.startswith("REASONING:"):
                reasoning = line.split(":", 1)[1].strip()

        # If multiline reasoning, capture it
        if reasoning is None:
            reasoning_start = False
            reasoning_lines = []
            for line in lines:
                if line.startswith("REASONING:"):
                    reasoning_start = True
                    reasoning_lines.append(line.split(":", 1)[1].strip())
                elif reasoning_start and line:
                    reasoning_lines.append(line)
            if reasoning_lines:
                reasoning = " ".join(reasoning_lines)

        return {
            "detected": detected if detected is not None else False,
            "confidence": confidence,
            "reasoning": reasoning or "No reasoning provided",
        }

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization"""
        return {
            "name": self.name,
            "description": self.description,
            "detection_prompt": self.detection_prompt,
            "enabled": self.enabled,
            "conditions": asdict(self.conditions),
            "actions": {
                "mode": self.actions.mode.value,
                "parameter_changes": self.actions.parameter_changes,
                "custom_message": self.actions.custom_message,
                "webhook_url": self.actions.webhook_url,
            },
            "confidence_threshold": self.confidence_threshold.value,
            "use_temporal_context": self.use_temporal_context,
            "temporal_context_size": self.temporal_context_size,
            "created_at": self.created_at.isoformat(),
            "modified_at": self.modified_at.isoformat(),
            "detection_count": self.detection_count,
            "run_count": self.run_count,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Detector":
        """Create detector from dictionary"""
        # Parse dates
        created_at = (
            datetime.fromisoformat(data["created_at"]) if "created_at" in data else datetime.now()
        )
        modified_at = (
            datetime.fromisoformat(data["modified_at"]) if "modified_at" in data else datetime.now()
        )

        # Parse conditions
        conditions = DetectorConditions(**data.get("conditions", {}))

        # Parse actions
        actions_data = data.get("actions", {})
        actions = DetectorActions(
            mode=DetectionMode(actions_data.get("mode", "recommend")),
            parameter_changes=actions_data.get("parameter_changes"),
            custom_message=actions_data.get("custom_message"),
            webhook_url=actions_data.get("webhook_url"),
        )

        # Parse confidence threshold
        confidence_threshold = ConfidenceLevel(data.get("confidence_threshold", "MEDIUM"))

        return cls(
            name=data["name"],
            description=data["description"],
            detection_prompt=data["detection_prompt"],
            enabled=data.get("enabled", True),
            conditions=conditions,
            actions=actions,
            confidence_threshold=confidence_threshold,
            use_temporal_context=data.get("use_temporal_context", True),
            temporal_context_size=data.get("temporal_context_size", 5),
            created_at=created_at,
            modified_at=modified_at,
            detection_count=data.get("detection_count", 0),
            run_count=data.get("run_count", 0),
        )
