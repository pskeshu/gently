"""
Detector registry for managing all configured detectors
"""

import json
import logging
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

from .detector import (  # noqa: E402
    ConfidenceLevel,
    Detector,
)


class DetectorRegistry:
    """
    Central registry for all configured detectors

    Handles CRUD operations, persistence, and querying of detectors.
    """

    def __init__(self, storage_path: Path | None = None):
        """
        Parameters
        ----------
        storage_path : Path, optional
            Where to save detector registry JSON
        """
        self.detectors: dict[str, Detector] = {}
        self.storage_path = storage_path or Path("./detector_registry.json")

        # Load existing detectors if file exists
        if self.storage_path.exists():
            self.load()

    def add(self, detector: Detector) -> bool:
        """
        Add a new detector

        Parameters
        ----------
        detector : Detector
            Detector to add

        Returns
        -------
        bool
            True if added, False if name already exists
        """
        if detector.name in self.detectors:
            return False

        self.detectors[detector.name] = detector
        self.save()
        return True

    def remove(self, name: str) -> bool:
        """
        Remove a detector

        Parameters
        ----------
        name : str
            Detector name

        Returns
        -------
        bool
            True if removed, False if not found
        """
        if name not in self.detectors:
            return False

        del self.detectors[name]
        self.save()
        return True

    def get(self, name: str) -> Detector | None:
        """Get detector by name"""
        return self.detectors.get(name)

    def list_all(self) -> list[Detector]:
        """Get all detectors"""
        return list(self.detectors.values())

    def list_enabled(self) -> list[Detector]:
        """Get all enabled detectors"""
        return [d for d in self.detectors.values() if d.enabled]

    def enable(self, name: str) -> bool:
        """
        Enable a detector

        Parameters
        ----------
        name : str
            Detector name

        Returns
        -------
        bool
            True if enabled, False if not found
        """
        detector = self.get(name)
        if not detector:
            return False

        detector.enabled = True
        detector.modified_at = datetime.now()
        self.save()
        return True

    def disable(self, name: str) -> bool:
        """
        Disable a detector

        Parameters
        ----------
        name : str
            Detector name

        Returns
        -------
        bool
            True if disabled, False if not found
        """
        detector = self.get(name)
        if not detector:
            return False

        detector.enabled = False
        detector.modified_at = datetime.now()
        self.save()
        return True

    def update(self, name: str, **kwargs) -> bool:
        """
        Update detector attributes

        Parameters
        ----------
        name : str
            Detector name
        **kwargs
            Attributes to update

        Returns
        -------
        bool
            True if updated, False if not found
        """
        detector = self.get(name)
        if not detector:
            return False

        for key, value in kwargs.items():
            if hasattr(detector, key):
                setattr(detector, key, value)

        detector.modified_at = datetime.now()
        self.save()
        return True

    def get_stats(self) -> dict:
        """
        Get registry statistics

        Returns
        -------
        dict
            Statistics summary
        """
        total = len(self.detectors)
        enabled = len([d for d in self.detectors.values() if d.enabled])
        total_detections = sum(d.detection_count for d in self.detectors.values())
        total_runs = sum(d.run_count for d in self.detectors.values())

        return {
            "total_detectors": total,
            "enabled_detectors": enabled,
            "disabled_detectors": total - enabled,
            "total_detections_fired": total_detections,
            "total_runs": total_runs,
            "detectors": {
                name: {
                    "enabled": d.enabled,
                    "detection_count": d.detection_count,
                    "run_count": d.run_count,
                }
                for name, d in self.detectors.items()
            },
        }

    def save(self):
        """Save registry to disk"""
        data = {
            "version": "1.0",
            "saved_at": datetime.now().isoformat(),
            "detectors": {name: detector.to_dict() for name, detector in self.detectors.items()},
        }

        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.storage_path, "w") as f:
            json.dump(data, f, indent=2)

    def load(self):
        """Load registry from disk"""
        if not self.storage_path.exists():
            return

        with open(self.storage_path) as f:
            data = json.load(f)

        self.detectors = {}
        for name, detector_data in data.get("detectors", {}).items():
            try:
                detector = Detector.from_dict(detector_data)
                self.detectors[name] = detector
            except Exception as e:
                logger.warning("Failed to load detector '%s': %s", name, e)

    def create_preset_detector(self, preset_name: str) -> Detector | None:
        """
        Create a detector from preset

        Parameters
        ----------
        preset_name : str
            Preset name (e.g., 'hatching', 'comma', 'pretzel')

        Returns
        -------
        Detector or None
            Detector if preset exists, None otherwise
        """
        presets = get_detector_presets()
        if preset_name not in presets:
            return None

        preset_data = presets[preset_name]
        detector = Detector(
            name=preset_data["name"],
            description=preset_data["description"],
            detection_prompt=preset_data["prompt"],
            use_temporal_context=preset_data.get("use_temporal_context", True),
            temporal_context_size=preset_data.get("temporal_context_size", 5),
            confidence_threshold=ConfidenceLevel(preset_data.get("confidence_threshold", "MEDIUM")),
        )

        return detector


# Re-export detector presets via the active organism plugin
def get_detector_presets():
    """Get detector presets from the active organism module."""
    from gently.organisms import get_organism

    org = get_organism()
    presets_module = __import__(
        f"gently.organisms.{org.ORGANISM_NAME}.detector_presets",
        fromlist=["get_detector_presets"],
    )
    return presets_module.get_detector_presets()
