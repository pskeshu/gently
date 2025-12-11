"""
Detector registry for managing all configured detectors
"""

import json
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

from .detector import Detector, DetectorConditions, DetectorActions, DetectionMode, ConfidenceLevel


class DetectorRegistry:
    """
    Central registry for all configured detectors

    Handles CRUD operations, persistence, and querying of detectors.
    """

    def __init__(self, storage_path: Optional[Path] = None):
        """
        Parameters
        ----------
        storage_path : Path, optional
            Where to save detector registry JSON
        """
        self.detectors: Dict[str, Detector] = {}
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

    def get(self, name: str) -> Optional[Detector]:
        """Get detector by name"""
        return self.detectors.get(name)

    def list_all(self) -> List[Detector]:
        """Get all detectors"""
        return list(self.detectors.values())

    def list_enabled(self) -> List[Detector]:
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

    def get_stats(self) -> Dict:
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
            'total_detectors': total,
            'enabled_detectors': enabled,
            'disabled_detectors': total - enabled,
            'total_detections_fired': total_detections,
            'total_runs': total_runs,
            'detectors': {
                name: {
                    'enabled': d.enabled,
                    'detection_count': d.detection_count,
                    'run_count': d.run_count
                }
                for name, d in self.detectors.items()
            }
        }

    def save(self):
        """Save registry to disk"""
        data = {
            'version': '1.0',
            'saved_at': datetime.now().isoformat(),
            'detectors': {
                name: detector.to_dict()
                for name, detector in self.detectors.items()
            }
        }

        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.storage_path, 'w') as f:
            json.dump(data, f, indent=2)

    def load(self):
        """Load registry from disk"""
        if not self.storage_path.exists():
            return

        with open(self.storage_path, 'r') as f:
            data = json.load(f)

        self.detectors = {}
        for name, detector_data in data.get('detectors', {}).items():
            try:
                detector = Detector.from_dict(detector_data)
                self.detectors[name] = detector
            except Exception as e:
                print(f"Warning: Failed to load detector '{name}': {e}")

    def create_preset_detector(self, preset_name: str) -> Optional[Detector]:
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
            name=preset_data['name'],
            description=preset_data['description'],
            detection_prompt=preset_data['prompt'],
            use_temporal_context=preset_data.get('use_temporal_context', True),
            temporal_context_size=preset_data.get('temporal_context_size', 5),
            confidence_threshold=ConfidenceLevel(preset_data.get('confidence_threshold', 'MEDIUM')),
        )

        return detector


def get_detector_presets() -> Dict:
    """
    Get predefined detector presets for common C. elegans stages

    Returns
    -------
    dict
        Preset detector configurations
    """
    return {
        'hatching': {
            'name': 'hatching',
            'description': 'Detects when C. elegans embryo hatches from eggshell',
            'prompt': """Analyze this C. elegans embryo image and determine if the embryo has HATCHED.

Key characteristics of hatching:
- Visible breach or rupture in the eggshell (vitelline membrane)
- Embryo emerging or partially emerged from shell
- Clear separation between larva and eggshell
- Change in overall morphology from constrained to free-moving

Focus on the CURRENT/LATEST image (the final one shown).

Respond in this exact format:
DETECTED: [YES/NO]
CONFIDENCE: [HIGH/MEDIUM/LOW]
REASONING: [Brief explanation of what you observe]""",
            'use_temporal_context': True,
            'temporal_context_size': 10,
            'confidence_threshold': 'HIGH',
            'stop_timelapse': True,  # Auto-stop when hatching detected
        },

        'comma': {
            'name': 'comma',
            'description': 'Detects comma stage (major morphogenesis)',
            'prompt': """Analyze this C. elegans embryo and determine if it has reached the COMMA STAGE.

Key characteristics of comma stage (~400 minutes, ~6.5 hours):
- Distinct comma or bean shape (ventral curvature)
- Clear anterior-posterior elongation
- Visible head/tail differentiation
- Movement patterns visible
- Still within eggshell

Focus on the CURRENT/LATEST image.

Respond in this exact format:
DETECTED: [YES/NO]
CONFIDENCE: [HIGH/MEDIUM/LOW]
REASONING: [Brief explanation]""",
            'use_temporal_context': True,
            'temporal_context_size': 5,
            'confidence_threshold': 'MEDIUM'
        },

        'pretzel': {
            'name': 'pretzel',
            'description': 'Detects pretzel/3-fold stage (highly elongated)',
            'prompt': """Analyze this C. elegans embryo and determine if it has reached the PRETZEL/3-FOLD STAGE.

Key characteristics of 3-fold stage (~550 minutes, ~9 hours):
- Highly elongated, approximately 3x the eggshell length
- Tightly folded/coiled within eggshell (pretzel-like)
- Active movement visible
- Clear segmentation and pharynx structure
- Still within eggshell

Focus on the CURRENT/LATEST image.

Respond in this exact format:
DETECTED: [YES/NO]
CONFIDENCE: [HIGH/MEDIUM/LOW]
REASONING: [Brief explanation]""",
            'use_temporal_context': True,
            'temporal_context_size': 5,
            'confidence_threshold': 'MEDIUM'
        },

        'gastrulation': {
            'name': 'gastrulation',
            'description': 'Detects onset of gastrulation',
            'prompt': """Analyze this C. elegans embryo and determine if GASTRULATION has begun.

Key characteristics of gastrulation (~210 minutes, ~3.5 hours):
- Visible internalization of cells (especially E cells - gut precursors)
- Loss of clear spherical shape
- Cell movements visible
- Typically after ~26-28 cell stage

Focus on the CURRENT/LATEST image.

Respond in this exact format:
DETECTED: [YES/NO]
CONFIDENCE: [HIGH/MEDIUM/LOW]
REASONING: [Brief explanation]""",
            'use_temporal_context': True,
            'temporal_context_size': 5,
            'confidence_threshold': 'MEDIUM'
        },

        'first_division': {
            'name': 'first_division',
            'description': 'Detects first cell division (1-cell to 2-cell)',
            'prompt': """Analyze this C. elegans embryo and determine if FIRST CELL DIVISION has occurred.

Key characteristics:
- Transition from single large cell to two cells
- Unequal division: larger AB cell (anterior) and smaller P1 cell (posterior)
- Clear cell boundary/cleavage plane visible
- Occurs ~40-50 minutes after fertilization

Focus on the CURRENT/LATEST image.

Respond in this exact format:
DETECTED: [YES/NO]
CONFIDENCE: [HIGH/MEDIUM/LOW]
REASONING: [Brief explanation]""",
            'use_temporal_context': True,
            'temporal_context_size': 3,
            'confidence_threshold': 'HIGH'
        },
    }
