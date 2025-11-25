"""
Conversational Microscopy Copilot

AI agent that acts as a scientific collaborator for diSPIM microscopy experiments.
"""

from .copilot import MicroscopyCopilot
from .state import EmbryoState, ExperimentState, ImageRecord
from .plan_synthesis import PlanSynthesizer, PlanValidator
from .image_manager import ImageManager
from .detector import Detector, DetectionResult, DetectorConditions, DetectorActions, DetectionMode, ConfidenceLevel
from .detector_registry import DetectorRegistry, get_detector_presets
from .detection_queue import DetectionQueue
from .rich_cli import run_rich_cli, RichCopilotCLI
from .autocomplete import create_completer, CopilotCompleter
from .device_factory import create_devices_from_mmcore
from .microscope_client import MicroscopeClient
from .queue_server_client import QueueServerClient

__all__ = [
    'MicroscopyCopilot',
    'MicroscopeClient',
    'QueueServerClient',
    'EmbryoState',
    'ExperimentState',
    'ImageRecord',
    'PlanSynthesizer',
    'PlanValidator',
    'ImageManager',
    'Detector',
    'DetectionResult',
    'DetectorConditions',
    'DetectorActions',
    'DetectionMode',
    'ConfidenceLevel',
    'DetectorRegistry',
    'DetectionQueue',
    'get_detector_presets',
    'run_rich_cli',
    'RichCopilotCLI',
    'create_completer',
    'CopilotCompleter',
    'create_devices_from_mmcore',
]
