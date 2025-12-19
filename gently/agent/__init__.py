"""
Conversational Microscopy Copilot

AI agent that acts as a scientific collaborator for diSPIM microscopy experiments.
"""

from .copilot import MicroscopyCopilot
from .state import EmbryoState, ExperimentState, ImageRecord
from .plan_synthesis import PlanSynthesizer, PlanValidator
from .image_manager import ImageManager
from .perception import PerceptionManager, PerceptionResult, PerceptionSession
from .rich_cli import run_rich_cli, RichCopilotCLI
from .autocomplete import create_completer, CopilotCompleter
from .device_factory import create_devices_from_mmcore
from .microscope_client import MicroscopeClient
from .queue_server_client import QueueServerClient
from .tool_registry import ToolRegistry, get_tool_registry, tool, ToolCategory

# Import tools package to register all tools
from . import tools

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
    # Perception system
    'PerceptionManager',
    'PerceptionResult',
    'PerceptionSession',
    'run_rich_cli',
    'RichCopilotCLI',
    'create_completer',
    'CopilotCompleter',
    'create_devices_from_mmcore',
    # Tool registry
    'ToolRegistry',
    'get_tool_registry',
    'tool',
    'ToolCategory',
]
