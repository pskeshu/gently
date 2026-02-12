"""
Conversational Microscopy Copilot

AI agent that acts as a scientific collaborator for diSPIM microscopy experiments.
"""

from .copilot import MicroscopyCopilot
from .state import EmbryoState, ExperimentState, ImageRecord
from .plan_synthesis import PlanSynthesizer, PlanValidator
from .perception import PerceptionManager, PerceptionResult, PerceptionSession
from .device_factory import create_devices_from_mmcore
from .queue_server_client import QueueServerClient
from .tool_registry import ToolRegistry, get_tool_registry, tool, ToolCategory
from .benchmark import run_benchmark, BenchmarkResults, print_benchmark_results

# Import tools package to register all tools
from . import tools

__all__ = [
    'MicroscopyCopilot',
    'QueueServerClient',
    'EmbryoState',
    'ExperimentState',
    'ImageRecord',
    'PlanSynthesizer',
    'PlanValidator',
    # Perception system
    'PerceptionManager',
    'PerceptionResult',
    'PerceptionSession',
    'create_devices_from_mmcore',
    # Tool registry
    'ToolRegistry',
    'get_tool_registry',
    'tool',
    'ToolCategory',
    # Benchmark
    'run_benchmark',
    'BenchmarkResults',
    'print_benchmark_results',
]
