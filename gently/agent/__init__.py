"""
Conversational Microscopy Agent

AI agent that acts as a scientific collaborator for diSPIM experiments.

This module re-exports from the new locations for backward compatibility.
"""

from gently.app.agent import MicroscopyCopilot  # legacy name
from gently.harness.state import EmbryoState, ExperimentState, ImageRecord
from gently.harness.orchestration.plan_synthesis import PlanSynthesizer, PlanValidator
from gently.harness.perception import PerceptionManager, PerceptionResult, PerceptionSession
try:
    from gently.hardware.dispim.device_factory import create_devices_from_mmcore
except ImportError:
    create_devices_from_mmcore = None
from gently.app.queue_server_client import QueueServerClient
from gently.harness.tools.registry import ToolRegistry, get_tool_registry, tool, ToolCategory
from gently.app.benchmark import run_benchmark, BenchmarkResults, print_benchmark_results

# Import tools package to register all tools
from gently.app import tools

# Alias
MicroscopyAgent = MicroscopyCopilot

__all__ = [
    'MicroscopyCopilot',
    'MicroscopyAgent',
    'QueueServerClient',
    'EmbryoState',
    'ExperimentState',
    'ImageRecord',
    'PlanSynthesizer',
    'PlanValidator',
    'PerceptionManager',
    'PerceptionResult',
    'PerceptionSession',
    'create_devices_from_mmcore',
    'ToolRegistry',
    'get_tool_registry',
    'tool',
    'ToolCategory',
    'run_benchmark',
    'BenchmarkResults',
    'print_benchmark_results',
]
