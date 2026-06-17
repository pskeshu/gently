"""
Conversational Microscopy Agent

AI agent that acts as a scientific collaborator for diSPIM experiments.

This module re-exports from the new locations for backward compatibility.
"""

from gently_perception import Perceiver, PerceptionOutput
from gently_perception import Session as PerceptionSession

from gently.app.agent import MicroscopyAgent
from gently.harness.orchestration.plan_synthesis import PlanSynthesizer, PlanValidator
from gently.harness.state import EmbryoState, ExperimentState, ImageRecord

try:
    from gently.hardware.dispim.device_factory import create_devices_from_mmcore
except ImportError:
    create_devices_from_mmcore = None  # type: ignore[assignment]
# Import tools package to register all tools
from gently.app import tools  # noqa: F401
from gently.app.benchmark import (
    BenchmarkResults,
    print_benchmark_results,
    run_benchmark,
)
from gently.app.queue_server_client import QueueServerClient
from gently.harness.tools.registry import (
    ToolCategory,
    ToolRegistry,
    get_tool_registry,
    tool,
)

__all__ = [
    "MicroscopyAgent",
    "QueueServerClient",
    "EmbryoState",
    "ExperimentState",
    "ImageRecord",
    "PlanSynthesizer",
    "PlanValidator",
    "Perceiver",
    "PerceptionOutput",
    "PerceptionSession",
    "create_devices_from_mmcore",
    "ToolRegistry",
    "get_tool_registry",
    "tool",
    "ToolCategory",
    "run_benchmark",
    "BenchmarkResults",
    "print_benchmark_results",
]
