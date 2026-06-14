"""
Perception benchmark framework.

Tools for offline testing of perception engine accuracy against ground truth labels.
"""

from .ground_truth import GroundTruth
from .metrics import PerceptionMetrics
from .runner import BenchmarkConfig, EmbryoResult, PerceptionBenchmark
from .testset import OfflineTestset, TestCase

__all__ = [
    "GroundTruth",
    "OfflineTestset",
    "TestCase",
    "PerceptionBenchmark",
    "BenchmarkConfig",
    "EmbryoResult",
    "PerceptionMetrics",
]
