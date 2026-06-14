"""
Gently ML — Structured machine learning training framework.

Provides:
- Architecture registry with suitability metadata
- Local single-GPU trainer (subprocess-based)
- Dataset loading from FileStore
- Evaluation metrics and reporting
- Federated averaging for distributed training
"""

from .architectures import ARCHITECTURE_REGISTRY, get_suitable_architectures
from .models import (
    DataSplit,
    MLPipeline,
    ModelArchitectureType,
    ModelConfig,
    TrainingConfig,
    TrainingRun,
    TrainingStatus,
)

__all__ = [
    "ARCHITECTURE_REGISTRY",
    "DataSplit",
    "MLPipeline",
    "ModelArchitectureType",
    "ModelConfig",
    "TrainingConfig",
    "TrainingRun",
    "TrainingStatus",
    "get_suitable_architectures",
]
