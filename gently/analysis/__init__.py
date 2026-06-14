"""
Composable Analysis Pipeline System

Provides a flexible framework for chaining analysis operations:
- VLM (Claude Vision) analysis
- SAM segmentation
- Classical computer vision
- Custom analysis steps

Pipelines are composable, configurable, and track data lineage via UIDs.
"""

from .pipeline import (
    AnalysisResult,
    AnalysisStep,
    Pipeline,
    PipelineBuilder,
    StepType,
    create_embryo_detection_pipeline,
    create_hatching_detection_pipeline,
    create_morphology_analysis_pipeline,
)
from .steps import (
    BlobDetectionStep,
    MaxProjectionStep,
    MorphologyStep,
    SAMStep,
    ThresholdStep,
    VLMStep,
)

__all__ = [
    # Core
    "AnalysisStep",
    "AnalysisResult",
    "Pipeline",
    "PipelineBuilder",
    "StepType",
    # Steps
    "VLMStep",
    "SAMStep",
    "MaxProjectionStep",
    "ThresholdStep",
    "MorphologyStep",
    "BlobDetectionStep",
    # Pre-built pipelines
    "create_embryo_detection_pipeline",
    "create_hatching_detection_pipeline",
    "create_morphology_analysis_pipeline",
]
