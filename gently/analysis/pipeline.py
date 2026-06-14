"""
Core Pipeline Framework

Provides the base classes and execution engine for composable analysis pipelines.
"""

import logging
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any

from ..settings import settings

logger = logging.getLogger(__name__)


class StepType(Enum):
    """Types of analysis steps"""

    VLM = auto()  # Vision Language Model (Claude)
    SAM = auto()  # Segment Anything Model
    CLASSICAL = auto()  # Classical CV (OpenCV, scikit-image)
    PROJECTION = auto()  # Dimension reduction (max proj, etc.)
    THRESHOLD = auto()  # Thresholding operations
    MORPHOLOGY = auto()  # Morphological operations
    DETECTION = auto()  # Object detection
    CUSTOM = auto()  # Custom step


@dataclass
class AnalysisResult:
    """
    Result from an analysis step or pipeline

    All results carry:
    - UID for tracking
    - Parent UID for lineage
    - Metadata about the analysis
    - The actual result data
    """

    uid: str = field(default_factory=lambda: str(uuid.uuid4()))
    step_name: str = ""
    step_type: StepType = StepType.CUSTOM
    parent_uid: str | None = None
    timestamp: datetime = field(default_factory=datetime.now)

    # Result data
    data: Any = None  # Primary result (image, mask, dict, etc.)
    metadata: dict[str, Any] = field(default_factory=dict)

    # Execution info
    duration_ms: float = 0.0
    success: bool = True
    error: str | None = None

    def __str__(self) -> str:
        status = "+" if self.success else "x"
        return f"{status} {self.step_name} ({self.step_type.name}) [{self.uid[:8]}]"

    def to_dict(self) -> dict:
        """Serialize to dictionary"""
        return {
            "uid": self.uid,
            "step_name": self.step_name,
            "step_type": self.step_type.name,
            "parent_uid": self.parent_uid,
            "timestamp": self.timestamp.isoformat(),
            "metadata": self.metadata,
            "duration_ms": self.duration_ms,
            "success": self.success,
            "error": self.error,
        }


class AnalysisStep(ABC):
    """
    Base class for analysis pipeline steps

    Each step:
    - Has a name and type
    - Takes input data and produces AnalysisResult
    - Can be sync or async
    - Tracks execution time
    """

    def __init__(
        self,
        name: str | None = None,
        step_type: StepType = StepType.CUSTOM,
        config: dict | None = None,
    ):
        self.name = name or self.__class__.__name__
        self.step_type = step_type
        self.config = config or {}

    @abstractmethod
    async def execute(
        self,
        input_data: Any,
        context: dict | None = None,
    ) -> AnalysisResult:
        """
        Execute the analysis step

        Parameters
        ----------
        input_data : any
            Input data (image, volume, previous result, etc.)
        context : dict, optional
            Shared context (embryo_id, timepoint, etc.)

        Returns
        -------
        AnalysisResult
            Result with data and metadata
        """
        pass

    async def __call__(
        self,
        input_data: Any,
        context: dict | None = None,
    ) -> AnalysisResult:
        """Allow calling step directly"""
        return await self.execute(input_data, context)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name='{self.name}')"


class Pipeline:
    """
    A sequence of analysis steps executed in order

    Features:
    - Sequential execution with result passing
    - Parallel branch support
    - Error handling and recovery
    - Full lineage tracking
    """

    def __init__(
        self,
        name: str = "pipeline",
        steps: list[AnalysisStep] | None = None,
        store_intermediate: bool = False,
    ):
        self.name = name
        self.steps: list[AnalysisStep] = steps or []
        self.store_intermediate = store_intermediate
        self._data_store = None

    def add_step(self, step: AnalysisStep) -> "Pipeline":
        """Add a step to the pipeline (fluent)"""
        self.steps.append(step)
        return self

    def set_data_store(self, store):
        """Set data store for result persistence"""
        self._data_store = store

    async def execute(
        self,
        input_data: Any,
        context: dict | None = None,
    ) -> AnalysisResult:
        """
        Execute the full pipeline

        Parameters
        ----------
        input_data : any
            Initial input data
        context : dict, optional
            Shared context passed to all steps

        Returns
        -------
        AnalysisResult
            Final result with full lineage
        """
        context = context or {}
        current_data = input_data
        parent_uid = context.get("input_uid")
        results: list[AnalysisResult] = []

        start_time = time.time()

        for i, step in enumerate(self.steps):
            step_start = time.time()

            try:
                # Execute step
                result = await step.execute(current_data, context)
                result.parent_uid = parent_uid

                # Track timing
                result.duration_ms = (time.time() - step_start) * 1000

                # Store intermediate result if configured
                if self.store_intermediate and self._data_store:
                    try:
                        ref = self._data_store.store(
                            data=result.data,
                            data_type="analysis",
                            metadata={
                                "step_name": result.step_name,
                                "step_type": result.step_type.name,
                                "pipeline": self.name,
                                "step_index": i,
                                **result.metadata,
                            },
                            parent_uid=parent_uid,
                        )
                        result.uid = ref.uid
                    except Exception as e:
                        logger.warning(f"Failed to store intermediate result: {e}")

                results.append(result)
                parent_uid = result.uid

                # Pass result data to next step
                current_data = result.data

                logger.debug(f"Step {i + 1}/{len(self.steps)}: {result}")

            except Exception as e:
                logger.error(f"Pipeline step {step.name} failed: {e}")
                return AnalysisResult(
                    step_name=step.name,
                    step_type=step.step_type,
                    parent_uid=parent_uid,
                    success=False,
                    error=str(e),
                    metadata={"failed_at_step": i, "pipeline": self.name},
                )

        # Create final result
        total_duration = (time.time() - start_time) * 1000
        final_result = AnalysisResult(
            step_name=self.name,
            step_type=StepType.CUSTOM,
            parent_uid=context.get("input_uid"),
            data=current_data,
            duration_ms=total_duration,
            success=True,
            metadata={
                "pipeline": self.name,
                "num_steps": len(self.steps),
                "step_results": [r.to_dict() for r in results],
            },
        )

        # Store final result
        if self._data_store:
            try:
                ref = self._data_store.store(
                    data=final_result.data,
                    data_type="analysis",
                    metadata={
                        "pipeline": self.name,
                        "final": True,
                        **final_result.metadata,
                    },
                    parent_uid=parent_uid,
                )
                final_result.uid = ref.uid
            except Exception as e:
                logger.warning(f"Failed to store final result: {e}")

        return final_result

    async def __call__(
        self,
        input_data: Any,
        context: dict | None = None,
    ) -> AnalysisResult:
        """Allow calling pipeline directly"""
        return await self.execute(input_data, context)

    def __repr__(self) -> str:
        step_names = [s.name for s in self.steps]
        return f"Pipeline(name='{self.name}', steps={step_names})"


class PipelineBuilder:
    """
    Fluent builder for creating analysis pipelines

    Usage:
        pipeline = (PipelineBuilder("embryo_detection")
            .max_projection()
            .threshold(method="otsu")
            .morphology(operation="open", kernel_size=5)
            .sam_segment(prompt="embryo")
            .vlm_analyze(prompt="Count embryos")
            .build())
    """

    def __init__(self, name: str = "pipeline"):
        self.name = name
        self._steps: list[AnalysisStep] = []
        self._store_intermediate = False
        self._data_store = None

    def add(self, step: AnalysisStep) -> "PipelineBuilder":
        """Add a custom step"""
        self._steps.append(step)
        return self

    def max_projection(self, axis: int = 0) -> "PipelineBuilder":
        """Add max projection step"""
        from .steps import MaxProjectionStep

        self._steps.append(MaxProjectionStep(axis=axis))
        return self

    def threshold(
        self,
        method: str = "otsu",
        value: float | None = None,
    ) -> "PipelineBuilder":
        """Add threshold step"""
        from .steps import ThresholdStep

        self._steps.append(ThresholdStep(method=method, value=value))
        return self

    def morphology(
        self,
        operation: str = "open",
        kernel_size: int = 3,
    ) -> "PipelineBuilder":
        """Add morphological operation step"""
        from .steps import MorphologyStep

        self._steps.append(MorphologyStep(operation=operation, kernel_size=kernel_size))
        return self

    def blob_detection(
        self,
        min_sigma: float = 10,
        max_sigma: float = 50,
        threshold: float = 0.1,
    ) -> "PipelineBuilder":
        """Add blob detection step"""
        from .steps import BlobDetectionStep

        self._steps.append(
            BlobDetectionStep(
                min_sigma=min_sigma,
                max_sigma=max_sigma,
                threshold=threshold,
            )
        )
        return self

    def sam_segment(
        self,
        prompt: str | None = None,
        points: list | None = None,
    ) -> "PipelineBuilder":
        """Add SAM segmentation step"""
        from .steps import SAMStep

        self._steps.append(SAMStep(prompt=prompt, points=points))
        return self

    def vlm_analyze(
        self,
        prompt: str,
        model: str = settings.models.perception,
        max_tokens: int = 1024,
    ) -> "PipelineBuilder":
        """Add VLM analysis step"""
        from .steps import VLMStep

        self._steps.append(VLMStep(prompt=prompt, model=model, max_tokens=max_tokens))
        return self

    def store_intermediate(self, store=None) -> "PipelineBuilder":
        """Enable storing intermediate results"""
        self._store_intermediate = True
        self._data_store = store
        return self

    def build(self) -> Pipeline:
        """Build the pipeline"""
        pipeline = Pipeline(
            name=self.name,
            steps=self._steps,
            store_intermediate=self._store_intermediate,
        )
        if self._data_store:
            pipeline.set_data_store(self._data_store)
        return pipeline


# =============================================================================
# Pre-built pipelines for common analysis tasks
# =============================================================================


def create_embryo_detection_pipeline(
    use_sam: bool = True,
    use_vlm_verification: bool = False,
) -> Pipeline:
    """
    Create a pipeline for embryo detection

    Parameters
    ----------
    use_sam : bool
        Whether to use SAM for segmentation
    use_vlm_verification : bool
        Whether to verify with Claude Vision

    Returns
    -------
    Pipeline
        Configured embryo detection pipeline
    """
    builder = PipelineBuilder("embryo_detection")

    # Preprocessing
    builder.max_projection(axis=0)
    builder.threshold(method="percentile_bright", value=99)
    builder.morphology(operation="close", kernel_size=5)

    # Detection
    if use_sam:
        builder.sam_segment(prompt="embryo")
    else:
        builder.blob_detection(min_sigma=20, max_sigma=80)

    # Verification
    if use_vlm_verification:
        builder.vlm_analyze(
            prompt="Verify these are C. elegans embryos. "
            "Count the number of valid embryos and note any false positives."
        )

    return builder.build()


def create_hatching_detection_pipeline() -> Pipeline:
    """
    Create a pipeline for hatching detection

    Returns
    -------
    Pipeline
        Configured hatching detection pipeline
    """
    builder = PipelineBuilder("hatching_detection")
    builder.max_projection(axis=0)
    builder.vlm_analyze(
        prompt="""Analyze this C. elegans embryo image for hatching.

Look for:
1. Empty or broken eggshell
2. Larva outside the egg
3. Changed morphology from previous timepoints

Respond with:
HATCHED: [YES/NO]
CONFIDENCE: [HIGH/MEDIUM/LOW]
REASONING: [brief explanation]"""
    )
    return builder.build()


def create_morphology_analysis_pipeline() -> Pipeline:
    """
    Create a pipeline for morphology analysis

    Returns
    -------
    Pipeline
        Configured morphology analysis pipeline
    """
    builder = PipelineBuilder("morphology_analysis")
    builder.max_projection(axis=0)
    builder.threshold(method="otsu")
    builder.morphology(operation="close", kernel_size=3)
    builder.vlm_analyze(
        prompt="""Analyze the morphology of this C. elegans embryo.

Describe:
1. Developmental stage (early cleavage, gastrula, comma, pretzel, etc.)
2. Overall shape and symmetry
3. Any abnormalities observed
4. Estimated cell count if visible

Format your response as structured data."""
    )
    return builder.build()
