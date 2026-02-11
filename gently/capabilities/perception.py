"""
Perception capability — wraps the VLM perception system.

Provides stage classification and feature detection for the agent.
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class StageClassification:
    """Result of stage classification."""
    stage: str
    confidence: float
    is_transitional: bool = False
    transition_between: Optional[List[str]] = None
    reasoning: str = ""
    observed_features: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Detection:
    """A detected feature."""
    feature: str
    confidence: float
    location: Optional[Dict[str, float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Detections:
    """Collection of detected features."""
    items: List[Detection] = field(default_factory=list)
    embryo_id: Optional[str] = None
    timepoint: Optional[int] = None


class PerceptionCapability:
    """
    Wraps the existing perception/VLM system.

    Provides stage classification and feature detection.
    """

    def __init__(self, perception_manager: Optional[Any] = None):
        """
        Parameters
        ----------
        perception_manager : Any, optional
            Existing perception manager. If None, perception is simulated.
        """
        self.perception = perception_manager

    async def classify_stage(
        self,
        image_b64: Optional[str] = None,
        image_path: Optional[str] = None,
        embryo_id: Optional[str] = None,
        timepoint: Optional[int] = None,
        volume: Optional[Any] = None,
    ) -> StageClassification:
        """
        Classify the developmental stage of an embryo.

        Parameters
        ----------
        image_b64 : str, optional
            Base64-encoded image
        image_path : str, optional
            Path to image file
        embryo_id : str, optional
            Embryo ID for context
        timepoint : int, optional
            Timepoint for context
        volume : np.ndarray, optional
            3D volume data for multi-view analysis

        Returns
        -------
        StageClassification
            Classification result
        """
        logger.info(f"Classifying stage: embryo={embryo_id}, tp={timepoint}")

        if self.perception is None:
            # Simulated classification
            return StageClassification(
                stage="comma",
                confidence=0.85,
                reasoning="Simulated classification result",
            )

        try:
            # Call actual perception system
            result = await self.perception.classify_stage(
                image_b64=image_b64,
                image_path=image_path,
                embryo_id=embryo_id,
                timepoint=timepoint,
                volume=volume,
            )
            return StageClassification(
                stage=result.stage,
                confidence=result.confidence,
                is_transitional=result.is_transitional,
                transition_between=result.transition_between,
                reasoning=result.reasoning,
                observed_features=result.observed_features.model_dump()
                if hasattr(result.observed_features, "model_dump")
                else {},
            )
        except Exception as e:
            logger.error(f"Stage classification failed: {e}")
            return StageClassification(
                stage="unknown",
                confidence=0.0,
                reasoning=f"Classification failed: {e}",
            )

    async def detect_features(
        self,
        image_b64: Optional[str] = None,
        image_path: Optional[str] = None,
        embryo_id: Optional[str] = None,
        features: Optional[List[str]] = None,
    ) -> Detections:
        """
        Detect features in an image.

        Parameters
        ----------
        image_b64 : str, optional
            Base64-encoded image
        image_path : str, optional
            Path to image file
        embryo_id : str, optional
            Embryo ID
        features : List[str], optional
            Specific features to look for

        Returns
        -------
        Detections
            Detected features
        """
        logger.info(f"Detecting features: embryo={embryo_id}, features={features}")

        if self.perception is None:
            # Simulated detection
            return Detections(
                items=[
                    Detection(feature="head", confidence=0.9),
                    Detection(feature="tail", confidence=0.85),
                ],
                embryo_id=embryo_id,
            )

        try:
            # Call actual detection
            result = await self.perception.detect_features(
                image_b64=image_b64,
                image_path=image_path,
                features=features,
            )
            return Detections(
                items=[
                    Detection(
                        feature=d.feature,
                        confidence=d.confidence,
                        location=d.location,
                    )
                    for d in result
                ],
                embryo_id=embryo_id,
            )
        except Exception as e:
            logger.error(f"Feature detection failed: {e}")
            return Detections(embryo_id=embryo_id)

    async def check_hatching(
        self,
        image_b64: Optional[str] = None,
        image_path: Optional[str] = None,
        embryo_id: Optional[str] = None,
    ) -> tuple[bool, float]:
        """
        Check if an embryo is hatching.

        Parameters
        ----------
        image_b64 : str, optional
            Base64-encoded image
        image_path : str, optional
            Path to image file
        embryo_id : str, optional
            Embryo ID

        Returns
        -------
        (is_hatching, confidence)
            Whether hatching is occurring and confidence level
        """
        logger.info(f"Checking hatching: embryo={embryo_id}")

        if self.perception is None:
            # Simulated
            return False, 0.95

        try:
            result = await self.perception.check_hatching(
                image_b64=image_b64,
                image_path=image_path,
            )
            return result.is_hatching, result.confidence
        except Exception as e:
            logger.error(f"Hatching check failed: {e}")
            return False, 0.0
