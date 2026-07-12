"""
Developmental Stage Tracker for C. elegans Embryos

Uses Claude Vision to classify embryo developmental stages and
predict time-to-hatching based on observed progression.

This module lives in the organism package because the stage enum,
timing tables, and classification prompt are all C. elegans-specific.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, cast

import anthropic
from anthropic.types import MessageParam, TextBlock

from ...settings import settings

logger = logging.getLogger(__name__)


class DevelopmentalStage(str, Enum):
    """C. elegans developmental stages in order (tracker-resolution)

    This enum is finer-grained than the perception stages — it includes
    individual cell-division stages that the perception VLM doesn't
    distinguish.  The perception enum (in stages.py) maps "early" to
    everything before comma.
    """

    ONE_CELL = "1-cell"
    TWO_CELL = "2-cell"
    FOUR_CELL = "4-cell"
    EIGHT_CELL = "8-cell"
    GASTRULATION = "gastrulation"
    COMMA = "comma"
    ONE_POINT_FIVE_FOLD = "1.5-fold"
    TWO_FOLD = "2-fold"
    PRETZEL = "pretzel"  # 3-fold
    PRE_HATCHING = "pre-hatching"
    HATCHING = "hatching"
    HATCHED = "hatched"
    UNKNOWN = "unknown"
    DEAD = "dead"


# Approximate timing at 20°C (minutes from fertilization)
STAGE_TIMING_20C = {
    DevelopmentalStage.ONE_CELL: 0,
    DevelopmentalStage.TWO_CELL: 40,
    DevelopmentalStage.FOUR_CELL: 55,
    DevelopmentalStage.EIGHT_CELL: 80,
    DevelopmentalStage.GASTRULATION: 210,
    DevelopmentalStage.COMMA: 400,
    DevelopmentalStage.ONE_POINT_FIVE_FOLD: 450,
    DevelopmentalStage.TWO_FOLD: 500,
    DevelopmentalStage.PRETZEL: 550,
    DevelopmentalStage.PRE_HATCHING: 750,
    DevelopmentalStage.HATCHING: 800,
    DevelopmentalStage.HATCHED: 840,
}

# Time from each stage to hatching (minutes at 20°C)
TIME_TO_HATCHING = {
    DevelopmentalStage.ONE_CELL: 800,
    DevelopmentalStage.TWO_CELL: 760,
    DevelopmentalStage.FOUR_CELL: 745,
    DevelopmentalStage.EIGHT_CELL: 720,
    DevelopmentalStage.GASTRULATION: 590,
    DevelopmentalStage.COMMA: 400,
    DevelopmentalStage.ONE_POINT_FIVE_FOLD: 350,
    DevelopmentalStage.TWO_FOLD: 300,
    DevelopmentalStage.PRETZEL: 250,
    DevelopmentalStage.PRE_HATCHING: 50,
    DevelopmentalStage.HATCHING: 0,
    DevelopmentalStage.HATCHED: 0,
}

# Variability in timing (±minutes) for confidence intervals
TIMING_VARIABILITY = {
    DevelopmentalStage.ONE_CELL: 60,
    DevelopmentalStage.TWO_CELL: 50,
    DevelopmentalStage.FOUR_CELL: 45,
    DevelopmentalStage.EIGHT_CELL: 40,
    DevelopmentalStage.GASTRULATION: 60,
    DevelopmentalStage.COMMA: 45,
    DevelopmentalStage.ONE_POINT_FIVE_FOLD: 40,
    DevelopmentalStage.TWO_FOLD: 35,
    DevelopmentalStage.PRETZEL: 30,
    DevelopmentalStage.PRE_HATCHING: 20,
    DevelopmentalStage.HATCHING: 10,
    DevelopmentalStage.HATCHED: 0,
}


@dataclass
class HatchingPrediction:
    """Prediction of time to hatching with confidence interval"""

    embryo_id: str
    current_stage: DevelopmentalStage
    predicted_minutes: int
    min_minutes: int  # Lower bound (optimistic)
    max_minutes: int  # Upper bound (conservative)
    confidence: str  # Based on stage classification confidence
    timestamp: datetime = field(default_factory=datetime.now)

    @property
    def predicted_hours(self) -> float:
        return self.predicted_minutes / 60

    @property
    def range_hours(self) -> tuple[float, float]:
        return (self.min_minutes / 60, self.max_minutes / 60)

    def to_dict(self) -> dict:
        return {
            "embryo_id": self.embryo_id,
            "current_stage": self.current_stage.value,
            "predicted_minutes": self.predicted_minutes,
            "predicted_hours": self.predicted_hours,
            "min_minutes": self.min_minutes,
            "max_minutes": self.max_minutes,
            "range_hours": self.range_hours,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat(),
        }

    def __str__(self) -> str:
        return (
            f"{self.predicted_hours:.1f}h "
            f"(range: {self.range_hours[0]:.1f}-{self.range_hours[1]:.1f}h, "
            f"{self.confidence} confidence)"
        )


@dataclass
class StageClassification:
    """Result of a stage classification"""

    stage: DevelopmentalStage
    confidence: str  # HIGH, MEDIUM, LOW
    reasoning: str
    timestamp: datetime = field(default_factory=datetime.now)
    timepoint: int = 0
    predicted_minutes_to_hatching: int | None = None

    def to_dict(self) -> dict:
        return {
            "stage": self.stage.value,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "timestamp": self.timestamp.isoformat(),
            "timepoint": self.timepoint,
            "predicted_minutes_to_hatching": self.predicted_minutes_to_hatching,
        }


STAGE_CLASSIFICATION_PROMPT = """Analyze this C. elegans embryo image and determine its
DEVELOPMENTAL STAGE.

Stages in order (earliest to latest):
- 1-cell: Single cell, spherical, no division
- 2-cell: Two cells (AB larger, P1 smaller)
- 4-cell: Four cells in diamond pattern
- 8-cell: Eight cells
- gastrulation: Cells internalizing, ~26+ cells
- comma: Bean/comma shape, ventral curvature, ~400 min
- 1.5-fold: Elongating, 1.5x eggshell length
- 2-fold: More elongated, 2x length, folding
- pretzel: Highly elongated, 3x length, tightly coiled (3-fold)
- pre-hatching: Active movement, pushing against shell
- hatching: Shell breach visible, emerging
- hatched: Larva outside shell
- dead: Cytoplasmic blebbing, loss of cell boundaries
- unknown: Cannot determine

Focus on the CURRENT/LATEST image shown.

Respond in this exact format:
STAGE: [stage name from list above]
CONFIDENCE: [HIGH/MEDIUM/LOW]
REASONING: [Brief explanation of what you observe that indicates this stage]"""


class DevelopmentalTracker:
    """
    Tracks embryo development across timepoints using Claude Vision.

    Features:
    - Stage classification
    - Time-to-hatching prediction
    - Stage history tracking
    - Progression validation
    """

    def __init__(
        self,
        claude_client: anthropic.Anthropic | None = None,
        model: str = settings.models.perception,
    ):
        """
        Parameters
        ----------
        claude_client : Anthropic, optional
            Claude client (will create one if not provided)
        model : str
            Claude model to use
        """
        self.claude = claude_client or anthropic.Anthropic()
        self.model = model

        # Stage history per embryo
        self._stage_history: dict[str, list[StageClassification]] = {}

    def classify_stage(
        self,
        image_b64: str,
        embryo_id: str,
        timepoint: int = 0,
        recent_images: list[dict] | None = None,
    ) -> StageClassification:
        """
        Classify the developmental stage of an embryo

        Parameters
        ----------
        image_b64 : str
            Base64-encoded JPEG image
        embryo_id : str
            Embryo identifier
        timepoint : int
            Current timepoint
        recent_images : list, optional
            Recent images for temporal context

        Returns
        -------
        StageClassification
            Classification result
        """
        # Build content for Claude Vision
        content: list[dict[str, Any]] = []

        # Add temporal context if available
        if recent_images and len(recent_images) > 1:
            content.append(
                {
                    "type": "text",
                    "text": f"Recent images from {embryo_id} (for temporal context):",
                }
            )
            for img in recent_images[:-1]:  # All but last
                content.append({"type": "text", "text": f"Timepoint {img.get('timepoint', '?')}"})
                content.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": img["b64_image"],
                        },
                    }
                )

        # Add current image
        content.append(
            {
                "type": "text",
                "text": f"CURRENT image (timepoint {timepoint}) - classify this one:",
            }
        )
        content.append(
            {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": image_b64,
                },
            }
        )

        # Add prompt
        content.append({"type": "text", "text": STAGE_CLASSIFICATION_PROMPT})

        try:
            messages: list[MessageParam] = [{"role": "user", "content": cast(Any, content)}]
            response = self.claude.messages.create(
                model=self.model,
                max_tokens=500,
                messages=messages,
            )

            result = self._parse_classification(cast(TextBlock, response.content[0]).text)
            result.timepoint = timepoint

            # Calculate time to hatching
            if result.stage in TIME_TO_HATCHING:
                result.predicted_minutes_to_hatching = TIME_TO_HATCHING[result.stage]

            # Store in history
            if embryo_id not in self._stage_history:
                self._stage_history[embryo_id] = []
            self._stage_history[embryo_id].append(result)

            logger.info(
                f"Stage classification for {embryo_id} t{timepoint}: "
                f"{result.stage.value} ({result.confidence})"
            )

            return result

        except Exception as e:
            logger.error(f"Stage classification failed: {e}")
            return StageClassification(
                stage=DevelopmentalStage.UNKNOWN,
                confidence="LOW",
                reasoning=f"Classification failed: {str(e)}",
                timepoint=timepoint,
            )

    def _parse_classification(self, response_text: str) -> StageClassification:
        """Parse Claude's response into a classification"""
        stage = DevelopmentalStage.UNKNOWN
        confidence = "LOW"
        reasoning = ""

        lines = response_text.strip().split("\n")

        for line in lines:
            line = line.strip()
            if line.startswith("STAGE:"):
                stage_str = line.split(":", 1)[1].strip().lower()
                # Map to enum
                stage = self._parse_stage_name(stage_str)
            elif line.startswith("CONFIDENCE:"):
                confidence = line.split(":", 1)[1].strip().upper()
            elif line.startswith("REASONING:"):
                reasoning = line.split(":", 1)[1].strip()

        # Capture multi-line reasoning
        if not reasoning:
            in_reasoning = False
            reasoning_lines = []
            for line in lines:
                if line.startswith("REASONING:"):
                    in_reasoning = True
                    reasoning_lines.append(line.split(":", 1)[1].strip())
                elif in_reasoning and line:
                    reasoning_lines.append(line)
            if reasoning_lines:
                reasoning = " ".join(reasoning_lines)

        return StageClassification(
            stage=stage,
            confidence=confidence,
            reasoning=reasoning or "No reasoning provided",
        )

    def _parse_stage_name(self, name: str) -> DevelopmentalStage:
        """Parse stage name string to enum"""
        name = name.lower().strip()

        # Direct matches
        mappings = {
            "1-cell": DevelopmentalStage.ONE_CELL,
            "one-cell": DevelopmentalStage.ONE_CELL,
            "2-cell": DevelopmentalStage.TWO_CELL,
            "two-cell": DevelopmentalStage.TWO_CELL,
            "4-cell": DevelopmentalStage.FOUR_CELL,
            "four-cell": DevelopmentalStage.FOUR_CELL,
            "8-cell": DevelopmentalStage.EIGHT_CELL,
            "eight-cell": DevelopmentalStage.EIGHT_CELL,
            "gastrulation": DevelopmentalStage.GASTRULATION,
            "comma": DevelopmentalStage.COMMA,
            "1.5-fold": DevelopmentalStage.ONE_POINT_FIVE_FOLD,
            "1.5 fold": DevelopmentalStage.ONE_POINT_FIVE_FOLD,
            "2-fold": DevelopmentalStage.TWO_FOLD,
            "2 fold": DevelopmentalStage.TWO_FOLD,
            "pretzel": DevelopmentalStage.PRETZEL,
            "3-fold": DevelopmentalStage.PRETZEL,
            "3 fold": DevelopmentalStage.PRETZEL,
            "pre-hatching": DevelopmentalStage.PRE_HATCHING,
            "prehatching": DevelopmentalStage.PRE_HATCHING,
            "hatching": DevelopmentalStage.HATCHING,
            "hatched": DevelopmentalStage.HATCHED,
            "dead": DevelopmentalStage.DEAD,
            "unknown": DevelopmentalStage.UNKNOWN,
        }

        return mappings.get(name, DevelopmentalStage.UNKNOWN)

    def get_stage_history(self, embryo_id: str) -> list[StageClassification]:
        """Get stage classification history for an embryo"""
        return self._stage_history.get(embryo_id, [])

    def get_current_stage(self, embryo_id: str) -> StageClassification | None:
        """Get the most recent stage classification"""
        history = self._stage_history.get(embryo_id, [])
        return history[-1] if history else None

    def predict_time_to_hatching(self, embryo_id: str) -> timedelta | None:
        """
        Predict time to hatching based on current stage

        Parameters
        ----------
        embryo_id : str
            Embryo to predict for

        Returns
        -------
        timedelta or None
            Predicted time to hatching, or None if unknown
        """
        current = self.get_current_stage(embryo_id)
        if not current or current.stage == DevelopmentalStage.UNKNOWN:
            return None

        if current.predicted_minutes_to_hatching is not None:
            return timedelta(minutes=current.predicted_minutes_to_hatching)

        return None

    def predict_time_to_stage(
        self,
        embryo_id: str,
        target_stage: DevelopmentalStage,
    ) -> timedelta | None:
        """
        Predict time until embryo reaches target stage

        Parameters
        ----------
        embryo_id : str
            Embryo to predict for
        target_stage : DevelopmentalStage
            Target stage

        Returns
        -------
        timedelta or None
            Predicted time, or None if cannot predict
        """
        current = self.get_current_stage(embryo_id)
        if not current or current.stage == DevelopmentalStage.UNKNOWN:
            return None

        # Get timing for both stages
        current_timing = STAGE_TIMING_20C.get(current.stage)
        target_timing = STAGE_TIMING_20C.get(target_stage)

        if current_timing is None or target_timing is None:
            return None

        # Already past target?
        if current_timing >= target_timing:
            return timedelta(minutes=0)

        # Estimate time
        minutes = target_timing - current_timing
        return timedelta(minutes=minutes)

    def get_progression_summary(self, embryo_id: str) -> dict[str, Any]:
        """
        Get a summary of stage progression for an embryo

        Parameters
        ----------
        embryo_id : str
            Embryo to summarize

        Returns
        -------
        dict
            Progression summary
        """
        history = self._stage_history.get(embryo_id, [])

        if not history:
            return {
                "embryo_id": embryo_id,
                "observations": 0,
                "current_stage": None,
                "stages_observed": [],
                "predicted_hatching": None,
            }

        current = history[-1]
        stages_observed = list(set(h.stage.value for h in history))

        return {
            "embryo_id": embryo_id,
            "observations": len(history),
            "current_stage": current.stage.value,
            "current_confidence": current.confidence,
            "stages_observed": stages_observed,
            "first_observation": history[0].timestamp.isoformat(),
            "last_observation": current.timestamp.isoformat(),
            "predicted_minutes_to_hatching": current.predicted_minutes_to_hatching,
        }

    def get_hatching_prediction(self, embryo_id: str) -> HatchingPrediction | None:
        """
        Get detailed hatching prediction with confidence interval

        Parameters
        ----------
        embryo_id : str
            Embryo to predict for

        Returns
        -------
        HatchingPrediction or None
            Prediction with range, or None if cannot predict
        """
        current = self.get_current_stage(embryo_id)
        if not current or current.stage == DevelopmentalStage.UNKNOWN:
            return None

        if current.stage in (DevelopmentalStage.HATCHED, DevelopmentalStage.DEAD):
            return None

        base_time = TIME_TO_HATCHING.get(current.stage)
        variability = TIMING_VARIABILITY.get(current.stage, 60)

        if base_time is None:
            return None

        # Adjust confidence interval based on classification confidence
        confidence_multiplier = {
            "HIGH": 1.0,
            "MEDIUM": 1.5,
            "LOW": 2.0,
        }.get(current.confidence, 2.0)

        adjusted_variability = int(variability * confidence_multiplier)

        return HatchingPrediction(
            embryo_id=embryo_id,
            current_stage=current.stage,
            predicted_minutes=base_time,
            min_minutes=max(0, base_time - adjusted_variability),
            max_minutes=base_time + adjusted_variability,
            confidence=current.confidence,
        )

    def get_all_predictions(self, embryo_ids: list[str]) -> dict[str, HatchingPrediction]:
        """
        Get predictions for multiple embryos

        Parameters
        ----------
        embryo_ids : list of str
            Embryos to predict for

        Returns
        -------
        dict
            embryo_id -> HatchingPrediction
        """
        predictions = {}
        for embryo_id in embryo_ids:
            pred = self.get_hatching_prediction(embryo_id)
            if pred:
                predictions[embryo_id] = pred
        return predictions

    def estimate_development_rate(self, embryo_id: str) -> float | None:
        """
        Estimate relative development rate compared to standard

        Based on observed stage progression timing.

        Parameters
        ----------
        embryo_id : str
            Embryo to analyze

        Returns
        -------
        float or None
            Rate multiplier (1.0 = normal, >1 = faster, <1 = slower)
        """
        history = self._stage_history.get(embryo_id, [])

        if len(history) < 2:
            return None

        # Need at least two different stages
        stages_seen = [
            (h.stage, h.timestamp) for h in history if h.stage != DevelopmentalStage.UNKNOWN
        ]
        if len(stages_seen) < 2:
            return None

        # Find first and last distinct stages
        first_stage, first_time = stages_seen[0]
        last_stage, last_time = stages_seen[-1]

        if first_stage == last_stage:
            return None

        # Calculate expected vs actual time
        expected_first = STAGE_TIMING_20C.get(first_stage, 0)
        expected_last = STAGE_TIMING_20C.get(last_stage, 0)

        if expected_last <= expected_first:
            return None

        expected_duration = expected_last - expected_first
        actual_duration = (last_time - first_time).total_seconds() / 60

        if actual_duration <= 0:
            return None

        # Rate = expected / actual (faster embryo = higher rate)
        rate = expected_duration / actual_duration
        return rate
