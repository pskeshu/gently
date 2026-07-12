"""
Detection queue - executes all enabled detectors on volumes
"""

import asyncio
from collections.abc import Callable
from datetime import datetime
from typing import Any, cast

import anthropic
from anthropic.types import Message, TextBlock

from gently.settings import settings

from ..state import EmbryoState
from .detector import ConfidenceLevel, DetectionResult, Detector
from .registry import DetectorRegistry


class DetectionQueue:
    """
    Execution engine for running detectors on volumes

    Manages detector execution, result storage, and action triggers.
    """

    def __init__(
        self,
        registry: DetectorRegistry,
        claude_client: anthropic.Anthropic,
        model: str = settings.models.perception,
        on_detection_callback: Callable | None = None,
        on_evaluation_callback: Callable | None = None,
    ):
        """
        Parameters
        ----------
        registry : DetectorRegistry
            Detector registry
        claude_client : anthropic.Anthropic
            Claude API client
        model : str
            Claude model to use
        on_detection_callback : callable, optional
            Called when detector fires (detected=True): callback(detector, embryo_id, result)
        on_evaluation_callback : callable, optional
            Called for every evaluation: callback(detector, embryo_id, result, embryo_state)
        """
        self.registry = registry
        self.claude = claude_client
        self.model = model
        self.on_detection_callback = on_detection_callback
        self.on_evaluation_callback = on_evaluation_callback

    async def run_detectors(
        self, embryo_state: EmbryoState, timepoint: int
    ) -> list[DetectionResult]:
        """
        Run all applicable detectors for an embryo/timepoint

        Parameters
        ----------
        embryo_state : EmbryoState
            Embryo to analyze
        timepoint : int
            Current timepoint

        Returns
        -------
        list of DetectionResult
            Results from all detectors that ran
        """
        embryo_id = embryo_state.id
        results = []

        # Get enabled detectors
        detectors = self.registry.list_enabled()

        for detector in detectors:
            # Check if should run
            if not detector.should_run(embryo_id, timepoint):
                continue

            # Run detector
            result = await self._run_single_detector(detector, embryo_state, timepoint)

            results.append(result)

            # Store result
            embryo_state.add_detection_result(detector.name, result.to_dict())

            # Mark detector as run
            detector.mark_run(embryo_id, timepoint)

            # Always trigger evaluation callback (for all results, not just positive)
            if self.on_evaluation_callback:
                await self.on_evaluation_callback(detector, embryo_id, result, embryo_state)

            # Check if detected and meets confidence threshold
            if result.detected and self._meets_confidence_threshold(result, detector):
                detector.mark_detected(embryo_id)

                # Trigger detection callback (for positive detections only)
                if self.on_detection_callback:
                    await self.on_detection_callback(detector, embryo_id, result)

        return results

    async def _run_single_detector(
        self, detector: Detector, embryo_state: EmbryoState, timepoint: int
    ) -> DetectionResult:
        """
        Run a single detector

        Parameters
        ----------
        detector : Detector
            Detector to run
        embryo_state : EmbryoState
            Embryo to analyze
        timepoint : int
            Current timepoint

        Returns
        -------
        DetectionResult
            Detection result
        """
        embryo_id = embryo_state.id
        start_time = datetime.now()

        try:
            # Get recent images
            num_images = detector.temporal_context_size if detector.use_temporal_context else 1
            recent_images = (
                embryo_state.recent_images[-num_images:] if embryo_state.recent_images else []
            )

            if not recent_images:
                # No images available
                return DetectionResult(
                    detector_name=detector.name,
                    embryo_id=embryo_id,
                    timepoint=timepoint,
                    timestamp=datetime.now(),
                    detected=False,
                    error=True,
                    error_message="No images available",
                )

            # Build image data for detector
            image_data = [
                {
                    "timepoint": img.timepoint,
                    "b64_image": img.max_projection_b64,
                    "size": img.size_kb,
                }
                for img in recent_images
            ]

            # Build content for Claude
            content = detector.build_detection_content(image_data, embryo_id, timepoint)

            # Call Claude Vision API
            response = await asyncio.to_thread(
                cast("Callable[..., Message]", self.claude.messages.create),
                model=self.model,
                max_tokens=1024,
                messages=[{"role": "user", "content": content}],
            )

            response_text = cast("TextBlock", response.content[0]).text
            api_duration = (datetime.now() - start_time).total_seconds()

            # Parse response
            parsed = detector.parse_detection_response(response_text)

            return DetectionResult(
                detector_name=detector.name,
                embryo_id=embryo_id,
                timepoint=timepoint,
                timestamp=datetime.now(),
                detected=parsed["detected"],
                confidence=parsed["confidence"],
                reasoning=parsed["reasoning"],
                error=False,
                api_duration=api_duration,
                num_images=len(image_data),
                full_response=response_text,
            )

        except Exception as e:
            # Handle errors gracefully
            api_duration = (datetime.now() - start_time).total_seconds()

            return DetectionResult(
                detector_name=detector.name,
                embryo_id=embryo_id,
                timepoint=timepoint,
                timestamp=datetime.now(),
                detected=False,
                error=True,
                error_message=str(e),
                api_duration=api_duration,
            )

    def _meets_confidence_threshold(self, result: DetectionResult, detector: Detector) -> bool:
        """
        Check if detection result meets confidence threshold

        Parameters
        ----------
        result : DetectionResult
            Detection result
        detector : Detector
            Detector configuration

        Returns
        -------
        bool
            True if meets threshold
        """
        if not result.confidence:
            return False

        # Map confidence levels to numeric values
        confidence_map = {
            ConfidenceLevel.LOW: 1,
            ConfidenceLevel.MEDIUM: 2,
            ConfidenceLevel.HIGH: 3,
        }

        result_value = confidence_map.get(result.confidence, 0)
        threshold_value = confidence_map.get(detector.confidence_threshold, 0)

        return result_value >= threshold_value

    async def test_detector(
        self,
        detector_name: str,
        embryo_state: EmbryoState,
        timepoint: int | None = None,
    ) -> DetectionResult | None:
        """
        Test a detector on a specific embryo/timepoint

        Useful for testing detector configuration without running full pipeline.

        Parameters
        ----------
        detector_name : str
            Detector to test
        embryo_state : EmbryoState
            Embryo to test on
        timepoint : int, optional
            Specific timepoint (default: latest)

        Returns
        -------
        DetectionResult or None
            Result if detector exists, None otherwise
        """
        detector = self.registry.get(detector_name)
        if not detector:
            return None

        # Use latest timepoint if not specified
        if timepoint is None and embryo_state.recent_images:
            timepoint = embryo_state.recent_images[-1].timepoint
        elif timepoint is None:
            return None

        # Run detector (ignore conditions for testing)
        result = await self._run_single_detector(detector, embryo_state, timepoint)

        return result

    def get_detection_summary(self, embryo_states: dict[str, EmbryoState]) -> dict:
        """
        Get summary of all detections across all embryos

        Parameters
        ----------
        embryo_states : dict
            embryo_id -> EmbryoState

        Returns
        -------
        dict
            Summary of detections
        """
        summary: dict[str, Any] = {"detectors": {}, "embryos": {}}

        # Per-detector summary
        for detector in self.registry.list_all():
            detector_summary: dict[str, Any] = {
                "name": detector.name,
                "description": detector.description,
                "enabled": detector.enabled,
                "total_runs": detector.run_count,
                "total_detections": detector.detection_count,
                "embryos_detected": [],
            }

            for embryo_id, embryo_state in embryo_states.items():
                if embryo_state.was_detected(detector.name):
                    latest = embryo_state.get_latest_detection(detector.name)
                    if latest is not None:
                        detector_summary["embryos_detected"].append(
                            {
                                "embryo_id": embryo_id,
                                "timepoint": latest.get("timepoint"),
                                "confidence": latest.get("confidence"),
                            }
                        )

            summary["detectors"][detector.name] = detector_summary

        # Per-embryo summary
        for embryo_id, embryo_state in embryo_states.items():
            embryo_summary: dict[str, Any] = {"embryo_id": embryo_id, "detections": {}}

            for detector_name in embryo_state.detection_results.keys():
                latest = embryo_state.get_latest_detection(detector_name)
                embryo_summary["detections"][detector_name] = {
                    "detected": latest.get("detected", False) if latest else False,
                    "timepoint": latest.get("timepoint") if latest else None,
                    "confidence": latest.get("confidence") if latest else None,
                }

            summary["embryos"][embryo_id] = embryo_summary

        return summary
