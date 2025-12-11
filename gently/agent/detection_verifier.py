"""
Detection Verifier - Challenger Agent for Detection Verification

Provides a verification layer before AUTO mode detection actions execute.
Uses multiple strategies to validate detections and prevent false positives.

Strategies:
1. Adversarial - Ask Claude to find counter-evidence
2. Independent - Fresh analysis without knowing previous result
3. Temporal - Compare to previous timepoints for actual change
"""

import asyncio
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import anthropic

from .detector import Detector, DetectionResult, ConfidenceLevel
from .state import EmbryoState

logger = logging.getLogger(__name__)


@dataclass
class AdversarialResult:
    """Result of adversarial verification strategy"""
    found_counter_evidence: bool
    concerns: List[str]
    confidence_in_original: Optional[ConfidenceLevel]
    raw_response: str


@dataclass
class IndependentResult:
    """Result of independent verification strategy"""
    detected: bool
    confidence: Optional[ConfidenceLevel]
    key_evidence: str
    raw_response: str


@dataclass
class TemporalResult:
    """Result of temporal comparison strategy"""
    change_detected: bool
    description: str
    confidence: Optional[ConfidenceLevel]
    raw_response: str


@dataclass
class VerificationResult:
    """Combined result of all verification strategies"""
    original_detected: bool
    original_confidence: Optional[ConfidenceLevel]

    # Strategy results
    adversarial: AdversarialResult
    independent: IndependentResult
    temporal: TemporalResult

    # Consensus
    consensus: bool
    consensus_reasoning: str

    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    verification_duration_seconds: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary"""
        return {
            'original_detected': self.original_detected,
            'original_confidence': self.original_confidence.value if self.original_confidence else None,
            'adversarial': {
                'found_counter_evidence': self.adversarial.found_counter_evidence,
                'concerns': self.adversarial.concerns,
                'confidence_in_original': self.adversarial.confidence_in_original.value if self.adversarial.confidence_in_original else None,
            },
            'independent': {
                'detected': self.independent.detected,
                'confidence': self.independent.confidence.value if self.independent.confidence else None,
                'key_evidence': self.independent.key_evidence,
            },
            'temporal': {
                'change_detected': self.temporal.change_detected,
                'description': self.temporal.description,
                'confidence': self.temporal.confidence.value if self.temporal.confidence else None,
            },
            'consensus': self.consensus,
            'consensus_reasoning': self.consensus_reasoning,
            'timestamp': self.timestamp.isoformat(),
            'verification_duration_seconds': self.verification_duration_seconds,
        }


class DetectionVerifier:
    """
    Verifies detections before critical actions using multiple strategies.

    Acts as a "challenger agent" that questions detections to prevent
    false positives from triggering serious actions like stopping timelapse.
    """

    def __init__(
        self,
        claude_client: anthropic.Anthropic,
        model: str = "claude-haiku-4-5-20251001",
    ):
        """
        Parameters
        ----------
        claude_client : anthropic.Anthropic
            Claude API client
        model : str
            Model to use for verification (Haiku for speed/cost)
        """
        self.claude = claude_client
        self.model = model

    async def verify(
        self,
        detector: Detector,
        embryo_state: EmbryoState,
        original_result: DetectionResult,
        timepoint: int,
    ) -> VerificationResult:
        """
        Run all verification strategies on a detection.

        Parameters
        ----------
        detector : Detector
            The detector that fired
        embryo_state : EmbryoState
            Current embryo state with recent images
        original_result : DetectionResult
            The original detection result to verify
        timepoint : int
            Current timepoint number

        Returns
        -------
        VerificationResult
            Combined verification result with consensus
        """
        start_time = datetime.now()

        # Run all strategies in parallel for speed
        adversarial_task = self._run_adversarial(
            detector, embryo_state, original_result, timepoint
        )
        independent_task = self._run_independent(
            detector, embryo_state, timepoint
        )
        temporal_task = self._run_temporal_check(
            detector, embryo_state, timepoint
        )

        adversarial, independent, temporal = await asyncio.gather(
            adversarial_task, independent_task, temporal_task
        )

        # Determine consensus
        consensus, reasoning = self._evaluate_consensus(
            original_result, adversarial, independent, temporal
        )

        duration = (datetime.now() - start_time).total_seconds()

        result = VerificationResult(
            original_detected=original_result.detected,
            original_confidence=original_result.confidence,
            adversarial=adversarial,
            independent=independent,
            temporal=temporal,
            consensus=consensus,
            consensus_reasoning=reasoning,
            verification_duration_seconds=duration,
        )

        logger.info(
            f"Verification complete for {detector.name}: "
            f"consensus={consensus}, duration={duration:.2f}s"
        )

        return result

    async def _run_adversarial(
        self,
        detector: Detector,
        embryo_state: EmbryoState,
        original_result: DetectionResult,
        timepoint: int,
    ) -> AdversarialResult:
        """
        Run adversarial verification - look for counter-evidence.

        Asks Claude to act as a critical reviewer and find reasons
        why the detection might be WRONG.
        """
        try:
            # Get recent images
            images = self._get_image_content(embryo_state, num_images=3)
            if not images:
                return AdversarialResult(
                    found_counter_evidence=False,
                    concerns=["No images available for verification"],
                    confidence_in_original=None,
                    raw_response="",
                )

            prompt = f"""You are reviewing a detection result for a C. elegans embryo.

The system detected: {detector.name}
Original confidence: {original_result.confidence.value if original_result.confidence else 'unknown'}
Original reasoning: {original_result.reasoning or 'not provided'}

NOW ACT AS A CRITICAL REVIEWER. Your job is to find reasons why this detection might be INCORRECT:
- Could this be a false positive?
- Are there artifacts, noise, or imaging issues that could be misleading?
- Is the evidence actually conclusive, or could it be interpreted differently?
- Are there alternative explanations for what is observed?

Analyze the image(s) carefully and respond in EXACTLY this format:
COUNTER_EVIDENCE_FOUND: [YES/NO]
CONCERNS: [list specific doubts or alternative explanations, separated by semicolons]
CONFIDENCE_IN_ORIGINAL: [HIGH/MEDIUM/LOW]
"""

            content = [{"type": "text", "text": prompt}] + images

            response = await asyncio.to_thread(
                self.claude.messages.create,
                model=self.model,
                max_tokens=500,
                messages=[{"role": "user", "content": content}]
            )

            response_text = response.content[0].text
            return self._parse_adversarial_response(response_text)

        except Exception as e:
            logger.error(f"Adversarial verification failed: {e}")
            return AdversarialResult(
                found_counter_evidence=False,
                concerns=[f"Verification error: {str(e)}"],
                confidence_in_original=None,
                raw_response="",
            )

    async def _run_independent(
        self,
        detector: Detector,
        embryo_state: EmbryoState,
        timepoint: int,
    ) -> IndependentResult:
        """
        Run independent verification - fresh analysis without bias.

        Asks Claude to analyze the image without knowing the previous
        detection result, providing an unbiased second opinion.
        """
        try:
            # Get only the current image for independent analysis
            images = self._get_image_content(embryo_state, num_images=1)
            if not images:
                return IndependentResult(
                    detected=False,
                    confidence=None,
                    key_evidence="No images available",
                    raw_response="",
                )

            # Use a neutral prompt that doesn't reveal the previous detection
            prompt = f"""Analyze this C. elegans embryo image at timepoint {timepoint}.

Question: Has '{detector.name}' occurred in this embryo?

{detector.description}

Provide an independent assessment based SOLELY on what you observe in this image.
Do not assume any prior state - analyze only what is visible now.

Respond in EXACTLY this format:
DETECTED: [YES/NO]
CONFIDENCE: [HIGH/MEDIUM/LOW]
KEY_EVIDENCE: [what specifically do you observe that supports your conclusion?]
"""

            content = [{"type": "text", "text": prompt}] + images

            response = await asyncio.to_thread(
                self.claude.messages.create,
                model=self.model,
                max_tokens=400,
                messages=[{"role": "user", "content": content}]
            )

            response_text = response.content[0].text
            return self._parse_independent_response(response_text)

        except Exception as e:
            logger.error(f"Independent verification failed: {e}")
            return IndependentResult(
                detected=False,
                confidence=None,
                key_evidence=f"Verification error: {str(e)}",
                raw_response="",
            )

    async def _run_temporal_check(
        self,
        detector: Detector,
        embryo_state: EmbryoState,
        timepoint: int,
    ) -> TemporalResult:
        """
        Run temporal comparison - check for actual change.

        Compares current timepoint to previous timepoints to verify
        that an actual transition/change occurred.
        """
        try:
            # Need at least 2 images for comparison
            if len(embryo_state.recent_images) < 2:
                return TemporalResult(
                    change_detected=True,  # Can't disprove without history
                    description="Insufficient temporal history for comparison",
                    confidence=ConfidenceLevel.LOW,
                    raw_response="",
                )

            # Get current and previous images
            current_images = self._get_image_content(embryo_state, num_images=1)

            # Get 2 previous images
            prev_images = []
            for img in embryo_state.recent_images[-3:-1]:
                if img.max_projection_b64:
                    prev_images.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": img.max_projection_b64,
                        }
                    })

            if not prev_images:
                return TemporalResult(
                    change_detected=True,
                    description="No previous images available",
                    confidence=ConfidenceLevel.LOW,
                    raw_response="",
                )

            prompt = f"""Compare these sequential timepoints of a C. elegans embryo.

PREVIOUS TIMEPOINTS (shown first):
These are from t={timepoint-2} to t={timepoint-1}

CURRENT TIMEPOINT (shown last):
This is t={timepoint}

Question: Is there a clear CHANGE consistent with '{detector.name}'?

For {detector.name}, look for:
- Actual transition or change between frames
- Not just a static state that could have existed before
- Clear evidence of progression or event occurrence

Respond in EXACTLY this format:
CHANGE_DETECTED: [YES/NO]
DESCRIPTION: [what specific change do you see between the previous and current frames?]
CONFIDENCE: [HIGH/MEDIUM/LOW]
"""

            # Combine: previous images first, then prompt, then current
            content = prev_images + [{"type": "text", "text": prompt}] + current_images

            response = await asyncio.to_thread(
                self.claude.messages.create,
                model=self.model,
                max_tokens=400,
                messages=[{"role": "user", "content": content}]
            )

            response_text = response.content[0].text
            return self._parse_temporal_response(response_text)

        except Exception as e:
            logger.error(f"Temporal verification failed: {e}")
            return TemporalResult(
                change_detected=True,  # Don't block on error
                description=f"Verification error: {str(e)}",
                confidence=None,
                raw_response="",
            )

    def _get_image_content(
        self,
        embryo_state: EmbryoState,
        num_images: int = 1
    ) -> List[Dict]:
        """Get image content blocks for Claude API"""
        images = []
        recent = embryo_state.recent_images[-num_images:] if embryo_state.recent_images else []

        for img in recent:
            if img.max_projection_b64:
                images.append({
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": img.max_projection_b64,
                    }
                })

        return images

    def _parse_adversarial_response(self, response: str) -> AdversarialResult:
        """Parse adversarial strategy response"""
        found_counter = False
        concerns = []
        confidence = None

        for line in response.split('\n'):
            line = line.strip()
            if line.startswith('COUNTER_EVIDENCE_FOUND:'):
                value = line.split(':', 1)[1].strip().upper()
                found_counter = value == 'YES'
            elif line.startswith('CONCERNS:'):
                concerns_str = line.split(':', 1)[1].strip()
                concerns = [c.strip() for c in concerns_str.split(';') if c.strip()]
            elif line.startswith('CONFIDENCE_IN_ORIGINAL:'):
                value = line.split(':', 1)[1].strip().upper()
                try:
                    confidence = ConfidenceLevel(value)
                except ValueError:
                    pass

        return AdversarialResult(
            found_counter_evidence=found_counter,
            concerns=concerns,
            confidence_in_original=confidence,
            raw_response=response,
        )

    def _parse_independent_response(self, response: str) -> IndependentResult:
        """Parse independent strategy response"""
        detected = False
        confidence = None
        evidence = ""

        for line in response.split('\n'):
            line = line.strip()
            if line.startswith('DETECTED:'):
                value = line.split(':', 1)[1].strip().upper()
                detected = value == 'YES'
            elif line.startswith('CONFIDENCE:'):
                value = line.split(':', 1)[1].strip().upper()
                try:
                    confidence = ConfidenceLevel(value)
                except ValueError:
                    pass
            elif line.startswith('KEY_EVIDENCE:'):
                evidence = line.split(':', 1)[1].strip()

        return IndependentResult(
            detected=detected,
            confidence=confidence,
            key_evidence=evidence,
            raw_response=response,
        )

    def _parse_temporal_response(self, response: str) -> TemporalResult:
        """Parse temporal strategy response"""
        change_detected = False
        description = ""
        confidence = None

        for line in response.split('\n'):
            line = line.strip()
            if line.startswith('CHANGE_DETECTED:'):
                value = line.split(':', 1)[1].strip().upper()
                change_detected = value == 'YES'
            elif line.startswith('DESCRIPTION:'):
                description = line.split(':', 1)[1].strip()
            elif line.startswith('CONFIDENCE:'):
                value = line.split(':', 1)[1].strip().upper()
                try:
                    confidence = ConfidenceLevel(value)
                except ValueError:
                    pass

        return TemporalResult(
            change_detected=change_detected,
            description=description,
            confidence=confidence,
            raw_response=response,
        )

    def _evaluate_consensus(
        self,
        original: DetectionResult,
        adversarial: AdversarialResult,
        independent: IndependentResult,
        temporal: TemporalResult,
    ) -> tuple[bool, str]:
        """
        Evaluate consensus across all verification strategies.

        Returns (consensus_reached, reasoning)
        """
        # Count agreements
        agreements = 0
        disagreements = []

        # Check adversarial: should NOT find strong counter-evidence
        if not adversarial.found_counter_evidence:
            agreements += 1
        else:
            disagreements.append(f"Adversarial found counter-evidence: {', '.join(adversarial.concerns[:2])}")

        # Check independent: should also detect
        if independent.detected:
            agreements += 1
        else:
            disagreements.append(f"Independent analysis did not detect: {independent.key_evidence}")

        # Check temporal: should see change
        if temporal.change_detected:
            agreements += 1
        else:
            disagreements.append(f"No temporal change detected: {temporal.description}")

        # Consensus requires all 3 strategies to agree
        consensus = agreements == 3

        if consensus:
            reasoning = (
                f"All verification strategies agree: "
                f"no counter-evidence found, independent analysis confirms detection, "
                f"temporal change observed."
            )
        else:
            reasoning = (
                f"Verification disagreement ({agreements}/3 agree): "
                f"{'; '.join(disagreements)}"
            )

        return consensus, reasoning
