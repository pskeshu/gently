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
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import anthropic

from gently.core import EventType, get_event_bus
from gently.settings import settings

from ..state import EmbryoState
from .detector import ConfidenceLevel, DetectionResult, Detector

logger = logging.getLogger(__name__)


@dataclass
class AdversarialResult:
    """Result of adversarial verification strategy"""

    found_counter_evidence: bool
    concerns: list[str]
    confidence_in_original: ConfidenceLevel | None
    raw_response: str


@dataclass
class IndependentResult:
    """Result of independent verification strategy"""

    detected: bool
    confidence: ConfidenceLevel | None
    key_evidence: str
    raw_response: str


@dataclass
class TemporalResult:
    """Result of temporal comparison strategy"""

    change_detected: bool
    description: str
    confidence: ConfidenceLevel | None
    raw_response: str


@dataclass
class EnsembleResult:
    """Result of ensemble voting strategy"""

    votes_yes: int
    votes_no: int
    total_votes: int
    agreement_ratio: float  # votes_yes / total_votes if detected, votes_no / total_votes if not
    consensus_detected: bool  # True if >70% agree on YES
    raw_responses: list[str] = field(default_factory=list)


@dataclass
class HardwareContextResult:
    """Result of hardware context analysis strategy"""

    suspicious: bool  # True if hardware errors could have caused false positive
    concerns: list[str]  # Specific concerns identified
    reasoning: str
    raw_response: str


@dataclass
class VerificationResult:
    """Combined result of all verification strategies"""

    original_detected: bool
    original_confidence: ConfidenceLevel | None

    # Strategy results
    adversarial: AdversarialResult
    independent: IndependentResult
    temporal: TemporalResult
    ensemble: EnsembleResult | None = None  # Only for hatching detection
    hardware_context: HardwareContextResult | None = None  # Only when errors present

    # Consensus
    consensus: bool = False
    consensus_reasoning: str = ""

    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    verification_duration_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary"""
        result: dict[str, Any] = {
            "original_detected": self.original_detected,
            "original_confidence": self.original_confidence.value
            if self.original_confidence
            else None,
            "adversarial": {
                "found_counter_evidence": self.adversarial.found_counter_evidence,
                "concerns": self.adversarial.concerns,
                "confidence_in_original": self.adversarial.confidence_in_original.value
                if self.adversarial.confidence_in_original
                else None,
            },
            "independent": {
                "detected": self.independent.detected,
                "confidence": self.independent.confidence.value
                if self.independent.confidence
                else None,
                "key_evidence": self.independent.key_evidence,
            },
            "temporal": {
                "change_detected": self.temporal.change_detected,
                "description": self.temporal.description,
                "confidence": self.temporal.confidence.value if self.temporal.confidence else None,
            },
            "consensus": self.consensus,
            "consensus_reasoning": self.consensus_reasoning,
            "timestamp": self.timestamp.isoformat(),
            "verification_duration_seconds": self.verification_duration_seconds,
        }
        if self.ensemble:
            result["ensemble"] = {
                "votes_yes": self.ensemble.votes_yes,
                "votes_no": self.ensemble.votes_no,
                "total_votes": self.ensemble.total_votes,
                "agreement_ratio": self.ensemble.agreement_ratio,
                "consensus_detected": self.ensemble.consensus_detected,
            }
        if self.hardware_context:
            result["hardware_context"] = {
                "suspicious": self.hardware_context.suspicious,
                "concerns": self.hardware_context.concerns,
                "reasoning": self.hardware_context.reasoning,
            }
        return result


class DetectionVerifier:
    """
    Verifies detections before critical actions using multiple strategies.

    Acts as a "challenger agent" that questions detections to prevent
    false positives from triggering serious actions like stopping timelapse.
    """

    def __init__(
        self,
        claude_client: anthropic.Anthropic,
        model: str = settings.models.fast,
        ensemble_model: str = settings.models.fast,
        ensemble_size: int = 50,
        ensemble_threshold: float = 0.70,
        event_bus=None,
    ):
        """
        Parameters
        ----------
        claude_client : anthropic.Anthropic
            Claude API client
        model : str
            Model to use for verification strategies (Opus for accuracy)
        ensemble_model : str
            Model to use for ensemble voting (Haiku for cost efficiency)
        ensemble_size : int
            Number of parallel calls for ensemble voting (default: 50)
        ensemble_threshold : float
            Agreement ratio required for ensemble consensus (default: 0.70 = 70%)
        event_bus : EventBus, optional
            Event bus for emitting verification events to viz server
        """
        self.claude = claude_client
        self.model = model
        self.ensemble_model = ensemble_model
        self.ensemble_size = ensemble_size
        self.ensemble_threshold = ensemble_threshold
        self._event_bus = event_bus or get_event_bus()

    def _emit_event(self, event_type: EventType, data: dict):
        """Emit event to viz server"""
        if self._event_bus:
            self._event_bus.publish(event_type, data)

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
        adversarial_task = self._run_adversarial(detector, embryo_state, original_result, timepoint)
        independent_task = self._run_independent(detector, embryo_state, timepoint)
        temporal_task = self._run_temporal_check(detector, embryo_state, timepoint)

        # For hatching detection, also run ensemble voting
        ensemble_result = None
        if detector.name == "hatching":
            ensemble_task = self._run_ensemble_hatching(embryo_state)
            adversarial, independent, temporal, ensemble_result = await asyncio.gather(
                adversarial_task, independent_task, temporal_task, ensemble_task
            )
        else:
            adversarial, independent, temporal = await asyncio.gather(
                adversarial_task, independent_task, temporal_task
            )

        # Determine consensus
        consensus, reasoning = self._evaluate_consensus(
            original_result, adversarial, independent, temporal, ensemble_result
        )

        duration = (datetime.now() - start_time).total_seconds()

        result = VerificationResult(
            original_detected=original_result.detected,
            original_confidence=original_result.confidence,
            adversarial=adversarial,
            independent=independent,
            temporal=temporal,
            ensemble=ensemble_result,
            consensus=consensus,
            consensus_reasoning=reasoning,
            verification_duration_seconds=duration,
        )

        logger.info(
            f"Verification complete for {detector.name}: "
            f"consensus={consensus}, duration={duration:.2f}s"
            + (
                f", ensemble={ensemble_result.votes_yes}/{ensemble_result.total_votes} YES"
                if ensemble_result
                else ""
            )
        )

        return result

    async def verify_with_context(
        self,
        detector: Detector,
        embryo_state: EmbryoState,
        original_result: DetectionResult,
        timepoint: int,
        global_error_context: str = "",
    ) -> VerificationResult:
        """
        Run all verification strategies including hardware context analysis.

        This is the enhanced verification that includes cross-embryo error correlation.

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
        global_error_context : str
            Compiled error log from GlobalErrorLog.compile_for_verification()

        Returns
        -------
        VerificationResult
            Combined verification result with consensus
        """
        start_time = datetime.now()

        # Run all strategies in parallel for speed
        adversarial_task = self._run_adversarial(detector, embryo_state, original_result, timepoint)
        independent_task = self._run_independent(detector, embryo_state, timepoint)
        temporal_task = self._run_temporal_check(detector, embryo_state, timepoint)

        # For hatching detection, also run ensemble voting
        ensemble_result = None
        hardware_result = None

        if detector.name == "hatching":
            ensemble_task = self._run_ensemble_hatching(embryo_state)

            # Run hardware context analysis if there are errors
            if global_error_context and "No hardware errors" not in global_error_context:
                hardware_task = self._run_hardware_context_analysis(
                    global_error_context, embryo_state.id
                )
                (
                    adversarial,
                    independent,
                    temporal,
                    ensemble_result,
                    hardware_result,
                ) = await asyncio.gather(
                    adversarial_task,
                    independent_task,
                    temporal_task,
                    ensemble_task,
                    hardware_task,
                )
            else:
                (
                    adversarial,
                    independent,
                    temporal,
                    ensemble_result,
                ) = await asyncio.gather(
                    adversarial_task, independent_task, temporal_task, ensemble_task
                )
        else:
            adversarial, independent, temporal = await asyncio.gather(
                adversarial_task, independent_task, temporal_task
            )

        # Emit strategy result events for viz server
        embryo_id = embryo_state.id
        strategies_complete = 0
        total_strategies = 3 + (1 if ensemble_result else 0) + (1 if hardware_result else 0)

        # Adversarial result
        strategies_complete += 1
        self._emit_event(
            EventType.VERIFICATION_STRATEGY,
            {
                "embryo_id": embryo_id,
                "detector_name": detector.name,
                "strategy": "adversarial",
                "passed": not adversarial.found_counter_evidence,
                "summary": (
                    "Counter-evidence: "
                    + (
                        "YES - " + ", ".join(adversarial.concerns)
                        if adversarial.found_counter_evidence
                        else "None found"
                    )
                ),
                "confidence": adversarial.confidence_in_original.value
                if adversarial.confidence_in_original
                else None,
            },
        )
        self._emit_event(
            EventType.VERIFICATION_PROGRESS,
            {
                "embryo_id": embryo_id,
                "strategies_complete": strategies_complete,
                "total_strategies": total_strategies,
            },
        )

        # Independent result
        strategies_complete += 1
        self._emit_event(
            EventType.VERIFICATION_STRATEGY,
            {
                "embryo_id": embryo_id,
                "detector_name": detector.name,
                "strategy": "independent",
                "passed": independent.detected,
                "summary": (
                    f"Independent detection: {'YES' if independent.detected else 'NO'}"
                    f" - {independent.key_evidence}"
                ),
                "confidence": independent.confidence.value if independent.confidence else None,
            },
        )
        self._emit_event(
            EventType.VERIFICATION_PROGRESS,
            {
                "embryo_id": embryo_id,
                "strategies_complete": strategies_complete,
                "total_strategies": total_strategies,
            },
        )

        # Temporal result
        strategies_complete += 1
        self._emit_event(
            EventType.VERIFICATION_STRATEGY,
            {
                "embryo_id": embryo_id,
                "detector_name": detector.name,
                "strategy": "temporal",
                "passed": temporal.change_detected,
                "summary": (
                    f"Change detected: {'YES' if temporal.change_detected else 'NO'}"
                    f" - {temporal.description}"
                ),
                "confidence": temporal.confidence.value if temporal.confidence else None,
            },
        )
        self._emit_event(
            EventType.VERIFICATION_PROGRESS,
            {
                "embryo_id": embryo_id,
                "strategies_complete": strategies_complete,
                "total_strategies": total_strategies,
            },
        )

        # Ensemble result (if applicable)
        if ensemble_result:
            strategies_complete += 1
            self._emit_event(
                EventType.VERIFICATION_STRATEGY,
                {
                    "embryo_id": embryo_id,
                    "detector_name": detector.name,
                    "strategy": "ensemble",
                    "passed": ensemble_result.consensus_detected,
                    "summary": (
                        f"Ensemble vote: {ensemble_result.votes_yes}/{ensemble_result.total_votes}"
                        f" YES ({ensemble_result.agreement_ratio * 100:.0f}%)"
                    ),
                    "votes_yes": ensemble_result.votes_yes,
                    "votes_no": ensemble_result.votes_no,
                    "total_votes": ensemble_result.total_votes,
                },
            )
            self._emit_event(
                EventType.VERIFICATION_PROGRESS,
                {
                    "embryo_id": embryo_id,
                    "strategies_complete": strategies_complete,
                    "total_strategies": total_strategies,
                },
            )

        # Hardware context result (if applicable)
        if hardware_result:
            strategies_complete += 1
            self._emit_event(
                EventType.VERIFICATION_STRATEGY,
                {
                    "embryo_id": embryo_id,
                    "detector_name": detector.name,
                    "strategy": "hardware_context",
                    "passed": not hardware_result.suspicious,
                    "summary": (
                        "Hardware errors suspicious: "
                        + (
                            "YES - " + ", ".join(hardware_result.concerns)
                            if hardware_result.suspicious
                            else "No"
                        )
                    ),
                    "reasoning": hardware_result.reasoning,
                },
            )
            self._emit_event(
                EventType.VERIFICATION_PROGRESS,
                {
                    "embryo_id": embryo_id,
                    "strategies_complete": strategies_complete,
                    "total_strategies": total_strategies,
                },
            )

        # Determine consensus (with hardware context)
        consensus, reasoning = self._evaluate_consensus_with_hardware(
            original_result,
            adversarial,
            independent,
            temporal,
            ensemble_result,
            hardware_result,
        )

        duration = (datetime.now() - start_time).total_seconds()

        result = VerificationResult(
            original_detected=original_result.detected,
            original_confidence=original_result.confidence,
            adversarial=adversarial,
            independent=independent,
            temporal=temporal,
            ensemble=ensemble_result,
            hardware_context=hardware_result,
            consensus=consensus,
            consensus_reasoning=reasoning,
            verification_duration_seconds=duration,
        )

        logger.info(
            f"Verification (with context) complete for {detector.name}: "
            f"consensus={consensus}, duration={duration:.2f}s"
            + (
                f", ensemble={ensemble_result.votes_yes}/{ensemble_result.total_votes} YES"
                if ensemble_result
                else ""
            )
            + (f", hardware_suspicious={hardware_result.suspicious}" if hardware_result else "")
        )

        # Emit VERIFICATION_COMPLETED event with full summary
        self._emit_event(
            EventType.VERIFICATION_COMPLETED,
            {
                "embryo_id": embryo_id,
                "detector_name": detector.name,
                "consensus": consensus,
                "reasoning": reasoning,
                "duration_seconds": duration,
                "strategies": {
                    "adversarial": not adversarial.found_counter_evidence,
                    "independent": independent.detected,
                    "temporal": temporal.change_detected,
                    "ensemble": ensemble_result.consensus_detected if ensemble_result else None,
                    "hardware_context": (not hardware_result.suspicious)
                    if hardware_result
                    else None,
                },
                "ensemble_votes": f"{ensemble_result.votes_yes}/{ensemble_result.total_votes}"
                if ensemble_result
                else None,
            },
        )

        return result

    async def _run_hardware_context_analysis(
        self,
        global_error_context: str,
        embryo_id: str,
    ) -> HardwareContextResult:
        """
        Analyze if hardware errors could have caused a false positive detection.

        Uses Haiku to analyze the global error log and determine if any errors
        could have affected the detection for this embryo.

        Parameters
        ----------
        global_error_context : str
            Compiled error log from GlobalErrorLog
        embryo_id : str
            The embryo being verified

        Returns
        -------
        HardwareContextResult
            Analysis result
        """
        try:
            prompt = f"""You are analyzing hardware error context for a microscopy detection
verification.

GLOBAL ERROR LOG:
{global_error_context}

DETECTION: Hatching detected for {embryo_id}

QUESTION: Could any of these hardware errors have caused a FALSE POSITIVE detection?

Consider these correlations:
- Stage positioning errors could cause wrong embryo to be imaged
- Acquisition timeouts could cause partial/blank images (blank images look like empty FOV = hatched)
- Camera errors could produce corrupted data
- Errors on OTHER embryos in the same round could indicate systemic issues
  (stage drift, hardware instability)
- Multiple errors in quick succession suggests hardware problems

If ANY errors occurred that could have affected the image quality or positioning for
{embryo_id}, report as SUSPICIOUS.

Respond in EXACTLY this format:
SUSPICIOUS: [YES/NO]
CONCERNS: [list specific concerns, separated by semicolons]
REASONING: [brief explanation of your analysis]
"""

            response = await asyncio.to_thread(
                self.claude.messages.create,
                model=self.ensemble_model,  # Use Haiku for speed
                max_tokens=300,
                messages=[{"role": "user", "content": prompt}],
            )

            response_text = response.content[0].text
            return self._parse_hardware_context_response(response_text)

        except Exception as e:
            logger.error(f"Hardware context analysis failed: {e}")
            return HardwareContextResult(
                suspicious=True,  # Be conservative on error
                concerns=[f"Analysis error: {str(e)}"],
                reasoning="Could not analyze hardware context due to error",
                raw_response="",
            )

    def _parse_hardware_context_response(self, response: str) -> HardwareContextResult:
        """Parse hardware context analysis response"""
        suspicious = False
        concerns = []
        reasoning = ""

        for line in response.split("\n"):
            line = line.strip()
            if line.startswith("SUSPICIOUS:"):
                value = line.split(":", 1)[1].strip().upper()
                suspicious = value == "YES"
            elif line.startswith("CONCERNS:"):
                concerns_str = line.split(":", 1)[1].strip()
                concerns = [c.strip() for c in concerns_str.split(";") if c.strip()]
            elif line.startswith("REASONING:"):
                reasoning = line.split(":", 1)[1].strip()

        return HardwareContextResult(
            suspicious=suspicious,
            concerns=concerns,
            reasoning=reasoning,
            raw_response=response,
        )

    def _evaluate_consensus_with_hardware(
        self,
        original: DetectionResult,
        adversarial: AdversarialResult,
        independent: IndependentResult,
        temporal: TemporalResult,
        ensemble: EnsembleResult | None = None,
        hardware_context: HardwareContextResult | None = None,
    ) -> tuple[bool, str]:
        """
        Evaluate consensus across all verification strategies including hardware context.

        Returns (consensus_reached, reasoning)
        """
        # Count agreements
        agreements = 0
        disagreements = []
        total_strategies = 3  # Base: adversarial, independent, temporal

        # Check adversarial: should NOT find strong counter-evidence
        if not adversarial.found_counter_evidence:
            agreements += 1
        else:
            disagreements.append(
                f"Adversarial found counter-evidence: {', '.join(adversarial.concerns[:2])}"
            )

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

        # Check ensemble (if available, for hatching detection)
        if ensemble is not None:
            total_strategies += 1
            if ensemble.consensus_detected:
                agreements += 1
            else:
                disagreements.append(
                    f"Ensemble voting disagrees: {ensemble.votes_yes}/{ensemble.total_votes} YES "
                    f"({ensemble.agreement_ratio:.0%}), threshold is {self.ensemble_threshold:.0%}"
                )

        # Check hardware context (if available)
        if hardware_context is not None:
            total_strategies += 1
            if not hardware_context.suspicious:
                agreements += 1
            else:
                disagreements.append(
                    f"Hardware context suspicious: {hardware_context.reasoning}; "
                    f"Concerns: {', '.join(hardware_context.concerns[:2])}"
                )

        # Consensus requires all strategies to agree
        consensus = agreements == total_strategies

        if consensus:
            parts = [
                "no counter-evidence found",
                "independent analysis confirms",
                "temporal change observed",
            ]
            if ensemble:
                parts.append(f"ensemble confirms ({ensemble.votes_yes}/{ensemble.total_votes} YES)")
            if hardware_context:
                parts.append("no hardware error concerns")
            reasoning = (
                f"All verification strategies agree ({total_strategies}/{total_strategies}): "
                + ", ".join(parts)
            )
        else:
            reasoning = (
                f"Verification disagreement ({agreements}/{total_strategies} agree): "
                f"{'; '.join(disagreements)}"
            )

        return consensus, reasoning

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

            # Build detector-specific critical review guidance
            if detector.name == "hatching":
                specific_guidance = """
For HATCHING specifically, look for these common FALSE POSITIVE patterns:
- Is the worm still COILED/PRETZEL-SHAPED inside the eggshell?
- Is the eggshell just EXPANDED/STRETCHED but worm still contained?
- Is the worm filling the shell but NOT actually OUTSIDE the boundary?
- Could this be late 3-fold stage (tightly packed) rather than hatched?

TRUE hatching requires the worm to be OUTSIDE the shell (free-floating, elongated, or field empty).
"""
            else:
                specific_guidance = ""

            prompt = f"""You are reviewing a detection result for a C. elegans embryo
(diSPIM max projection).

The system detected: {detector.name}
Original confidence: {original_result.confidence.value if original_result.confidence else "unknown"}
Original reasoning: {original_result.reasoning or "not provided"}

NOW ACT AS A CRITICAL REVIEWER. Your job is to find reasons why this detection might be INCORRECT:
- Could this be a false positive?
- Are there artifacts, noise, or imaging issues that could be misleading?
- Is the evidence actually conclusive, or could it be interpreted differently?
- Are there alternative explanations for what is observed?
{specific_guidance}
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
                messages=[{"role": "user", "content": content}],
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

            # Build detector-specific criteria
            if detector.name == "hatching":
                criteria = """
TRUE HATCHING criteria (must meet at least one):
- Worm body is OUTSIDE the eggshell boundary (free-floating, elongated)
- Empty field of view (worm has left)
- Worm moving in/out of frame (not confined to egg location)

NOT HATCHING indicators:
- Worm is still coiled/pretzel-shaped INSIDE the eggshell
- Expanded/stretched shell with worm still contained"""
            else:
                criteria = detector.description

            # Use a neutral prompt that doesn't reveal the previous detection
            prompt = f"""Analyze this C. elegans embryo image (diSPIM max projection) at
timepoint {timepoint}.

Question: Has '{detector.name}' occurred in this embryo?

{criteria}

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
                messages=[{"role": "user", "content": content}],
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
                    prev_images.append(
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": img.max_projection_b64,
                            },
                        }
                    )

            if not prev_images:
                return TemporalResult(
                    change_detected=True,
                    description="No previous images available",
                    confidence=ConfidenceLevel.LOW,
                    raw_response="",
                )

            # Build detector-specific temporal criteria
            if detector.name == "hatching":
                temporal_criteria = """For HATCHING, look for:
- A visible BREACH appearing in the eggshell boundary (not just expansion)
- The worm physically EXITING the shell (part of body moves outside)
- A transition from INSIDE to OUTSIDE the shell
- NOT just shell expansion or increased movement while still contained"""
            else:
                temporal_criteria = f"""For {detector.name}, look for:
- Actual transition or change between frames
- Not just a static state that could have existed before
- Clear evidence of progression or event occurrence"""

            prompt = f"""Compare these sequential timepoints of a C. elegans embryo
(diSPIM max projection).

PREVIOUS TIMEPOINTS (shown first):
These are from t={timepoint - 2} to t={timepoint - 1}

CURRENT TIMEPOINT (shown last):
This is t={timepoint}

Question: Is there a clear CHANGE consistent with '{detector.name}'?

{temporal_criteria}

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
                messages=[{"role": "user", "content": content}],
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

    def _get_image_content(self, embryo_state: EmbryoState, num_images: int = 1) -> list[dict]:
        """Get image content blocks for Claude API"""
        images = []
        recent = embryo_state.recent_images[-num_images:] if embryo_state.recent_images else []

        for img in recent:
            if img.max_projection_b64:
                images.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": img.max_projection_b64,
                        },
                    }
                )

        return images

    async def _run_ensemble_hatching(self, embryo_state: EmbryoState) -> EnsembleResult:
        """
        Run ensemble voting for hatching detection using many parallel Haiku calls.

        This provides statistical noise reduction through collective wisdom.
        Each call makes an independent YES/NO hatching determination.

        Returns
        -------
        EnsembleResult
            Voting results with agreement ratio
        """
        try:
            # Get the current image
            images = self._get_image_content(embryo_state, num_images=1)
            if not images:
                return EnsembleResult(
                    votes_yes=0,
                    votes_no=0,
                    total_votes=0,
                    agreement_ratio=0.0,
                    consensus_detected=False,
                    raw_responses=["No images available"],
                )

            # Simple, focused prompt for hatching detection
            ensemble_prompt = """Look at this C. elegans embryo image (diSPIM max projection).

Answer ONE question: Has the embryo HATCHED?

HATCHED means: The worm body is OUTSIDE the eggshell (free-floating, elongated, or field is
empty because worm left).
NOT HATCHED means: The worm is still INSIDE the eggshell (coiled/pretzel-shaped, even if
shell looks expanded).

Respond with ONLY: YES or NO"""

            async def single_vote() -> str:
                """Make a single API call for one vote"""
                try:
                    content = [{"type": "text", "text": ensemble_prompt}] + images
                    response = await asyncio.to_thread(
                        self.claude.messages.create,
                        model=self.ensemble_model,
                        max_tokens=10,  # Very short response expected
                        messages=[{"role": "user", "content": content}],
                    )
                    return response.content[0].text.strip().upper()
                except Exception as e:
                    logger.debug(f"Ensemble vote failed: {e}")
                    return "ERROR"

            # Run all votes in parallel
            logger.info(
                f"[ENSEMBLE] Running {self.ensemble_size} parallel Haiku calls"
                " for hatching verification"
            )
            tasks = [single_vote() for _ in range(self.ensemble_size)]
            responses = await asyncio.gather(*tasks)

            # Count votes
            votes_yes = sum(1 for r in responses if "YES" in r)
            votes_no = sum(1 for r in responses if "NO" in r)
            errors = sum(1 for r in responses if "ERROR" in r)
            total_valid = votes_yes + votes_no

            if total_valid == 0:
                return EnsembleResult(
                    votes_yes=0,
                    votes_no=0,
                    total_votes=0,
                    agreement_ratio=0.0,
                    consensus_detected=False,
                    raw_responses=responses,
                )

            # Calculate agreement ratio
            agreement_ratio = votes_yes / total_valid
            consensus_detected = agreement_ratio >= self.ensemble_threshold

            logger.info(
                f"[ENSEMBLE] Results: {votes_yes} YES, {votes_no} NO, {errors} errors. "
                f"Agreement: {agreement_ratio:.1%}. Consensus: {consensus_detected}"
            )

            return EnsembleResult(
                votes_yes=votes_yes,
                votes_no=votes_no,
                total_votes=total_valid,
                agreement_ratio=agreement_ratio,
                consensus_detected=consensus_detected,
                raw_responses=responses[:10],  # Keep only first 10 for debugging
            )

        except Exception as e:
            logger.error(f"Ensemble verification failed: {e}")
            return EnsembleResult(
                votes_yes=0,
                votes_no=0,
                total_votes=0,
                agreement_ratio=0.0,
                consensus_detected=False,
                raw_responses=[f"Error: {str(e)}"],
            )

    def _parse_adversarial_response(self, response: str) -> AdversarialResult:
        """Parse adversarial strategy response"""
        found_counter = False
        concerns = []
        confidence = None

        for line in response.split("\n"):
            line = line.strip()
            if line.startswith("COUNTER_EVIDENCE_FOUND:"):
                value = line.split(":", 1)[1].strip().upper()
                found_counter = value == "YES"
            elif line.startswith("CONCERNS:"):
                concerns_str = line.split(":", 1)[1].strip()
                concerns = [c.strip() for c in concerns_str.split(";") if c.strip()]
            elif line.startswith("CONFIDENCE_IN_ORIGINAL:"):
                value = line.split(":", 1)[1].strip().upper()
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

        for line in response.split("\n"):
            line = line.strip()
            if line.startswith("DETECTED:"):
                value = line.split(":", 1)[1].strip().upper()
                detected = value == "YES"
            elif line.startswith("CONFIDENCE:"):
                value = line.split(":", 1)[1].strip().upper()
                try:
                    confidence = ConfidenceLevel(value)
                except ValueError:
                    pass
            elif line.startswith("KEY_EVIDENCE:"):
                evidence = line.split(":", 1)[1].strip()

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

        for line in response.split("\n"):
            line = line.strip()
            if line.startswith("CHANGE_DETECTED:"):
                value = line.split(":", 1)[1].strip().upper()
                change_detected = value == "YES"
            elif line.startswith("DESCRIPTION:"):
                description = line.split(":", 1)[1].strip()
            elif line.startswith("CONFIDENCE:"):
                value = line.split(":", 1)[1].strip().upper()
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
        ensemble: EnsembleResult | None = None,
    ) -> tuple[bool, str]:
        """
        Evaluate consensus across all verification strategies.

        Returns (consensus_reached, reasoning)
        """
        # Count agreements
        agreements = 0
        disagreements = []
        total_strategies = 3  # Base: adversarial, independent, temporal

        # Check adversarial: should NOT find strong counter-evidence
        if not adversarial.found_counter_evidence:
            agreements += 1
        else:
            disagreements.append(
                f"Adversarial found counter-evidence: {', '.join(adversarial.concerns[:2])}"
            )

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

        # Check ensemble (if available, for hatching detection)
        if ensemble is not None:
            total_strategies = 4
            if ensemble.consensus_detected:
                agreements += 1
            else:
                disagreements.append(
                    f"Ensemble voting disagrees: {ensemble.votes_yes}/{ensemble.total_votes} YES "
                    f"({ensemble.agreement_ratio:.0%}), threshold is {self.ensemble_threshold:.0%}"
                )

        # Consensus requires all strategies to agree
        consensus = agreements == total_strategies

        if consensus:
            if ensemble:
                reasoning = (
                    f"All verification strategies agree ({total_strategies}/{total_strategies}): "
                    f"no counter-evidence found, independent analysis confirms detection, "
                    f"temporal change observed, ensemble voting confirms "
                    f"({ensemble.votes_yes}/{ensemble.total_votes}"
                    f" = {ensemble.agreement_ratio:.0%} YES)."
                )
            else:
                reasoning = (
                    "All verification strategies agree: "
                    "no counter-evidence found, independent analysis confirms detection, "
                    "temporal change observed."
                )
        else:
            reasoning = (
                f"Verification disagreement ({agreements}/{total_strategies} agree): "
                f"{'; '.join(disagreements)}"
            )

        return consensus, reasoning
