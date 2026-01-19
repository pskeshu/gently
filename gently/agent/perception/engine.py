"""
Simple Perception Engine.

Show reference examples, show current image, ask what stage.
No probability distributions, no tiered models, no complex parsing.
"""

import asyncio
import base64
import io
import json
import logging
import re
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import anthropic
import numpy as np
from PIL import Image
from scipy import ndimage

from .session import (
    Observation,
    ObservedFeatures,
    ContrastiveReasoning,
    ReasoningStep,
    ReasoningTrace,
    PerceptionResult,
    PerceptionSession,
    CandidateStage,
    StageComparison,
    MultiPhaseReasoningTrace,
    PhaseTrace,
)
from .example_store import ExampleStore
from .stages import STAGES
from .verification import VerificationEngine

logger = logging.getLogger(__name__)


def render_volume_view(
    volume: np.ndarray,
    rotation_x: float = 0,
    rotation_y: float = 0,
    threshold: float = 0.2,
) -> str:
    """
    Render a 3D volume from a specific viewing angle using alpha compositing.

    This produces a depth-aware view similar to the 3D viewer, where you can
    see the embryo's shape and structure, not just a flat max projection.

    Parameters
    ----------
    volume : np.ndarray
        3D volume (Z, Y, X)
    rotation_x : float
        Rotation around X axis in degrees (-90 to 90)
    rotation_y : float
        Rotation around Y axis in degrees (-180 to 180)
    threshold : float
        Intensity threshold for transparency (0-1)

    Returns
    -------
    str
        Base64-encoded JPEG image
    """
    # Handle 4D volumes (Views, Z, Y, X) - take first view
    if volume.ndim == 4:
        volume = volume[0]

    # Normalize to 0-1
    vol = volume.astype(np.float32)
    p1, p99 = np.percentile(vol, [1, 99])
    vol = np.clip((vol - p1) / (p99 - p1 + 1e-8), 0, 1)

    # Apply rotations
    if rotation_y != 0:
        vol = ndimage.rotate(vol, rotation_y, axes=(0, 2), reshape=False, order=1)
    if rotation_x != 0:
        vol = ndimage.rotate(vol, rotation_x, axes=(0, 1), reshape=False, order=1)

    # Alpha composite from back to front (same as Three.js stacked slices)
    z_depth = vol.shape[0]
    result = np.zeros(vol.shape[1:], dtype=np.float32)
    accumulated_alpha = np.zeros_like(result)

    for z in range(z_depth):
        slice_val = vol[z]
        # Alpha based on intensity above threshold
        alpha = np.clip((slice_val - threshold) / (1 - threshold + 1e-8), 0, 1) * 0.3

        # Front-to-back compositing
        result += slice_val * alpha * (1 - accumulated_alpha)
        accumulated_alpha += alpha * (1 - accumulated_alpha)

    # Normalize result to 0-255
    if result.max() > 0:
        result = (result / result.max() * 255).astype(np.uint8)
    else:
        result = result.astype(np.uint8)

    # Convert to JPEG base64
    pil_image = Image.fromarray(result)

    # Resize if too large (max 800px)
    max_dim = 800
    if max(pil_image.size) > max_dim:
        scale = max_dim / max(pil_image.size)
        new_size = (int(pil_image.size[0] * scale), int(pil_image.size[1] * scale))
        pil_image = pil_image.resize(new_size, Image.Resampling.LANCZOS)

    buffer = io.BytesIO()
    pil_image.save(buffer, format='JPEG', quality=85)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

# Tool definitions for interleaved reasoning
PERCEPTION_TOOLS = [
    {
        "name": "view_previous_timepoint",
        "description": (
            "View the microscopy image from a previous timepoint to compare morphological "
            "changes over time. Use this when you're uncertain about the current stage and "
            "need to see how the embryo looked earlier to assess progression.\n\n"
            "Example inputs:\n"
            "- {\"offset\": 1, \"reason\": \"Need to see if the fold has progressed since last timepoint\"}\n"
            "- {\"offset\": 2, \"reason\": \"Checking if this is a new stage or was already folded\"}"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "offset": {
                    "type": "integer",
                    "description": "How many timepoints back to look (1 = immediately previous, 2 = two timepoints back, etc.)",
                    "minimum": 1,
                    "maximum": 5,
                },
                "reason": {
                    "type": "string",
                    "description": "Brief explanation of why you need to see this previous timepoint",
                },
            },
            "required": ["offset", "reason"],
        },
    },
    # NOTE: view_reference_example removed - reference images are already included in the prompt
    # NOTE: view_reference_example_3d removed - reference images are already included in the prompt
    {
        "name": "view_embryo",
        "description": (
            "View the CURRENT embryo's 3D volume from a specific angle. This renders a depth-aware view "
            "that shows the embryo's shape and structure (not a flat projection). Use this when:\n"
            "- You need to see the embryo's 3D morphology from a different perspective\n"
            "- The default top-down view doesn't clearly show folding or coiling\n"
            "- You want to verify if a structure is real or an imaging artifact\n"
            "- You need to see head-tail orientation or body segments\n\n"
            "Example inputs:\n"
            "- {\"rotation_x\": 45, \"rotation_y\": 0, \"reason\": \"Check if apparent fold is real\"}\n"
            "- {\"rotation_x\": 0, \"rotation_y\": 90, \"reason\": \"View from side to assess coiling\"}\n"
            "- {\"rotation_x\": 30, \"rotation_y\": 45, \"reason\": \"Get angled view to see body shape\"}"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "rotation_x": {
                    "type": "number",
                    "description": "Rotation around X axis in degrees (-90 to 90). 0 = top-down, positive = tilt forward",
                    "minimum": -90,
                    "maximum": 90,
                },
                "rotation_y": {
                    "type": "number",
                    "description": "Rotation around Y axis in degrees (-180 to 180). 0 = front view, 90 = side view",
                    "minimum": -180,
                    "maximum": 180,
                },
                "timepoint": {
                    "type": "integer",
                    "description": "Timepoint to view (optional, defaults to current). Use to compare with previous timepoints.",
                },
                "reason": {
                    "type": "string",
                    "description": "Brief explanation of why you need this viewing angle",
                },
            },
            "required": ["reason"],
        },
    },
    {
        "name": "request_verification",
        "description": (
            "Request parallel verification subagents to compare specific stage pairs when your initial "
            "classification has low confidence. This spawns separate focused agents that compare the "
            "embryo against reference examples for specific stages.\n\n"
            "USE THIS WHEN:\n"
            "- Your initial confidence is < 0.7\n"
            "- You're uncertain between 2-3 adjacent stages\n"
            "- The embryo appears transitional\n"
            "- Morphological features are ambiguous\n\n"
            "Example inputs:\n"
            "- {\"initial_stage\": \"comma\", \"confidence\": 0.55, \"comparisons\": [{\"stage_a\": \"comma\", \"stage_b\": \"1.5fold\", \"reason\": \"Uncertain if fold is beginning\", \"use_3d\": false}], \"key_question\": \"Has folding begun?\"}\n"
            "- {\"initial_stage\": \"1.5fold\", \"confidence\": 0.6, \"comparisons\": [{\"stage_a\": \"comma\", \"stage_b\": \"1.5fold\", \"reason\": \"Curvature vs fold\", \"use_3d\": true}, {\"stage_a\": \"1.5fold\", \"stage_b\": \"2fold\", \"reason\": \"Count folds\", \"use_3d\": true}], \"key_question\": \"How many folds are present?\"}"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "initial_stage": {
                    "type": "string",
                    "description": "Your best initial guess for the stage",
                    "enum": ["early", "bean", "comma", "1.5fold", "2fold", "pretzel", "hatching", "hatched"],
                },
                "confidence": {
                    "type": "number",
                    "description": "Your confidence in the initial stage (0.0-1.0). Only request verification if < 0.7",
                    "minimum": 0.0,
                    "maximum": 1.0,
                },
                "comparisons": {
                    "type": "array",
                    "description": "Stage pairs to compare (max 3). Each comparison spawns a focused subagent.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "stage_a": {
                                "type": "string",
                                "description": "First stage to compare",
                                "enum": ["early", "bean", "comma", "1.5fold", "2fold", "pretzel"],
                            },
                            "stage_b": {
                                "type": "string",
                                "description": "Second stage to compare",
                                "enum": ["early", "bean", "comma", "1.5fold", "2fold", "pretzel"],
                            },
                            "reason": {
                                "type": "string",
                                "description": "Why this comparison is needed",
                            },
                            "use_3d": {
                                "type": "boolean",
                                "description": "Whether to include 3D reference views for this comparison",
                            },
                        },
                        "required": ["stage_a", "stage_b", "reason", "use_3d"],
                    },
                    "maxItems": 3,
                },
                "key_question": {
                    "type": "string",
                    "description": "The key uncertainty that verification should resolve",
                },
            },
            "required": ["initial_stage", "confidence", "comparisons", "key_question"],
        },
    },
]


class PerceptionEngine:
    """
    Simple perception engine.

    Just: show examples, show image, ask what stage.
    """

    MODEL = "claude-opus-4-5-20251101"

    def __init__(
        self,
        claude_client: anthropic.Anthropic,
        example_store: Optional[ExampleStore] = None,
        examples_path: Optional[Path] = None,
        volume_accessor: Optional[Callable[[str, int], Optional[np.ndarray]]] = None,
        enable_verification: bool = True,
    ):
        """
        Parameters
        ----------
        claude_client : anthropic.Anthropic
            Anthropic client for API calls
        example_store : ExampleStore, optional
            Pre-loaded example store
        examples_path : Path, optional
            Path to examples directory
        volume_accessor : callable, optional
            Function (embryo_id, timepoint) -> volume array for 3D viewing.
            If provided, enables the view_embryo tool.
        enable_verification : bool
            Whether to enable multi-phase verification. Defaults to True.
        """
        self.claude = claude_client
        self.volume_accessor = volume_accessor
        self.enable_verification = enable_verification

        # Load examples if provided
        if example_store:
            self.example_store = example_store
        elif examples_path:
            self.example_store = ExampleStore(examples_path)
        else:
            self.example_store = None

        # Cache loaded examples (with descriptions)
        self._examples_cache: Optional[Dict[str, List[Dict]]] = None

        # Temporary state during perceive() call
        self._current_volume: Optional[np.ndarray] = None
        self._current_embryo_id: Optional[str] = None
        self._current_timepoint: Optional[int] = None
        self._current_image_b64: Optional[str] = None  # For verification subagents

        # Verification engine for multi-phase perception
        self.verification_engine = VerificationEngine(
            claude_client=claude_client,
            example_store=self.example_store,
        ) if enable_verification else None

    def _load_all_examples(self) -> Dict[str, List[Dict]]:
        """Load all stage examples with descriptions (cached)."""
        if self._examples_cache is not None:
            return self._examples_cache

        if not self.example_store:
            return {}

        examples = {}
        for stage in STAGES:
            stage_examples = self.example_store.get_stage_examples_with_descriptions(
                stage, max_examples=2
            )
            if stage_examples:
                examples[stage] = stage_examples

        self._examples_cache = examples
        return examples

    async def perceive(
        self,
        image_b64: str,
        session: PerceptionSession,
        timepoint: int,
        volume: Optional[np.ndarray] = None,
        top_image_b64: Optional[str] = None,
        side_image_b64: Optional[str] = None,
    ) -> PerceptionResult:
        """
        Perceive the current image using interleaved reasoning.

        The VLM can request additional context (previous timepoints, reference
        examples, 3D views) via tool use when uncertain about the classification.

        Parameters
        ----------
        image_b64 : str
            Base64-encoded current image (combined view for backward compatibility)
        session : PerceptionSession
            Session with previous observations
        timepoint : int
            Current timepoint number
        volume : np.ndarray, optional
            Current 3D volume for view_embryo tool. If provided, enables
            3D viewing from arbitrary angles.
        top_image_b64 : str, optional
            Base64-encoded TOP view image (looking down). If provided along with
            side_image_b64, sends views as separate images for better analysis.
        side_image_b64 : str, optional
            Base64-encoded SIDE view image (looking from front).

        Returns
        -------
        PerceptionResult
            Stage classification and hatching status with full reasoning trace
        """
        # Store current image for potential future reference
        session.store_image(timepoint, image_b64)

        # Set up temporary state for tool access
        self._current_volume = volume
        self._current_embryo_id = session.embryo_id
        self._current_timepoint = timepoint
        self._current_image_b64 = image_b64  # For verification subagents

        try:
            # Build initial prompt - use separate images if available
            content = self._build_prompt(
                image_b64, session, timepoint,
                top_image_b64=top_image_b64,
                side_image_b64=side_image_b64
            )

            # Run interleaved reasoning loop with tool use
            result, trace = await self._run_reasoning_loop(
                content=content,
                session=session,
                timepoint=timepoint,
            )

            # Attach reasoning trace to result
            result.reasoning_trace = trace

            logger.info(
                f"[{session.embryo_id}] T{timepoint}: "
                f"stage={result.stage}, hatching={result.is_hatching}, "
                f"confidence={result.confidence:.0%}, "
                f"tool_calls={trace.total_tool_calls}, "
                f"reasoning={result.reasoning[:50] if result.reasoning else 'EMPTY'}..."
            )
        finally:
            # Clear temporary state to release memory
            self._current_volume = None
            self._current_embryo_id = None
            self._current_timepoint = None
            self._current_image_b64 = None

        return result

    async def _run_reasoning_loop(
        self,
        content: List[Dict],
        session: PerceptionSession,
        timepoint: int,
        max_iterations: int = 5,
    ) -> tuple:
        """
        Run the interleaved reasoning loop with tool use.

        Returns (PerceptionResult, ReasoningTrace)
        """
        trace = ReasoningTrace()

        # Add temporal context as first step in trace (for observability)
        temporal = session.compute_temporal_analysis()
        if temporal:
            temporal_content = (
                f"Stage: {temporal.current_stage}, "
                f"Time: {temporal.time_in_current_stage_min:.0f}min, "
                f"Overtime: {temporal.overtime_ratio:.1f}x"
            )
            if temporal.is_potentially_arrested:
                temporal_content += f" [ARREST WARNING: {temporal.arrest_reason}]"

            trace.add_step(ReasoningStep(
                step_type="temporal_context",
                content=temporal_content,
            ))

        messages = [{"role": "user", "content": content}]

        for iteration in range(max_iterations):
            # Call Claude with tools
            response = await self._call_claude_with_tools(messages)

            # Extract thinking content if present (extended thinking)
            thinking_content = None
            for block in response.content:
                if block.type == "thinking":
                    thinking_content = block.thinking
                    # Log thinking summary for debugging
                    thinking_preview = thinking_content[:500] if thinking_content else ""
                    logger.debug(f"Extended thinking: {thinking_preview}...")
                    break

            # Record thinking in trace if present
            if thinking_content:
                trace.add_step(ReasoningStep(
                    step_type="thinking",
                    content=thinking_content[:2000],  # Truncate for storage
                ))

            # Check if we got a final response (no tool use)
            if response.stop_reason == "end_turn":
                # Extract text response
                text_response = ""
                for block in response.content:
                    if block.type == "text":
                        text_response = block.text

                # Record final decision
                trace.add_step(ReasoningStep(
                    step_type="final_decision",
                    content=text_response,
                ))

                # Parse and return
                result = self._parse_response(text_response)
                return result, trace

            # Handle tool use
            if response.stop_reason == "tool_use":
                # Check for verification request first (special handling)
                verification_block = None
                for block in response.content:
                    if block.type == "tool_use" and block.name == "request_verification":
                        verification_block = block
                        break

                # If verification requested, run subagents and return result
                if verification_block and self.verification_engine:
                    # Record the verification request in trace
                    for block in response.content:
                        if block.type == "text":
                            trace.add_step(ReasoningStep(
                                step_type="initial_analysis",
                                content=block.text,
                            ))
                        elif block.type == "tool_use":
                            trace.add_step(ReasoningStep(
                                step_type="tool_call",
                                content=f"Requesting {block.name}",
                                tool_name=block.name,
                                tool_input=block.input,
                            ))

                    # Run verification and return result
                    result = await self._handle_verification_request(
                        verification_block.input,
                        trace,
                    )
                    return result, trace

                # Build assistant message with the response
                assistant_content = []
                for block in response.content:
                    if block.type == "text":
                        # Record initial analysis (full content)
                        trace.add_step(ReasoningStep(
                            step_type="initial_analysis",
                            content=block.text,
                        ))
                        assistant_content.append({
                            "type": "text",
                            "text": block.text,
                        })
                    elif block.type == "tool_use":
                        # Record tool call
                        trace.add_step(ReasoningStep(
                            step_type="tool_call",
                            content=f"Requesting {block.name}",
                            tool_name=block.name,
                            tool_input=block.input,
                        ))
                        assistant_content.append({
                            "type": "tool_use",
                            "id": block.id,
                            "name": block.name,
                            "input": block.input,
                        })

                messages.append({"role": "assistant", "content": assistant_content})

                # Process tool calls and build tool results
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        result_content, result_summary, image_tp, image_type = self._handle_tool_call(
                            tool_name=block.name,
                            tool_input=block.input,
                            session=session,
                            timepoint=timepoint,
                        )

                        # Record tool result
                        trace.add_step(ReasoningStep(
                            step_type="tool_result",
                            content=result_summary,
                            tool_name=block.name,
                            tool_result_summary=result_summary,
                            image_timepoint=image_tp,
                            image_type=image_type,
                        ))

                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result_content,
                        })

                messages.append({"role": "user", "content": tool_results})

        # Max iterations reached - parse last response
        logger.warning(f"Max reasoning iterations ({max_iterations}) reached")
        return self._parse_response(""), trace

    def _handle_tool_call(
        self,
        tool_name: str,
        tool_input: Dict,
        session: PerceptionSession,
        timepoint: int,
    ) -> tuple:
        """
        Handle a tool call and return the result.

        Returns (content_for_claude, summary, image_timepoint, image_type)
        """
        if tool_name == "view_previous_timepoint":
            offset = tool_input.get("offset", 1)
            reason = tool_input.get("reason", "")

            prev_data = session.get_previous_image(timepoint, offset)
            if prev_data:
                prev_tp, prev_image = prev_data
                logger.info(f"Tool: Providing image from T{prev_tp} (offset={offset})")

                # Return image content
                content = [
                    {"type": "text", "text": f"Here is the image from timepoint T{prev_tp} (requested because: {reason}):"},
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": prev_image,
                        }
                    },
                ]
                summary = f"Showed T{prev_tp} image"
                return content, summary, prev_tp, "previous_timepoint"
            else:
                available = session.get_available_timepoints()
                content = [{"type": "text", "text": f"Image not available for T{timepoint - offset}. Available timepoints: {available}"}]
                summary = f"T{timepoint - offset} not available"
                return content, summary, None, None

        # NOTE: view_reference_example and view_reference_example_3d handlers removed
        # Reference images are already included in the prompt, no need for tools

        elif tool_name == "view_embryo":
            rotation_x = tool_input.get("rotation_x", 0)
            rotation_y = tool_input.get("rotation_y", 0)
            target_tp = tool_input.get("timepoint", self._current_timepoint)
            reason = tool_input.get("reason", "")

            # Get the volume
            vol = None
            if target_tp == self._current_timepoint and self._current_volume is not None:
                vol = self._current_volume
            elif self.volume_accessor is not None and self._current_embryo_id is not None:
                vol = self.volume_accessor(self._current_embryo_id, target_tp)

            if vol is not None:
                logger.info(
                    f"Tool: Rendering 3D view at rotation_x={rotation_x}, "
                    f"rotation_y={rotation_y}, timepoint={target_tp}"
                )

                # Render the view
                rendered_b64 = render_volume_view(vol, rotation_x, rotation_y)

                content = [
                    {
                        "type": "text",
                        "text": (
                            f"Here is a 3D rendered view of the embryo at T{target_tp} "
                            f"(rotation_x={rotation_x}°, rotation_y={rotation_y}°). "
                            f"Requested because: {reason}"
                        ),
                    },
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": rendered_b64,
                        },
                    },
                ]
                summary = f"3D view at rx={rotation_x}, ry={rotation_y}"
                return content, summary, target_tp, "volume_view"
            else:
                content = [
                    {
                        "type": "text",
                        "text": f"3D volume not available for timepoint {target_tp}. Volume viewing requires volume data to be passed to perceive().",
                    }
                ]
                summary = "Volume not available"
                return content, summary, None, None

        else:
            content = [{"type": "text", "text": f"Unknown tool: {tool_name}"}]
            return content, "Unknown tool", None, None

    async def _handle_verification_request(
        self,
        tool_input: Dict,
        trace: ReasoningTrace,
    ) -> PerceptionResult:
        """
        Handle a request_verification tool call by running parallel subagents.

        Returns PerceptionResult with verification data populated.
        """
        initial_stage = tool_input.get("initial_stage", "early")
        initial_confidence = float(tool_input.get("confidence", 0.5))
        comparisons_data = tool_input.get("comparisons", [])
        key_question = tool_input.get("key_question", "")

        logger.info(
            f"Verification requested: initial={initial_stage} ({initial_confidence:.0%}), "
            f"comparisons={len(comparisons_data)}, question='{key_question}'"
        )

        # Record verification request in trace
        trace.add_step(ReasoningStep(
            step_type="verification_requested",
            content=f"Running {len(comparisons_data)} verification subagents for: {key_question}",
        ))

        # Parse comparisons into StageComparison objects
        comparisons = []
        for comp in comparisons_data[:3]:  # Max 3 comparisons
            comparisons.append(StageComparison(
                stage_a=comp.get("stage_a", ""),
                stage_b=comp.get("stage_b", ""),
                reason=comp.get("reason", ""),
                use_3d=comp.get("use_3d", False),
            ))

        if not comparisons:
            logger.warning("No valid comparisons in verification request")
            return PerceptionResult(
                stage=initial_stage,
                is_hatching=(initial_stage == "hatching"),
                confidence=initial_confidence,
                reasoning=f"Verification requested but no comparisons provided. Using initial: {initial_stage}",
                verification_triggered=True,
                phase_count=2,
            )

        # Run verification subagents in parallel
        aggregation, subagent_traces = await self.verification_engine.run_verification(
            comparisons=comparisons,
            image_b64=self._current_image_b64,
            initial_stage=initial_stage,
            initial_confidence=initial_confidence,
            key_question=key_question,
            volume=self._current_volume,
        )

        # Record subagent results in trace
        for subagent_trace in subagent_traces:
            result_text = "FAILED"
            if subagent_trace.result:
                result_text = (
                    f"prefers {subagent_trace.result.preferred_stage} "
                    f"({subagent_trace.result.confidence:.0%})"
                )
            trace.add_step(ReasoningStep(
                step_type="verification_subagent",
                content=(
                    f"{subagent_trace.comparison.stage_a} vs {subagent_trace.comparison.stage_b}: "
                    f"{result_text}"
                ),
                tool_name="verification_subagent",
                tool_input=subagent_trace.comparison.to_dict(),
                tool_result_summary=result_text,
            ))

        # Record aggregation result
        trace.add_step(ReasoningStep(
            step_type="verification_result",
            content=(
                f"Aggregation: {aggregation.winning_stage} ({aggregation.aggregated_confidence:.0%}), "
                f"override={aggregation.should_override_initial}, "
                f"agreement={aggregation.subagents_agree}"
            ),
        ))

        # Build reasoning string
        reasoning_parts = [
            f"Initial assessment: {initial_stage} ({initial_confidence:.0%})",
            f"Key question: {key_question}",
        ]
        for subagent_trace in subagent_traces:
            if subagent_trace.result:
                reasoning_parts.append(
                    f"Subagent ({subagent_trace.comparison.stage_a} vs {subagent_trace.comparison.stage_b}): "
                    f"prefers {subagent_trace.result.preferred_stage} ({subagent_trace.result.confidence:.0%})"
                )
        reasoning_parts.append(
            f"Final: {aggregation.winning_stage} ({aggregation.aggregated_confidence:.0%})"
        )

        # Build candidate stages from comparisons
        candidate_stages = [
            CandidateStage(
                stage=initial_stage,
                confidence=initial_confidence,
                evidence_for=["Initial perception assessment"],
                evidence_against=[],
            )
        ]
        for subagent_trace in subagent_traces:
            if subagent_trace.result and subagent_trace.result.preferred_stage != initial_stage:
                candidate_stages.append(CandidateStage(
                    stage=subagent_trace.result.preferred_stage,
                    confidence=subagent_trace.result.confidence,
                    evidence_for=subagent_trace.result.evidence_for_preferred,
                    evidence_against=subagent_trace.result.evidence_against_other,
                ))

        logger.info(
            f"Verification complete: {initial_stage} -> {aggregation.winning_stage} "
            f"(override={aggregation.should_override_initial})"
        )

        return PerceptionResult(
            stage=aggregation.winning_stage,
            is_hatching=(aggregation.winning_stage == "hatching"),
            confidence=aggregation.aggregated_confidence,
            reasoning=" | ".join(reasoning_parts),
            verification_triggered=True,
            verification_result=aggregation,
            candidate_stages=candidate_stages,
            phase_count=3,
            should_stop=(aggregation.winning_stage == "hatched"),
        )

    def _build_cached_system_prompt(self) -> List[Dict]:
        """Build TEXT-ONLY system prompt for caching.

        Note: Anthropic API only allows text in system prompts, not images.
        Images (reference examples) must go in the user message.

        Uses 1-hour TTL for caching across timelapse sessions.
        """
        return [{
            "type": "text",
            "text": """You are an expert microscopy perception system analyzing C. elegans embryo development.

IMPORTANT PRINCIPLES:
1. DESCRIBE FIRST: Always describe what you actually see BEFORE classifying
2. EMBRACE TRANSITIONS: If features suggest a transitional state, SAY SO
3. CALIBRATE CONFIDENCE: Hedging words (slight, subtle, beginning) = lower confidence (0.5-0.7)

Development is a SPECTRUM, not discrete jumps. Trust your observations over expectations.

IMAGE FORMAT - Each image shows THREE ORTHOGONAL VIEWS:
┌─────────┬─────────┐
│   XY    │   YZ    │  (TOP ROW)
├─────────┴─────────┤
│        XZ         │  (BOTTOM ROW)
└───────────────────┘

- XY (top-left): Looking DOWN - Best for end asymmetry, ventral indentation, folding
- YZ (top-right): Looking from SIDE - Best for body height/thickness
- XZ (bottom): Looking from FRONT - CRITICAL for early→bean transition (look for "peanut" or central constriction)

**ALWAYS ANALYZE XZ VIEW**: The XZ view often shows bean-stage features (central constriction, "peanut" shape) BEFORE they're visible in XY. If XZ shows ANY central narrowing or figure-8 appearance, this suggests bean stage even if XY looks symmetric.

DEVELOPMENTAL STAGES:

EARLY: Elongated oval (~2:1), SYMMETRIC ENDS, both edges CONVEX, NO central constriction in XZ
BEAN: Even SUBTLE end asymmetry OR central constriction/"peanut" shape in XZ view, edges still CONVEX
COMMA: One edge FLAT or curves INWARD (ventral indentation). XZ shows side-by-side lobes (horizontal figure-8)
1.5-FOLD: Body folding back. XZ shows STACKED horizontal layers (two parallel bands, one above the other)
2-FOLD: Body doubled back completely. XZ shows TWO DISTINCT HORIZONTAL LINES with dark gap between
PRETZEL: Tightly coiled, 3+ body segments visible as multiple stacked layers
HATCHED: Worm exited shell

CRITICAL FOR EARLY vs BEAN vs COMMA:
- EARLY: Both ends symmetric AND both edges convex AND no central constriction in XZ
- BEAN: ANY of these: subtle end tapering, central constriction in XZ, "peanut" shape - edges still convex
- COMMA: One edge is flat or curves INWARD (not convex)

CRITICAL FOR COMMA vs FOLD STAGES (examine XZ view carefully):
- COMMA XZ: Lobes are SIDE-BY-SIDE horizontally (peanut/figure-8 from ventral indentation) - single body plane
- 1.5FOLD/2FOLD XZ: Layers are STACKED VERTICALLY (parallel horizontal lines with gap) - body folded back on itself
The difference is orientation: comma = horizontal constriction, fold = vertical stacking/layering

EARLY→BEAN SENSITIVITY: Err on the side of detecting bean early. If you see ANY hint of:
- One end slightly more tapered than the other
- Central narrowing or "waist" in XZ view
- Figure-8 or peanut appearance in any view
Mark as TRANSITIONAL (early→bean) or BEAN with appropriate confidence.

Respond with JSON:
{
  "observed_features": {"shape": "...", "curvature": "...", "shell_status": "...", "emergence": "..."},
  "contrastive_reasoning": {"why_not_previous_stage": "...", "why_not_next_stage": "..."},
  "stage": "early|bean|comma|1.5fold|2fold|pretzel|hatching|hatched|arrested",
  "is_transitional": true/false,
  "transition_between": ["stage1", "stage2"] or null,
  "confidence": 0.0-1.0,
  "reasoning": "Brief explanation"
}""",
            "cache_control": {"type": "ephemeral", "ttl": "1h"}
        }]

    async def _call_claude_with_tools(self, messages: List[Dict]) -> Any:
        """Call Claude API with tools enabled for interleaved reasoning."""
        try:
            response = await asyncio.to_thread(
                self.claude.messages.create,
                model=self.MODEL,
                max_tokens=8000,
                system=self._build_cached_system_prompt(),
                tools=PERCEPTION_TOOLS,
                messages=messages,
            )

            # Log cache metrics
            usage = response.usage
            cache_read = getattr(usage, 'cache_read_input_tokens', 0) or 0
            cache_create = getattr(usage, 'cache_creation_input_tokens', 0) or 0
            if cache_read > 0 or cache_create > 0:
                logger.info(f"Cache: read={cache_read:,} tokens, created={cache_create:,} tokens")

            return response

        except Exception as e:
            logger.error(f"Claude API call with tools failed: {e}")
            raise

    def _build_prompt(
        self,
        image_b64: str,
        session: PerceptionSession,
        timepoint: int = 0,
        top_image_b64: Optional[str] = None,
        side_image_b64: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Build the perception prompt.

        Structure for optimal caching:
        1. Reference images (STATIC) - marked with cache_control for 1h TTL
        2. Dynamic content (timepoint, observations, current image)

        The reference images are identical across all timepoints, so they will
        be cached and reused, dramatically reducing token costs.
        """
        content = []

        # Get available previous timepoints for context
        available_tps = session.get_available_timepoints()
        prev_available = [tp for tp in available_tps if tp < timepoint]

        # 1. Reference images (STATIC - will be cached across timepoints)
        content.append({
            "type": "text",
            "text": "REFERENCE EXAMPLES FOR EACH STAGE:"
        })

        examples = self._load_all_examples()
        for stage in STAGES:
            if stage in examples:
                stage_desc = ""
                if self.example_store:
                    stage_desc = self.example_store.get_stage_description(stage)

                header = f"\n{stage.upper()}"
                if stage_desc:
                    header += f": {stage_desc}"
                content.append({"type": "text", "text": header})

                for example in examples[stage]:
                    content.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": example["image"],
                        }
                    })

        # Mark reference images for caching (1-hour TTL)
        # Everything up to here is STATIC and will be cached across timepoints
        if content:
            content[-1]["cache_control"] = {"type": "ephemeral", "ttl": "1h"}

        # 2. Dynamic content starts here (NOT cached)
        content.append({
            "type": "text",
            "text": f"\n=== ANALYZE EMBRYO AT T{timepoint} ==="
        })

        # 2. Previous observations (last 3)
        recent = session.get_recent_observations(3)
        if recent:
            obs_text = "\nPREVIOUS OBSERVATIONS:\n"
            for obs in recent:
                obs_text += f"- T{obs.timepoint}: {obs.stage}"
                if obs.is_hatching:
                    obs_text += " (hatching in progress)"
                obs_text += "\n"
            content.append({"type": "text", "text": obs_text})

        # 3. Temporal context (for detecting arrested/dead embryos)
        temporal = session.compute_temporal_analysis()
        if temporal:
            temporal_text = f"""
TEMPORAL CONTEXT:
- Current stage: {temporal.current_stage}
- Time at this stage: {temporal.time_in_current_stage_min:.0f} minutes
- Expected duration: {temporal.expected_duration_min or 'N/A'} minutes
- Overtime ratio: {temporal.overtime_ratio:.1f}x (>2x is unusual, >3x is concerning)
"""
            if temporal.is_potentially_arrested:
                temporal_text += f"""
WARNING - POTENTIAL DEVELOPMENTAL ARREST:
- {temporal.arrest_reason}
"""
            content.append({"type": "text", "text": temporal_text})

        # 4. Current image to analyze
        use_separate_images = top_image_b64 is not None and side_image_b64 is not None

        if use_separate_images:
            content.append({
                "type": "text",
                "text": "\n=== CURRENT EMBRYO (T{}) ===\n\n**TOP VIEW**:".format(timepoint)
            })
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": top_image_b64,
                }
            })
            content.append({
                "type": "text",
                "text": "\n**SIDE VIEW**:"
            })
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": side_image_b64,
                }
            })
        else:
            # Send combined three-view image
            content.append({
                "type": "text",
                "text": f"\n=== CURRENT EMBRYO (T{timepoint}) ==="
            })
            content.append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/jpeg",
                    "data": image_b64,
                }
            })

        # 5. Note about available previous timepoints (for view_previous_timepoint tool)
        if prev_available:
            content.append({
                "type": "text",
                "text": f"(Previous timepoints available: T{', T'.join(map(str, prev_available[-5:]))})"
            })

        return content

    def _parse_response(self, response: str) -> PerceptionResult:
        """Parse VLM response into PerceptionResult."""
        try:
            # Try multiple JSON extraction strategies
            data = None

            # Strategy 1: Find JSON in code block (most reliable)
            json_match = re.search(r'```json?\s*(.*?)\s*```', response, re.DOTALL)
            if json_match:
                try:
                    data = json.loads(json_match.group(1))
                except json.JSONDecodeError:
                    pass

            # Strategy 2: Find balanced braces (handles nested content)
            if data is None:
                start = response.find('{')
                if start >= 0:
                    depth = 0
                    end = start
                    for i, c in enumerate(response[start:], start):
                        if c == '{':
                            depth += 1
                        elif c == '}':
                            depth -= 1
                            if depth == 0:
                                end = i + 1
                                break
                    try:
                        data = json.loads(response[start:end])
                    except json.JSONDecodeError:
                        pass

            # Strategy 3: Try parsing the whole response
            if data is None:
                try:
                    data = json.loads(response.strip())
                except json.JSONDecodeError:
                    pass

            if data is None:
                raise ValueError("No JSON found in response")

            stage = data.get("stage", "early")
            if stage not in STAGES:
                stage = "early"

            # Parse observed features
            observed_features = None
            features_data = data.get("observed_features", {})
            if features_data:
                observed_features = ObservedFeatures(
                    shape=features_data.get("shape", ""),
                    curvature=features_data.get("curvature", ""),
                    shell_status=features_data.get("shell_status", ""),
                    body_segments_visible=features_data.get("body_segments_visible", ""),
                    emergence=features_data.get("emergence", ""),
                )
                logger.info(
                    f"Observed: shape={observed_features.shape}, "
                    f"curve={observed_features.curvature}, "
                    f"shell={observed_features.shell_status}, "
                    f"emergence={observed_features.emergence}"
                )

            # Parse contrastive reasoning
            contrastive = None
            contrast_data = data.get("contrastive_reasoning", {})
            if contrast_data:
                contrastive = ContrastiveReasoning(
                    why_not_previous_stage=contrast_data.get("why_not_previous_stage", ""),
                    why_not_next_stage=contrast_data.get("why_not_next_stage", ""),
                )
                logger.debug(
                    f"Contrastive: not_prev={contrastive.why_not_previous_stage[:50]}, "
                    f"not_next={contrastive.why_not_next_stage[:50]}"
                )

            # Parse transitional status
            is_transitional = data.get("is_transitional", False)
            transition_between = data.get("transition_between")

            # Determine is_hatching from stage
            is_hatching = (stage == "hatching")
            raw_confidence = float(data.get("confidence", 0.5))
            reasoning = data.get("reasoning", "")

            # Calibrate confidence based on hedging language and transitional status
            confidence = self._calibrate_confidence(raw_confidence, reasoning, is_transitional)

            if is_transitional and transition_between:
                logger.info(f"TRANSITIONAL between {transition_between}")

            return PerceptionResult(
                stage=stage,
                is_hatching=is_hatching,
                confidence=confidence,
                reasoning=reasoning,
                observed_features=observed_features,
                contrastive_reasoning=contrastive,
                is_transitional=is_transitional,
                transition_between=transition_between,
                should_stop=(stage == "hatched"),
            )

        except Exception as e:
            logger.warning(f"Failed to parse response: {e}")
            logger.debug(f"Raw response: {response[:500]}")

            # Return safe default
            return PerceptionResult(
                stage="early",
                is_hatching=False,
                confidence=0.0,
                reasoning=f"Parse error: {e}",
                should_stop=False,
            )

    def _calibrate_confidence(
        self,
        raw_confidence: float,
        reasoning: str,
        is_transitional: bool,
    ) -> float:
        """
        Calibrate confidence based on hedging language and transitional status.

        Reduces confidence when the VLM uses hedging words (indicating uncertainty)
        or when the observation is marked as transitional.

        Parameters
        ----------
        raw_confidence : float
            Original confidence from VLM (0.0-1.0)
        reasoning : str
            VLM's reasoning text to check for hedging
        is_transitional : bool
            Whether this is a transitional observation

        Returns
        -------
        float
            Calibrated confidence (0.3-1.0)
        """
        HEDGING_WORDS = [
            "subtle", "slight", "beginning", "partial", "maybe", "hint",
            "possibly", "appears to", "seems", "might", "could be",
            "starting to", "early signs", "not quite", "borderline",
        ]

        reasoning_lower = reasoning.lower()
        hedging_count = sum(1 for word in HEDGING_WORDS if word in reasoning_lower)

        # Apply penalty for hedging language
        penalty = min(0.25, hedging_count * 0.06)

        # Additional penalty for transitional observations
        if is_transitional:
            penalty += 0.10

        calibrated = max(0.3, raw_confidence - penalty)

        if penalty > 0:
            logger.debug(
                f"Confidence calibration: {raw_confidence:.2f} -> {calibrated:.2f} "
                f"(hedging={hedging_count}, transitional={is_transitional})"
            )

        return calibrated
