"""
Verification subagent system for multi-phase perception.

This module handles Phase 3 of the perceive-think-verify loop:
- Spawning parallel verification subagents (separate Claude API calls)
- Running focused stage comparisons
- Aggregating subagent results via confidence-weighted voting
"""

import asyncio
import json
import logging
import re
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import anthropic
import numpy as np

from ...settings import settings
from .session import (
    CandidateStage,
    StageComparison,
    SubagentResult,
    SubagentTrace,
    VerificationAggregation,
    ReasoningStep,
)
from .example_store import ExampleStore
from gently.imaging import render_volume_view

logger = logging.getLogger(__name__)


# Subagent uses a faster, cheaper model for focused comparisons
SUBAGENT_MODEL = settings.models.fast


# Limited tools for subagents (focused comparison only)
def get_subagent_tools(stage_a: str, stage_b: str, use_3d: bool) -> List[Dict]:
    """
    Get tools available to a verification subagent.

    Limited to reference examples for the two comparison stages only.
    """
    tools = [
        {
            "name": "view_reference_example",
            "description": (
                f"View a reference example for {stage_a.upper()} or {stage_b.upper()} stage. "
                "Use this to compare the current embryo against known examples of these specific stages."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "stage": {
                        "type": "string",
                        "enum": [stage_a, stage_b],
                        "description": f"Which stage reference to view ({stage_a} or {stage_b})",
                    },
                    "reason": {
                        "type": "string",
                        "description": "Brief explanation of why you need to see this reference",
                    },
                },
                "required": ["stage", "reason"],
            },
        },
    ]

    if use_3d:
        tools.append({
            "name": "view_reference_example_3d",
            "description": (
                f"View a 3D reference for {stage_a.upper()} or {stage_b.upper()} stage from a specific angle. "
                "Use this for detailed morphological comparison when 2D views are ambiguous."
            ),
            "input_schema": {
                "type": "object",
                "properties": {
                    "stage": {
                        "type": "string",
                        "enum": [stage_a, stage_b],
                        "description": f"Which stage 3D reference to view ({stage_a} or {stage_b})",
                    },
                    "rotation_x": {
                        "type": "number",
                        "description": "Rotation around X axis (-90 to 90 degrees)",
                        "minimum": -90,
                        "maximum": 90,
                    },
                    "rotation_y": {
                        "type": "number",
                        "description": "Rotation around Y axis (-180 to 180 degrees)",
                        "minimum": -180,
                        "maximum": 180,
                    },
                    "reason": {
                        "type": "string",
                        "description": "Brief explanation of why you need this 3D view",
                    },
                },
                "required": ["stage", "reason"],
            },
        })

    return tools


def build_subagent_prompt(
    comparison: StageComparison,
    image_b64: str,
    initial_stage: str,
    initial_confidence: float,
    key_question: str,
) -> str:
    """Build the system prompt for a verification subagent."""
    return f"""You are a verification subagent focused on a SINGLE comparison task.

TASK: Determine if this embryo is more likely {comparison.stage_a.upper()} or {comparison.stage_b.upper()}.

CONTEXT:
- Initial perception classified this as: {initial_stage} (confidence: {initial_confidence:.0%})
- Key question to resolve: {key_question}
- Comparison reason: {comparison.reason}

Your job is to compare the current embryo image against reference examples of {comparison.stage_a.upper()} and {comparison.stage_b.upper()} and make a definitive choice between these TWO options.

PROCESS:
1. Examine the current embryo image carefully
2. Use view_reference_example to see what {comparison.stage_a} and {comparison.stage_b} look like
3. Compare key morphological features
4. Make a decision with confidence

After examining references, respond with ONLY this JSON:
{{
  "preferred_stage": "{comparison.stage_a}" or "{comparison.stage_b}",
  "confidence": 0.5-1.0,
  "evidence_for_preferred": ["feature 1 matching preferred", "feature 2 matching preferred"],
  "evidence_against_other": ["feature 1 that rules out other stage", "feature 2 that rules out other stage"]
}}

CONFIDENCE GUIDE:
- 0.5: Truly uncertain, could be either
- 0.6-0.7: Slight preference based on subtle features
- 0.8-0.9: Clear preference based on visible features
- 1.0: Definitive match with no ambiguity

DO NOT respond with anything other than the JSON. No explanations outside the JSON."""


class VerificationEngine:
    """
    Handles verification subagent spawning and result aggregation.

    This is Phase 3 of the perceive-think-verify loop.
    """

    def __init__(
        self,
        claude_client: anthropic.Anthropic,
        example_store: Optional[ExampleStore] = None,
    ):
        self.claude = claude_client
        self.example_store = example_store

    async def run_verification(
        self,
        comparisons: List[StageComparison],
        image_b64: str,
        initial_stage: str,
        initial_confidence: float,
        key_question: str,
        volume: Optional[np.ndarray] = None,
        embryo_id: Optional[str] = None,
        timepoint: Optional[int] = None,
    ) -> Tuple[VerificationAggregation, List[SubagentTrace]]:
        """
        Run parallel verification subagents.

        Parameters
        ----------
        comparisons : List[StageComparison]
            Stage comparisons to verify (max 3)
        image_b64 : str
            Base64-encoded current image
        initial_stage : str
            Initial classification from Phase 1
        initial_confidence : float
            Initial confidence from Phase 1
        key_question : str
            The key uncertainty being resolved
        volume : np.ndarray, optional
            3D volume for comparisons that need 3D views
        embryo_id : str, optional
            Embryo identifier for logging context
        timepoint : int, optional
            Timepoint for logging context

        Returns
        -------
        Tuple[VerificationAggregation, List[SubagentTrace]]
            Aggregated result and individual subagent traces
        """
        # Run subagents in parallel
        tasks = [
            self._run_single_subagent(
                comparison=comp,
                image_b64=image_b64,
                initial_stage=initial_stage,
                initial_confidence=initial_confidence,
                key_question=key_question,
                volume=volume if comp.use_3d else None,
                embryo_id=embryo_id,
                timepoint=timepoint,
            )
            for comp in comparisons
        ]

        traces = await asyncio.gather(*tasks)

        # Extract results from traces
        subagent_results = [
            trace.result for trace in traces
            if trace.result is not None
        ]

        # Aggregate results
        aggregation = self._aggregate_results(
            initial_stage=initial_stage,
            initial_confidence=initial_confidence,
            subagent_results=subagent_results,
        )

        return aggregation, list(traces)

    async def _run_single_subagent(
        self,
        comparison: StageComparison,
        image_b64: str,
        initial_stage: str,
        initial_confidence: float,
        key_question: str,
        volume: Optional[np.ndarray] = None,
        embryo_id: Optional[str] = None,
        timepoint: Optional[int] = None,
    ) -> SubagentTrace:
        """
        Run a single verification subagent.

        Returns
        -------
        SubagentTrace
            Trace of the subagent execution with result
        """
        trace = SubagentTrace(
            comparison=comparison,
            started_at=datetime.now(),
        )

        try:
            # Build prompt and tools
            system_prompt = build_subagent_prompt(
                comparison=comparison,
                image_b64=image_b64,
                initial_stage=initial_stage,
                initial_confidence=initial_confidence,
                key_question=key_question,
            )

            tools = get_subagent_tools(
                stage_a=comparison.stage_a,
                stage_b=comparison.stage_b,
                use_3d=comparison.use_3d,
            )

            # Build initial message with the current image
            messages = [{
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": f"Here is the embryo image to classify as either {comparison.stage_a.upper()} or {comparison.stage_b.upper()}:",
                    },
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": image_b64,
                        },
                    },
                    {
                        "type": "text",
                        "text": "Use the tools to view reference examples, then make your decision.",
                    },
                ],
            }]

            # Run reasoning loop with limited iterations
            result = await self._subagent_reasoning_loop(
                system_prompt=system_prompt,
                tools=tools,
                messages=messages,
                comparison=comparison,
                volume=volume,
                trace=trace,
                max_iterations=3,
                embryo_id=embryo_id,
                timepoint=timepoint,
            )

            trace.result = result

        except Exception as e:
            ctx = f"[{embryo_id}] T{timepoint}: " if embryo_id else ""
            logger.error(f"{ctx}Subagent failed for {comparison.stage_a} vs {comparison.stage_b}: {e}")
            trace.error = str(e)

        trace.complete()
        return trace

    async def _subagent_reasoning_loop(
        self,
        system_prompt: str,
        tools: List[Dict],
        messages: List[Dict],
        comparison: StageComparison,
        volume: Optional[np.ndarray],
        trace: SubagentTrace,
        max_iterations: int = 3,
        embryo_id: Optional[str] = None,
        timepoint: Optional[int] = None,
    ) -> Optional[SubagentResult]:
        """Run the subagent's tool-use reasoning loop."""

        for iteration in range(max_iterations):
            # Call Claude
            response = await asyncio.to_thread(
                self.claude.messages.create,
                model=SUBAGENT_MODEL,
                max_tokens=2000,
                system=system_prompt,
                tools=tools,
                messages=messages,
            )

            # Check for final response
            if response.stop_reason == "end_turn":
                text_response = ""
                for block in response.content:
                    if block.type == "text":
                        text_response = block.text

                trace.steps.append(ReasoningStep(
                    step_type="final_decision",
                    content=text_response,
                ))

                return self._parse_subagent_response(text_response, comparison)

            # Handle tool use
            if response.stop_reason == "tool_use":
                assistant_content = []

                for block in response.content:
                    if block.type == "text":
                        trace.steps.append(ReasoningStep(
                            step_type="analysis",
                            content=block.text,
                        ))
                        assistant_content.append({
                            "type": "text",
                            "text": block.text,
                        })
                    elif block.type == "tool_use":
                        trace.steps.append(ReasoningStep(
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

                # Process tool calls
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        result_content, summary = self._handle_subagent_tool(
                            tool_name=block.name,
                            tool_input=block.input,
                            comparison=comparison,
                            volume=volume,
                        )

                        trace.steps.append(ReasoningStep(
                            step_type="tool_result",
                            content=summary,
                            tool_name=block.name,
                            tool_result_summary=summary,
                        ))

                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result_content,
                        })

                messages.append({"role": "user", "content": tool_results})

        ctx = f"[{embryo_id}] T{timepoint}: " if embryo_id else ""
        logger.warning(f"{ctx}Subagent max iterations reached for {comparison.stage_a} vs {comparison.stage_b}")
        return None

    def _handle_subagent_tool(
        self,
        tool_name: str,
        tool_input: Dict,
        comparison: StageComparison,
        volume: Optional[np.ndarray],
    ) -> Tuple[List[Dict], str]:
        """Handle a tool call from a subagent."""

        if tool_name == "view_reference_example":
            stage = tool_input.get("stage")
            reason = tool_input.get("reason", "")

            # Validate stage is one of the comparison stages
            if stage not in [comparison.stage_a, comparison.stage_b]:
                return [{"type": "text", "text": f"Invalid stage: {stage}. Only {comparison.stage_a} or {comparison.stage_b} allowed."}], f"Invalid stage {stage}"

            if self.example_store:
                examples = self.example_store.get_stage_examples_with_descriptions(stage, max_examples=1)
                if examples:
                    example = examples[0]
                    content = [
                        {"type": "text", "text": f"Reference example of {stage.upper()} (requested: {reason}):"},
                    ]
                    if example.get("description"):
                        content.append({"type": "text", "text": f"Description: {example['description']}"})
                    content.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": example["image"],
                        }
                    })
                    return content, f"Showed {stage} reference"

            return [{"type": "text", "text": f"No reference available for {stage}"}], f"No {stage} reference"

        elif tool_name == "view_reference_example_3d":
            stage = tool_input.get("stage")
            rotation_x = tool_input.get("rotation_x", 0)
            rotation_y = tool_input.get("rotation_y", 0)
            reason = tool_input.get("reason", "")

            if stage not in [comparison.stage_a, comparison.stage_b]:
                return [{"type": "text", "text": f"Invalid stage: {stage}"}], f"Invalid stage {stage}"

            if self.example_store and self.example_store.has_volume(stage):
                vol = self.example_store.get_stage_volume(stage)
                if vol is not None:
                    rendered_b64 = render_volume_view(vol, rotation_x, rotation_y)
                    content = [
                        {"type": "text", "text": f"3D view of {stage.upper()} reference (rx={rotation_x}°, ry={rotation_y}°). {reason}"},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": rendered_b64,
                            }
                        }
                    ]
                    return content, f"Showed {stage} 3D at ({rotation_x}, {rotation_y})"

            return [{"type": "text", "text": f"No 3D volume available for {stage}"}], f"No {stage} 3D"

        return [{"type": "text", "text": f"Unknown tool: {tool_name}"}], "Unknown tool"

    def _parse_subagent_response(
        self,
        response: str,
        comparison: StageComparison,
    ) -> Optional[SubagentResult]:
        """Parse subagent JSON response."""
        try:
            # Try to extract JSON
            data = None

            # Strategy 1: Code block
            json_match = re.search(r'```json?\s*(.*?)\s*```', response, re.DOTALL)
            if json_match:
                try:
                    data = json.loads(json_match.group(1))
                except json.JSONDecodeError:
                    pass

            # Strategy 2: Find balanced braces
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

            if data is None:
                logger.warning(f"No JSON in subagent response: {response[:200]}")
                return None

            preferred = data.get("preferred_stage")
            if preferred not in [comparison.stage_a, comparison.stage_b]:
                logger.warning(f"Invalid preferred_stage: {preferred}")
                return None

            return SubagentResult(
                preferred_stage=preferred,
                confidence=min(1.0, max(0.5, float(data.get("confidence", 0.5)))),
                evidence_for_preferred=data.get("evidence_for_preferred", []),
                evidence_against_other=data.get("evidence_against_other", []),
            )

        except Exception as e:
            logger.warning(f"Failed to parse subagent response: {e}")
            return None

    def _aggregate_results(
        self,
        initial_stage: str,
        initial_confidence: float,
        subagent_results: List[SubagentResult],
    ) -> VerificationAggregation:
        """
        Aggregate subagent results using confidence-weighted voting.

        The initial assessment also contributes to the vote.
        """
        stage_votes: Dict[str, float] = defaultdict(float)

        # Initial assessment contributes
        stage_votes[initial_stage] += initial_confidence

        # Each subagent votes with confidence weight
        for result in subagent_results:
            stage_votes[result.preferred_stage] += result.confidence

        # Determine winning stage
        if not stage_votes:
            winning_stage = initial_stage
        else:
            winning_stage = max(stage_votes, key=lambda k: stage_votes[k])

        # Calculate aggregated confidence
        total_votes = sum(stage_votes.values())
        if total_votes > 0:
            aggregated_confidence = stage_votes[winning_stage] / total_votes
        else:
            aggregated_confidence = initial_confidence

        # Check if subagents agree
        subagent_stages = [r.preferred_stage for r in subagent_results]
        subagents_agree = len(set(subagent_stages)) <= 1 if subagent_stages else True

        # Did verification change the result?
        should_override = winning_stage != initial_stage

        return VerificationAggregation(
            stage_votes=dict(stage_votes),
            winning_stage=winning_stage,
            aggregated_confidence=aggregated_confidence,
            subagents_agree=subagents_agree,
            should_override_initial=should_override,
        )
