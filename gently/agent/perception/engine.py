"""
Simple Perception Engine.

Show reference examples, show current image, ask what stage.
No probability distributions, no tiered models, no complex parsing.
"""

import asyncio
import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import anthropic

from .session import (
    Observation,
    ObservedFeatures,
    ContrastiveReasoning,
    ReasoningStep,
    ReasoningTrace,
    PerceptionResult,
    PerceptionSession,
)
from .example_store import ExampleStore
from .stages import STAGES

logger = logging.getLogger(__name__)

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
    {
        "name": "view_reference_example",
        "description": (
            "View a reference example image for a specific developmental stage. Use this when "
            "you need to compare the current embryo's morphology against a known example of a "
            "particular stage.\n\n"
            "Example inputs:\n"
            "- {\"stage\": \"comma\", \"reason\": \"Verifying if current curve matches comma stage\"}\n"
            "- {\"stage\": \"pretzel\", \"reason\": \"Comparing coil tightness with known pretzel\"}"
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "stage": {
                    "type": "string",
                    "enum": ["early", "comma", "1.5fold", "pretzel", "hatching", "hatched"],
                    "description": "The developmental stage to view an example of",
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


class PerceptionEngine:
    """
    Simple perception engine.

    Just: show examples, show image, ask what stage.
    """

    MODEL = "claude-sonnet-4-5-20250929"

    def __init__(
        self,
        claude_client: anthropic.Anthropic,
        example_store: Optional[ExampleStore] = None,
        examples_path: Optional[Path] = None,
    ):
        self.claude = claude_client

        # Load examples if provided
        if example_store:
            self.example_store = example_store
        elif examples_path:
            self.example_store = ExampleStore(examples_path)
        else:
            self.example_store = None

        # Cache loaded examples (with descriptions)
        self._examples_cache: Optional[Dict[str, List[Dict]]] = None

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
    ) -> PerceptionResult:
        """
        Perceive the current image using interleaved reasoning.

        The VLM can request additional context (previous timepoints, reference
        examples) via tool use when uncertain about the classification.

        Parameters
        ----------
        image_b64 : str
            Base64-encoded current image
        session : PerceptionSession
            Session with previous observations
        timepoint : int
            Current timepoint number

        Returns
        -------
        PerceptionResult
            Stage classification and hatching status with full reasoning trace
        """
        # Store current image for potential future reference
        session.store_image(timepoint, image_b64)

        # Build initial prompt
        content = self._build_prompt(image_b64, session, timepoint)

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
                    content=text_response[:500],  # Truncate for storage
                ))

                # Parse and return
                result = self._parse_response(text_response)
                return result, trace

            # Handle tool use
            if response.stop_reason == "tool_use":
                # Build assistant message with the response
                assistant_content = []
                for block in response.content:
                    if block.type == "text":
                        # Record initial analysis
                        trace.add_step(ReasoningStep(
                            step_type="initial_analysis",
                            content=block.text[:500],
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

        elif tool_name == "view_reference_example":
            stage = tool_input.get("stage", "early")
            reason = tool_input.get("reason", "")

            examples = self._load_all_examples()
            if stage in examples and examples[stage]:
                example = examples[stage][0]  # Get first example
                logger.info(f"Tool: Providing reference example for {stage}")

                content = [
                    {"type": "text", "text": f"Here is a reference example of {stage.upper()} stage (requested because: {reason}):"},
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
                summary = f"Showed {stage} reference"
                return content, summary, None, f"reference_{stage}"
            else:
                content = [{"type": "text", "text": f"No reference example available for stage: {stage}"}]
                summary = f"No {stage} reference"
                return content, summary, None, None

        else:
            content = [{"type": "text", "text": f"Unknown tool: {tool_name}"}]
            return content, "Unknown tool", None, None

    async def _call_claude_with_tools(self, messages: List[Dict]) -> Any:
        """Call Claude API with tools enabled for interleaved reasoning."""
        try:
            response = await asyncio.to_thread(
                self.claude.messages.create,
                model=self.MODEL,
                max_tokens=8000,
                system=(
                    "You are an expert microscopy perception system analyzing C. elegans "
                    "embryo development. You have access to tools that let you request "
                    "additional context when uncertain about a classification. Use tools "
                    "judiciously - only when genuinely uncertain and additional context "
                    "would help. When confident, provide your classification directly."
                ),
                tools=PERCEPTION_TOOLS,
                messages=messages,
            )
            return response

        except Exception as e:
            logger.error(f"Claude API call with tools failed: {e}")
            raise

    def _build_prompt(
        self,
        image_b64: str,
        session: PerceptionSession,
        timepoint: int = 0,
    ) -> List[Dict[str, Any]]:
        """Build the perception prompt."""
        content = []

        # Get available previous timepoints for context
        available_tps = session.get_available_timepoints()
        prev_available = [tp for tp in available_tps if tp < timepoint]

        # 1. Instructions
        content.append({
            "type": "text",
            "text": f"""You are analyzing a C. elegans embryo in microscopy images.
Current timepoint: T{timepoint}

Each image shows TWO VIEWS side-by-side:
- LEFT: TOP view (looking down at the embryo)
- RIGHT: SIDE view (looking at the embryo from the side)

Both views together give you 3D information about the embryo's morphology.

Your task: Identify the developmental stage and whether hatching is occurring.

DEVELOPMENTAL STAGES (in order):

EARLY (gastrulation through early morphogenesis):
- Oval/elliptical shape, relatively uniform cellular mass
- Grainy texture showing many individual cells (~100+ cell stage)
- No clear C-curve yet (may have subtle asymmetry)
- Side view shows compact, rounded or slightly elongated blob

COMMA:
- Clear elongation - distinctly longer shape
- Pronounced bend/curve forming C-shape
- Body axis now established, head/tail distinguishable
- Side view shows curvature

1.5-FOLD:
- Embryo starting to fold back on itself
- Body clearly longer than egg width
- Partial fold - about 1.5x original length folded

PRETZEL:
- Tightly coiled "pretzel" shape
- 2-3 body segments visible
- Maximum compaction within eggshell
- Twitching/movement may be visible

HATCHED:
- Worm has exited the eggshell
- Elongated worm-like body clearly visible
- No longer contained in oval eggshell shape

REFERENCE EXAMPLES:
"""
        })

        # 2. Reference examples for each stage
        examples = self._load_all_examples()
        for stage in STAGES:
            if stage in examples:
                # Get stage description from metadata
                stage_desc = ""
                if self.example_store:
                    stage_desc = self.example_store.get_stage_description(stage)

                header = f"\n{stage.upper()} stage"
                if stage_desc:
                    header += f": {stage_desc}"
                content.append({"type": "text", "text": header})

                for example in examples[stage]:
                    # Add description before each image if available
                    if example.get("description"):
                        content.append({
                            "type": "text",
                            "text": f"  - {example['description']}"
                        })
                    content.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": example["image"],
                        }
                    })

        # Mark static content for caching (instructions + reference images)
        # Dynamic content (observations, current image) follows - not cached
        if content:
            content[-1]["cache_control"] = {"type": "ephemeral"}

        # 3. Previous observations (last 3)
        recent = session.get_recent_observations(3)
        if recent:
            obs_text = "\nPREVIOUS OBSERVATIONS:\n"
            for obs in recent:
                obs_text += f"- T{obs.timepoint}: {obs.stage}"
                if obs.is_hatching:
                    obs_text += " (hatching in progress)"
                obs_text += "\n"
            content.append({"type": "text", "text": obs_text})

        # 3.5. Temporal context (for detecting arrested/dead embryos)
        temporal = session.compute_temporal_analysis()
        if temporal:
            temporal_text = f"""
TEMPORAL CONTEXT:
- Current stage: {temporal.current_stage}
- Time at this stage: {temporal.time_in_current_stage_min:.0f} minutes
- Expected duration: {temporal.expected_duration_min or 'N/A'} minutes
- Overtime ratio: {temporal.overtime_ratio:.1f}x (>2x is unusual, >3x is concerning)
- Observations at this stage: {temporal.observations_in_current_stage}
"""
            if temporal.is_potentially_arrested:
                temporal_text += f"""
WARNING - POTENTIAL DEVELOPMENTAL ARREST:
- {temporal.arrest_reason}
- If the embryo shows NO visible progression or morphological change compared to
  previous timepoints, consider classifying as "arrested"
- Look for signs of degradation, fragmentation, or abnormal texture
"""
            content.append({"type": "text", "text": temporal_text})

        # 4. Current image
        content.append({
            "type": "text",
            "text": "\nCURRENT IMAGE TO ANALYZE:"
        })
        content.append({
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": image_b64,
            }
        })

        # 5. Output format - force explicit morphological description before classification
        content.append({
            "type": "text",
            "text": """
CRITICAL: Do NOT jump to a classification. Follow this process:

STEP 1 - DESCRIBE what you see in the CURRENT IMAGE:
- What is the overall shape? (oval, curved, folded, coiled, elongated worm)
- Is there a C-curve? How pronounced?
- Is the shell intact or breached?
- Can you see body segments? How many?
- Is any part of the embryo OUTSIDE the shell boundary?

STEP 2 - CONTRASTIVE REASONING:
- Why is this NOT the previous stage? (what feature rules it out?)
- Why is this NOT the next stage? (what feature is missing?)

STEP 3 - CLASSIFY with appropriate confidence:
- If clearly one stage: high confidence (0.8-1.0)
- If transitional between stages: note this, medium confidence (0.5-0.7)
- If uncertain: low confidence (<0.5)

Respond with JSON:
{
  "observed_features": {
    "shape": "describe the actual shape you see",
    "curvature": "none/slight/pronounced C-curve/folded/tightly coiled",
    "shell_status": "intact oval boundary / irregular boundary / breached / absent",
    "body_segments_visible": "none/1/2/3/coiled mass",
    "emergence": "none / partial (part outside shell) / complete (fully out)"
  },
  "contrastive_reasoning": {
    "why_not_previous_stage": "feature that rules out earlier stage",
    "why_not_next_stage": "feature missing for later stage"
  },
  "stage": "early" | "comma" | "1.5fold" | "pretzel" | "hatching" | "hatched" | "arrested",
  "is_transitional": true/false,
  "transition_between": ["stage1", "stage2"] or null,
  "confidence": 0.0-1.0,
  "reasoning": "Brief explanation grounded in observed features"
}

KEY DISCRIMINATORS:
- EARLY vs COMMA: Does it have a clear C-curve? No curve = early, clear curve = comma
- COMMA vs 1.5FOLD: Is body folding back on itself? Just curved = comma, folding back = 1.5fold
- 1.5FOLD vs PRETZEL: How tight is the coil? Partial fold = 1.5fold, tight coil filling shell = pretzel
- PRETZEL vs HATCHING: Is shell intact? Intact = pretzel, breached with part outside = hatching
- HATCHING vs HATCHED: Is any part still in shell? Part inside = hatching, fully out = hatched
- ANY STAGE vs ARRESTED: Is there visible progression over time? If embryo looks identical across many
  timepoints with NO morphological change, and/or shows degradation/fragmentation/abnormal texture,
  classify as "arrested". Only use if TEMPORAL CONTEXT shows significant overtime.

IMPORTANT: Describe what you ACTUALLY SEE, not what you expect based on previous observations.

TOOLS AVAILABLE (use if uncertain):
If you are uncertain about the classification, you can request additional context:

1. view_previous_timepoint - See how the embryo looked at an earlier timepoint to assess progression
   Use when: You're unsure if there's been a stage transition, or need to compare morphology changes

2. view_reference_example - See a reference example of a specific stage
   Use when: You want to directly compare the current embryo against a known example

Only use tools when genuinely uncertain. If the classification is clear, proceed directly to the JSON response.
"""
        })

        # Add note about available previous timepoints
        if prev_available:
            content.append({
                "type": "text",
                "text": f"\n(Previous timepoint images available: T{', T'.join(map(str, prev_available[-5:]))})"
            })

        return content

    async def _call_claude(self, content: List[Dict]) -> str:
        """Call Claude API with extended thinking enabled."""
        try:
            response = await asyncio.to_thread(
                self.claude.messages.create,
                model=self.MODEL,
                max_tokens=16000,  # Extended thinking requires higher max_tokens
                thinking={
                    "type": "enabled",
                    "budget_tokens": 8000,  # Allow thorough reasoning for stage classification
                },
                messages=[{"role": "user", "content": content}],
            )

            # With extended thinking, response has thinking blocks followed by text
            text_response = ""
            thinking_summary = ""

            for block in response.content:
                if block.type == "thinking":
                    # Log thinking for debugging (truncated)
                    thinking_summary = block.thinking[:200] if block.thinking else ""
                    logger.debug(f"VLM thinking: {thinking_summary}...")
                elif block.type == "text":
                    text_response = block.text

            return text_response

        except Exception as e:
            logger.error(f"Claude API call failed: {e}")
            raise

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
            confidence = float(data.get("confidence", 0.5))
            reasoning = data.get("reasoning", "")

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
