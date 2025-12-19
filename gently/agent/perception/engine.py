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

from .session import Observation, PerceptionResult, PerceptionSession
from .example_store import ExampleStore

logger = logging.getLogger(__name__)

# Stages in order
STAGES = ["early", "bean", "comma", "1.5fold", "2fold", "3fold", "hatched"]


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

        # Cache loaded examples
        self._examples_cache: Optional[Dict[str, List[str]]] = None

    def _load_all_examples(self) -> Dict[str, List[str]]:
        """Load all stage examples (cached)."""
        if self._examples_cache is not None:
            return self._examples_cache

        if not self.example_store:
            return {}

        examples = {}
        for stage in STAGES:
            stage_examples = self.example_store.get_stage_examples(stage, max_examples=2)
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
        Perceive the current image.

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
            Stage classification and hatching status
        """
        # Build prompt
        content = self._build_prompt(image_b64, session)

        # Call VLM
        response = await self._call_claude(content)

        # Parse response
        result = self._parse_response(response)

        logger.info(
            f"[{session.embryo_id}] T{timepoint}: "
            f"stage={result.stage}, hatching={result.is_hatching}, "
            f"confidence={result.confidence:.0%}"
        )

        return result

    def _build_prompt(
        self,
        image_b64: str,
        session: PerceptionSession,
    ) -> List[Dict[str, Any]]:
        """Build the perception prompt."""
        content = []

        # 1. Instructions
        content.append({
            "type": "text",
            "text": """You are analyzing a C. elegans embryo in microscopy images.

Each image shows TWO VIEWS side-by-side:
- LEFT: TOP view (looking down at the embryo)
- RIGHT: SIDE view (looking at the embryo from the side)

Both views together give you 3D information about the embryo's morphology.

Your task: Identify the developmental stage and whether hatching is occurring.

DEVELOPMENTAL STAGES (in order):

EARLY (gastrulation):
- Oval/elliptical shape, relatively uniform cellular mass
- Grainy texture showing many individual cells (~100+ cell stage)
- No elongation or asymmetry
- Side view shows compact, rounded blob

BEAN:
- Slightly more elongated than early stage
- Beginning asymmetry - one end slightly narrower
- Subtle kidney/bean shape forming
- Still mostly a compact cellular mass

COMMA:
- Clear elongation - distinctly longer shape
- Pronounced bend/curve forming C-shape
- Body axis now established
- Side view shows curvature

1.5-FOLD:
- Embryo starting to fold back on itself
- Two distinct regions visible where body is doubling
- Partial fold - about 1.5x original length folded

2-FOLD:
- Compact "pretzel" shape
- Embryo folded completely back on itself
- Two parallel body segments visible in top view
- Twitching/movement may begin at this stage

3-FOLD:
- Tightly coiled - three body segments visible
- Nearly fully developed worm, coiled within eggshell
- Active movement/twitching expected
- Complex folded morphology

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
                content.append({"type": "text", "text": f"\n{stage.upper()} stage:"})
                for example_b64 in examples[stage]:
                    content.append({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": example_b64,
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

        # 5. Output format
        content.append({
            "type": "text",
            "text": """
Compare the current image to the reference examples above.

Respond with JSON:
{
  "stage": "early" | "bean" | "comma" | "1.5fold" | "2fold" | "3fold" | "hatched",
  "is_hatching": true/false,
  "confidence": 0.0-1.0,
  "reasoning": "Brief explanation"
}

Notes:
- is_hatching means the worm is actively emerging (breach visible, worm exiting)
- For "hatched", the worm should be fully outside the shell
- Be honest about confidence - if uncertain, say so
"""
        })

        return content

    async def _call_claude(self, content: List[Dict]) -> str:
        """Call Claude API."""
        try:
            response = await asyncio.to_thread(
                self.claude.messages.create,
                model=self.MODEL,
                max_tokens=500,
                messages=[{"role": "user", "content": content}],
            )

            for block in response.content:
                if hasattr(block, "text"):
                    return block.text

            return ""

        except Exception as e:
            logger.error(f"Claude API call failed: {e}")
            raise

    def _parse_response(self, response: str) -> PerceptionResult:
        """Parse VLM response into PerceptionResult."""
        try:
            # Extract JSON
            json_match = re.search(r'\{[^{}]*\}', response, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
            else:
                # Try to find JSON in code block
                json_match = re.search(r'```json?\s*(.*?)\s*```', response, re.DOTALL)
                if json_match:
                    data = json.loads(json_match.group(1))
                else:
                    raise ValueError("No JSON found in response")

            stage = data.get("stage", "early")
            if stage not in STAGES:
                stage = "early"

            is_hatching = data.get("is_hatching", False)
            confidence = float(data.get("confidence", 0.5))
            reasoning = data.get("reasoning", "")

            return PerceptionResult(
                stage=stage,
                is_hatching=is_hatching,
                confidence=confidence,
                reasoning=reasoning,
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
