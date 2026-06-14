"""
Hatching detector — Claude-vision check for whether an embryo has hatched.

Useful for TestEmbryos that lack the nuclear marker the standard perception
pipeline trains on. The dopaminergic-signal detector already returns
``has_hatched`` as part of its richer schema; this is a lighter-weight
yes/no for use cases where structure / intensity assessment isn't needed.

The verdict comes back as a forced tool call (``tool_choice`` pins the model
to ``record_hatching``), so the structured fields arrive already parsed as
``block.input`` — no JSON-from-prose scraping, no silent-default parse layer.
"""

import asyncio
import logging
import time
from typing import Any

import numpy as np

from .base import Detector, DetectorResult
from .dopaminergic_signal import _volume_to_b64

logger = logging.getLogger(__name__)


_HATCHING_PROMPT = """\
You are observing a C. elegans embryo on a microscope. Decide whether the embryo has HATCHED,
then record your decision with the record_hatching tool.

A HATCHED embryo:
- Has visibly broken out of the eggshell
- Often shows the elongated larva moving outside the eggshell outline
- The eggshell itself may be visible as a broken / collapsed outline

An UNHATCHED embryo:
- Is still contained within an intact eggshell
- May be at any pre-hatching stage (bean, comma, 1.5-fold, 2-fold, pretzel)

Default to has_hatched=false unless you are confident. Don't over-call hatching.
"""


# Forced tool schema — the model is pinned to this via tool_choice, so the
# fields come back as a validated dict on the tool_use block. The conservative
# "default to false" guidance lives in the prompt. We deliberately do NOT ask
# the model to self-rate confidence — that's a heuristics-era artifact; the
# has_hatched judgment is the signal.
_HATCHING_TOOL = {
    "name": "record_hatching",
    "description": "Record whether the C. elegans embryo has hatched, with brief reasoning.",
    "input_schema": {
        "type": "object",
        "properties": {
            "has_hatched": {
                "type": "boolean",
                "description": "True only if the embryo has visibly broken out of the eggshell.",
            },
            "reasoning": {
                "type": "string",
                "description": "One short sentence citing the visual evidence for the call.",
            },
        },
        "required": ["has_hatched", "reasoning"],
    },
}


class HatchingDetector(Detector):
    """Claude-vision hatching yes/no."""

    name = "hatching"

    def __init__(self, claude_client=None, model: str | None = None):
        self._claude = claude_client
        self._model = model

    async def run(
        self,
        volume: np.ndarray,
        context: dict[str, Any],
    ) -> DetectorResult:
        import json

        import anthropic

        from gently.settings import settings

        embryo_id = context.get("embryo_id", "?")
        timepoint = int(context.get("timepoint", 0))
        start = time.time()
        claude = self._claude or context.get("claude")
        if claude is None:
            return DetectorResult(
                detector_name=self.name,
                embryo_id=embryo_id,
                timepoint=timepoint,
                error="No Claude client available",
                elapsed_ms=(time.time() - start) * 1000,
            )

        b64_image = _volume_to_b64(volume, context.get("calibration"))
        if b64_image is None:
            return DetectorResult(
                detector_name=self.name,
                embryo_id=embryo_id,
                timepoint=timepoint,
                findings={"has_hatched": False},
                reasoning="Empty / unreadable volume",
                elapsed_ms=(time.time() - start) * 1000,
            )

        try:
            response = await asyncio.to_thread(
                claude.messages.create,
                model=self._model or settings.models.fast,
                max_tokens=256,
                tools=[_HATCHING_TOOL],
                tool_choice={"type": "tool", "name": _HATCHING_TOOL["name"]},
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": _HATCHING_PROMPT},
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": b64_image,
                                },
                            },
                        ],
                    }
                ],
            )

            # Forced tool_choice guarantees a tool_use block; read its parsed
            # input directly. No regex, no JSON-from-prose fallback.
            tool_input = next(
                (b.input for b in response.content if getattr(b, "type", None) == "tool_use"),
                None,
            )

            findings = {"has_hatched": False}
            reasoning = None
            err = None
            if isinstance(tool_input, dict):
                findings["has_hatched"] = bool(tool_input.get("has_hatched", False))
                reasoning = tool_input.get("reasoning")
            else:
                # Shouldn't happen with forced tool_choice — keep the
                # conservative default and record why.
                err = "no tool_use block in response"

            return DetectorResult(
                detector_name=self.name,
                embryo_id=embryo_id,
                timepoint=timepoint,
                findings=findings,
                reasoning=reasoning,
                raw_response=json.dumps(tool_input) if isinstance(tool_input, dict) else None,
                elapsed_ms=(time.time() - start) * 1000,
                error=err,
            )

        except (
            anthropic.APIConnectionError,
            anthropic.RateLimitError,
            anthropic.APIStatusError,
        ) as e:
            logger.error("[%s] Claude API error for %s: %s", self.name, embryo_id, e)
            return DetectorResult(
                detector_name=self.name,
                embryo_id=embryo_id,
                timepoint=timepoint,
                error=f"API error: {e}",
                elapsed_ms=(time.time() - start) * 1000,
            )
        except Exception as e:
            logger.exception("[%s] unexpected error", self.name)
            return DetectorResult(
                detector_name=self.name,
                embryo_id=embryo_id,
                timepoint=timepoint,
                error=str(e),
                elapsed_ms=(time.time() - start) * 1000,
            )
