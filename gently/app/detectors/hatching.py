"""
Hatching detector — Claude-vision check for whether an embryo has hatched.

Useful for TestEmbryos that lack the nuclear marker the standard perception
pipeline trains on. The dopaminergic-signal detector already returns
``has_hatched`` as part of its richer schema; this is a lighter-weight
yes/no for use cases where structure / intensity assessment isn't needed.
"""

import asyncio
import logging
import time
from typing import Any, Dict, Optional

import numpy as np

from .base import Detector, DetectorResult
from .dopaminergic_signal import _volume_to_b64

logger = logging.getLogger(__name__)


_HATCHING_PROMPT = """You are observing a C. elegans embryo on a microscope. Decide whether the embryo has HATCHED.

A HATCHED embryo:
- Has visibly broken out of the eggshell
- Often shows the elongated larva moving outside the eggshell outline
- The eggshell itself may be visible as a broken / collapsed outline

An UNHATCHED embryo:
- Is still contained within an intact eggshell
- May be at any pre-hatching stage (bean, comma, 1.5-fold, 2-fold, pretzel)

Respond with ONLY a JSON object exactly matching this schema:

{
  "has_hatched": true|false,
  "confidence": "LOW|MEDIUM|HIGH",
  "reasoning": "..."
}

Default to false unless you are confident. Don't over-call hatching.
"""


class HatchingDetector(Detector):
    """Claude-vision hatching yes/no, with confidence."""

    name = "hatching"

    def __init__(self, claude_client=None, model: Optional[str] = None):
        self._claude = claude_client
        self._model = model

    async def run(
        self,
        volume: np.ndarray,
        context: Dict[str, Any],
    ) -> DetectorResult:
        from gently.settings import settings
        import json, re
        import anthropic

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
                findings={"has_hatched": False, "confidence": "LOW"},
                reasoning="Empty / unreadable volume",
                elapsed_ms=(time.time() - start) * 1000,
            )

        try:
            response = await asyncio.to_thread(
                claude.messages.create,
                model=self._model or settings.models.fast,
                max_tokens=200,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": _HATCHING_PROMPT},
                        {"type": "image", "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": b64_image,
                        }},
                    ],
                }],
            )
            raw = response.content[0].text if response.content else ""

            findings = {"has_hatched": False, "confidence": "LOW"}
            reasoning = None
            err = None
            try:
                m = re.search(r"\{.*?\}", raw, re.DOTALL)
                blob = m.group(0) if m else raw.strip()
                parsed = json.loads(blob)
                findings["has_hatched"] = bool(parsed.get("has_hatched", False))
                confidence = str(parsed.get("confidence", "LOW")).upper()
                if confidence not in {"LOW", "MEDIUM", "HIGH"}:
                    confidence = "LOW"
                findings["confidence"] = confidence
                reasoning = parsed.get("reasoning")
            except (json.JSONDecodeError, AttributeError) as e:
                err = f"parse error: {e}"

            return DetectorResult(
                detector_name=self.name,
                embryo_id=embryo_id,
                timepoint=timepoint,
                findings=findings,
                reasoning=reasoning,
                raw_response=raw,
                elapsed_ms=(time.time() - start) * 1000,
                error=err,
            )

        except (anthropic.APIConnectionError, anthropic.RateLimitError, anthropic.APIStatusError) as e:
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
