"""
BlankImageDetector — Claude yes/no on whether the volume is blank / corrupted.

Direct refactor of ``AgentMicroscope.check_blank_image`` (gently/app/agent.py:1031)
into the Detector interface. The original method is kept as a thin shim
for backward compatibility; new callers should use this class.
"""

import asyncio
import base64
import io
import logging
import time
from typing import Any

import numpy as np

from .base import Detector, DetectorResult

logger = logging.getLogger(__name__)


_BLANK_PROMPT = """Look at this microscopy image. Is this a VALID microscopy image or a
BLANK/CORRUPTED image?

A BLANK or CORRUPTED image shows:
- Mostly uniform gray/black with no structure
- No visible biological features
- Static noise only
- Hardware artifacts (stripes, patterns) without actual sample

A VALID image shows:
- Visible biological structure (embryo, cells, etc.)
- Even if the embryo is small or faint, there should be clear structure

Respond with ONLY: VALID or BLANK"""


class BlankImageDetector(Detector):
    """Yes/no blank-image check via Claude vision."""

    name = "blank_image"

    def __init__(self, claude_client=None, model: str | None = None):
        self._claude = claude_client
        self._model = model

    async def run(
        self,
        volume: np.ndarray,
        context: dict[str, Any],
    ) -> DetectorResult:
        import anthropic
        from PIL import Image as PILImage

        from gently.settings import settings

        embryo_id = context.get("embryo_id", "?")
        timepoint = int(context.get("timepoint", 0))
        start = time.time()

        # Numerical short-circuit (no API call needed)
        vol = np.squeeze(volume) if volume is not None else None
        if vol is None or vol.size == 0:
            return DetectorResult(
                detector_name=self.name,
                embryo_id=embryo_id,
                timepoint=timepoint,
                findings={"is_blank": True},
                reasoning="Empty volume",
                elapsed_ms=(time.time() - start) * 1000,
            )

        max_proj = np.max(vol, axis=0) if vol.ndim == 3 else vol
        if np.std(max_proj) < 1.0 or np.max(max_proj) < 10:
            return DetectorResult(
                detector_name=self.name,
                embryo_id=embryo_id,
                timepoint=timepoint,
                findings={"is_blank": True},
                reasoning="Numerical check (low std / max)",
                elapsed_ms=(time.time() - start) * 1000,
            )

        claude = self._claude or context.get("claude")
        if claude is None:
            return DetectorResult(
                detector_name=self.name,
                embryo_id=embryo_id,
                timepoint=timepoint,
                findings={"is_blank": False},
                reasoning="No Claude client; deferred to numerical check (passed)",
                elapsed_ms=(time.time() - start) * 1000,
            )

        # Normalize, encode, ask Claude
        if max_proj.max() == 0:
            return DetectorResult(
                detector_name=self.name,
                embryo_id=embryo_id,
                timepoint=timepoint,
                findings={"is_blank": True},
                reasoning="Max projection is all zeros",
                elapsed_ms=(time.time() - start) * 1000,
            )
        normalized = (max_proj / max_proj.max() * 255).astype(np.uint8)
        buf = io.BytesIO()
        PILImage.fromarray(normalized).save(buf, format="PNG")
        b64_image = base64.b64encode(buf.getvalue()).decode("ascii")

        try:
            response = await asyncio.to_thread(
                claude.messages.create,
                model=self._model or settings.models.fast,
                max_tokens=10,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": _BLANK_PROMPT},
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
            text = (response.content[0].text if response.content else "").strip().upper()
            is_blank = "BLANK" in text
            return DetectorResult(
                detector_name=self.name,
                embryo_id=embryo_id,
                timepoint=timepoint,
                findings={"is_blank": is_blank},
                reasoning=text,
                raw_response=text,
                elapsed_ms=(time.time() - start) * 1000,
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
