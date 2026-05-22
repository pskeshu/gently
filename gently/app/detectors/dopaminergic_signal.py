"""
Dopaminergic signal detector — ad-hoc Claude vision observer for
TestEmbryos in the dopaminergic-reporter experiment.

Replaces a hand-coded brightness analysis. Reads a max-projection +
optional dark/flat/edge_roi preprocessing from calibration, sends it to
Haiku, and parses a structured JSON response describing intensity level,
structure quality, and hatched state.

Modeled on ``AgentMicroscope.check_blank_image`` (gently/app/agent.py:1031).
"""

import asyncio
import base64
import io
import json
import logging
import re
import time
from typing import Any, Dict, Optional

import numpy as np

from .base import Detector, DetectorResult

logger = logging.getLogger(__name__)


_PROMPT_TEMPLATE = """You are observing a C. elegans embryo carrying a sparse dopaminergic-neuron fluorescent reporter (e.g. ADE / CEP / PDE — head and tail neurons that typically light up around the 3-fold / pretzel developmental stage). Most of the time the embryo is BLANK; only after expression begins do bright punctate neurons appear in the head region.

You will see a max-projection of an acquired 488 nm volume. Decide:

1. `intensity_level` — overall brightness inside the embryo body:
   - "NONE": no signal above background (blank embryo)
   - "WEAK": faint signal beginning to appear
   - "MEDIUM": clearly visible signal
   - "STRONG": bright, well-resolved signal
   - "SATURATING": signal saturates the camera (>1% of pixels at sensor max)

2. `structure_quality` — how interpretable the dopaminergic structure is:
   - "NONE": no neurons visible
   - "PARTIAL": some neuron(s) visible but not all expected ones
   - "GOOD": the expected sparse neuron pattern is clearly resolved
     (this is the burst-mode trigger threshold)

3. `has_hatched` — true ONLY if the embryo has clearly hatched
   (visible larva moving outside the eggshell, or the eggshell has cracked).
   False otherwise. Do not over-call; default false unless certain.

4. `reasoning` — one sentence explaining your assessment.

Respond ONLY with a JSON object exactly matching this schema:

{
  "intensity_level": "NONE|WEAK|MEDIUM|STRONG|SATURATING",
  "structure_quality": "NONE|PARTIAL|GOOD",
  "has_hatched": true|false,
  "reasoning": "..."
}
"""


class DopaminergicSignalDetector(Detector):
    """Claude-vision detector for fluorescent reporter onset + structure."""

    name = "dopaminergic_signal"

    def __init__(self, claude_client=None, model: Optional[str] = None):
        self._claude = claude_client
        self._model = model  # resolved at run-time from settings if None

    async def run(
        self,
        volume: np.ndarray,
        context: Dict[str, Any],
    ) -> DetectorResult:
        from gently.settings import settings
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

        try:
            # Preprocess (calibration-aware if provided) → max projection → PNG → b64
            b64_image = _volume_to_b64(volume, context.get("calibration"))
            if b64_image is None:
                return DetectorResult(
                    detector_name=self.name,
                    embryo_id=embryo_id,
                    timepoint=timepoint,
                    findings={"intensity_level": "NONE", "structure_quality": "NONE",
                              "has_hatched": False},
                    reasoning="Empty / unreadable volume",
                    elapsed_ms=(time.time() - start) * 1000,
                )

            model_name = self._model or settings.models.fast
            response = await asyncio.to_thread(
                claude.messages.create,
                model=model_name,
                max_tokens=400,
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": _PROMPT_TEMPLATE},
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/png",
                                "data": b64_image,
                            },
                        },
                    ],
                }],
            )

            raw = response.content[0].text if response.content else ""
            findings, parse_err = _parse_response(raw)

            return DetectorResult(
                detector_name=self.name,
                embryo_id=embryo_id,
                timepoint=timepoint,
                findings={k: v for k, v in findings.items() if k != "reasoning"},
                reasoning=findings.get("reasoning"),
                raw_response=raw,
                elapsed_ms=(time.time() - start) * 1000,
                error=parse_err,
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


def _volume_to_b64(volume: np.ndarray, calibration: Optional[Dict] = None) -> Optional[str]:
    """Max-project a volume (with optional dark/flat correction + edge ROI)
    and return a base64-encoded PNG."""
    from PIL import Image as PILImage

    if volume is None or volume.size == 0:
        return None

    vol = np.squeeze(volume)
    if vol.ndim == 4:  # multi-view: take first view
        vol = vol[0]

    # Max projection along Z
    if vol.ndim == 3:
        proj = np.max(vol, axis=0)
    elif vol.ndim == 2:
        proj = vol
    else:
        return None

    # Two-point correction (Phase 6 calibration: dark + flat)
    if calibration:
        dark = calibration.get("dark")
        flat = calibration.get("flat")
        if dark is not None and flat is not None and dark.shape == proj.shape:
            denom = (flat.astype(np.float32) - dark.astype(np.float32))
            denom[denom <= 0] = 1.0
            proj = ((proj.astype(np.float32) - dark.astype(np.float32)) / denom * 255.0)
            proj = np.clip(proj, 0, 255).astype(np.uint8)

        # Edge ROI mask
        edge_bbox = calibration.get("edge_bbox")  # (x0, y0, x1, y1)
        if edge_bbox is not None:
            x0, y0, x1, y1 = map(int, edge_bbox)
            x0, y0 = max(0, x0), max(0, y0)
            x1 = min(proj.shape[1], x1)
            y1 = min(proj.shape[0], y1)
            if x1 > x0 and y1 > y0:
                proj = proj[y0:y1, x0:x1]

    # Numerical sanity: blank image short-circuit
    if proj.size == 0 or (np.std(proj) < 1.0 and np.max(proj) < 10):
        # Return a minimal black PNG so Claude still gets a frame
        img = PILImage.fromarray(np.zeros((16, 16), dtype=np.uint8))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return base64.b64encode(buf.getvalue()).decode("ascii")

    # Normalize to uint8
    if proj.dtype != np.uint8:
        lo, hi = float(proj.min()), float(proj.max())
        if hi > lo:
            proj = ((proj.astype(np.float32) - lo) / (hi - lo) * 255.0).astype(np.uint8)
        else:
            proj = np.zeros_like(proj, dtype=np.uint8)

    img = PILImage.fromarray(proj)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


_DEFAULT_FINDINGS = {
    "intensity_level": "NONE",
    "structure_quality": "NONE",
    "has_hatched": False,
    "reasoning": "Unparseable response",
}


def _parse_response(raw: str) -> tuple:
    """Defensive JSON parse — Claude sometimes wraps JSON in prose / code fences.

    Returns ``(findings_dict, error_or_None)``.
    """
    if not raw or not raw.strip():
        return dict(_DEFAULT_FINDINGS), "empty response"

    # Try direct parse first.
    try:
        parsed = json.loads(raw.strip())
        return _normalize_findings(parsed), None
    except json.JSONDecodeError:
        pass

    # Strip markdown code fences.
    m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if m:
        try:
            return _normalize_findings(json.loads(m.group(1))), None
        except json.JSONDecodeError:
            pass

    # Find any JSON-looking blob.
    m = re.search(r"\{[^{}]*\}", raw, re.DOTALL)
    if m:
        try:
            return _normalize_findings(json.loads(m.group(0))), None
        except json.JSONDecodeError:
            pass

    return dict(_DEFAULT_FINDINGS), f"could not parse response: {raw[:120]!r}"


def _normalize_findings(d: Dict[str, Any]) -> Dict[str, Any]:
    """Coerce a parsed JSON dict to the expected schema, filling defaults
    for missing keys and validating enum values."""
    intensity_choices = {"NONE", "WEAK", "MEDIUM", "STRONG", "SATURATING"}
    structure_choices = {"NONE", "PARTIAL", "GOOD"}

    intensity = str(d.get("intensity_level", "NONE")).upper()
    if intensity not in intensity_choices:
        intensity = "NONE"

    structure = str(d.get("structure_quality", "NONE")).upper()
    if structure not in structure_choices:
        structure = "NONE"

    hatched = bool(d.get("has_hatched", False))
    reasoning = str(d.get("reasoning", "")).strip() or None

    return {
        "intensity_level": intensity,
        "structure_quality": structure,
        "has_hatched": hatched,
        "reasoning": reasoning,
    }
