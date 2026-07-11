"""
Dopaminergic signal detector — two-stage LLM pipeline for TestEmbryos in
the dopaminergic-reporter experiment.

Architecture:
- Stage 1: Perceiver (VLM call). Sees the image, answers a few open-ended
  visual questions in plain prose. Does no classification.
- Stage 2: Classifier (text-only Claude call). Reads the prose, fits it
  against a fixed rubric, emits structured findings. Supports an
  UNCERTAIN escape on each field so it can refuse to commit when the
  description is ambiguous.

The split separates "what's in the image?" from "what category does that
correspond to?" — the failure mode of the previous single-call detector
(WEAK on noise-only frames after per-image min-max stretch) was exactly
the conflation those two jobs.
"""

import asyncio
import base64
import io
import json
import logging
import re
import time
from typing import Any

import numpy as np

from .base import Detector, DetectorResult

logger = logging.getLogger(__name__)


_PERCEIVER_PROMPT = """You are looking at the max projection of a volume of C. elegans imaged
with a 488 nm light-sheet microscope. The embryo is expressing a fluorophore that lights up
when certain neurons are born.

We are trying to image the birth of these neurons and their continued existence. Your
description will be read by a classifier that decides how to guide the imaging: it will speed
up imaging once the neuronal structures appear, and trigger a 1-minute burst of fast
acquisitions once they look stably bright. Be specific so the classifier has something
concrete to act on.

You may initially see a faint outline of the embryo — autofluorescence from the body.

Eventually, you may see puncta-like structures in the embryo region of the image. If there
are any puncta outside the embryo region, ignore them — those are likely gut granules.

The nerve cells, as they begin to express, will first appear as a faint blob, then a brighter
blob, then a further-brighter blob, and will start to emit thread-like structures from them —
the nerve body. These are what we want to image.

The embryo may also eventually hatch. This can look like the embryo structure disappearing
entirely from the field of view. Mention it if you see this.

Describe what you see in a few sentences of plain prose.
"""


_CLASSIFIER_PROMPT = """You are reading a microscopist's description of an image of a C. elegans
embryo expressing a dopaminergic-neuron reporter (dat-1::mNeonGreen).

Classify the description against the rubric below. Output ONLY a JSON object — no prose, no
markdown fences. Your output drives the timelapse orchestrator's next imaging decision.

The orchestrator can take these actions based on your output:
- speed_up — accelerate imaging cadence to 1-minute intervals (when neurons begin to appear).
- burst — fire a 1-minute burst of fast acquisitions (when neurons are stably bright and
  well-resolved).
- ramp_down_power — step the 488 nm laser down (when signal saturates the camera).
- stop — stop imaging this embryo (when it has hatched / left the field).
- none — keep the base cadence, take no action (when the description is uncertain or shows
  only background).

Schema:
{
  "intensity_level": "NONE" | "WEAK" | "MEDIUM" | "STRONG" | "SATURATING" | "UNCERTAIN",
  # signal strength — is the signal absent, faint, average, or stronger?
  "structure_quality": "NONE" | "PARTIAL" | "GOOD" | "UNCERTAIN",
  # are the neuronal structures absent, emerging, fully emerged with good signal, or can't tell?
  "has_hatched": true | false,
  # has the embryo hatched or left the field of view? — drives the stop action
  "reasoning": "one short sentence quoting which words drove your choice"
}

intensity_level rubric (drives speed_up / ramp_down_power):
- NONE: description says no puncta in the embryo / blank / nothing above background. → none
- WEAK: description mentions 1 dim spot, OR signal explicitly described as "barely visible" /
  "could be noise" / "very faint". → none (could be noise)
- MEDIUM: description mentions 2+ clearly discrete bright spots above background. → speed_up
- STRONG: description mentions multiple bright, well-resolved spots. → speed_up
- SATURATING: description explicitly says the signal saturates the camera. → ramp_down_power
- UNCERTAIN: description hedges, doesn't address signal, or you cannot tell from the prose
  alone. → none

structure_quality rubric (drives burst):
- NONE: no puncta described. → none
- PARTIAL: puncta present but no neurites / no curved connecting traces. → none (not yet stable)
- GOOD: description mentions curved/elongated traces between puncta or recognizable neurite
  structure. → burst
- UNCERTAIN: description is silent on structure or ambiguous. → none

has_hatched: true ONLY if the description explicitly says the embryo has hatched, OR the
embryo structure has disappeared from the field of view. Default false. → stop

When the description is ambiguous, choose UNCERTAIN over guessing. When between two adjacent
levels, choose the more conservative (lower) one — false negatives (missing onset by a
timepoint) are cheap, false positives (burning photodose on noise) are expensive.

Description to classify:
---
{DESCRIPTION}
---
"""


class DopaminergicSignalDetector(Detector):
    """Two-stage Claude detector: VLM perceiver → text classifier."""

    name = "dopaminergic_signal"

    def __init__(
        self,
        claude_client=None,
        perceiver_model: str | None = None,
        classifier_model: str | None = None,
        # Back-compat: callers passing the old single-model kwarg.
        model: str | None = None,
    ):
        self._claude = claude_client
        self._perceiver_model = perceiver_model or model
        self._classifier_model = classifier_model
        self._calibration_notice_logged = False  # once-per-detector-instance

    async def run(
        self,
        volume: np.ndarray,
        context: dict[str, Any],
    ) -> DetectorResult:
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

        try:
            calibration = context.get("calibration")
            has_two_point = bool(
                calibration
                and calibration.get("dark") is not None
                and calibration.get("flat") is not None
            )
            if not has_two_point and not self._calibration_notice_logged:
                # Once per detector instance, at DEBUG so it's only
                # surfaced when explicitly asking for verbose logs; this
                # is a routine fallback, not an error.
                logger.debug(
                    "[%s] running without dark/flat calibration; using "
                    "fixed dynamic-range scaling (-100 baseline, /4).",
                    self.name,
                )
                self._calibration_notice_logged = True

            b64_image = _volume_to_b64(volume, calibration)
            if b64_image is None:
                return DetectorResult(
                    detector_name=self.name,
                    embryo_id=embryo_id,
                    timepoint=timepoint,
                    findings={
                        "intensity_level": "NONE",
                        "structure_quality": "NONE",
                        "has_hatched": False,
                    },
                    reasoning="Empty / unreadable volume",
                    elapsed_ms=(time.time() - start) * 1000,
                )

            # Stage 1: Perceiver (image → prose)
            perceiver_model = self._perceiver_model or settings.models.perception
            description, perceiver_raw = await self._call_perceiver(
                claude,
                perceiver_model,
                b64_image,
            )

            # Stage 2: Classifier (prose → findings)
            classifier_model = self._classifier_model or settings.models.main
            findings, classifier_raw, parse_err = await self._call_classifier(
                claude,
                classifier_model,
                description,
            )

            return DetectorResult(
                detector_name=self.name,
                embryo_id=embryo_id,
                timepoint=timepoint,
                findings={k: v for k, v in findings.items() if k != "reasoning"},
                reasoning=findings.get("reasoning"),
                raw_response={
                    "description": description,
                    "perceiver_raw": perceiver_raw,
                    "classifier_raw": classifier_raw,
                    "perceiver_model": perceiver_model,
                    "classifier_model": classifier_model,
                },
                elapsed_ms=(time.time() - start) * 1000,
                error=parse_err,
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

    async def _call_perceiver(
        self,
        claude,
        model: str,
        b64_image: str,
    ) -> tuple[str, str]:
        """Stage 1: image → free prose description. Stateless — each
        timepoint is evaluated independently."""
        response = await asyncio.to_thread(
            claude.messages.create,
            model=model,
            max_tokens=400,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": _PERCEIVER_PROMPT},
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
        if response.stop_reason == "refusal" or not response.content:
            return "(perception model declined the request)", ""
        raw = response.content[0].text
        return raw.strip(), raw

    async def _call_classifier(
        self,
        claude,
        model: str,
        description: str,
    ) -> tuple[dict[str, Any], str, str | None]:
        """Stage 2: description prose → structured findings."""
        prompt = _CLASSIFIER_PROMPT.replace("{DESCRIPTION}", description or "(no description)")
        response = await asyncio.to_thread(
            claude.messages.create,
            model=model,
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        if response.stop_reason == "refusal" or not response.content:
            return dict(_DEFAULT_FINDINGS), "", "Safety refusal"
        raw = response.content[0].text
        findings, parse_err = _parse_response(raw)
        return findings, raw, parse_err


def _volume_to_b64(volume: np.ndarray, calibration: dict | None = None) -> str | None:
    """Max-project a volume (with optional dark/flat correction + edge ROI)
    and return a base64-encoded PNG.

    Brightness preservation: when calibration with dark/flat is present we
    use the two-point-corrected image directly. When it is absent we scale
    by sensor dynamic range (16-bit → /256, 12-bit-in-uint16 → /16) rather
    than per-image min-max — that latter stretch makes a noise-only blank
    frame visually indistinguishable from a true onset frame and was the
    root cause of false WEAK detections.
    """
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

    # Dual-view side-by-side layout: when the projection is much wider
    # than it is tall (typical 512x2048 layout = two views concatenated
    # along X), crop to the left half (side A only). The right half is
    # currently empty / dim duplicate and drags the VLM's overall
    # impression toward "dark." This matches the user's "we are only
    # showing side A" assertion.
    h, w = proj.shape[-2], proj.shape[-1]
    if w >= 2 * h:
        proj = proj[..., : w // 2]

    # Two-point correction (Phase 6 calibration: dark + flat). When this
    # is available the result is already a meaningful uint8 image with
    # background near zero — no further normalization needed.
    calibrated = False
    if calibration:
        dark = calibration.get("dark")
        flat = calibration.get("flat")
        if dark is not None and flat is not None and dark.shape == proj.shape:
            denom = flat.astype(np.float32) - dark.astype(np.float32)
            denom[denom <= 0] = 1.0
            proj = (proj.astype(np.float32) - dark.astype(np.float32)) / denom * 255.0
            proj = np.clip(proj, 0, 255).astype(np.uint8)
            calibrated = True

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

    # Convert to uint8 preserving absolute brightness scale.
    if proj.dtype != np.uint8:
        if calibrated:
            # Already in 0..255 from two-point correction above.
            proj = np.clip(proj, 0, 255).astype(np.uint8)
        else:
            # No calibration: subtract a fixed dark baseline of 100 counts
            # (raw background floor for this camera) and floor at 0, then
            # scale to make the embryo body autofluorescence visible.
            # We use /4 (not /16) because the body autofluor only sits
            # ~5-35 counts above the background floor — /16 maps that
            # range to 0-2 out of 255, invisible to the VLM. /4 maps it
            # to 1-9 (faintly visible) and lets bright puncta clip to
            # white at 255, which is fine — the classifier reasons about
            # STRONG vs SATURATING from prose, not from raw pixel values.
            arr = proj.astype(np.float32)
            arr = np.maximum(arr - 100.0, 0.0)
            proj = np.clip(arr / 4.0, 0, 255).astype(np.uint8)

    img = PILImage.fromarray(proj)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


_DEFAULT_FINDINGS = {
    "intensity_level": "UNCERTAIN",
    "structure_quality": "UNCERTAIN",
    "has_hatched": False,
    "reasoning": "Unparseable classifier response",
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


def _normalize_findings(d: dict[str, Any]) -> dict[str, Any]:
    """Coerce a parsed JSON dict to the expected schema, filling defaults
    for missing keys and validating enum values. UNCERTAIN is a valid
    value for both intensity_level and structure_quality and downstream
    rules treat it as 'no trigger'."""
    intensity_choices = {"NONE", "WEAK", "MEDIUM", "STRONG", "SATURATING", "UNCERTAIN"}
    structure_choices = {"NONE", "PARTIAL", "GOOD", "UNCERTAIN"}

    intensity = str(d.get("intensity_level", "UNCERTAIN")).upper()
    if intensity not in intensity_choices:
        intensity = "UNCERTAIN"

    structure = str(d.get("structure_quality", "UNCERTAIN")).upper()
    if structure not in structure_choices:
        structure = "UNCERTAIN"

    hatched = bool(d.get("has_hatched", False))
    reasoning = str(d.get("reasoning", "")).strip() or None

    return {
        "intensity_level": intensity,
        "structure_quality": structure,
        "has_hatched": hatched,
        "reasoning": reasoning,
    }
