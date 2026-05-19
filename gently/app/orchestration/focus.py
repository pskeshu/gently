"""
FocusController — owns the calibration journey for each embryo in a session.

The orchestrator (the conversational agent) interacts with this controller
through a deliberately narrow surface: certificate(embryo_id), is_ready_to_image(
embryo_id), and the high-level calibration tools. The orchestrator never reads
R-squared values or sweep curves directly — those live behind the certificate.

Two flavors of verification feed every certificate:

  1. A synchronous rule engine that inspects R-squared at top/bottom, the slope
     value, and the slope-vs-session-prior consistency. Runs the moment
     calibration finishes.
  2. An asynchronous VLM check, scheduled as an asyncio.Task. It asks Claude
     Vision whether the best focus image at each calibration position actually
     shows the embryo (using the existing detect_embryo_presence call). While
     the task is in flight, certificate['verified'] reads 'pending'.

The certificate format is:

  {
    'verified': True | False | 'pending',
    'r_squared_top': float,
    'r_squared_bottom': float,
    'slope_um_per_deg': float,
    'concerns': [str, ...],          # human-readable issues from rules + VLM
    'rules_passed': bool,            # synchronous rule outcome (stable)
    'vlm_check': {                   # populated when async verification lands
       'passed': True | False | None,
       'top_score': int,             # 0-10 from detect_embryo_presence
       'bottom_score': int,
       'note': str,
    } | None,
    'checked_at': ISO-8601 timestamp,
  }
"""

from __future__ import annotations

import asyncio
import logging
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Rule engine
# ---------------------------------------------------------------------------

# Threshold used in calibration code; keep these aligned with the values in
# gently.app.tools.calibration_tools._adaptive_focus_sweep (MIN_R_SQUARED).
MIN_R_SQUARED = 0.75
SLOPE_MIN_UM_PER_DEG = 60.0
SLOPE_MAX_UM_PER_DEG = 150.0
SLOPE_PRIOR_DEVIATION_FRACTION = 0.4  # >40% off the session prior is suspect


def apply_calibration_rules(
    calibration: Dict[str, Any],
    session_prior: Optional[Any] = None,
) -> Tuple[bool, List[str]]:
    """Inspect a freshly written calibration dict and produce concerns.

    Pure function — no I/O, no async. Returns (rules_passed, concerns).
    Concerns is a list of human-readable strings, each tagged with a short
    code at the front so downstream code (or the agent) can match patterns
    without parsing prose.
    """
    concerns: List[str] = []

    r_top = calibration.get('r_squared_top')
    r_bot = calibration.get('r_squared_bottom')
    slope = calibration.get('slope_um_per_deg')

    if r_top is not None and r_top < MIN_R_SQUARED:
        concerns.append(
            f"[r2_top_low] R-squared at top calib position = {r_top:.2f}, "
            f"below threshold {MIN_R_SQUARED:.2f}"
        )
    if r_bot is not None and r_bot < MIN_R_SQUARED:
        concerns.append(
            f"[r2_bottom_low] R-squared at bottom calib position = {r_bot:.2f}, "
            f"below threshold {MIN_R_SQUARED:.2f}"
        )

    if slope is not None:
        if slope < SLOPE_MIN_UM_PER_DEG or slope > SLOPE_MAX_UM_PER_DEG:
            concerns.append(
                f"[slope_anomalous] Slope = {slope:.1f} um/deg, outside expected "
                f"range [{SLOPE_MIN_UM_PER_DEG:.0f}, {SLOPE_MAX_UM_PER_DEG:.0f}]"
            )
        # Compare to session prior if we have one
        prior_slope = None
        if session_prior is not None:
            # The CalibrationPrior class exposes slope_um_per_deg only after
            # at least one good calibration has been recorded.
            prior_slope = getattr(session_prior, 'slope_um_per_deg', None)
            n_prior = getattr(session_prior, 'num_calibrations', 0)
            if prior_slope and n_prior >= 1:
                deviation = abs(slope - prior_slope) / max(abs(prior_slope), 1e-6)
                if deviation > SLOPE_PRIOR_DEVIATION_FRACTION:
                    concerns.append(
                        f"[slope_prior_mismatch] Slope = {slope:.1f} um/deg deviates "
                        f"{deviation*100:.0f}% from session prior {prior_slope:.1f}"
                    )

    rules_passed = len(concerns) == 0
    return rules_passed, concerns


# ---------------------------------------------------------------------------
# FocusController
# ---------------------------------------------------------------------------

class FocusController:
    """Per-session controller that owns calibration certificates.

    Constructed once on the agent (`agent.focus`) and shared across tools.
    Stateless across sessions — the persistent certificate lives on
    embryo.calibration['certificate'] which is serialized by FileStore.
    """

    def __init__(self, agent: Any):
        self.agent = agent
        # embryo_id -> asyncio.Task currently running the VLM check
        self._pending_tasks: Dict[str, asyncio.Task] = {}
        # embryo_id -> (top_image, bottom_image) held in memory between
        # calibration finishing and the async verification completing
        self._verification_images: Dict[str, Tuple[Optional[np.ndarray],
                                                   Optional[np.ndarray]]] = {}

    # ------------------------------------------------------------------ entry

    def record_calibration(
        self,
        embryo_id: str,
        calibration: Dict[str, Any],
        top_image: Optional[np.ndarray] = None,
        bottom_image: Optional[np.ndarray] = None,
        schedule_vlm: bool = True,
        extra_concerns: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Apply rules, write the certificate, optionally schedule VLM check.

        Returns the newly written certificate dict. Called by
        ``calibrate_embryo`` right after it computes slope and offset.

        ``extra_concerns`` lets the caller supply already-detected problems
        (e.g. pre-flight quality rejections from the edge-detection phase)
        that should flow into the certificate alongside the rule engine's
        own findings. Any non-empty extra_concerns forces ``rules_passed`` to
        False so the gate refuses downstream imaging.
        """
        embryo = self._get_embryo(embryo_id)
        if embryo is None:
            logger.warning("record_calibration: no embryo %s", embryo_id)
            return {}

        session_prior = getattr(self.agent.experiment, 'calibration_prior', None)
        rules_passed, concerns = apply_calibration_rules(calibration, session_prior)
        if extra_concerns:
            concerns = list(concerns) + list(extra_concerns)
            rules_passed = False

        certificate: Dict[str, Any] = {
            'verified': True if rules_passed else False,
            'rules_passed': rules_passed,
            'r_squared_top': calibration.get('r_squared_top'),
            'r_squared_bottom': calibration.get('r_squared_bottom'),
            'slope_um_per_deg': calibration.get('slope_um_per_deg'),
            'concerns': concerns,
            'vlm_check': None,
            'checked_at': datetime.now().isoformat(),
        }

        # Write the certificate onto the embryo's calibration dict so it
        # round-trips through FileStore for free.
        if not isinstance(embryo.calibration, dict):
            embryo.calibration = {}
        embryo.calibration['certificate'] = certificate

        # If we have images and the user wants async verification, schedule it.
        # The certificate is marked pending so the orchestrator's
        # is_ready_to_image gate keeps blocking until the task lands.
        if schedule_vlm and top_image is not None and bottom_image is not None:
            self._verification_images[embryo_id] = (top_image, bottom_image)
            certificate['verified'] = 'pending'
            self._spawn_verification(embryo_id)

        return certificate

    # ------------------------------------------------------------------ query

    def certificate(self, embryo_id: str) -> Optional[Dict[str, Any]]:
        embryo = self._get_embryo(embryo_id)
        if embryo is None or not isinstance(embryo.calibration, dict):
            return None
        return embryo.calibration.get('certificate')

    def is_ready_to_image(self, embryo_id: str) -> Tuple[bool, str]:
        """Single gate the orchestrator should consult before imaging.

        Returns (ready, reason). ``reason`` is always a short, agent-readable
        sentence — safe to surface in tool output verbatim.
        """
        embryo = self._get_embryo(embryo_id)
        if embryo is None:
            return False, f"embryo '{embryo_id}' not found"
        if not isinstance(embryo.calibration, dict) or not embryo.calibration:
            return False, f"{embryo_id} has not been calibrated"

        cert = embryo.calibration.get('certificate')
        if cert is None:
            # Legacy calibration without certificate — treat as a soft warning
            # but allow imaging. The remediation is a recalibration which will
            # write a certificate on the next pass.
            return True, f"{embryo_id} calibrated (legacy, no certificate)"

        verified = cert.get('verified')
        if verified == 'pending':
            return False, f"{embryo_id} calibration verification still in progress"
        if verified is False:
            concerns = cert.get('concerns') or ['unknown calibration concern']
            return False, (
                f"{embryo_id} calibration has unresolved concerns: "
                + "; ".join(concerns)
            )
        return True, f"{embryo_id} calibration verified"

    def gate_many(self, embryo_ids: List[str]) -> Tuple[bool, List[str]]:
        """Batch version of is_ready_to_image — returns (all_ready, refusals).

        ``refusals`` is empty when all embryos pass; otherwise it lists the
        per-embryo refusal reasons so a caller can render a single error.
        """
        refusals: List[str] = []
        for eid in embryo_ids:
            ready, reason = self.is_ready_to_image(eid)
            if not ready:
                refusals.append(reason)
        return len(refusals) == 0, refusals

    def has_pending(self, embryo_id: Optional[str] = None) -> bool:
        if embryo_id is None:
            return any(not t.done() for t in self._pending_tasks.values())
        task = self._pending_tasks.get(embryo_id)
        return task is not None and not task.done()

    async def await_pending(
        self,
        embryo_id: Optional[str] = None,
        timeout: float = 60.0,
    ) -> Dict[str, Any]:
        """Block until pending verifications finish (or timeout).

        Returns a summary dict ``{embryo_id: certificate_or_state}``. Safe to
        call when nothing is pending — returns an empty dict in that case.
        """
        if embryo_id is not None:
            tasks_to_wait = (
                {embryo_id: self._pending_tasks[embryo_id]}
                if embryo_id in self._pending_tasks else {}
            )
        else:
            tasks_to_wait = dict(self._pending_tasks)

        if not tasks_to_wait:
            return {}

        try:
            await asyncio.wait_for(
                asyncio.gather(*tasks_to_wait.values(), return_exceptions=True),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "Focus verification timed out after %.0fs (embryos: %s)",
                timeout, list(tasks_to_wait.keys()),
            )

        return {eid: self.certificate(eid) for eid in tasks_to_wait.keys()}

    # ----------------------------------------------------------- async worker

    def _spawn_verification(self, embryo_id: str) -> None:
        """Start a new asyncio task that runs the VLM verification.

        If a task is already running for this embryo, it is cancelled first —
        a re-calibration supersedes the previous verification.
        """
        existing = self._pending_tasks.get(embryo_id)
        if existing is not None and not existing.done():
            existing.cancel()

        try:
            loop = asyncio.get_event_loop()
        except RuntimeError:
            logger.warning(
                "No running event loop for VLM verification of %s; skipping",
                embryo_id,
            )
            return

        task = loop.create_task(self._verify_via_vlm(embryo_id))
        self._pending_tasks[embryo_id] = task

    async def _verify_via_vlm(self, embryo_id: str) -> None:
        """Body of the verification task.

        Asks Claude Vision whether each calibration position actually shows
        the embryo. Uses ``detect_embryo_presence`` rather than a montage
        prompt because that path is already exercised at calibration time and
        returns a clean ``(visible, feature_score, description)`` triple.
        """
        try:
            top_img, bot_img = self._verification_images.get(
                embryo_id, (None, None)
            )
            if top_img is None or bot_img is None:
                self._finalize(embryo_id, vlm_passed=None,
                               top_score=0, bottom_score=0,
                               note="no calibration images retained — skipping VLM check")
                return

            # Lazy import: avoid pulling the hardware module at controller import
            from gently.hardware.dispim.claude_client import AsyncClaudeClient
            from PIL import Image

            vlm = AsyncClaudeClient()

            top_path = self._save_temp_png(top_img)
            bot_path = self._save_temp_png(bot_img)

            try:
                top_present, top_score, top_desc = await vlm.detect_embryo_presence(top_path)
                bot_present, bot_score, bot_desc = await vlm.detect_embryo_presence(bot_path)
            finally:
                for p in (top_path, bot_path):
                    try:
                        p.unlink(missing_ok=True)
                    except Exception:
                        pass

            # Decision rule: BOTH positions must show the embryo, and both
            # must have at least sparse features. Score threshold is
            # intentionally lenient (>=3) — calibration positions sit at the
            # embryo edges by design, so they won't always score 7-10.
            VLM_MIN_SCORE = 3
            both_visible = bool(top_present) and bool(bot_present)
            both_featured = top_score >= VLM_MIN_SCORE and bot_score >= VLM_MIN_SCORE
            passed = both_visible and both_featured

            note_parts = []
            if not top_present:
                note_parts.append(f"VLM saw no embryo at top calib position ({top_desc[:80]})")
            elif top_score < VLM_MIN_SCORE:
                note_parts.append(f"VLM saw only sparse features at top (score {top_score}/10)")
            if not bot_present:
                note_parts.append(f"VLM saw no embryo at bottom calib position ({bot_desc[:80]})")
            elif bot_score < VLM_MIN_SCORE:
                note_parts.append(f"VLM saw only sparse features at bottom (score {bot_score}/10)")
            if passed:
                note_parts.append(
                    f"VLM confirms embryo visible at both calib positions "
                    f"(top {top_score}/10, bottom {bot_score}/10)"
                )

            self._finalize(
                embryo_id,
                vlm_passed=passed,
                top_score=int(top_score),
                bottom_score=int(bot_score),
                note=" | ".join(note_parts) if note_parts else "VLM check complete",
            )

        except asyncio.CancelledError:
            # Cancelled by a fresh calibration — no certificate update,
            # the new calibration will write its own.
            raise
        except Exception as exc:
            logger.exception("VLM verification failed for %s", embryo_id)
            self._finalize(
                embryo_id,
                vlm_passed=None,
                top_score=0,
                bottom_score=0,
                note=f"verification error: {exc}",
            )

    def _finalize(
        self,
        embryo_id: str,
        vlm_passed: Optional[bool],
        top_score: int,
        bottom_score: int,
        note: str,
    ) -> None:
        embryo = self._get_embryo(embryo_id)
        if embryo is None or not isinstance(embryo.calibration, dict):
            self._verification_images.pop(embryo_id, None)
            self._pending_tasks.pop(embryo_id, None)
            return

        cert = embryo.calibration.get('certificate') or {}
        rules_passed = bool(cert.get('rules_passed'))
        concerns = list(cert.get('concerns') or [])

        cert['vlm_check'] = {
            'passed': vlm_passed,
            'top_score': top_score,
            'bottom_score': bottom_score,
            'note': note,
        }

        if vlm_passed is False:
            concerns.append(f"[vlm_failed] {note}")

        cert['concerns'] = concerns
        # Final verified state: rules must pass AND VLM must not actively
        # disagree. A None VLM result (skipped / errored) is treated as
        # neutral — the rules decide.
        if not rules_passed:
            cert['verified'] = False
        elif vlm_passed is False:
            cert['verified'] = False
        else:
            cert['verified'] = True
        cert['checked_at'] = datetime.now().isoformat()
        embryo.calibration['certificate'] = cert

        # Persist updated calibration so the certificate survives restarts.
        # The agent's persistence entry point is save_session(); we tolerate
        # its absence (test harnesses) and any I/O failure so a flaky write
        # never crashes the background verification task.
        try:
            saver = getattr(self.agent, 'save_session', None)
            if callable(saver):
                saver()
        except Exception as save_err:
            logger.warning("Failed to persist updated certificate: %s", save_err)

        self._verification_images.pop(embryo_id, None)
        self._pending_tasks.pop(embryo_id, None)

    # ----------------------------------------------------------- helpers

    def _get_embryo(self, embryo_id: str):
        try:
            return self.agent.experiment.get_embryo_by_any_name(embryo_id)
        except Exception:
            return None

    @staticmethod
    def _save_temp_png(image: np.ndarray) -> Path:
        from PIL import Image
        img = image
        if img.ndim == 3:
            # Pick view 0 if a multi-view image was passed in
            img = img[0] if img.shape[0] in (1, 2) else img.max(axis=-1)
        if img.dtype != np.uint8:
            mn, mx = float(img.min()), float(img.max())
            denom = (mx - mn) if (mx - mn) > 0 else 1.0
            img = ((img - mn) / denom * 255.0).clip(0, 255).astype(np.uint8)
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            path = Path(f.name)
        Image.fromarray(img).save(path)
        return path


def render_certificate_for_summary(cert: Optional[Dict[str, Any]]) -> str:
    """Tiny renderer used by the experiment summary line."""
    if not cert:
        return "not calibrated"
    verified = cert.get('verified')
    r_top = cert.get('r_squared_top')
    r_bot = cert.get('r_squared_bottom')
    parts = []
    if verified is True:
        parts.append("calibration verified")
    elif verified == 'pending':
        parts.append("calibration verification pending")
    else:
        parts.append("calibration has concerns")
    if r_top is not None and r_bot is not None:
        parts.append(f"R2 top={r_top:.2f} bot={r_bot:.2f}")
    concerns = cert.get('concerns') or []
    if concerns:
        parts.append(f"{len(concerns)} concern(s)")
    return " | ".join(parts)
