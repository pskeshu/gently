"""
Simple Perception Manager.

Orchestrates perception sessions for embryos.
No belief states, no schedulers, no anomaly detectors - just simple tracking.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import anthropic

from .session import PerceptionSession, PerceptionResult
from .engine import PerceptionEngine
from .example_store import ExampleStore

logger = logging.getLogger(__name__)


class PerceptionManager:
    """
    Simple manager for perception sessions.

    One session per embryo, tracks observations over time.
    """

    def __init__(
        self,
        claude_client: anthropic.Anthropic,
        examples_path: Path,
        event_bus: Optional[Any] = None,
    ):
        """
        Parameters
        ----------
        claude_client : anthropic.Anthropic
            Claude API client
        examples_path : Path
            Root directory for few-shot example images
        event_bus : EventBus, optional
            Event bus for emitting perception events
        """
        self.example_store = ExampleStore(examples_path)
        self.engine = PerceptionEngine(
            claude_client=claude_client,
            example_store=self.example_store,
        )
        self._event_bus = event_bus

        # Active sessions (one per embryo)
        self.sessions: Dict[str, PerceptionSession] = {}

        # Callbacks
        self._on_hatching_detected: Optional[Callable] = None
        self._on_hatched: Optional[Callable] = None

    def get_or_create_session(self, embryo_id: str) -> PerceptionSession:
        """Get existing session or create new one."""
        if embryo_id not in self.sessions:
            self.sessions[embryo_id] = PerceptionSession(
                embryo_id=embryo_id,
                created_at=datetime.now(),
            )
            logger.info(f"Created new perception session for {embryo_id}")

        return self.sessions[embryo_id]

    async def process_image(
        self,
        embryo_id: str,
        timepoint: int,
        image_b64: str,
        volume=None,
    ) -> PerceptionResult:
        """
        Process an image through the perception system.

        Parameters
        ----------
        embryo_id : str
            Embryo identifier
        timepoint : int
            Current timepoint number
        image_b64 : str
            Base64-encoded image
        volume : np.ndarray, optional
            3D volume data for view_embryo tool. If provided, enables
            rotated views during analysis.

        Returns
        -------
        PerceptionResult
            Stage classification and hatching status
        """
        session = self.get_or_create_session(embryo_id)

        # Skip if already hatched
        if session.is_complete():
            logger.info(f"[{embryo_id}] Already hatched, skipping perception")
            return PerceptionResult(
                stage="hatched",
                is_hatching=False,
                confidence=1.0,
                reasoning="Already hatched",
                should_stop=True,
            )

        # Run perception
        try:
            result = await self.engine.perceive(
                image_b64=image_b64,
                session=session,
                timepoint=timepoint,
                volume=volume,
            )
        except Exception as e:
            logger.error(f"Perception failed for {embryo_id}: {e}")
            return PerceptionResult(
                stage=session.get_current_stage() or "early",
                is_hatching=False,
                confidence=0.0,
                reasoning=f"Error: {e}",
                should_stop=False,
            )

        # Add observation to session
        session.add_observation(
            timepoint=timepoint,
            stage=result.stage,
            is_hatching=result.is_hatching,
            confidence=result.confidence,
            reasoning=result.reasoning,
            is_transitional=result.is_transitional,
            transition_between=result.transition_between,
        )

        # Handle hatching events
        if result.is_hatching and session.hatching_started_at == timepoint:
            # First time detecting hatching
            logger.info(f"[{embryo_id}] Hatching started at T{timepoint}")
            if self._on_hatching_detected:
                await self._on_hatching_detected(
                    embryo_id=embryo_id,
                    timepoint=timepoint,
                    confidence=result.confidence,
                )
            self._emit_hatching_event(embryo_id, timepoint, result)

        if result.stage == "hatched" and session.hatching_complete_at == timepoint:
            # First time detecting hatched
            logger.info(f"[{embryo_id}] Hatching complete at T{timepoint}")
            if self._on_hatched:
                await self._on_hatched(
                    embryo_id=embryo_id,
                    timepoint=timepoint,
                    confidence=result.confidence,
                )
            self._emit_hatched_event(embryo_id, timepoint, result)

        # Note: DETECTOR_EVALUATED event is emitted by timelapse_orchestrator
        # to avoid duplicate events and ensure volume_uid is included

        return result

    def _emit_hatching_event(
        self,
        embryo_id: str,
        timepoint: int,
        result: PerceptionResult,
    ) -> None:
        """Emit hatching detected event."""
        if not self._event_bus:
            return

        try:
            from ...core import EventType

            self._event_bus.publish(
                EventType.HATCHING_DETECTED,
                {
                    "embryo_id": embryo_id,
                    "timepoint": timepoint,
                    "confidence": result.confidence,
                },
                source="perception_manager",
            )
        except Exception as e:
            logger.debug(f"Failed to emit hatching event: {e}")

    def _emit_hatched_event(
        self,
        embryo_id: str,
        timepoint: int,
        result: PerceptionResult,
    ) -> None:
        """Emit hatched event."""
        if not self._event_bus:
            return

        try:
            from ...core import EventType

            # Try HATCHING_COMPLETE or fall back to HATCHING_DETECTED
            event_type = getattr(EventType, "HATCHING_COMPLETE", None)
            if event_type is None:
                event_type = EventType.HATCHING_DETECTED

            self._event_bus.publish(
                event_type,
                {
                    "embryo_id": embryo_id,
                    "timepoint": timepoint,
                    "confidence": result.confidence,
                    "stage": "hatched",
                },
                source="perception_manager",
            )
        except Exception as e:
            logger.debug(f"Failed to emit hatched event: {e}")

    def get_session(self, embryo_id: str) -> Optional[PerceptionSession]:
        """Get session for an embryo (if exists)."""
        return self.sessions.get(embryo_id)

    def get_current_stage(self, embryo_id: str) -> Optional[str]:
        """Get current stage for an embryo."""
        session = self.sessions.get(embryo_id)
        return session.get_current_stage() if session else None

    def get_all_sessions(self) -> Dict[str, PerceptionSession]:
        """Get all active sessions."""
        return self.sessions.copy()

    def clear_session(self, embryo_id: str) -> bool:
        """Clear session for an embryo (reset perception)."""
        if embryo_id in self.sessions:
            del self.sessions[embryo_id]
            return True
        return False

    def set_callbacks(
        self,
        on_hatching_detected: Optional[Callable] = None,
        on_hatched: Optional[Callable] = None,
    ) -> None:
        """Set callbacks for perception events."""
        self._on_hatching_detected = on_hatching_detected
        self._on_hatched = on_hatched

    def to_dict(self) -> Dict[str, Any]:
        """Serialize manager state for session persistence."""
        return {
            "sessions": {
                embryo_id: session.to_dict()
                for embryo_id, session in self.sessions.items()
            },
        }

    def restore_sessions(self, data: Dict[str, Any]) -> None:
        """Restore sessions from serialized data."""
        sessions_data = data.get("sessions", {})
        for embryo_id, session_dict in sessions_data.items():
            try:
                self.sessions[embryo_id] = PerceptionSession.from_dict(session_dict)
                logger.info(f"Restored perception session for {embryo_id}")
            except Exception as e:
                logger.warning(f"Failed to restore session for {embryo_id}: {e}")


# Backwards compatibility alias
async def process_volume(
    manager: PerceptionManager,
    embryo_id: str,
    timepoint: int,
    current_image_b64: str,
    **kwargs,  # Ignore extra args from old interface
) -> PerceptionResult:
    """Backwards compatible wrapper."""
    return await manager.process_image(
        embryo_id=embryo_id,
        timepoint=timepoint,
        image_b64=current_image_b64,
    )
