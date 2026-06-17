"""
Structured Interaction Logger for Agent Sessions

Logs structured records of all agent interactions for:
- Post-hoc analysis of usage patterns
- Identifying recurring issues and corrections
- Research into self-improving agent systems

Each interaction captures:
- User prompt and system state at time of request
- Tool calls made and their results
- Any errors encountered
- Whether the user corrected the behavior in the next turn
"""

import json
import logging
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class ToolCallRecord:
    """Record of a single tool call"""

    tool_name: str
    tool_input: dict[str, Any]
    result: str
    duration_seconds: float
    is_error: bool = False
    error_message: str | None = None


@dataclass
class InteractionRecord:
    """
    Complete record of a single user<->agent interaction

    An interaction is one user message and the agent's response,
    including any tool calls made during that response.
    """

    # Unique ID for this interaction
    interaction_id: str

    # The prompt
    user_prompt: str
    timestamp: datetime

    # System state snapshot at time of request
    system_state: dict[str, Any] = field(default_factory=dict)

    # What happened
    tool_calls: list[ToolCallRecord] = field(default_factory=list)
    assistant_response: str = ""
    total_duration_seconds: float = 0.0

    # Errors
    error: str | None = None
    error_traceback: str | None = None

    # Correction detection (filled in after next turn)
    was_corrected: bool = False
    correction_prompt: str | None = None
    correction_indicators: list[str] = field(default_factory=list)

    # Metadata
    session_id: str = ""
    codebase_version: str = ""
    model: str = ""

    def to_dict(self) -> dict:
        """Serialize to dictionary for JSON storage"""
        d = asdict(self)
        # Convert datetime to ISO format
        d["timestamp"] = self.timestamp.isoformat()
        # Convert tool calls
        d["tool_calls"] = [asdict(tc) for tc in self.tool_calls]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "InteractionRecord":
        """Deserialize from dictionary"""
        d = d.copy()
        d["timestamp"] = datetime.fromisoformat(d["timestamp"])
        d["tool_calls"] = [ToolCallRecord(**tc) for tc in d.get("tool_calls", [])]
        return cls(**d)


class InteractionLogger:
    """
    Logs structured interaction records for agent sessions

    Automatically detects correction patterns to identify when
    the agent's response didn't match user intent.
    """

    # Phrases that indicate the user is correcting the agent
    CORRECTION_INDICATORS = [
        "no,",
        "no ",
        "not that",
        "i meant",
        "i said",
        "wrong",
        "incorrect",
        "that's not",
        "thats not",
        "don't",
        "dont",
        "stop",
        "cancel",
        "undo",
        "actually",
        "instead",
        "try again",
        "let me clarify",
        "what i meant",
        "i wanted",
    ]

    def __init__(
        self,
        storage_path: Path,
        session_id: str,
        model: str = "",
    ):
        """
        Parameters
        ----------
        storage_path : Path
            Base directory for storing interaction logs
        session_id : str
            Current session ID
        model : str
            Claude model being used
        """
        self.storage_path = Path(storage_path)
        self.session_id = session_id
        self.model = model

        # Create storage directory
        self.logs_dir = self.storage_path / "interaction_logs"
        self.logs_dir.mkdir(parents=True, exist_ok=True)

        # Current session log file
        self.log_file = self.logs_dir / f"{session_id}.jsonl"

        # In-memory buffer of recent interactions (for correction detection)
        self._recent_interactions: list[InteractionRecord] = []
        self._max_recent = 10

        # Get codebase version (git commit)
        self._codebase_version = self._get_git_version()

        # Interaction counter for unique IDs
        self._interaction_count = 0

        logger.info(f"InteractionLogger initialized: {self.log_file}")

    def _get_git_version(self) -> str:
        """Get current git commit hash"""
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--short", "HEAD"],
                capture_output=True,
                text=True,
                cwd=str(self.storage_path.parent),
                timeout=5,
            )
            if result.returncode == 0:
                return result.stdout.strip()
        except Exception:
            pass
        return "unknown"

    def start_interaction(
        self,
        user_prompt: str,
        system_state: dict[str, Any],
    ) -> InteractionRecord:
        """
        Start recording a new interaction

        Call this when a user message is received, before processing.

        Parameters
        ----------
        user_prompt : str
            The user's message
        system_state : dict
            Current system state (embryos, detectors, etc.)

        Returns
        -------
        InteractionRecord
            The new interaction record (will be updated as processing continues)
        """
        self._interaction_count += 1

        interaction = InteractionRecord(
            interaction_id=f"{self.session_id}_{self._interaction_count:04d}",
            user_prompt=user_prompt,
            timestamp=datetime.now(),
            system_state=self._sanitize_state(system_state),
            session_id=self.session_id,
            codebase_version=self._codebase_version,
            model=self.model,
        )

        return interaction

    def record_tool_call(
        self,
        interaction: InteractionRecord,
        tool_name: str,
        tool_input: dict[str, Any],
        result: str,
        duration_seconds: float,
        is_error: bool = False,
        error_message: str | None = None,
    ):
        """
        Record a tool call within an interaction

        Parameters
        ----------
        interaction : InteractionRecord
            The interaction being recorded
        tool_name : str
            Name of the tool called
        tool_input : dict
            Tool input parameters
        result : str
            Tool result (may be truncated for large results)
        duration_seconds : float
            How long the tool took
        is_error : bool
            Whether the tool call failed
        error_message : str, optional
            Error message if failed
        """
        # Truncate large results
        if len(result) > 2000:
            result = result[:2000] + f"... [truncated, total {len(result)} chars]"

        # Sanitize input (remove large/binary data)
        sanitized_input = self._sanitize_tool_input(tool_input)

        tool_record = ToolCallRecord(
            tool_name=tool_name,
            tool_input=sanitized_input,
            result=result,
            duration_seconds=duration_seconds,
            is_error=is_error,
            error_message=error_message,
        )

        interaction.tool_calls.append(tool_record)

    def complete_interaction(
        self,
        interaction: InteractionRecord,
        assistant_response: str,
        total_duration_seconds: float,
        error: str | None = None,
        error_traceback: str | None = None,
    ):
        """
        Complete and save an interaction record

        Parameters
        ----------
        interaction : InteractionRecord
            The interaction to complete
        assistant_response : str
            The final assistant response
        total_duration_seconds : float
            Total time for the interaction
        error : str, optional
            Error message if the interaction failed
        error_traceback : str, optional
            Full traceback if error occurred
        """
        interaction.assistant_response = assistant_response
        interaction.total_duration_seconds = total_duration_seconds
        interaction.error = error
        interaction.error_traceback = error_traceback

        # Check if previous interaction was corrected by this one
        self._detect_correction(interaction)

        # Add to recent buffer
        self._recent_interactions.append(interaction)
        if len(self._recent_interactions) > self._max_recent:
            self._recent_interactions.pop(0)

        # Save to disk
        self._save_interaction(interaction)

        logger.debug(
            f"Logged interaction {interaction.interaction_id}: "
            f"{len(interaction.tool_calls)} tools, "
            f"{total_duration_seconds:.2f}s"
        )

    def _detect_correction(self, current: InteractionRecord):
        """
        Check if current interaction is correcting the previous one

        Updates the previous interaction's was_corrected field.
        """
        if not self._recent_interactions:
            return

        previous = self._recent_interactions[-1]
        prompt_lower = current.user_prompt.lower()

        # Check for correction indicators
        indicators_found = []
        for indicator in self.CORRECTION_INDICATORS:
            if indicator in prompt_lower:
                indicators_found.append(indicator)

        if indicators_found:
            # Mark previous as corrected
            previous.was_corrected = True
            previous.correction_prompt = current.user_prompt
            previous.correction_indicators = indicators_found

            # Re-save the updated previous interaction
            self._save_interaction(previous, append=False)

            logger.info(
                f"Detected correction: {previous.interaction_id} "
                f"corrected by {current.interaction_id} "
                f"(indicators: {indicators_found})"
            )

    def _save_interaction(self, interaction: InteractionRecord, append: bool = True):
        """Save interaction to JSONL file"""
        try:
            if append:
                # Append to log file
                with open(self.log_file, "a", encoding="utf-8") as f:
                    f.write(json.dumps(interaction.to_dict()) + "\n")
            else:
                # Need to update existing record - rewrite file
                # This is less efficient but corrections are rare
                self._rewrite_with_update(interaction)
        except Exception as e:
            logger.error(f"Failed to save interaction: {e}")

    def _rewrite_with_update(self, updated: InteractionRecord):
        """Rewrite log file with updated interaction"""
        if not self.log_file.exists():
            return

        # Read all interactions
        interactions = []
        try:
            with open(self.log_file, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        record = InteractionRecord.from_dict(json.loads(line))
                        if record.interaction_id == updated.interaction_id:
                            interactions.append(updated)
                        else:
                            interactions.append(record)
        except Exception as e:
            logger.error(f"Failed to read log file for update: {e}")
            return

        # Rewrite file
        try:
            with open(self.log_file, "w", encoding="utf-8") as f:
                for record in interactions:
                    f.write(json.dumps(record.to_dict()) + "\n")
        except Exception as e:
            logger.error(f"Failed to rewrite log file: {e}")

    def _sanitize_state(self, state: dict[str, Any]) -> dict[str, Any]:
        """Remove large/sensitive data from state snapshot"""
        sanitized: dict[str, Any] = {}

        # Keep summary info
        if "embryos" in state:
            sanitized["embryo_count"] = len(state["embryos"])
            sanitized["embryo_ids"] = list(state["embryos"].keys())

        if "detectors" in state:
            sanitized["detector_count"] = len(state["detectors"])

        if "acquisition_status" in state:
            sanitized["acquisition_status"] = state["acquisition_status"]

        return sanitized

    def _sanitize_tool_input(self, tool_input: dict[str, Any]) -> dict[str, Any]:
        """Remove large/binary data from tool input"""
        sanitized: dict[str, Any] = {}
        for key, value in tool_input.items():
            if isinstance(value, (str, int, float, bool, type(None))):
                if isinstance(value, str) and len(value) > 500:
                    sanitized[key] = value[:500] + "..."
                else:
                    sanitized[key] = value
            elif isinstance(value, list):
                if len(value) > 10:
                    sanitized[key] = f"[list of {len(value)} items]"
                else:
                    sanitized[key] = value
            elif isinstance(value, dict):
                sanitized[key] = f"[dict with {len(value)} keys]"
            else:
                sanitized[key] = f"[{type(value).__name__}]"
        return sanitized

    def get_session_stats(self) -> dict[str, Any]:
        """Get statistics for current session"""
        if not self.log_file.exists():
            return {
                "total_interactions": 0,
                "corrections": 0,
                "errors": 0,
                "tool_calls": 0,
            }

        total = 0
        corrections = 0
        errors = 0
        tool_calls = 0

        try:
            with open(self.log_file, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        record = json.loads(line)
                        total += 1
                        if record.get("was_corrected"):
                            corrections += 1
                        if record.get("error"):
                            errors += 1
                        tool_calls += len(record.get("tool_calls", []))
        except Exception:
            pass

        return {
            "total_interactions": total,
            "corrections": corrections,
            "errors": errors,
            "tool_calls": tool_calls,
            "correction_rate": corrections / total if total > 0 else 0,
        }

    def load_session_interactions(self) -> list[InteractionRecord]:
        """Load all interactions from current session"""
        if not self.log_file.exists():
            return []

        interactions = []
        try:
            with open(self.log_file, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        record = InteractionRecord.from_dict(json.loads(line))
                        interactions.append(record)
        except Exception as e:
            logger.error(f"Failed to load interactions: {e}")

        return interactions
