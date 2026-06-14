"""
Global Error Log for Timelapse Acquisition

Tracks errors across embryos and timepoints for cross-embryo correlation
and debugging system-level issues.
"""

import logging
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class ErrorEntry:
    """Single error entry in the global log"""

    timestamp: datetime
    round_number: int
    embryo_id: str
    timepoint: int
    error_type: str
    message: str
    exception: Exception | None = None


class GlobalErrorLog:
    """
    Global error log for tracking errors across all embryos during timelapse.

    Used to detect system-level issues (e.g., all embryos failing at the same time)
    and provide context for debugging.
    """

    def __init__(self, max_entries: int = 100):
        """
        Parameters
        ----------
        max_entries : int
            Maximum number of entries to keep (oldest are dropped)
        """
        self._entries: list[ErrorEntry] = []
        self._max_entries = max_entries

    def log_error(
        self,
        round_number: int,
        embryo_id: str,
        timepoint: int,
        error_type: str,
        message: str,
        exception: Exception | None = None,
    ):
        """
        Log an error during timelapse acquisition.

        Parameters
        ----------
        round_number : int
            Current acquisition round
        embryo_id : str
            Embryo where error occurred
        timepoint : int
            Timepoint at which error occurred
        error_type : str
            Category of error (acquisition, perception, etc.)
        message : str
            Human-readable error message
        exception : Exception, optional
            The underlying exception if available
        """
        entry = ErrorEntry(
            timestamp=datetime.now(),
            round_number=round_number,
            embryo_id=embryo_id,
            timepoint=timepoint,
            error_type=error_type,
            message=message,
            exception=exception,
        )

        self._entries.append(entry)

        # Trim old entries
        if len(self._entries) > self._max_entries:
            self._entries = self._entries[-self._max_entries :]

        # Also log to standard logger
        logger.warning(f"[{error_type}] Round {round_number}, {embryo_id} t{timepoint}: {message}")

    def get_recent_errors(self, limit: int = 10) -> list[ErrorEntry]:
        """Get most recent errors"""
        return self._entries[-limit:]

    def get_errors_for_embryo(self, embryo_id: str) -> list[ErrorEntry]:
        """Get all errors for a specific embryo"""
        return [e for e in self._entries if e.embryo_id == embryo_id]

    def get_errors_in_round(self, round_number: int) -> list[ErrorEntry]:
        """Get all errors from a specific round"""
        return [e for e in self._entries if e.round_number == round_number]

    def compile_for_verification(self) -> str:
        """
        Compile error log into a format suitable for LLM verification prompts.

        Returns
        -------
        str
            Formatted error log for context
        """
        if not self._entries:
            return "No errors recorded."

        lines = ["Recent errors:"]
        for entry in self._entries[-10:]:
            lines.append(
                f"  - [{entry.error_type}] {entry.embryo_id} t{entry.timepoint}: {entry.message}"
            )
        return "\n".join(lines)

    def clear(self):
        """Clear all entries"""
        self._entries.clear()

    @property
    def error_count(self) -> int:
        """Total number of errors logged"""
        return len(self._entries)

    def has_system_wide_errors(self, round_number: int, threshold: int = 2) -> bool:
        """
        Check if there are system-wide errors (multiple embryos failing in same round).

        Parameters
        ----------
        round_number : int
            Round to check
        threshold : int
            Number of embryos that must fail to consider it system-wide

        Returns
        -------
        bool
            True if system-wide issue detected
        """
        round_errors = self.get_errors_in_round(round_number)
        unique_embryos = set(e.embryo_id for e in round_errors)
        return len(unique_embryos) >= threshold
