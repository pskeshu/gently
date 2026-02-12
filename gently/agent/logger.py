"""
Logging infrastructure for Microscopy Copilot

Provides structured plain-text log file output.
All session output is saved for later review.
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


class CopilotLogger:
    """
    Session logger with structured plain-text file output.

    All copilot activity is written to a timestamped log file
    for later review and debugging.
    """

    def __init__(
        self,
        log_dir: Path,
        session_name: Optional[str] = None,
    ):
        """
        Initialize the logger.

        Parameters
        ----------
        log_dir : Path
            Directory for log files.
        session_name : str, optional
            Custom session identifier.
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Generate session-specific log filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session = session_name or "session"
        self.log_file = self.log_dir / f"copilot_{session}_{timestamp}.log"

        # File handle for logging
        self._file_handle = open(self.log_file, "w", encoding="utf-8")

        # Track session start
        self.session_start = datetime.now()

        # Write log header
        self._write_header()

    def _write_header(self):
        """Write session header to log file."""
        header = (
            "=" * 80 + "\n"
            "MICROSCOPY COPILOT SESSION LOG\n"
            f"Started: {self.session_start.strftime('%Y-%m-%d %H:%M:%S')}\n"
            "=" * 80 + "\n\n"
        )
        self._file_handle.write(header)
        self._file_handle.flush()

    def _timestamp(self) -> str:
        """Get current timestamp string."""
        return datetime.now().strftime("%H:%M:%S")

    def _log_to_file(self, category: str, message: str):
        """Write a line to the log file."""
        ts = self._timestamp()
        lines = message.split("\n")
        first_line = f"[{ts}] {category:8}| {lines[0]}\n"
        self._file_handle.write(first_line)

        for line in lines[1:]:
            self._file_handle.write(f"{'':12}| {line}\n")

        self._file_handle.flush()

    def log_system(self, message: str):
        """Log a system message."""
        self._log_to_file("SYSTEM", message)
        logger.info(message)

    def log_device(self, device_name: str, status: str, message: str = ""):
        """Log a device status."""
        full_msg = f"{device_name}"
        if message:
            full_msg += f" - {message}"
        self._log_to_file("DEVICE", full_msg)
        logger.info("Device %s: %s %s", device_name, status, message)

    def log_user(self, message: str):
        """Log user input."""
        self._log_to_file("USER", message)

    def log_copilot(self, message: str):
        """Log copilot response."""
        self._log_to_file("COPILOT", message)

    def log_tool(self, tool_name: str, params: dict, duration: Optional[float] = None):
        """Log a tool call."""
        lines = [tool_name]
        for key, value in params.items():
            lines.append(f"  {key}: {value}")
        if duration is not None:
            lines.append(f"  duration: {duration:.2f}s")
        self._log_to_file("TOOL", "\n".join(lines))

    def log_error(self, message: str, traceback: Optional[str] = None):
        """Log an error."""
        self._log_to_file("ERROR", message)
        if traceback:
            self._log_to_file("ERROR", traceback)
        logger.error(message)

    def close(self):
        """Close file handles."""
        duration = datetime.now() - self.session_start
        footer = (
            "\n" + "=" * 80 + "\n"
            "SESSION ENDED\n"
            f"Duration: {duration}\n"
            f"Log file: {self.log_file}\n"
            "=" * 80 + "\n"
        )
        self._file_handle.write(footer)
        self._file_handle.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
