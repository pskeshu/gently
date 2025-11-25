"""
Logging infrastructure for Microscopy Copilot

Provides dual output: Rich console (with colors) + plain text log file.
All session output is saved for later review.
"""

from datetime import datetime
from pathlib import Path
from typing import Optional, Any

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich import box

from .theme import get_theme


class CopilotLogger:
    """
    Unified logger with Rich console and file output

    All console output is mirrored to a plain-text log file
    with timestamps for later review.
    """

    def __init__(
        self,
        log_dir: Path,
        session_name: Optional[str] = None,
        console: Optional[Console] = None
    ):
        """
        Initialize the logger

        Parameters
        ----------
        log_dir : Path
            Directory for log files
        session_name : str, optional
            Custom session identifier
        console : Console, optional
            Existing Rich console to use
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # Generate session-specific log filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session = session_name or "session"
        self.log_file = self.log_dir / f"copilot_{session}_{timestamp}.log"

        # Rich console for terminal output (with colors)
        self.console = console or Console()

        # File handle for logging
        self._file_handle = open(self.log_file, 'w', encoding='utf-8')

        # File console (no colors, for logging)
        self.file_console = Console(
            file=self._file_handle,
            force_terminal=False,
            no_color=True,
            width=120,
        )

        # Track session start
        self.session_start = datetime.now()

        # Write log header
        self._write_header()

    def _write_header(self):
        """Write session header to log file"""
        theme = get_theme()
        header = f"""================================================================================
MICROSCOPY COPILOT SESSION LOG
Started: {self.session_start.strftime('%Y-%m-%d %H:%M:%S')}
Theme: {theme.name}
================================================================================

"""
        self._file_handle.write(header)
        self._file_handle.flush()

    def _timestamp(self) -> str:
        """Get current timestamp string"""
        return datetime.now().strftime("%H:%M:%S")

    def _log_to_file(self, category: str, message: str):
        """Write a line to the log file"""
        ts = self._timestamp()
        # Format with fixed-width category column
        lines = message.split('\n')
        first_line = f"[{ts}] {category:8}| {lines[0]}\n"
        self._file_handle.write(first_line)

        # Continuation lines
        for line in lines[1:]:
            self._file_handle.write(f"{'':12}| {line}\n")

        self._file_handle.flush()

    def print(self, *args, **kwargs):
        """Print to both console and log file"""
        # Print to terminal with colors
        self.console.print(*args, **kwargs)

        # Print to file without colors
        self.file_console.print(*args, **kwargs)

    def log_system(self, message: str):
        """Log a system message"""
        theme = get_theme()
        self._log_to_file("SYSTEM", message)
        self.console.print(f"[{theme.system}]{theme.icon_system}[/] {message}")

    def log_device(self, device_name: str, status: str, message: str = ""):
        """Log a device status"""
        theme = get_theme()
        icon = theme.icon_success if status == "success" else theme.icon_error if status == "error" else theme.icon_warning
        color = theme.success if status == "success" else theme.error if status == "error" else theme.warning

        full_msg = f"{icon} {device_name}"
        if message:
            full_msg += f" - {message}"

        self._log_to_file("DEVICE", full_msg)
        self.console.print(f"  [{color}]{icon}[/] [bold]{device_name}[/]" + (f" - {message}" if message else ""))

    def log_user(self, message: str):
        """Log user input"""
        self._log_to_file("USER", message)

    def log_copilot(self, message: str):
        """Log copilot response"""
        self._log_to_file("COPILOT", message)

    def log_tool(self, tool_name: str, params: dict, duration: Optional[float] = None):
        """Log a tool call"""
        lines = [tool_name]
        for key, value in params.items():
            lines.append(f"  {key}: {value}")
        if duration is not None:
            lines.append(f"  duration: {duration:.2f}s")

        self._log_to_file("TOOL", "\n".join(lines))

    def log_error(self, message: str, traceback: Optional[str] = None):
        """Log an error"""
        theme = get_theme()
        self._log_to_file("ERROR", message)
        if traceback:
            self._log_to_file("ERROR", traceback)

        self.console.print(f"[{theme.error}]{theme.icon_error} {message}[/]")

    def close(self):
        """Close file handles"""
        # Write footer
        duration = datetime.now() - self.session_start
        footer = f"""
================================================================================
SESSION ENDED
Duration: {duration}
Log file: {self.log_file}
================================================================================
"""
        self._file_handle.write(footer)
        self._file_handle.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False
