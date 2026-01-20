"""
Animated startup sequence for Microscopy Copilot

Provides a visually appealing initialization display with
progress bars and device status reporting.
"""

from typing import Dict, List, Tuple, Optional, Generator, Callable, Any
from datetime import datetime

from rich.console import Console
from rich.panel import Panel
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TaskProgressColumn,
    TimeElapsedColumn,
)
from rich.table import Table
from rich.text import Text
from rich import box

from .theme import get_theme
from .logger import CopilotLogger


class StartupSequence:
    """
    Animated startup sequence with hardware initialization progress

    Displays a banner, progress bar during device creation, and
    a summary table of connected devices.
    """

    # Default device initialization steps
    DEFAULT_STEPS = [
        ("core", "Micro-Manager Core"),
        ("scanner", "Scanner Device"),
        ("piezo", "Piezo Stage"),
        ("camera", "Lightsheet Camera"),
        ("laser", "Laser Control"),
        ("xy_stage", "XY Stage"),
        ("led", "LED Illumination"),
        ("bottom_cam", "Bottom Camera"),
        ("volume", "Volume Scanner"),
        ("lightsheet", "Lightsheet Snap"),
        ("bluesky", "Bluesky RunEngine"),
        ("databroker", "Databroker"),
    ]

    def __init__(
        self,
        console: Optional[Console] = None,
        logger: Optional[CopilotLogger] = None
    ):
        """
        Initialize startup sequence

        Parameters
        ----------
        console : Console, optional
            Rich console for output
        logger : CopilotLogger, optional
            Logger for file output
        """
        self.console = console or Console()
        self.logger = logger
        self.device_status: Dict[str, Tuple[str, str]] = {}  # device_id -> (status, message)

    def _log(self, message: str):
        """Log to both console and file if logger is available"""
        if self.logger:
            self.logger.log_system(message)

    def show_banner(self):
        """Display the startup banner"""
        theme = get_theme()

        banner_text = Text()
        banner_text.append("\n")
        banner_text.append("        M I C R O S C O P Y   C O P I L O T\n", style=f"bold {theme.primary}")
        banner_text.append("        ", style="dim")
        banner_text.append("─" * 39 + "\n", style=f"dim {theme.secondary}")
        banner_text.append("        AI-Powered Adaptive diSPIM Control\n", style=theme.muted)
        banner_text.append("\n")

        panel = Panel(
            banner_text,
            box=box.DOUBLE,
            border_style=theme.primary,
            padding=(0, 2),
        )
        self.console.print(panel)

    def run_with_progress(
        self,
        init_generator: Generator[Tuple[str, str, str, Optional[str]], None, None],
        total_steps: Optional[int] = None
    ) -> Dict[str, Tuple[str, str]]:
        """
        Run initialization with animated progress bar

        Parameters
        ----------
        init_generator : Generator
            Generator yielding (step_id, step_name, status, message) tuples
            Status should be: 'loading', 'success', 'error', 'skipped'
        total_steps : int, optional
            Total number of steps (for progress bar)

        Returns
        -------
        dict
            Device status dictionary: {device_id: (status, message)}
        """
        theme = get_theme()

        if total_steps is None:
            total_steps = len(self.DEFAULT_STEPS)

        self.console.print()  # Spacing

        with Progress(
            SpinnerColumn("dots12", style=theme.primary),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(
                bar_width=30,
                style=theme.secondary,
                complete_style=theme.success,
                finished_style=theme.success,
            ),
            TaskProgressColumn(),
            TimeElapsedColumn(),
            console=self.console,
            transient=False,
        ) as progress:
            main_task = progress.add_task(
                f"[{theme.info}]Initializing...[/]",
                total=total_steps
            )

            for step_id, step_name, status, message in init_generator:
                # Update description
                progress.update(
                    main_task,
                    description=f"[{theme.info}]Loading {step_name}...[/]"
                )

                # Store status
                self.device_status[step_id] = (status, message or "")

                # Log the result
                if self.logger:
                    self.logger.log_device(step_name, status, message or "")

                # Advance progress
                progress.advance(main_task)

            # Final update
            progress.update(
                main_task,
                description=f"[{theme.success}]Initialization complete[/]"
            )

        self.console.print()  # Spacing

        return self.device_status

    def show_summary(self):
        """Display summary table of device status"""
        theme = get_theme()

        # Count statuses
        success_count = sum(1 for s, _ in self.device_status.values() if s == "success")
        error_count = sum(1 for s, _ in self.device_status.values() if s == "error")
        skipped_count = sum(1 for s, _ in self.device_status.values() if s == "skipped")

        # Create summary table
        table = Table(
            show_header=True,
            header_style=f"bold {theme.primary}",
            box=box.ROUNDED,
            border_style=theme.muted,
        )
        table.add_column("Device", style="bold")
        table.add_column("Status", justify="center")
        table.add_column("Details", style=theme.muted)

        for device_id, (status, message) in self.device_status.items():
            if status == "success":
                status_text = Text(f"{theme.icon_success} Ready", style=theme.success)
            elif status == "error":
                status_text = Text(f"{theme.icon_error} Error", style=theme.error)
            elif status == "skipped":
                status_text = Text(f"{theme.icon_warning} Skipped", style=theme.warning)
            else:
                status_text = Text(f"{theme.icon_info} Unknown", style=theme.muted)

            table.add_row(device_id, status_text, message)

        self.console.print(table)

        # Print summary line
        summary = Text()
        summary.append(f"\n  {theme.icon_success} ", style=theme.success)
        summary.append(f"{success_count} ready", style=theme.success)

        if error_count > 0:
            summary.append(f"  {theme.icon_error} ", style=theme.error)
            summary.append(f"{error_count} failed", style=theme.error)

        if skipped_count > 0:
            summary.append(f"  {theme.icon_warning} ", style=theme.warning)
            summary.append(f"{skipped_count} skipped", style=theme.warning)

        summary.append("\n")
        self.console.print(summary)

    def show_log_location(self, log_file):
        """Show where the log file is saved"""
        theme = get_theme()
        self.console.print(f"[{theme.muted}]Log: {log_file}[/]\n")


def create_simple_init_generator(
    device_factory_func: Callable,
    *args,
    **kwargs
) -> Generator[Tuple[str, str, str, Optional[str]], None, Dict]:
    """
    Helper to create an init generator from a device factory function

    This wraps the device creation process to yield progress updates.

    Parameters
    ----------
    device_factory_func : Callable
        Function that creates devices and returns a dict
    *args, **kwargs
        Arguments to pass to the factory function

    Yields
    ------
    tuple
        (step_id, step_name, status, message)
    """
    # This is a simplified version - the actual implementation
    # would need to hook into the device factory more deeply
    yield ("core", "Micro-Manager Core", "loading", None)

    try:
        devices = device_factory_func(*args, **kwargs)
        yield ("core", "Micro-Manager Core", "success", "Connected")

        for device_id, device in devices.items():
            yield (device_id, device_id, "success", type(device).__name__)

    except Exception as e:
        yield ("core", "Micro-Manager Core", "error", str(e))
