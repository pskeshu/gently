"""
CV Agent Terminal Display

Rich terminal display for the CV subagent service.
Shows streaming thinking, tool calls, and analysis results.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.markdown import Markdown
from rich import box

# Import theme from copilot for consistency
try:
    from gently.agent.theme import get_theme, Theme
except ImportError:
    # Fallback if theme not available
    get_theme = None
    Theme = None


class CVAgentDisplay:
    """
    Rich terminal display for CV Agent

    Runs in the CV service terminal (separate from copilot).
    Displays:
    - Task start/completion
    - Streaming thinking/reasoning
    - Tool calls with parameters and results
    """

    def __init__(self, console: Optional[Console] = None):
        self.console = console or Console()
        self._current_task_id: Optional[str] = None

    def _get_colors(self) -> Dict[str, str]:
        """Get colors from theme or use defaults"""
        if get_theme:
            theme = get_theme()
            return {
                "primary": theme.primary,
                "tool": theme.tool,
                "success": theme.success,
                "error": theme.error,
                "info": theme.info,
                "warning": theme.warning,
                "muted": theme.muted,
                "copilot": theme.copilot,
                "accent": theme.accent,
            }
        # Fallback colors (vibrant theme)
        return {
            "primary": "#7C3AED",
            "tool": "#EC4899",
            "success": "#22C55E",
            "error": "#EF4444",
            "info": "#06B6D4",
            "warning": "#EAB308",
            "muted": "dim #9CA3AF",
            "copilot": "#3B82F6",
            "accent": "#F59E0B",
        }

    def print_welcome(self):
        """Print CV Agent welcome banner"""
        colors = self._get_colors()
        banner = Panel(
            Text.from_markup(
                f"[bold {colors['primary']}]CV Agent[/] - Computer Vision Analysis\n"
                f"[{colors['muted']}]Claude-powered embryo analysis[/]"
            ),
            border_style=colors["primary"],
            box=box.DOUBLE,
            padding=(0, 2),
        )
        self.console.print(banner)
        self.console.print()

    def print_task_started(self, task_id: str, intent: str, embryo_id: str):
        """Print task start notification"""
        colors = self._get_colors()
        timestamp = datetime.now().strftime("%H:%M:%S")
        self._current_task_id = task_id

        # Truncate task_id for display
        task_short = task_id[:8] if len(task_id) > 8 else task_id

        content = Text()
        content.append("Task: ", style="bold")
        content.append(f"{task_short}\n", style=colors["muted"])
        content.append("Intent: ", style="bold")
        content.append(f"{intent}\n", style="white")
        content.append("Embryo: ", style="bold")
        content.append(embryo_id, style=colors["accent"])

        panel = Panel(
            content,
            title=f"[{colors['muted']}]{timestamp}[/] [{colors['info']}]Analysis Started[/]",
            title_align="left",
            border_style=colors["info"],
            box=box.ROUNDED,
            padding=(0, 1),
        )
        self.console.print(panel)

    def print_thinking(self, thinking: str, iteration: int):
        """Print agent thinking/reasoning block"""
        colors = self._get_colors()

        # Truncate very long thinking for readability
        max_len = 400
        display_text = thinking[:max_len] + "..." if len(thinking) > max_len else thinking

        # Clean up the text
        display_text = display_text.strip()

        panel = Panel(
            Text(display_text, style=colors["muted"]),
            title=f"[{colors['copilot']}]Thinking[/] [{colors['muted']}](iter {iteration})[/]",
            title_align="left",
            border_style=colors["copilot"],
            box=box.SIMPLE,
            padding=(0, 1),
        )
        self.console.print(panel)

    def print_tool_call(self, tool_name: str, tool_input: Dict[str, Any]):
        """Print tool call with parameters"""
        colors = self._get_colors()
        timestamp = datetime.now().strftime("%H:%M:%S")

        # Build parameter display
        content = Text()
        content.append(f"{tool_name}\n", style=f"bold {colors['tool']}")

        for key, value in tool_input.items():
            content.append(f"  {key}: ", style="bold")
            # Truncate long values
            str_val = str(value)
            if len(str_val) > 60:
                str_val = str_val[:60] + "..."
            content.append(f"{str_val}\n", style=colors["muted"])

        panel = Panel(
            content,
            title=f"[{colors['muted']}]{timestamp}[/] [{colors['tool']}]Tool[/]",
            title_align="left",
            border_style=colors["tool"],
            box=box.SIMPLE,
            padding=(0, 1),
        )
        self.console.print(panel)

    def print_tool_result(
        self,
        tool_name: str,
        result: Any,
        is_error: bool = False,
        duration_ms: Optional[float] = None
    ):
        """Print tool result (success or error)"""
        colors = self._get_colors()

        if is_error:
            style = colors["error"]
            label = "Error"
        else:
            style = colors["success"]
            label = "Result"

        # Format result for display
        if isinstance(result, dict):
            # Show key metrics from result
            lines = []
            for key, value in list(result.items())[:5]:  # Limit to 5 items
                str_val = str(value)
                if len(str_val) > 50:
                    str_val = str_val[:50] + "..."
                lines.append(f"  {key}: {str_val}")
            result_text = "\n".join(lines)
            if len(result) > 5:
                result_text += f"\n  ... +{len(result) - 5} more"
        else:
            result_text = str(result)[:200]

        content = Text()
        content.append(f"{tool_name} ", style="bold")
        if duration_ms:
            content.append(f"({duration_ms:.0f}ms)\n", style=colors["muted"])
        else:
            content.append("\n")
        content.append(result_text, style=style if is_error else colors["muted"])

        panel = Panel(
            content,
            title=f"[{style}]{label}[/]",
            title_align="left",
            border_style=style,
            box=box.SIMPLE,
            padding=(0, 1),
        )
        self.console.print(panel)

    def print_task_completed(
        self,
        task_id: str,
        summary: str,
        duration_ms: float,
        tools_used: List[str],
        iterations: int
    ):
        """Print task completion summary"""
        colors = self._get_colors()
        timestamp = datetime.now().strftime("%H:%M:%S")

        # Build stats line
        stats = Text()
        stats.append("Duration: ", style="bold")
        stats.append(f"{duration_ms/1000:.1f}s", style=colors["accent"])
        stats.append("  |  ", style=colors["muted"])
        stats.append("Iterations: ", style="bold")
        stats.append(f"{iterations}", style=colors["accent"])
        stats.append("  |  ", style=colors["muted"])
        stats.append("Tools: ", style="bold")
        stats.append(", ".join(tools_used) if tools_used else "none", style=colors["tool"])

        self.console.print(stats)
        self.console.print()

        # Summary panel
        panel = Panel(
            Markdown(summary),
            title=f"[{colors['muted']}]{timestamp}[/] [{colors['success']}]Analysis Complete[/]",
            title_align="left",
            border_style=colors["success"],
            box=box.ROUNDED,
            padding=(0, 1),
        )
        self.console.print(panel)
        self._current_task_id = None

    def print_task_failed(self, task_id: str, error: str):
        """Print task failure"""
        colors = self._get_colors()
        timestamp = datetime.now().strftime("%H:%M:%S")

        panel = Panel(
            Text(error, style=colors["error"]),
            title=f"[{colors['muted']}]{timestamp}[/] [{colors['error']}]Analysis Failed[/]",
            title_align="left",
            border_style=colors["error"],
            box=box.ROUNDED,
            padding=(0, 1),
        )
        self.console.print(panel)
        self._current_task_id = None

    def print_status(self, message: str, style: str = "info"):
        """Print a status message"""
        colors = self._get_colors()
        color = colors.get(style, colors["info"])
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.console.print(f"[{colors['muted']}]{timestamp}[/] [{color}]{message}[/]")
