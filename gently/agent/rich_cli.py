"""
Rich terminal interface for Microscopy Copilot

Provides:
- Semantic color coding (user/copilot/system/tool)
- Live status dashboard
- Streaming responses
- Command autocomplete
- Progress indicators
- Rich formatting with panels and tables
"""

import asyncio
from typing import Optional, Dict, Any, AsyncIterator
from datetime import datetime
from pathlib import Path

from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.live import Live
from rich.layout import Layout
from rich.text import Text
from rich.markdown import Markdown
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn
from rich.syntax import Syntax
from rich import box
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory

from .autocomplete import create_completer


class ColorScheme:
    """Semantic color scheme for copilot CLI"""
    USER = "green"
    USER_BOLD = "bold green"
    COPILOT = "blue"
    COPILOT_DIM = "dim blue"
    SYSTEM = "yellow"
    TOOL = "magenta"
    ERROR = "bold red"
    SUCCESS = "bright_green"
    WARNING = "yellow"
    INFO = "cyan"
    MUTED = "dim white"
    TIMESTAMP = "dim cyan"


class RichCopilotCLI:
    """
    Rich terminal interface for microscopy copilot

    Features:
    - Live status dashboard showing experiment state
    - Streaming response display
    - Command autocomplete and history
    - Progress indicators for long operations
    - Rich formatting with colors and panels
    """

    def __init__(self, copilot, history_file: Optional[Path] = None):
        """
        Initialize Rich CLI

        Parameters
        ----------
        copilot : MicroscopyCopilot
            Copilot instance
        history_file : Path, optional
            Path to command history file (default: ~/.copilot_history)
        """
        self.copilot = copilot
        self.console = Console()
        self.layout = None
        self.status_task = None
        self.live_display = None

        # Command history
        if history_file is None:
            history_file = Path.home() / '.copilot_history'
        self.history_file = history_file

        # Prompt session with autocomplete
        self.session = PromptSession(
            history=FileHistory(str(self.history_file)),
            auto_suggest=AutoSuggestFromHistory(),
            completer=create_completer(copilot),
            complete_while_typing=True,
            mouse_support=False,  # Disable mouse support to allow terminal scrolling
        )

        # State
        self._running = False
        self._last_status_update = None

    def print_welcome(self):
        """Print welcome banner"""
        welcome = Panel(
            Text.from_markup(
                "[bold blue]Microscopy Copilot v2.0[/]\n"
                "[dim]AI-powered adaptive microscopy control[/]\n\n"
                "[cyan]Commands:[/]\n"
                "  • Type naturally to interact with copilot\n"
                "  • Use [yellow]/detectors[/], [yellow]/status[/], [yellow]/embryos[/] for quick info\n"
                "  • Press [yellow]Tab[/] for autocomplete\n"
                "  • Press [yellow]↑/↓[/] for command history\n"
                "  • Press [yellow]Ctrl+C[/] to exit\n"
            ),
            title="Welcome",
            border_style="blue",
            box=box.ROUNDED,
        )
        self.console.print(welcome)
        self.console.print()

    def print_user_message(self, message: str):
        """Print user message with formatting"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        panel = Panel(
            Text(message, style=ColorScheme.USER),
            title=f"[{ColorScheme.TIMESTAMP}]{timestamp}[/] [bold]You[/]",
            title_align="left",
            border_style=ColorScheme.USER,
            box=box.ROUNDED,
        )
        self.console.print(panel)

    def print_copilot_message(self, message: str, is_markdown: bool = True):
        """Print copilot message with formatting"""
        timestamp = datetime.now().strftime("%H:%M:%S")

        if is_markdown:
            content = Markdown(message)
        else:
            content = Text(message, style=ColorScheme.COPILOT)

        panel = Panel(
            content,
            title=f"[{ColorScheme.TIMESTAMP}]{timestamp}[/] [bold blue]Copilot[/]",
            title_align="left",
            border_style=ColorScheme.COPILOT,
            box=box.ROUNDED,
        )
        self.console.print(panel)

    def print_system_message(self, message: str):
        """Print system message"""
        self.console.print(
            Text(f"[System] {message}", style=ColorScheme.SYSTEM)
        )

    def print_tool_call(self, tool_name: str, tool_input: Dict[str, Any], duration: Optional[float] = None):
        """Print tool call information"""
        # Format input
        input_lines = []
        for key, value in tool_input.items():
            if isinstance(value, dict):
                input_lines.append(f"  {key}:")
                for k, v in value.items():
                    input_lines.append(f"    {k}: {v}")
            else:
                input_lines.append(f"  {key}: {value}")

        content = f"[bold]{tool_name}[/]\n" + "\n".join(input_lines)
        if duration is not None:
            content += f"\n\n⏱  {duration:.2f}s"

        panel = Panel(
            Text.from_markup(content),
            title="[Tool Call]",
            title_align="left",
            border_style=ColorScheme.TOOL,
            box=box.SIMPLE,
        )
        self.console.print(panel)

    def print_error(self, error: str):
        """Print error message"""
        panel = Panel(
            Text(error, style=ColorScheme.ERROR),
            title="[bold red]Error[/]",
            border_style=ColorScheme.ERROR,
            box=box.HEAVY,
        )
        self.console.print(panel)

    def print_success(self, message: str):
        """Print success message"""
        self.console.print(
            Text(f"✓ {message}", style=ColorScheme.SUCCESS)
        )

    def create_status_panel(self) -> Panel:
        """Create status dashboard panel"""
        try:
            # Get experiment state
            experiment = self.copilot.experiment
            detector_registry = self.copilot.detector_registry

            # Experiment status
            status_lines = []
            status = experiment.status.value if hasattr(experiment, 'status') else 'unknown'
            status_color = {
                'running': ColorScheme.SUCCESS,
                'idle': ColorScheme.INFO,
                'paused': ColorScheme.WARNING,
                'completed': ColorScheme.COPILOT,
            }.get(status, ColorScheme.MUTED)

            status_lines.append(Text(f"Status: ", style=ColorScheme.MUTED) + Text(status.upper(), style=status_color))

            # Embryo count
            embryo_count = len(experiment.embryos)
            active_embryos = sum(1 for e in experiment.embryos.values() if not getattr(e, 'skip', False))
            status_lines.append(Text(f"Embryos: {active_embryos}/{embryo_count}", style=ColorScheme.INFO))

            # Detector count
            all_detectors = detector_registry.list_all()
            enabled_detectors = len([d for d in all_detectors if d.enabled])
            total_detectors = len(all_detectors)
            status_lines.append(Text(f"Detectors: {enabled_detectors}/{total_detectors}", style=ColorScheme.INFO))

            # Last imaging time
            last_imaging = "Never"
            for embryo in experiment.embryos.values():
                if hasattr(embryo, 'last_imaging_time') and embryo.last_imaging_time:
                    elapsed = (datetime.now() - embryo.last_imaging_time).total_seconds()
                    if elapsed < 60:
                        last_imaging = f"{int(elapsed)}s ago"
                    else:
                        last_imaging = f"{int(elapsed / 60)}m ago"
                    break

            status_lines.append(Text(f"Last image: {last_imaging}", style=ColorScheme.MUTED))

            # Recent detections (last 5)
            recent_detections = []
            for embryo_id, embryo in experiment.embryos.items():
                if hasattr(embryo, 'detection_results'):
                    for detector_name, results in embryo.detection_results.items():
                        for result in results[-3:]:  # Last 3 per detector
                            if result.get('detected'):
                                recent_detections.append({
                                    'detector': detector_name,
                                    'embryo': embryo_id,
                                    'timepoint': result.get('timepoint', 0),
                                })

            # Sort by timepoint, take last 5
            recent_detections.sort(key=lambda x: x['timepoint'], reverse=True)
            recent_detections = recent_detections[:5]

            if recent_detections:
                status_lines.append(Text(""))
                status_lines.append(Text("Recent Detections:", style="bold"))
                for det in recent_detections:
                    status_lines.append(
                        Text(f"  • {det['detector']} ", style=ColorScheme.TOOL) +
                        Text(f"({det['embryo']})", style=ColorScheme.MUTED)
                    )

            content = Group(*status_lines)

            return Panel(
                content,
                title="[bold]Status Dashboard[/]",
                border_style=ColorScheme.INFO,
                box=box.ROUNDED,
                padding=(1, 2),
            )

        except Exception as e:
            return Panel(
                Text(f"Status unavailable: {e}", style=ColorScheme.ERROR),
                title="Status",
                border_style=ColorScheme.ERROR,
            )

    def print_detector_table(self, detectors: list):
        """Print formatted detector table"""
        table = Table(
            title="Detectors",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold cyan",
        )

        table.add_column("Name", style=ColorScheme.INFO)
        table.add_column("Status", justify="center")
        table.add_column("Mode", style=ColorScheme.MUTED)
        table.add_column("Runs", justify="right", style=ColorScheme.MUTED)
        table.add_column("Detections", justify="right", style=ColorScheme.SUCCESS)

        for detector in detectors:
            status = "✓" if detector.enabled else "✗"
            status_style = ColorScheme.SUCCESS if detector.enabled else ColorScheme.ERROR

            # Get stats
            stats = detector.stats if hasattr(detector, 'stats') else {}
            total_runs = stats.get('total_runs', 0)
            total_detections = stats.get('total_detections', 0)

            table.add_row(
                detector.name,
                Text(status, style=status_style),
                detector.actions.mode.value,
                str(total_runs),
                str(total_detections),
            )

        self.console.print(table)

    def print_embryo_table(self, embryos: Dict[str, Any]):
        """Print formatted embryo table"""
        table = Table(
            title="Embryos",
            box=box.ROUNDED,
            show_header=True,
            header_style="bold cyan",
        )

        table.add_column("ID", style=ColorScheme.INFO)
        table.add_column("Status", justify="center")
        table.add_column("Last Imaging", style=ColorScheme.MUTED)
        table.add_column("Detections", style=ColorScheme.SUCCESS)

        for embryo_id, embryo in embryos.items():
            # Status
            skip = getattr(embryo, 'skip', False)
            status = "✗ Skipped" if skip else "✓ Active"
            status_style = ColorScheme.ERROR if skip else ColorScheme.SUCCESS

            # Last imaging
            last_time = "Never"
            if hasattr(embryo, 'last_imaging_time') and embryo.last_imaging_time:
                elapsed = (datetime.now() - embryo.last_imaging_time).total_seconds()
                if elapsed < 60:
                    last_time = f"{int(elapsed)}s ago"
                elif elapsed < 3600:
                    last_time = f"{int(elapsed / 60)}m ago"
                else:
                    last_time = f"{int(elapsed / 3600)}h ago"

            # Detections
            detections = []
            if hasattr(embryo, 'detection_results'):
                for detector_name in embryo.detection_results.keys():
                    if embryo.was_detected(detector_name):
                        detections.append(detector_name)

            detections_str = ", ".join(detections) if detections else "None"

            table.add_row(
                embryo_id,
                Text(status, style=status_style),
                last_time,
                detections_str,
            )

        self.console.print(table)

    async def stream_copilot_response(self, message: str):
        """
        Handle message with streaming response display

        Parameters
        ----------
        message : str
            User message
        """
        timestamp = datetime.now().strftime("%H:%M:%S")

        # Accumulated response
        response_text = ""

        # Create live display for streaming
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            transient=True,
        ) as progress:
            task = progress.add_task("[cyan]Thinking...", total=None)

            # Get streaming response
            async for chunk in self.copilot.handle_message_stream(message):
                if chunk.get('type') == 'text':
                    response_text += chunk.get('text', '')
                    progress.update(task, description=f"[cyan]Copilot is responding...")
                elif chunk.get('type') == 'tool_call':
                    # Show tool call
                    progress.stop()
                    self.print_tool_call(
                        chunk.get('tool_name', 'unknown'),
                        chunk.get('tool_input', {}),
                        chunk.get('duration')
                    )
                    progress.start()

        # Print final response
        panel = Panel(
            Markdown(response_text),
            title=f"[{ColorScheme.TIMESTAMP}]{timestamp}[/] [bold blue]Copilot[/]",
            title_align="left",
            border_style=ColorScheme.COPILOT,
            box=box.ROUNDED,
        )
        self.console.print(panel)

    async def handle_slash_command(self, command: str) -> Optional[bool]:
        """
        Handle slash commands directly

        Returns
        -------
        True if should quit, False if handled (continue loop), None if not a slash command
        """
        cmd = command.strip().lower()

        if cmd in ['/quit', '/exit', '/q']:
            return True  # Signal to quit

        elif cmd == '/clear':
            self.console.clear()
            self.print_welcome()
            return False  # Handled, continue loop

        elif cmd == '/help':
            self.print_help()
            return False  # Handled, continue loop

        elif cmd == '/status':
            panel = self.create_status_panel()
            self.console.print(panel)
            return False  # Handled, continue loop

        elif cmd == '/detectors':
            # List detectors
            detectors = self.copilot.detector_registry.list_all()
            self.print_detector_table(detectors)
            return False  # Handled, continue loop

        elif cmd == '/embryos':
            # List embryos
            self.print_embryo_table(self.copilot.experiment.embryos)
            return False  # Handled, continue loop

        elif cmd == '/history':
            # Show recent conversation history
            self.print_conversation_history()
            return False  # Handled, continue loop

        return None  # Not a slash command, send to copilot

    def print_conversation_history(self, limit: int = 10):
        """Print recent conversation history"""
        history = self.copilot.conversation_history[-limit:]

        if not history:
            self.console.print("No conversation history yet.", style=ColorScheme.MUTED)
            return

        self.console.print(Panel(
            Text(f"Showing last {len(history)} messages", style=ColorScheme.INFO),
            title="Conversation History",
            border_style=ColorScheme.INFO,
        ))
        self.console.print()

        for msg in history:
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')

            # Handle text content
            if isinstance(content, str):
                text = content[:500] + "..." if len(content) > 500 else content

                if role == 'user':
                    self.console.print(Panel(
                        Text(text, style=ColorScheme.USER),
                        title="You",
                        border_style=ColorScheme.USER,
                        box=box.SIMPLE,
                    ))
                elif role == 'assistant':
                    self.console.print(Panel(
                        Text(text, style=ColorScheme.COPILOT),
                        title="Copilot",
                        border_style=ColorScheme.COPILOT,
                        box=box.SIMPLE,
                    ))

            # Handle content blocks (may include tool calls)
            elif isinstance(content, list):
                text_parts = []
                for block in content:
                    if hasattr(block, 'text'):
                        text_parts.append(block.text)
                    elif hasattr(block, 'type') and block.type == 'tool_use':
                        text_parts.append(f"[Tool: {block.name}]")

                text = " ".join(text_parts)[:500]
                if role == 'assistant':
                    self.console.print(Panel(
                        Text(text, style=ColorScheme.COPILOT),
                        title="Copilot",
                        border_style=ColorScheme.COPILOT,
                        box=box.SIMPLE,
                    ))

        self.console.print()

    def print_help(self):
        """Print help message"""
        help_text = """
# Copilot Commands

## Natural Language
Just type what you want! Examples:
- "What detectors do we have?"
- "Add a detector for comma stage"
- "Test hatching detector on embryo 1"
- "Show me the status"
- "Start imaging all embryos"

## Slash Commands
- `/detectors` - List all detectors
- `/embryos` - List all embryos
- `/status` - Show experiment status
- `/history` - Show recent conversation
- `/help` - Show this help
- `/clear` - Clear screen
- `/quit` - Exit

## Keyboard Shortcuts
- `Tab` - Autocomplete commands/IDs
- `↑/↓` - Browse command history
- `Ctrl+C` - Exit
- `Ctrl+L` - Clear screen
- `Ctrl+R` - Reverse search history
        """
        self.console.print(Markdown(help_text))

    async def run(self):
        """Run interactive CLI loop"""
        self._running = True
        self.print_welcome()

        try:
            while self._running:
                try:
                    # Get user input with autocomplete
                    user_input = await self.session.prompt_async(
                        [(ColorScheme.USER_BOLD, '> ')],
                    )

                    if not user_input.strip():
                        continue

                    # Clear the input line to avoid double display
                    self.console.print()

                    # Handle slash commands
                    if user_input.startswith('/'):
                        result = await self.handle_slash_command(user_input)
                        if result is True:  # Quit command
                            break
                        elif result is False:  # Handled, continue loop
                            continue
                        # result is None means not recognized, fall through to copilot

                    # Stream copilot response
                    try:
                        await self.stream_copilot_response(user_input)
                    except Exception as e:
                        self.print_error(f"Error processing message: {e}")
                        import traceback
                        self.console.print(traceback.format_exc(), style=ColorScheme.ERROR)

                    self.console.print()  # Add spacing

                except KeyboardInterrupt:
                    self.console.print()
                    try:
                        confirm = await self.session.prompt_async("Exit? (y/n): ")
                        if confirm.lower().startswith('y'):
                            break
                    except (KeyboardInterrupt, EOFError):
                        break

                except EOFError:
                    break

                except Exception as e:
                    self.print_error(f"Unexpected error: {e}")
                    import traceback
                    self.console.print(traceback.format_exc(), style=ColorScheme.ERROR)
                    self.console.print()
                    # Continue loop instead of breaking

        finally:
            self._running = False
            self.console.print()
            self.console.print(Text("Goodbye! 👋", style=ColorScheme.COPILOT))


async def run_rich_cli(copilot, history_file: Optional[Path] = None):
    """
    Run rich CLI interface

    Parameters
    ----------
    copilot : MicroscopyCopilot
        Copilot instance
    history_file : Path, optional
        Command history file path
    """
    cli = RichCopilotCLI(copilot, history_file)
    await cli.run()
