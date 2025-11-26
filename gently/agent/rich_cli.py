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
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.application import Application
from prompt_toolkit.layout import Layout
from prompt_toolkit.layout.containers import Window, HSplit
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.formatted_text import HTML

from .autocomplete import create_completer
from .theme import get_theme, set_theme, list_themes, Theme


# Backwards compatibility - ColorScheme now wraps theme
class ColorScheme:
    """Semantic color scheme for copilot CLI (now wraps theme system)"""
    @property
    def USER(self):
        return get_theme().user

    @property
    def USER_BOLD(self):
        return f"bold {get_theme().user}"

    @property
    def COPILOT(self):
        return get_theme().copilot

    @property
    def COPILOT_DIM(self):
        return f"dim {get_theme().copilot}"

    @property
    def SYSTEM(self):
        return get_theme().system

    @property
    def TOOL(self):
        return get_theme().tool

    @property
    def ERROR(self):
        return f"bold {get_theme().error}"

    @property
    def SUCCESS(self):
        return get_theme().success

    @property
    def WARNING(self):
        return get_theme().warning

    @property
    def INFO(self):
        return get_theme().info

    @property
    def MUTED(self):
        return get_theme().muted

    @property
    def TIMESTAMP(self):
        return get_theme().muted


# Single instance for compatibility
_color_scheme = ColorScheme()


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
        theme = get_theme()
        welcome = Panel(
            Text.from_markup(
                f"[bold {theme.primary}]Microscopy Copilot v2.0[/]\n"
                f"[{theme.muted}]AI-powered adaptive microscopy control[/]\n\n"
                f"[{theme.secondary}]Commands:[/]\n"
                f"  {theme.icon_info} Type naturally to interact with copilot\n"
                f"  {theme.icon_info} Use [{theme.tool}]/detectors[/], [{theme.tool}]/status[/], [{theme.tool}]/embryos[/], [{theme.tool}]/theme[/]\n"
                f"  {theme.icon_info} Press [{theme.tool}]Tab[/] for autocomplete\n"
                f"  {theme.icon_info} Press [{theme.tool}]Ctrl+C[/] to exit\n"
            ),
            title=f"[bold {theme.primary}]Welcome[/]",
            border_style=theme.primary,
            box=box.ROUNDED,
        )
        self.console.print(welcome)
        self.console.print()

    def print_user_message(self, message: str):
        """Print user message with formatting"""
        theme = get_theme()
        timestamp = datetime.now().strftime("%H:%M:%S")
        panel = Panel(
            Text(message, style=theme.user),
            title=f"[{theme.muted}]{timestamp}[/] [{theme.user} bold]{theme.icon_user}[/]",
            title_align="left",
            border_style=theme.user,
            box=box.ROUNDED,
        )
        self.console.print(panel)

    def print_copilot_message(self, message: str, is_markdown: bool = True):
        """Print copilot message with formatting"""
        theme = get_theme()
        timestamp = datetime.now().strftime("%H:%M:%S")

        if is_markdown:
            content = Markdown(message)
        else:
            content = Text(message, style=theme.copilot)

        panel = Panel(
            content,
            title=f"[{theme.muted}]{timestamp}[/] [{theme.copilot} bold]{theme.icon_copilot}[/]",
            title_align="left",
            border_style=theme.copilot,
            box=box.ROUNDED,
        )
        self.console.print(panel)

    def print_system_message(self, message: str):
        """Print system message"""
        theme = get_theme()
        self.console.print(
            Text(f"[System] {message}", style=theme.system)
        )

    def print_tool_call(self, tool_name: str, tool_input: Dict[str, Any], duration: Optional[float] = None):
        """Print tool call information"""
        theme = get_theme()
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
            content += f"\n\n[{theme.muted}]{duration:.2f}s[/]"

        panel = Panel(
            Text.from_markup(content),
            title=f"[{theme.tool}]{theme.icon_tool}[/]",
            title_align="left",
            border_style=theme.tool,
            box=box.SIMPLE,
        )
        self.console.print(panel)

    def print_error(self, error: str):
        """Print error message"""
        theme = get_theme()
        panel = Panel(
            Text(error, style=f"bold {theme.error}"),
            title=f"[bold {theme.error}]{theme.icon_error} Error[/]",
            border_style=theme.error,
            box=box.HEAVY,
        )
        self.console.print(panel)

    def print_success(self, message: str):
        """Print success message"""
        theme = get_theme()
        self.console.print(
            Text(f"{theme.icon_success} {message}", style=theme.success)
        )

    def create_status_panel(self) -> Panel:
        """Create status dashboard panel"""
        theme = get_theme()
        try:
            # Get experiment state
            experiment = self.copilot.experiment
            detector_registry = self.copilot.detector_registry

            status_lines = []

            # Microscope connection status
            has_hardware = self.copilot._has_hardware() if hasattr(self.copilot, '_has_hardware') else False
            devices = getattr(self.copilot, 'devices', {}) or {}

            if has_hardware:
                status_lines.append(
                    Text(f"{theme.icon_success} ", style=theme.success) +
                    Text("Microscope: ", style=theme.muted) +
                    Text("CONNECTED", style=f"bold {theme.success}")
                )
                # Show device count
                device_count = len(devices)
                status_lines.append(Text(f"   Devices: {device_count} loaded", style=theme.muted))
            else:
                status_lines.append(
                    Text(f"{theme.icon_error} ", style=theme.error) +
                    Text("Microscope: ", style=theme.muted) +
                    Text("NOT CONNECTED", style=f"bold {theme.error}")
                )

            status_lines.append(Text(""))  # Spacer

            # Session info
            session_id = self.copilot.session_id or "none"
            status_lines.append(Text(f"Session: ", style=theme.muted) + Text(session_id, style=theme.info))

            # Experiment status
            status = experiment.status.value if hasattr(experiment, 'status') else 'unknown'
            status_color = {
                'running': theme.success,
                'idle': theme.info,
                'paused': theme.warning,
                'completed': theme.copilot,
            }.get(status, theme.muted)

            status_lines.append(Text(f"Experiment: ", style=theme.muted) + Text(status.upper(), style=status_color))

            # Embryo count
            embryo_count = len(experiment.embryos)
            active_embryos = sum(1 for e in experiment.embryos.values() if not getattr(e, 'skip', False))
            status_lines.append(Text(f"Embryos: {active_embryos}/{embryo_count}", style=theme.info))

            # Detector count
            all_detectors = detector_registry.list_all()
            enabled_detectors = len([d for d in all_detectors if d.enabled])
            total_detectors = len(all_detectors)
            status_lines.append(Text(f"Detectors: {enabled_detectors}/{total_detectors}", style=theme.info))

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

            status_lines.append(Text(f"Last image: {last_imaging}", style=theme.muted))

            # Show connected devices if any
            if devices:
                status_lines.append(Text(""))
                status_lines.append(Text("Hardware:", style="bold"))
                for dev_name in sorted(devices.keys()):
                    status_lines.append(Text(f"  {theme.icon_success} {dev_name}", style=theme.muted))

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
                        Text(f"  {theme.icon_info} {det['detector']} ", style=theme.tool) +
                        Text(f"({det['embryo']})", style=theme.muted)
                    )

            content = Group(*status_lines)

            return Panel(
                content,
                title=f"[bold {theme.primary}]Status Dashboard[/]",
                border_style=theme.info,
                box=box.ROUNDED,
                padding=(1, 2),
            )

        except Exception as e:
            return Panel(
                Text(f"Status unavailable: {e}", style=theme.error),
                title="Status",
                border_style=theme.error,
            )

    def print_detector_table(self, detectors: list):
        """Print formatted detector table"""
        theme = get_theme()
        table = Table(
            title="Detectors",
            box=box.ROUNDED,
            show_header=True,
            header_style=f"bold {theme.secondary}",
        )

        table.add_column("Name", style=theme.info)
        table.add_column("Status", justify="center")
        table.add_column("Mode", style=theme.muted)
        table.add_column("Runs", justify="right", style=theme.muted)
        table.add_column("Detections", justify="right", style=theme.success)

        for detector in detectors:
            status = theme.icon_success if detector.enabled else theme.icon_error
            status_style = theme.success if detector.enabled else theme.error

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

    def print_sessions_table(self):
        """Print formatted sessions table"""
        theme = get_theme()
        sessions = self.copilot.list_sessions()

        if not sessions:
            self.console.print(f"[{theme.muted}]No saved sessions found.[/]")
            self.console.print(f"[{theme.muted}]Current session: {self.copilot.session_id}[/]")
            return

        table = Table(
            title="Available Sessions",
            box=box.ROUNDED,
            show_header=True,
            header_style=f"bold {theme.secondary}",
        )

        table.add_column("ID", style=theme.info)
        table.add_column("Name", style=theme.muted)
        table.add_column("Embryos", justify="center")
        table.add_column("Messages", justify="center")
        table.add_column("Last Active", style=theme.muted)

        current_session_id = self.copilot.session_id

        for session in sessions:
            session_id = session.get('session_id', 'unknown')
            is_current = session_id == current_session_id

            # Format last active time
            last_active = session.get('last_active', '')
            if last_active:
                try:
                    from datetime import datetime
                    dt = datetime.fromisoformat(last_active)
                    elapsed = (datetime.now() - dt).total_seconds()
                    if elapsed < 60:
                        last_active = "just now"
                    elif elapsed < 3600:
                        last_active = f"{int(elapsed / 60)}m ago"
                    elif elapsed < 86400:
                        last_active = f"{int(elapsed / 3600)}h ago"
                    else:
                        last_active = f"{int(elapsed / 86400)}d ago"
                except:
                    pass

            # Mark current session
            id_display = f"{session_id}" + (" *" if is_current else "")
            id_style = f"bold {theme.success}" if is_current else theme.info

            table.add_row(
                Text(id_display, style=id_style),
                session.get('name') or '-',
                str(session.get('embryo_count', 0)),
                str(session.get('message_count', 0)),
                last_active,
            )

        self.console.print(table)
        self.console.print()
        self.console.print(f"[{theme.muted}]* = current session[/]")
        self.console.print(f"[{theme.muted}]Press Enter to select a session, or type /resume <id>[/]")

    async def interactive_session_picker(self) -> Optional[str]:
        """
        Show interactive session picker - keyboard only (↑/↓ + Enter/Esc).

        Returns
        -------
        str or None
            Selected session ID, or None if cancelled
        """
        theme = get_theme()
        sessions = self.copilot.list_sessions()

        if not sessions:
            self.console.print(f"[{theme.muted}]No saved sessions found.[/]")
            return None

        current_session_id = self.copilot.session_id

        # Build session items
        items = []
        for session in sessions:
            session_id = session.get('session_id', session.get('id', 'unknown'))
            name = session.get('name', '')
            embryo_count = session.get('embryo_count', 0)
            message_count = session.get('message_count', 0)

            # Format last active time
            last_active = session.get('last_active', '')
            time_str = ''
            if last_active:
                try:
                    dt = datetime.fromisoformat(last_active)
                    elapsed = (datetime.now() - dt).total_seconds()
                    if elapsed < 60:
                        time_str = "just now"
                    elif elapsed < 3600:
                        time_str = f"{int(elapsed / 60)}m ago"
                    elif elapsed < 86400:
                        time_str = f"{int(elapsed / 3600)}h ago"
                    else:
                        time_str = f"{int(elapsed / 86400)}d ago"
                except:
                    pass

            is_current = session_id == current_session_id
            items.append({
                'id': session_id,
                'name': name,
                'embryos': embryo_count,
                'messages': message_count,
                'time': time_str,
                'current': is_current
            })

        # State for the picker
        selected_idx = [0]
        result = [None]
        cancelled = [False]

        # Key bindings
        kb = KeyBindings()

        @kb.add('up')
        @kb.add('k')
        def move_up(event):
            selected_idx[0] = max(0, selected_idx[0] - 1)

        @kb.add('down')
        @kb.add('j')
        def move_down(event):
            selected_idx[0] = min(len(items) - 1, selected_idx[0] + 1)

        @kb.add('enter')
        def select(event):
            result[0] = items[selected_idx[0]]['id']
            event.app.exit()

        @kb.add('escape')
        @kb.add('q')
        def cancel(event):
            cancelled[0] = True
            event.app.exit()

        def get_formatted_text():
            text = '<b>Select a session:</b> (↑/↓ navigate, Enter select, Esc cancel)\n'
            text += '─' * 70 + '\n'

            for i, item in enumerate(items):
                is_selected = (i == selected_idx[0])
                marker = '▶ ' if is_selected else '  '
                current = ' <style fg="green">(current)</style>' if item['current'] else ''

                if is_selected:
                    line = f'<style bg="#0066cc" fg="white"><b>{marker}{item["id"]}</b>{current}'
                    line += f' │ {item["embryos"]} embryos │ {item["messages"]} msgs'
                    if item['time']:
                        line += f' │ {item["time"]}'
                    line += '</style>\n'
                else:
                    line = f'{marker}<style fg="#aaaaaa">{item["id"]}</style>{current}'
                    line += f' <style fg="#666666">│ {item["embryos"]} embryos │ {item["messages"]} msgs'
                    if item['time']:
                        line += f' │ {item["time"]}'
                    line += '</style>\n'

                text += line

            text += '─' * 70
            return HTML(text)

        # Create the application
        layout = Layout(
            HSplit([
                Window(
                    content=FormattedTextControl(get_formatted_text),
                    wrap_lines=False
                )
            ])
        )

        app = Application(
            layout=layout,
            key_bindings=kb,
            full_screen=False,
            mouse_support=False
        )

        try:
            await app.run_async()
            if cancelled[0]:
                return None
            return result[0]
        except Exception as e:
            self.console.print(f"[{theme.error}]Error: {e}[/]")
            return None

    def print_embryo_table(self, embryos: Dict[str, Any]):
        """Print formatted embryo table"""
        theme = get_theme()
        table = Table(
            title="Embryos",
            box=box.ROUNDED,
            show_header=True,
            header_style=f"bold {theme.secondary}",
        )

        table.add_column("ID", style=theme.info)
        table.add_column("Status", justify="center")
        table.add_column("Last Imaging", style=theme.muted)
        table.add_column("Detections", style=theme.success)

        for embryo_id, embryo in embryos.items():
            # Status
            skip = getattr(embryo, 'skip', False)
            status = f"{theme.icon_error} Skipped" if skip else f"{theme.icon_success} Active"
            status_style = theme.error if skip else theme.success

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
        theme = get_theme()
        panel = Panel(
            Markdown(response_text),
            title=f"[{theme.muted}]{timestamp}[/] [{theme.copilot} bold]{theme.icon_copilot}[/]",
            title_align="left",
            border_style=theme.copilot,
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

        elif cmd.startswith('/theme'):
            # Theme switching
            parts = cmd.split()
            if len(parts) > 1:
                theme_name = parts[1]
                try:
                    set_theme(theme_name)
                    theme = get_theme()
                    self.console.print(f"[{theme.success}]+ Theme changed to: {theme.name}[/]")
                except ValueError as e:
                    theme = get_theme()
                    self.console.print(f"[{theme.error}]x {e}[/]")
            else:
                # List available themes
                theme = get_theme()
                self.console.print(f"\n[bold {theme.primary}]Available themes:[/]")
                for name, t in list_themes().items():
                    marker = " [dim](current)[/]" if t.name == theme.name else ""
                    self.console.print(f"  [{t.primary}]{name}[/]{marker}")
                self.console.print(f"\n[{theme.muted}]Usage: /theme <name>[/]\n")
            return False  # Handled, continue loop

        elif cmd == '/sessions':
            # Show interactive session picker
            theme = get_theme()
            self.console.print(f"\n[bold {theme.primary}]Session Manager[/]\n")
            selected_id = await self.interactive_session_picker()

            if selected_id:
                if self.copilot.resume_session(selected_id):
                    session = self.copilot.session_manager.current_session
                    self.console.print(f"\n[{theme.success}]✓ Session resumed: {selected_id}[/]")
                    self.console.print(f"[{theme.muted}]  {session.embryo_count} embryos, {session.message_count} messages[/]")
                else:
                    self.console.print(f"\n[{theme.error}]✗ Failed to resume session '{selected_id}'[/]")
            else:
                self.console.print(f"\n[{theme.muted}]Session selection cancelled[/]")
            return False  # Handled, continue loop

        elif cmd.startswith('/resume'):
            # Resume a session by ID (for autocomplete/direct use)
            parts = command.strip().split()
            if len(parts) > 1:
                session_id = parts[1]
                if self.copilot.resume_session(session_id):
                    theme = get_theme()
                    session = self.copilot.session_manager.current_session
                    self.console.print(f"[{theme.success}]✓ Session resumed: {session_id}[/]")
                    self.console.print(f"[{theme.muted}]  {session.embryo_count} embryos, {session.message_count} messages[/]")
                else:
                    theme = get_theme()
                    self.console.print(f"[{theme.error}]✗ Session '{session_id}' not found[/]")
            else:
                # No session ID provided - show picker
                theme = get_theme()
                self.console.print(f"\n[bold {theme.primary}]Select a session to resume:[/]\n")
                selected_id = await self.interactive_session_picker()

                if selected_id:
                    if self.copilot.resume_session(selected_id):
                        session = self.copilot.session_manager.current_session
                        self.console.print(f"\n[{theme.success}]✓ Session resumed: {selected_id}[/]")
                        self.console.print(f"[{theme.muted}]  {session.embryo_count} embryos, {session.message_count} messages[/]")
                    else:
                        self.console.print(f"\n[{theme.error}]✗ Failed to resume session '{selected_id}'[/]")
                else:
                    self.console.print(f"\n[{theme.muted}]Session selection cancelled[/]")
            return False  # Handled, continue loop

        elif cmd == '/save':
            # Save current session
            if self.copilot.save_session():
                theme = get_theme()
                self.console.print(f"[{theme.success}]+ Session saved: {self.copilot.session_id}[/]")
            else:
                theme = get_theme()
                self.console.print(f"[{theme.error}]x Failed to save session[/]")
            return False  # Handled, continue loop

        return None  # Not a slash command, send to copilot

    def print_conversation_history(self, limit: int = 10):
        """Print recent conversation history"""
        theme = get_theme()
        history = self.copilot.conversation_history[-limit:]

        if not history:
            self.console.print("No conversation history yet.", style=theme.muted)
            return

        self.console.print(Panel(
            Text(f"Showing last {len(history)} messages", style=theme.info),
            title="Conversation History",
            border_style=theme.info,
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
                        Text(text, style=theme.user),
                        title=theme.icon_user,
                        border_style=theme.user,
                        box=box.SIMPLE,
                    ))
                elif role == 'assistant':
                    self.console.print(Panel(
                        Text(text, style=theme.copilot),
                        title=theme.icon_copilot,
                        border_style=theme.copilot,
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
                        Text(text, style=theme.copilot),
                        title=theme.icon_copilot,
                        border_style=theme.copilot,
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
- `/sessions` - Browse and select saved sessions (interactive)
- `/resume [id]` - Resume a session (interactive picker if no ID given)
- `/save` - Save current session
- `/theme [name]` - Switch theme (vibrant, scientific, claude, monochrome)
- `/help` - Show this help
- `/clear` - Clear screen
- `/quit` - Exit

## Keyboard Shortcuts
- `Tab` - Autocomplete commands/IDs
- `Ctrl+C` - Exit
- `Ctrl+L` - Clear screen
- `Ctrl+R` - Reverse search history
        """
        self.console.print(Markdown(help_text))

    async def run(self):
        """Run interactive CLI loop"""
        self._running = True
        self.print_welcome()

        # Show restored conversation history if session was resumed
        if self.copilot.conversation_history:
            theme = get_theme()
            num_messages = len(self.copilot.conversation_history)
            self.console.print(Panel(
                Text.from_markup(
                    f"[{theme.secondary}]Session restored with {num_messages} previous messages[/]"
                ),
                title=f"[{theme.secondary}]Restored Session[/]",
                border_style=theme.secondary,
                box=box.SIMPLE,
            ))
            self.print_conversation_history()
            self.console.print(Text(
                f"{'─' * 40} Current Session {'─' * 40}",
                style=theme.secondary
            ))
            self.console.print()

        try:
            while self._running:
                try:
                    # Get user input with autocomplete
                    theme = get_theme()
                    user_input = await self.session.prompt_async(
                        [(f"bold {theme.user}", '> ')],
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
                        self.console.print(traceback.format_exc(), style=theme.error)

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
                    theme = get_theme()
                    self.console.print(traceback.format_exc(), style=theme.error)
                    self.console.print()
                    # Continue loop instead of breaking

        finally:
            self._running = False
            theme = get_theme()
            self.console.print()
            self.console.print(Text("Goodbye!", style=theme.copilot))


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
