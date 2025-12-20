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
from rich.prompt import Prompt
from rich import box
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.application import Application
from prompt_toolkit.layout import Layout
from prompt_toolkit.layout.containers import Window, HSplit
from prompt_toolkit.layout.controls import FormattedTextControl
from prompt_toolkit.formatted_text import HTML

from .autocomplete import create_completer, create_auto_suggest
from .theme import get_theme, set_theme, list_themes, Theme
from .timeline import TimelineManager, TimelineEvent, parse_time_delta


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
        self._history = FileHistory(str(self.history_file))

        # Prompt session with autocomplete and command-aware auto-suggest
        self.session = PromptSession(
            history=self._history,
            auto_suggest=create_auto_suggest(copilot, history=self._history),
            completer=create_completer(copilot),
            complete_while_typing=True,
            mouse_support=False,  # Disable mouse support to allow terminal scrolling
            bottom_toolbar=self._get_status_bar,
        )

        # Wire up choice handler for interactive selection UI
        self.copilot.choice_handler = self.interactive_choice_picker

        # State
        self._running = False
        self._last_status_update = None

    def _get_status_bar(self):
        """Generate bottom status bar content"""
        from prompt_toolkit.formatted_text import HTML

        parts = []

        # Context size (what Claude sees per call) and cost
        context_tokens = getattr(self.copilot, 'current_context_tokens', 0)

        # Cost calculation from cumulative tokens
        cache_read = getattr(self.copilot, 'cache_read_tokens', 0)
        cache_created = getattr(self.copilot, 'cache_creation_tokens', 0)
        input_tokens = self.copilot.total_input_tokens
        output_tokens = self.copilot.total_output_tokens

        if input_tokens > 0 or output_tokens > 0:
            # Sonnet pricing: input $3/M, output $15/M, cache_read $0.30/M, cache_write $6/M (1h TTL)
            input_cost = input_tokens * 0.003 / 1000
            cache_read_cost = cache_read * 0.0003 / 1000
            cache_write_cost = cache_created * 0.006 / 1000
            output_cost = output_tokens * 0.015 / 1000
            cost = input_cost + cache_read_cost + cache_write_cost + output_cost

            # Show context size and cost
            context_k = context_tokens / 1000
            if cache_read > 0:
                parts.append(f"<b>Context:</b> {context_k:.1f}K (${cost:.3f}) <style fg='green'>⚡</style>")
            else:
                parts.append(f"<b>Context:</b> {context_k:.1f}K (${cost:.3f})")
        else:
            parts.append("<b>Context:</b> 0")

        # Session ID (truncated)
        session_id = self.copilot.session_id or "none"
        if len(session_id) > 20:
            session_id = session_id[:17] + "..."
        parts.append(f"<b>Session:</b> {session_id}")

        # Embryo count
        embryo_count = len(self.copilot.experiment.embryos) if self.copilot.experiment else 0
        parts.append(f"<b>Embryos:</b> {embryo_count}")

        # Connection status
        if self.copilot.client and self.copilot.client.is_connected:
            parts.append("<style fg='green'>● Connected</style>")
        else:
            parts.append("<style fg='yellow'>○ Offline</style>")

        return HTML(" │ ".join(parts))

    def print_welcome(self):
        """Print welcome banner with categorized commands"""
        from .command_registry import get_command_registry, CommandCategory

        theme = get_theme()
        registry = get_command_registry()

        # Build categorized command list
        category_labels = {
            CommandCategory.NAVIGATION: "Navigate",
            CommandCategory.INSPECTION: "Inspect",
            CommandCategory.SESSION: "Session",
            CommandCategory.APPEARANCE: "Style",
        }

        cmd_lines = []
        for category in CommandCategory:
            cmds = registry.get_by_category(category)
            if cmds:
                label = category_labels.get(category, category.name)
                cmd_names = ", ".join(f"[{theme.tool}]{c.name}[/]" for c in cmds)
                cmd_lines.append(f"  [{theme.muted}]{label:8}[/] {cmd_names}")

        cmd_section = "\n".join(cmd_lines)

        welcome = Panel(
            Text.from_markup(
                f"[bold {theme.primary}]Microscopy Copilot v2.0[/]\n"
                f"[{theme.muted}]AI-powered adaptive microscopy control[/]\n\n"
                f"[{theme.secondary}]Commands:[/]\n"
                f"{cmd_section}\n\n"
                f"  {theme.icon_info} Type naturally or use commands above\n"
                f"  {theme.icon_info} [{theme.tool}]Tab[/] autocomplete, [{theme.tool}]Right Arrow[/] accept suggestion\n"
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

            status_lines = []

            # Microscope connection status
            has_hardware = self.copilot.client and self.copilot.client.is_connected
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
            status = experiment.acquisition_status or 'idle'
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

            # Perception system status
            perception_mgr = getattr(self.copilot, 'perception_manager', None)
            if perception_mgr:
                session_count = len(perception_mgr.sessions)
                status_lines.append(Text(f"Perception: {session_count} sessions active", style=theme.info))
            else:
                status_lines.append(Text(f"Perception: not initialized", style=theme.muted))

            # Last imaging time - find most recent across all embryos
            last_imaging = "Never"
            most_recent = None
            for embryo in experiment.embryos.values():
                if embryo.last_imaged:
                    if most_recent is None or embryo.last_imaged > most_recent:
                        most_recent = embryo.last_imaged
            if most_recent:
                elapsed = (datetime.now() - most_recent).total_seconds()
                if elapsed < 60:
                    last_imaging = f"{int(elapsed)}s ago"
                elif elapsed < 3600:
                    last_imaging = f"{int(elapsed / 60)}m ago"
                else:
                    last_imaging = f"{elapsed / 3600:.1f}h ago"

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

            # Get stats from detector attributes
            total_runs = detector.run_count
            total_detections = detector.detection_count

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

    async def interactive_choice_picker(self, choice_data: Dict) -> str:
        """
        Show interactive choice picker for tool-generated choices.

        Parameters
        ----------
        choice_data : dict
            Choice request from ask_user_choice tool containing:
            - question: str
            - options: List[{id, label, description?}]
            - allow_multiple: bool
            - default_id: str?

        Returns
        -------
        str
            Selected option ID (or comma-separated IDs if multiple)
        """
        theme = get_theme()

        question = choice_data.get('question', 'Select an option:')
        options = choice_data.get('options', [])
        allow_multiple = choice_data.get('allow_multiple', False)
        default_id = choice_data.get('default_id')

        if not options:
            return "Error: No options provided"

        # Find default index
        default_idx = 0
        if default_id:
            for i, opt in enumerate(options):
                if opt.get('id') == default_id:
                    default_idx = i
                    break

        # State for the picker (using lists for mutability in closures)
        selected_idx = [default_idx]
        selected_ids = [set()]  # For multiple selection

        # Key bindings
        kb = KeyBindings()

        @kb.add('up')
        @kb.add('k')
        def move_up(event):
            selected_idx[0] = max(0, selected_idx[0] - 1)

        @kb.add('down')
        @kb.add('j')
        def move_down(event):
            selected_idx[0] = min(len(options) - 1, selected_idx[0] + 1)

        @kb.add('space')
        def toggle_select(event):
            if allow_multiple:
                opt_id = options[selected_idx[0]].get('id')
                if opt_id in selected_ids[0]:
                    selected_ids[0].discard(opt_id)
                else:
                    selected_ids[0].add(opt_id)

        @kb.add('enter')
        def select(event):
            # Use app.exit(result=...) to directly return the selected value
            if allow_multiple:
                if selected_ids[0]:
                    event.app.exit(result=','.join(sorted(selected_ids[0])))
                else:
                    event.app.exit(result=options[selected_idx[0]].get('id') or f"option_{selected_idx[0]}")
            else:
                event.app.exit(result=options[selected_idx[0]].get('id') or f"option_{selected_idx[0]}")

        @kb.add('escape')
        @kb.add('q')
        def cancel(event):
            event.app.exit(result="cancelled")

        def get_formatted_text():
            text = f'<b>{question}</b>\n'
            if allow_multiple:
                text += '<style fg="#888888">(↑/↓ navigate, Space toggle, Enter confirm, Esc cancel)</style>\n'
            else:
                text += '<style fg="#888888">(↑/↓ navigate, Enter select, Esc cancel)</style>\n'
            text += '─' * 60 + '\n'

            for i, opt in enumerate(options):
                is_cursor = (i == selected_idx[0])
                is_selected = opt.get('id') in selected_ids[0] if allow_multiple else False

                if allow_multiple:
                    check = '✓' if is_selected else '○'
                    marker = f'{check} '
                else:
                    marker = ''

                cursor = '▶ ' if is_cursor else '  '

                label = opt.get('label', opt.get('id', f'Option {i+1}'))
                desc = opt.get('description', '')

                if is_cursor:
                    line = f'<style bg="#0066cc" fg="white"><b>{cursor}{marker}{label}</b>'
                    if desc:
                        line += f' <style fg="#cccccc">- {desc}</style>'
                    line += '</style>\n'
                else:
                    line = f'{cursor}{marker}<style fg="#aaaaaa">{label}</style>'
                    if desc:
                        line += f' <style fg="#666666">- {desc}</style>'
                    line += '\n'

                text += line

            text += '─' * 60
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

        self.console.print()  # Add spacing

        try:
            # Run the picker synchronously in a thread to avoid event loop conflicts
            def run_picker():
                return app.run()

            # app.run() returns the value passed to app.exit(result=...)
            selection = await asyncio.get_event_loop().run_in_executor(None, run_picker)

            # Return the selection (will be "cancelled" if user pressed Escape/q)
            if selection:
                return selection

            # Fallback if somehow selection is None/empty
            fallback = options[selected_idx[0]].get('id') or f"option_{selected_idx[0]}"
            return fallback
        except Exception as e:
            self.console.print(f"[{theme.error}]Error: {e}[/]")
            return "error"

    def print_timeline_horizontal(self, events: list):
        """
        Print annotated vertical timeline (git-log style)

        Shows events in chronological order with visual timeline connector
        """
        theme = get_theme()

        if not events:
            self.console.print(f"[{theme.muted}]No events to display[/]")
            return

        # Get time range for title
        start_time = events[0].timestamp
        end_time = events[-1].timestamp
        start_label = start_time.strftime("%H:%M")
        end_label = end_time.strftime("%H:%M")

        lines = []

        for i, event in enumerate(events):
            time_str = event.timestamp.strftime("%H:%M:%S")

            # Determine style based on event type
            if event.event_type == 'timelapse':
                marker_style = theme.success
                type_label = "TL"
            else:
                marker_style = theme.info
                type_label = "DET"

            # Event line with marker
            event_line = Text()
            event_line.append(f"  {time_str}  ", style=theme.muted)
            event_line.append("*", style=f"bold {marker_style}")
            event_line.append(f" {type_label} ", style=marker_style)
            event_line.append(f"{event.event_subtype}", style=f"bold {theme.secondary}")

            # Add embryo if present
            if event.embryo_id:
                event_line.append(f" [{event.embryo_id}]", style=theme.info)

            # Add timepoint if present
            if event.timepoint is not None:
                event_line.append(f" t={event.timepoint}", style=theme.accent)

            lines.append(event_line)

            # Description line
            desc_line = Text()
            desc_line.append("             ", style=theme.muted)  # Align with above
            if i < len(events) - 1:
                desc_line.append("|  ", style=theme.muted)
            else:
                desc_line.append("   ", style=theme.muted)
            desc_line.append(event.description, style=theme.muted)
            lines.append(desc_line)

            # Connector line (except for last event)
            if i < len(events) - 1:
                connector = Text()
                connector.append("             ", style=theme.muted)
                connector.append("|", style=theme.muted)
                lines.append(connector)

        # Build panel
        content = Group(*lines)
        panel = Panel(
            content,
            title=f"[bold {theme.primary}]Timeline ({start_label} - {end_label})[/]",
            border_style=theme.secondary,
            box=box.ROUNDED,
            padding=(1, 2),
        )

        self.console.print(panel)

    def print_timeline_axis(self, events: list):
        """
        Print horizontal axis timeline with simple markers
        """
        theme = get_theme()

        if not events:
            self.console.print(f"[{theme.muted}]No events to display[/]")
            return

        # Get time range
        start_time = events[0].timestamp
        end_time = events[-1].timestamp
        duration = (end_time - start_time).total_seconds()

        width = 60
        start_label = start_time.strftime("%H:%M")
        end_label = end_time.strftime("%H:%M")

        def get_position(timestamp):
            if duration == 0:
                return width // 2
            elapsed = (timestamp - start_time).total_seconds()
            return max(0, min(int((elapsed / duration) * (width - 1)), width - 1))

        # Build marker line
        marker_line = list("-" * width)
        for event in events:
            pos = get_position(event.timestamp)
            marker = "T" if event.event_type == 'timelapse' else "D"
            marker_line[pos] = marker

        # Time labels
        time_line = [" "] * width
        time_line[0:len(start_label)] = list(start_label)
        time_line[width - len(end_label):] = list(end_label)

        # Count events
        tl_count = sum(1 for e in events if e.event_type == 'timelapse')
        det_count = sum(1 for e in events if e.event_type == 'detection')

        lines = []
        lines.append(Text("".join(time_line), style=theme.muted))
        lines.append(Text("|" + "".join(marker_line) + "|", style=theme.secondary))
        lines.append(Text(""))

        summary = Text()
        summary.append(f"{len(events)} events: ", style=theme.muted)
        summary.append(f"T", style=theme.success)
        summary.append(f"={tl_count} timelapse  ", style=theme.muted)
        summary.append(f"D", style=theme.info)
        summary.append(f"={det_count} detection", style=theme.muted)
        lines.append(summary)

        panel = Panel(
            Group(*lines),
            title=f"[bold {theme.primary}]Timeline[/]",
            border_style=theme.secondary,
            box=box.ROUNDED,
            padding=(1, 2),
        )
        self.console.print(panel)

    def print_timeline_letters(self, events: list):
        """
        Print horizontal timeline with lettered markers and legend
        """
        theme = get_theme()

        if not events:
            self.console.print(f"[{theme.muted}]No events to display[/]")
            return

        start_time = events[0].timestamp
        end_time = events[-1].timestamp
        duration = (end_time - start_time).total_seconds()

        width = 60
        start_label = start_time.strftime("%H:%M")
        end_label = end_time.strftime("%H:%M")

        def get_position(timestamp):
            if duration == 0:
                return width // 2
            elapsed = (timestamp - start_time).total_seconds()
            return max(0, min(int((elapsed / duration) * (width - 1)), width - 1))

        def get_marker(idx):
            if idx < 26:
                return chr(ord('A') + idx)
            return str(idx - 25)

        # Build marker line
        marker_line = list("-" * width)
        event_markers = []

        for idx, event in enumerate(events[-20:]):
            pos = get_position(event.timestamp)
            marker = get_marker(idx)
            # Find free position
            actual_pos = pos
            while actual_pos < width and marker_line[actual_pos] != '-':
                actual_pos += 1
            if actual_pos < width:
                marker_line[actual_pos] = marker
                event_markers.append((marker, event))

        # Time labels
        time_line = [" "] * width
        time_line[0:len(start_label)] = list(start_label)
        time_line[width - len(end_label):] = list(end_label)

        lines = []
        lines.append(Text("".join(time_line), style=theme.muted))
        lines.append(Text("|" + "".join(marker_line) + "|", style=theme.secondary))
        lines.append(Text(""))
        lines.append(Text("Legend:", style=f"bold {theme.primary}"))

        for marker, event in event_markers:
            time_str = event.timestamp.strftime("%H:%M:%S")
            type_style = theme.success if event.event_type == 'timelapse' else theme.info
            type_label = "TL" if event.event_type == 'timelapse' else "DET"

            line = Text()
            line.append(f"  {marker} ", style=f"bold {theme.accent}")
            line.append(f"[{time_str}] ", style=theme.muted)
            line.append(f"{type_label} ", style=type_style)
            line.append(event.description, style=theme.muted)
            lines.append(line)

        panel = Panel(
            Group(*lines),
            title=f"[bold {theme.primary}]Timeline ({start_label} - {end_label})[/]",
            border_style=theme.secondary,
            box=box.ROUNDED,
            padding=(1, 2),
        )
        self.console.print(panel)

    def print_timeline_list(self, events: list, title: str = "Timeline Events", compact: bool = False):
        """
        Print timeline events as a vertical list

        Parameters
        ----------
        events : list of TimelineEvent
            Events to display
        title : str
            Panel title
        compact : bool
            If True, show one-line-per-event format
        """
        theme = get_theme()

        if not events:
            self.console.print(f"[{theme.muted}]No events to display[/]")
            return

        if compact:
            # Compact table format
            table = Table(
                title=title,
                box=box.SIMPLE,
                show_header=True,
                header_style=f"bold {theme.secondary}",
                padding=(0, 1),
            )
            table.add_column("Time", style=theme.muted, width=10)
            table.add_column("Type", style=theme.info, width=10)
            table.add_column("Embryo", style=theme.secondary, width=10)
            table.add_column("Details", style=theme.muted)

            for event in events:
                time_str = event.timestamp.strftime("%H:%M:%S")

                # Event type with color based on severity
                type_style = theme.info
                if event.severity == 'success':
                    type_style = theme.success
                elif event.severity == 'error':
                    type_style = theme.error
                elif event.severity == 'warning':
                    type_style = theme.warning

                type_text = f"{event.short_label} {event.event_subtype[:6]}"

                embryo = event.embryo_id or "-"
                details = event.description[:30] if len(event.description) > 30 else event.description

                table.add_row(
                    time_str,
                    Text(type_text, style=type_style),
                    embryo,
                    details,
                )

            self.console.print(table)
        else:
            # Detailed list format
            lines = []
            current_date = None

            for event in events:
                # Add date header if changed
                event_date = event.timestamp.date()
                if event_date != current_date:
                    current_date = event_date
                    if event_date == datetime.now().date():
                        date_label = "TODAY"
                    else:
                        date_label = event_date.strftime("%Y-%m-%d")
                    lines.append(Text(f"\n{date_label}", style=f"bold {theme.primary}"))

                # Event line
                time_str = event.timestamp.strftime("%H:%M:%S")

                # Icon and color based on severity
                icon = event.icon
                if event.severity == 'success':
                    style = theme.success
                elif event.severity == 'error':
                    style = theme.error
                elif event.severity == 'warning':
                    style = theme.warning
                else:
                    style = theme.info

                event_text = Text()
                event_text.append(f"  {time_str}  ", style=theme.muted)
                event_text.append(f"{icon} ", style=style)
                event_text.append(f"{event.event_type}/{event.event_subtype}", style=style)
                lines.append(event_text)

                # Description line (indented)
                desc_text = Text()
                desc_text.append("            ", style=theme.muted)  # Indent
                desc_text.append(event.description, style=theme.muted)
                lines.append(desc_text)

            content = Group(*lines)
            panel = Panel(
                content,
                title=f"[bold {theme.primary}]{title}[/]",
                border_style=theme.secondary,
                box=box.ROUNDED,
                padding=(1, 2),
            )
            self.console.print(panel)

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
        table.add_column("XY (µm)", style=theme.secondary)
        table.add_column("Status", justify="center")
        table.add_column("Last Imaging", style=theme.info)
        table.add_column("Exposures", style=theme.warning)
        table.add_column("Stage", style=theme.success)

        for embryo_id, embryo in embryos.items():
            # XY position
            pos = getattr(embryo, 'stage_position', {})
            x = pos.get('x', 0)
            y = pos.get('y', 0)
            xy_str = f"{x:.0f}, {y:.0f}"

            # Status
            skip = getattr(embryo, 'skip', False)
            status = f"{theme.icon_error} Skipped" if skip else f"{theme.icon_success} Active"
            status_style = theme.error if skip else theme.success

            # Last imaging
            last_time = "Never"
            if hasattr(embryo, 'last_imaged') and embryo.last_imaged:
                elapsed = (datetime.now() - embryo.last_imaged).total_seconds()
                if elapsed < 60:
                    last_time = f"{int(elapsed)}s ago"
                elif elapsed < 3600:
                    last_time = f"{int(elapsed / 60)}m ago"
                else:
                    last_time = f"{int(elapsed / 3600)}h ago"

            # Exposures
            exposure_count = getattr(embryo, 'exposure_count', 0)
            total_exposure_ms = getattr(embryo, 'total_exposure_ms', 0)
            if exposure_count > 0:
                total_sec = total_exposure_ms / 1000
                if total_sec < 1:
                    exposure_str = f"{exposure_count} ({total_exposure_ms:.0f}ms)"
                else:
                    exposure_str = f"{exposure_count} ({total_sec:.1f}s)"
            else:
                exposure_str = "0"

            # Perception stage info
            perception_mgr = getattr(self.copilot, 'perception_manager', None)
            stage_str = "-"
            if perception_mgr:
                session = perception_mgr.get_session(embryo_id)
                if session:
                    stage_str = session.get_current_stage() or "-"

            table.add_row(
                embryo_id,
                xy_str,
                Text(status, style=status_style),
                last_time,
                exposure_str,
                stage_str,
            )

        self.console.print(table)

    def print_embryo_details(self, embryo):
        """Print detailed view of a single embryo"""
        theme = get_theme()

        # Build title with nickname if present
        title = f"{embryo.id}"
        if embryo.nickname:
            title += f' ("{embryo.nickname}")'
        if embryo.user_label:
            title += f" [{embryo.user_label}]"

        sections = []

        # === Identity & Position ===
        pos = embryo.stage_position or {}
        x = pos.get('x', 0)
        y = pos.get('y', 0)
        identity_lines = [
            f"[{theme.secondary}]Stage Position:[/] X={x:.1f} µm, Y={y:.1f} µm",
            f"[{theme.secondary}]Detection Confidence:[/] {embryo.detection_confidence:.1%}" if embryo.detection_confidence else "",
        ]
        sections.append(("Position", [l for l in identity_lines if l]))

        # === Calibration ===
        cal = embryo.calibration or {}
        if cal:
            cal_lines = []
            if 'piezo_center' in cal:
                cal_lines.append(f"[{theme.secondary}]Piezo Center:[/] {cal['piezo_center']:.2f} µm")
            if 'piezo_amplitude' in cal:
                cal_lines.append(f"[{theme.secondary}]Piezo Amplitude:[/] {cal['piezo_amplitude']:.2f} µm")
            if 'galvo_center' in cal:
                cal_lines.append(f"[{theme.secondary}]Galvo Center:[/] {cal['galvo_center']:.3f} V")
            if 'galvo_amplitude' in cal:
                cal_lines.append(f"[{theme.secondary}]Galvo Amplitude:[/] {cal['galvo_amplitude']:.3f} V")

            # Any other calibration keys
            shown_keys = {'piezo_center', 'piezo_amplitude', 'galvo_center', 'galvo_amplitude'}
            for key, val in cal.items():
                if key not in shown_keys:
                    cal_lines.append(f"[{theme.secondary}]{key}:[/] {val}")

            if cal_lines:
                sections.append(("Calibration", cal_lines))
        else:
            sections.append(("Calibration", [f"[{theme.muted}]Not calibrated[/]"]))

        # === Acquisition Settings ===
        acq_lines = [
            f"[{theme.secondary}]Slices:[/] {embryo.num_slices}",
            f"[{theme.secondary}]Exposure:[/] {embryo.exposure_ms} ms",
            f"[{theme.secondary}]Priority:[/] {embryo.priority}",
        ]
        sections.append(("Acquisition Settings", acq_lines))

        # === Status ===
        status_lines = []
        if embryo.should_skip:
            status_lines.append(f"[{theme.error}]⏸ Skipped:[/] {embryo.skip_reason or 'No reason given'}")
        else:
            status_lines.append(f"[{theme.success}]● Active[/]")

        status_lines.append(f"[{theme.secondary}]Timepoints Acquired:[/] {embryo.timepoints_acquired}")

        if embryo.last_imaged:
            elapsed = (datetime.now() - embryo.last_imaged).total_seconds()
            if elapsed < 60:
                time_str = f"{int(elapsed)}s ago"
            elif elapsed < 3600:
                time_str = f"{int(elapsed / 60)}m ago"
            else:
                time_str = embryo.last_imaged.strftime("%Y-%m-%d %H:%M:%S")
            status_lines.append(f"[{theme.secondary}]Last Imaged:[/] {time_str}")
        else:
            status_lines.append(f"[{theme.secondary}]Last Imaged:[/] Never")

        sections.append(("Status", status_lines))

        # === Light Exposure ===
        exposure_lines = []
        if hasattr(embryo, 'exposure_count') and embryo.exposure_count > 0:
            exposure_lines.append(f"[{theme.secondary}]Exposures:[/] {embryo.exposure_count}")
            total_ms = getattr(embryo, 'total_exposure_ms', 0)
            if total_ms < 1000:
                time_str = f"{total_ms:.0f} ms"
            elif total_ms < 60000:
                time_str = f"{total_ms / 1000:.1f} s"
            else:
                time_str = f"{total_ms / 60000:.1f} min"
            exposure_lines.append(f"[{theme.secondary}]Total Laser Time:[/] {time_str}")
        else:
            exposure_lines.append(f"[{theme.muted}]No light exposure recorded[/]")
        sections.append(("Light Exposure", exposure_lines))

        # === Hatching Status ===
        if embryo.hatching_status:
            hatch_lines = []
            if embryo.hatching_status.get('hatched') or embryo.hatching_status.get('detected'):
                hatch_lines.append(f"[{theme.success}]✓ Hatched[/]")
                if 'timepoint' in embryo.hatching_status:
                    hatch_lines.append(f"[{theme.secondary}]Timepoint:[/] {embryo.hatching_status['timepoint']}")
                if 'confidence' in embryo.hatching_status:
                    hatch_lines.append(f"[{theme.secondary}]Confidence:[/] {embryo.hatching_status['confidence']}")
            else:
                hatch_lines.append(f"[{theme.muted}]Not hatched[/]")
            sections.append(("Hatching", hatch_lines))

        # === Perception Results ===
        perception_mgr = getattr(self.copilot, 'perception_manager', None)
        if perception_mgr:
            session = perception_mgr.get_session(embryo.id)
            if session and session.observations:
                perc_lines = []
                current_stage = session.get_current_stage() or "unknown"
                perc_lines.append(f"[{theme.success}]Current Stage:[/] {current_stage}")
                perc_lines.append(f"[{theme.secondary}]Observations:[/] {len(session.observations)}")

                # Show last few observations
                recent = session.get_recent_observations(3)
                if recent:
                    perc_lines.append(f"[{theme.secondary}]Recent:[/]")
                    for obs in recent:
                        conf_pct = f"{obs.confidence:.0%}" if obs.confidence else "?"
                        hatching = " (hatching)" if obs.is_hatching else ""
                        perc_lines.append(f"    T{obs.timepoint}: {obs.stage} [{conf_pct}]{hatching}")
                sections.append(("Perception", perc_lines))
            else:
                sections.append(("Perception", [f"[{theme.muted}]No observations yet[/]"]))
        else:
            sections.append(("Perception", [f"[{theme.muted}]Perception not initialized[/]"]))

        # === Focus History ===
        if hasattr(embryo, 'focus_history') and embryo.focus_history:
            focus_summary = embryo.get_focus_summary()
            focus_lines = focus_summary.split('\n')
            sections.append(("Focus History", focus_lines))

        # === Build the panel content ===
        content_parts = []
        for section_name, lines in sections:
            content_parts.append(f"[bold {theme.info}]{section_name}[/]")
            for line in lines:
                content_parts.append(f"  {line}")
            content_parts.append("")

        content = "\n".join(content_parts).rstrip()

        panel = Panel(
            content,
            title=f"[bold]{title}[/]",
            border_style=theme.secondary,
            padding=(1, 2),
        )
        self.console.print(panel)

    async def stream_copilot_response(self, message: str):
        """
        Handle message with streaming response display.

        Text is printed in segments - before each tool call and after all tools complete.
        This provides better UX than waiting for everything to finish.

        Parameters
        ----------
        message : str
            User message
        """
        theme = get_theme()

        # Accumulated response for current segment
        response_text = ""
        segment_count = 0

        def flush_text_segment():
            """Print accumulated text as a panel and reset"""
            nonlocal response_text, segment_count
            if response_text.strip():
                timestamp = datetime.now().strftime("%H:%M:%S")
                panel = Panel(
                    Markdown(response_text),
                    title=f"[{theme.muted}]{timestamp}[/] [{theme.copilot} bold]{theme.icon_copilot}[/]",
                    title_align="left",
                    border_style=theme.copilot,
                    box=box.ROUNDED,
                )
                self.console.print(panel)
                segment_count += 1
                response_text = ""

        # Get the stream iterator so we can manage Progress context and asend() separately
        stream_iter = self.copilot.handle_message_stream(message).__aiter__()
        pending_choice_result = None  # Result to send back via asend()

        while True:
            # Use Progress context for normal streaming, exit it for interactive UI
            progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                transient=True,
            )
            progress.start()
            task = progress.add_task("[cyan]Thinking...", total=None)

            try:
                while True:
                    try:
                        # Use asend() if we have a choice result to send back
                        if pending_choice_result is not None:
                            chunk = await stream_iter.asend(pending_choice_result)
                            pending_choice_result = None
                        else:
                            chunk = await stream_iter.__anext__()
                    except StopAsyncIteration:
                        progress.stop()
                        flush_text_segment()
                        return  # Done with stream

                    if chunk.get('type') == 'text':
                        response_text += chunk.get('text', '')
                        progress.update(task, description=f"[cyan]Copilot is responding...")
                    elif chunk.get('type') == 'choice_request':
                        # COMPLETELY stop Progress before running interactive picker
                        progress.stop()
                        flush_text_segment()

                        # Run the interactive picker OUTSIDE the Progress context
                        choice_data = chunk.get('choice_data', {})
                        user_selection = await self.interactive_choice_picker(choice_data)
                        pending_choice_result = user_selection

                        break  # Exit inner loop, will send result on next iteration
                    elif chunk.get('type') == 'tool_start':
                        tool_name = chunk.get('tool_name', 'unknown')

                        # Flush any accumulated text BEFORE showing tool
                        progress.stop()
                        flush_text_segment()

                        # Skip tool panel for ask_user_choice - the interactive picker handles UI
                        if tool_name != 'ask_user_choice':
                            self.print_tool_call(
                                tool_name,
                                chunk.get('tool_input', {}),
                                None  # No duration yet - tool is starting
                            )
                        # Reset progress to show tool is running
                        progress = Progress(
                            SpinnerColumn(),
                            TextColumn("[progress.description]{task.description}"),
                            transient=True,
                        )
                        progress.start()
                        task = progress.add_task(f"[cyan]Running {tool_name}...", total=None)
                    elif chunk.get('type') == 'tool_call':
                        # Tool finished - update progress description with duration
                        tool_name = chunk.get('tool_name', 'unknown')
                        duration = chunk.get('duration')

                        progress.stop()
                        # Print duration if we want to show it
                        if duration:
                            self.console.print(f"   [dim]{duration:.2f}s[/dim]")

                        # Reset progress for next operation
                        progress = Progress(
                            SpinnerColumn(),
                            TextColumn("[progress.description]{task.description}"),
                            transient=True,
                        )
                        progress.start()
                        task = progress.add_task("[cyan]Thinking...", total=None)
            finally:
                progress.stop()

    async def handle_slash_command(self, command: str) -> Optional[bool]:
        """
        Handle slash commands directly

        Returns
        -------
        True if should quit, False if handled (continue loop), None if not a slash command
        """
        from .command_registry import get_command_registry

        cmd = command.strip().lower()

        # Extract command name (first word) for registry validation
        cmd_name = cmd.split()[0] if cmd.split() else cmd
        registry = get_command_registry()

        # Validate command exists in registry (prevents ghost commands)
        if not registry.get(cmd_name):
            return None  # Not a registered command, send to copilot

        if cmd in ['/quit', '/exit', '/q']:
            return True  # Signal to quit

        elif cmd == '/clear':
            self.console.clear()
            self.print_welcome()
            return False  # Handled, continue loop

        elif cmd == '/help' or cmd.startswith('/help '):
            # Support /help and /help <command>
            parts = command.strip().split(maxsplit=1)
            help_cmd = parts[1] if len(parts) > 1 else None
            self.print_help(help_cmd)
            return False  # Handled, continue loop

        elif cmd == '/status':
            panel = self.create_status_panel()
            self.console.print(panel)
            return False  # Handled, continue loop

        elif cmd == '/detectors':
            # Show perception system status (replaces old detector registry)
            theme = get_theme()
            perception_mgr = getattr(self.copilot, 'perception_manager', None)
            if perception_mgr and perception_mgr.sessions:
                self.console.print(f"[{theme.info}]Perception Sessions:[/]")
                for embryo_id, session in perception_mgr.sessions.items():
                    stage = session.get_current_stage() or "unknown"
                    obs_count = len(session.observations)
                    self.console.print(f"  {embryo_id}: stage={stage}, {obs_count} observations")
            else:
                self.console.print(f"[{theme.muted}]No active perception sessions[/]")
            return False  # Handled, continue loop

        elif cmd.startswith('/embryos'):
            # List embryos or show details for specific embryo
            parts = cmd.split(maxsplit=1)
            if len(parts) > 1:
                # Show details for specific embryo
                embryo_id = parts[1].strip()
                embryo = self.copilot.experiment.get_embryo_by_any_name(embryo_id)
                if embryo:
                    self.print_embryo_details(embryo)
                else:
                    theme = get_theme()
                    available = list(self.copilot.experiment.embryos.keys())
                    self.console.print(f"[{theme.error}]Embryo '{embryo_id}' not found.[/]")
                    if available:
                        self.console.print(f"[{theme.muted}]Available: {', '.join(available)}[/]")
            else:
                # List all embryos
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

        elif cmd == '/tokens':
            # Show token usage for session
            theme = get_theme()
            self.console.print(f"\n[bold {theme.primary}]Token Usage[/]")

            # Current context size
            context_tokens = self.copilot.current_context_tokens
            self.console.print(f"  [bold]Current context:[/] {context_tokens:,} tokens (~{context_tokens/1000:.1f}K)")

            # Cumulative usage breakdown
            cache_read = self.copilot.cache_read_tokens
            cache_created = self.copilot.cache_creation_tokens
            total_input = self.copilot.total_input_tokens + cache_read + cache_created
            total_output = self.copilot.total_output_tokens
            total_cumulative = total_input + total_output

            self.console.print(f"  [bold]Cumulative (billed):[/] {total_cumulative:,} tokens")
            self.console.print(f"    Input: {total_input:,} (cached: {cache_read:,})")
            self.console.print(f"    Output: {total_output:,}")
            self.console.print(f"    API calls: {self.copilot.api_call_count}")

            # Cost
            input_cost = self.copilot.total_input_tokens * 0.003 / 1000
            cache_read_cost = cache_read * 0.0003 / 1000
            cache_write_cost = cache_created * 0.006 / 1000
            output_cost = total_output * 0.015 / 1000
            total_cost = input_cost + cache_read_cost + cache_write_cost + output_cost
            self.console.print(f"  [bold]Est. cost:[/] ${total_cost:.3f}\n")

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

        elif cmd.startswith('/import-embryos'):
            # Import embryos from another session
            theme = get_theme()
            parts = cmd.split(maxsplit=1)

            # If session ID or 'last' provided
            if len(parts) > 1:
                arg = parts[1].strip().lower()

                # Handle 'last' shortcut - import from most recent session with embryos
                if arg == 'last':
                    sessions = self.copilot.list_sessions()
                    sessions_with_embryos = [s for s in sessions if s.get('embryo_count', 0) > 0]
                    if not sessions_with_embryos:
                        self.console.print(f"[{theme.muted}]No sessions with embryos found.[/]")
                        return False
                    # Most recent is first (sessions are sorted by last_active desc)
                    session_id = sessions_with_embryos[0].get('session_id', '')
                    if not session_id:
                        self.console.print(f"[{theme.error}]✗ Could not find last session[/]")
                        return False
                else:
                    session_id = parts[1].strip()  # Use original case for session ID

                result = self.copilot.import_embryos_from_session(session_id)
                if result.get('success'):
                    imported = result.get('imported', [])
                    self.console.print(f"[{theme.success}]✓ Imported {len(imported)} embryo(s) from {session_id}[/]")
                    if imported:
                        self.console.print(f"[{theme.info}]  {', '.join(imported)}[/]")
                    if result.get('skipped'):
                        self.console.print(f"[{theme.muted}]  Skipped (exist): {', '.join(result['skipped'])}[/]")
                else:
                    self.console.print(f"[{theme.error}]✗ {result.get('error', 'Import failed')}[/]")
            else:
                # Show session picker
                sessions = self.copilot.list_sessions()
                if not sessions:
                    self.console.print(f"[{theme.muted}]No sessions found.[/]")
                else:
                    # Filter to sessions with embryos
                    sessions_with_embryos = [s for s in sessions if s.get('embryo_count', 0) > 0]
                    if not sessions_with_embryos:
                        self.console.print(f"[{theme.muted}]No sessions with embryos found.[/]")
                    else:
                        self.console.print(f"\n[{theme.info}]Sessions with embryos:[/]")
                        for i, s in enumerate(sessions_with_embryos[:10], 1):
                            sid = s.get('session_id', '')[:30]
                            embryo_count = s.get('embryo_count', 0)
                            last_active = s.get('last_active', '')[:16]
                            self.console.print(f"  [{theme.secondary}]{i}.[/] {sid} [{theme.info}]({embryo_count} embryos)[/] [{theme.muted}]{last_active}[/]")

                        self.console.print(f"\n[{theme.muted}]Enter number to import, or session ID:[/]")
                        choice = Prompt.ask("Import from", default="")

                        if choice:
                            # Check if it's a number
                            try:
                                idx = int(choice) - 1
                                if 0 <= idx < len(sessions_with_embryos):
                                    session_id = sessions_with_embryos[idx]['session_id']
                                else:
                                    self.console.print(f"[{theme.error}]Invalid selection[/]")
                                    return False
                            except ValueError:
                                session_id = choice

                            result = self.copilot.import_embryos_from_session(session_id)
                            if result.get('success'):
                                imported = result.get('imported', [])
                                self.console.print(f"[{theme.success}]✓ Imported {len(imported)} embryo(s)[/]")
                                if imported:
                                    self.console.print(f"[{theme.info}]  {', '.join(imported)}[/]")
                            else:
                                self.console.print(f"[{theme.error}]✗ {result.get('error', 'Import failed')}[/]")

            return False  # Handled, continue loop

        elif cmd.startswith('/make-video'):
            # Generate timelapse video from volumes
            from .video_maker import make_session_videos, discover_volumes, create_timelapse_video
            from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

            theme = get_theme()
            parts = cmd.split()

            # Parse arguments
            embryo_id = None
            fps = 10

            i = 1
            while i < len(parts):
                if parts[i] == '--fps' and i + 1 < len(parts):
                    try:
                        fps = int(parts[i + 1])
                    except ValueError:
                        pass
                    i += 2
                elif not parts[i].startswith('--'):
                    embryo_id = parts[i]
                    i += 1
                else:
                    i += 1

            session_id = self.copilot.session_id
            if not session_id:
                self.console.print(f"[{theme.error}]No active session[/]")
                return False

            # Get storage path
            storage_path = self.copilot.storage_path
            session_images_dir = storage_path / "images" / session_id

            if not session_images_dir.exists():
                self.console.print(f"[{theme.error}]No images found for session {session_id}[/]")
                return False

            # Discover volumes
            all_volumes = discover_volumes(session_images_dir, embryo_id)

            if not all_volumes:
                self.console.print(f"[{theme.muted}]No timelapse volumes found[/]")
                return False

            self.console.print(f"\n[bold {theme.primary}]Creating Timelapse Videos[/]")
            self.console.print(f"[{theme.muted}]Session: {session_id}[/]")
            self.console.print(f"[{theme.muted}]FPS: {fps}[/]\n")

            for eid, volumes in all_volumes.items():
                self.console.print(f"[{theme.info}]{eid}:[/] {len(volumes)} frames")

            self.console.print()

            # Generate videos with progress
            output_dir = storage_path / "videos" / session_id
            results = {}

            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                console=self.console
            ) as progress:
                for eid, volumes in all_volumes.items():
                    task = progress.add_task(f"[cyan]{eid}", total=len(volumes))

                    output_path = output_dir / f"{eid}_timelapse.mp4"

                    def update_progress(current, total):
                        progress.update(task, completed=current)

                    result = create_timelapse_video(
                        volume_paths=volumes,
                        output_path=output_path,
                        fps=fps,
                        add_timestamps=True,
                        embryo_id=eid,
                        progress_callback=update_progress
                    )

                    progress.update(task, completed=len(volumes))
                    results[eid] = result

            # Print results
            self.console.print()
            for eid, result in results.items():
                if result.get('success'):
                    path = result['output_path']
                    frames = result['frame_count']
                    duration = result['duration_seconds']
                    self.console.print(f"[{theme.success}]+ {eid}:[/] {frames} frames, {duration:.1f}s")
                    self.console.print(f"  [{theme.muted}]{path}[/]")
                else:
                    self.console.print(f"[{theme.error}]x {eid}:[/] {result.get('error', 'Failed')}")

            self.console.print()
            return False  # Handled, continue loop

        elif cmd == '/timelapse' or cmd == '/timelapse watch':
            # Show timelapse status (with optional live watch mode)
            watch_mode = 'watch' in cmd
            theme = get_theme()

            if not (hasattr(self.copilot, 'timelapse_orchestrator') and self.copilot.timelapse_orchestrator):
                self.console.print(f"\n[{theme.muted}]No timelapse running[/]")
                self.console.print()
                return False

            from .timelapse_orchestrator import TimelapseStatus

            def build_timelapse_display():
                """Build the timelapse status display"""
                from rich.table import Table
                from rich.panel import Panel
                from rich.text import Text
                from datetime import datetime

                state = self.copilot.timelapse_orchestrator.get_status()
                is_running = state.status == TimelapseStatus.RUNNING

                # Build table
                table = Table(show_header=True, header_style=f"bold {theme.primary}", box=box.SIMPLE)
                table.add_column("Embryo")
                table.add_column("Timepoints")
                table.add_column("Last Imaged")
                table.add_column("Next In", justify="right")
                table.add_column("Status")

                if state.embryos:
                    now = datetime.now()
                    # Round-based: all embryos share the same next round time
                    next_round_secs = state.seconds_until_next_round

                    for eid, emb_state in state.embryos.items():
                        # Timepoints acquired shows progress
                        tp_str = str(emb_state.timepoints_acquired)

                        # Next acquisition countdown (shared for all active embryos)
                        if emb_state.is_complete:
                            next_str = "-"
                            status = f"[{theme.success}]done[/]"
                        elif next_round_secs is not None:
                            if next_round_secs <= 0:
                                next_str = f"[bold {theme.warning}]NOW[/]"
                            elif next_round_secs < 60:
                                next_str = f"[bold {theme.info}]{int(next_round_secs)}s[/]"
                            else:
                                mins = int(next_round_secs // 60)
                                secs = int(next_round_secs % 60)
                                next_str = f"{mins}m {secs}s"
                            status = f"[{theme.info}]active[/]"
                        else:
                            next_str = "?"
                            status = f"[{theme.muted}]waiting[/]"

                        table.add_row(
                            eid,
                            tp_str,
                            f"round {state.current_round}",
                            next_str,
                            status
                        )

                # Header with status
                status_color = theme.success if is_running else theme.muted
                status_text = state.status.value.upper()
                header = Text()
                header.append("Timelapse ", style=f"bold {theme.primary}")
                header.append(f"[{status_text}]", style=status_color)
                if watch_mode:
                    header.append(" (press Ctrl+C to exit)", style=theme.muted)

                return Panel(table, title=header, border_style=theme.primary)

            if watch_mode:
                # Live updating display
                from rich.live import Live
                import time

                self.console.print(f"\n[{theme.info}]Starting live timelapse monitor...[/]")

                try:
                    with Live(build_timelapse_display(), refresh_per_second=1, console=self.console) as live:
                        while True:
                            time.sleep(1)
                            state = self.copilot.timelapse_orchestrator.get_status()
                            live.update(build_timelapse_display())
                            # Exit if timelapse stopped
                            if state.status != TimelapseStatus.RUNNING:
                                break
                except KeyboardInterrupt:
                    self.console.print(f"\n[{theme.muted}]Exited watch mode[/]")
            else:
                # Static display
                self.console.print()
                self.console.print(build_timelapse_display())

            self.console.print()
            return False  # Handled, continue loop

        elif cmd.startswith('/timeline'):
            # Show timeline of timelapse and detection events
            theme = get_theme()
            timeline = self.copilot.timeline_manager

            if not timeline:
                self.console.print(f"[{theme.muted}]Timeline not available[/]")
                return False

            # Parse command arguments
            parts = command.strip().split()
            args = parts[1:] if len(parts) > 1 else []

            # Check for subcommands
            if args and args[0] == 'clear':
                # Clear timeline
                before_time = None
                if len(args) > 1 and args[1] == '--before':
                    if len(args) > 2:
                        delta = parse_time_delta(args[2])
                        if delta:
                            before_time = datetime.now() - delta
                count = timeline.clear_events(before=before_time)
                self.console.print(f"[{theme.success}]Cleared {count} timeline events[/]")
                return False

            # Parse filter options
            event_filter = None
            embryo_filter = None
            since_time = None
            show_all_sessions = False
            style = "letters"  # Default style: letters, log, table, axis

            i = 0
            while i < len(args):
                arg = args[i].lower()
                if arg == '--filter' and i + 1 < len(args):
                    event_filter = args[i + 1].lower()
                    i += 2
                elif arg == '--embryo' and i + 1 < len(args):
                    embryo_filter = args[i + 1]
                    i += 2
                elif arg == '--since' and i + 1 < len(args):
                    delta = parse_time_delta(args[i + 1])
                    if delta:
                        since_time = datetime.now() - delta
                    i += 2
                elif arg == '--style' and i + 1 < len(args):
                    style = args[i + 1].lower()
                    i += 2
                elif arg == '--all':
                    show_all_sessions = True
                    i += 1
                # Shortcuts for styles
                elif arg in ('--log', '--table', '--axis', '--letters'):
                    style = arg[2:]  # Remove '--'
                    i += 1
                else:
                    i += 1

            # Get filtered events (default: current session only)
            events = timeline.get_events(
                event_type=event_filter,
                embryo_id=embryo_filter,
                since=since_time,
                session_id="all" if show_all_sessions else "current",
                limit=50,
            )

            if not events:
                self.console.print(f"[{theme.muted}]No timeline events found[/]")
                return False

            # Display based on style
            if style == "table":
                self.print_timeline_list(events, compact=True)
            elif style == "axis":
                self.print_timeline_axis(events)
            elif style == "letters":
                self.print_timeline_letters(events)
            else:  # Default: log style
                self.print_timeline_horizontal(events)

            return False  # Handled, continue loop

        return None  # Not a slash command, send to copilot

    def _extract_text_from_content(self, content) -> str:
        """Extract text from message content (handles both str and list of blocks)"""
        if isinstance(content, str):
            return content

        if isinstance(content, list):
            text_parts = []
            for block in content:
                # Handle Claude API objects
                if hasattr(block, 'text'):
                    text_parts.append(block.text)
                # Handle dicts (from JSON restore)
                elif isinstance(block, dict) and block.get('text'):
                    text_parts.append(block['text'])
                # Skip tool_use blocks for display
            return "\n\n".join(text_parts)

        return str(content) if content else ""

    def print_conversation_history(self, limit: int = 10, show_header: bool = True):
        """Print recent conversation history with same formatting as live session"""
        theme = get_theme()
        history = self.copilot.conversation_history[-limit:]

        if not history:
            self.console.print("No conversation history yet.", style=theme.muted)
            return

        if show_header:
            self.console.print(Panel(
                Text(f"Showing last {len(history)} messages", style=theme.info),
                title="Conversation History",
                border_style=theme.info,
            ))
            self.console.print()

        for msg in history:
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')
            text = self._extract_text_from_content(content)

            if not text:
                continue

            if role == 'user':
                # Same format as print_user_message
                panel = Panel(
                    Text(text, style=theme.user),
                    title=f"[{theme.user} bold]{theme.icon_user}[/]",
                    title_align="left",
                    border_style=theme.user,
                    box=box.ROUNDED,
                )
                self.console.print(panel)

            elif role == 'assistant':
                # Same format as print_copilot_message (with markdown)
                panel = Panel(
                    Markdown(text),
                    title=f"[{theme.copilot} bold]{theme.icon_copilot}[/]",
                    title_align="left",
                    border_style=theme.copilot,
                    box=box.ROUNDED,
                )
                self.console.print(panel)

        self.console.print()

    def print_help(self, command: Optional[str] = None):
        """Print help message (auto-generated from command registry)

        Parameters
        ----------
        command : str, optional
            Specific command to show detailed help for (e.g., "timeline")
        """
        from .command_registry import get_command_registry
        registry = get_command_registry()

        if command:
            # Show detailed help for specific command
            cmd_name = command if command.startswith('/') else f'/{command}'
            help_text = registry.generate_command_help(cmd_name)
            if help_text:
                self.console.print(Markdown(help_text))
            else:
                theme = get_theme()
                self.console.print(f"[{theme.error}]Unknown command: {command}[/]")
                self.console.print(f"[{theme.muted}]Use /help to see all commands[/]")
        else:
            # Show full help
            help_text = registry.generate_help_markdown()
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
            self.print_conversation_history(show_header=False)
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

                    # Check if thinking mode will be triggered
                    if self.copilot._should_use_thinking(user_input):
                        self.console.print(f"[{theme.info}]💭 Extended thinking enabled[/]")

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

            # Clean up viz server
            if self.copilot.viz_server:
                await self.copilot.stop_viz_server()

            # Clean up client session
            if self.copilot.client:
                await self.copilot.client.disconnect()


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
