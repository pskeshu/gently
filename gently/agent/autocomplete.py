"""
Autocomplete support for Rich CLI

Provides:
- Dropdown completion for slash commands with descriptions
- Sub-command and option completion
- Shadow text suggestions with argument hints
- Dynamic completions (embryo IDs, session IDs, themes)
"""

from typing import Iterable, List, Optional

from prompt_toolkit.auto_suggest import AutoSuggest, Suggestion
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.completion import Completer, Completion
from prompt_toolkit.document import Document
from prompt_toolkit.history import History

from .command_registry import get_command_registry, CommandDefinition


class CopilotCompleter(Completer):
    """
    Dropdown autocompleter for microscopy copilot CLI

    Uses CommandRegistry for:
    - Slash commands with descriptions in meta
    - Sub-commands after command names
    - Options/flags after commands
    - Dynamic values (embryo IDs, session IDs, theme names)
    """

    # Common natural language patterns
    COMMON_PHRASES = [
        'What detectors do we have?',
        'Show me the detection summary',
        'Add a detector for',
        'Test the',
        'Enable',
        'Disable',
        "What's the status?",
        'How is embryo',
        'Analyze embryo',
        'List all embryos',
        'Show me embryo',
        'Generate a prompt for detecting',
        'Start imaging',
    ]

    def __init__(self, copilot=None):
        """
        Initialize completer

        Parameters
        ----------
        copilot : MicroscopyCopilot, optional
            Copilot instance for dynamic completions
        """
        self.copilot = copilot
        self._registry = get_command_registry()

    def get_embryo_ids(self) -> List[str]:
        """Get list of embryo IDs from copilot"""
        if self.copilot and hasattr(self.copilot, 'experiment'):
            return list(self.copilot.experiment.embryos.keys())
        return []

    def get_detector_names(self) -> List[str]:
        """Get list of detector names from copilot"""
        if self.copilot and hasattr(self.copilot, 'detector_registry'):
            return [d.name for d in self.copilot.detector_registry.list_all()]
        return []

    def get_session_ids(self) -> List[str]:
        """Get list of session IDs from copilot"""
        if self.copilot and hasattr(self.copilot, 'list_sessions'):
            try:
                sessions = self.copilot.list_sessions()
                return [s.get('session_id', s.get('id', '')) for s in sessions if s]
            except Exception:
                pass
        return []

    def get_theme_names(self) -> List[str]:
        """Get list of available theme names"""
        try:
            from .theme import list_themes
            return list(list_themes().keys())
        except Exception:
            return ['vibrant', 'scientific', 'claude', 'monochrome']

    def get_completions(self, document: Document, complete_event) -> Iterable[Completion]:
        """Generate completions based on current input"""
        text = document.text_before_cursor

        # Handle slash commands
        if text.startswith('/'):
            yield from self._complete_slash_command(text)
            return

        # Handle natural language
        yield from self._complete_natural_language(document)

    def _complete_slash_command(self, text: str) -> Iterable[Completion]:
        """Complete slash commands, sub-commands, and options"""
        parts = text.split()
        cmd_name = parts[0].lower() if parts else ""

        # Case 1: Still typing command name (e.g., "/ti")
        if len(parts) == 1 and not text.endswith(' '):
            yield from self._complete_command_name(text)
            return

        # Get command definition
        cmd_def = self._registry.get(cmd_name)
        if not cmd_def:
            # Unknown command - still offer command completion
            yield from self._complete_command_name(cmd_name)
            return

        # What comes after the command
        after_cmd = text[len(cmd_name):].lstrip()
        current_word = parts[-1] if len(parts) > 1 else ""
        at_space = text.endswith(' ')

        # Case 2: Completing sub-command or positional
        if len(parts) == 1 and at_space:
            # Just finished typing command, show sub-commands and positional hints
            yield from self._complete_subcommand(cmd_def, "", at_space)
            yield from self._complete_positional(cmd_def, "", at_space)
            yield from self._complete_options(cmd_def, after_cmd, "", at_space)
            return

        if len(parts) == 2 and not current_word.startswith('-'):
            # Typing first argument - could be sub-command or positional
            if not at_space:
                yield from self._complete_subcommand(cmd_def, current_word, at_space)
                yield from self._complete_positional(cmd_def, current_word, at_space)
            else:
                # Finished first arg, now show options
                yield from self._complete_options(cmd_def, after_cmd, "", at_space)
            return

        # Case 3: Completing option (e.g., "--f" -> "--filter")
        if current_word.startswith('-') and not at_space:
            yield from self._complete_options(cmd_def, after_cmd, current_word, at_space)
            return

        # Case 4: At space after option - show option values or more options
        if at_space:
            # Check if previous word was an option that takes a value
            prev_word = parts[-1] if parts else ""
            yield from self._complete_option_value(cmd_def, prev_word, "", at_space)
            yield from self._complete_options(cmd_def, after_cmd, "", at_space)
            return

        # Case 5: Typing option value
        if len(parts) >= 3:
            prev_word = parts[-2]
            yield from self._complete_option_value(cmd_def, prev_word, current_word, at_space)

    def _complete_command_name(self, text: str) -> Iterable[Completion]:
        """Complete slash command names with descriptions"""
        # Track what we've already yielded to avoid duplicates
        yielded = set()
        search_text = text[1:] if text.startswith('/') else text  # Strip leading /

        # First pass: prefix matches (higher priority)
        for cmd in self._registry.get_all():
            if cmd.name.startswith(text):
                yielded.add(cmd.name)
                yield Completion(
                    cmd.name,
                    start_position=-len(text),
                    display=cmd.name,
                    display_meta=cmd.description,
                )
            # Match aliases
            for alias in cmd.aliases:
                if alias.startswith(text):
                    yielded.add(alias)
                    yield Completion(
                        alias,
                        start_position=-len(text),
                        display=alias,
                        display_meta=f"{cmd.description} (→{cmd.name})",
                    )

        # Second pass: substring matches (lower priority)
        for cmd in self._registry.get_all():
            if cmd.name not in yielded and search_text in cmd.name:
                yield Completion(
                    cmd.name,
                    start_position=-len(text),
                    display=cmd.name,
                    display_meta=cmd.description,
                )

    def _complete_subcommand(self, cmd: CommandDefinition, current: str,
                             at_space: bool) -> Iterable[Completion]:
        """Complete sub-commands like 'watch' for /timelapse"""
        for sub in cmd.subcommands:
            if at_space or sub.name.startswith(current):
                yield Completion(
                    sub.name,
                    start_position=-len(current) if not at_space else 0,
                    display=sub.name,
                    display_meta=sub.description,
                )

    def _complete_positional(self, cmd: CommandDefinition, current: str,
                             at_space: bool) -> Iterable[Completion]:
        """Complete positional arguments (embryo IDs, session IDs, themes)"""
        if not cmd.positional_arg:
            return

        choices = []

        # Dynamic choices based on command
        if cmd.name in ["/resume", "/import-embryos"]:
            choices = self.get_session_ids()
            # Add 'last' for import-embryos
            if cmd.name == "/import-embryos":
                choices = ['last'] + choices
        elif cmd.name == "/embryos":
            choices = self.get_embryo_ids()
        elif cmd.name == "/theme":
            choices = self.get_theme_names()
        elif cmd.name == "/help":
            # Complete with command names (without /)
            choices = [c.name[1:] for c in self._registry.get_all()]

        for choice in choices:
            if at_space or choice.startswith(current):
                yield Completion(
                    choice,
                    start_position=-len(current) if not at_space else 0,
                    display=choice,
                    display_meta=cmd.positional_arg,
                )

    def _complete_options(self, cmd: CommandDefinition, after_cmd: str,
                          current: str, at_space: bool) -> Iterable[Completion]:
        """Complete options like --filter for /timeline"""
        # Collect all options (from command and active subcommand)
        all_options = list(cmd.options)

        # Check if a subcommand is active
        for sub in cmd.subcommands:
            if sub.name in after_cmd.split():
                all_options.extend(sub.options)

        # Filter out already-used options
        used_options = {p for p in after_cmd.split() if p.startswith('-')}

        for opt in all_options:
            if opt.name in used_options:
                continue

            if at_space or opt.name.startswith(current):
                # Build display with value hint
                display = opt.name
                if opt.takes_value:
                    display += f" {opt.value_hint or 'VALUE'}"

                # Build meta with choices
                meta = opt.description
                if opt.value_choices:
                    meta += f" ({'/'.join(opt.value_choices)})"

                yield Completion(
                    opt.name,
                    start_position=-len(current) if current.startswith('-') else 0,
                    display=display,
                    display_meta=meta,
                )

    def _complete_option_value(self, cmd: CommandDefinition, option: str,
                               current: str, at_space: bool) -> Iterable[Completion]:
        """Complete option values like 'timelapse' after --filter"""
        # Find the option definition
        all_options = list(cmd.options)
        for sub in cmd.subcommands:
            all_options.extend(sub.options)

        opt = None
        for o in all_options:
            if o.name == option or o.short == option:
                opt = o
                break

        if not opt or not opt.takes_value:
            return

        # Static choices from option definition
        for choice in opt.value_choices:
            if at_space or choice.startswith(current):
                yield Completion(
                    choice,
                    start_position=-len(current) if not at_space else 0,
                    display=choice,
                    display_meta=f"value for {opt.name}",
                )

        # Dynamic choices based on option semantics
        if opt.name == "--embryo":
            for eid in self.get_embryo_ids():
                if at_space or eid.startswith(current):
                    yield Completion(
                        eid,
                        start_position=-len(current) if not at_space else 0,
                        display=eid,
                        display_meta="embryo",
                    )

    def _complete_natural_language(self, document: Document) -> Iterable[Completion]:
        """Complete natural language phrases and entity names"""
        text = document.text_before_cursor
        word = document.get_word_before_cursor(WORD=True)

        # Embryo ID completion
        if 'embryo' in text.lower():
            for eid in self.get_embryo_ids():
                if eid.startswith(word):
                    yield Completion(
                        eid,
                        start_position=-len(word),
                        display=eid,
                        display_meta='embryo'
                    )
                # Support "embryo 1" -> "embryo_1"
                elif eid.startswith('embryo_') and word.isdigit():
                    embryo_num = eid.split('_')[1]
                    if embryo_num.lstrip('0') == word:
                        yield Completion(
                            eid,
                            start_position=-len(word),
                            display=eid,
                            display_meta='embryo'
                        )

        # Detector name completion
        elif any(t in text.lower() for t in ['detector', 'test', 'enable', 'disable']):
            for name in self.get_detector_names():
                if name.startswith(word):
                    yield Completion(
                        name,
                        start_position=-len(word),
                        display=name,
                        display_meta='detector'
                    )

        # Common phrase completion (only for short inputs)
        elif len(text.strip()) < 20:
            for phrase in self.COMMON_PHRASES:
                if phrase.lower().startswith(text.lower().strip()):
                    yield Completion(
                        phrase,
                        start_position=-len(text.strip()),
                        display=phrase,
                        display_meta='suggestion'
                    )


class CommandAutoSuggest(AutoSuggest):
    """
    Shadow text auto-suggest for commands

    Shows ghost text with:
    - Command completion (e.g., "/ti" -> "/timelapse")
    - Argument hints (e.g., "/timelapse" -> "/timelapse [watch]")

    Works alongside history suggestions (history takes priority).
    """

    def __init__(self, copilot=None, history: Optional[History] = None):
        """
        Initialize auto-suggest

        Parameters
        ----------
        copilot : MicroscopyCopilot, optional
            Copilot instance for dynamic completions
        history : History, optional
            Command history for history-based suggestions
        """
        self.copilot = copilot
        self.history = history
        self._registry = get_command_registry()

    def get_suggestion(self, buffer: Buffer, document: Document) -> Optional[Suggestion]:
        """Get suggestion for current input"""
        text = document.text_before_cursor

        # Try history first (if available)
        if self.history:
            history_suggestion = self._get_history_suggestion(text)
            if history_suggestion:
                return history_suggestion

        # Then try command suggestions
        if text.startswith('/'):
            return self._get_command_suggestion(text)

        return None

    def _get_history_suggestion(self, text: str) -> Optional[Suggestion]:
        """Get suggestion from history"""
        if not text or not self.history:
            return None

        # Search history for matching entries
        for entry in reversed(list(self.history.get_strings())):
            if entry.startswith(text) and entry != text:
                # Return the part after what user has typed
                return Suggestion(entry[len(text):])

        return None

    def _get_command_suggestion(self, text: str) -> Optional[Suggestion]:
        """Get command completion suggestion with argument hints"""
        parts = text.split()
        cmd_text = parts[0].lower() if parts else ""

        # Case 1: Completing command name
        if len(parts) == 1 and not text.endswith(' '):
            # Find first matching command
            for cmd in self._registry.get_all():
                if cmd.name.startswith(cmd_text) and cmd.name != cmd_text:
                    # Suggest rest of command name + arg hints
                    rest = cmd.name[len(cmd_text):]
                    hint = cmd.arg_hint_string()
                    if hint:
                        return Suggestion(f"{rest} {hint}")
                    return Suggestion(rest)

                # Check aliases
                for alias in cmd.aliases:
                    if alias.startswith(cmd_text) and alias != cmd_text:
                        rest = alias[len(cmd_text):]
                        return Suggestion(rest)

            return None

        # Case 2: Command complete, show arg hints
        cmd_def = self._registry.get(cmd_text)
        if not cmd_def:
            return None

        # If user just typed the command (with space), show hints
        if len(parts) == 1 and text.endswith(' '):
            hint = cmd_def.arg_hint_string()
            if hint:
                return Suggestion(hint)
            return None

        # Case 3: Completing sub-command
        if len(parts) == 2 and not text.endswith(' '):
            current = parts[1]
            for sub in cmd_def.subcommands:
                if sub.name.startswith(current) and sub.name != current:
                    return Suggestion(sub.name[len(current):])

        # Case 4: Completing option name
        if len(parts) >= 2:
            current = parts[-1]
            if current.startswith('-') and not text.endswith(' '):
                for opt in cmd_def.options:
                    if opt.name.startswith(current) and opt.name != current:
                        rest = opt.name[len(current):]
                        if opt.takes_value:
                            return Suggestion(f"{rest} {opt.value_hint or 'VALUE'}")
                        return Suggestion(rest)

        return None


def create_completer(copilot=None) -> CopilotCompleter:
    """
    Create dropdown autocompleter for copilot CLI

    Parameters
    ----------
    copilot : MicroscopyCopilot, optional
        Copilot instance for dynamic completions

    Returns
    -------
    CopilotCompleter
        Configured completer
    """
    return CopilotCompleter(copilot=copilot)


def create_auto_suggest(copilot=None, history: Optional[History] = None) -> CommandAutoSuggest:
    """
    Create shadow text auto-suggest for copilot CLI

    Parameters
    ----------
    copilot : MicroscopyCopilot, optional
        Copilot instance for dynamic completions
    history : History, optional
        Command history for history-based suggestions

    Returns
    -------
    CommandAutoSuggest
        Configured auto-suggest
    """
    return CommandAutoSuggest(copilot=copilot, history=history)
