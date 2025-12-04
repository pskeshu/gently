"""
Unified Slash Command Registry for Microscopy Copilot CLI

Provides a single source of truth for:
- Command definitions with metadata
- Sub-commands and options
- Autocomplete support
- Auto-generated help text
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Callable, Dict, List, Optional


class CommandCategory(Enum):
    """Categories for organizing commands in help and welcome"""
    NAVIGATION = auto()    # /quit, /clear, /help
    INSPECTION = auto()    # /status, /detectors, /embryos, /timelapse, /timeline
    SESSION = auto()       # /sessions, /resume, /save, /import-embryos
    APPEARANCE = auto()    # /theme, /history, /tokens


@dataclass
class CommandOption:
    """Definition of a command option/flag"""
    name: str                                    # e.g., "--filter"
    description: str = ""
    short: Optional[str] = None                  # e.g., "-f"
    takes_value: bool = False                    # True if option requires a value
    value_choices: List[str] = field(default_factory=list)  # Possible values
    value_hint: str = ""                         # e.g., "TYPE" for "--filter TYPE"
    is_flag: bool = False                        # True for boolean flags (no value)


@dataclass
class SubCommand:
    """Definition of a sub-command"""
    name: str                                    # e.g., "watch", "clear"
    description: str = ""
    options: List[CommandOption] = field(default_factory=list)


@dataclass
class CommandDefinition:
    """
    Complete definition of a slash command

    Contains all metadata needed for:
    - Autocomplete with rich descriptions
    - Help text generation
    - Welcome message generation
    """
    name: str                                    # e.g., "/timelapse" (with leading slash)
    description: str                             # Short description for autocomplete
    help_text: str = ""                          # Detailed help (multi-line OK)
    aliases: List[str] = field(default_factory=list)  # e.g., ["/q", "/exit"]
    category: CommandCategory = CommandCategory.NAVIGATION

    # Positional argument
    positional_arg: Optional[str] = None         # e.g., "embryo_id"
    positional_hint: str = ""                    # e.g., "ID or 'last'"

    # Sub-commands (e.g., /timelapse watch)
    subcommands: List[SubCommand] = field(default_factory=list)

    # Options/flags (e.g., /timeline --filter)
    options: List[CommandOption] = field(default_factory=list)

    def usage_string(self) -> str:
        """Generate usage string like '/timeline [clear] [--filter TYPE]'"""
        parts = [self.name]

        if self.subcommands:
            sub_names = [s.name for s in self.subcommands]
            parts.append(f"[{'/'.join(sub_names)}]")

        if self.positional_arg:
            hint = self.positional_hint or self.positional_arg
            parts.append(f"[{hint}]")

        for opt in self.options[:3]:  # Show first 3 options
            if opt.is_flag:
                parts.append(f"[{opt.name}]")
            elif opt.takes_value:
                parts.append(f"[{opt.name} {opt.value_hint or 'VALUE'}]")

        if len(self.options) > 3:
            parts.append("[...]")

        return " ".join(parts)

    def arg_hint_string(self) -> str:
        """Generate hint string for shadow text (shows what args are available)"""
        hints = []

        if self.subcommands:
            sub_names = [s.name for s in self.subcommands]
            hints.append(f"[{'/'.join(sub_names)}]")

        if self.positional_arg:
            hint = self.positional_hint or self.positional_arg
            hints.append(f"[{hint}]")

        # Show key options
        key_opts = [o for o in self.options if not o.is_flag][:2]
        for opt in key_opts:
            hints.append(f"[{opt.name} {opt.value_hint or 'VALUE'}]")

        return " ".join(hints)


class CommandRegistry:
    """
    Central registry for all slash commands

    Features:
    - Single source of truth for command definitions
    - Lookup by name or alias
    - Category filtering
    - Autocomplete data generation
    - Help text generation
    """

    def __init__(self):
        self._commands: Dict[str, CommandDefinition] = {}
        self._aliases: Dict[str, str] = {}  # alias -> canonical name

    def register(self, command: CommandDefinition) -> None:
        """Register a command definition"""
        self._commands[command.name] = command
        for alias in command.aliases:
            self._aliases[alias] = command.name

    def get(self, name: str) -> Optional[CommandDefinition]:
        """Get command by name or alias"""
        name = name.lower()
        canonical = self._aliases.get(name, name)
        return self._commands.get(canonical)

    def get_all(self) -> List[CommandDefinition]:
        """Get all registered commands"""
        return list(self._commands.values())

    def get_by_category(self, category: CommandCategory) -> List[CommandDefinition]:
        """Get commands in a category"""
        return [c for c in self._commands.values() if c.category == category]

    def get_all_names_and_aliases(self) -> List[str]:
        """Get all command names and aliases for autocomplete"""
        names = list(self._commands.keys())
        names.extend(self._aliases.keys())
        return sorted(set(names))

    def generate_help_markdown(self) -> str:
        """Generate complete help text in markdown format"""
        lines = [
            "# Copilot Commands",
            "",
            "## Natural Language",
            "Just type what you want! Examples:",
            "- \"What detectors do we have?\"",
            "- \"Add a detector for comma stage\"",
            "- \"Test hatching detector on embryo 1\"",
            "- \"Start imaging all embryos\"",
            "",
            "## Slash Commands",
            "",
        ]

        # Group by category
        category_names = {
            CommandCategory.NAVIGATION: "Navigation",
            CommandCategory.INSPECTION: "Inspection",
            CommandCategory.SESSION: "Session",
            CommandCategory.APPEARANCE: "Appearance",
        }

        for category in CommandCategory:
            cmds = self.get_by_category(category)
            if not cmds:
                continue

            lines.append(f"### {category_names.get(category, category.name)}")
            lines.append("")

            for cmd in sorted(cmds, key=lambda c: c.name):
                # Command with usage
                usage = cmd.usage_string()
                lines.append(f"- `{usage}` - {cmd.description}")

                # Show aliases
                if cmd.aliases:
                    aliases_str = ", ".join(f"`{a}`" for a in cmd.aliases)
                    lines.append(f"  - Aliases: {aliases_str}")

                # Show sub-commands
                for sub in cmd.subcommands:
                    lines.append(f"  - `{sub.name}` - {sub.description}")

                # Show key options (up to 4)
                for opt in cmd.options[:4]:
                    desc = opt.description
                    if opt.value_choices:
                        desc += f" ({'/'.join(opt.value_choices)})"
                    lines.append(f"  - `{opt.name}` - {desc}")

                if len(cmd.options) > 4:
                    lines.append(f"  - ... and {len(cmd.options) - 4} more options")

            lines.append("")

        # Keyboard shortcuts
        lines.extend([
            "## Keyboard Shortcuts",
            "- `Tab` - Autocomplete commands/options",
            "- `Right Arrow` - Accept shadow suggestion",
            "- `Ctrl+C` - Exit",
            "- `Ctrl+L` - Clear screen",
            "- `Ctrl+R` - Reverse search history",
        ])

        return "\n".join(lines)

    def generate_command_help(self, name: str) -> Optional[str]:
        """Generate detailed help for a specific command"""
        cmd = self.get(name)
        if not cmd:
            return None

        lines = [
            f"# {cmd.name}",
            "",
            cmd.help_text or cmd.description,
            "",
            f"**Usage:** `{cmd.usage_string()}`",
            "",
        ]

        if cmd.aliases:
            aliases_str = ", ".join(f"`{a}`" for a in cmd.aliases)
            lines.append(f"**Aliases:** {aliases_str}")
            lines.append("")

        if cmd.subcommands:
            lines.append("**Sub-commands:**")
            for sub in cmd.subcommands:
                lines.append(f"- `{sub.name}` - {sub.description}")
                for opt in sub.options:
                    opt_str = opt.name
                    if opt.takes_value:
                        opt_str += f" {opt.value_hint or 'VALUE'}"
                    lines.append(f"  - `{opt_str}` - {opt.description}")
            lines.append("")

        if cmd.options:
            lines.append("**Options:**")
            for opt in cmd.options:
                opt_str = opt.name
                if opt.short:
                    opt_str = f"{opt.short}, {opt.name}"
                if opt.takes_value:
                    opt_str += f" {opt.value_hint or 'VALUE'}"
                desc = opt.description
                if opt.value_choices:
                    desc += f" (choices: {', '.join(opt.value_choices)})"
                lines.append(f"- `{opt_str}` - {desc}")
            lines.append("")

        return "\n".join(lines)


# ============================================================================
# Default Commands Registration
# ============================================================================

def _register_default_commands(registry: CommandRegistry) -> None:
    """Register all built-in commands"""

    # === Navigation Commands ===
    registry.register(CommandDefinition(
        name="/quit",
        description="Exit the copilot",
        help_text="Exit the interactive copilot session.",
        aliases=["/exit", "/q"],
        category=CommandCategory.NAVIGATION,
    ))

    registry.register(CommandDefinition(
        name="/clear",
        description="Clear screen",
        help_text="Clear the terminal screen and show welcome banner.",
        category=CommandCategory.NAVIGATION,
    ))

    registry.register(CommandDefinition(
        name="/help",
        description="Show help",
        help_text="Show help for all commands or a specific command.\n\nUsage:\n- `/help` - Show all commands\n- `/help timeline` - Show detailed help for /timeline",
        positional_arg="command",
        positional_hint="command",
        category=CommandCategory.NAVIGATION,
    ))

    # === Inspection Commands ===
    registry.register(CommandDefinition(
        name="/status",
        description="Show experiment status",
        help_text="Display current experiment status including microscope connection, active embryos, and detector status.",
        category=CommandCategory.INSPECTION,
    ))

    registry.register(CommandDefinition(
        name="/detectors",
        description="List all detectors",
        help_text="Show a table of all registered detectors with their status, type, and configuration.",
        category=CommandCategory.INSPECTION,
    ))

    registry.register(CommandDefinition(
        name="/embryos",
        description="List embryos or show details",
        help_text="List all embryos in the current experiment. Provide an embryo ID to see detailed information about a specific embryo.",
        positional_arg="embryo_id",
        positional_hint="ID",
        category=CommandCategory.INSPECTION,
    ))

    registry.register(CommandDefinition(
        name="/timelapse",
        description="Timelapse status [watch]",
        help_text="Display timelapse acquisition status for all embryos.\n\nUse 'watch' for live updating countdown view that refreshes every second.",
        subcommands=[
            SubCommand(
                name="watch",
                description="Live countdown mode (Ctrl+C to exit)",
            ),
        ],
        category=CommandCategory.INSPECTION,
    ))

    registry.register(CommandDefinition(
        name="/timeline",
        description="Event timeline [--filter, clear]",
        help_text="""Display timeline of timelapse and detection events.

Shows events from the current session by default. Use --all to see events from all sessions.""",
        subcommands=[
            SubCommand(
                name="clear",
                description="Clear timeline history",
                options=[
                    CommandOption(
                        name="--before",
                        description="Clear events before time",
                        takes_value=True,
                        value_hint="TIME",
                    ),
                ],
            ),
        ],
        options=[
            CommandOption(
                name="--filter",
                description="Filter by event type",
                takes_value=True,
                value_choices=["timelapse", "detection"],
                value_hint="TYPE",
            ),
            CommandOption(
                name="--embryo",
                description="Filter by embryo ID",
                takes_value=True,
                value_hint="ID",
            ),
            CommandOption(
                name="--since",
                description="Show events from time period",
                takes_value=True,
                value_hint="TIME",
            ),
            CommandOption(
                name="--all",
                description="Show events from all sessions",
                is_flag=True,
            ),
            CommandOption(
                name="--letters",
                description="Lettered markers with legend (default)",
                is_flag=True,
            ),
            CommandOption(
                name="--log",
                description="Git-log style vertical timeline",
                is_flag=True,
            ),
            CommandOption(
                name="--table",
                description="Compact table view",
                is_flag=True,
            ),
            CommandOption(
                name="--axis",
                description="Simple horizontal axis",
                is_flag=True,
            ),
        ],
        category=CommandCategory.INSPECTION,
    ))

    # === Session Commands ===
    registry.register(CommandDefinition(
        name="/sessions",
        description="Browse saved sessions",
        help_text="Open interactive session browser to view and select from saved sessions.",
        category=CommandCategory.SESSION,
    ))

    registry.register(CommandDefinition(
        name="/resume",
        description="Resume a session",
        help_text="Resume a previously saved session. Opens interactive picker if no session ID is provided.",
        positional_arg="session_id",
        positional_hint="ID",
        category=CommandCategory.SESSION,
    ))

    registry.register(CommandDefinition(
        name="/save",
        description="Save current session",
        help_text="Save the current session including embryo states and conversation history.",
        category=CommandCategory.SESSION,
    ))

    registry.register(CommandDefinition(
        name="/import-embryos",
        description="Import embryos from session",
        help_text="""Import embryo definitions from another session into the current session.

Use 'last' to import from the most recent session with embryos.""",
        positional_arg="session_id",
        positional_hint="ID|last",
        category=CommandCategory.SESSION,
    ))

    # === Appearance Commands ===
    registry.register(CommandDefinition(
        name="/theme",
        description="Switch color theme",
        help_text="Change the CLI color theme.\n\nAvailable themes: vibrant, scientific, claude, monochrome",
        positional_arg="name",
        positional_hint="name",
        category=CommandCategory.APPEARANCE,
    ))

    registry.register(CommandDefinition(
        name="/history",
        description="Show conversation history",
        help_text="Display recent conversation history with the copilot.",
        category=CommandCategory.APPEARANCE,
    ))

    registry.register(CommandDefinition(
        name="/tokens",
        description="Show API token usage",
        help_text="Display token usage statistics and estimated cost for the current session.",
        category=CommandCategory.APPEARANCE,
    ))


# ============================================================================
# Global Registry
# ============================================================================

_command_registry: Optional[CommandRegistry] = None


def get_command_registry() -> CommandRegistry:
    """Get or create the global command registry"""
    global _command_registry
    if _command_registry is None:
        _command_registry = CommandRegistry()
        _register_default_commands(_command_registry)
    return _command_registry
