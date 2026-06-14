"""
Unified Slash Command Registry for Microscopy Agent CLI

Provides a single source of truth for:
- Command definitions with metadata
- Sub-commands and options
- Autocomplete support
- Auto-generated help text
"""

from dataclasses import dataclass, field
from enum import Enum, auto


class CommandCategory(Enum):
    """Categories for organizing commands in help and welcome"""

    NAVIGATION = auto()  # /quit, /clear, /help
    INSPECTION = auto()  # /status, /detectors, /embryos, /timelapse, /timeline
    SESSION = auto()  # /sessions, /resume, /save, /import-embryos
    PLANNING = auto()  # /plan
    APPEARANCE = auto()  # /theme, /history, /tokens
    DIAGNOSTICS = auto()  # /test-device


@dataclass
class CommandOption:
    """Definition of a command option/flag"""

    name: str  # e.g., "--filter"
    description: str = ""
    short: str | None = None  # e.g., "-f"
    takes_value: bool = False  # True if option requires a value
    value_choices: list[str] = field(default_factory=list)  # Possible values
    value_hint: str = ""  # e.g., "TYPE" for "--filter TYPE"
    is_flag: bool = False  # True for boolean flags (no value)


@dataclass
class SubCommand:
    """Definition of a sub-command"""

    name: str  # e.g., "watch", "clear"
    description: str = ""
    options: list[CommandOption] = field(default_factory=list)


@dataclass
class CommandDefinition:
    """
    Complete definition of a slash command

    Contains all metadata needed for:
    - Autocomplete with rich descriptions
    - Help text generation
    - Welcome message generation
    """

    name: str  # e.g., "/timelapse" (with leading slash)
    description: str  # Short description for autocomplete
    help_text: str = ""  # Detailed help (multi-line OK)
    aliases: list[str] = field(default_factory=list)  # e.g., ["/q", "/exit"]
    category: CommandCategory = CommandCategory.NAVIGATION

    # Positional argument
    positional_arg: str | None = None  # e.g., "embryo_id"
    positional_hint: str = ""  # e.g., "ID or 'last'"

    # Sub-commands (e.g., /timelapse watch)
    subcommands: list[SubCommand] = field(default_factory=list)

    # Options/flags (e.g., /timeline --filter)
    options: list[CommandOption] = field(default_factory=list)

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
        self._commands: dict[str, CommandDefinition] = {}
        self._aliases: dict[str, str] = {}  # alias -> canonical name

    def register(self, command: CommandDefinition) -> None:
        """Register a command definition"""
        self._commands[command.name] = command
        for alias in command.aliases:
            self._aliases[alias] = command.name

    def get(self, name: str) -> CommandDefinition | None:
        """Get command by name or alias"""
        name = name.lower()
        canonical = self._aliases.get(name, name)
        return self._commands.get(canonical)

    def get_all(self) -> list[CommandDefinition]:
        """Get all registered commands"""
        return list(self._commands.values())

    def get_by_category(self, category: CommandCategory) -> list[CommandDefinition]:
        """Get commands in a category"""
        return [c for c in self._commands.values() if c.category == category]

    def get_all_names_and_aliases(self) -> list[str]:
        """Get all command names and aliases for autocomplete"""
        names = list(self._commands.keys())
        names.extend(self._aliases.keys())
        return sorted(set(names))

    def generate_help_markdown(self) -> str:
        """Generate complete help text in markdown format"""
        lines = [
            "# Agent Commands",
            "",
            "## Natural Language",
            "Just type what you want! Examples:",
            '- "What detectors do we have?"',
            '- "Add a detector for comma stage"',
            '- "Test hatching detector on embryo 1"',
            '- "Start imaging all embryos"',
            "",
            "## Slash Commands",
            "",
        ]

        # Group by category
        category_names = {
            CommandCategory.NAVIGATION: "Navigation",
            CommandCategory.INSPECTION: "Inspection",
            CommandCategory.SESSION: "Session",
            CommandCategory.PLANNING: "Planning",
            CommandCategory.APPEARANCE: "Appearance",
            CommandCategory.DIAGNOSTICS: "Diagnostics",
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
        lines.extend(
            [
                "## Keyboard Shortcuts",
                "- `Tab` - Autocomplete commands/options",
                "- `Right Arrow` - Accept shadow suggestion",
                "- `Ctrl+C` - Exit",
                "- `Ctrl+L` - Clear screen",
                "- `Ctrl+R` - Reverse search history",
            ]
        )

        return "\n".join(lines)

    def generate_command_help(self, name: str) -> str | None:
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
    registry.register(
        CommandDefinition(
            name="/quit",
            description="Exit the agent",
            help_text="Exit the interactive agent session.",
            aliases=["/exit", "/q"],
            category=CommandCategory.NAVIGATION,
        )
    )

    registry.register(
        CommandDefinition(
            name="/clear",
            description="Clear screen",
            help_text="Clear the terminal screen and show welcome banner.",
            category=CommandCategory.NAVIGATION,
        )
    )

    registry.register(
        CommandDefinition(
            name="/help",
            description="Show help",
            help_text=(
                "Show help for all commands or a specific command.\n\nUsage:\n"
                "- `/help` - Show all commands\n"
                "- `/help timeline` - Show detailed help for /timeline"
            ),
            positional_arg="command",
            positional_hint="command",
            category=CommandCategory.NAVIGATION,
        )
    )

    # === Inspection Commands ===
    registry.register(
        CommandDefinition(
            name="/status",
            description="Show experiment status",
            help_text=(
                "Display current experiment status including microscope connection,"
                " active embryos, and detector status."
            ),
            category=CommandCategory.INSPECTION,
        )
    )

    registry.register(
        CommandDefinition(
            name="/detectors",
            description="List all detectors",
            help_text=(
                "Show a table of all registered detectors with their status, type,"
                " and configuration."
            ),
            category=CommandCategory.INSPECTION,
        )
    )

    registry.register(
        CommandDefinition(
            name="/embryos",
            description="List embryos or show details",
            help_text=(
                "List all embryos in the current experiment. Provide an embryo ID to see"
                " detailed information about a specific embryo."
            ),
            positional_arg="embryo_id",
            positional_hint="ID",
            category=CommandCategory.INSPECTION,
        )
    )

    registry.register(
        CommandDefinition(
            name="/timelapse",
            description="Timelapse status [watch]",
            help_text=(
                "Display timelapse acquisition status for all embryos.\n\n"
                "Use 'watch' for live updating countdown view that refreshes every second."
            ),
            subcommands=[
                SubCommand(
                    name="watch",
                    description="Live countdown mode (Ctrl+C to exit)",
                ),
            ],
            category=CommandCategory.INSPECTION,
        )
    )

    registry.register(
        CommandDefinition(
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
        )
    )

    # === Session Commands ===
    registry.register(
        CommandDefinition(
            name="/sessions",
            description="Browse saved sessions",
            help_text="Open interactive session browser to view and select from saved sessions.",
            category=CommandCategory.SESSION,
        )
    )

    registry.register(
        CommandDefinition(
            name="/resume",
            description="Resume a session",
            help_text=(
                "Resume a previously saved session."
                " Opens interactive picker if no session ID is provided."
            ),
            positional_arg="session_id",
            positional_hint="ID",
            category=CommandCategory.SESSION,
        )
    )

    registry.register(
        CommandDefinition(
            name="/save",
            description="Save current session",
            help_text="Save the current session including embryo states and conversation history.",
            category=CommandCategory.SESSION,
        )
    )

    registry.register(
        CommandDefinition(
            name="/import-embryos",
            description="Import embryos from session",
            help_text="""Import embryo definitions from another session into the current session.

Use 'last' to import from the most recent session with embryos.""",
            positional_arg="session_id",
            positional_hint="ID|last",
            category=CommandCategory.SESSION,
        )
    )

    registry.register(
        CommandDefinition(
            name="/make-video",
            description="Create timelapse video",
            help_text="""Generate MP4 video from timelapse volumes in current session.

Creates max projection videos for each embryo. Optionally specify embryo ID to generate
video for a single embryo.

Options:
  --fps N     Frames per second (default: 10)
  --all       Include all embryos""",
            positional_arg="embryo_id",
            positional_hint="embryo_id",
            options=[
                CommandOption(
                    name="--fps",
                    description="Frames per second",
                    takes_value=True,
                    value_hint="N",
                ),
            ],
            category=CommandCategory.SESSION,
        )
    )

    # === Appearance Commands ===
    registry.register(
        CommandDefinition(
            name="/theme",
            description="Switch color theme",
            help_text=(
                "Change the CLI color theme.\n\n"
                "Available themes: vibrant, scientific, claude, monochrome"
            ),
            positional_arg="name",
            positional_hint="name",
            category=CommandCategory.APPEARANCE,
        )
    )

    registry.register(
        CommandDefinition(
            name="/history",
            description="Show conversation history",
            help_text="Display recent conversation history with the agent.",
            category=CommandCategory.APPEARANCE,
        )
    )

    registry.register(
        CommandDefinition(
            name="/tokens",
            description="Show API token usage",
            help_text="Display token usage statistics and estimated cost for the current session.",
            category=CommandCategory.APPEARANCE,
        )
    )

    # === Diagnostics Commands ===
    registry.register(
        CommandDefinition(
            name="/test-device",
            aliases=["/benchmark"],
            description="Test device layer pipeline (acquisition FPS benchmark)",
            help_text="""Run end-to-end volume acquisition benchmark.

Measures the full pipeline latency:
- Acquisition: HTTP → device layer → hardware → file written
- Storage: FileStore registration (projection + file-based storage)
- Viz push: Push to visualization server (if running)

Requires microscope connection and at least one registered embryo.""",
            options=[
                CommandOption(
                    name="--volumes",
                    short="-n",
                    description="Number of volumes to acquire",
                    takes_value=True,
                    value_hint="N",
                ),
                CommandOption(
                    name="--slices",
                    short="-s",
                    description="Slices per volume",
                    takes_value=True,
                    value_hint="N",
                ),
                CommandOption(
                    name="--warmup",
                    short="-w",
                    description="Warmup volumes (not timed)",
                    takes_value=True,
                    value_hint="N",
                ),
                CommandOption(
                    name="--save",
                    description="Save results to CSV",
                    is_flag=True,
                ),
            ],
            category=CommandCategory.DIAGNOSTICS,
        )
    )

    # === Planning Commands ===
    registry.register(
        CommandDefinition(
            name="/campaign",
            description="View or manage campaigns",
            help_text="""Browse and manage campaigns and experimental plans.

Usage:
  /campaign              List all campaigns with progress summary
  /campaign <id>         Show detailed view of a specific campaign
  /campaign delete <id>  Delete a campaign and all its plan items
  /campaign share <id>   Share a campaign on the mesh
  /campaign unshare <id> Stop sharing a campaign

Use plan mode (/plan) to create and modify campaigns.""",
            aliases=["/campaigns"],
            positional_arg="campaign_id",
            positional_hint="ID",
            subcommands=[
                SubCommand(
                    name="delete",
                    description="Delete a campaign and its plan items",
                ),
                SubCommand(
                    name="share",
                    description="Share a campaign on the mesh for coordination",
                ),
                SubCommand(
                    name="unshare",
                    description="Stop sharing a campaign on the mesh",
                ),
            ],
            category=CommandCategory.PLANNING,
        )
    )

    registry.register(
        CommandDefinition(
            name="/plan",
            description="Switch to plan mode for experimental design",
            help_text="""Enter plan mode to design experiments with the agent.

In plan mode, the agent acts as a scientific collaborator — helping
design campaigns, choose strains, set imaging parameters, plan controls,
and track progress across sessions.

Sub-commands:
  /plan          Enter plan mode (or show status if already in plan mode)
  /plan status   Show current plan progress
  /plan exit     Return to run mode""",
            subcommands=[
                SubCommand(
                    name="status",
                    description="Show current plan progress",
                ),
                SubCommand(
                    name="exit",
                    description="Return to run mode",
                ),
            ],
            category=CommandCategory.PLANNING,
        )
    )

    registry.register(
        CommandDefinition(
            name="/reset-context",
            description="Clear the context database (for testing)",
            help_text=(
                "Wipe all campaigns, learnings, session intents, and other context.\n"
                "The startup wizard will run again on next launch."
            ),
            category=CommandCategory.DIAGNOSTICS,
        )
    )

    registry.register(
        CommandDefinition(
            name="/wizard",
            description="Run the startup wizard",
            help_text=(
                "Re-run the onboarding wizard to set organism, campaign, and session intent.\n"
                "Useful after /reset-context or to change your current setup."
            ),
            category=CommandCategory.SESSION,
        )
    )

    registry.register(
        CommandDefinition(
            name="/peers",
            description="Show mesh peers on the network",
            help_text="""List all Gently instances discovered on the LAN.
Shows hostname, capabilities (GPU, SAM, microscope), and status for each peer.

Usage:
  /peers                        List all peers
  /peers <hostname> campaigns   Show shared campaigns on a peer""",
            aliases=["/mesh"],
            positional_arg="hostname",
            positional_hint="HOSTNAME",
            subcommands=[
                SubCommand(
                    name="campaigns",
                    description="Show shared campaigns on a peer",
                ),
            ],
            category=CommandCategory.INSPECTION,
        )
    )

    # === Mesh coordination commands ===
    registry.register(
        CommandDefinition(
            name="/join-campaign",
            description="Join a shared campaign on a peer",
            help_text="""Join a campaign shared by a mesh peer.

Usage:
  /join-campaign <hostname> <campaign_id>

After joining, use /claim to claim items for execution.""",
            positional_hint="HOSTNAME CAMPAIGN_ID",
            category=CommandCategory.PLANNING,
        )
    )

    registry.register(
        CommandDefinition(
            name="/claim",
            description="Claim a plan item from a shared campaign",
            help_text="""Claim a plan item from a joined remote campaign.

Usage:
  /claim <item_id>

Requires an active remote campaign (via /join-campaign).""",
            positional_hint="ITEM_ID",
            category=CommandCategory.PLANNING,
        )
    )

    registry.register(
        CommandDefinition(
            name="/pair",
            description="Pair with a mesh peer for secure communication",
            help_text="""Bluetooth-style pairing with mesh peers.

Usage:
  /pair <hostname>   Initiate pairing with a peer (shows PIN)
  /pair accept       Accept a pending pairing request
  /pair reject       Reject a pending pairing request
  /pair list         Show all trusted peers
  /pair unpair <id>  Remove trust for a peer (hostname or instance_id)
  /pair scopes       Show scopes for all peers
  /pair scopes <hostname> <scope1,scope2>  Set scopes for a peer""",
            positional_arg="target",
            positional_hint="HOSTNAME|accept|reject|list|unpair|scopes",
            subcommands=[
                SubCommand(name="accept", description="Accept a pending pairing request"),
                SubCommand(name="reject", description="Reject a pending pairing request"),
                SubCommand(name="list", description="Show all trusted peers"),
                SubCommand(name="unpair", description="Remove trust for a peer"),
                SubCommand(
                    name="scopes",
                    description="View or set permission scopes for a peer",
                ),
            ],
            category=CommandCategory.PLANNING,
        )
    )


# ============================================================================
# Global Registry
# ============================================================================

_command_registry: CommandRegistry | None = None


def get_command_registry() -> CommandRegistry:
    """Get or create the global command registry"""
    global _command_registry
    if _command_registry is None:
        _command_registry = CommandRegistry()
        _register_default_commands(_command_registry)
    return _command_registry
