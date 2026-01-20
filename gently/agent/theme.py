"""
Theme system for Microscopy Copilot CLI

Provides multiple color themes with runtime switching support.
"""

from dataclasses import dataclass
from typing import Dict


@dataclass
class Theme:
    """Theme definition with colors and text indicators"""
    name: str

    # Primary colors
    primary: str
    secondary: str
    accent: str

    # Semantic colors for different message types
    user: str
    copilot: str
    system: str
    tool: str

    # Status colors
    success: str
    warning: str
    error: str
    info: str
    muted: str

    # Text indicators (minimal, no emojis)
    icon_success: str = "+"
    icon_error: str = "x"
    icon_warning: str = "!"
    icon_info: str = ">"
    icon_user: str = "You"
    icon_copilot: str = "Copilot"
    icon_tool: str = "Tool"
    icon_system: str = "System"


# Theme definitions
THEMES: Dict[str, Theme] = {
    "vibrant": Theme(
        name="Vibrant",
        primary="#7C3AED",      # Vivid purple
        secondary="#06B6D4",    # Cyan
        accent="#F59E0B",       # Amber
        user="#10B981",         # Emerald green
        copilot="#3B82F6",      # Bright blue
        system="#F59E0B",       # Amber
        tool="#EC4899",         # Pink/magenta
        success="#22C55E",      # Green
        warning="#EAB308",      # Yellow
        error="#EF4444",        # Red
        info="#06B6D4",         # Cyan
        muted="dim #9CA3AF",    # Gray
    ),

    "scientific": Theme(
        name="Scientific",
        primary="#1E3A5F",      # Navy blue
        secondary="#2E7D32",    # Forest green
        accent="#FF8F00",       # Amber
        user="#2E7D32",         # Forest green
        copilot="#1565C0",      # Blue
        system="#6A1B9A",       # Purple
        tool="#00838F",         # Teal
        success="#2E7D32",      # Forest green
        warning="#F57C00",      # Orange
        error="#C62828",        # Dark red
        info="#0277BD",         # Light blue
        muted="dim #607D8B",    # Blue gray
    ),

    "claude": Theme(
        name="Claude",
        primary="#D97706",      # Claude orange/amber
        secondary="#1F2937",    # Dark gray
        accent="#D97706",       # Amber
        user="#059669",         # Teal green
        copilot="#D97706",      # Claude amber
        system="#6B7280",       # Gray
        tool="#7C3AED",         # Purple
        success="#10B981",      # Emerald
        warning="#F59E0B",      # Amber
        error="#EF4444",        # Red
        info="#3B82F6",         # Blue
        muted="dim #9CA3AF",    # Gray
    ),

    "monochrome": Theme(
        name="Monochrome",
        primary="white",
        secondary="bright_white",
        accent="bold white",
        user="green",
        copilot="white",
        system="yellow",
        tool="cyan",
        success="green",
        warning="yellow",
        error="red",
        info="cyan",
        muted="dim",
    ),
}


# Current theme (module-level state)
_current_theme: Theme = THEMES["vibrant"]


def get_theme() -> Theme:
    """Get the current active theme"""
    return _current_theme


def set_theme(name: str) -> None:
    """
    Set the active theme by name

    Parameters
    ----------
    name : str
        Theme name (vibrant, scientific, claude, monochrome)

    Raises
    ------
    ValueError
        If theme name is not found
    """
    global _current_theme
    if name in THEMES:
        _current_theme = THEMES[name]
    else:
        available = ", ".join(THEMES.keys())
        raise ValueError(f"Unknown theme: '{name}'. Available themes: {available}")


def list_themes() -> Dict[str, Theme]:
    """Get all available themes"""
    return THEMES.copy()
