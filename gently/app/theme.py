"""
Theme system for Microscopy Agent

Provides multiple color themes with dark/light variants
and runtime switching support via /theme command.
"""

from dataclasses import dataclass


@dataclass
class Theme:
    """Theme definition with colors and text indicators."""

    name: str
    color_mode: str  # "dark" or "light"

    # Primary colors
    primary: str
    secondary: str
    accent: str

    # Semantic colors for different message types
    user: str
    agent: str
    system: str
    tool: str

    # Status colors
    success: str
    warning: str
    error: str
    info: str
    muted: str

    # Text indicators
    icon_success: str = "+"
    icon_error: str = "x"
    icon_warning: str = "!"
    icon_info: str = ">"
    icon_user: str = "You"
    icon_agent: str = "Agent"
    icon_tool: str = "Tool"
    icon_system: str = "System"


THEMES: dict[str, Theme] = {
    "vibrant": Theme(
        name="Vibrant",
        color_mode="dark",
        primary="#7C3AED",
        secondary="#06B6D4",
        accent="#F59E0B",
        user="#10B981",
        agent="#3B82F6",
        system="#F59E0B",
        tool="#EC4899",
        success="#22C55E",
        warning="#EAB308",
        error="#EF4444",
        info="#06B6D4",
        muted="#9CA3AF",
    ),
    "vibrant-light": Theme(
        name="Vibrant Light",
        color_mode="light",
        primary="#6D28D9",
        secondary="#0891B2",
        accent="#D97706",
        user="#059669",
        agent="#2563EB",
        system="#D97706",
        tool="#DB2777",
        success="#16A34A",
        warning="#CA8A04",
        error="#DC2626",
        info="#0891B2",
        muted="#6B7280",
    ),
    "scientific": Theme(
        name="Scientific",
        color_mode="dark",
        primary="#1E3A5F",
        secondary="#2E7D32",
        accent="#FF8F00",
        user="#2E7D32",
        agent="#1565C0",
        system="#6A1B9A",
        tool="#00838F",
        success="#2E7D32",
        warning="#F57C00",
        error="#C62828",
        info="#0277BD",
        muted="#607D8B",
    ),
    "scientific-light": Theme(
        name="Scientific Light",
        color_mode="light",
        primary="#1565C0",
        secondary="#388E3C",
        accent="#F57C00",
        user="#2E7D32",
        agent="#1565C0",
        system="#7B1FA2",
        tool="#00838F",
        success="#388E3C",
        warning="#EF6C00",
        error="#C62828",
        info="#0277BD",
        muted="#78909C",
    ),
    "claude": Theme(
        name="Claude",
        color_mode="dark",
        primary="#D97706",
        secondary="#1F2937",
        accent="#D97706",
        user="#059669",
        agent="#D97706",
        system="#6B7280",
        tool="#7C3AED",
        success="#10B981",
        warning="#F59E0B",
        error="#EF4444",
        info="#3B82F6",
        muted="#9CA3AF",
    ),
    "claude-light": Theme(
        name="Claude Light",
        color_mode="light",
        primary="#B45309",
        secondary="#374151",
        accent="#B45309",
        user="#047857",
        agent="#B45309",
        system="#4B5563",
        tool="#6D28D9",
        success="#059669",
        warning="#D97706",
        error="#DC2626",
        info="#2563EB",
        muted="#6B7280",
    ),
    "monochrome": Theme(
        name="Monochrome",
        color_mode="dark",
        primary="#FFFFFF",
        secondary="#E5E7EB",
        accent="#FFFFFF",
        user="#22C55E",
        agent="#FFFFFF",
        system="#EAB308",
        tool="#06B6D4",
        success="#22C55E",
        warning="#EAB308",
        error="#EF4444",
        info="#06B6D4",
        muted="#6B7280",
    ),
    "monochrome-light": Theme(
        name="Monochrome Light",
        color_mode="light",
        primary="#111827",
        secondary="#374151",
        accent="#111827",
        user="#16A34A",
        agent="#111827",
        system="#CA8A04",
        tool="#0891B2",
        success="#16A34A",
        warning="#CA8A04",
        error="#DC2626",
        info="#0891B2",
        muted="#9CA3AF",
    ),
}

_current_theme: Theme = THEMES["vibrant"]


def get_theme() -> Theme:
    """Get the current active theme."""
    return _current_theme


def set_theme(name: str) -> None:
    """Set the active theme by name."""
    global _current_theme
    if name in THEMES:
        _current_theme = THEMES[name]
    else:
        available = ", ".join(THEMES.keys())
        raise ValueError(f"Unknown theme: '{name}'. Available: {available}")


def list_themes() -> dict[str, Theme]:
    """Get all available themes."""
    return THEMES.copy()
