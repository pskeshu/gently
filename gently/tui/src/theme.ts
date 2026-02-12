/**
 * Theme system — port of Python theme.py
 *
 * Four themes matching the Python CLI so switching between
 * the Rich CLI and the Ink TUI feels consistent.
 */

import type { ThemeColors } from "./types.js";

export const THEMES: Record<string, ThemeColors> = {
  vibrant: {
    name: "Vibrant",
    primary: "#7C3AED",
    secondary: "#06B6D4",
    accent: "#F59E0B",
    user: "#10B981",
    copilot: "#3B82F6",
    system: "#F59E0B",
    tool: "#EC4899",
    success: "#22C55E",
    warning: "#EAB308",
    error: "#EF4444",
    info: "#06B6D4",
    muted: "#9CA3AF",
  },

  scientific: {
    name: "Scientific",
    primary: "#1E3A5F",
    secondary: "#2E7D32",
    accent: "#FF8F00",
    user: "#2E7D32",
    copilot: "#1565C0",
    system: "#6A1B9A",
    tool: "#00838F",
    success: "#2E7D32",
    warning: "#F57C00",
    error: "#C62828",
    info: "#0277BD",
    muted: "#607D8B",
  },

  claude: {
    name: "Claude",
    primary: "#D97706",
    secondary: "#1F2937",
    accent: "#D97706",
    user: "#059669",
    copilot: "#D97706",
    system: "#6B7280",
    tool: "#7C3AED",
    success: "#10B981",
    warning: "#F59E0B",
    error: "#EF4444",
    info: "#3B82F6",
    muted: "#9CA3AF",
  },

  monochrome: {
    name: "Monochrome",
    primary: "#FFFFFF",
    secondary: "#E5E7EB",
    accent: "#FFFFFF",
    user: "#22C55E",
    copilot: "#FFFFFF",
    system: "#EAB308",
    tool: "#06B6D4",
    success: "#22C55E",
    warning: "#EAB308",
    error: "#EF4444",
    info: "#06B6D4",
    muted: "#6B7280",
  },
};

let currentTheme: ThemeColors = THEMES.vibrant!;

export function getTheme(): ThemeColors {
  return currentTheme;
}

export function setTheme(name: string): void {
  const theme = THEMES[name];
  if (!theme) {
    const available = Object.keys(THEMES).join(", ");
    throw new Error(`Unknown theme: '${name}'. Available: ${available}`);
  }
  currentTheme = theme;
}

export function listThemes(): Record<string, ThemeColors> {
  return { ...THEMES };
}
