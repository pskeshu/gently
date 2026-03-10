/**
 * Theme system with dark/light mode support.
 *
 * Each base theme (vibrant, scientific, claude, monochrome) has a dark
 * and light variant. Dark is the default. Switch via /theme command:
 *   /theme vibrant-light
 *   /theme claude
 *   /theme monochrome-light
 */

import type { ThemeColors } from "./types.js";

export const THEMES: Record<string, ThemeColors> = {
  // ── Vibrant (dark) ──────────────────────────────────────────
  vibrant: {
    name: "Vibrant",
    colorMode: "dark",
    primary: "#7C3AED",
    secondary: "#06B6D4",
    accent: "#F59E0B",
    user: "#10B981",
    agent: "#3B82F6",
    system: "#F59E0B",
    tool: "#EC4899",
    success: "#22C55E",
    warning: "#EAB308",
    error: "#EF4444",
    info: "#06B6D4",
    muted: "#9CA3AF",
    userMessageBg: "#374151",
    surfaceBg: "#1f2937",
  },

  "vibrant-light": {
    name: "Vibrant Light",
    colorMode: "light",
    primary: "#6D28D9",
    secondary: "#0891B2",
    accent: "#D97706",
    user: "#059669",
    agent: "#2563EB",
    system: "#D97706",
    tool: "#DB2777",
    success: "#16A34A",
    warning: "#CA8A04",
    error: "#DC2626",
    info: "#0891B2",
    muted: "#6B7280",
    userMessageBg: "#f1f5f9",
    surfaceBg: "#e2e8f0",
  },

  // ── Scientific (dark) ───────────────────────────────────────
  scientific: {
    name: "Scientific",
    colorMode: "dark",
    primary: "#1E3A5F",
    secondary: "#2E7D32",
    accent: "#FF8F00",
    user: "#2E7D32",
    agent: "#1565C0",
    system: "#6A1B9A",
    tool: "#00838F",
    success: "#2E7D32",
    warning: "#F57C00",
    error: "#C62828",
    info: "#0277BD",
    muted: "#607D8B",
    userMessageBg: "#37474f",
    surfaceBg: "#263238",
  },

  "scientific-light": {
    name: "Scientific Light",
    colorMode: "light",
    primary: "#1565C0",
    secondary: "#388E3C",
    accent: "#F57C00",
    user: "#2E7D32",
    agent: "#1565C0",
    system: "#7B1FA2",
    tool: "#00838F",
    success: "#388E3C",
    warning: "#EF6C00",
    error: "#C62828",
    info: "#0277BD",
    muted: "#78909C",
    userMessageBg: "#eceff1",
    surfaceBg: "#cfd8dc",
  },

  // ── Claude (dark) ───────────────────────────────────────────
  claude: {
    name: "Claude",
    colorMode: "dark",
    primary: "#D97706",
    secondary: "#1F2937",
    accent: "#D97706",
    user: "#059669",
    agent: "#D97706",
    system: "#6B7280",
    tool: "#7C3AED",
    success: "#10B981",
    warning: "#F59E0B",
    error: "#EF4444",
    info: "#3B82F6",
    muted: "#9CA3AF",
    userMessageBg: "#44403c",
    surfaceBg: "#292524",
  },

  "claude-light": {
    name: "Claude Light",
    colorMode: "light",
    primary: "#B45309",
    secondary: "#374151",
    accent: "#B45309",
    user: "#047857",
    agent: "#B45309",
    system: "#4B5563",
    tool: "#6D28D9",
    success: "#059669",
    warning: "#D97706",
    error: "#DC2626",
    info: "#2563EB",
    muted: "#6B7280",
    userMessageBg: "#fef3c7",
    surfaceBg: "#fffbeb",
  },

  // ── Monochrome (dark) ───────────────────────────────────────
  monochrome: {
    name: "Monochrome",
    colorMode: "dark",
    primary: "#FFFFFF",
    secondary: "#E5E7EB",
    accent: "#FFFFFF",
    user: "#22C55E",
    agent: "#FFFFFF",
    system: "#EAB308",
    tool: "#06B6D4",
    success: "#22C55E",
    warning: "#EAB308",
    error: "#EF4444",
    info: "#06B6D4",
    muted: "#6B7280",
    userMessageBg: "#374151",
    surfaceBg: "#1f2937",
  },

  "monochrome-light": {
    name: "Monochrome Light",
    colorMode: "light",
    primary: "#111827",
    secondary: "#374151",
    accent: "#111827",
    user: "#16A34A",
    agent: "#111827",
    system: "#CA8A04",
    tool: "#0891B2",
    success: "#16A34A",
    warning: "#CA8A04",
    error: "#DC2626",
    info: "#0891B2",
    muted: "#9CA3AF",
    userMessageBg: "#f3f4f6",
    surfaceBg: "#e5e7eb",
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
