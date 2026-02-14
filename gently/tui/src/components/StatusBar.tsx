/**
 * Persistent bottom status bar — always visible below the input.
 *
 * Layout:
 *   ─────────────────────────────────────────────────────────────
 *   live · ● Device connected · 3 embryos · 12.4k tokens      gently v0.4.0
 *
 * Left side: mode badge, device status, embryos, token usage.
 * Right side: version (right-aligned).
 *
 * Notifications overlay temporarily when they fire, then fade back
 * to the persistent status line.
 */

import React, { useEffect, useState } from "react";
import { Box, Text } from "ink";
import type { ThemeColors, TokenSnapshot } from "../types.js";

interface StatusBarProps {
  theme: ThemeColors;
  version: string;
  sessionId: string;
  deviceConnected: boolean;
  offline: boolean;
  embryoCount: number;
  tokens: TokenSnapshot;
  notification: { level: string; title: string; body?: string } | null;
  onClearNotification: () => void;
  wizardActive?: boolean;
  copilotMode?: string;
}

function formatTokens(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}k`;
  return String(n);
}

function ModeBadge({ mode, theme }: { mode: string; theme: ThemeColors }) {
  if (mode === "plan") {
    return <Text color={theme.info} bold>plan</Text>;
  }
  return <Text color={theme.muted}>run</Text>;
}

export function StatusBar({
  theme,
  version,
  sessionId,
  deviceConnected,
  offline,
  embryoCount,
  tokens,
  notification,
  onClearNotification,
  wizardActive,
  copilotMode = "run",
}: StatusBarProps) {
  // Auto-dismiss notifications after 5 seconds
  const [showNotification, setShowNotification] = useState(false);

  useEffect(() => {
    if (!notification) {
      setShowNotification(false);
      return;
    }
    setShowNotification(true);
    const timer = setTimeout(() => {
      setShowNotification(false);
      onClearNotification();
    }, 5000);
    return () => clearTimeout(timer);
  }, [notification, onClearNotification]);

  const sep = <Text color={theme.muted}> · </Text>;

  // Notification overlay
  if (showNotification && notification) {
    const levelColor =
      notification.level === "error"
        ? theme.error
        : notification.level === "warning"
          ? theme.warning
          : notification.level === "success"
            ? theme.success
            : theme.info;

    return (
      <Box justifyContent="space-between">
        <Box>
          <ModeBadge mode={copilotMode} theme={theme} />
          {sep}
          <Text color={levelColor} bold>
            {notification.title}
          </Text>
          {notification.body ? (
            <Text color={theme.muted}> — {notification.body}</Text>
          ) : null}
        </Box>
        {version ? (
          <Text color={theme.muted}>gently v{version}</Text>
        ) : null}
      </Box>
    );
  }

  // Wizard active indicator
  if (wizardActive) {
    return (
      <Box justifyContent="space-between">
        <Box>
          <ModeBadge mode={copilotMode} theme={theme} />
          {sep}
          <Text color={theme.info} bold>setting up</Text>
          <Text color={theme.muted}> — answer a few questions to get started</Text>
        </Box>
        {version ? (
          <Text color={theme.muted}>gently v{version}</Text>
        ) : null}
      </Box>
    );
  }

  // Device status indicator
  let deviceDot: { char: string; color: string; label: string };
  if (offline) {
    deviceDot = { char: "○", color: theme.warning, label: "offline" };
  } else if (deviceConnected) {
    deviceDot = { char: "●", color: theme.success, label: "connected" };
  } else {
    deviceDot = { char: "●", color: theme.error, label: "disconnected" };
  }

  return (
    <Box justifyContent="space-between">
      {/* Left: mode, device status, session, embryos, tokens */}
      <Box>
        <ModeBadge mode={copilotMode} theme={theme} />
        {sep}
        <Text color={deviceDot.color}>{deviceDot.char}</Text>
        <Text color={theme.muted}> {deviceDot.label}</Text>

        {sessionId ? (
          <>
            {sep}
            <Text color={theme.muted}>{sessionId.slice(0, 8)}</Text>
          </>
        ) : null}

        {embryoCount > 0 ? (
          <>
            {sep}
            <Text color={theme.muted}>
              {embryoCount} embryo{embryoCount !== 1 ? "s" : ""}
            </Text>
          </>
        ) : null}

        {tokens.total_tokens > 0 ? (
          <>
            {sep}
            <Text color={theme.muted}>
              {formatTokens(tokens.total_tokens)} tokens
            </Text>
          </>
        ) : null}
      </Box>

      {/* Right: version */}
      {version ? (
        <Text color={theme.muted}>gently v{version}</Text>
      ) : null}
    </Box>
  );
}
