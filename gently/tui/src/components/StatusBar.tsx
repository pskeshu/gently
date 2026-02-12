/**
 * Persistent bottom status bar — always visible below the input.
 *
 * Layout (like Claude Code):
 *   ─────────────────────────────────────────────────────
 *   gently v0.4.0 · session abc123 · ● Connected · 3 embryos · 12.4k tokens
 *
 * Notifications overlay temporarily when they fire, then fade back
 * to the persistent status line.
 */

import React, { useEffect, useState } from "react";
import { Box, Text } from "ink";
import type { ConnectionStatus, ThemeColors, TokenSnapshot } from "../types.js";

interface StatusBarProps {
  theme: ThemeColors;
  version: string;
  sessionId: string;
  connectionStatus: ConnectionStatus;
  embryoCount: number;
  tokens: TokenSnapshot;
  notification: { level: string; title: string; body?: string } | null;
  onClearNotification: () => void;
}

function formatTokens(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}k`;
  return String(n);
}

export function StatusBar({
  theme,
  version,
  sessionId,
  connectionStatus,
  embryoCount,
  tokens,
  notification,
  onClearNotification,
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

  // Connection indicator
  const connDot =
    connectionStatus === "connected"
      ? { char: "●", color: theme.success, label: "Connected" }
      : connectionStatus === "connecting"
        ? { char: "○", color: theme.warning, label: "Connecting" }
        : { char: "●", color: theme.error, label: "Disconnected" };

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
      <Box>
        <Text color={levelColor} bold>
          {notification.title}
        </Text>
        {notification.body ? (
          <Text color={theme.muted}> — {notification.body}</Text>
        ) : null}
      </Box>
    );
  }

  // Persistent status line
  const sep = <Text color={theme.muted}> · </Text>;

  return (
    <Box>
      {version ? (
        <>
          <Text color={theme.muted}>gently v{version}</Text>
          {sep}
        </>
      ) : null}

      {sessionId ? (
        <>
          <Text color={theme.muted}>{sessionId.slice(0, 8)}</Text>
          {sep}
        </>
      ) : null}

      <Text color={connDot.color}>{connDot.char}</Text>
      <Text color={theme.muted}> {connDot.label}</Text>

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

      {tokens.api_calls > 0 ? (
        <>
          {sep}
          <Text color={theme.muted}>
            {tokens.api_calls} call{tokens.api_calls !== 1 ? "s" : ""}
          </Text>
        </>
      ) : null}
    </Box>
  );
}
