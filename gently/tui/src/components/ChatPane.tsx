/**
 * Active streaming area — renders only the currently-streaming message.
 *
 * Completed messages are handled by <Static> in App.tsx and scroll up
 * naturally. This component only shows the live/active content so it
 * can re-render on every text chunk without touching history.
 *
 * When isThinking is true (message sent, no response yet), shows a
 * spinner with a live elapsed timer (like Claude Code).
 */

import React, { useEffect, useState } from "react";
import { Box, Text } from "ink";
import Spinner from "ink-spinner";
import { MessageBubble } from "./MessageBubble.js";
import type { ChatEntry, ThemeColors, TokenSnapshot } from "../types.js";

interface ActiveMessageProps {
  entry: ChatEntry | null;
  theme: ThemeColors;
  tokens?: TokenSnapshot;
}

function formatElapsed(ms: number): string {
  const s = Math.floor(ms / 1000);
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  const rem = s % 60;
  return `${m}m ${rem}s`;
}

function formatTokens(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}k`;
  return String(n);
}

export function ActiveMessage({ entry, theme, tokens }: ActiveMessageProps) {
  const [elapsed, setElapsed] = useState(0);

  // Live timer — ticks every second while thinking
  useEffect(() => {
    if (!entry?.isThinking) {
      setElapsed(0);
      return;
    }
    const start = entry.timestamp;
    setElapsed(Date.now() - start);
    const interval = setInterval(() => {
      setElapsed(Date.now() - start);
    }, 1000);
    return () => clearInterval(interval);
  }, [entry?.isThinking, entry?.timestamp]);

  if (!entry) return null;

  // Thinking state — spinner with elapsed timer and token activity
  if (entry.isThinking) {
    const totalTokens = tokens?.total_tokens ?? 0;

    return (
      <Box marginBottom={1}>
        <Text color={theme.copilot}>
          <Spinner type="dots" />
        </Text>
        <Text color={theme.muted}>
          {" "}Thinking{elapsed >= 1000 ? ` ${formatElapsed(elapsed)}` : ""}
        </Text>
        {totalTokens > 0 ? (
          <Text color={theme.muted}> · {formatTokens(totalTokens)} tokens</Text>
        ) : null}
      </Box>
    );
  }

  return (
    <Box flexDirection="column">
      <MessageBubble entry={entry} theme={theme} />
    </Box>
  );
}
