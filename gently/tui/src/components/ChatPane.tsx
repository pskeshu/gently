/**
 * Active streaming area — renders only the currently-streaming message.
 *
 * Completed messages are handled by <Static> in App.tsx and scroll up
 * naturally. This component only shows the live/active content so it
 * can re-render on every text chunk without touching history.
 *
 * When isThinking is true (message sent, no response yet), shows a
 * spinner with a live elapsed timer and session token stats. During
 * streaming, shows elapsed time and approximate output tokens.
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
  streamStartedAt?: number;
  streamCharsReceived?: number;
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

export function ActiveMessage({ entry, theme, tokens, streamStartedAt, streamCharsReceived }: ActiveMessageProps) {
  const [elapsed, setElapsed] = useState(0);

  // Live timer — ticks every second while streaming or thinking
  useEffect(() => {
    if (!streamStartedAt) {
      setElapsed(0);
      return;
    }
    setElapsed(Date.now() - streamStartedAt);
    const interval = setInterval(() => {
      setElapsed(Date.now() - streamStartedAt);
    }, 1000);
    return () => clearInterval(interval);
  }, [streamStartedAt]);

  if (!entry) return null;

  // Thinking state — spinner with elapsed timer and session stats
  if (entry.isThinking) {
    const apiCalls = tokens?.api_calls ?? 0;
    const totalTokens = tokens?.total_tokens ?? 0;

    // Build stats suffix: show session totals if we have them, otherwise
    // just show "calling API..." so the user knows something is happening.
    let stats = "";
    if (apiCalls > 0) {
      stats = ` · ${formatTokens(totalTokens)} tokens · ${apiCalls} API call${apiCalls !== 1 ? "s" : ""}`;
    } else if (elapsed >= 2000) {
      stats = " · calling API...";
    }

    return (
      <Box marginBottom={1}>
        <Text color={theme.copilot}>
          <Spinner type="dots" />
        </Text>
        <Text color={theme.muted}>
          {" "}Thinking{elapsed >= 1000 ? ` ${formatElapsed(elapsed)}` : ""}
          {stats}
        </Text>
      </Box>
    );
  }

  // Streaming text — show message with elapsed time and approx output tokens
  const approxTokens = Math.round((streamCharsReceived ?? 0) / 4);

  return (
    <Box flexDirection="column">
      <MessageBubble entry={entry} theme={theme} />
      {entry.isStreaming && elapsed >= 1000 ? (
        <Box>
          <Text color={theme.muted}>
            {"  "}{formatElapsed(elapsed)}
            {approxTokens > 0 ? ` · ~${approxTokens} output tokens` : ""}
          </Text>
        </Box>
      ) : null}
    </Box>
  );
}
