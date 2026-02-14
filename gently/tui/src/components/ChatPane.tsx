/**
 * Active streaming area — renders only the currently-streaming message.
 *
 * Completed messages are handled by <Static> in App.tsx and scroll up
 * naturally. This component only shows the live/active content so it
 * can re-render on every text chunk without touching history.
 *
 * When isThinking is true (message sent, no response yet), shows a
 * spinner with a live elapsed timer. During streaming, shows elapsed
 * time and approximate output tokens alongside the text.
 */

import React, { useEffect, useState } from "react";
import { Box, Text } from "ink";
import Spinner from "ink-spinner";
import { MessageBubble } from "./MessageBubble.js";
import type { ChatEntry, ThemeColors } from "../types.js";

interface ActiveMessageProps {
  entry: ChatEntry | null;
  theme: ThemeColors;
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

export function ActiveMessage({ entry, theme, streamStartedAt, streamCharsReceived }: ActiveMessageProps) {
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

  // Thinking state — spinner with elapsed timer
  if (entry.isThinking) {
    return (
      <Box marginBottom={1}>
        <Text color={theme.copilot}>
          <Spinner type="dots" />
        </Text>
        <Text color={theme.muted}>
          {" "}Thinking{elapsed >= 1000 ? ` ${formatElapsed(elapsed)}` : ""}
        </Text>
      </Box>
    );
  }

  // Streaming text — show message with elapsed time and approx tokens
  const approxTokens = Math.round((streamCharsReceived ?? 0) / 4);

  return (
    <Box flexDirection="column">
      <MessageBubble entry={entry} theme={theme} />
      {entry.isStreaming && elapsed >= 1000 ? (
        <Box>
          <Text color={theme.muted}>
            {"  "}{formatElapsed(elapsed)}
            {approxTokens > 0 ? ` · ~${approxTokens} tokens` : ""}
          </Text>
        </Box>
      ) : null}
    </Box>
  );
}
