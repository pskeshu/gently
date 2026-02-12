/**
 * Active streaming area — renders only the currently-streaming message.
 *
 * Completed messages are handled by <Static> in App.tsx and scroll up
 * naturally. This component only shows the live/active content so it
 * can re-render on every text chunk without touching history.
 *
 * When isThinking is true (message sent, no response yet), shows a
 * spinner to indicate the copilot is processing.
 */

import React from "react";
import { Box, Text } from "ink";
import Spinner from "ink-spinner";
import { MessageBubble } from "./MessageBubble.js";
import type { ChatEntry, ThemeColors } from "../types.js";

interface ActiveMessageProps {
  entry: ChatEntry | null;
  theme: ThemeColors;
}

export function ActiveMessage({ entry, theme }: ActiveMessageProps) {
  if (!entry) return null;

  // Thinking state — waiting for first response chunk
  if (entry.isThinking) {
    return (
      <Box marginBottom={1}>
        <Text color={theme.copilot}>
          <Spinner type="dots" />
        </Text>
        <Text color={theme.muted}> Thinking...</Text>
      </Box>
    );
  }

  return (
    <Box flexDirection="column">
      <MessageBubble entry={entry} theme={theme} />
    </Box>
  );
}
