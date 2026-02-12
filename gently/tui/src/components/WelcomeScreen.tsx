/**
 * Welcome screen shown on first connect before any messages.
 * Disappears once the user sends their first message.
 */

import React from "react";
import { Box, Text } from "ink";
import type { ThemeColors } from "../types.js";

interface WelcomeScreenProps {
  theme: ThemeColors;
  version: string;
  sessionId: string;
  embryoCount: number;
}

export function WelcomeScreen({
  theme,
  version,
  sessionId,
  embryoCount,
}: WelcomeScreenProps) {
  const shortSession = sessionId.slice(0, 8);

  return (
    <Box flexDirection="column" paddingLeft={2} paddingTop={1} paddingBottom={1}>
      {/* Branding */}
      <Box>
        <Text bold color={theme.primary}>
          {"✦ gently"}
        </Text>
        <Text color={theme.muted}> v{version}</Text>
      </Box>

      <Box height={1} />

      {/* Session info */}
      <Box>
        <Text color={theme.muted}>
          Session {shortSession} · {embryoCount} embryo
          {embryoCount !== 1 ? "s" : ""}
        </Text>
      </Box>

      <Box height={1} />

      {/* Quick-start hints */}
      <Box flexDirection="column">
        <Text color={theme.muted}>Quick start:</Text>
        <Box paddingLeft={2} flexDirection="column">
          <Text color={theme.muted}>
            Type a message to chat with the copilot
          </Text>
          <Text color={theme.muted}>
            {"Use "}
            <Text color={theme.accent}>/help</Text>
            {" for available commands"}
          </Text>
          <Text color={theme.muted}>
            {"Press "}
            <Text color={theme.accent}>Esc</Text>
            {" to cancel a running response"}
          </Text>
        </Box>
      </Box>
    </Box>
  );
}
