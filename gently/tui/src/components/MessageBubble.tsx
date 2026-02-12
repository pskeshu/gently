/**
 * Single chat message — user, copilot, system, or tool.
 *
 * User messages get a distinctive left-border highlight and bold
 * treatment so they stand out from copilot responses — similar
 * to how Claude Code renders user vs assistant messages.
 */

import React from "react";
import { Box, Text } from "ink";
import type { ChatEntry, ThemeColors } from "../types.js";
import { MarkdownText } from "./MarkdownText.js";

interface MessageBubbleProps {
  entry: ChatEntry;
  theme: ThemeColors;
}

export function MessageBubble({ entry, theme }: MessageBubbleProps) {
  const time = new Date(entry.timestamp).toLocaleTimeString("en-US", {
    hour12: false,
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });

  switch (entry.role) {
    case "user":
      return (
        <Box flexDirection="column" marginBottom={1}>
          <Box>
            <Text bold color={theme.user}>
              {"❯ You"}
            </Text>
            <Text color={theme.muted}> {time}</Text>
          </Box>
          <Box
            borderStyle="single"
            borderLeft
            borderRight={false}
            borderTop={false}
            borderBottom={false}
            borderColor={theme.user}
            paddingLeft={1}
          >
            <Text bold>{entry.text}</Text>
          </Box>
        </Box>
      );

    case "copilot":
      return (
        <Box flexDirection="column" marginBottom={1}>
          <Box>
            <Text bold color={theme.copilot}>
              {"✦ Copilot"}
            </Text>
            <Text color={theme.muted}> {time}</Text>
            {entry.isStreaming ? (
              <Text color={theme.copilot}> ▍</Text>
            ) : null}
          </Box>
          <Box paddingLeft={2}>
            <MarkdownText theme={theme}>{entry.text}</MarkdownText>
          </Box>
        </Box>
      );

    case "tool":
      return (
        <Box marginBottom={0} paddingLeft={2}>
          {entry.isStreaming ? (
            <Text>
              <Text color={theme.tool}>{"⠸ "}</Text>
              <Text color={theme.tool} dimColor>
                {entry.text}
              </Text>
            </Text>
          ) : (
            <Text>
              <Text color={theme.success}>{"✓ "}</Text>
              <Text color={theme.muted}>{entry.text}</Text>
            </Text>
          )}
        </Box>
      );

    case "system":
      return (
        <Box marginBottom={1} paddingLeft={2}>
          <MarkdownText theme={theme}>{entry.text}</MarkdownText>
        </Box>
      );

    default:
      return <Text>{entry.text}</Text>;
  }
}
