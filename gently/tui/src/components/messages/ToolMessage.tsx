/**
 * Tool call message — spinner glyph while running, then a completed
 * bullet with optional duration.
 */

import React, { memo } from "react";
import { Box, Text } from "ink";
import type { ChatEntry, ThemeColors } from "../../types.js";

interface Props {
  entry: ChatEntry;
  theme: ThemeColors;
}

function ToolMessageImpl({ entry, theme }: Props) {
  const name = entry.toolName ?? entry.text;
  const summary = entry.toolSummary;
  const answer = entry.toolAnswer;

  if (entry.isStreaming) {
    return (
      <Box flexDirection="column" marginBottom={0} paddingLeft={2}>
        <Text>
          <Text color={theme.tool}>{"⠸ "}</Text>
          <Text color={theme.tool} dimColor>
            {name}
          </Text>
          {summary ? (
            <Text color={theme.muted} dimColor>{" — "}{summary}</Text>
          ) : null}
        </Text>
        {answer ? (
          <Box paddingLeft={2}>
            <Text color={theme.muted} dimColor>{"↳ "}{answer}</Text>
          </Box>
        ) : null}
      </Box>
    );
  }

  return (
    <Box flexDirection="column" marginBottom={0} paddingLeft={2}>
      <Text>
        <Text color={theme.success}>{"● "}</Text>
        <Text color={theme.muted}>{name}</Text>
        {summary ? <Text dimColor>{" — "}{summary}</Text> : null}
      </Text>
      {answer ? (
        <Box paddingLeft={2}>
          <Text color={theme.muted} dimColor>{"↳ "}{answer}</Text>
        </Box>
      ) : null}
    </Box>
  );
}

export const ToolMessage = memo(ToolMessageImpl);
