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
  const dur = entry.toolDuration;
  const showDuration = dur && dur > 0.1;

  if (entry.isStreaming) {
    return (
      <Box marginBottom={0} paddingLeft={2}>
        <Text>
          <Text color={theme.tool}>{"⠸ "}</Text>
          <Text color={theme.tool} dimColor>
            {name}
          </Text>
          {summary ? (
            <Text color={theme.muted} dimColor>{" — "}{summary}</Text>
          ) : null}
        </Text>
      </Box>
    );
  }

  return (
    <Box marginBottom={0} paddingLeft={2}>
      <Text>
        <Text color={theme.success}>{"● "}</Text>
        <Text color={theme.muted}>{name}</Text>
        {summary ? <Text dimColor>{" — "}{summary}</Text> : null}
        {showDuration ? <Text dimColor>{` (${dur.toFixed(1)}s)`}</Text> : null}
      </Text>
    </Box>
  );
}

export const ToolMessage = memo(ToolMessageImpl);
