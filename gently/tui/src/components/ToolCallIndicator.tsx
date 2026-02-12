/**
 * Tool call indicator — spinner while running, checkmark when done.
 */

import React from "react";
import { Box, Text } from "ink";
import Spinner from "ink-spinner";
import type { ThemeColors } from "../types.js";

interface ToolCallIndicatorProps {
  toolName: string;
  isRunning: boolean;
  duration?: number;
  theme: ThemeColors;
}

export function ToolCallIndicator({
  toolName,
  isRunning,
  duration,
  theme,
}: ToolCallIndicatorProps) {
  return (
    <Box gap={1}>
      {isRunning ? (
        <>
          <Text color={theme.tool}>
            <Spinner type="dots" />
          </Text>
          <Text color={theme.tool}>Running {toolName}...</Text>
        </>
      ) : (
        <>
          <Text color={theme.success}>✓</Text>
          <Text color={theme.tool}>{toolName}</Text>
          {duration !== undefined ? (
            <Text color={theme.muted}>({duration.toFixed(2)}s)</Text>
          ) : null}
        </>
      )}
    </Box>
  );
}
