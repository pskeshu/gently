/**
 * Agent message — markdown-rendered text with a streaming cursor when
 * `entry.isStreaming` is true. The streaming variant intentionally is
 * NOT memoized via React.memo's default props comparison because
 * `entry` is mutated (text grows). It's only used inside <ActiveMessage>
 * during streaming; completed agent entries pass `isStreaming=false`
 * and benefit from memoization.
 */

import React, { memo } from "react";
import { Box, Text } from "ink";
import { MarkdownText } from "../MarkdownText.js";
import type { ChatEntry, ThemeColors } from "../../types.js";
import { formatTime } from "./time.js";

interface Props {
  entry: ChatEntry;
  theme: ThemeColors;
}

function AgentMessageImpl({ entry, theme }: Props) {
  return (
    <Box flexDirection="column" marginBottom={1}>
      <Box>
        <Text bold color={theme.agent}>
          {"✦ Gently"}
        </Text>
        <Text color={theme.muted}> {formatTime(entry.timestamp)}</Text>
        {entry.isStreaming ? <Text color={theme.agent}> ▍</Text> : null}
      </Box>
      <Box paddingLeft={2}>
        <MarkdownText theme={theme}>{entry.text}</MarkdownText>
      </Box>
    </Box>
  );
}

export const AgentMessage = memo(AgentMessageImpl);
