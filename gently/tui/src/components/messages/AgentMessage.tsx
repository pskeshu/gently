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

interface Props {
  entry: ChatEntry;
  theme: ThemeColors;
}

function AgentMessageImpl({ entry, theme }: Props) {
  const text = entry.text.replace(/^\n+/, "");
  return (
    <Box flexDirection="row" marginBottom={1}>
      <Text color={theme.agent}>{"✦ "}</Text>
      <Box flexDirection="column" flexGrow={1}>
        <MarkdownText theme={theme}>{text}</MarkdownText>
      </Box>
    </Box>
  );
}

export const AgentMessage = memo(AgentMessageImpl);
