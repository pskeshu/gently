/**
 * System / command-result message — markdown-rendered, indented.
 */

import React, { memo } from "react";
import { Box } from "ink";
import { MarkdownText } from "../MarkdownText.js";
import type { ChatEntry, ThemeColors } from "../../types.js";

interface Props {
  entry: ChatEntry;
  theme: ThemeColors;
}

function SystemMessageImpl({ entry, theme }: Props) {
  return (
    <Box marginBottom={1} paddingLeft={2}>
      <MarkdownText theme={theme}>{entry.text}</MarkdownText>
    </Box>
  );
}

export const SystemMessage = memo(SystemMessageImpl);
