/**
 * User message — left-border highlight + bold treatment so it stands
 * out from agent responses. Memoized: the entry is immutable once
 * committed to the completed list, so re-renders should be no-ops.
 */

import React, { memo } from "react";
import { Box, Text } from "ink";
import type { ChatEntry, ThemeColors } from "../../types.js";
import { formatTime } from "./time.js";

interface Props {
  entry: ChatEntry;
  theme: ThemeColors;
}

function UserMessageImpl({ entry, theme }: Props) {
  const fg = theme.colorMode === "light" ? "#1f2937" : "#f9fafb";
  return (
    <Box flexDirection="column" marginBottom={1}>
      <Box>
        <Text bold color={theme.user}>
          {"❯ You"}
        </Text>
        <Text color={theme.muted}> {formatTime(entry.timestamp)}</Text>
      </Box>
      <Box
        borderStyle="single"
        borderLeft
        borderRight={false}
        borderTop={false}
        borderBottom={false}
        borderColor={theme.user}
      >
        <Text bold color={fg} backgroundColor={theme.userMessageBg}>
          {" "}{entry.text}{" "}
        </Text>
      </Box>
    </Box>
  );
}

export const UserMessage = memo(UserMessageImpl);
