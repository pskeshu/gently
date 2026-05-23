/**
 * MessageBubble — thin role dispatcher.
 *
 * Each role has its own component in `messages/` so it can own its
 * rendering and be memoized in isolation. Adding a new role (plan
 * approval, rate-limit notice, etc.) is now a matter of adding one
 * file and one switch case.
 */

import React from "react";
import { Text } from "ink";
import { AgentMessage } from "./messages/AgentMessage.js";
import { UserMessage } from "./messages/UserMessage.js";
import { ToolMessage } from "./messages/ToolMessage.js";
import { SystemMessage } from "./messages/SystemMessage.js";
import { SpecPanel } from "./messages/SpecPanel.js";
import type { ChatEntry, ThemeColors } from "../types.js";

interface MessageBubbleProps {
  entry: ChatEntry;
  theme: ThemeColors;
}

export function MessageBubble({ entry, theme }: MessageBubbleProps) {
  if (entry.isSpecCard) {
    return <SpecPanel entry={entry} theme={theme} />;
  }
  switch (entry.role) {
    case "user":
      return <UserMessage entry={entry} theme={theme} />;
    case "agent":
      return <AgentMessage entry={entry} theme={theme} />;
    case "tool":
      return <ToolMessage entry={entry} theme={theme} />;
    case "system":
      return <SystemMessage entry={entry} theme={theme} />;
    default:
      return <Text>{entry.text}</Text>;
  }
}
