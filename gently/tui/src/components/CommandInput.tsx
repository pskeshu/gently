/**
 * Persistent input bar — always at the bottom, always accepts typing.
 *
 * When the agent is streaming, new messages are queued and sent
 * automatically once the stream finishes. A queue indicator shows
 * how many messages are waiting.
 *
 * Slash-command autocomplete shows when input starts with "/".
 */

import React, { useState } from "react";
import { Box, Text, useInput } from "ink";
import TextInput from "ink-text-input";
import { getCompletions } from "../commands.js";
import type { CommandDef, ThemeColors } from "../types.js";

interface CommandInputProps {
  commands: CommandDef[];
  theme: ThemeColors;
  isStreaming: boolean;
  queueLength: number;
  onSubmit: (text: string) => void;
  onOpenBrowser?: () => void;
  browserOpen?: boolean;
}

export function CommandInput({
  commands,
  theme,
  isStreaming,
  queueLength,
  onSubmit,
  onOpenBrowser,
  browserOpen,
}: CommandInputProps) {
  const [value, setValue] = useState("");
  const [completionIdx, setCompletionIdx] = useState(-1);
  // Incrementing key forces TextInput to re-mount, resetting cursor to end
  const [inputKey, setInputKey] = useState(0);

  const completions = value.startsWith("/")
    ? getCompletions(value, commands)
    : [];

  const showCompletions = completions.length > 0 && completionIdx < 0;

  useInput((input, key) => {
    if (key.downArrow && value === "" && onOpenBrowser && !browserOpen) {
      onOpenBrowser();
      return;
    }
    if (browserOpen) return;
    if (key.tab && !key.shift && completions.length > 0) {
      const idx =
        completionIdx < 0 ? 0 : (completionIdx + 1) % completions.length;
      setCompletionIdx(idx);
      const cmd = completions[idx];
      if (cmd) {
        setValue(cmd.name + " ");
        setInputKey((k) => k + 1);
      }
    }
  });

  function handleChange(v: string) {
    setValue(v);
    setCompletionIdx(-1);
  }

  function handleSubmit(v: string) {
    const trimmed = v.trim();
    if (!trimmed) return;
    setValue("");
    setCompletionIdx(-1);
    onSubmit(trimmed);
  }

  return (
    <Box flexDirection="column">
      {/* Autocomplete dropdown */}
      {showCompletions ? (
        <Box flexDirection="column" marginBottom={0}>
          {completions.slice(0, 5).map((cmd) => (
            <Text key={cmd.name}>
              <Text color={theme.info}>  {cmd.name}</Text>
              <Text color={theme.muted}> — {cmd.description}</Text>
            </Text>
          ))}
        </Box>
      ) : null}

      {/* Separator line */}
      <Text color={theme.muted}>{"─".repeat(process.stdout.columns || 80)}</Text>

      {/* Input line */}
      <Box>
        <Text color={theme.user} bold>
          {"❯ "}
        </Text>
        <TextInput
          key={inputKey}
          value={value}
          onChange={handleChange}
          onSubmit={handleSubmit}
          placeholder="Send a message..."
          focus={!browserOpen}
        />
        {/* Queue indicator */}
        {queueLength > 0 ? (
          <Text color={theme.warning}>
            {" "}
            ({queueLength} queued)
          </Text>
        ) : null}
      </Box>

      {/* Bottom separator before status bar */}
      <Text color={theme.muted}>{"─".repeat(process.stdout.columns || 80)}</Text>
    </Box>
  );
}
