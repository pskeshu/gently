/**
 * Persistent input bar — always at the bottom, always accepts typing.
 *
 * When the agent is streaming, new messages are queued and sent
 * automatically once the stream finishes. The queue is rendered as
 * a small panel above the input so the user can see (and clear)
 * what's pending.
 *
 * Slash-command autocomplete shows when input starts with "/".
 */

import React, { useState } from "react";
import { Box, Text, useInput } from "ink";
import TextInput from "ink-text-input";
import { getCompletions } from "../commands.js";
import { useTuiSelector, useTuiStoreApi } from "../context.js";

interface CommandInputProps {
  onSubmit: (text: string) => void;
}

// One-line preview of a queued message: collapse whitespace, truncate.
function previewQueued(text: string, maxLen: number): string {
  const collapsed = text.replace(/\s+/g, " ").trim();
  if (collapsed.length <= maxLen) return collapsed;
  return collapsed.slice(0, Math.max(0, maxLen - 1)) + "…";
}

export function CommandInput({ onSubmit }: CommandInputProps) {
  const store = useTuiStoreApi();
  const commands = useTuiSelector((s) => s.commands);
  const theme = useTuiSelector((s) => s.theme);
  const messageQueue = useTuiSelector((s) => s.messageQueue);
  const queueLength = messageQueue.length;
  const browserOpen = useTuiSelector((s) => s.browserOpen);

  const onOpenBrowser = () => store.getState().setBrowserOpen(true);

  const [value, setValue] = useState("");
  const [completionIdx, setCompletionIdx] = useState(-1);
  // Incrementing key forces TextInput to re-mount, resetting cursor to end
  const [inputKey, setInputKey] = useState(0);

  const completions = value.startsWith("/")
    ? getCompletions(value, commands)
    : [];

  const showCompletions = completions.length > 0 && completionIdx < 0;

  useInput((input, key) => {
    if (key.downArrow && value === "" && !browserOpen) {
      onOpenBrowser();
      return;
    }
    if (browserOpen) return;
    // Ctrl+X — clear the pending message queue.
    if (key.ctrl && input === "x" && queueLength > 0) {
      store.getState().clearQueue();
      return;
    }
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

  // Width budget for queued previews — column width minus padding for
  // the prefix ("  N. ") and trailing hint.
  const cols = process.stdout.columns || 80;
  const previewWidth = Math.max(20, cols - 8);

  // Show up to 3 most recent queued messages; collapse the rest.
  const MAX_VISIBLE = 3;
  const visibleStart = Math.max(0, queueLength - MAX_VISIBLE);
  const visible = messageQueue.slice(visibleStart);
  const hiddenCount = visibleStart;

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

      {/* Queued messages panel — visible only when something is queued. */}
      {queueLength > 0 ? (
        <Box flexDirection="column" marginBottom={0}>
          <Text>
            <Text color={theme.warning} bold>
              {`⏳ Queued (${queueLength})`}
            </Text>
            <Text color={theme.muted}>{"  ·  ctrl+x clear · sends when stream ends"}</Text>
          </Text>
          {hiddenCount > 0 ? (
            <Text color={theme.muted}>
              {`    … ${hiddenCount} earlier`}
            </Text>
          ) : null}
          {visible.map((msg, i) => {
            const idx = visibleStart + i + 1;
            return (
              <Text key={`${idx}-${msg.slice(0, 12)}`}>
                <Text color={theme.muted}>{`  ${String(idx).padStart(2, " ")}. `}</Text>
                <Text color={theme.warning}>{previewQueued(msg, previewWidth)}</Text>
              </Text>
            );
          })}
        </Box>
      ) : null}

      {/* Separator line */}
      <Text color={theme.muted}>{"─".repeat(cols)}</Text>

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
          placeholder={
            queueLength > 0
              ? `Send another… (${queueLength} queued)`
              : "Send a message..."
          }
          focus={!browserOpen}
        />
      </Box>

      {/* Bottom separator before status bar */}
      <Text color={theme.muted}>{"─".repeat(cols)}</Text>
    </Box>
  );
}
