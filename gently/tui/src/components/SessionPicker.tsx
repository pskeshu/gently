/**
 * Standalone session picker — arrow-key navigation, enter to select.
 *
 * Used in the two-phase TUI launch: Python spawns this as a standalone
 * Ink app, captures the selected session ID from stdout, then launches
 * the full TUI with that session.
 *
 * Output protocol: prints `SESSION:<id>` (or `SESSION:` for new session)
 * to stdout, then exits.
 */

import React, { useState } from "react";
import { Box, Text, useApp, useInput } from "ink";
import { getTheme } from "../theme.js";

export interface SessionItem {
  session_id: string;
  embryo_count: number;
  time: string;
}

interface SessionPickerProps {
  sessions: SessionItem[];
}

export function SessionPicker({ sessions }: SessionPickerProps) {
  const theme = getTheme();
  const { exit } = useApp();

  // Build items: "New Session" at index 0, then real sessions
  const items = [
    { id: "", label: "+ New Session", detail: "start fresh", isNew: true },
    ...sessions.map((s) => ({
      id: s.session_id,
      label: s.session_id,
      detail: `${s.embryo_count} embryo${s.embryo_count !== 1 ? "s" : ""}${s.time ? ` · ${s.time}` : ""}`,
      isNew: false,
    })),
  ];

  const [cursor, setCursor] = useState(0);

  useInput((_input, key) => {
    if (key.upArrow) {
      setCursor((c) => Math.max(0, c - 1));
    } else if (key.downArrow) {
      setCursor((c) => Math.min(items.length - 1, c + 1));
    } else if (key.return) {
      const selected = items[cursor]!;
      // Write selection to stdout for Python to capture
      process.stdout.write(`SESSION:${selected.id}\n`);
      exit();
    } else if (key.escape) {
      // Cancel = new session
      process.stdout.write(`SESSION:\n`);
      exit();
    }
  });

  return (
    <Box flexDirection="column" paddingLeft={2} paddingTop={1}>
      <Box marginBottom={1}>
        <Text bold color={theme.primary}>
          {"✦ "}
        </Text>
        <Text bold>Select a session</Text>
      </Box>
      <Text color={theme.muted}>
        {"  ↑/↓ navigate · Enter select · Esc new session"}
      </Text>
      <Box flexDirection="column" marginTop={1}>
        {items.map((item, i) => {
          const isCursor = i === cursor;
          const marker = isCursor ? "▶ " : "  ";

          if (item.isNew) {
            return (
              <Box key="__new__">
                <Text color={isCursor ? theme.success : theme.muted} bold={isCursor}>
                  {marker}
                </Text>
                <Text color={isCursor ? theme.success : theme.muted} bold={isCursor}>
                  {item.label}
                </Text>
                <Text color={theme.muted}>{" (start fresh)"}</Text>
              </Box>
            );
          }

          return (
            <Box key={item.id}>
              <Text color={isCursor ? theme.info : undefined} bold={isCursor}>
                {marker}
                {item.label}
              </Text>
              <Text color={theme.muted}>
                {" · "}
                {item.detail}
              </Text>
            </Box>
          );
        })}
      </Box>
    </Box>
  );
}
