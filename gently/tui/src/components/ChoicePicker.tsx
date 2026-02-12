/**
 * Interactive choice picker — arrow-key navigation, multi-select,
 * enter to submit, escape to cancel.
 *
 * Modeled after Claude Code's AskUserQuestion component.
 */

import React, { useState } from "react";
import { Box, Text, useInput } from "ink";
import type { ChoiceRequest, ThemeColors } from "../types.js";

interface ChoicePickerProps {
  choice: ChoiceRequest;
  theme: ThemeColors;
  onSelect: (selected: string) => void;
  onCancel: () => void;
}

export function ChoicePicker({
  choice,
  theme,
  onSelect,
  onCancel,
}: ChoicePickerProps) {
  const options = choice.choice_data.options;
  const allowMultiple = choice.choice_data.allow_multiple;

  const [cursor, setCursor] = useState(0);
  const [selected, setSelected] = useState<Set<string>>(() => {
    const defaultId = choice.choice_data.default_id;
    return defaultId ? new Set([defaultId]) : new Set();
  });

  useInput((input, key) => {
    if (key.upArrow) {
      setCursor((c) => Math.max(0, c - 1));
    } else if (key.downArrow) {
      setCursor((c) => Math.min(options.length - 1, c + 1));
    } else if (key.return) {
      if (allowMultiple) {
        // Submit all selected
        onSelect(Array.from(selected).join(","));
      } else {
        // Submit the one under cursor
        const opt = options[cursor];
        if (opt && !opt.disabled) {
          onSelect(opt.id);
        }
      }
    } else if (input === " " && allowMultiple) {
      const opt = options[cursor];
      if (opt && !opt.disabled) {
        setSelected((prev) => {
          const next = new Set(prev);
          if (next.has(opt.id)) {
            next.delete(opt.id);
          } else {
            next.add(opt.id);
          }
          return next;
        });
      }
    } else if (key.escape) {
      onCancel();
    }
  });

  return (
    <Box
      flexDirection="column"
      borderStyle="round"
      borderColor={theme.info}
      paddingX={1}
      marginBottom={1}
    >
      <Text bold color={theme.info}>
        {choice.choice_data.question}
      </Text>
      <Text color={theme.muted}>
        {allowMultiple
          ? "↑/↓ navigate · Space toggle · Enter submit · Esc cancel"
          : "↑/↓ navigate · Enter select · Esc cancel"}
      </Text>
      <Box flexDirection="column" marginTop={1}>
        {options.map((opt, i) => {
          const isCursor = i === cursor;
          const isSelected = selected.has(opt.id);
          const isDisabled = opt.disabled;

          let marker = "  ";
          if (allowMultiple) {
            marker = isSelected ? "◉ " : "○ ";
          } else if (isCursor) {
            marker = "▶ ";
          }

          const color = isDisabled
            ? theme.muted
            : isCursor
              ? theme.info
              : undefined;

          return (
            <Box key={opt.id} flexDirection="column">
              <Text
                color={color}
                bold={isCursor}
                dimColor={isDisabled}
              >
                {marker}
                {opt.label}
              </Text>
              {opt.description && isCursor ? (
                <Text color={theme.muted}>    {opt.description}</Text>
              ) : null}
            </Box>
          );
        })}
      </Box>
    </Box>
  );
}
