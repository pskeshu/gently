/**
 * Interactive choice picker — arrow-key navigation, multi-select,
 * enter to submit, escape to cancel.
 *
 * Every picker automatically gets a "Something else..." option at the
 * bottom with an inline text input, so the user can always type a
 * custom response.  Backend options with id "__custom__" are merged
 * rather than duplicated.
 */

import React, { useState } from "react";
import { Box, Text, useInput } from "ink";
import TextInput from "ink-text-input";
import { useTuiSelector } from "../context.js";

interface ChoicePickerProps {
  onSelect: (selected: string) => void;
  onCancel: () => void;
}

export function ChoicePicker({ onSelect, onCancel }: ChoicePickerProps) {
  const choice = useTuiSelector((s) => s.choiceQueue[0]?.request ?? null);
  const theme = useTuiSelector((s) => s.theme);
  if (!choice) return null;
  const allowMultiple = choice.choice_data.allow_multiple;

  // Auto-append a "Something else..." option if none exists
  const rawOptions = choice.choice_data.options;
  const hasCustom = rawOptions.some((o) => o.id === "__custom__");
  const options = hasCustom
    ? rawOptions
    : [
        ...rawOptions,
        { id: "__custom__", label: "Something else..." },
      ];

  const [cursor, setCursor] = useState(0);
  const [selected, setSelected] = useState<Set<string>>(() => {
    const defaultId = choice.choice_data.default_id;
    return defaultId ? new Set([defaultId]) : new Set();
  });

  // Inline text input for __custom__ options
  const [customText, setCustomText] = useState("");
  const cursorOption = options[cursor];
  const isCustom = cursorOption?.id === "__custom__";

  useInput((input, key) => {
    if (key.escape) {
      onCancel();
    } else if (key.upArrow) {
      setCursor((c) => Math.max(0, c - 1));
    } else if (key.downArrow) {
      setCursor((c) => Math.min(options.length - 1, c + 1));
    } else if (key.return && !isCustom) {
      if (allowMultiple) {
        onSelect(Array.from(selected).join(","));
      } else {
        const opt = options[cursor];
        if (opt && !opt.disabled) {
          onSelect(opt.id);
        }
      }
    } else if (input === " " && allowMultiple && !isCustom) {
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
    }
  });

  function handleCustomSubmit(value: string) {
    const trimmed = value.trim();
    if (trimmed) {
      onSelect(trimmed);
    }
  }

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
        {isCustom
          ? "Type your answer · Enter submit · Esc cancel"
          : allowMultiple
            ? "↑/↓ navigate · Space toggle · Enter submit · Esc cancel"
            : "↑/↓ navigate · Enter select · Esc cancel"}
      </Text>
      <Box flexDirection="column" marginTop={1}>
        {options.map((opt, i) => {
          const isCursor = i === cursor;
          const isSelected = selected.has(opt.id);
          const isDisabled = opt.disabled;
          const isThisCustom = opt.id === "__custom__";

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
              {opt.description && isCursor && !isThisCustom ? (
                <Text color={theme.muted}>    {opt.description}</Text>
              ) : null}
              {isThisCustom && isCursor ? (
                <Box marginLeft={2} marginTop={0}>
                  <Text color={theme.info}>❯ </Text>
                  <TextInput
                    value={customText}
                    onChange={setCustomText}
                    onSubmit={handleCustomSubmit}
                    placeholder="Type here..."
                  />
                </Box>
              ) : null}
            </Box>
          );
        })}
      </Box>
    </Box>
  );
}
