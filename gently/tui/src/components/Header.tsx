/**
 * Minimal header — just branding. All status info lives in the
 * bottom StatusBar now (like Claude Code).
 */

import React from "react";
import { Box, Text } from "ink";
import type { ThemeColors } from "../types.js";

interface HeaderProps {
  theme: ThemeColors;
}

export function Header({ theme }: HeaderProps) {
  return (
    <Box paddingX={1}>
      <Text bold color={theme.primary}>
        ✦ Gently
      </Text>
    </Box>
  );
}
