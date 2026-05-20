/**
 * Minimal header — just branding. All status info lives in the
 * bottom StatusBar now (like Claude Code).
 */

import React from "react";
import { Box, Text } from "ink";
import { useTuiSelector } from "../context.js";

export function Header() {
  const primary = useTuiSelector((s) => s.theme.primary);
  return (
    <Box paddingX={1}>
      <Text bold color={primary}>
        ✦ Gently
      </Text>
    </Box>
  );
}
