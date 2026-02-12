/**
 * Welcome screen shown on first connect before any messages.
 * Disappears once the user sends their first message.
 *
 * Two-column layout: branding, session info, device status, and paths
 * on the left; quick-start hints and viz URL on the right.
 */

import React from "react";
import { Box, Text } from "ink";
import type { ThemeColors } from "../types.js";

interface WelcomeScreenProps {
  theme: ThemeColors;
  version: string;
  sessionId: string;
  embryoCount: number;
  deviceConnected: boolean;
  samAvailable: boolean;
  offline: boolean;
  storePath: string;
  vizUrl: string | null;
  logPath: string;
  resumed: boolean;
}

/** Colored status icon: + green, x red, ! yellow */
function StatusIcon({
  ok,
  theme,
  warn,
}: {
  ok: boolean;
  theme: ThemeColors;
  warn?: boolean;
}) {
  if (ok) return <Text color={theme.success}>{"+"}</Text>;
  if (warn) return <Text color={theme.warning}>{"!"}</Text>;
  return <Text color={theme.error}>{"x"}</Text>;
}

export function WelcomeScreen({
  theme,
  version,
  sessionId,
  embryoCount,
  deviceConnected,
  samAvailable,
  offline,
  storePath,
  vizUrl,
  logPath,
  resumed,
}: WelcomeScreenProps) {
  const shortSession = sessionId.slice(0, 8);

  // Truncate long paths to keep the layout compact (must fit within
  // the 40-char left column minus the "Store: " / "Log: " prefix).
  const maxPathLen = 32;
  const trimPath = (p: string) =>
    p.length > maxPathLen ? "..." + p.slice(p.length - maxPathLen + 3) : p;

  return (
    <Box paddingLeft={2} paddingTop={1} paddingBottom={1} gap={6}>
      {/* Left column — branding, session, device status, paths */}
      <Box flexDirection="column" width={40}>
        {/* Branding */}
        <Box>
          <Text bold color={theme.primary}>
            {"✦ gently"}
          </Text>
          <Text color={theme.muted}> v{version}</Text>
        </Box>

        {/* Blank spacer line */}
        <Text>{""}</Text>

        {/* Session info */}
        <Box>
          <Text color={theme.muted}>
            {resumed ? "Resumed" : "Session"} {shortSession}
            {" · "}
            {embryoCount} embryo{embryoCount !== 1 ? "s" : ""}
          </Text>
        </Box>

        {/* Device status lines */}
        {offline ? (
          <Box>
            <StatusIcon ok={false} theme={theme} warn />
            <Text color={theme.warning}>{" Offline mode"}</Text>
          </Box>
        ) : (
          <>
            <Box>
              <StatusIcon ok={deviceConnected} theme={theme} />
              <Text color={deviceConnected ? theme.success : theme.error}>
                {" Device Layer  "}
              </Text>
              <Text color={deviceConnected ? theme.muted : theme.error}>
                {deviceConnected ? "connected" : "not connected"}
              </Text>
            </Box>
            <Box>
              <StatusIcon ok={samAvailable} theme={theme} warn={!samAvailable} />
              <Text color={samAvailable ? theme.success : theme.warning}>
                {" SAM Detection "}
              </Text>
              <Text color={samAvailable ? theme.muted : theme.warning}>
                {samAvailable ? "available" : "not available"}
              </Text>
            </Box>
          </>
        )}

        {/* Blank spacer */}
        <Text>{""}</Text>

        {/* Paths */}
        {storePath ? (
          <Text color={theme.muted}>Store: {trimPath(storePath)}</Text>
        ) : null}
        {logPath ? (
          <Text color={theme.muted}>Log: {trimPath(logPath)}</Text>
        ) : null}
      </Box>

      {/* Right column — quick-start hints + links */}
      <Box flexDirection="column">
        <Text color={theme.muted}>Quick start:</Text>
        <Box paddingLeft={1} flexDirection="column">
          <Text color={theme.muted}>
            {"Type a message to chat"}
          </Text>
          <Text color={theme.muted}>
            {"Use "}
            <Text color={theme.accent}>/help</Text>
            {" for commands"}
          </Text>
          <Text color={theme.muted}>
            {"Press "}
            <Text color={theme.accent}>Esc</Text>
            {" to cancel"}
          </Text>
        </Box>

        {/* Blank spacer */}
        <Text>{""}</Text>

        {vizUrl ? (
          <>
            <Text color={theme.muted}>Links:</Text>
            <Box paddingLeft={1}>
              <Text color={theme.muted}>
                {"Viz: "}
                <Text color={theme.info}>{vizUrl}</Text>
              </Text>
            </Box>
          </>
        ) : null}
      </Box>
    </Box>
  );
}
