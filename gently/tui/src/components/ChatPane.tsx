/**
 * Active streaming area — renders only the currently-streaming message.
 *
 * Completed messages are handled by <Static> in App.tsx and scroll up
 * naturally. This component only shows the live/active content so it
 * can re-render on every text chunk without touching history.
 *
 * Stream metrics (elapsed time, chars received) live as mutable refs on
 * the store (not reactive state) so chunk arrivals don't trigger
 * re-renders elsewhere. We poll them once per second via setInterval.
 */

import React, { useEffect, useRef, useState } from "react";
import { Box, Text } from "ink";
import Spinner from "ink-spinner";
import { MessageBubble } from "./MessageBubble.js";
import { useTuiSelector, useTuiStoreApi } from "../context.js";
import type { ChatEntry, ThemeColors } from "../types.js";

function formatElapsed(ms: number): string {
  const s = Math.floor(ms / 1000);
  if (s < 60) return `${s}s`;
  const m = Math.floor(s / 60);
  const rem = s % 60;
  return `${m}m ${rem}s`;
}

function formatTokens(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}k`;
  return String(n);
}

export function ActiveMessage() {
  const store = useTuiStoreApi();
  const entry = useTuiSelector((s) => s.activeMessage);
  const theme = useTuiSelector((s) => s.theme);

  // Tokens are only needed for the thinking-state stats line.
  const apiCalls = useTuiSelector((s) => s.tokens.api_calls);
  const totalTokens = useTuiSelector((s) => s.tokens.total_tokens);

  // Tick once a second to refresh elapsed/approxTokens from the ref.
  const [tick, setTick] = useState(0);
  const startedAtRef = useRef(0);
  useEffect(() => {
    const startedAt = store.getState().streamMetrics.startedAt;
    startedAtRef.current = startedAt;
    if (!startedAt) {
      setTick(0);
      return;
    }
    setTick((t) => t + 1);
    const interval = setInterval(() => setTick((t) => t + 1), 1000);
    return () => clearInterval(interval);
  }, [entry?.id, store]);

  if (!entry) return null;

  const metrics = store.getState().streamMetrics;
  const elapsed = metrics.startedAt ? Date.now() - metrics.startedAt : 0;

  // Thinking state — spinner with elapsed timer and session stats.
  if (entry.isThinking) {
    let stats = "";
    if (apiCalls > 0) {
      stats = ` · ${formatTokens(totalTokens)} tokens · ${apiCalls} API call${apiCalls !== 1 ? "s" : ""}`;
    } else if (elapsed >= 2000) {
      stats = " · calling API...";
    }

    return (
      <Box marginBottom={1}>
        <Text color={theme.agent}>
          <Spinner type="dots" />
        </Text>
        <Text color={theme.muted}>
          {" "}Thinking{elapsed >= 1000 ? ` ${formatElapsed(elapsed)}` : ""}
          {stats}
        </Text>
      </Box>
    );
  }

  // Streaming text — show message with elapsed time and approx output tokens.
  const approxTokens = Math.round(metrics.charsReceived / 4);
  // Reference `tick` so React re-runs this branch on the interval.
  void tick;

  return (
    <Box flexDirection="column">
      <MessageBubble entry={entry} theme={theme} />
      {entry.isStreaming && elapsed >= 1000 ? (
        <Box>
          <Text color={theme.muted}>
            {"  "}{formatElapsed(elapsed)}
            {approxTokens > 0 ? ` · ~${approxTokens} output tokens` : ""}
          </Text>
        </Box>
      ) : null}
    </Box>
  );
}

// Kept exported for any external imports; the props are now derived from
// the store internally so callers can render <ActiveMessage /> directly.
export type { ChatEntry, ThemeColors };
