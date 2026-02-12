#!/usr/bin/env node
/**
 * Gently TUI — entry point.
 *
 * Parses CLI args, creates the Zustand store, and renders the
 * Ink <App> component which owns the WebSocket connection.
 *
 * Modes:
 *   node dist/index.js --ws-url ws://localhost:8080/ws/copilot
 *   node dist/index.js --pick-session '<json>'   (standalone picker)
 */

import React from "react";
import { render } from "ink";
import { App } from "./components/App.js";
import { SessionPicker } from "./components/SessionPicker.js";
import type { SessionItem } from "./components/SessionPicker.js";
import { createTuiStore } from "./store.js";

// ---------------------------------------------------------------------------
// Parse args
// ---------------------------------------------------------------------------

function parseArgs(): { wsUrl: string; pickSession: string | null } {
  const args = process.argv.slice(2);
  let wsUrl = "ws://localhost:8080/ws/copilot";
  let pickSession: string | null = null;

  for (let i = 0; i < args.length; i++) {
    if (args[i] === "--ws-url" && args[i + 1]) {
      wsUrl = args[i + 1]!;
      i++;
    } else if (args[i] === "--pick-session" && args[i + 1]) {
      pickSession = args[i + 1]!;
      i++;
    }
  }

  return { wsUrl, pickSession };
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

const { wsUrl, pickSession } = parseArgs();

if (pickSession) {
  // Two-phase launch: standalone session picker mode.
  // Parse the JSON session list, render picker, print selection to stdout.
  let sessions: SessionItem[] = [];
  try {
    sessions = JSON.parse(pickSession) as SessionItem[];
  } catch {
    process.stderr.write("Error: invalid --pick-session JSON\n");
    process.exit(1);
  }

  // Render to stderr so the UI is visible to the user, while stdout
  // is captured by the parent Python process for the SESSION: protocol.
  const { waitUntilExit } = render(<SessionPicker sessions={sessions} />, {
    stdout: process.stderr,
  });
  waitUntilExit().then(() => {
    process.exit(0);
  });
} else {
  // Normal mode: full TUI with WebSocket connection.
  const store = createTuiStore();
  const { waitUntilExit } = render(<App wsUrl={wsUrl} store={store} />);
  waitUntilExit().then(() => {
    process.exit(0);
  });
}
