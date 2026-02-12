#!/usr/bin/env node
/**
 * Gently TUI — entry point.
 *
 * Parses CLI args, creates the Zustand store, and renders the
 * Ink <App> component which owns the WebSocket connection.
 *
 * Usage:
 *   node dist/index.js --ws-url ws://localhost:8080/ws/copilot
 */

import React from "react";
import { render } from "ink";
import { App } from "./components/App.js";
import { createTuiStore } from "./store.js";

// ---------------------------------------------------------------------------
// Parse args
// ---------------------------------------------------------------------------

function parseArgs(): { wsUrl: string } {
  const args = process.argv.slice(2);
  let wsUrl = "ws://localhost:8080/ws/copilot";

  for (let i = 0; i < args.length; i++) {
    if (args[i] === "--ws-url" && args[i + 1]) {
      wsUrl = args[i + 1]!;
      i++;
    }
  }

  return { wsUrl };
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

const { wsUrl } = parseArgs();
const store = createTuiStore();

const { waitUntilExit } = render(<App wsUrl={wsUrl} store={store} />);

waitUntilExit().then(() => {
  process.exit(0);
});
