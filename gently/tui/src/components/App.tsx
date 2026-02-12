/**
 * Root application component — Claude Code-style persistent layout.
 *
 * Architecture:
 *   <Static>         Completed messages — rendered once, scroll up naturally
 *   <ActiveMessage>  Currently streaming copilot/tool message (re-renders on each chunk)
 *   <ChoicePicker>   Interactive picker when copilot asks a question
 *   <CommandInput>   Always-visible input bar at the bottom
 *   <StatusBar>      Notifications
 *
 * The input bar is ALWAYS active. When the copilot is streaming, new
 * messages are queued and auto-sent once the stream finishes.
 */

import React, { useCallback, useEffect, useSyncExternalStore } from "react";
import { Box, Static, Text } from "ink";
import type { StoreApi } from "zustand/vanilla";
import type { TuiStore } from "../store.js";
import { useWebSocket } from "../hooks/useWebSocket.js";
import { useKeyboard } from "../hooks/useKeyboard.js";
import { isSlashCommand } from "../commands.js";
import { setTheme, listThemes } from "../theme.js";

import { Header } from "./Header.js";
import { MessageBubble } from "./MessageBubble.js";
import { ActiveMessage } from "./ChatPane.js";
import { ChoicePicker } from "./ChoicePicker.js";
import { CommandInput } from "./CommandInput.js";
import { StatusBar } from "./StatusBar.js";
import { WelcomeScreen } from "./WelcomeScreen.js";

interface AppProps {
  wsUrl: string;
  store: StoreApi<TuiStore>;
}

export function App({ wsUrl, store }: AppProps) {
  const state = useSyncExternalStore(store.subscribe, store.getState);

  const { send } = useWebSocket(wsUrl, store);
  useKeyboard(store, send);

  // ------------------------------------------------------------------
  // Send a message (or queue it if copilot is busy)
  // ------------------------------------------------------------------
  // ------------------------------------------------------------------
  // Handle /theme locally (no server round-trip needed)
  // ------------------------------------------------------------------
  const handleThemeCommand = useCallback(
    (cmd: string): boolean => {
      const parts = cmd.trim().split(/\s+/);
      if (parts[0] !== "/theme") return false;

      const s = store.getState();
      if (parts.length > 1) {
        // /theme <name> — switch theme
        const name = parts[1]!;
        try {
          setTheme(name);
          const newTheme = listThemes()[name]!;
          s.setTheme(newTheme);
          s.addSystemMessage(`Theme changed to ${newTheme.name}`);
          // Also update Python side
          send({ type: "command", command: cmd });
        } catch {
          const available = Object.keys(listThemes()).join(", ");
          s.addSystemMessage(`Unknown theme: '${name}'. Available: ${available}`);
        }
      } else {
        // /theme — show interactive picker
        const themes = listThemes();
        const current = s.theme.name;
        const options = Object.entries(themes).map(([key, t]) => ({
          id: key,
          label: `${t.name}${t.name === current ? " (current)" : ""}`,
          description: t.colorMode === "dark" ? "Dark mode" : "Light mode",
        }));
        s.setChoice(
          {
            type: "choice_request",
            choice_data: {
              _type: "single",
              question: "Choose a theme",
              options,
              allow_multiple: false,
            },
          },
          "local:theme",
        );
      }
      return true;
    },
    [send, store],
  );

  const sendMessage = useCallback(
    (text: string) => {
      if (isSlashCommand(text)) {
        // Handle /theme locally
        if (handleThemeCommand(text)) return;
        // Other commands go to server
        send({ type: "command", command: text });
        return;
      }

      if (store.getState().isStreaming) {
        // Copilot is busy — queue the message
        store.getState().enqueueMessage(text);
      } else {
        store.getState().addUserMessage(text);
        send({ type: "chat", text });
      }
    },
    [send, store],
  );

  // ------------------------------------------------------------------
  // Auto-drain queue when streaming finishes
  // ------------------------------------------------------------------
  useEffect(() => {
    if (state.isStreaming) return;

    const next = store.getState().dequeueMessage();
    if (next) {
      store.getState().addUserMessage(next);
      send({ type: "chat", text: next });
    }
  }, [state.isStreaming, send, store]);

  // ------------------------------------------------------------------
  // Choice picker callbacks
  // ------------------------------------------------------------------
  const handleChoiceSelect = useCallback(
    (selected: string) => {
      const s = store.getState();
      const requestId = s.pendingChoiceRequestId;

      // Local theme picker — handle without server round-trip
      if (requestId === "local:theme") {
        s.clearChoice();
        try {
          setTheme(selected);
          const newTheme = listThemes()[selected]!;
          s.setTheme(newTheme);
          s.addSystemMessage(`Theme changed to ${newTheme.name}`);
          // Also update Python side
          send({ type: "command", command: `/theme ${selected}` });
        } catch {
          const available = Object.keys(listThemes()).join(", ");
          s.addSystemMessage(`Unknown theme: '${selected}'. Available: ${available}`);
        }
        return;
      }

      // Server choice — show selection without committing active tool
      const choice = s.pendingChoice;
      if (choice) {
        const option = choice.choice_data.options.find((o) => o.id === selected);
        const label = option?.label ?? selected;
        s.addUserSelection(label);
      }

      send({
        type: "choice_response",
        request_id: requestId,
        selected,
      });
      s.clearChoice();
    },
    [send, store],
  );

  const handleChoiceCancel = useCallback(() => {
    const s = store.getState();
    const requestId = s.pendingChoiceRequestId;

    // Local picker — just dismiss
    if (requestId.startsWith("local:")) {
      s.clearChoice();
      return;
    }

    // Server choice — send empty response
    send({
      type: "choice_response",
      request_id: requestId,
      selected: "",
    });
    s.clearChoice();
    // Use addSystemMessage instead of addUserMessage so we don't
    // trigger a new thinking indicator for the cancelled choice.
    s.addSystemMessage("(cancelled)");
    s.finishStreaming();
  }, [send, store]);

  const handleClearNotification = useCallback(() => {
    store.getState().setNotification(null);
  }, [store]);

  return (
    <Box flexDirection="column">
      {/* ── Header ──────────────────────────────────────────── */}
      <Header theme={state.theme} />

      {/* ── Welcome screen (before any messages) ────────────── */}
      {state.connectionStatus === "connected" &&
        state.completedMessages.length === 0 &&
        !state.activeMessage ? (
          <WelcomeScreen
            theme={state.theme}
            version={state.version}
            sessionId={state.sessionId}
            embryoCount={state.embryoCount}
            deviceConnected={state.deviceConnected}
            samAvailable={state.samAvailable}
            offline={state.offline}
            storePath={state.storePath}
            vizUrl={state.vizUrl}
            logPath={state.logPath}
            resumed={state.resumed}
          />
        ) : null}

      {/* ── Completed messages (rendered once, scroll up) ────── */}
      <Static items={state.completedMessages}>
        {(entry) => (
          <Box key={entry.id} flexDirection="column">
            <MessageBubble entry={entry} theme={state.theme} />
          </Box>
        )}
      </Static>

      {/* ── Active streaming message (re-renders on each chunk) ─ */}
      <ActiveMessage entry={state.activeMessage} theme={state.theme} />

      {/* ── Choice picker (when copilot asks a question) ──────── */}
      {state.pendingChoice ? (
        <ChoicePicker
          choice={state.pendingChoice}
          theme={state.theme}
          onSelect={handleChoiceSelect}
          onCancel={handleChoiceCancel}
        />
      ) : null}

      {/* ── Persistent input bar (always at the bottom) ───────── */}
      <CommandInput
        commands={state.commands}
        theme={state.theme}
        isStreaming={state.isStreaming}
        queueLength={state.messageQueue.length}
        onSubmit={sendMessage}
      />

      {/* ── Persistent status bar ─────────────────────────────── */}
      <StatusBar
        theme={state.theme}
        version={state.version}
        sessionId={state.sessionId}
        deviceConnected={state.deviceConnected}
        offline={state.offline}
        embryoCount={state.embryoCount}
        tokens={state.tokens}
        notification={state.notification}
        onClearNotification={handleClearNotification}
      />
    </Box>
  );
}
