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
  const sendMessage = useCallback(
    (text: string) => {
      if (isSlashCommand(text)) {
        // Commands always go immediately
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
      // Show the user's selection in the chat
      const choice = store.getState().pendingChoice;
      if (choice) {
        const option = choice.choice_data.options.find((o) => o.id === selected);
        const label = option?.label ?? selected;
        store.getState().addUserMessage(label);
      }

      send({
        type: "choice_response",
        request_id: state.pendingChoiceRequestId,
        selected,
      });
      store.getState().clearChoice();
    },
    [send, state.pendingChoiceRequestId, store],
  );

  const handleChoiceCancel = useCallback(() => {
    // Show cancellation in the chat
    store.getState().addUserMessage("(cancelled)");

    send({
      type: "choice_response",
      request_id: state.pendingChoiceRequestId,
      selected: "",
    });
    store.getState().clearChoice();
  }, [send, state.pendingChoiceRequestId, store]);

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
        connectionStatus={state.connectionStatus}
        embryoCount={state.embryoCount}
        tokens={state.tokens}
        notification={state.notification}
        onClearNotification={handleClearNotification}
      />
    </Box>
  );
}
