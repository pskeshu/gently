/**
 * Zustand store — single source of truth for TUI state.
 *
 * Holds chat messages, connection status, pending choice pickers,
 * command registry, theme, and a message queue.
 *
 * Messages are split into two lists:
 *   - `completedMessages` — finished, rendered via <Static> (never re-render)
 *   - `activeMessage`     — the currently-streaming copilot/tool message
 *
 * This split is what gives us the persistent-input-bar behaviour:
 * completed content scrolls up, active content + input stays at the bottom.
 */

import { createStore } from "zustand/vanilla";
import type {
  ChatEntry,
  ChoiceRequest,
  CommandDef,
  ConnectionStatus,
  ThemeColors,
  TokenSnapshot,
} from "./types.js";
import { getTheme } from "./theme.js";

// ---------------------------------------------------------------------------
// State shape
// ---------------------------------------------------------------------------

export interface TuiState {
  // Connection
  connectionStatus: ConnectionStatus;
  sessionId: string;

  // Metadata from server
  version: string;
  tokens: TokenSnapshot;
  embryoCount: number;

  // Chat — split for <Static> vs dynamic rendering
  completedMessages: ChatEntry[];
  activeMessage: ChatEntry | null;

  // Queue: messages typed while copilot is busy
  messageQueue: string[];

  // Active choice picker (null when none)
  pendingChoice: ChoiceRequest | null;
  pendingChoiceRequestId: string;

  // Commands from server
  commands: CommandDef[];

  // Theme
  theme: ThemeColors;

  // Notifications
  notification: { level: string; title: string; body?: string } | null;

  // Whether copilot is currently streaming a response
  isStreaming: boolean;
}

// ---------------------------------------------------------------------------
// Actions
// ---------------------------------------------------------------------------

export interface TuiActions {
  setConnected: (meta: {
    sessionId: string;
    commands: CommandDef[];
    version: string;
    tokens: TokenSnapshot;
    embryoCount: number;
  }) => void;
  setDisconnected: () => void;
  setConnecting: () => void;
  updateTokens: (tokens: TokenSnapshot) => void;

  addUserMessage: (text: string) => void;
  appendCopilotText: (text: string) => void;
  addToolStart: (toolName: string, toolInput: Record<string, unknown>) => void;
  addToolCall: (toolName: string, duration?: number) => void;
  addSystemMessage: (text: string) => void;
  addCommandResult: (command: string, content: string) => void;
  finishStreaming: () => void;

  // Queue
  enqueueMessage: (text: string) => void;
  dequeueMessage: () => string | undefined;

  clearMessages: () => void;
  setChoice: (choice: ChoiceRequest, requestId: string) => void;
  clearChoice: () => void;

  setTheme: (theme: ThemeColors) => void;
  setNotification: (n: { level: string; title: string; body?: string } | null) => void;

  setStreaming: (v: boolean) => void;
  cancelStream: () => void;
}

export type TuiStore = TuiState & TuiActions;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

let _nextId = 0;
function nextId(): string {
  return `msg_${++_nextId}`;
}

/** Move activeMessage into completedMessages (mark finished). */
function commitActive(state: TuiState): {
  completedMessages: ChatEntry[];
  activeMessage: null;
} {
  const completed = [...state.completedMessages];
  if (state.activeMessage) {
    completed.push({ ...state.activeMessage, isStreaming: false });
  }
  return { completedMessages: completed, activeMessage: null };
}

// ---------------------------------------------------------------------------
// Store factory
// ---------------------------------------------------------------------------

export function createTuiStore() {
  return createStore<TuiStore>((set, get) => ({
    // Initial state
    connectionStatus: "connecting",
    sessionId: "",
    version: "",
    tokens: { input_tokens: 0, output_tokens: 0, total_tokens: 0, api_calls: 0 },
    embryoCount: 0,
    completedMessages: [],
    activeMessage: null,
    messageQueue: [],
    pendingChoice: null,
    pendingChoiceRequestId: "",
    commands: [],
    theme: getTheme(),
    notification: null,
    isStreaming: false,

    // Connection
    setConnected: (meta) =>
      set({
        connectionStatus: "connected",
        sessionId: meta.sessionId,
        commands: meta.commands,
        version: meta.version,
        tokens: meta.tokens,
        embryoCount: meta.embryoCount,
      }),
    setDisconnected: () => set({ connectionStatus: "disconnected" }),
    setConnecting: () => set({ connectionStatus: "connecting" }),
    updateTokens: (tokens) => set({ tokens }),

    // ------------------------------------------------------------------
    // Chat mutations
    // ------------------------------------------------------------------

    addUserMessage: (text) =>
      set((s) => {
        // Commit any active message first, then add user message to completed
        const { completedMessages } = commitActive(s);
        completedMessages.push({
          id: nextId(),
          role: "user",
          text,
          timestamp: Date.now(),
        });
        return {
          completedMessages,
          // Show a thinking indicator while waiting for first response chunk
          activeMessage: {
            id: nextId(),
            role: "copilot",
            text: "",
            timestamp: Date.now(),
            isThinking: true,
          },
          isStreaming: true,
        };
      }),

    appendCopilotText: (text) =>
      set((s) => {
        if (s.activeMessage && s.activeMessage.role === "copilot") {
          // Append to current streaming copilot message (clears thinking state)
          return {
            activeMessage: {
              ...s.activeMessage,
              text: s.activeMessage.text + text,
              isThinking: false,
              isStreaming: true,
            },
          };
        }
        // Commit previous active message, start new copilot message
        const { completedMessages } = commitActive(s);
        return {
          completedMessages,
          activeMessage: {
            id: nextId(),
            role: "copilot",
            text,
            timestamp: Date.now(),
            isStreaming: true,
          },
        };
      }),

    addToolStart: (toolName, _toolInput) =>
      set((s) => {
        // Commit any active message, then set tool as active
        const { completedMessages } = commitActive(s);
        return {
          completedMessages,
          activeMessage: {
            id: nextId(),
            role: "tool",
            text: `Running ${toolName}...`,
            toolName,
            timestamp: Date.now(),
            isStreaming: true,
          },
        };
      }),

    addToolCall: (toolName, duration) =>
      set((s) => {
        // Show thinking indicator after tool completes if stream is still active
        const thinkingMessage: ChatEntry | null = s.isStreaming
          ? { id: nextId(), role: "copilot", text: "", timestamp: Date.now(), isThinking: true }
          : null;

        // If active message is the matching tool, complete it
        if (
          s.activeMessage?.role === "tool" &&
          s.activeMessage.toolName === toolName
        ) {
          const finished: ChatEntry = {
            ...s.activeMessage,
            text: duration
              ? `${toolName} (${duration.toFixed(2)}s)`
              : toolName,
            toolDuration: duration,
            isStreaming: false,
          };
          return {
            completedMessages: [...s.completedMessages, finished],
            activeMessage: thinkingMessage,
          };
        }
        // Otherwise commit active and add as completed directly
        const { completedMessages } = commitActive(s);
        completedMessages.push({
          id: nextId(),
          role: "tool",
          text: duration
            ? `${toolName} (${duration.toFixed(2)}s)`
            : toolName,
          toolName,
          toolDuration: duration,
          timestamp: Date.now(),
        });
        return { completedMessages, activeMessage: thinkingMessage };
      }),

    addSystemMessage: (text) =>
      set((s) => {
        const { completedMessages } = commitActive(s);
        completedMessages.push({
          id: nextId(),
          role: "system",
          text,
          timestamp: Date.now(),
        });
        return { completedMessages, activeMessage: null };
      }),

    addCommandResult: (command, content) =>
      set((s) => {
        const { completedMessages } = commitActive(s);
        completedMessages.push({
          id: nextId(),
          role: "system",
          text: content,
          timestamp: Date.now(),
        });
        return { completedMessages, activeMessage: null };
      }),

    finishStreaming: () =>
      set((s) => ({
        ...commitActive(s),
        isStreaming: false,
      })),

    // ------------------------------------------------------------------
    // Message queue
    // ------------------------------------------------------------------

    enqueueMessage: (text) =>
      set((s) => ({ messageQueue: [...s.messageQueue, text] })),

    dequeueMessage: () => {
      const s = get();
      if (s.messageQueue.length === 0) return undefined;
      const [next, ...rest] = s.messageQueue;
      set({ messageQueue: rest });
      return next;
    },

    // Clear all messages
    clearMessages: () =>
      set({ completedMessages: [], activeMessage: null }),

    // Choice picker
    setChoice: (choice, requestId) =>
      set({ pendingChoice: choice, pendingChoiceRequestId: requestId }),
    clearChoice: () =>
      set({ pendingChoice: null, pendingChoiceRequestId: "" }),

    // Theme / notifications
    setTheme: (theme) => set({ theme }),
    setNotification: (n) => set({ notification: n }),
    setStreaming: (v) => set({ isStreaming: v }),

    cancelStream: () =>
      set((s) => {
        // If thinking (no text yet), just clear. If streaming text,
        // commit what we have and mark as cancelled.
        if (s.activeMessage?.isThinking) {
          return {
            activeMessage: null,
            isStreaming: false,
          };
        }
        return {
          ...commitActive(s),
          isStreaming: false,
        };
      }),
  }));
}
