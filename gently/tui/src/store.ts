/**
 * Zustand store — single source of truth for TUI state.
 *
 * Holds chat messages, connection status, pending choice pickers,
 * command registry, theme, and a message queue.
 *
 * Messages are split into two lists:
 *   - `completedMessages` — finished, rendered via <Static> (never re-render)
 *   - `activeMessage`     — the currently-streaming agent/tool message
 *
 * This split is what gives us the persistent-input-bar behaviour:
 * completed content scrolls up, active content + input stays at the bottom.
 */

import { createStore } from "zustand/vanilla";
import type {
  AppliedSpec,
  BrowserCampaign,
  BrowserPeer,
  BrowserPlanItem,
  ChatEntry,
  ChoiceRequest,
  CommandDef,
  ConnectionStatus,
  ThemeColors,
  TokenSnapshot,
  WizardMeta,
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
  campaignCount: number;

  // Launch status (from connected message)
  deviceConnected: boolean;
  samAvailable: boolean;
  offline: boolean;
  storePath: string;
  vizUrl: string | null;
  logPath: string;
  resumed: boolean;

  // Mesh peers
  peerCount: number;

  // Chat — split for <Static> vs dynamic rendering
  completedMessages: ChatEntry[];
  activeMessage: ChatEntry | null;

  // Queue: messages typed while agent is busy
  messageQueue: string[];

  /**
   * FIFO of choice requests awaiting user response. Head (index 0) is
   * shown by <ChoicePicker>; responding pops the head and reveals the
   * next. Storing as a queue prevents loss when two requests land in
   * quick succession (e.g. agent question while a local picker is open).
   */
  choiceQueue: { request: ChoiceRequest; requestId: string }[];

  // Commands from server
  commands: CommandDef[];

  // Theme
  theme: ThemeColors;

  // Notifications
  notification: { level: string; title: string; body?: string } | null;

  // Whether agent is currently streaming a response
  isStreaming: boolean;

  /**
   * Non-reactive stream metrics. Mutated directly (never via `set`) so
   * per-chunk updates don't trigger React re-renders — `ActiveMessage`
   * polls this via its own setInterval. Resets on stream start/finish.
   */
  streamMetrics: { startedAt: number; charsReceived: number };

  // Agent mode ("run" or "plan")
  agentMode: string;

  // Startup wizard
  wizardActive: boolean;
  wizardWeight: string;

  // Browser panel
  browserOpen: boolean;
  browserCampaigns: BrowserCampaign[];
  browserPeers: BrowserPeer[];
  peerCampaignItems: BrowserPlanItem[];
  peerCampaignMeta: { hostname: string; campaign_id: string } | null;
  campaignBrowserOpen: boolean;
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
    campaignCount: number;
    peerCount: number;
    deviceConnected: boolean;
    samAvailable: boolean;
    offline: boolean;
    storePath: string;
    vizUrl: string | null;
    logPath: string;
    resumed: boolean;
    wizard?: WizardMeta;
    mode?: string;
  }) => void;
  setDisconnected: () => void;
  setConnecting: () => void;
  updateTokens: (tokens: TokenSnapshot) => void;

  addUserMessage: (text: string) => void;
  addUserSelection: (text: string) => void;
  appendAgentText: (text: string) => void;
  showThinking: () => void;
  addToolStart: (toolName: string, toolInput: Record<string, unknown>, toolLabel?: string) => void;
  addToolCall: (toolName: string, duration?: number) => void;
  addSystemMessage: (text: string) => void;
  addCommandResult: (command: string, content: string) => void;
  addSpecCard: (spec: AppliedSpec) => void;
  finishStreaming: () => void;

  // Queue
  enqueueMessage: (text: string) => void;
  dequeueMessage: () => string | undefined;
  clearQueue: () => void;
  removeQueuedAt: (index: number) => void;

  clearMessages: () => void;
  enqueueChoice: (request: ChoiceRequest, requestId: string) => void;
  dismissCurrentChoice: () => void;

  setTheme: (theme: ThemeColors) => void;
  setNotification: (n: { level: string; title: string; body?: string } | null) => void;

  setStreaming: (v: boolean) => void;
  cancelStream: () => void;

  setPeerCount: (count: number) => void;
  setWizardActive: (active: boolean) => void;
  setAgentMode: (mode: string) => void;

  // Browser panel
  setBrowserOpen: (open: boolean) => void;
  setBrowserCampaigns: (campaigns: BrowserCampaign[]) => void;
  setBrowserPeers: (peers: BrowserPeer[]) => void;
  setPeerCampaignItems: (items: BrowserPlanItem[], hostname: string, campaign_id: string) => void;
  clearPeerCampaignItems: () => void;
  setCampaignBrowserOpen: (open: boolean) => void;
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
    // Don't commit empty thinking placeholders — they're transient
    // indicators, not real messages.
    const isEmptyThinking =
      state.activeMessage.isThinking && !state.activeMessage.text;
    if (!isEmptyThinking) {
      // Dedup guard: skip if identical to the last completed message
      // (prevents duplicate rendering from race conditions)
      const last = completed[completed.length - 1];
      const isDuplicate =
        last &&
        last.role === state.activeMessage.role &&
        last.text === state.activeMessage.text &&
        last.text !== "";
      if (!isDuplicate) {
        completed.push({ ...state.activeMessage, isStreaming: false });
      }
    }
  }
  return { completedMessages: completed, activeMessage: null };
}

/** Extract a short summary from tool input for display. */
function extractToolSummary(toolName: string, input: Record<string, unknown>): string {
  // Choice picker — no summary needed
  if (toolName === "ask_user_choice") return "";

  // Look for common descriptive keys in priority order
  const keys = ["query", "question", "description", "title", "campaign_id", "item_id", "shorthand"];
  for (const key of keys) {
    if (input[key] && typeof input[key] === "string") {
      const val = input[key] as string;
      // Truncate long values
      return val.length > 60 ? val.slice(0, 57) + "..." : val;
    }
  }

  // For tools with a "type" field, show it
  if (input["type"] && typeof input["type"] === "string") {
    return input["type"] as string;
  }

  return "";
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
    campaignCount: 0,
    peerCount: 0,
    deviceConnected: false,
    samAvailable: false,
    offline: false,
    storePath: "",
    vizUrl: null,
    logPath: "",
    resumed: false,
    completedMessages: [],
    activeMessage: null,
    messageQueue: [],
    choiceQueue: [],
    commands: [],
    theme: getTheme(),
    notification: null,
    isStreaming: false,
    streamMetrics: { startedAt: 0, charsReceived: 0 },
    agentMode: "run",
    wizardActive: false,
    wizardWeight: "none",
    browserOpen: false,
    browserCampaigns: [],
    browserPeers: [],
    peerCampaignItems: [],
    peerCampaignMeta: null,
    campaignBrowserOpen: false,

    // Connection
    setConnected: (meta) =>
      set({
        connectionStatus: "connected",
        sessionId: meta.sessionId,
        commands: meta.commands,
        version: meta.version,
        tokens: meta.tokens,
        embryoCount: meta.embryoCount,
        campaignCount: meta.campaignCount,
        peerCount: meta.peerCount,
        deviceConnected: meta.deviceConnected,
        samAvailable: meta.samAvailable,
        offline: meta.offline,
        storePath: meta.storePath,
        vizUrl: meta.vizUrl,
        logPath: meta.logPath,
        resumed: meta.resumed,
        agentMode: meta.mode ?? "run",
        wizardActive: meta.wizard?.wizard_needed ?? false,
        wizardWeight: meta.wizard?.conversation_weight ?? "none",
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
        // Reset stream metrics ref in-place (no re-render)
        s.streamMetrics.startedAt = Date.now();
        s.streamMetrics.charsReceived = 0;
        return {
          completedMessages,
          // Show a thinking indicator while waiting for first response chunk
          activeMessage: {
            id: nextId(),
            role: "agent",
            text: "",
            timestamp: Date.now(),
            isThinking: true,
          },
          isStreaming: true,
        };
      }),

    addUserSelection: (text) =>
      set((s) => {
        // If a choice-tool is currently active, attach the answer to the
        // tool entry itself so the question and answer commit together
        // (mirrors src's AskUserQuestionResultMessage pattern). Avoids the
        // ordering bug where the answer was appended to completedMessages
        // before the tool entry, putting the answer above the question.
        if (s.activeMessage?.role === "tool") {
          return {
            activeMessage: {
              ...s.activeMessage,
              toolAnswer: text,
            },
          };
        }
        // Fallback: no active tool — append as a standalone selection row.
        return {
          completedMessages: [
            ...s.completedMessages,
            { id: nextId(), role: "user" as const, text, timestamp: Date.now(), isSelection: true },
          ],
        };
      }),

    appendAgentText: (text) =>
      set((s) => {
        // Mutate stream metrics ref in-place (no re-render trigger)
        s.streamMetrics.charsReceived += text.length;
        if (s.activeMessage && s.activeMessage.role === "agent") {
          // Append to current streaming agent message (clears thinking state)
          return {
            activeMessage: {
              ...s.activeMessage,
              text: s.activeMessage.text + text,
              isThinking: false,
              isStreaming: true,
            },
          };
        }
        // Commit previous active message, start new agent message
        const { completedMessages } = commitActive(s);
        return {
          completedMessages,
          activeMessage: {
            id: nextId(),
            role: "agent",
            text,
            timestamp: Date.now(),
            isStreaming: true,
          },
        };
      }),

    showThinking: () =>
      set((s) => {
        const { completedMessages } = commitActive(s);
        return {
          completedMessages,
          activeMessage: {
            id: nextId(),
            role: "agent",
            text: "",
            timestamp: Date.now(),
            isThinking: true,
          },
          isStreaming: true,
        };
      }),

    addToolStart: (toolName, toolInput, toolLabel?) =>
      set((s) => {
        // Commit any active message, then set tool as active
        const { completedMessages } = commitActive(s);
        // Prefer server-provided label (resolves IDs to names), fall back to client-side extraction
        const summary = toolLabel || extractToolSummary(toolName, toolInput);
        return {
          completedMessages,
          activeMessage: {
            id: nextId(),
            role: "tool",
            text: summary ? `${toolName} — ${summary}` : toolName,
            toolName,
            toolSummary: summary || undefined,
            timestamp: Date.now(),
            isStreaming: true,
          },
        };
      }),

    addToolCall: (toolName, duration) =>
      set((s) => {
        // Show thinking indicator after tool completes if stream is still active
        const thinkingMessage: ChatEntry | null = s.isStreaming
          ? { id: nextId(), role: "agent", text: "", timestamp: Date.now(), isThinking: true }
          : null;

        // Build display text: tool name + summary + meaningful duration
        const isChoice = toolName === "ask_user_choice";
        const buildText = (summary?: string) => {
          let text = toolName;
          if (summary) text += ` — ${summary}`;
          // Only show duration if > 0.1s and not a choice picker (user wait time)
          if (duration && duration > 0.1 && !isChoice) {
            text += ` (${duration.toFixed(1)}s)`;
          }
          return text;
        };

        // If active message is the matching tool, complete it
        if (
          s.activeMessage?.role === "tool" &&
          s.activeMessage.toolName === toolName
        ) {
          const finished: ChatEntry = {
            ...s.activeMessage,
            text: buildText(s.activeMessage.toolSummary),
            toolDuration: isChoice ? undefined : duration,
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
          text: buildText(),
          toolName,
          toolDuration: isChoice ? undefined : duration,
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

    addSpecCard: (spec) =>
      set((s) => {
        const { completedMessages } = commitActive(s);
        completedMessages.push({
          id: nextId(),
          role: "system",
          text: "",
          timestamp: Date.now(),
          isSpecCard: true,
          specData: spec,
        });
        return { completedMessages, activeMessage: null };
      }),

    finishStreaming: () =>
      set((s) => {
        s.streamMetrics.startedAt = 0;
        s.streamMetrics.charsReceived = 0;
        return {
          ...commitActive(s),
          isStreaming: false,
        };
      }),

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

    clearQueue: () => set({ messageQueue: [] }),

    removeQueuedAt: (index) =>
      set((s) => ({
        messageQueue: s.messageQueue.filter((_, i) => i !== index),
      })),

    // Clear all messages
    clearMessages: () =>
      set({ completedMessages: [], activeMessage: null }),

    // Choice picker (FIFO queue)
    enqueueChoice: (request, requestId) =>
      set((s) => ({ choiceQueue: [...s.choiceQueue, { request, requestId }] })),
    dismissCurrentChoice: () =>
      set((s) => ({ choiceQueue: s.choiceQueue.slice(1) })),

    // Theme / notifications
    setTheme: (theme) => set({ theme }),
    setNotification: (n) => set({ notification: n }),
    setStreaming: (v) => set({ isStreaming: v }),

    cancelStream: () =>
      set((s) => {
        s.streamMetrics.startedAt = 0;
        s.streamMetrics.charsReceived = 0;
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

    setPeerCount: (count) => set({ peerCount: count }),
    setWizardActive: (active) => set({ wizardActive: active }),
    setAgentMode: (mode) => set({ agentMode: mode }),

    // Browser panel
    setBrowserOpen: (open) => set({ browserOpen: open }),
    setBrowserCampaigns: (campaigns) => set({ browserCampaigns: campaigns }),
    setBrowserPeers: (peers) => set({ browserPeers: peers }),
    setPeerCampaignItems: (items, hostname, campaign_id) =>
      set({ peerCampaignItems: items, peerCampaignMeta: { hostname, campaign_id } }),
    clearPeerCampaignItems: () =>
      set({ peerCampaignItems: [], peerCampaignMeta: null }),
    setCampaignBrowserOpen: (open) => set({ campaignBrowserOpen: open }),
  }));
}
