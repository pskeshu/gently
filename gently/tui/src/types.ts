/**
 * Shared TypeScript types for the Gently TUI.
 *
 * Mirrors the WebSocket message protocol between the Ink TUI
 * and the Python copilot backend.
 */

// ---------------------------------------------------------------------------
// Server → Client messages
// ---------------------------------------------------------------------------

export interface TokenSnapshot {
  input_tokens: number;
  output_tokens: number;
  total_tokens: number;
  api_calls: number;
}

export interface ConnectedMessage {
  type: "connected";
  session_id: string;
  commands: CommandDef[];
  version: string;
  tokens: TokenSnapshot;
  embryo_count: number;
  timestamp: string;
}

export interface StreamEndMessage {
  type: "stream_end";
  tokens: TokenSnapshot;
}

export interface TextChunk {
  type: "text";
  text: string;
}

export interface ToolStartChunk {
  type: "tool_start";
  tool_name: string;
  tool_input: Record<string, unknown>;
}

export interface ToolCallChunk {
  type: "tool_call";
  tool_name: string;
  tool_input?: Record<string, unknown>;
  duration?: number;
  result_summary?: string;
}

export interface ChoiceOption {
  id: string;
  label: string;
  description?: string;
  disabled?: boolean;
}

export interface ChoiceRequest {
  type: "choice_request";
  choice_data: {
    _type: string;
    question: string;
    options: ChoiceOption[];
    allow_multiple: boolean;
    default_id?: string;
  };
  request_id?: string;
}

export interface CommandResult {
  type: "command_result";
  command: string;
  content?: Record<string, unknown>;
  error?: string;
  action?: string;
}

export interface StateUpdate {
  type: "state_update";
  state: Record<string, unknown>;
}

export interface NotificationMessage {
  type: "notification";
  level: "info" | "warning" | "error" | "success";
  title: string;
  body?: string;
}

export interface ErrorMessage {
  type: "error";
  error: string;
}

export interface PingMessage {
  type: "ping";
}

export interface PongMessage {
  type: "pong";
}

export type ServerMessage =
  | ConnectedMessage
  | StreamEndMessage
  | TextChunk
  | ToolStartChunk
  | ToolCallChunk
  | ChoiceRequest
  | CommandResult
  | StateUpdate
  | NotificationMessage
  | ErrorMessage
  | PingMessage
  | PongMessage;

// ---------------------------------------------------------------------------
// Client → Server messages
// ---------------------------------------------------------------------------

export interface ChatMessage {
  type: "chat";
  text: string;
}

export interface ChoiceResponse {
  type: "choice_response";
  request_id: string;
  selected: string;
}

export interface CommandMessage {
  type: "command";
  command: string;
}

export interface ClientPing {
  type: "ping";
}

export interface CancelMessage {
  type: "cancel";
}

export type ClientMessage =
  | ChatMessage
  | ChoiceResponse
  | CommandMessage
  | CancelMessage
  | ClientPing;

// ---------------------------------------------------------------------------
// UI models
// ---------------------------------------------------------------------------

export type MessageRole = "user" | "copilot" | "system" | "tool";

export interface ChatEntry {
  id: string;
  role: MessageRole;
  text: string;
  timestamp: number;
  toolName?: string;
  toolDuration?: number;
  isStreaming?: boolean;
  isThinking?: boolean;
}

export interface CommandDef {
  name: string;
  description: string;
  aliases: string[];
  category: string;
  usage: string;
  arg_hint: string;
  subcommands: { name: string; description: string }[];
}

export type ConnectionStatus = "connecting" | "connected" | "disconnected";

// ---------------------------------------------------------------------------
// Theme
// ---------------------------------------------------------------------------

export interface ThemeColors {
  name: string;
  primary: string;
  secondary: string;
  accent: string;
  user: string;
  copilot: string;
  system: string;
  tool: string;
  success: string;
  warning: string;
  error: string;
  info: string;
  muted: string;
}
