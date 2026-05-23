/**
 * Shared TypeScript types for the Gently TUI.
 *
 * Mirrors the WebSocket message protocol between the Ink TUI
 * and the Python agent backend.
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

export interface WizardMeta {
  wizard_needed: boolean;
  conversation_weight: string;
  is_first_launch: boolean;
  readiness: number;
}

export interface ConnectedMessage {
  type: "connected";
  session_id: string;
  commands: CommandDef[];
  version: string;
  tokens: TokenSnapshot;
  embryo_count: number;
  campaign_count?: number;
  timestamp: string;
  // Launch status metadata (populated by TUI launch path)
  device_connected: boolean;
  sam_available: boolean;
  offline: boolean;
  store_path: string;
  viz_url: string | null;
  log_path: string;
  resumed: boolean;
  // Mesh peers
  peer_count?: number;
  // Agent mode
  mode?: string;
  // Startup wizard metadata
  wizard?: WizardMeta;
}

export interface StreamEndMessage {
  type: "stream_end";
  tokens: TokenSnapshot;
  wizard_complete?: boolean;
  mode?: string;
}

export interface TextChunk {
  type: "text";
  text: string;
}

export interface ThinkingChunk {
  type: "thinking";
}

export interface ToolStartChunk {
  type: "tool_start";
  tool_name: string;
  tool_input: Record<string, unknown>;
  tool_label?: string;
}

export interface ToolCallChunk {
  type: "tool_call";
  tool_name: string;
  tool_input?: Record<string, unknown>;
  duration?: number;
  result_summary?: string;
}

export interface ChoiceOptionMeta {
  /** Campaign label (shorthand or short description). */
  campaign?: string;
  /** Position in sequence, e.g. "1 of 21 · 0 done". */
  sequence?: string;
  /** Plan-item status when non-default (e.g. "in_progress"). */
  status?: string;
  /** Acquisition spec for session-resolution candidates. */
  spec?: {
    strain?: string;
    temperature_c?: number;
    num_slices?: number;
    exposure_ms?: number;
    interval_s?: number;
    stop_condition?: string;
    success_criteria?: string;
  };
}

export interface ChoiceOption {
  id: string;
  label: string;
  description?: string;
  disabled?: boolean;
  /** Extra metadata for richer rendering (used by session_resolution). */
  meta?: ChoiceOptionMeta;
}

export interface ChoiceRequest {
  type: "choice_request";
  choice_data: {
    _type: string;
    /** Optional sub-kind hint for richer rendering (e.g. "session_resolution"). */
    _kind?: string;
    question: string;
    options: ChoiceOption[];
    allow_multiple: boolean;
    default_id?: string;
  };
  request_id?: string;
}

export interface AppliedSpec {
  plan_item_id?: string;
  plan_item_title?: string;
  strain?: string;
  temperature_c?: number;
  num_slices?: number;
  exposure_ms?: number;
  interval_s?: number;
  stop_condition?: string;
  detectors?: string[] | null;
  success_criteria?: string;
  adaptive_intervals?: Record<string, unknown> | null;
}

export interface AppliedSpecMessage {
  type: "applied_spec";
  spec: AppliedSpec;
}

export interface StreamStartMessage {
  type: "stream_start";
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
  | StreamStartMessage
  | StreamEndMessage
  | TextChunk
  | ThinkingChunk
  | ToolStartChunk
  | ToolCallChunk
  | ChoiceRequest
  | AppliedSpecMessage
  | CommandResult
  | StateUpdate
  | NotificationMessage
  | ErrorMessage
  | BrowseResult
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

export interface ClientPong {
  type: "pong";
}

export interface CancelMessage {
  type: "cancel";
}

export type ClientMessage =
  | ChatMessage
  | ChoiceResponse
  | CommandMessage
  | BrowseRequest
  | CancelMessage
  | ClientPing
  | ClientPong;

// ---------------------------------------------------------------------------
// UI models
// ---------------------------------------------------------------------------

export type MessageRole = "user" | "agent" | "system" | "tool";

export interface ChatEntry {
  id: string;
  role: MessageRole;
  text: string;
  timestamp: number;
  toolName?: string;
  toolDuration?: number;
  toolSummary?: string;
  toolAnswer?: string;
  isStreaming?: boolean;
  isThinking?: boolean;
  isSelection?: boolean;
  /** Renders as a structured applied-spec panel instead of text. */
  isSpecCard?: boolean;
  specData?: AppliedSpec;
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
// Browser panel
// ---------------------------------------------------------------------------

export interface BrowserPlanItem {
  id: string;
  title: string;
  status: string;
  type?: string;
  claimed_by_hostname?: string;
}

export interface BrowserCampaign {
  id: string;
  shorthand: string;
  description: string;
  target: string;
  status: string;
  is_shared: boolean;
  total: number;
  completed: number;
  in_progress: number;
  subcampaigns: BrowserCampaign[];
  items: BrowserPlanItem[];
}

export interface BrowserPeer {
  instance_id: string;
  hostname: string;
  ip_address: string;
  viz_port: number;
  mode: string;
  embryo_count: number;
  is_trusted: boolean;
  tls_enabled: boolean;
  shared_campaigns: BrowserCampaign[];
}

export interface BrowseRequest {
  type: "browse";
  target: "campaigns" | "peers" | "peer_campaigns" | "peer_campaign_items";
  hostname?: string;
  campaign_id?: string;
}

export interface BrowseResult {
  type: "browse_result";
  target: "campaigns" | "peers" | "peer_campaigns" | "peer_campaign_items";
  data: unknown[];
  campaign_id?: string;
  hostname?: string;
}

// ---------------------------------------------------------------------------
// Theme
// ---------------------------------------------------------------------------

export interface ThemeColors {
  name: string;
  colorMode: "dark" | "light";
  primary: string;
  secondary: string;
  accent: string;
  user: string;
  agent: string;
  system: string;
  tool: string;
  success: string;
  warning: string;
  error: string;
  info: string;
  muted: string;
  // Background colors
  userMessageBg: string;
  surfaceBg: string;
}
