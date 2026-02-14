/**
 * React hook that owns the WebSocket lifecycle and dispatches
 * incoming server messages into the Zustand store.
 */

import { useEffect, useRef } from "react";
import { WsClient } from "../ws-client.js";
import type { TuiStore } from "../store.js";
import type { ServerMessage } from "../types.js";
import type { StoreApi } from "zustand/vanilla";

export function useWebSocket(
  url: string,
  store: StoreApi<TuiStore>,
): { send: WsClient["send"] } {
  const clientRef = useRef<WsClient | null>(null);

  useEffect(() => {
    const actions = store.getState();

    const client = new WsClient({
      url,
      onConnect: () => {
        // connected message will set full state
      },
      onDisconnect: () => {
        store.getState().setDisconnected();
      },
      onMessage: (msg: ServerMessage) => {
        const s = store.getState();
        switch (msg.type) {
          case "connected":
            s.setConnected({
              sessionId: msg.session_id,
              commands: msg.commands,
              version: msg.version ?? "",
              tokens: msg.tokens ?? { input_tokens: 0, output_tokens: 0, total_tokens: 0, api_calls: 0 },
              embryoCount: msg.embryo_count ?? 0,
              deviceConnected: msg.device_connected ?? false,
              samAvailable: msg.sam_available ?? false,
              offline: msg.offline ?? false,
              storePath: msg.store_path ?? "",
              vizUrl: msg.viz_url ?? null,
              logPath: msg.log_path ?? "",
              resumed: msg.resumed ?? false,
              wizard: msg.wizard,
              mode: msg.mode,
            });
            break;

          case "stream_end":
            s.updateTokens(msg.tokens);
            s.finishStreaming();
            if (msg.wizard_complete) {
              s.setWizardActive(false);
            }
            break;

          case "thinking":
            // Show thinking spinner (wizard uses this during LLM calls)
            s.showThinking();
            break;

          case "text":
            s.appendCopilotText(msg.text);
            break;

          case "tool_start":
            s.addToolStart(msg.tool_name, msg.tool_input);
            break;

          case "tool_call":
            s.addToolCall(msg.tool_name, msg.duration);
            break;

          case "choice_request": {
            const requestId =
              msg.request_id ?? `req_${Date.now()}`;
            s.setChoice(msg, requestId);
            break;
          }

          case "command_result": {
            if (msg.action === "quit") {
              process.exit(0);
            }
            if (msg.action === "clear") {
              s.clearMessages();
              break;
            }
            // Update copilot mode when /plan command returns it
            if (msg.content?.mode && typeof msg.content.mode === "string") {
              s.setCopilotMode(msg.content.mode);
            }
            const text = msg.error
              ? `Error: ${msg.error}`
              : formatCommandResult(msg.command, msg.content);
            s.addCommandResult(msg.command, text);
            break;
          }

          case "notification":
            s.setNotification({
              level: msg.level,
              title: msg.title,
              body: msg.body,
            });
            break;

          case "error":
            s.addSystemMessage(`Error: ${msg.error}`);
            s.finishStreaming();
            break;

          case "pong":
            break;

          default:
            // state_update, etc. — ignore for now
            break;
        }
      },
    });

    clientRef.current = client;
    store.getState().setConnecting();
    client.connect();

    return () => {
      client.close();
    };
  }, [url, store]);

  return {
    send: (msg) => clientRef.current?.send(msg),
  };
}

/**
 * Format a command result into a display string.
 */
function formatCommandResult(
  command: string,
  content?: Record<string, unknown>,
): string {
  if (!content) return `${command}: (no data)`;

  // /status
  if (content.session_id !== undefined) {
    const c = content as Record<string, unknown>;
    return [
      `Session: ${c.session_id}`,
      `Connected: ${c.connected}`,
      `Embryos: ${c.embryo_count}`,
      c.has_sam ? "SAM: available" : "SAM: not available",
    ].join("\n");
  }

  // /help or text-based results
  if (typeof content.text === "string") {
    return content.text;
  }

  // /embryos list
  if (Array.isArray(content.embryos)) {
    const list = content.embryos as Array<Record<string, string>>;
    if (list.length === 0) return "No embryos registered.";
    return list
      .map((e) => {
        let line = e.id ?? "?";
        if (e.nickname) line += ` (${e.nickname})`;
        return line;
      })
      .join("\n");
  }

  // /tokens
  if (content.total_tokens !== undefined) {
    return [
      `Input tokens:  ${content.input_tokens}`,
      `Output tokens: ${content.output_tokens}`,
      `Total tokens:  ${content.total_tokens}`,
    ].join("\n");
  }

  // /theme
  if (content.themes) {
    const themes = content.themes as Record<string, string>;
    const current = content.current as string;
    return Object.entries(themes)
      .map(([k, v]) => `  ${k}${v === current ? " (current)" : ""}`)
      .join("\n");
  }

  // /sessions
  if (Array.isArray(content.sessions)) {
    const sessions = content.sessions as Array<Record<string, unknown>>;
    if (sessions.length === 0) return "No saved sessions.";
    return sessions
      .map((s) => `${s.session_id} — ${s.embryo_count} embryos`)
      .join("\n");
  }

  // Fallback
  return JSON.stringify(content, null, 2);
}
