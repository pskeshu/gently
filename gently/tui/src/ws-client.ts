/**
 * WebSocket client with auto-reconnect and typed message handling.
 *
 * Connects to the Python viz server at /ws/copilot and dispatches
 * incoming messages to a callback.
 */

import WebSocket from "ws";
import type { ServerMessage, ClientMessage } from "./types.js";

export interface WsClientOptions {
  url: string;
  onMessage: (msg: ServerMessage) => void;
  onConnect: () => void;
  onDisconnect: () => void;
  /** Max reconnect attempts (0 = infinite). Default: 0 */
  maxRetries?: number;
}

export class WsClient {
  private ws: WebSocket | null = null;
  private url: string;
  private onMessage: (msg: ServerMessage) => void;
  private onConnect: () => void;
  private onDisconnect: () => void;
  private maxRetries: number;
  private retryCount = 0;
  private retryTimer: ReturnType<typeof setTimeout> | null = null;
  private intentionallyClosed = false;

  constructor(opts: WsClientOptions) {
    this.url = opts.url;
    this.onMessage = opts.onMessage;
    this.onConnect = opts.onConnect;
    this.onDisconnect = opts.onDisconnect;
    this.maxRetries = opts.maxRetries ?? 0;
  }

  connect(): void {
    this.intentionallyClosed = false;
    this._connect();
  }

  private _connect(): void {
    try {
      this.ws = new WebSocket(this.url);
    } catch {
      this._scheduleReconnect();
      return;
    }

    this.ws.on("open", () => {
      this.retryCount = 0;
      this.onConnect();
    });

    this.ws.on("message", (data) => {
      try {
        const msg = JSON.parse(data.toString()) as ServerMessage;
        // Auto-respond to pings
        if (msg.type === "ping") {
          this.send({ type: "ping" });
          return;
        }
        this.onMessage(msg);
      } catch {
        // Ignore unparseable messages
      }
    });

    this.ws.on("close", () => {
      this.onDisconnect();
      if (!this.intentionallyClosed) {
        this._scheduleReconnect();
      }
    });

    this.ws.on("error", () => {
      // Error is followed by close — reconnect handled there
    });
  }

  send(msg: ClientMessage): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(msg));
    }
  }

  close(): void {
    this.intentionallyClosed = true;
    if (this.retryTimer) {
      clearTimeout(this.retryTimer);
      this.retryTimer = null;
    }
    this.ws?.close();
  }

  private _scheduleReconnect(): void {
    if (this.intentionallyClosed) return;
    if (this.maxRetries > 0 && this.retryCount >= this.maxRetries) return;

    // Exponential backoff: 500ms, 1s, 2s, 4s … capped at 10s
    const delay = Math.min(500 * 2 ** this.retryCount, 10_000);
    this.retryCount++;

    this.retryTimer = setTimeout(() => {
      this.retryTimer = null;
      this._connect();
    }, delay);
  }
}
