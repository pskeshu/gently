/**
 * When the agent finishes streaming, pop the next queued message off
 * the user's pending-message queue and send it.
 */

import { useEffect } from "react";
import { useTuiSelector, useTuiStoreApi } from "../context.js";
import type { WsClient } from "../ws-client.js";

export function useMessageQueueDrain(send: WsClient["send"]): void {
  const store = useTuiStoreApi();
  const isStreaming = useTuiSelector((s) => s.isStreaming);
  useEffect(() => {
    if (isStreaming) return;
    const next = store.getState().dequeueMessage();
    if (!next) return;
    store.getState().addUserMessage(next);
    send({ type: "chat", text: next });
  }, [isStreaming, send, store]);
}
