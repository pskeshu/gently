/**
 * Global keyboard shortcut hook.
 *
 * - Escape: cancel in-flight stream (interruptible thinking)
 */

import { useInput } from "ink";
import type { StoreApi } from "zustand/vanilla";
import type { TuiStore } from "../store.js";
import type { WsClient } from "../ws-client.js";

export function useKeyboard(
  store: StoreApi<TuiStore>,
  send: WsClient["send"],
): void {
  useInput((_input, key) => {
    // Escape — cancel in-flight stream
    if (key.escape) {
      const s = store.getState();
      if (s.isStreaming) {
        send({ type: "cancel" });
        s.cancelStream();
      }
    }
  });
}
