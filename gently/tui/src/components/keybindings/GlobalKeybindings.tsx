/**
 * Global keybinding handler — claims Esc and Shift+Tab.
 *
 * Mounted as a sibling inside the React tree; only listens to key
 * events when `isActive` is true. When a modal (choice picker, campaign
 * browser) is open, App passes `isActive={false}` and the modal's own
 * useInput handlers own the keys — preventing Esc from both cancelling
 * the picker AND cancelling the stream behind it.
 *
 * Mirrors src's pattern of layered handler components (each with its
 * own isActive gate) rather than a single shared useInput.
 */

import React from "react";
import { useInput } from "ink";
import { useTuiStoreApi } from "../../context.js";
import type { WsClient } from "../../ws-client.js";

interface Props {
  send: WsClient["send"];
  isActive: boolean;
}

export function GlobalKeybindings({ send, isActive }: Props) {
  const store = useTuiStoreApi();
  useInput(
    (_input, key) => {
      // Esc — cancel in-flight stream.
      if (key.escape) {
        const s = store.getState();
        if (s.isStreaming) {
          send({ type: "cancel" });
          s.cancelStream();
        }
        return;
      }
      // Shift+Tab — toggle plan/run mode.
      if (key.tab && key.shift) {
        const s = store.getState();
        if (s.isStreaming) return;
        const cmd = s.agentMode === "plan" ? "/plan exit" : "/plan";
        send({ type: "command", command: cmd });
      }
    },
    { isActive },
  );
  return null;
}
