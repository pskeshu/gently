/**
 * Returns the unified send-or-queue function. Slash commands route
 * through the local-command registry first, then fall through to the
 * server. Chat text gets queued if the agent is still streaming.
 */

import { useCallback } from "react";
import { isSlashCommand } from "../commands.js";
import { tryLocalCommand } from "../localCommands.js";
import { useTuiStoreApi } from "../context.js";
import type { WsClient } from "../ws-client.js";

export function useSendMessage(send: WsClient["send"]): (text: string) => void {
  const store = useTuiStoreApi();
  return useCallback(
    (text: string) => {
      if (isSlashCommand(text)) {
        if (tryLocalCommand(text, store, send)) return;
        send({ type: "command", command: text });
        return;
      }
      const s = store.getState();
      if (s.isStreaming) {
        s.enqueueMessage(text);
      } else {
        s.addUserMessage(text);
        send({ type: "chat", text });
      }
    },
    [send, store],
  );
}
