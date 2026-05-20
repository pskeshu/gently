/**
 * Choice-picker callbacks (`onSelect`, `onCancel`) for the head of the
 * choice queue. Knows about the "local:" requestId convention so local
 * pickers (e.g. /theme) finish without a server response.
 */

import { useCallback, useMemo } from "react";
import { setTheme, listThemes } from "../theme.js";
import { useTuiStoreApi } from "../context.js";
import type { WsClient } from "../ws-client.js";

export interface ChoiceCallbacks {
  onSelect: (selected: string) => void;
  onCancel: () => void;
}

export function useChoiceCallbacks(send: WsClient["send"]): ChoiceCallbacks {
  const store = useTuiStoreApi();

  const onSelect = useCallback(
    (selected: string) => {
      const s = store.getState();
      const head = s.choiceQueue[0];
      if (!head) return;
      const { request, requestId } = head;

      // Local pickers handle their own follow-up; no server response.
      if (requestId === "local:theme") {
        s.dismissCurrentChoice();
        try {
          setTheme(selected);
          const newTheme = listThemes()[selected]!;
          s.setTheme(newTheme);
          s.addSystemMessage(`Theme changed to ${newTheme.name}`);
          send({ type: "command", command: `/theme ${selected}` });
        } catch {
          const available = Object.keys(listThemes()).join(", ");
          s.addSystemMessage(
            `Unknown theme: '${selected}'. Available: ${available}`,
          );
        }
        return;
      }

      const option = request.choice_data.options.find((o) => o.id === selected);
      const label = option?.label ?? selected;
      s.addUserSelection(label);
      send({ type: "choice_response", request_id: requestId, selected });
      s.dismissCurrentChoice();
    },
    [send, store],
  );

  const onCancel = useCallback(() => {
    const s = store.getState();
    const head = s.choiceQueue[0];
    if (!head) return;
    const { requestId } = head;
    if (requestId.startsWith("local:")) {
      s.dismissCurrentChoice();
      return;
    }
    send({ type: "choice_response", request_id: requestId, selected: "" });
    s.dismissCurrentChoice();
    s.addSystemMessage("(cancelled)");
    s.finishStreaming();
  }, [send, store]);

  return useMemo(() => ({ onSelect, onCancel }), [onSelect, onCancel]);
}
