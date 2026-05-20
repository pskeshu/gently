/**
 * Root application component — Claude Code-style persistent layout.
 *
 * Plugs concrete content into the <Layout> primitive's named slots.
 * All logic lives in dedicated hooks:
 *   - useWebSocket: WS lifecycle and server-message dispatch
 *   - GlobalKeybindings: Esc/Shift+Tab handler (gated by isActive when a modal opens)
 *   - useSendMessage: routes user input through local commands or to the server
 *   - useMessageQueueDrain: pops the next queued chat message when streaming finishes
 *   - useChoiceCallbacks: choice-picker onSelect/onCancel (handles local: pickers)
 *
 * App itself just subscribes to the few slices that decide which slots
 * are populated, and otherwise stays declarative.
 */

import React from "react";
import { Box, Static } from "ink";
import { useWebSocket } from "../hooks/useWebSocket.js";
import { useSendMessage } from "../hooks/useSendMessage.js";
import { useMessageQueueDrain } from "../hooks/useMessageQueueDrain.js";
import { useChoiceCallbacks } from "../hooks/useChoiceCallbacks.js";
import { useTuiSelector, useTuiStoreApi } from "../context.js";

import { GlobalKeybindings } from "./keybindings/GlobalKeybindings.js";
import { Layout } from "./Layout.js";
import { MessageBubble } from "./MessageBubble.js";
import { ActiveMessage } from "./ChatPane.js";
import { CampaignBrowser } from "./CampaignBrowser.js";
import { ChoicePicker } from "./ChoicePicker.js";
import { CommandInput } from "./CommandInput.js";
import { StatusBar } from "./StatusBar.js";
import { WelcomeScreen } from "./WelcomeScreen.js";

interface AppProps {
  wsUrl: string;
}

export function App({ wsUrl }: AppProps) {
  const store = useTuiStoreApi();

  const { send } = useWebSocket(wsUrl);

  const sendMessage = useSendMessage(send);
  useMessageQueueDrain(send);
  const choiceCallbacks = useChoiceCallbacks(send);

  // ── Slice subscriptions: just what App needs for layout routing ──
  const completedMessages = useTuiSelector((s) => s.completedMessages);
  const hasActive = useTuiSelector((s) => s.activeMessage !== null);
  const connectionStatus = useTuiSelector((s) => s.connectionStatus);
  const wizardActive = useTuiSelector((s) => s.wizardActive);
  const choicePending = useTuiSelector((s) => s.choiceQueue.length > 0);
  const campaignBrowserOpen = useTuiSelector((s) => s.campaignBrowserOpen);
  // Theme is needed for the Static block — pulled here so a theme
  // change re-runs the Static render for new messages. Ink's Static
  // caches already-rendered items, matching prior behavior.
  const theme = useTuiSelector((s) => s.theme);

  const showWelcome =
    connectionStatus === "connected" &&
    completedMessages.length === 0 &&
    !hasActive &&
    !wizardActive;

  // Modal slot. Listed in priority order: a server-issued choice
  // request preempts a campaign browse.
  const modal = choicePending ? (
    <ChoicePicker {...choiceCallbacks} />
  ) : campaignBrowserOpen ? (
    <CampaignBrowser
      send={send}
      onClose={() => store.getState().setCampaignBrowserOpen(false)}
    />
  ) : null;

  return (
    <Layout
      header={<GlobalKeybindings send={send} isActive={!modal} />}
      welcome={showWelcome ? <WelcomeScreen /> : null}
      transcript={
        <>
          <Static items={completedMessages}>
            {(entry) => (
              <Box key={entry.id} flexDirection="column">
                <MessageBubble entry={entry} theme={theme} />
              </Box>
            )}
          </Static>
          <ActiveMessage />
        </>
      }
      modal={modal}
      bottom={<CommandInput onSubmit={sendMessage} />}
      statusBar={<StatusBar send={send} />}
    />
  );
}
