/**
 * Local slash-command registry.
 *
 * Some commands (e.g. /theme, /campaign) are handled entirely on the
 * client side — they manipulate the TUI store directly, or open a
 * built-in picker, without a server round-trip. Each local command
 * declares its name, description, and a handler that returns `true`
 * when it consumed the input. Anything not handled here falls through
 * to the server via `{ type: "command", command }`.
 *
 * To add a new local command:
 *   1. Push an entry below (or a separate `localCommands.<feature>.ts`
 *      that re-exports into the array).
 *   2. The dispatcher (`useSlashCommands`) iterates in order; first
 *      match wins. Order by specificity.
 */

import type { StoreApi } from "zustand/vanilla";
import type { TuiStore } from "./store.js";
import type { WsClient } from "./ws-client.js";
import { setTheme, listThemes } from "./theme.js";

export interface LocalCommandContext {
  /** Raw command text as typed, e.g. "/theme dark". */
  command: string;
  /** Args parsed by whitespace, excluding the command name. */
  args: string[];
  store: StoreApi<TuiStore>;
  send: WsClient["send"];
}

export interface LocalCommand {
  /** Primary name including the leading slash, e.g. "/theme". */
  name: string;
  /** Optional alias names (also with leading slash). */
  aliases?: string[];
  /** Short description (used by completion / help). */
  description: string;
  /**
   * Run the command. Return `true` to consume the input, `false` to
   * fall through (e.g. lookup failed and the server should try).
   */
  handler: (ctx: LocalCommandContext) => boolean;
}

/** /theme — switch themes (with picker if no arg given). */
const themeCommand: LocalCommand = {
  name: "/theme",
  description: "Switch UI theme",
  handler: ({ args, store, send }) => {
    const s = store.getState();
    if (args.length > 0) {
      const name = args[0]!;
      try {
        setTheme(name);
        const newTheme = listThemes()[name]!;
        s.setTheme(newTheme);
        s.addSystemMessage(`Theme changed to ${newTheme.name}`);
        // Also tell the server so any future render uses the same theme.
        send({ type: "command", command: `/theme ${name}` });
      } catch {
        const available = Object.keys(listThemes()).join(", ");
        s.addSystemMessage(`Unknown theme: '${name}'. Available: ${available}`);
      }
      return true;
    }
    // No arg — interactive picker. Enqueued with a "local:" requestId
    // so the picker callbacks know to handle it client-side.
    const themes = listThemes();
    const current = s.theme.name;
    const options = Object.entries(themes).map(([key, t]) => ({
      id: key,
      label: `${t.name}${t.name === current ? " (current)" : ""}`,
      description: t.colorMode === "dark" ? "Dark mode" : "Light mode",
    }));
    s.enqueueChoice(
      {
        type: "choice_request",
        choice_data: {
          _type: "single",
          question: "Choose a theme",
          options,
          allow_multiple: false,
        },
      },
      "local:theme",
    );
    return true;
  },
};

/** /campaign — open the campaign browser overlay. */
const campaignCommand: LocalCommand = {
  name: "/campaign",
  aliases: ["/campaigns"],
  description: "Browse campaigns and peers",
  handler: ({ store, send }) => {
    store.getState().setCampaignBrowserOpen(true);
    send({ type: "browse", target: "campaigns" });
    return true;
  },
};

export const LOCAL_COMMANDS: LocalCommand[] = [themeCommand, campaignCommand];

/**
 * Local-side completion source. Returns matches whose name (or alias)
 * has `prefix` (with leading slash) as a prefix. Server-supplied
 * commands are merged separately by the input component.
 */
export function getLocalCompletions(prefix: string): { name: string; description: string }[] {
  const p = prefix.toLowerCase();
  const seen = new Set<string>();
  const out: { name: string; description: string }[] = [];
  for (const cmd of LOCAL_COMMANDS) {
    for (const n of [cmd.name, ...(cmd.aliases ?? [])]) {
      if (!seen.has(n) && n.startsWith(p)) {
        seen.add(n);
        out.push({ name: n, description: cmd.description });
      }
    }
  }
  return out;
}

/**
 * Try every registered local command against the input. Returns true
 * if one consumed it; false otherwise (caller should forward to server).
 */
export function tryLocalCommand(
  input: string,
  store: StoreApi<TuiStore>,
  send: WsClient["send"],
): boolean {
  const trimmed = input.trim();
  if (!trimmed.startsWith("/")) return false;
  const parts = trimmed.split(/\s+/);
  const head = parts[0]!.toLowerCase();
  const args = parts.slice(1);
  for (const cmd of LOCAL_COMMANDS) {
    const names = [cmd.name, ...(cmd.aliases ?? [])];
    if (names.includes(head)) {
      return cmd.handler({ command: trimmed, args, store, send });
    }
  }
  return false;
}
