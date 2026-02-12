/**
 * Client-side slash command helpers.
 *
 * Handles parsing, matching, and autocomplete for slash commands.
 * The command registry comes from the server on connect.
 */

import type { CommandDef } from "./types.js";

/**
 * Check if the input looks like a slash command.
 */
export function isSlashCommand(input: string): boolean {
  return input.trimStart().startsWith("/");
}

/**
 * Get autocomplete suggestions for a partial input.
 */
export function getCompletions(
  input: string,
  commands: CommandDef[],
): CommandDef[] {
  const trimmed = input.trimStart().toLowerCase();
  if (!trimmed.startsWith("/")) return [];

  return commands.filter(
    (cmd) =>
      cmd.name.startsWith(trimmed) ||
      cmd.aliases.some((a) => a.startsWith(trimmed)),
  );
}

/**
 * Find the best matching command for a full input string.
 */
export function matchCommand(
  input: string,
  commands: CommandDef[],
): CommandDef | undefined {
  const name = input.trimStart().toLowerCase().split(/\s/)[0] ?? "";
  return commands.find(
    (cmd) => cmd.name === name || cmd.aliases.includes(name),
  );
}
