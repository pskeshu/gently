/**
 * Interactive campaign browser — navigable tree with actions.
 *
 * Opened by `/campaign` command. Renders in the chat area (replaces
 * CommandInput while active, same pattern as ChoicePicker).
 *
 * Navigation:
 *   Level 1: Root campaigns
 *   Level 2: Actions first, then subcampaigns or items
 *   Level 3: Items within expanded subcampaigns
 */

import React, { useState } from "react";
import { Box, Text, useInput } from "ink";
import type { BrowserCampaign, ThemeColors } from "../types.js";
import type { WsClient } from "../ws-client.js";

interface CampaignBrowserProps {
  campaigns: BrowserCampaign[];
  theme: ThemeColors;
  send: WsClient["send"];
  onClose: () => void;
}

// ── Helpers ──────────────────────────────────────────────────

function statusBadge(status: string, theme: ThemeColors) {
  if (status === "paused") return { char: "\u23F8", color: theme.warning };
  if (status === "completed") return { char: "\u2713", color: theme.muted };
  return { char: "\u25CF", color: theme.success };
}

function itemIcon(status: string, theme: ThemeColors) {
  if (status === "completed" || status === "done") return { char: "\u2713", color: theme.success };
  if (status === "running" || status === "in_progress") return { char: "\u25CF", color: theme.info };
  if (status === "blocked") return { char: "\u2717", color: theme.error };
  if (status === "skipped") return { char: "\u2013", color: theme.muted };
  return { char: "\u25CB", color: theme.muted };
}

function typeTag(type?: string): string {
  if (!type) return "";
  const map: Record<string, string> = {
    imaging: "[img]", bench: "[lab]", genetics: "[gen]",
    analysis: "[ana]", decision_point: "[dec]",
  };
  return map[type] || "";
}

function getActions(c: BrowserCampaign): string[] {
  const actions: string[] = [];
  actions.push(c.is_shared ? "[Unshare]" : "[Share on mesh]");
  if (c.status !== "completed") {
    actions.push(c.status === "active" ? "[Pause campaign]" : "[Resume campaign]");
  }
  return actions;
}

function getActionCommand(c: BrowserCampaign, actionIdx: number): string | null {
  const commands: string[] = [];
  commands.push(c.is_shared ? `/campaign unshare ${c.id}` : `/campaign share ${c.id}`);
  if (c.status !== "completed") {
    commands.push(c.status === "active" ? `/campaign pause ${c.id}` : `/campaign resume ${c.id}`);
  }
  return commands[actionIdx] ?? null;
}

// ── Component ────────────────────────────────────────────────

export function CampaignBrowser({ campaigns, theme, send, onClose }: CampaignBrowserProps) {
  const [cursor, setCursor] = useState(0);
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [subCursor, setSubCursor] = useState(0);
  const [expandedSubId, setExpandedSubId] = useState<string | null>(null);
  const [itemCursor, setItemCursor] = useState(0);

  // Resolve expanded campaign
  const expandedCamp = expandedId ? campaigns.find((c) => c.id === expandedId) : null;
  const actionCount = expandedCamp ? getActions(expandedCamp).length : 0;
  const hasSubcampaigns = expandedCamp ? expandedCamp.subcampaigns.length > 0 : false;

  // Level 2: actions first, then content (subcampaigns or items)
  // Layout: [action0, action1, ..., content0, content1, ...]
  const level2Max = (() => {
    if (!expandedCamp) return -1;
    const contentCount = hasSubcampaigns
      ? expandedCamp.subcampaigns.length
      : expandedCamp.items.length;
    return actionCount + contentCount - 1;
  })();

  // Expanded subcampaign (level 3 — items only, no actions)
  const expandedSub = expandedSubId && expandedCamp
    ? expandedCamp.subcampaigns.find((s) => s.id === expandedSubId)
    : null;

  const level3Max = expandedSub ? expandedSub.items.length - 1 : -1;

  const dispatchAction = (campaign: BrowserCampaign, actionIdx: number) => {
    const cmd = getActionCommand(campaign, actionIdx);
    if (cmd) {
      send({ type: "command", command: cmd });
      setTimeout(() => send({ type: "browse", target: "campaigns" }), 300);
    }
  };

  useInput((_input, key) => {
    if (key.escape) {
      if (expandedSubId) {
        setExpandedSubId(null);
        setItemCursor(0);
      } else if (expandedId) {
        setExpandedId(null);
        setSubCursor(0);
      } else {
        onClose();
      }
      return;
    }

    if (key.upArrow) {
      if (expandedSubId) {
        if (itemCursor > 0) setItemCursor((c) => c - 1);
        else { setExpandedSubId(null); setItemCursor(0); }
      } else if (expandedId) {
        if (subCursor > 0) setSubCursor((c) => c - 1);
        else { setExpandedId(null); setSubCursor(0); }
      } else {
        if (cursor > 0) setCursor((c) => c - 1);
      }
      return;
    }

    if (key.downArrow) {
      if (expandedSubId) {
        setItemCursor((c) => Math.min(level3Max, c + 1));
      } else if (expandedId) {
        setSubCursor((c) => Math.min(level2Max, c + 1));
      } else {
        setCursor((c) => Math.min(campaigns.length - 1, c + 1));
      }
      return;
    }

    if (key.return) {
      // Level 3: read-only items in subcampaign
      if (expandedSubId) return;

      // Level 2: actions first, then content
      if (expandedId && expandedCamp) {
        if (subCursor < actionCount) {
          // Action row
          dispatchAction(expandedCamp, subCursor);
        } else if (hasSubcampaigns) {
          // Expand subcampaign
          const sc = expandedCamp.subcampaigns[subCursor - actionCount];
          if (sc) { setExpandedSubId(sc.id); setItemCursor(0); }
        }
        // Items are read-only — no action on Enter
        return;
      }

      // Level 1: expand/collapse
      const c = campaigns[cursor];
      if (!c) return;
      if (expandedId === c.id) {
        setExpandedId(null);
        setSubCursor(0);
        setExpandedSubId(null);
        setItemCursor(0);
      } else {
        setExpandedId(c.id);
        setSubCursor(0);
        setExpandedSubId(null);
        setItemCursor(0);
      }
    }
  });

  // ── Rendering ──────────────────────────────────────────────

  const hintText = expandedSubId
    ? "\u2191/\u2193 navigate \u00B7 Esc back"
    : expandedId
      ? "\u2191/\u2193 navigate \u00B7 Enter act \u00B7 Esc back"
      : "\u2191/\u2193 navigate \u00B7 Enter expand \u00B7 Esc close";

  return (
    <Box
      flexDirection="column"
      borderStyle="round"
      borderColor={theme.info}
      paddingX={1}
      marginBottom={1}
    >
      <Text bold color={theme.info}>Campaigns</Text>
      {campaigns.length === 0 ? (
        <Text color={theme.muted}>No campaigns</Text>
      ) : (
        <Box flexDirection="column" marginTop={1}>
          {campaigns.map((c, i) => {
            const isCursor = i === cursor && !expandedId;
            const isExpanded = expandedId === c.id;
            const marker = isExpanded ? "\u25BC" : isCursor ? "\u25B6" : " ";
            const badge = statusBadge(c.status, theme);

            return (
              <Box key={c.id} flexDirection="column">
                <Text color={isCursor || isExpanded ? theme.info : undefined} bold={isCursor}>
                  {`${marker} `}
                  <Text color={badge.color}>{badge.char}</Text>
                  {` ${c.shorthand || c.id.slice(0, 8)}`}
                  {c.is_shared ? <Text color={theme.success}>{" \u27C6"}</Text> : null}
                  <Text color={theme.muted}>{` (${c.completed}/${c.total})`}</Text>
                </Text>
                {isExpanded && c.description ? (
                  <Text color={theme.muted}>
                    {"    "}{c.description.length > 70 ? c.description.slice(0, 67) + "..." : c.description}
                  </Text>
                ) : null}
                {isExpanded && c.target ? (
                  <Text color={theme.muted}>{"    "}Target: {c.target}</Text>
                ) : null}
                {isExpanded ? (
                  <>
                    {renderActions(c, subCursor, "    ", theme)}
                    {c.subcampaigns.length > 0
                      ? renderSubcampaigns(c.subcampaigns, subCursor - getActions(c).length, expandedSubId, itemCursor, theme)
                      : renderItems(c.items, subCursor - getActions(c).length, "    ", theme)}
                  </>
                ) : null}
              </Box>
            );
          })}
        </Box>
      )}
      <Text color={theme.muted}>{hintText}</Text>
    </Box>
  );
}

// ── Sub-renderers ────────────────────────────────────────────

/** Render action rows (share/unshare, pause/resume) — shown first when expanded. */
function renderActions(
  campaign: BrowserCampaign,
  activeCursor: number,
  indent: string,
  theme: ThemeColors,
) {
  const actions = getActions(campaign);
  return (
    <>
      {actions.map((label, idx) => {
        const isActive = idx === activeCursor;
        return (
          <Text
            key={label}
            color={isActive ? theme.success : theme.muted}
            bold={isActive}
          >
            {indent}{isActive ? "\u25B6 " : "  "}{label}
          </Text>
        );
      })}
    </>
  );
}

/** Render subcampaigns list. */
function renderSubcampaigns(
  subcampaigns: BrowserCampaign[],
  contentCursor: number,
  expandedSubId: string | null,
  itemCursor: number,
  theme: ThemeColors,
) {
  return subcampaigns.map((s, j) => {
    const isSub = j === contentCursor && !expandedSubId;
    const isExpanded = expandedSubId === s.id;
    const marker = isExpanded ? "\u25BC" : isSub ? "\u25B6" : " ";
    const badge = statusBadge(s.status, theme);

    return (
      <Box key={s.id} flexDirection="column">
        <Text color={isSub || isExpanded ? theme.info : theme.muted} bold={isSub}>
          {"    "}{marker}{" "}
          <Text color={badge.color}>{badge.char}</Text>
          {` ${s.shorthand || s.id.slice(0, 8)}`}
          {s.is_shared ? <Text color={theme.success}>{" \u27C6"}</Text> : null}
          <Text color={theme.muted}>{` (${s.completed}/${s.total})`}</Text>
        </Text>
        {isExpanded ? renderItems(s.items, itemCursor, "        ", theme) : null}
      </Box>
    );
  });
}

/** Render plan items list. */
function renderItems(
  items: BrowserCampaign["items"],
  activeCursor: number,
  indent: string,
  theme: ThemeColors,
) {
  return (
    <>
      {items.map((item, k) => {
        const isActive = k === activeCursor;
        const icon = itemIcon(item.status, theme);
        const tt = typeTag(item.type);
        return (
          <Text key={item.id} color={isActive ? theme.info : theme.muted}>
            {indent}
            <Text color={icon.color}>{icon.char}</Text>
            {tt ? <Text color={theme.muted}> {tt}</Text> : null}
            {` ${item.title}`}
            {item.claimed_by_hostname ? (
              <Text color={theme.muted}> @{item.claimed_by_hostname}</Text>
            ) : null}
          </Text>
        );
      })}
    </>
  );
}
