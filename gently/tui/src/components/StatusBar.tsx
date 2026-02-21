/**
 * Persistent bottom status bar — always visible below the input.
 *
 * Layout (normal):
 *   run · ● connected · 3 embryos · 2 campaigns · solo · 12.4k tokens      gently v0.4.0
 *
 * When `browserOpen` is true, the "campaigns" and "peers" segments
 * become navigable (highlighted).  Arrow-down from the input puts
 * the cursor here; left/right switches between them, Enter expands
 * a short detail list inline below the status line, Esc returns to input.
 */

import React, { useEffect, useState } from "react";
import { Box, Text, useInput } from "ink";
import type {
  BrowserCampaign,
  BrowserPeer,
  BrowserPlanItem,
  ThemeColors,
  TokenSnapshot,
} from "../types.js";
import type { WsClient } from "../ws-client.js";

interface StatusBarProps {
  theme: ThemeColors;
  version: string;
  sessionId: string;
  deviceConnected: boolean;
  offline: boolean;
  embryoCount: number;
  campaignCount: number;
  peerCount: number;
  tokens: TokenSnapshot;
  notification: { level: string; title: string; body?: string } | null;
  onClearNotification: () => void;
  wizardActive?: boolean;
  copilotMode?: string;
  // Browser mode
  browserOpen: boolean;
  onCloseBrowser: () => void;
  send: WsClient["send"];
  campaigns: BrowserCampaign[];
  peers: BrowserPeer[];
  peerCampaignItems: BrowserPlanItem[];
  peerCampaignMeta: { hostname: string; campaign_id: string } | null;
  onClearPeerCampaignItems: () => void;
}

function formatTokens(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}k`;
  return String(n);
}

function ModeBadge({ mode, theme }: { mode: string; theme: ThemeColors }) {
  if (mode === "plan") {
    return <Text color={theme.info} bold>plan</Text>;
  }
  return <Text color={theme.muted}>run</Text>;
}

type BrowserView = "campaigns" | "peers";

export function StatusBar({
  theme,
  version,
  sessionId,
  deviceConnected,
  offline,
  embryoCount,
  campaignCount,
  peerCount,
  tokens,
  notification,
  onClearNotification,
  wizardActive,
  copilotMode = "run",
  browserOpen,
  onCloseBrowser,
  send,
  campaigns,
  peers,
  peerCampaignItems,
  peerCampaignMeta,
  onClearPeerCampaignItems,
}: StatusBarProps) {
  // Auto-dismiss notifications after 5 seconds
  const [showNotification, setShowNotification] = useState(false);

  useEffect(() => {
    if (!notification) {
      setShowNotification(false);
      return;
    }
    setShowNotification(true);
    const timer = setTimeout(() => {
      setShowNotification(false);
      onClearNotification();
    }, 5000);
    return () => clearTimeout(timer);
  }, [notification, onClearNotification]);

  // ── Browser navigation state ─────────────────────────────
  const views: BrowserView[] = [];
  if (campaignCount > 0) views.push("campaigns");
  views.push("peers");

  const [activeView, setActiveView] = useState<BrowserView>(views[0] || "peers");
  const [cursor, setCursor] = useState(0);
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [subCursor, setSubCursor] = useState(0);
  // Third level: expanded campaign within a peer
  const [expandedCampaignId, setExpandedCampaignId] = useState<string | null>(null);
  const [itemCursor, setItemCursor] = useState(0);

  // Reset and fetch when browser opens/closes
  useEffect(() => {
    if (browserOpen) {
      const v = campaignCount > 0 ? "campaigns" : "peers";
      setActiveView(v);
      setCursor(0);
      setExpandedId(null);
      setSubCursor(0);
      setExpandedCampaignId(null);
      setItemCursor(0);
      onClearPeerCampaignItems();
      if (campaignCount > 0) send({ type: "browse", target: "campaigns" });
      send({ type: "browse", target: "peers" });
    }
  }, [browserOpen]);

  const listItems = activeView === "campaigns" ? campaigns : peers;

  // Sub-items for expanded item
  let subItems: unknown[] = [];
  if (expandedId) {
    if (activeView === "campaigns") {
      subItems = campaigns.find((c) => c.id === expandedId)?.items ?? [];
    } else {
      subItems = peers.find((p) => p.instance_id === expandedId)?.shared_campaigns ?? [];
    }
  }

  // Third-level items: campaign items within a peer's campaign (+ Join action)
  // Items count = peerCampaignItems.length + 1 (for Join action)
  const hasExpandedCampaign = activeView === "peers" && expandedId && expandedCampaignId;
  const thirdLevelCount = hasExpandedCampaign ? peerCampaignItems.length + 1 : 0;

  // ── Keyboard handling ────────────────────────────────────
  useInput((_input, key) => {
    if (!browserOpen) return;

    if (key.escape) {
      if (expandedCampaignId) {
        // Close third level → back to campaign list
        setExpandedCampaignId(null);
        setItemCursor(0);
        onClearPeerCampaignItems();
      } else if (expandedId) {
        setExpandedId(null);
        setSubCursor(0);
      } else {
        onCloseBrowser();
      }
      return;
    }

    if (key.leftArrow && !expandedId) {
      const idx = views.indexOf(activeView);
      if (idx > 0) {
        setActiveView(views[idx - 1]!);
        setCursor(0);
      }
      return;
    }

    if (key.rightArrow && !expandedId) {
      const idx = views.indexOf(activeView);
      if (idx < views.length - 1) {
        setActiveView(views[idx + 1]!);
        setCursor(0);
      }
      return;
    }

    if (key.upArrow) {
      if (expandedCampaignId) {
        // Third level navigation
        if (itemCursor > 0) {
          setItemCursor((c) => c - 1);
        } else {
          // Back to campaign list
          setExpandedCampaignId(null);
          setItemCursor(0);
          onClearPeerCampaignItems();
        }
      } else if (expandedId) {
        if (subCursor > 0) {
          setSubCursor((c) => c - 1);
        } else {
          setExpandedId(null);
          setSubCursor(0);
        }
      } else {
        if (cursor > 0) {
          setCursor((c) => c - 1);
        } else {
          onCloseBrowser();
        }
      }
      return;
    }

    if (key.downArrow) {
      if (expandedCampaignId) {
        // Third level: items + [Join campaign]
        setItemCursor((c) => Math.min(thirdLevelCount - 1, c + 1));
      } else if (expandedId) {
        setSubCursor((c) => Math.min(subItems.length - 1, c + 1));
      } else {
        setCursor((c) => Math.min(listItems.length - 1, c + 1));
      }
      return;
    }

    if (key.return) {
      // Third level: action on campaign item or Join
      if (expandedCampaignId) {
        if (itemCursor === peerCampaignItems.length) {
          // [Join campaign] action
          const peer = peers.find((p) => p.instance_id === expandedId);
          if (peer) {
            send({
              type: "command",
              command: `/join-campaign ${peer.hostname} ${expandedCampaignId}`,
            });
            onCloseBrowser();
          }
        }
        // Items themselves are read-only, no action on Enter
        return;
      }

      if (expandedId) {
        // Second level: expand campaign to show items
        if (activeView === "peers") {
          const peer = peers.find((p) => p.instance_id === expandedId);
          const camp = subItems[subCursor] as BrowserCampaign | undefined;
          if (peer && camp) {
            setExpandedCampaignId(camp.id);
            setItemCursor(0);
            send({
              type: "browse",
              target: "peer_campaign_items",
              hostname: peer.hostname,
              campaign_id: camp.id,
            });
          }
        }
        return;
      }

      // First level: expand / collapse list item
      const item = listItems[cursor];
      if (!item) return;

      if (activeView === "campaigns") {
        const c = item as BrowserCampaign;
        if (expandedId === c.id) {
          setExpandedId(null);
        } else {
          setExpandedId(c.id);
          setSubCursor(0);
        }
      } else {
        const p = item as BrowserPeer;
        if (!p.is_trusted) {
          // Unpaired peer — initiate pairing
          send({ type: "command", command: `/pair ${p.hostname}` });
          onCloseBrowser();
          return;
        }
        if (expandedId === p.instance_id) {
          setExpandedId(null);
        } else {
          setExpandedId(p.instance_id);
          setSubCursor(0);
          setExpandedCampaignId(null);
          setItemCursor(0);
          onClearPeerCampaignItems();
          send({ type: "browse", target: "peer_campaigns", hostname: p.hostname });
        }
      }
    }
  });

  // ── Rendering helpers ────────────────────────────────────

  const sep = <Text color={theme.muted}> · </Text>;
  const campActive = browserOpen && activeView === "campaigns";
  const peerActive = browserOpen && activeView === "peers";

  // Notification overlay
  if (showNotification && notification) {
    const levelColor =
      notification.level === "error"
        ? theme.error
        : notification.level === "warning"
          ? theme.warning
          : notification.level === "success"
            ? theme.success
            : theme.info;

    return (
      <Box justifyContent="space-between">
        <Box>
          <ModeBadge mode={copilotMode} theme={theme} />
          {sep}
          <Text color={levelColor} bold>
            {notification.title}
          </Text>
          {notification.body ? (
            <Text color={theme.muted}> — {notification.body}</Text>
          ) : null}
        </Box>
        {version ? (
          <Text color={theme.muted}>gently v{version}</Text>
        ) : null}
      </Box>
    );
  }

  // Wizard active indicator
  if (wizardActive) {
    return (
      <Box justifyContent="space-between">
        <Box>
          <ModeBadge mode={copilotMode} theme={theme} />
          {sep}
          <Text color={theme.info} bold>setting up</Text>
          <Text color={theme.muted}> — answer a few questions to get started</Text>
        </Box>
        {version ? (
          <Text color={theme.muted}>gently v{version}</Text>
        ) : null}
      </Box>
    );
  }

  // Device status indicator
  let deviceDot: { char: string; color: string; label: string };
  if (offline) {
    deviceDot = { char: "○", color: theme.warning, label: "offline" };
  } else if (deviceConnected) {
    deviceDot = { char: "●", color: theme.success, label: "connected" };
  } else {
    deviceDot = { char: "●", color: theme.error, label: "disconnected" };
  }

  // ── Main render ──────────────────────────────────────────

  return (
    <Box flexDirection="column">
      {/* Expanded list above the status line */}
      {browserOpen ? (
        <Box flexDirection="column">
          {activeView === "campaigns"
            ? renderCampaignList(campaigns, cursor, expandedId, subCursor, theme)
            : renderPeerList(peers, cursor, expandedId, subCursor, subItems as BrowserCampaign[], expandedCampaignId, itemCursor, peerCampaignItems, theme)}
        </Box>
      ) : null}

      {/* Status line (always at the bottom) */}
      <Box justifyContent="space-between">
        <Box>
          <ModeBadge mode={copilotMode} theme={theme} />
          {sep}
          <Text color={deviceDot.color}>{deviceDot.char}</Text>
          <Text color={theme.muted}> {deviceDot.label}</Text>

          {sessionId ? (
            <>
              {sep}
              <Text color={theme.muted}>{sessionId.slice(0, 8)}</Text>
            </>
          ) : null}

          {embryoCount > 0 ? (
            <>
              {sep}
              <Text color={theme.muted}>
                {embryoCount} embryo{embryoCount !== 1 ? "s" : ""}
              </Text>
            </>
          ) : null}

          {campaignCount > 0 ? (
            <>
              {sep}
              <Text color={campActive ? theme.info : theme.muted} bold={campActive}>
                {`${campActive ? "▼ " : ""}${campaignCount} campaign${campaignCount !== 1 ? "s" : ""}`}
              </Text>
            </>
          ) : null}

          {sep}
          <Text color={peerActive ? theme.info : theme.muted} bold={peerActive}>
            {`${peerActive ? "▼ " : ""}${peerCount > 0 ? `${peerCount} peer${peerCount !== 1 ? "s" : ""}` : "solo"}`}
          </Text>

          {tokens.total_tokens > 0 ? (
            <>
              {sep}
              <Text color={theme.muted}>
                {formatTokens(tokens.total_tokens)} tokens
              </Text>
            </>
          ) : null}
        </Box>

        {/* Right side: hints when browsing, version otherwise */}
        {browserOpen ? (
          <Text color={theme.muted}>
            {views.length > 1 ? "←/→ " : ""}↑/↓ · Esc
          </Text>
        ) : version ? (
          <Text color={theme.muted}>gently v{version}</Text>
        ) : null}
      </Box>
    </Box>
  );
}

// ── Campaign list rendering ──────────────────────────────────

function renderCampaignList(
  campaigns: BrowserCampaign[],
  cursor: number,
  expandedId: string | null,
  subCursor: number,
  theme: ThemeColors,
) {
  if (campaigns.length === 0) {
    return <Text color={theme.muted}>  No campaigns</Text>;
  }

  return campaigns.map((c, i) => {
    const isCursor = i === cursor && !expandedId;
    const isExpanded = expandedId === c.id;
    const done = c.completed >= c.total && c.total > 0;
    const marker = isExpanded ? "▼" : isCursor ? "▶" : " ";

    return (
      <Box key={c.id} flexDirection="column">
        <Text
          color={isCursor || isExpanded ? theme.info : undefined}
          bold={isCursor}
        >
          {`  ${marker} ${c.shorthand || c.id.slice(0, 8)}`}
          <Text color={theme.muted}>
            {` (${c.completed}/${c.total})`}{done ? " ✓" : ""}
          </Text>
        </Text>
        {isExpanded
          ? c.items.map((item, j) => {
              const isSub = j === subCursor;
              const icon =
                item.status === "completed" || item.status === "done"
                  ? "✓"
                  : item.status === "running" || item.status === "in_progress"
                    ? "●"
                    : "○";
              const iconColor =
                item.status === "completed" || item.status === "done"
                  ? theme.success
                  : item.status === "running" || item.status === "in_progress"
                    ? theme.info
                    : theme.muted;

              return (
                <Text key={item.id} color={isSub ? theme.info : theme.muted}>
                  {"      "}
                  <Text color={iconColor}>{icon}</Text>
                  {` ${item.title}`}
                  {item.claimed_by_hostname ? (
                    <Text color={theme.muted}> @{item.claimed_by_hostname}</Text>
                  ) : null}
                </Text>
              );
            })
          : null}
      </Box>
    );
  });
}

// ── Peer list rendering ──────────────────────────────────────

function renderPeerList(
  peers: BrowserPeer[],
  cursor: number,
  expandedId: string | null,
  subCursor: number,
  subCampaigns: BrowserCampaign[],
  expandedCampaignId: string | null,
  itemCursor: number,
  peerCampaignItems: BrowserPlanItem[],
  theme: ThemeColors,
) {
  if (peers.length === 0) {
    return <Text color={theme.muted}>  No peers discovered</Text>;
  }

  return peers.map((p, i) => {
    const isCursor = i === cursor && !expandedId;
    const isExpanded = expandedId === p.instance_id;
    const marker = isExpanded ? "▼" : isCursor ? "▶" : " ";

    // Trust indicator
    let trustIcon: string;
    let trustColor: string;
    let trustSuffix = "";
    if (p.is_trusted && p.tls_enabled) {
      trustIcon = "🔒";
      trustColor = theme.success;
    } else if (p.is_trusted) {
      trustIcon = "🛡";
      trustColor = theme.warning;
    } else {
      trustIcon = "?";
      trustColor = theme.error;
      trustSuffix = " (unpaired)";
    }

    return (
      <Box key={p.instance_id} flexDirection="column">
        <Text
          color={isCursor || isExpanded ? theme.info : undefined}
          bold={isCursor}
        >
          {`  ${marker} `}
          <Text color={trustColor}>{trustIcon}</Text>
          {` ${p.hostname}${trustSuffix}`}
          <Text color={theme.muted}>
            {` (${p.ip_address}) · ${p.mode} · ${p.embryo_count} embryo${p.embryo_count !== 1 ? "s" : ""}`}
          </Text>
        </Text>
        {isExpanded
          ? p.shared_campaigns.length > 0
            ? p.shared_campaigns.map((c, j) => {
                const isSub = j === subCursor && !expandedCampaignId;
                const isCampExpanded = expandedCampaignId === c.id;
                const campMarker = isCampExpanded ? "▼" : isSub ? "▶" : " ";
                const done = c.completed >= c.total && c.total > 0;

                return (
                  <Box key={c.id} flexDirection="column">
                    <Text
                      color={isSub || isCampExpanded ? theme.info : theme.muted}
                      bold={isSub}
                    >
                      {"      "}
                      {campMarker} {c.shorthand || c.id.slice(0, 8)}
                      <Text color={theme.muted}>
                        {` (${c.completed}/${c.total})`}{done ? " ✓" : ""}
                      </Text>
                    </Text>
                    {isCampExpanded
                      ? renderPeerCampaignItems(peerCampaignItems, itemCursor, theme)
                      : null}
                  </Box>
                );
              })
            : <Text color={theme.muted}>{"      "}No shared campaigns</Text>
          : null}
      </Box>
    );
  });
}

// ── Peer campaign items (third level) ────────────────────────

function renderPeerCampaignItems(
  items: BrowserPlanItem[],
  itemCursor: number,
  theme: ThemeColors,
) {
  const rows: React.ReactNode[] = [];

  if (items.length === 0) {
    rows.push(
      <Text key="loading" color={theme.muted}>{"          "}Loading items...</Text>
    );
  } else {
    for (let k = 0; k < items.length; k++) {
      const item = items[k]!;
      const isActive = k === itemCursor;
      const icon =
        item.status === "completed" || item.status === "done"
          ? "✓"
          : item.status === "running" || item.status === "in_progress"
            ? "●"
            : "○";
      const iconColor =
        item.status === "completed" || item.status === "done"
          ? theme.success
          : item.status === "running" || item.status === "in_progress"
            ? theme.info
            : theme.muted;

      rows.push(
        <Text key={item.id} color={isActive ? theme.info : theme.muted}>
          {"          "}
          <Text color={iconColor}>{icon}</Text>
          {` ${item.title}`}
          {item.claimed_by_hostname ? (
            <Text color={theme.muted}> @{item.claimed_by_hostname}</Text>
          ) : null}
        </Text>
      );
    }
  }

  // [Join campaign] action at the bottom
  const isJoinActive = itemCursor === items.length && items.length > 0;
  rows.push(
    <Text
      key="__join__"
      color={isJoinActive ? theme.success : theme.muted}
      bold={isJoinActive}
    >
      {"          "}
      {isJoinActive ? "▶ " : "  "}[Join campaign]
    </Text>
  );

  return <>{rows}</>;
}
