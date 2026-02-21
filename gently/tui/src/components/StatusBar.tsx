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

  // Sub-items for expanded item (level 2)
  let subItems: unknown[] = [];
  let campaignLevel2Mode: "subcampaigns" | "items" = "items";
  if (expandedId) {
    if (activeView === "campaigns") {
      const ec = campaigns.find((c) => c.id === expandedId);
      if (ec && ec.subcampaigns.length > 0) {
        subItems = ec.subcampaigns;
        campaignLevel2Mode = "subcampaigns";
      } else {
        subItems = ec?.items ?? [];
        campaignLevel2Mode = "items";
      }
    } else {
      subItems = peers.find((p) => p.instance_id === expandedId)?.shared_campaigns ?? [];
    }
  }

  // Action count for a campaign (share toggle + optional pause/resume)
  const getActionCount = (c: BrowserCampaign) => c.status === "completed" ? 1 : 2;

  // Level-2 max cursor for campaigns (items/subcampaigns + optional action rows)
  const campaignLevel2Max = (() => {
    if (activeView !== "campaigns" || !expandedId || expandedCampaignId) return subItems.length - 1;
    const ec = campaigns.find((c) => c.id === expandedId);
    if (!ec) return subItems.length - 1;
    if (campaignLevel2Mode === "subcampaigns") return ec.subcampaigns.length - 1;
    return ec.items.length + getActionCount(ec) - 1;
  })();

  // Third-level count for campaigns (subcampaign items + action rows)
  const campaignLevel3Max = (() => {
    if (activeView !== "campaigns" || !expandedId || !expandedCampaignId) return -1;
    const ec = campaigns.find((c) => c.id === expandedId);
    if (!ec) return -1;
    const sc = ec.subcampaigns.find((s) => s.id === expandedCampaignId);
    if (!sc) return -1;
    return sc.items.length + getActionCount(sc) - 1;
  })();

  // Third-level for peers (peer campaign items + Join action)
  const hasExpandedPeerCampaign = activeView === "peers" && expandedId && expandedCampaignId;
  const peerThirdLevelCount = hasExpandedPeerCampaign ? peerCampaignItems.length + 1 : 0;

  // Helper: dispatch a campaign action command and refresh the browser
  const dispatchCampaignAction = (campaign: BrowserCampaign, actionIdx: number) => {
    const actions: string[] = [];
    actions.push(
      campaign.is_shared
        ? `/campaign unshare ${campaign.id}`
        : `/campaign share ${campaign.id}`,
    );
    if (campaign.status !== "completed") {
      actions.push(
        campaign.status === "active"
          ? `/campaign pause ${campaign.id}`
          : `/campaign resume ${campaign.id}`,
      );
    }
    const action = actions[actionIdx];
    if (action) {
      send({ type: "command", command: action });
      setTimeout(() => send({ type: "browse", target: "campaigns" }), 300);
    }
  };

  // ── Keyboard handling ────────────────────────────────────
  useInput((_input, key) => {
    if (!browserOpen) return;

    if (key.escape) {
      if (expandedCampaignId) {
        setExpandedCampaignId(null);
        setItemCursor(0);
        if (activeView === "peers") onClearPeerCampaignItems();
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
        if (itemCursor > 0) {
          setItemCursor((c) => c - 1);
        } else {
          setExpandedCampaignId(null);
          setItemCursor(0);
          if (activeView === "peers") onClearPeerCampaignItems();
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
        if (activeView === "campaigns") {
          setItemCursor((c) => Math.min(campaignLevel3Max, c + 1));
        } else {
          setItemCursor((c) => Math.min(peerThirdLevelCount - 1, c + 1));
        }
      } else if (expandedId) {
        if (activeView === "campaigns") {
          setSubCursor((c) => Math.min(campaignLevel2Max, c + 1));
        } else {
          setSubCursor((c) => Math.min(subItems.length - 1, c + 1));
        }
      } else {
        setCursor((c) => Math.min(listItems.length - 1, c + 1));
      }
      return;
    }

    if (key.return) {
      // ── Third level ──
      if (expandedCampaignId) {
        if (activeView === "campaigns") {
          // Campaign third level: items + actions
          const ec = campaigns.find((c) => c.id === expandedId);
          const sc = ec?.subcampaigns.find((s) => s.id === expandedCampaignId);
          if (sc) {
            const actionIdx = itemCursor - sc.items.length;
            if (actionIdx >= 0) dispatchCampaignAction(sc, actionIdx);
          }
        } else {
          // Peer third level: items + [Join]
          if (itemCursor === peerCampaignItems.length) {
            const peer = peers.find((p) => p.instance_id === expandedId);
            if (peer) {
              send({
                type: "command",
                command: `/join-campaign ${peer.hostname} ${expandedCampaignId}`,
              });
              onCloseBrowser();
            }
          }
        }
        return;
      }

      // ── Second level ──
      if (expandedId) {
        if (activeView === "campaigns") {
          const ec = campaigns.find((c) => c.id === expandedId);
          if (!ec) return;
          if (campaignLevel2Mode === "subcampaigns") {
            // Expand subcampaign to level 3
            const sc = ec.subcampaigns[subCursor];
            if (sc) {
              setExpandedCampaignId(sc.id);
              setItemCursor(0);
            }
          } else {
            // Leaf campaign: items + actions
            const actionIdx = subCursor - ec.items.length;
            if (actionIdx >= 0) dispatchCampaignAction(ec, actionIdx);
          }
        } else if (activeView === "peers") {
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

      // ── First level ──
      const item = listItems[cursor];
      if (!item) return;

      if (activeView === "campaigns") {
        const c = item as BrowserCampaign;
        if (expandedId === c.id) {
          setExpandedId(null);
          setSubCursor(0);
          setExpandedCampaignId(null);
          setItemCursor(0);
        } else {
          setExpandedId(c.id);
          setSubCursor(0);
          setExpandedCampaignId(null);
          setItemCursor(0);
        }
      } else {
        const p = item as BrowserPeer;
        if (!p.is_trusted) {
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
            ? renderCampaignList(campaigns, cursor, expandedId, subCursor, expandedCampaignId, itemCursor, theme)
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

function campaignStatusBadge(status: string, theme: ThemeColors): { char: string; color: string } {
  if (status === "paused") return { char: "⏸", color: theme.warning };
  if (status === "completed") return { char: "✓", color: theme.muted };
  return { char: "●", color: theme.success };
}

function itemStatusIcon(status: string, theme: ThemeColors): { char: string; color: string } {
  if (status === "completed" || status === "done") return { char: "✓", color: theme.success };
  if (status === "running" || status === "in_progress") return { char: "●", color: theme.info };
  if (status === "blocked") return { char: "✗", color: theme.error };
  if (status === "skipped") return { char: "–", color: theme.muted };
  return { char: "○", color: theme.muted };
}

function typeBadge(type?: string): string {
  if (!type) return "";
  const map: Record<string, string> = {
    imaging: "[img]", bench: "[lab]", genetics: "[gen]",
    analysis: "[ana]", decision_point: "[dec]",
  };
  return map[type] || "";
}

function renderCampaignActions(
  campaign: BrowserCampaign,
  actionIdx: number,
  indent: string,
  theme: ThemeColors,
) {
  const actions: string[] = [];
  actions.push(campaign.is_shared ? "[Unshare]" : "[Share on mesh]");
  if (campaign.status !== "completed") {
    actions.push(campaign.status === "active" ? "[Pause campaign]" : "[Resume campaign]");
  }
  return actions.map((label, idx) => {
    const isActive = idx === actionIdx;
    return (
      <Text
        key={label}
        color={isActive ? theme.success : theme.muted}
        bold={isActive}
      >
        {indent}{isActive ? "▶ " : "  "}{label}
      </Text>
    );
  });
}

function renderCampaignList(
  campaigns: BrowserCampaign[],
  cursor: number,
  expandedId: string | null,
  subCursor: number,
  expandedCampaignId: string | null,
  itemCursor: number,
  theme: ThemeColors,
) {
  if (campaigns.length === 0) {
    return <Text color={theme.muted}>  No campaigns</Text>;
  }

  return campaigns.map((c, i) => {
    const isCursor = i === cursor && !expandedId;
    const isExpanded = expandedId === c.id;
    const marker = isExpanded ? "▼" : isCursor ? "▶" : " ";
    const badge = campaignStatusBadge(c.status, theme);

    return (
      <Box key={c.id} flexDirection="column">
        <Text
          color={isCursor || isExpanded ? theme.info : undefined}
          bold={isCursor}
        >
          {`  ${marker} `}
          <Text color={badge.color}>{badge.char}</Text>
          {` ${c.shorthand || c.id.slice(0, 8)}`}
          {c.is_shared ? <Text color={theme.success}>{" \u27C6"}</Text> : null}
          <Text color={theme.muted}>
            {` (${c.completed}/${c.total})`}
          </Text>
        </Text>
        {/* Description + target when expanded */}
        {isExpanded && c.description ? (
          <Text color={theme.muted}>
            {"      "}{c.description.length > 70 ? c.description.slice(0, 67) + "..." : c.description}
          </Text>
        ) : null}
        {isExpanded && c.target ? (
          <Text color={theme.muted}>{"      "}Target: {c.target}</Text>
        ) : null}
        {/* Level 2: subcampaigns or items + actions */}
        {isExpanded
          ? c.subcampaigns.length > 0
            ? renderSubcampaigns(c.subcampaigns, subCursor, expandedCampaignId, itemCursor, theme)
            : (
              <>
                {c.items.map((item, j) => {
                  const isSub = j === subCursor && !expandedCampaignId;
                  const icon = itemStatusIcon(item.status, theme);
                  const tb = typeBadge(item.type);
                  return (
                    <Text key={item.id} color={isSub ? theme.info : theme.muted}>
                      {"      "}
                      <Text color={icon.color}>{icon.char}</Text>
                      {tb ? <Text color={theme.muted}> {tb}</Text> : null}
                      {` ${item.title}`}
                      {item.claimed_by_hostname ? (
                        <Text color={theme.muted}> @{item.claimed_by_hostname}</Text>
                      ) : null}
                    </Text>
                  );
                })}
                {renderCampaignActions(c, subCursor - c.items.length, "      ", theme)}
              </>
            )
          : null}
      </Box>
    );
  });
}

function renderSubcampaigns(
  subcampaigns: BrowserCampaign[],
  subCursor: number,
  expandedCampaignId: string | null,
  itemCursor: number,
  theme: ThemeColors,
) {
  return subcampaigns.map((s, j) => {
    const isSub = j === subCursor && !expandedCampaignId;
    const isCampExpanded = expandedCampaignId === s.id;
    const campMarker = isCampExpanded ? "▼" : isSub ? "▶" : " ";
    const badge = campaignStatusBadge(s.status, theme);

    return (
      <Box key={s.id} flexDirection="column">
        <Text
          color={isSub || isCampExpanded ? theme.info : theme.muted}
          bold={isSub}
        >
          {"      "}{campMarker}{" "}
          <Text color={badge.color}>{badge.char}</Text>
          {` ${s.shorthand || s.id.slice(0, 8)}`}
          {s.is_shared ? <Text color={theme.success}>{" \u27C6"}</Text> : null}
          <Text color={theme.muted}>
            {` (${s.completed}/${s.total})`}
          </Text>
        </Text>
        {isCampExpanded ? (
          <>
            {s.items.map((item, k) => {
              const isActive = k === itemCursor;
              const icon = itemStatusIcon(item.status, theme);
              const tb = typeBadge(item.type);
              return (
                <Text key={item.id} color={isActive ? theme.info : theme.muted}>
                  {"          "}
                  <Text color={icon.color}>{icon.char}</Text>
                  {tb ? <Text color={theme.muted}> {tb}</Text> : null}
                  {` ${item.title}`}
                  {item.claimed_by_hostname ? (
                    <Text color={theme.muted}> @{item.claimed_by_hostname}</Text>
                  ) : null}
                </Text>
              );
            })}
            {renderCampaignActions(s, itemCursor - s.items.length, "          ", theme)}
          </>
        ) : null}
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
