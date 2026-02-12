/**
 * Renders a subset of Markdown as native Ink <Text>/<Box> elements.
 *
 * Block-level:  headers, bullet lists, blockquotes, fenced code blocks
 * Inline:       **bold**, *italic*, `code`, ~~strikethrough~~
 *
 * Streaming-safe — unclosed markers render as literal text until the
 * next chunk completes them.
 */

import React from "react";
import { Box, Text } from "ink";
import type { ThemeColors } from "../types.js";

interface MarkdownTextProps {
  children: string;
  theme: ThemeColors;
}

/* ------------------------------------------------------------------ */
/*  Inline parser                                                      */
/* ------------------------------------------------------------------ */

const INLINE_RE =
  /(\*\*(.+?)\*\*)|(\*(.+?)\*)|(~~(.+?)~~)|(`([^`]+?)`)/g;

function renderInline(
  line: string,
  theme: ThemeColors,
  key: string,
): React.ReactNode[] {
  const nodes: React.ReactNode[] = [];
  let lastIndex = 0;
  let match: RegExpExecArray | null;
  let i = 0;

  INLINE_RE.lastIndex = 0;
  while ((match = INLINE_RE.exec(line)) !== null) {
    // Literal text before match
    if (match.index > lastIndex) {
      nodes.push(
        <Text key={`${key}-t${i++}`}>{line.slice(lastIndex, match.index)}</Text>,
      );
    }

    if (match[2] != null) {
      // **bold**
      nodes.push(
        <Text key={`${key}-b${i++}`} bold>
          {match[2]}
        </Text>,
      );
    } else if (match[4] != null) {
      // *italic*
      nodes.push(
        <Text key={`${key}-i${i++}`} italic>
          {match[4]}
        </Text>,
      );
    } else if (match[6] != null) {
      // ~~strikethrough~~
      nodes.push(
        <Text key={`${key}-s${i++}`} strikethrough>
          {match[6]}
        </Text>,
      );
    } else if (match[8] != null) {
      // `code`
      nodes.push(
        <Text key={`${key}-c${i++}`} color={theme.accent}>
          {match[8]}
        </Text>,
      );
    }

    lastIndex = match.index + match[0].length;
  }

  // Trailing literal
  if (lastIndex < line.length) {
    nodes.push(<Text key={`${key}-t${i++}`}>{line.slice(lastIndex)}</Text>);
  }

  // If nothing matched, return the whole line as-is
  if (nodes.length === 0) {
    nodes.push(<Text key={`${key}-raw`}>{line}</Text>);
  }

  return nodes;
}

/* ------------------------------------------------------------------ */
/*  Block-level parser                                                 */
/* ------------------------------------------------------------------ */

export function MarkdownText({ children, theme }: MarkdownTextProps) {
  const lines = children.split("\n");
  const blocks: React.ReactNode[] = [];
  let inCodeBlock = false;
  let idx = 0;

  for (let li = 0; li < lines.length; li++) {
    const raw = lines[li];

    // ── Fenced code block toggle ──────────────────────────────
    if (raw.trimStart().startsWith("```")) {
      inCodeBlock = !inCodeBlock;
      // Skip the fence line itself
      continue;
    }

    // ── Inside code block ─────────────────────────────────────
    if (inCodeBlock) {
      blocks.push(
        <Box key={idx++} paddingLeft={2}>
          <Text color={theme.secondary}>{raw}</Text>
        </Box>,
      );
      continue;
    }

    // ── Blank line → spacer ───────────────────────────────────
    if (raw.trim() === "") {
      blocks.push(<Box key={idx++} height={1} />);
      continue;
    }

    // ── Headers ───────────────────────────────────────────────
    const headerMatch = raw.match(/^(#{1,3})\s+(.+)$/);
    if (headerMatch) {
      const level = headerMatch[1].length;
      const color =
        level === 1
          ? theme.primary
          : level === 2
            ? theme.secondary
            : theme.accent;
      blocks.push(
        <Box key={idx++}>
          <Text bold color={color}>
            {headerMatch[2]}
          </Text>
        </Box>,
      );
      continue;
    }

    // ── Bullet list ───────────────────────────────────────────
    const bulletMatch = raw.match(/^(\s*)[-*+]\s+(.+)$/);
    if (bulletMatch) {
      const indent = Math.floor((bulletMatch[1]?.length ?? 0) / 2);
      blocks.push(
        <Box key={idx++} paddingLeft={1 + indent * 2}>
          <Text>{"  "}</Text>
          {renderInline(bulletMatch[2], theme, `bl-${idx}`)}
        </Box>,
      );
      continue;
    }

    // ── Blockquote ────────────────────────────────────────────
    const quoteMatch = raw.match(/^>\s?(.*)$/);
    if (quoteMatch) {
      blocks.push(
        <Box key={idx++} paddingLeft={1}>
          <Text dimColor color={theme.muted}>
            {"│ "}
            {quoteMatch[1]}
          </Text>
        </Box>,
      );
      continue;
    }

    // ── Paragraph (inline formatting) ─────────────────────────
    blocks.push(
      <Box key={idx++} flexDirection="row" flexWrap="wrap">
        {renderInline(raw, theme, `p-${idx}`)}
      </Box>,
    );
  }

  return <Box flexDirection="column">{blocks}</Box>;
}
