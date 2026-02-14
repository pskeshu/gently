/**
 * Renders a subset of Markdown as native Ink <Text>/<Box> elements.
 *
 * Block-level:  headers, bullet lists, numbered lists, blockquotes,
 *               fenced code blocks, tables, horizontal rules
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

/** Strip inline markdown markers to get plain text length. */
function stripInline(text: string): string {
  return text
    .replace(/\*\*(.+?)\*\*/g, "$1")
    .replace(/\*(.+?)\*/g, "$1")
    .replace(/~~(.+?)~~/g, "$1")
    .replace(/`([^`]+?)`/g, "$1");
}

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
/*  Table parser                                                       */
/* ------------------------------------------------------------------ */

interface TableData {
  headers: string[];
  alignments: ("left" | "center" | "right")[];
  rows: string[][];
}

function isTableSeparator(line: string): boolean {
  return /^\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)*\|?\s*$/.test(line.trim());
}

function isTableRow(line: string): boolean {
  const trimmed = line.trim();
  return trimmed.startsWith("|") && trimmed.endsWith("|") && trimmed.includes("|", 1);
}

function parseTableCells(line: string): string[] {
  const trimmed = line.trim();
  // Remove leading/trailing pipes
  const inner = trimmed.startsWith("|") ? trimmed.slice(1) : trimmed;
  const end = inner.endsWith("|") ? inner.slice(0, -1) : inner;
  return end.split("|").map((c) => c.trim());
}

function parseAlignment(sep: string): "left" | "center" | "right" {
  const trimmed = sep.trim();
  const left = trimmed.startsWith(":");
  const right = trimmed.endsWith(":");
  if (left && right) return "center";
  if (right) return "right";
  return "left";
}

function parseTable(lines: string[], startIdx: number): { table: TableData; consumed: number } | null {
  // Need at least header + separator + 1 data row
  if (startIdx + 2 >= lines.length) return null;

  const headerLine = lines[startIdx];
  const sepLine = lines[startIdx + 1];

  if (!isTableRow(headerLine) || !isTableSeparator(sepLine)) return null;

  const headers = parseTableCells(headerLine);
  const sepCells = parseTableCells(sepLine);
  const alignments = sepCells.map(parseAlignment);

  // Collect data rows
  const rows: string[][] = [];
  let consumed = 2; // header + separator
  for (let i = startIdx + 2; i < lines.length; i++) {
    if (!isTableRow(lines[i])) break;
    rows.push(parseTableCells(lines[i]));
    consumed++;
  }

  if (rows.length === 0) return null;

  return { table: { headers, alignments, rows }, consumed };
}

function padCell(text: string, width: number, align: "left" | "center" | "right"): string {
  const plain = stripInline(text);
  const pad = Math.max(0, width - plain.length);
  if (align === "right") return " ".repeat(pad) + text;
  if (align === "center") {
    const left = Math.floor(pad / 2);
    const right = pad - left;
    return " ".repeat(left) + text + " ".repeat(right);
  }
  return text + " ".repeat(pad);
}

function renderTable(table: TableData, theme: ThemeColors, key: string): React.ReactNode {
  const allRows = [table.headers, ...table.rows];
  const numCols = Math.max(table.headers.length, ...table.rows.map((r) => r.length));

  // Calculate column widths from plain text content
  const colWidths: number[] = Array(numCols).fill(0);
  for (const row of allRows) {
    for (let c = 0; c < numCols; c++) {
      const cell = row[c] ?? "";
      colWidths[c] = Math.max(colWidths[c], stripInline(cell).length);
    }
  }

  // Ensure minimum width for readability
  for (let c = 0; c < numCols; c++) {
    colWidths[c] = Math.max(colWidths[c], 3);
  }

  const align = table.alignments;

  // Build separator line: ─┼─
  const sepParts = colWidths.map((w) => "─".repeat(w + 2));
  const separator = "├" + sepParts.join("┼") + "┤";
  const topBorder = "┌" + colWidths.map((w) => "─".repeat(w + 2)).join("┬") + "┐";
  const bottomBorder = "└" + colWidths.map((w) => "─".repeat(w + 2)).join("┴") + "┘";

  const lines: React.ReactNode[] = [];

  // Top border
  lines.push(
    <Text key={`${key}-top`} color={theme.muted}>{topBorder}</Text>,
  );

  // Header row
  const headerCells = table.headers.map((h, c) =>
    padCell(h, colWidths[c], align[c] ?? "left"),
  );
  lines.push(
    <Box key={`${key}-hdr`} flexDirection="row">
      <Text color={theme.muted}>{"│"}</Text>
      {headerCells.map((cell, c) => (
        <React.Fragment key={c}>
          <Text> </Text>
          <Text bold>{renderInline(cell, theme, `${key}-h${c}`)}</Text>
          <Text> </Text>
          <Text color={theme.muted}>{"│"}</Text>
        </React.Fragment>
      ))}
    </Box>,
  );

  // Separator
  lines.push(
    <Text key={`${key}-sep`} color={theme.muted}>{separator}</Text>,
  );

  // Data rows
  for (let r = 0; r < table.rows.length; r++) {
    const row = table.rows[r];
    const cells = Array(numCols)
      .fill("")
      .map((_, c) => padCell(row[c] ?? "", colWidths[c], align[c] ?? "left"));

    lines.push(
      <Box key={`${key}-r${r}`} flexDirection="row">
        <Text color={theme.muted}>{"│"}</Text>
        {cells.map((cell, c) => (
          <React.Fragment key={c}>
            <Text> </Text>
            {renderInline(cell, theme, `${key}-r${r}c${c}`)}
            <Text> </Text>
            <Text color={theme.muted}>{"│"}</Text>
          </React.Fragment>
        ))}
      </Box>,
    );
  }

  // Bottom border
  lines.push(
    <Text key={`${key}-bot`} color={theme.muted}>{bottomBorder}</Text>,
  );

  return (
    <Box key={key} flexDirection="column" marginTop={0} marginBottom={0}>
      {lines}
    </Box>
  );
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

    // ── Horizontal rule ─────────────────────────────────────
    if (/^(\s*[-*_]\s*){3,}$/.test(raw)) {
      blocks.push(
        <Box key={idx++} marginTop={0} marginBottom={0}>
          <Text color={theme.muted}>{"─".repeat(60)}</Text>
        </Box>,
      );
      continue;
    }

    // ── Table ───────────────────────────────────────────────
    if (isTableRow(raw) && li + 1 < lines.length && isTableSeparator(lines[li + 1])) {
      const result = parseTable(lines, li);
      if (result) {
        blocks.push(renderTable(result.table, theme, `tbl-${idx++}`));
        li += result.consumed - 1; // -1 because the for loop will increment
        continue;
      }
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

    // ── Numbered list ─────────────────────────────────────────
    const numMatch = raw.match(/^(\s*)\d+[.)]\s+(.+)$/);
    if (numMatch) {
      const indent = Math.floor((numMatch[1]?.length ?? 0) / 2);
      // Extract the number prefix for display
      const numPrefix = raw.match(/^(\s*)(\d+[.)])\s/);
      blocks.push(
        <Box key={idx++} paddingLeft={1 + indent * 2}>
          <Text color={theme.muted}>{" "}{numPrefix?.[2] ?? "·"} </Text>
          {renderInline(numMatch[2], theme, `nl-${idx}`)}
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
