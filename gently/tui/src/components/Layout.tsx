/**
 * Layout primitive — named slots for the TUI shell.
 *
 * One reusable shell that every screen mode plugs into. App.tsx decides
 * *what* each slot renders; Layout owns the *order* and the rule that
 * `modal` (when present) replaces `bottom`.
 *
 * Slots:
 *   header     — branding row, top
 *   welcome    — landing page shown before any transcript content
 *   transcript — Static history + active streaming message
 *   modal      — overlay panel (ChoicePicker, CampaignBrowser, …);
 *                when set, `bottom` is hidden
 *   bottom     — persistent input bar (CommandInput, search bar, …)
 *   statusBar  — always-visible bottom status row
 *
 * Adding a new mode (e.g. a permission prompt, a diff viewer) is now
 * one App.tsx branch that swaps `modal` — no layout surgery.
 */

import React, { type ReactNode } from "react";
import { Box } from "ink";

interface LayoutProps {
  header?: ReactNode;
  welcome?: ReactNode;
  transcript?: ReactNode;
  modal?: ReactNode;
  bottom?: ReactNode;
  statusBar?: ReactNode;
}

export function Layout({
  header,
  welcome,
  transcript,
  modal,
  bottom,
  statusBar,
}: LayoutProps) {
  return (
    <Box flexDirection="column">
      {header}
      {welcome}
      {transcript}
      {modal}
      {modal ? null : bottom}
      {statusBar}
    </Box>
  );
}
