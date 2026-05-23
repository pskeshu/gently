/**
 * SpecPanel — structured "applied plan spec" card.
 *
 * Rendered after the resolution picker attaches a session to a plan
 * and the bridge runs `apply_plan_acquisition_spec`. Replaces the
 * older streamed-bullet recap from the agent ("All set! Here's what
 * got loaded: - Strain: SLS-11 - Acquisition: 80 slices, 10ms…").
 */

import React, { memo } from "react";
import { Box, Text } from "ink";
import type { AppliedSpec, ChatEntry, ThemeColors } from "../../types.js";

interface Props {
  entry: ChatEntry;
  theme: ThemeColors;
}

interface Row {
  label: string;
  value: string;
}

function buildRows(spec: AppliedSpec): Row[] {
  const rows: Row[] = [];
  if (spec.strain) rows.push({ label: "Strain", value: spec.strain });
  if (spec.temperature_c !== undefined && spec.temperature_c !== null) {
    rows.push({ label: "Temperature", value: `${spec.temperature_c}°C` });
  }

  const acquisitionParts: string[] = [];
  if (spec.num_slices !== undefined && spec.num_slices !== null) {
    acquisitionParts.push(`${spec.num_slices} slices`);
  }
  if (spec.exposure_ms !== undefined && spec.exposure_ms !== null) {
    acquisitionParts.push(`${spec.exposure_ms}ms exposure`);
  }
  if (spec.interval_s !== undefined && spec.interval_s !== null) {
    acquisitionParts.push(`every ${spec.interval_s}s`);
  }
  if (acquisitionParts.length > 0) {
    rows.push({ label: "Acquisition", value: acquisitionParts.join(" · ") });
  }

  if (spec.stop_condition) {
    rows.push({ label: "Stop", value: spec.stop_condition });
  }
  if (spec.success_criteria) {
    rows.push({ label: "Success", value: spec.success_criteria });
  }
  if (spec.detectors && spec.detectors.length > 0) {
    rows.push({ label: "Detectors", value: spec.detectors.join(", ") });
  }
  if (spec.adaptive_intervals && Object.keys(spec.adaptive_intervals).length > 0) {
    rows.push({
      label: "Adaptive",
      value: Object.keys(spec.adaptive_intervals).join(", "),
    });
  }
  return rows;
}

function SpecPanelImpl({ entry, theme }: Props) {
  const spec = entry.specData;
  if (!spec) return null;

  const title = spec.plan_item_title ?? "Plan spec loaded";
  const rows = buildRows(spec);
  const labelWidth = Math.max(...rows.map((r) => r.label.length), 0);

  return (
    <Box flexDirection="column" marginBottom={1} paddingLeft={2}>
      <Box flexDirection="row">
        <Text color={theme.accent}>{"▌ "}</Text>
        <Text color={theme.accent} bold>
          {title}
        </Text>
      </Box>
      <Box flexDirection="column" paddingLeft={2}>
        {rows.map((r) => (
          <Box key={r.label} flexDirection="row">
            <Text color={theme.muted}>
              {r.label.padEnd(labelWidth, " ")}
            </Text>
            <Text>{"  "}{r.value}</Text>
          </Box>
        ))}
      </Box>
    </Box>
  );
}

export const SpecPanel = memo(SpecPanelImpl);
