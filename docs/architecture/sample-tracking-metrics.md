# Sample Tracking Metrics

Sample tracking metrics describe what happened to a sample over time. They are
kept separate from organism-specific interpretation so they can support embryos,
wells, cells, organoids, or tissue regions.

## Metric Categories

| Category | Purpose | Examples |
| --- | --- | --- |
| Position | Where the sample was observed | coarse XY, fine XY, position history |
| Exposure | How much light/acquisition burden it received | exposure count, total exposure ms |
| Focus | Whether optical focus is stable | focus history, drift rate |
| Signal | Quantitative image measurements | channel intensity, photobleaching curve |
| Perception | Semantic interpretations | developmental stage, health state, event flags |
| Provenance | How the record was produced | detector id, model version, confidence |

## Universal Fields

These fields are broadly useful across sample types and should remain stable
where possible.

| Field | Type | Description |
| --- | --- | --- |
| `sample_id` | string | Stable id within a session. Current C. elegans code uses `embryo_id`. |
| `sample_uid` | string/null | Optional globally meaningful id. Current code uses `uid`/`embryo_uid`. |
| `role` | string | Experimental role such as test, calibration, control, unassigned. |
| `position_coarse` | object/null | Overview/manual position, usually XY stage coordinates. |
| `position_fine` | object/null | Refined acquisition position for the primary objective/modality. |
| `has_fine_position` | bool | True when fine position should override coarse for acquisition. |
| `position_history` | list | Optional time series of positions and sources. |
| `exposure_count` | integer | Number of acquisitions or exposure events. |
| `total_exposure_ms` | number | Integrated illumination/exposure time. |
| `last_imaged` | ISO datetime/null | Last successful observation time. |
| `focus_history` | list | Focus measurements keyed by position, modality, and score. |
| `signal_intensity_history` | object/list | Per-channel measurements over time. |
| `perception_runs` | list | Links to semantic classifications and reasoning traces. |

## Domain-Specific Fields

Domain fields are valuable, but should be clearly scoped to an organism/sample
plugin.

| Field | Current meaning | Scope |
| --- | --- | --- |
| `developmental_stage` | C. elegans morphology stage | C. elegans organism profile |
| `hatching_status` | C. elegans hatch state | C. elegans organism profile |
| `morphology_history` | Stage/shape observations | organism-specific |
| `custom_classifications` | User-defined labels | experiment-specific |

## Position Semantics

Gently distinguishes coarse and fine positions:

- `position_coarse` comes from an overview image, manual marking, or low-mag
  sample map.
- `position_fine` comes from a refined alignment step for the acquisition
  modality.
- `stage_position` is a compatibility/read convenience: fine if present,
  otherwise coarse.

Persistence and import/export code should preserve both positions. Updating the
coarse position should clear or invalidate fine position unless the update is
known not to affect fine alignment.

## Exposure Semantics

Exposure tracking should describe burden on the sample, not simply image count.

Minimum record:

```yaml
sample_id: embryo_1
timepoint: 12
modality: lightsheet_volume
frames: 50
exposure_ms: 10.0
channels:
  488nm:
    laser_power_pct: 5.0
    total_ms: 500.0
created_at: 2026-05-30T12:00:00
```

For multimodal acquisitions, store one record per channel/modality or include a
structured `channels` map. Do not collapse illumination wavelength and camera
exposure into a single ambiguous number.

## Focus Semantics

Focus measurements should include:

- acquisition modality or objective,
- stage position/context,
- focus actuator position,
- score algorithm,
- score value,
- timestamp,
- source: hardware autofocus, FFT, VLM, manual.

This allows drift estimates such as micrometers/hour without tying the schema to
one microscope.

## Perception Provenance

Every semantic decision should be traceable:

- model or detector id,
- input image/volume uid,
- prompt or detector config hash when applicable,
- confidence or score,
- reasoning trace path when available,
- timestamp and triggering event.

This is the bridge between sample tracking and the evaluation/trajectory
debugging systems.

## Compatibility Notes

Current `EmbryoState` fields map directly onto this schema. Future sample types
can either implement their own typed state object or reuse a generic
`SampleState` shape, provided API responses keep the universal fields above.
