# Sample and Hardware Domain Model

Gently's current production profile is a diSPIM imaging C. elegans embryos, but
the control model should be described in terms that also fit other microscopes
and sample types. This document defines that vocabulary.

## Design Rule

Application code should talk about samples, observations, acquisitions, and
calibration. Hardware modules translate those concepts to device-specific
plans, device names, and timing details.

The current `Embryo` state is therefore an implementation of a more general
sample state. It should remain useful for C. elegans while keeping fields and
operations that other sample models can reuse.

## Concept Mapping

| Gently concept | Current diSPIM implementation | Other modality examples |
| --- | --- | --- |
| Sample | C. elegans embryo | well, organoid, cell colony, tissue region |
| Sample overview | Bottom-camera widefield image | low-mag confocal tile, plate overview, brightfield montage |
| Sample detection | Embryo detection/marking | well detection, cell segmentation, ROI picking |
| Position calibration | Pixel-to-stage and SPIM alignment | tile registration, well-to-stage map, objective-specific alignment |
| Focus scan | Piezo/galvo focus sweep | Z-stack focus curve, autofocus objective sweep |
| 3D acquisition | Lightsheet volume | confocal Z-stack, widefield deconvolution stack |
| Timepoint | One scheduled sample observation | one image, stack, or multimodal acquisition at a sample/time |
| Perception | VLM/classifier stage reasoning | phenotype call, quality control, event detection |

## Standard Operation Names

Hardware profiles should expose conceptually named operations even when their
device implementation is modality-specific.

| Operation name | Meaning | diSPIM backing operation |
| --- | --- | --- |
| `sample_overview` | Capture a field that can locate samples | bottom camera capture |
| `detect_samples` | Produce sample candidates/ROIs | embryo detector or manual marking |
| `move_to_sample` | Move the instrument to a sample's resolved position | XY stage move |
| `focus_scan` | Search focus around a sample or plane | piezo/galvo focus sweep |
| `position_calibration` | Refine sample position/calibration state | center/verify and SPIM calibration |
| `acquire_volume` | Acquire a 3D observation | lightsheet volume scan |
| `acquire_snapshot` | Acquire a 2D observation | lightsheet snap or overview image |
| `set_illumination` | Configure light source state/power | laser, LED, room light controls |
| `read_device_state` | Report physical state without mutation | device-layer status endpoints |

Names in tool descriptions, plan metadata, logs, and docs should prefer these
general concepts. The hardware package may still contain files named for the
real devices (`bottom_camera`, `galvo`, `piezo`) because those are implementation
details inside the diSPIM profile.

## Layering

1. `gently.core` defines storage, events, coordinates, and image utilities.
2. `gently.harness` defines reusable agent, session, tool, prompt, and planning
   mechanics.
3. `gently.organisms` defines biological semantics, such as C. elegans stages.
4. `gently.hardware` defines device profiles and maps standard operations to
   concrete devices/plans.
5. `gently.app` composes one organism and one hardware profile into the
   microscopy agent.

Domain state should move upward as structured sample records. Raw device
details should stay in the hardware profile unless the UI is explicitly showing
hardware diagnostics.

## Plan Naming Convention

Use `*_plan` for Bluesky/device-layer plans and include the conceptual operation
in the metadata when possible:

```python
_md = {
    "plan_name": "acquire_volume",
    "operation": "acquire_volume",
    "hardware_profile": "dispim",
}
```

Recommended naming:

| Preferred | Avoid for new public API | Reason |
| --- | --- | --- |
| `sample_overview_plan` | `bottom_camera_plan` | overview generalizes beyond diSPIM |
| `focus_scan_plan` | `piezo_sweep_plan` | focus is the user-facing concept |
| `position_calibration_plan` | `center_embryo_plan` | calibration can apply to many samples |
| `acquire_volume_plan` | `lightsheet_only_plan` | volume acquisition is modality-neutral |

Existing diSPIM-specific names do not need churn. New public tools, docs, and
metadata should use the conceptual names and point to the diSPIM implementation.

## Extension Checklist

A new hardware profile should document:

- Which device or plan provides `sample_overview`.
- How sample coordinates map to stage coordinates.
- Which acquisition operations are supported: snapshot, volume, timelapse,
  multichannel, burst.
- Which state readings are available without moving hardware.
- Which safety limits are enforced in hardware/device classes.
- Which sample tracking metrics are populated by the profile.

Use `docs/architecture/hardware-profile-template.md` as the starting point.
