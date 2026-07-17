# Hardware Profile Template

Use this template when adding or documenting a new hardware profile.

## Profile Identity

- Profile name:
- Primary sample type:
- Primary acquisition modality:
- Device-layer entry point:
- Configuration file(s):

## Standard Operations

| Operation | Supported | Implementation | Notes |
| --- | --- | --- | --- |
| `sample_overview` | yes/no |  |  |
| `detect_samples` | yes/no |  |  |
| `move_to_sample` | yes/no |  |  |
| `focus_scan` | yes/no |  |  |
| `position_calibration` | yes/no |  |  |
| `acquire_snapshot` | yes/no |  |  |
| `acquire_volume` | yes/no |  |  |
| `set_illumination` | yes/no |  |  |
| `read_device_state` | yes/no |  |  |

## Coordinate Frames

List every coordinate frame exposed to the agent or UI.

| Frame | Units | Origin | Axes | Conversion owner |
| --- | --- | --- | --- | --- |
| overview pixels | px |  |  |  |
| stage |  |  |  |  |
| acquisition volume | voxels |  |  |  |

Required notes:

- Which frame is stored in `position_coarse`.
- Which frame is stored in `position_fine`.
- Whether any axis is inverted relative to the overview image.
- Which code owns pixel-to-stage transforms.

## Safety Limits

| Device/axis | Lower | Upper | Enforcement layer | How verified |
| --- | --- | --- | --- | --- |
|  |  |  | device `set()` |  |
|  |  |  | firmware |  |

Every motion source should have a lowest-layer safety limit. If joystick,
manual controls, or vendor UI can bypass Python checks, document the firmware or
operator procedure that closes that path.

## State Reporting

| State field | Source | Poll/callback | Frequency | Read-only |
| --- | --- | --- | --- | --- |
| stage position |  |  |  | yes |
| focus actuator |  |  |  | yes |
| illumination state |  |  |  | yes |
| temperature |  |  |  | yes |

State reporting endpoints must not mutate hardware.

## Data Products

| Product | Format | Storage owner | Metadata required |
| --- | --- | --- | --- |
| overview image |  |  | sample id, position, pixel size |
| snapshot |  |  | sample id, channel, exposure |
| volume |  |  | sample id, voxel size, channel, exposure |

Large arrays should be written by the device layer or persisted once, then
referenced by uid/path. Avoid repeated JSON/base64 transfer for production data.

## Sample Metrics Populated

Check the metrics this profile updates:

- [ ] `position_coarse`
- [ ] `position_fine`
- [ ] `position_history`
- [ ] `exposure_count`
- [ ] `total_exposure_ms`
- [ ] `focus_history`
- [ ] `signal_intensity_history`
- [ ] `perception_runs`

## Test Plan

- [ ] Offline import smoke test
- [ ] Mock device unit tests
- [ ] Hardware availability diagnostic
- [ ] Safety limit rejection test
- [ ] Acquisition dry-run or simulated run
- [ ] Live acquisition test with explicit operator opt-in
