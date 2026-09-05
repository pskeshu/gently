# Devices tab — component audit

> **Status, 2026-09-06.** Findings 1, 4, 5, 6, 7 and #108 are fixed — see the
> pull requests referenced inline. What remains needs the rig, or is
> presentation work listed at the end. The findings are kept rather than
> deleted, because the reasoning is what stops them coming back.

A read of every surface in the Devices tab, from the bottom camera through to
starting an experiment, judged against `docs/architecture/PANELS.md`.

Done without hardware, from the code. Anything marked **rig** needs the
microscope to confirm.

---

## Cross-cutting

### 1. Three renderings of one roster — FIXED

One `panels/roster.js`, mounted twice with actions declared per mount.


| function | host | shows | actions |
|---|---|---|---|
| `renderEmbryoRail` (`operate.js:1001`) | `op-erail-list` | label, XY | **delete** |
| `renderRoster` (`operate.js:1052`) | `op-roster` | label, XY | **role toggle**, **Centre** |
| `embryos.js:2655` | `embryos-count` | count only | — |

The two in Operate are the same list rendered twice, ~80% identical code, each
keeping its own count element. This is the root of #129 (counts disagree across
surfaces) and a plain rule 1 violation.

The action sets differ **arbitrarily**, not by design: on Bottom cam you can
delete an embryo but not centre on it; on Acquisition you can centre and assign
a role but not delete. Nothing about either pane justifies the split — it is
just where each button happened to be added.

*Fix:* one roster component with a declared action set per mount, the way
`CameraPanel` takes `{titled}`. It also becomes the thing the Atrium's EMBRYOS
window mounts.

### 2. The helpful empty state is on the pane you reach second — FIXED

The panel's empty state names the fix at every mount; the CTA is opt-in.


- Bottom cam: *"No embryos yet — detect on the bottom camera, then register."*
  — describes the fix, offers no way to do it. This is the pane you are already
  on.
- Acquisition: *"No embryos marked yet."* **+ a `Go to Bottom cam` button**.

So the actionable empty state is the one you only see after you have gone
somewhere else. Backwards.

### 3. One click still carries three meanings

`op-mark-hint` reads *"Click to mark · click a marker to remove · click a
registered embryo to centre on it"*. #105's fix widened the hit radius so the
third meaning stops firing accidentally, but the count is unchanged and #113
called it out separately. Centre already exists as a button in the Acquisition
roster — the click could drop that meaning entirely.

---

## Bottom cam

Now in reasonable shape after the panel work: the frame is 356×356 (was
163×163), the display range is a histogram panel beneath it, and Marking is a
bordered card under the frame with `MARKED` and `REGISTERED` as separate
counts.

Remaining:

- The `Camera` block holds `Start camera` and exposure; the focus gauge is a
  separate block. Reasonable grouping.
- **rig** `AT_TOL_UM = 50` and `PREVIEW_IDLE_MS = 45000` are guesses carrying
  `RIG-NOTE` markers, still untuned.

---

## SPIM head

- Light panel (LED, `BeamEnabled`, Laser config, per-line power, derived
  EMITTING), Camera exposure, F-drive gauge, sheet alignment.
- **#111 is felt here.** You can select a different embryo and the view now
  correctly blanks to *"Not at this embryo"* — but there is no way to travel to
  it. The pane tells you where you are not, and offers no way to go.
- **#112** — the F-drive still bands fixed step sizes rather than taking a
  distance.

---

## Acquisition — starting an experiment

This is the weakest surface, and two of the findings are workflow correctness
rather than presentation.

### 4. `Start` is four different verbs behind one label — FIXED

The label is the verb: *Acquire one volume* / *Start timelapse* /
*Run tactic* / *Brief the agent*.


`startRun` (`operate.js:1144`) branches on `_mode`:

| mode | what `Start` does | tells you |
|---|---|---|
| `single` | acquires **one volume** | "Volume acquired" |
| `adaptive` | starts a timelapse | "Adaptive timelapse started" |
| `library` | runs a saved tactic | "Tactic started" |
| `agent` | hands a prompt to the agent | — |

In `single` mode the primary button is not starting an experiment at all; it
takes one image and finishes. The label never changes, and the mode selector
sits in a different block above. An operator who has chosen a mode and then
looked away has no way to read what the button will now do.

### 5. `single` mode silently ignores the roster you built — FIXED

It now names the embryo it images, which is also what made the
calibration gate possible on that route.


Every other mode passes `subjectIds()`. `single` uses `_selected` alone. So the
roster — the thing the whole preceding workflow exists to produce — matters or
does not depending on a segmented control, with nothing saying so.

### 6. Nothing anywhere checks that calibration has been done — FIXED

`gently/harness/calibration_gate.py`, enforced server-side on
`timelapse/start`, `run-tactic`, `acquire/volume` and the agent tool.
The rail shows `CAL`/`UNCAL` so a refusal is visible beforehand.


Neither `startRun` nor `POST /api/devices/timelapse/start` looks at whether the
selected embryos carry a calibration fit. The route validates
`interval_seconds > 0` and the embryo ids exist, and that is all.

So an adaptive timelapse can be started on embryos that have never been
calibrated. It will run, and produce data from an uncalibrated stage.

This is the one that matters most, because it inverts the priority the team
actually stated. Ryan, 2026-08-07: *"the main thing is just making sure that we
can get the calibration to work"*. Kesavan, same call: *"Calibration has to
work. Embryo navigation has to work. Then timelapse setup has to work."* The
workflow has a hard dependency and the code enforces none of it.

*Fix:* a preflight on the run path — refuse, or warn explicitly, when a subject
embryo has no calibration. Server-side, so the agent path cannot route around
it.

### 7. `subjectIds()` falls back to imaging the references — FIXED

The fallback is gone and `haveSubjects()` refuses, distinguishing
"no embryos" from "no subjects among your embryos".


```js
const subs = _embryos.filter(e => e.role !== 'calibration').map(e => e.id);
return subs.length ? subs : _embryos.map(e => e.id);
```

If every embryo is marked `calibration`, the fallback returns **all** of them —
so a roster of nothing but reference embryos would be imaged as subjects. The
fallback is meant to be kind to an unassigned roster; it should distinguish
"nobody assigned roles" from "everybody is a reference".

### 8. Nothing disarms the beam when a run ends

`single` mode calls `forceLedOff()` in its `finally`, which is right. Nothing
touches `BeamEnabled`. Per #106, `configure_for_volume_acquisition()` leaves it
`No` — so the state after a run is now visible in the Light panel, but no code
restores or asserts it either way.

---

## Other subtabs

Not audited in depth; each has open issues already.

- **Map** — #107 (remap/recentre), #135 (mm scale bar on a micron map, no
  fit-to-container, no zoom, colliding labels).
- **Details / 3D** — not exercised in the walkthrough.
- **Manual** — holds the bespoke laser UI that the shared Light panel is meant
  to replace once it is proven on the rig. Until then it is a second control
  for illumination that does not know about the first.

---

## What actually blocks shipping

In order:

All three are done. What is left needs the microscope:

1. **#106** — whether the beam actually fires. The Light panel separates
   *armed* from *routed* from *powered*, and
   `GET /api/devices/properties?device=Coherent-Scientific%20Remote` reports
   what the controller says about itself, which works without the specimen in
   focus.
2. **#125** — camera ROI readout, for seeing nuclei well enough to calibrate.
3. **#149** — the per-frame display range. The histogram panel now makes the
   problem visible; the fix is server-side.
4. `AT_TOL_UM = 50` and `PREVIEW_IDLE_MS = 45000` carry `RIG-NOTE` markers and
   are still guesses.

Presentation items — the empty states, the three-meaning caption, the Map
legibility work — are real but do not stop a run from being trustworthy.
