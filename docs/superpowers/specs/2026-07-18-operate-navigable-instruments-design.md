# Operate: navigable steps + real instruments

> **SUPERSEDED (2026-07-20).** The premise below — that Operate is a guided
> sequence the software owns, made navigable rather than removed — was dropped.
> Operate is now three independent, always-live surfaces (Bottom cam / SPIM head
> / Acquisition) with no steps, phases or run-mode chooser; gating comes only
> from live hardware state. See `feat(operate): three instrument surfaces, no
> workflow`. Kept as the record of why the step model was abandoned rather than
> repaired: each round made disclosure more navigable without questioning the
> sequence itself. The safety analysis here (F-drive floor, XY interlock, the
> 15fps TDR cap, the half-frame bug) all still holds and carried over.

**Date:** 2026-07-18
**Context:** v1 release prep. Ryan is the first external user; he tests on the diSPIM with live samples tomorrow morning.
**Status:** approved shape, scoped for one night.

## The problem

Keshu's verdict after driving the Operate view: *"Operate itself is the rigid one."* His acceptance
criteria for this work, in his words: **you can go back and forth, the views are not cluttered, and it
looks professional.**

Three findings from the 2026-07-18 audit set the scope.

**Rigidity has one cause.** Progressive disclosure was implemented as an exclusive CSS selector list
(`operate.css:172-182` renders exactly one `.op-group`), and *every* control lives inside a
`.op-group`. The design has no concept of a control that outlives a step. `setStep()` is only ever
called forward; the header stepper is styled as a navigable breadcrumb but its click handler responds
to one of eight nodes (`operate.js:887-892`). Disclosure became imprisonment.

The data model is already correct: progress (`_states`, monotonic) is separate from the view cursor
(`_step`, free). Nothing structurally prevents backward navigation — no code path does it. So this is
a small change, not a rewrite.

**Ryan lands on the wrong surface.** The devices tab defaults to Map (`devices.js:149`); the finished
guided flow hides behind an unlabeled fifth button.

**Z is rendered as text, not as an instrument.** The device layer already returns
`{position, min, max, distance_to_floor}` on every axis call (`device_layer.py:2615-2636`) and the
frontend discards `min`/`max`. A 1 Hz `fdrive`/`bottom_z` position stream exists with zero
subscribers. The floor gauge fakes its scale with a hardcoded `MAXD=500` against real travel of
30–25000 µm.

## The shape

Keep the paged model. Make it reversible, decluttered, and instrumented.

### 1. The stepper becomes a real control

- Every node is clickable and moves `_step` in either direction.
- `_states` stays monotonic — progress is a record of what has happened, not a permission system.
- **Gates key off live hardware state, never step position.** Centering stays blocked while
  `_headLowered`; down-nudges stay greyed near the floor. Revisiting a step never re-arms a motion
  the hardware state forbids. This is what makes free navigation safe.
- Nodes are real `<button>`s: focusable, `aria-current` on the active one, arrow-key traversal.
- `is-locked` stops meaning "forbidden" and starts meaning "not reached yet" — reachable, visibly
  quieter.

### 2. A persistent control shelf

Controls that describe the instrument rather than the step move out of `.op-group` and are always
present: the two Z gauges, LED, camera toggle, and an always-visible **Cancel**. Escape aborts the
current step. The step rail keeps only the step's own action (Detect, Confirm, Center, Acquire).

`deactivate()` clears `_selected`, `_step`, `_runState` — but **never** `_headLowered`.

### 3. The Z instrument — the signature element

One component, two instances (bottom-focus Z, SPIM F-drive). A vertical travel track showing:

- the axis's real range from the `min`/`max` already being served;
- current position as a marker with a tabular-numeral readout in µm;
- for the F-drive, the floor as a hard band at the bottom of the track, with the approach zone
  shading as `distance_to_floor` shrinks;
- nudge buttons attached to the track and spatially arranged — up above, down below — so the control
  maps to the motion. Bottom-Z gets ±1/±10; the F-drive gains a fine ±1 alongside ±10/±100.

Fed by the existing 1 Hz stream so the marker tracks real motion instead of going stale after
activate.

This is where the design boldness is spent: it is the one element that should read as laboratory
instrumentation rather than a web form. Everything around it stays quiet.

### 4. Declutter

Five image surfaces, two streams. Reduce to one surface per stream:

- delete the Map's embedded bottom-camera panel (`index.html:778-822`);
- guard `renderStep()` on `_active` and both frame handlers on visibility, so a hidden module stops
  decoding JPEGs and stops fighting over stream ownership.

### 5. Visual bar

Work inside the existing token system (`main.css:8-93`) — no new palette. Professional here means
precision: consistent spacing rhythm, tabular numerals on every measurement, aligned control groups,
visible keyboard focus, and no half-built states. Instruments read as instruments; the rail reads as
one column, not a stack of unrelated boxes.

## Also shipping tonight

- **Operate becomes the devices-tab landing view** (`devices.js:149`, `index.html:448-449`).
- **Run chooser defaults to Manual**, not `adaptive` — the current default launches a timelapse and
  never reaches Center.
- **`_headLowered` derives from a live fdrive read on activate.** It is currently a client-only flag
  initialized `false`, so a page reload mid-embryo re-enables Center and will command an absolute XY
  move with the objective down while the strip reads "▲ up."
- **The half-frame bug.** Six copies of `if width > height*2: take left half` discard the right half
  of every 2048×512 SPIM frame (`imaging.py:351`, `server.py:465`, `images.py:93`, `agent.py:1567`,
  `timelapse.py:2581`, `embryo_dataset.py:1064`). This rig is single-camera, so it fires on every
  frame; an embryo right of centre yields a garbage projection, and it feeds the perceiver too. Must
  be verified against a synthetic 2048×512 volume, not only CI.

## Explicitly not tonight

- **Do not touch the encoder.** Image *quality* was never reduced — only frame rate
  (`_ls_interval_sec` 0→1/15, commit `6797e0a`). The 15 fps server cap is the only thing preventing a
  repeat of the 2026-06-29 `nvlddmkm` Video-TDR freeze, because Operate swaps `.src` per frame with
  no throttling. A hard freeze with a sample mounted and the head down is the worst outcome in the
  room. Raising resolution or FPS requires porting a client throttle and measuring on the rig first.
- **Map travel-area framing** (demoting the arbitrary 0,0 origin) — real, but Operate is the surface
  under test.
- **Retiring SAM.** The classical detector already exists in-tree and the ground-truth writer is
  live, but there is no precision/recall harness yet, and the labels captured so far are anchored to
  lossy thumbnails. Fix full-res capture before trusting the flywheel.
- **Server-side motion interlocks.** Every interlock is client-side today.
  `/api/devices/stage/move` validates only that x and y are numbers. The live-read stopgap above
  covers tomorrow; the durable fix is backend work with its own test burden.

## Acceptance

1. From any step, a microscopist can click any earlier step and return, without the hardware
   re-arming anything its live state forbids.
2. Escape and a visible Cancel leave any step, at any time.
3. Both Z axes show range, current position, and — for the F-drive — the floor, updating live.
4. One surface per camera stream; no hidden decoder running.
5. The devices tab opens on Operate.
6. A 2048×512 SPIM frame renders whole.
7. Chrome MCP UI audit passes on every Operate step in both themes: alignment, spacing rhythm,
   overflow, contrast, focus visibility.
