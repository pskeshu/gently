# Panels

A **panel** is a standard, reusable control surface that can be mounted in more
than one place and reads its state from one shared source. Panels are how
Gently's UI is meant to be assembled from here on.

This is a policy, not a description: most of the UI does not follow it yet.

## Why

The tabbed UI grew a control at a time, each written where it was needed. So
the same fact is rendered by unrelated code in several places, and the copies
disagree:

- The SPIM head has an LED toggle. Manual mode has a different laser panel.
  Neither knows what the other has done.
- `LASER: ON` in manual mode is set from an HTTP 200 with no hardware read-back
  (#106), so it says ON while the beam is disabled and nothing emits.
- Embryo and image counts disagree across header, footer, panel and agent
  (#129), because four surfaces each count separately.

The cost is not duplicated code. It is that an operator cannot trust the
instrument, and on a microscope with lasers and an objective that can hit a
coverslip, a confident wrong readout is worse than a blank one.

## The rules

**1. One panel per subject, mounted many times.**
A panel owns a subject — light, stage, camera, run state. Every surface that
needs the subject mounts the same panel rather than rendering its own version.

**2. State comes from `SharedState`, never from the caller.**
`status-store.js` is the single store; panels subscribe to a key and re-render.
Two mounted copies cannot disagree because there is only one value. A panel
never takes its reading as a constructor argument.

**3. Read back; never render a command as a fact.**
A value that was sent is not a value that is true. Panels render what the
device reported. A value that has not been read is `—`, not a plausible
default. This is the rule #106 was written by.

**4. Limits come from the server.**
Bounds, options and enumerations are fetched, not hardcoded in JS. Laser power
limits live in `DiSPIMLightSource.POWER_LIMITS_PCT` (488 is `2.0–6.0`, not
`0–100`); a hardcoded slider would offer settings the hardware rejects.

**5. Derived hazards are computed, not commanded.**
"Emitting" is *armed AND routed AND power > 0* — computed from read-back state.
It is never a flag someone remembered to set, because that is the flag someone
will forget to clear.

**6. Presence-driven, not click-driven.**
A panel that reports a transient condition appears when the condition holds and
retires when it stops. No open/close affordance for something the operator did
not ask to see. Someone watching the scope should be able to read the screen
without operating it. See `PREVIEW_IDLE_MS` in `operate.js` for the first
instance.

**7. A panel is a module, not a template fragment.**
Markup, styles and behaviour ship together and mount into a host element, so
adding it to a new surface is one call rather than a copy-paste of markup.

## Status

| Panel | Subject | Mounted in | Notes |
|---|---|---|---|
| Light | LED, beam, routed lines, per-line power | SPIM head | read-back; derives EMITTING |
| ImageView | zoom, pan, contrast, brightness | both camera surfaces | view state, stays local |
| Camera | exposure | both camera surfaces | untitled inside a block that already names the camera |
| Marking | pending marks vs registered roster, detect/register/clear | bottom camera | renders state, calls operate.js for the verbs |

Note what Marking does *not* own: `_markers`, the canvas and the frame
geometry stay in `operate.js`, because marks are placed in stage coordinates
derived from the live frame and that arithmetic belongs with the pixels. A
panel can own a readout and a set of verbs without owning the surface — which
is what lets it mount in the Atrium's EMBRYOS window, where there is no canvas
at all.

ImageView is the exception to rule 2: zoom and contrast are view state, not
instrument state, so they stay per-mount. Two people looking at one microscope
still want their own magnification.

Next: the calibration tab (#108) mounts Light and Camera. Manual mode's
bespoke laser UI is replaced once the shared one is proven on the rig —
deliberately not in the same change, because it is a surface nobody is testing
this week.
