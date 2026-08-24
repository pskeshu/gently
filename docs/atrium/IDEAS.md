# The Atrium — ideation capture

30 minutes of stream-of-consciousness design, 2026-08-24, pskeshu.
Reference implementation: `canvas-surface.html` (self-testing, `?selftest`).

> **The Atrium** is a spatial interface in which every capability permanently
> exists as a framed window; the only thing that changes is attention —
> allocated by pressure, bounded by a cap, and shared between an operator and
> an agent that can point but never build.

An atrium is an open middle you move through, ringed by fixed structure. That
is the shape of the thing, and the vocabulary follows the architecture:

| Term | Means |
|---|---|
| **atrium** | the whole surface |
| **courtyard** | the fixed cloister of screen-bound edges |
| **bench** | the canvas in the middle, where attention travels |
| **window** | one capability, always present, always framed |
| **gauge** | a window in its folded view |
| **density** | how many windows are open at once |
| **pressure** | what decides — at the window, the deck, and the human |
| **release ladder** | the channels a fact can reach a human through |
| **curator** | the agent that points at places, and never builds them |
Screenshots: `../../screenshots/canvas-*.png`.

Not a spec. A capture, so nothing from the session is lost. The spec is a
later, shorter document distilled from the **Rules** section below.

---

## The rules, in the order they arrived

1. **The surface is a map.** Pan and zoom, one continuous plane. Not tabs, not
   a dock — a Miro-like board where the UI renders as citizens of the canvas.
2. **Everything exists. Everything has a view. Few have attention.** No
   create/destroy lifecycle. Fixed geography, so muscle memory survives.
   Attention travels; unattended windows recede but never leave.
3. **The surface is protected.** No orphans, nothing half-rendered. Everything
   has a frame. Content too big for its frame gets its *own* internal zoom and
   never bleeds onto the board. The frame is the boundary between zooming the
   UI and zooming the sample.
4. **A specialist agent keeps the feed current.** Separate from the experiment
   agent. Its only job is what is live and where attention should be.
5. **Zero agent turns required.** Every move the curator makes, the operator can
   make by hand. The agent is an accelerant, never a dependency.
6. **A flight deck of folded and open views.** Folded is the gauge you glance
   at; open is the control you reach for. Two renderings of one window, not
   hidden/shown.
7. **View density is a count, not a zoom.** How many windows are open at once.
   The rest fold to gauges.
8. **Every window carries an information index** — update frequency × how
   crucial it is to the experiment. Density opens the top N by index.
9. **The courtyard.** An outer cloister of screen-bound windows — side, bottom,
   any edge — around an open middle the canvas moves through. Fixed attention
   vs travelling attention, two coordinate spaces on one screen.
10. **Attention layouts.** A layout is two decisions: which windows are pinned
    where, and where the canvas is looking. Different missions, different
    fixed/dynamic split.
11. **The courtyard design is itself configurable, at a high level**, with a
    meta window that previews it live. The designer is a window on the surface
    it configures.
12. **Everything sizes to its own container**, not the viewport. The same window
    renders differently at 200px in a rail and 500px on the canvas.
13. **Inner pressure.** Content that does not fit claims space outward, bounded
    by what is free. Pressure propagates: a pressed resident widens its edge.
14. **Information release management.** Which channel does a human learn a fact
    through, and when. Notify, email, open a window, flip a chip — all release
    channels with different cost to the human.
15. **Pressure is the core primitive**, at three scales — see below.

---

## Pressure and relief, unified (the framework's spine)

One mechanic at three scales. A demand exceeds a bounded container, propagates
outward, is relieved at the lowest level that can absorb it, and a **hard cap**
means pressure can never fully win. The cap is the safety property: it is why an
agent driving this cannot run away with the screen or the operator's evening.

| Scale | What presses | Container | Relief | Cap |
|---|---|---|---|---|
| within a window | content vs frame | the panel | grow · scroll · internal zoom | 78% of viewport |
| within the deck | windows vs open slots | density dial | top index open, rest fold | the density number |
| against the human | facts vs attention | release ladder | lowest channel seen in time | no email at 2am |

With time folded in, latency tolerance stops being a bolt-on:

```
pressure = criticality × (elapsed / tolerance)
```

A stale calibration presses harder the longer it is stale. The release ladder is
then a threshold on one number, not an authored policy table.

### Three quantities, not one

Attempting to run everything off a single number failed twice, and the
corrections are the model:

- **urgency = crit x overdue.** Zero when fresh. Drives the release ladder.
  Importance *amplifies* urgency; it does not manufacture it. Collapsing the
  two made a merely-critical window shout while perfectly fresh.
- **salience = crit + urgency.** Always positive. Drives deck ranking, so an
  important window still earns a slot before anything is overdue, and an
  overdue one climbs.
- **strain / give.** The spatial pair, for content against its frame. Same
  *shape* as urgency, different *quantity* — do not force one scalar to do both.

And one more distinction the clock exposed: **resolved is not refreshed.** A
solved transform is done. It must not restart its staleness clock and creep
back up the ladder a tolerance later.

### The release ladder — escalating cost to the human

| # | Channel | Costs |
|---|---|---|
| 0 | folded gauge | nothing, it is already in the courtyard |
| 1 | state chip flips `LIVE`→`STALE` | a glance, if they look |
| 2 | index rises, deck refolds, window opens | a glance they will take |
| 3 | curator pulses and offers to travel | a decision |
| 4 | attention seized, canvas travels unasked | the current task |
| 5 | OS notification | a context switch |
| 6 | email / SMS | their evening |

**Rule: release at the lowest channel that will still be seen in time.**

This also bounds the curator's action space to *assign channel and timing* —
small and auditable. It cannot build UI, so it cannot build bad UI.

---

## Built in the prototype

Canvas surface with cursor-pinned zoom · attention travel with glide and recede ·
authored home viewpoint and view density · fold/open with live gauges (4 Hz,
decoupled from data rate) · information index · density dial · courtyard with
configurable edges · attention layouts (survey / operate / acquire) · CONFIG-driven
geometry · courtyard designer with preset schematics and live preview · container
queries · inner pressure with caps and edge widening · curator that points rather
than builds · full keyboard operability with no agent.

## Not built yet

- Semantic zoom that *reflows content* at focus rather than only scaling it

## Known gaps in the prototype

- `#hint`, `#lay`, `#zoom` still float instead of being courtyard residents, so
  they collide once an edge is configured into their corner
- Canvas panels get sliced mid-title by the viewport edge — rule 3 one level up:
  a frame half off the viewport is as half-rendered as content escaping its frame
- Deck gauges truncate at the fixed panel width
- Window title and gauge state the same value twice (`EMBRYOS · 5` / `5 · e02 sel`)
- Index weights (`crit .65 / freq .35`) are a guess — they come from watching Ryan

## Bugs found while building (each one a design lesson)

- An in-flight glide was not cancelled by newer navigation or by grabbing the
  board → you land where you did not ask. *Operator input must beat animation.*
- Travel keys fired from focused controls → typing `40` into exposure flung the
  operator to home, then acquire. *Global keys must respect focus.*
- `buildCourtyard` wiped its host with `innerHTML=''`, deleting every pinned
  window. *Evacuate before demolishing — "everything exists" is a load-bearing
  invariant, not a slogan.*
- Pinning to an edge that config had removed. *A layout must degrade to the
  canvas, not throw.*
- Attention landed on a folded window, so you arrived at a gauge.
  *Attention implies the open view.*
- Pressure measured before container queries reflowed the content. *One
  measurement is stale by definition; settle over two passes.*
- One number for importance and urgency → a fresh but critical window flipped
  its own chip to `STALE` a second after being solved. *Over-unification is a
  bug too. Importance amplifies urgency, it does not create it.*
- Relief reset the staleness clock rather than marking the fact done, so a
  solved calibration went stale again one tolerance later. *Resolved is not
  refreshed.*
- `peek()` called `getElementById` on the panel still being constructed.
  *Pass the element in; a window cannot look itself up before it exists.*

## Open questions

- **Framework, spec, or paper?** Current position: build it inside Gently, write
  a one-page spec, publish the paradigm as a paper. Extract a package only when a
  second instrument asks — the second user is what teaches you which parts are
  general. Do not adopt tldraw / React Flow: they cover the ~40 lines that were
  cheap and none of the parts that are novel, and they cost the build-step-free
  edit-and-reload loop that makes this iterable against hardware.
- Corners policy — `sides` or `ends` as the default
- Whether low density should leave the board breathing or compact into the space

## Where this sits relative to the release

**Part of `1.0.0.dev1`.** Strategic call by pskeshu, 2026-08-24: this surface is
as important as the six issues from Ryan's walkthrough (#105–#110), not the
release after. dev1 therefore carries two shapes of work — bug fixes with a
known end, and a paradigm still being invented — and its acceptance test grows
accordingly: Ryan completes the walkthrough *and* reacts to the surface.
