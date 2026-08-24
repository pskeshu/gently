# The Atrium — specification

> A spatial interface in which every capability permanently exists as a framed
> window; the only thing that changes is **attention** — allocated by
> **pressure**, bounded by a **cap**, and shared between an operator and an
> agent that can point but never build.

Two implementations, deliberately:

- **Reference** — [`canvas-surface.html`](canvas-surface.html). Self-testing;
  load with `?selftest` and read the console. Fake data, full model: pressure,
  the release ladder, subwindows, the courtyard designer. This is where a rule
  gets tried before it is trusted.
- **Production** — `gently/ui/web/static/js/atrium.js` + `static/css/atrium.css`.
  Runs on the real UI behind `?atrium=1`, off by default. It adopts the ten
  existing `.tab-content` divs as windows rather than rewriting them. Carries
  **all ten rules**. Pressure is wired to genuine signals — the microscope
  dropping off `ConnectionStatus`, the telemetry socket going down — and its
  ladder is capped at `open` until an operator has seen it escalate. Adding a
  source is one line: give a window `tol` and `pressWhen`.

The reference is allowed to be ahead. When the two disagree, the reference is
the design and the production shell is the debt.
Migration into the live UI: [`MIGRATION.md`](MIGRATION.md).
How it was arrived at: [`IDEAS.md`](IDEAS.md).

An **atrium** is an open middle you move through, ringed by fixed structure.
That is the shape, and the vocabulary follows the architecture:

| Term | Means |
|---|---|
| **atrium** | the whole surface |
| **courtyard** | the fixed cloister of screen-bound edges |
| **bench** | the canvas middle, where attention travels |
| **window** | one capability — always present, always framed |
| **gauge** | a window in its folded rendering |
| **density** | how many windows are open at once |
| **urgency / salience** | what decides (see R5) |
| **release ladder** | the channels by which a fact reaches a human |
| **curator** | the agent that points at places, and never builds them |

---

## R1 — The surface is a map

The bench is one continuous pannable, zoomable plane. Not tabs, not a dock.

*Why:* a UI with nowhere to put new things crams them into existing containers,
which is how tabs grow their own internal switchers. Gently had four such
switchers before this spec — `embryos.js` for board/filmstrip/vitals,
`devices.js` hiding the Operate spine behind a `display:none` view toggle. An
infinite surface always has somewhere to put the next thing, so nothing nests
out of desperation.

*Test:* zoom keeps the point under the cursor pinned. Operator input beats any
running animation — a newer navigation or a grab on the board cancels an
in-flight glide, or you land where you did not ask.

## R2 — Everything exists; few have attention

Every capability has a window and every window always exists. Nothing is
created or destroyed. Attention travels to windows; unattended windows recede
but never leave.

*Why:* it removes an entire class of state — no lifecycle, no "is it mounted",
no add-panel menu. And it buys the thing adaptive layouts always lose: a fixed
geography, so muscle memory survives. The calibration window is in the same
place at density 2 as at density 8.

*Corollary — the agent's action space collapses to one verb: point.* It cannot
build UI, so it cannot build bad UI.

*Test:* every layout returns every window. `survey` puts all of them back on the
bench; nothing is lost across any reconfiguration.

## R3 — The surface is protected

Nothing renders on the surface unframed, orphaned, or half-complete. Content too
large for its frame gets its **own internal zoom**; it never bleeds onto the
board. An empty window still occupies its frame and says that it is empty.

*Why:* the frame edge is the semantic boundary between *zooming the UI* and
*zooming the sample*. Microscopy has two nested maps — the bench and the
specimen — and without a hard boundary an operator cannot tell which gesture
they just made.

*Test:* every window computes `overflow: hidden`. Every folded window carries a
non-empty gauge. A wheel over a frame's inner surface zooms the sample and stops
propagating; a wheel over the bench zooms the bench.

*One level up:* a frame half off the viewport is as half-rendered as content
escaping its frame. Both surfaces therefore fade at their boundary — the mask
sits on an inner layer, not the viewport itself, or the viewport goes
see-through at the edges and whatever is behind it ghosts in.

## R4 — Windows have two renderings

**Folded** is the gauge you glance at — one line, live. **Open** is the control
you reach for. Not hidden/shown: two renderings of one thing.

Density decides how many are open; the rest fold. Folding **moves nothing** —
a window keeps its coordinates and only changes height.

*Why:* semantic zoom falls out for free (focused *is* zoomed-in), and the
geography survives the density dial, so R2's muscle-memory guarantee holds at
every setting.

*Test:* density *N* leaves exactly *N* open. Travelling to a window opens it —
you never arrive at a gauge.

## R5 — Pressure and relief, under a cap

One mechanic at three scales. A demand exceeds a bounded container, propagates
outward, is relieved at the lowest level that can absorb it, and a **hard cap**
means pressure can never fully win.

| Scale | What presses | Container | Relief | Cap |
|---|---|---|---|---|
| within a window | content vs frame | the panel | grow · scroll · internal zoom | fraction of viewport |
| within the deck | windows vs open slots | density | highest salience open, rest fold | the density number |
| against the human | facts vs attention | release ladder (R6) | lowest sufficient channel | `maxChannel` |

**Three quantities, not one.** Collapsing them is a bug that was actually shipped
and caught by running the clock:

```
urgency  = crit × overdue      0 when fresh.  drives the release ladder.
salience = crit + urgency      always > 0.    drives deck ranking.
strain   = demand − capacity   the spatial pair, for content vs frame.
```

*Importance amplifies urgency; it does not manufacture it.* A merely-critical
window must not shout while perfectly fresh.

**Resolved is not refreshed.** A solved fact is done. It does not restart its
staleness clock and creep back up the ladder one tolerance later.

*Why the cap matters most:* it is the entire safety argument for letting an
agent drive this. Pressure yields to a limit, so nothing can run away with the
screen or the operator's evening.

*Test:* a fresh window has zero urgency and releases nothing. One tolerance
overdue equals `crit`. Absurdly overdue still respects `overdueCap`. A resolved
fact survives ninety-nine tolerances without climbing.

## R6 — Release at the lowest channel that will still be seen in time

| # | Channel | Costs the human |
|---|---|---|
| 0 | folded gauge | nothing — already in the courtyard |
| 1 | state chip flips | a glance, if they look |
| 2 | window opens | a glance they will take |
| 3 | curator offers to travel | a decision |
| 4 | attention seized | the current task |
| 5 | OS notification | a context switch |
| 6 | email / SMS | their evening |

A window climbs, one release per rung, never repeating. The ladder is a
threshold on `urgency` — not an authored policy table.

**The release log is part of the rule, not an extra.** Every release records
what went out, through which channel, and why. That log is how the policy gets
defended the first time it wakes someone up.

*Open question for the operator, not for us:* whether rung 4 (`seize` — taking
attention unasked, mid-task, at a microscope) is acceptable at all. It is
implemented because it completes the ladder. It may need to come out.

## R7 — Zero agent turns required

Every move the curator makes, the operator can make by hand. The agent is an
accelerant, never a dependency.

*Why:* the LLM is slow, occasionally wrong, and sometimes absent. It is also
what makes this a general-purpose instrument rather than a nicer Micromanager —
so it must add capability without becoming load-bearing.

*Test:* unplug the curator. Every window, layout, density and destination is
still reachable from the keyboard.

## R8 — Windows may hold addressable children

A window may contain subwindows. A child is a **full destination**
(`attend('p-embryos:vitals')`), carries its own crit/tolerance, and can be
**detached** into a window of its own, keeping its accumulated pressure.

Pressure originating in a child raises its parent, because the parent is how you
reach the child.

*Why:* this is what dissolves the nested switcher. Nesting is not the problem —
nesting *without an address* is. Give the nested thing a destination and it
stops being buried, which is exactly the failure that made Ryan ask us to build
a calibration tab that already existed.

*Test:* addressing a child activates it and unfolds its parent. A pressing child
raises its parent's salience. Detaching produces a real window without the
parent losing its remaining children.

## R9 — Everything is configured, nothing is hard-coded

Courtyard geometry, home viewpoint, density, index weights and layouts all come
from one `CONFIG` object. Adding an edge is one line, not a CSS rewrite.

*Corollary:* rebuilding the courtyard must **evacuate its residents first**.
Wiping the host with `innerHTML=''` deleted every pinned window and the chip
bar — R2 is a load-bearing invariant, not a slogan.

*Test:* every preset round-trips with all windows intact. Pinning to an edge
that config removed degrades to the bench rather than throwing.

## R10 — Windows size to their container, not the viewport

Container queries, not media queries. The same window renders differently at
200px in a rail and 500px on the bench — on any display.

*Test:* a window reflows on its own width with the viewport unchanged. Pressure
measurement settles over two passes, because a container query can reflow
content in response to the width just set — one measurement is stale by
definition.

---

## What this spec does not yet cover

- Layout persistence across sessions
- Attention history (there is no "back", and you feel it immediately)
- Off-screen virtualisation, needed somewhere past ~40 windows
- Keyboard navigation *between* windows (travel is digits and mouse only —
  an accessibility gap, not a nicety)
- Semantic zoom that *reflows* content at focus rather than only scaling it

## The rule about rules

Every rule above earned its place by being violated first. R5's three-quantity
split, R9's evacuation clause, R3's two-pass settle and R4's "travelling opens
it" were all bugs before they were rules. If a future rule cannot name the
failure it prevents, it is decoration — delete it.
