# Embryos tab — monitoring study

6 October 2026. A design pass over the Embryos tab (`gently/ui/web/templates/index.html` 255–340, `static/js/embryos.js`, `static/css/main.css`), prompted by a brightfield-only overnight run that the tab could not describe. Mockup: [`2026-10-06-embryos-monitoring.html`](2026-10-06-embryos-monitoring.html).

## What the tab does today, and where it breaks

`EmbryosManager` keeps `state.embryos`, `detectionReasoning` and `_dicFrames`. Everything on screen is derived from the first two; the third is an afterthought bolted to the header.

- `renderStatusBadge()` reads `this.state.status`, but short-circuits to "No active timelapse" whenever `Object.keys(this.state.embryos).length === 0`. A brightfield-only run has no embryos, so it is labelled idle while it runs.
- `renderSummary()` returns `''` when there are no embryos. The run's one live number (next frame due) is never shown.
- The Default view's `reasoning-panel` falls through to `renderSmartEmptyState('no-embryos')`: "No embryos yet — Start a timelapse acquisition… Go to Calibration" — under a strip of frames the run is taking.
- `renderBoardView`, `renderFilmstripView`, `renderVitalsView` all `return` on zero embryos with "No embryos to display". Film has a DIC row (`_filmDicRow`) but the early return means it is never reached in a brightfield-only run.
- `renderDicStrip()` shows `all.slice(-12)` at 110 px in the header panel. `openDicViewer()` is a full-screen modal with prev/next only. No scrub, no play, no facts beyond frame number, time and stage position. `/api/dic/frames` does not return exposure, light, or the reference record a frame was taken under, although `brightfield.py` files that record and `DicOverview` holds the exposure and light.
- `reconcileWithServerState()` never reads `serverState.volumes` or `serverState.dic`, though `TimelapseState.to_dict()` sends both. The UI cannot tell a brightfield-only run from a mixed one, or know when the next overview frame is due.
- `docs/atrium/SPEC.md` R1 already names this tab's view switcher as the example of a container that "grew its own internal switcher". Four views sharing one `renderDetailPanel()` confirm it.

## 1. Information architecture

What an operator wants at a glance is the same question in three tenses: *is it still going, what has it got, when is the next thing.* The header should answer those three in words, and nothing else.

Status comes from `serverState.status` alone. Stats come from `volumes` and `dic.enabled`. Embryo count is a stat, never a gate.

### (a) Brightfield-only, overnight

```
● Running · brightfield          48 frames   every 10 min   next 3:12   since 21:40 (6 h 20)   dark/flat ✓
```

- Badge: filled green circle (the STATUS SHAPES rule in `main.css`), text "Running · brightfield". The suffix says why there are no embryos.
- Stats, in this order: frames taken, cadence, next due (the one live number, in mono, updated each second), elapsed, references. "dark/flat ✓" is the Sessions tab's own phrasing (`review.js holdings()`); if none match the run's light and exposure it reads "no dark/flat" in amber.
- Nothing about embryos. No "0 Active · 0 Done".

### (b) Mixed

```
● Running          4 embryos, 3 going   round 112   next volume 1:30   overview 38 frames, next 8:12   since 21:40 (6 h 20)
```

- Two cadences, two "next" values, each labelled with what it is. The current `Next` label with no noun is why the header is cryptic.
- "3 going" replaces "Active"; "1 done" is implied and said on the tile.

### (c) Finished, being reviewed

```
○ Complete · ended 09:12          4 embryos   112 timepoints   38 frames   18 h 40          Open in Sessions →
```

- Badge: ring (done) with the end time. A run cut short: half-filled amber circle, "Stopped early · ended 03:12 after 48 of 112".
- Stats become totals; countdowns disappear. One link hands off to the Sessions tab, which already renders the full record. The Embryos tab should not grow a second review surface.

### Component

`.run-header`: one row, `baseline` aligned. Badge left; stats as a `<dl>` of `dt/dd` pairs rendered inline, label after value in `--text-muted` (as today), values in tabular numerals. No stat is upper-cased.

## 2. The brightfield stage

The experiment in a brightfield-only run *is* the series of dish frames. It needs the room the embryo rail and reasoning panel have today.

### Layout: `.bf-stage`

```
┌──────────────────────────────────────────────────┬───────────────────┐
│                                                  │ Frame 48 of 48    │
│                                                  │ Round 48          │
│          newest frame, fit to height             │ 03:12 · +6 h 20   │
│          (max 70vh, aspect preserved)            │ 1 240, 860 µm     │
│                                                  │ 20 ms · room light│
│                                                  │ dark/flat ✓ 21:38 │
│                                                  │ [Corrected|Raw]   │
├──────────────────────────────────────────────────┴───────────────────┤
│ ▏▏▏▏▏▏▏▏▏▏▏▏▏▏▏▏▏▏▏▏▏▏▏▏▏▏▏▏▏▏▏▏▏▏▏▏▏▏▏▏▏▏▏▏▏▏▏▏▌      timeline       │
│ 21:40                                                   03:12 ● live  │
│ ‹  ›  ▶ play 8/s        Latest                                        │
└──────────────────────────────────────────────────────────────────────┘
```

- **Frame** (`.bf-frame`): the newest frame, `object-fit: contain`, fills the width the facts column leaves. Replaces the 110 px thumbnail *and* the fixed modal `#dic-viewer`. A frame you can see does not need a lightbox.
- **Facts** (`.bf-facts`, 200 px, JetBrains Mono 0.78 rem): frame *n* of *N*, round, wall time and elapsed since start, stage position, exposure and light (`DicOverview.exposure_ms`, `light`, `led_intensity_pct`), the reference record that applies. These come from `/api/dic/frames` once it carries `exposure_ms`, `light`, `led_intensity_pct` and `reference` (the record name `brightfield.py` already writes into each frame's metadata).
- **Corrected | Raw** (`.bf-correct`): a two-segment control under the facts, shown only when `/api/brightfield/references?light=&exposure_ms=` returns a `match`. Default Corrected. The PNG route takes `?corrected=1` and applies `(frame − dark) / (flat − dark) · mean(flat − dark)` server-side; the browser stays dumb.
- **No references**: in place of the toggle, one amber line with the diamond prefix: "No dark or flat for 20 ms under room light. Take references →" linking to the brightfield panel in Calibration. Not a modal, not a banner across the tab.
- **Timeline** (`.bf-timeline`): full width, 48 px tall. Under ~40 frames it shows 64 px thumbnails; above that, one 3 px tick per frame on a time axis so a 112-frame night still fits one row with no horizontal scroll. Pointer movement over it scrubs the big frame (the `wireScrub()` pattern from `review.js`, `cursor: ew-resize`); `←`/`→` step; `Space` plays at 8 frames/s; `Home`/`End` jump. The playhead is a 2 px `--accent` line with the frame's time beneath it.
- **Live follow**: while the playhead is on the newest frame, a new frame replaces the picture and the "● live" tag stays lit. Scrubbing unpins; a **Latest** button (and `End`) re-pins. This is the auto-follow logic already in `renderFilmstripView()`'s `wasAtRightEdge`, moved to where it matters.

### Where it lives

**The Default view adapts; no fifth view.** Reasons: (1) in a brightfield-only run the stage is the whole body, and a view switcher with three dead buttons and one live one is worse than none; (2) in a mixed run the operator wants dish and embryos together, not in separate tabs; (3) SPEC R4 — the current `dic-strip` is the stage's *folded* rendering, so keep it as that: in a mixed run the stage opens folded (one row of thumbnails, 110 px, newest pinned) with a `Open ⌄` affordance that expands it in place to the layout above and pushes the embryo tiles down. In a brightfield-only run it opens expanded and cannot fold. Nothing moves; height changes.

## 3. Do the four views earn their keep?

| View | What it shows | Unique? |
|---|---|---|
| Default | 48 px emoji rail + reasoning panel (sparkline, trace, chat) | Trace and chat live only here |
| Board | per-embryo row: stage, clock, stereo, pace, ETA, sparkline, alert; row expands into the same detail panel | Dense; the only view that scales past ~6 embryos |
| Film | thumbnail per timepoint per embryo, DIC row, click → same detail panel | The only view with pictures over time |
| Vitals | confidence-over-time SVG per embryo + ON TRACK / SLOW / ARRESTED badge → same detail panel | Nothing: pace and arrest are Board columns; the chart is Board's sparkline with a y-axis |

All four end in `renderDetailPanel()`. The real content is: pictures, a stage bar, pace/alert, and the trace. That is one view with tiles plus one dense table.

**Recommendation: two views, `Watch` and `Table`.**

- **Watch** (replaces Default and Film): the stage, then embryo tiles (§4). Each tile's thumbnail scrubs through its own timepoints — that *is* the filmstrip, one embryo at a time, without the matrix. Selecting a tile opens the detail drawer beneath the tiles (the existing reasoning panel, trimmed).
- **Table** (replaces Board and Vitals): the current Board rows. The Vitals chart moves into the expanded row, where its y-axis has room. Drop the `ON TRACK` badge; the pace column already says it with `!`/`!!` prefixes.
- **Drop Film and Vitals.** Keyboard `1`/`2`. Delete `renderFilmstripView`, `_filmDicRow`, `renderVitalsView`, `_renderVitalsStrip`, their CSS (`.filmstrip-*`, `.vitals-*`, ~300 lines), and `dashboardConfig.filmstrip`.

With four or fewer embryos — every real run so far — Watch alone does the job. Table stays because a 12-embryo dish is a plausible future and a table is the honest answer to it.

## 4. Embryo tiles replacing the rail, and the detail drawer

The rail shows "E1" under a 🥨. The emoji carries no order (a pretzel is not visibly later than a comma), renders differently per OS, and is the only place the stage appears at a glance. The Sessions tab already solved this: `stageBar()` paints one cell per timepoint on `stageColor()`'s viridis ramp, so the whole history and its direction read as a gradient.

### `.embryo-tile` (160 px wide, grid, `gap: 12px`)

```
┌──────────────────┐
│ ● E1  dauer-ctrl │   status shape + id + nickname (STATUS SHAPES: ● going, ○ done, ◐ paused, ◆ error)
│ ┌──────────────┐ │
│ │  thumbnail   │ │   96 px square, latest projection; pointer scrubs timepoints; "t112" cap
│ └──────────────┘ │
│ ▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮▮ │   stage bar, 8 px, one cell per timepoint, stageColor ramp
│ ■ 2-fold · t112  │   current stage: 10 px swatch of its ramp colour + name; no emoji
│ next in 1:30     │   one live line; "done · hatched t140" when complete
└──────────────────┘
```

Selected tile: `border-color: var(--accent)`. Sort: going first, then by id (as now). An embryo with no evaluations yet shows a grey bar and "acquiring · t3".

### The detail drawer (`.embryo-detail`, replaces `.reasoning-panel`)

Keeps what only it has: `renderTimelineSparkline`, the inline trace, the chat. Changes:

- Header line becomes plain words: **"E1 · 2-fold since t98 · 112 timepoints"** with the swatch. Delete the `3 transitions · 112 evals · 112 tp` triple.
- The quick-jump badges (`🥨 Pretzel @ T98`) become clicks on the tile's stage bar: each colour change is a transition; clicking a cell scrolls the trace to that timepoint. A `.stage-bar-key` beneath names the first and last stage, as in Sessions.
- The 48 px rail, `getStageIcon()`, and `.embryo-rail-item` go.

Shape as well as colour for the stage: the swatch is a square; status is a circle/ring/diamond. Two glyph families, no collision.

## 5. Empty and error states — the copy

One idea per block. Say what is true now and what happens next. No emoji, no hourglasses. Buttons only where there is somewhere to go.

| State | Badge | Body |
|---|---|---|
| No run | ◌ **No run** (dashed hollow) | **Nothing is running.** Start a run from Calibration. Frames and embryos appear here as they are taken. `[Go to Calibration]` |
| Connecting | ◌ **Connecting** | **Connecting to the server.** |
| Brightfield, before first frame | ● **Running · brightfield** | *(in the stage, over a grey frame)* **First frame due in 0:42.** Then one every 10 minutes, under room light at 20 ms. — plus the references line (✓ or "No dark or flat for 20 ms under room light. Take references →") |
| Embryos, before first volume | ● **Running** | *(tiles present, grey thumbnails)* **First volume due in 1:58.** 4 embryos, one every 2 minutes. Stage calls start after the first volume. |
| Mixed, overview not yet taken | ● **Running** | *(folded stage row)* **First overview frame due in 9:40.** One every 10 minutes. |
| Paused | ◐ **Paused** (amber) | **Paused at 03:12, after frame 48.** Nothing is taken until you resume from Operations. |
| Frame failed, run continues | ◆ **Running** · amber line in facts | **Frame 49 failed: camera timed out.** The run continues; the next frame is due in 8:10. |
| Run failed | ◆ **Failed** (red) | **Stopped at 03:12: stage lost connection.** 48 frames are saved. Resume from Sessions when the rig is back. `[Open in Sessions]` |
| Complete | ○ **Complete · ended 09:12** | **Finished.** 112 frames over 18 h 40. `[Open in Sessions]` |
| Stopped early | ◐ **Stopped early · ended 03:12** | **Stopped after 48 of 112 frames.** `[Open in Sessions]` |
| Embryo with no evaluations | — | *(in drawer)* **No stage calls yet for E1.** The first comes after its first volume. |

Delete: "Select an embryo to view analysis history" (auto-select the first tile, as `reconcileWithServerState` already does), "Click on any embryo card in the left panel", "click a frame to view it", "No stage transitions yet", "Typical first detection: 2-4 hours after start", "Syncing with experiment data".

## 6. Density and hierarchy

**Remove**

- Two of four view buttons; the switcher becomes a two-segment control or, in a brightfield-only run, nothing.
- All emoji: rail icons, empty-state icons (`&#x1F52C;`, `&#x1F441;`, `&#x23F3;`), stage badges, detector icons.
- The fixed modal `#dic-viewer` and its three buttons.
- The `.embryo-list-header` ("Embryos" — the tab is called Embryos).
- The `TP · Active · Done · Dur · Next` labels; use nouns.
- `.dic-strip-hint`.
- The hover highlight on the whole header bar (`.embryos-header-bar:hover`): it is not a button.

**Enlarge**

- The frame: 110 px → the body. It is the experiment.
- The next-due countdown: 1.1 rem → 1.4 rem, mono. It is the one number that changes while you watch.
- Stage history: from a tooltip on a 48 px rail item to an 8 px bar across each 160 px tile.
- Status text: 0.85 rem → 0.95 rem, and it says *what* is running.

**Keep as is**: JetBrains Mono for facts; the STATUS SHAPES glyph rule; `renderDetailPanel`, the trace and the chat; the Board rows.

## Prioritised implementation

1. **Status and stats from the server, not from embryo count.** `renderStatusBadge()` reads `state.status` only; `renderSummary()` branches on `volumes` / `dic.enabled`; `reconcileWithServerState()` stores `serverState.volumes`, `serverState.dic`, `seconds_until_next_round`. Small diff, fixes the headline lie. Test: a RUNNING state with zero embryos and `volumes:false` renders "Running · brightfield" and a frame count.
2. **Empty-state copy.** Replace the `renderSmartEmptyState` table with §5; route brightfield-only runs to the stage's own waiting line, never to `no-embryos`. Delete the dead strings.
3. **The stage, in place.** New `renderStage()` owning `#dic-strip`'s DOM: big frame + facts + timeline with pointer scrub, arrow keys, play, Latest. Delete `openDicViewer`/`closeDicViewer`/`#dic-viewer`. Fold/unfold by `volumes`.
4. **Frame facts and correction on the API.** `/api/dic/frames` adds `exposure_ms`, `light`, `led_intensity_pct`, `reference`; `/api/dic/frames/{stem}.png?corrected=1` applies the matching record. The stage calls `/api/brightfield/references` once per run spec for the ✓ / prompt line.
5. **Embryo tiles.** `renderEmbryoCards()` → `renderTiles()` using `stageColor` (load `stage-colors.js` on this page), a scrubbing thumbnail via the existing projection route, and the Sessions `.stage-bar` CSS lifted into `main.css`. Delete `getStageIcon`, `.embryo-rail-item`.
6. **Trim the drawer header** to one sentence; stage-bar clicks replace quick-jump badges.
7. **Two views.** Rename Default → Watch, Board → Table; move the Vitals chart into the Board expanded row; delete Film and Vitals code and CSS; `dashboardConfig` loses `filmstrip` and `vitals`.
8. **Header polish**: nouns for stats, larger countdown, drop the hover, drop the list header and hint.

Items 1–2 are a day and remove the observed failure. 3–4 make the brightfield run watchable. 5–8 bring the tab in line with Sessions.

## Files

- `docs/superpowers/mockups/2026-10-06-embryos-study-monitoring.md` (this report)
- `docs/superpowers/mockups/2026-10-06-embryos-monitoring.html` (mockup: brightfield-only, mixed, finished; empty-state copy beneath)
