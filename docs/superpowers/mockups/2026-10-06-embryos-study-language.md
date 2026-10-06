# Embryos tab — design-language and accessibility study

Date: 2026-10-06. Scope: the Embryos tab (`index.html:255-340`, `embryos.js`,
its rules in `main.css`) read against the token palette (`main.css:1-110`),
the STATUS SHAPES section (`main.css:11659-11759`), the colorblind audit
(`docs/colorblind-audit.md` §1-3, §7), the Operate layer (`operate.css`), the
Sessions tab (`review.css`) and the shell (`shell.css`). Read-only: nothing
under `gently/` or `tests/` was changed. Line numbers are from today's tree.

## Headline

The tab predates the design language and shows it. It carries its own status
badge, its own stats row, four empty-state components, emoji as stage and
status icons, eleven font sizes, a mixed rem/px spacing rhythm, three stage
colour systems in one view (ramp, emoji, hardcoded hex with purple), and
clickable `<div>`s where Operate uses `<button>`. The redesign should not
invent anything: every idiom it needs exists in Operate, Sessions or the shell.

Two things are wrong, not just inconsistent, for a brightfield-only run: the
status badge and the embryo list both say "No active timelapse" while DIC
frames are arriving (`embryos.js:1801-1803`, `index.html:310`), and the
"waiting" copy promises a volume that such a run never produces.

---

## 1. Idioms the Embryos tab uses that the rest of the app does not

| # | Idiom | Where | Adopt instead |
|---|---|---|---|
| 1 | Emoji as stage icons (🥚🌱🌙🔄🔁🔃🥨🐣🐛⏸️⬜, 🔬 fallback) | `embryos.js:1948-1964` `getStageIcon`; used at `:585,616` (board), `:1931,1941` (rail), `:2107,2139` (header, quick-jump), `:3018,3027` (eval dots) | The ordinal ramp as a **fill swatch** plus a text label — the `stageBar()` pattern (`review.js:633-640`, `.stage-bar` `review.css:654-656`). No glyph per stage; a stage is ordinal, the ramp carries order. |
| 2 | Emoji as empty-state icons (🔍 👁 📊 🎬 📈 🔬 ⏸ ⏳) | `index.html:315`; `embryos.js:526, 823, 989, 2047, 2889-2910` | None. `.op-empty` (`operate.css:654-662`) is a dashed hairline box with one line of text and an optional `.op-btn`. If an icon is wanted, the inline SVG idiom of `.no-session-selected svg` (`review.css:187`), never a font emoji. |
| 3 | Own stats row `.header-stat` / `.stat-value` (1.1rem accent blue) / `.stat-label` | `main.css:4051-4075`; `embryos.js:1826-1858` | `.op-num` + `<i>unit</i>` (`operate.css:418-425`; markup at `index.html:450-452`), labels in `.op-label` (`operate.css:397-404`). Numbers in ink, not accent: accent means selected/done. |
| 4 | Own run-status badge `.timelapse-status` | `main.css:4086-4138`; `index.html:264-267`; `embryos.js:1792-1810` | It duplicates `.header-status` (`main.css:248-292`, same classes, already in STATUS SHAPES) and the shell strip (`body[data-run]`, `shell.css:222-275`), which shows the same state on every tab. Drop it; if a per-tab line is wanted, reuse `.header-status`. |
| 5 | Four empty-state components: `.empty-state`, `.reasoning-empty`, `.board-empty`, `.smart-empty-state` | `main.css:1759, 4738-4755, 3591-3600, 7938-7990` | One: `.op-empty`. |
| 6 | Own badge styles: tinted pills `.vitals-status.*`, green chips `.quick-jump-badge`, `.reasoning-condition`, unstyled `.filmstrip-terminated-badge` | `main.css:3842-3852, 5060-5085, 4790-4796`; `embryos.js:849-862` (no CSS exists for `.filmstrip-terminated-badge` or `.filmstrip-row.terminated`) | One outline pill: `.session-run-badge` (`review.css:~262-272`: 11px, `border:1px solid currentColor`, dashed for interrupted/stopped) — same shape as the shell's `.diag-badge` (`shell.css:277-290`). |
| 7 | Own list header `.embryo-list-header` (0.6rem, centred) | `main.css:4476-4486`; `index.html:308` | `.op-erail-head` (`operate.css:108-118`): 0.72rem, uppercase, 0.06em, with the count in `.op-num`. |
| 8 | Own section title `<h2 class="embryos-title">` (1.1rem) | `main.css:4078-4083`; `index.html:263` | Drop: the rail already names the tab (`index.html:50`). Sub-sections use `.op-block-head` (`operate.css:397-405`) or `.session-plan-title` (`review.css:~287`). |
| 9 | Own strip title via opacity (`.dic-strip-title` opacity .75, `.dic-strip-hint` .55) | `main.css:11457-11459` | `.session-dic-head` (`review.css:~299-306`) — same component, already in Sessions. Colour through `--text-muted`, never opacity. |
| 10 | Status by left stripe on rail items, default green | `main.css:4507-4536` | `.op-tcard.st-*` (`operate.css:634-646`): 2px stripe, done = `opacity:.6`; plus the glyph (see §2). |
| 11 | Hardcoded `rgba(96,165,250,…)` for hover/selection/glow | `main.css:3540, 3693, 4525, 4529, 5207` | `--accent-soft` (`main.css:35`, light `:87`) and `color-mix(in srgb, var(--accent) …)`. |
| 12 | Five independent pulses: status dot glow+scale, `.board-alert-critical`, `.eval-dot.hatching`, `.filmstrip-cell.pending` (two keyframes), `.ambient-pulse` | `main.css:4100-4104, 4138; 3578; 5293-5296; 3699-3743; 3435-3470` | Operate's rule: only the HALT state pulses (`operate.css:198-210`). The shell already pulses the run dot (`shell.css:229-234`). Keep that one; the rest become static shape + text. |
| 13 | Clickable `<div>`/`<span>` (rail item, board row, film cell, eval dot, quick-jump) | `embryos.js:1937-1944, 563, 877, 787, 3022-3027, 2106-2114` | `<button type="button">`, as the DIC strip already does (`embryos.js:272`) and every `.op-btn`. |
| 14 | `style="display:none"` on view containers | `index.html:321-325`; `embryos.js:371` | The `hidden` attribute (Operate panes, `index.html:446, 560`; guard rule `operate.css:22`). |
| 15 | Eleven font sizes: .55 .6 .65 .7 .75 .8 .85 .9 .95 1 1.1rem, plus 3rem icons | `main.css:3484-3880, 4024-4135, 4476-4555, 4738-4796, 7938-7975` | Operate's scale (§3). |
| 16 | Spacing in rem fractions (.25 .3 .35 .5 .75 1 1.25 1.5 2 3) and px (2 3 4 6 8 10) | same ranges | `--op-1..--op-6` (`operate.css:39-40`). |
| 17 | Fonts: system stack, `monospace`, `'SF Mono'`, JetBrains Mono only on DIC captions | `main.css:120, 4131, 290, 11458, 11477, 11491, 11556` | `--op-ui` (Inter Tight) and `--op-mono` (JetBrains Mono), both already loaded globally (`index.html:12`). |
| 18 | Whole header bar `cursor:pointer` with no control | `main.css:4036` | A real `<button aria-expanded>` if it collapses; otherwise remove the cursor. |
| 19 | Dead rulesets: `.embryo-card*` (full, sidebar, minimal) | `main.css:4188-4310, 4560-4700`; comment at `embryos.js:1947` "not used" | Delete in the redesign. |

---

## 2. Colour and shape

Vocabulary (`main.css:17-24`): green live, amber paused/warning, blue
done/selected, red error, grey idle; purple means AI-authored only. Shapes
(`main.css:11659-11672`): live filled circle, paused half-filled, done ring,
error diamond, idle hollow dashed; text-only states carry `✓ ✕ ! ●`.

| Place | Problem | Fix |
|---|---|---|
| `.board-status-dot` `embryos.js:613`, `main.css:3544-3551` | The same `●` is written for running, complete and error; only colour differs. The STATUS SHAPES comment (`main.css:11670`) claims the board dot "already receives a distinct character from the JS" — it does not. | Write `●` running, `✓` complete, `✕` error (the set at `main.css:11664`). `embryos.js:2097-2098` already does this for the reasoning header with `✔ ✘`; use `✓ ✕` in both. |
| `.embryo-rail-item` default vs `.complete` `main.css:4520, 4534` | Running = solid green stripe, complete = solid blue stripe. Paused is dashed and error double (`main.css:11750-11758`), but running vs complete is colour alone. | Complete = `opacity:.6` as `.op-tcard.st-done` (`operate.css:646`) plus `✓` before the label. |
| `.timelapse-status.idle` `embryos.js:1800` | Class is emitted; no rule exists. Idle renders as a solid grey 10px dot — the live shape in grey. | Add `.timelapse-status.idle .status-indicator` to the hollow-dashed list at `main.css:11735`, or emit `stopped`. (Moot if the badge goes, §1 #4.) |
| `.board-col-pace.pace-slow-bad` `embryos.js:697` + `main.css:11744` | JS writes `⚠ 2.0×`; CSS prepends `!! ` → `!! ⚠ 2.0×`. Same at `embryos.js:605-608` (`⚠ arrested`, `⚠ slow`) and `:676` (stereo `⚠`). | Remove every `⚠` from the JS. The CSS glyph is the cue. Give `.board-alert-critical` `✕ ` and `.board-alert-warn` `! ` in the same block. |
| Stage colour, three systems | (a) `stageColor()` ramp as **text/border colour** at `embryos.js:616, 870, 884, 1123, 1143, 1151`; (b) emoji `getStageIcon`; (c) `.eval-dot.stage-*` hardcoded hex from the old hue palette, with **purple** for 2fold/3fold (`main.css:5273-5281`) and green for hatching (`:5288-5296`). Purple is AI-only; green is live. The class is built as `stage-${stage.replace('.','')}` (`embryos.js:3016`), so `1_5_fold` → `stage-1_5_fold`, which no rule matches. | One system: the ramp as a fill swatch beside a `--text` label (the `.stage-bar` idiom). Delete `main.css:5253-5296`. For eval dots set `style="--stage:${stageColor(s)}"` and `border-color: var(--stage)`. Hatching is a stage, not a status: ramp yellow, no glow. |
| Ramp as text colour | `#440154` (early) on `#161b22` ≈ 1.2:1; `#3b528b` (bean) ≈ 2.5:1. Fails for `.board-stage-badge`, `.filmstrip-stage-label` (0.55rem), `.vitals-stage`. Sessions never does this — it uses the ramp only as a fill. | Rule: **the ramp is a fill, never a text colour.** |
| `.quick-jump-badge` `main.css:5060-5075` | Green tint and border on navigation chips. Green = live. | `.session-run-badge` outline in `--text-muted`, or `--accent-soft` background (blue = selected). |
| `.reasoning-embryo-name` `main.css:4786`, `.header-stat .stat-value` `main.css:4066` | Names and counts in accent blue. Blue = done/selected. | `--text` (Operate `.op-num` is ink). |
| `.ambient-pulse.warning/.critical` `main.css:3450-3458`, `embryos.js:472-481` | A tinted frame around the whole app, colour and animation only, no text (audit §2 "filled regions with no label"). Under reduced motion the warning tint is all that remains. | Drop it. The Alert column and the vitals badge already carry the same fact in words. |
| `.vitals-status.vitals-ok` `main.css:3850` | Warn and critical get `!` and `✕` (`main.css:11739-11747`); ok gets nothing but green tint. | `✓ ` before ok, or no colour on ok at all (on-pace is the default; do not decorate the default). |
| `.filmstrip-cell.pending` `main.css:3699-3743` | Correct by the audit (border + dot + italic), but `!important` twice to beat the inline stage colour. | Goes away once the stage colour is a swatch, not an inline border. |
| `.filmstrip-terminated-badge`, `.filmstrip-row.terminated` `embryos.js:855-862` | No rule exists; the badge renders as plain text and the row is not marked. | `.session-run-badge.is-stopped` (dashed outline, muted). |
| Light theme | `rgba(96,165,250,…)` at `main.css:3540, 3693, 4525, 4529, 5207` has no light value. | `--accent-soft`. |
| DIC strip `main.css:11449-11478` | Shape-clean: no status colour. `.dic-frame-blank` is a black box with no text when a frame has no preview. Captions use opacity for colour. | Caption reads "no preview"; `--text-muted` not opacity. |
| DIC viewer `main.css:11479-11497` | Hardcoded `#e6e8ec` / `#000` / `rgba(0,0,0,.82)` — acceptable, a scrim is theme-independent. `:disabled` nav at `opacity:.2` is colour-only but the button is also inert; fine. | No colour change. See §5 for focus and dialog semantics. |

---

## 3. Typography and spacing

### What each surface does

| | Operate (`operate.css`) | Sessions (`review.css`) | Shell (`shell.css`) | Embryos |
|---|---|---|---|---|
| UI face | Inter Tight (`--op-ui`, :43) | inherits system (`main.css:120`) | Inter Tight on controls (`:150, 161, 169`) | system |
| Numeral face | JetBrains Mono (`--op-mono`, :42) tabular | ui-monospace on the stage key (:656) | ui-monospace in the console (:196) | `monospace`, `'SF Mono'`, JetBrains Mono on DIC only |
| Label | .62rem / 600 / .10em / uppercase (:397) | 11px / 600 / .08em / uppercase (:287, :300) | 10-11px / .1em / uppercase (:21, :118) | .6rem centred (:4476); .7rem .05em (:3490); .75rem .5px (:4071); .7rem .08em (:11457) |
| Caption | .68rem / 1.45 (:426) | 12-13px | 12.5px | .55 .6 .65 .7 .75rem |
| Control | .72rem (:539), pad 8×12 | 14px tabs, 13px buttons | 12.5-13.5px | .75rem, pad 4×12 (:3380) |
| Number | .74rem mono (:420) | 13px | tabular 12.5px | 1.1rem accent (:4064) |
| Name | .74rem / 600 (:648) | 600, 14px | 13.5px / 600 | .9rem (:3555), 1rem / 700 (:3828), 1.1rem (:4786) |
| Floor | .58rem meta only (:651) | 11px | 10px | .55rem (8.8px) on rail and film labels |
| Spacing | one 4px scale `--op-1..6` (:39) | 4 6 8 10 12 14 16 24 px | 2 4 6 8 9 10 12 14 16 px | rem fractions + px, no scale |
| Surface | 1px hairline, radius 6, no shadow (:13, :389) | 1px, radius 10-12 | 1px, radius 8-14 | radius 4, 6, 8, 10, 12; glows (:4197, :3693) |

Three sections, three scales. Sessions and the shell agree loosely (11-14px,
system face); Operate is the only one with a declared scale, and it is the
surface built for the rig.

### One scale for the redesign

Adopt Operate's. It is already tokenised, already loaded, and the Embryos tab
is the other live-run surface, so the two should read as one instrument.

| Role | Size | Weight / case | Face | Source |
|---|---|---|---|---|
| Block label | .62rem | 600, uppercase, .10em | `--op-ui` | `.op-block-head` |
| Caption, empty text | .68rem / 1.45 | 400 | `--op-ui` | `.op-cap`, `.op-empty` |
| Control | .72rem | 500 | `--op-ui` | `.op-btn` |
| Number | .74rem, tabular | 500 | `--op-mono` | `.op-num` |
| Name (embryo id) | .74rem | 600 | `--op-ui` | `.op-tcard-name` |
| Meta (time, count) | .62rem | 400 | `--op-mono` | `.op-tcard-meta` raised from .58 |

Five sizes replace eleven. Floor is .62rem (9.9px) for uppercase labels and
mono meta; anything read as a sentence is .68rem or above. Nothing is 3rem.

Spacing: `--op-1..--op-6` (4 8 12 16 20 24). Row padding `--op-2 --op-3`,
block gap `--op-2`, pane gap `--op-3`. Radius 6 everywhere, 1px hairline in
`--op-rule`, no glow, no shadow.

Hoist: the eight custom properties at `operate.css:39-43` are declared on
`.operate`. Move them to `:root` in `main.css` (one block; `operate.css` keeps
working) so the Embryos tab can use `var(--op-2)` without being inside
`.operate`. Rename is optional and not worth the diff.

---

## 4. Copy

Voice from `docs/atrium/SPEC.md`: declarative, short, says what happens next.
No exclamation marks, no emoji, no "Click on any…", units with a space, 24-hour
time, lower-case state words inside a sentence.

**BF** marks strings that are wrong for a brightfield-only run (DIC frames
arriving, no registered embryos, no volumes).

### Header

| Where | Now | Rewrite |
|---|---|---|
| `index.html:263` | Embryo Monitoring | *(remove; the rail says Embryos)* |
| `index.html:266`, `embryos.js:1803` | No active timelapse **BF** — shown whenever `embryos.length === 0`, even while RUNNING | Key on `state.status` only. idle: `No run`. running with no embryos: `Running · overview only` |
| `embryos.js:1805-1808` | Running / Paused / Completed / Stopped | Running / Paused / **Done** / Stopped (vocabulary says done) |
| `index.html:268` title | Default view (1) | Rail and analysis (1) |
| `index.html:269` title | Status Board (2) | Board (2) |
| `index.html:270` title | Filmstrip (3) | Film (3) |
| `index.html:271` title | Vital Signs (4) | Vitals (4) |
| `index.html:268` label | Default | Rail |
| `embryos.js:1838` | TP | timepoints |
| `embryos.js:1842` | Active | active |
| `embryos.js:1846` | Done | done |
| `embryos.js:1851` | Dur | elapsed |
| `embryos.js:1856` | Next | next in |
| `embryos.js:1866, 1873` | `--:--` | — |
| `embryos.js:1822` | stats hidden when no embryos **BF** (elapsed vanishes during an overview-only run) | Show elapsed and frame count whenever `status !== IDLE`. |

### DIC strip and viewer

| Where | Now | Rewrite |
|---|---|---|
| `index.html:285` | DIC overview | DIC overview *(keep)* |
| `index.html:287` | click a frame to view it | Opens full size. Arrow keys step through. *(or remove: the frames are buttons)* |
| `embryos.js:264, 780` | 3 frames | 3 frames *(keep)* |
| `embryos.js:272` title | Open frame 3 | Frame 3, 14:12 |
| `embryos.js:274` cap | `3 · 02:12 PM` | `3 · 14:12` (`hour12:false`) |
| `embryos.js:291-292` | DIC overview · frame 3 of 12 · 10/6/2026, 2:12:05 PM · 1200, 340 µm | Frame 3 of 12 · 14:12:05 · x 1200 y 340 µm |
| `embryos.js:277` | *(blank box when no thumb)* | no preview |
| `index.html:293` aria-label | DIC overview frame | DIC overview, full size |
| `embryos.js:778-779` | DIC / overview | DIC / overview *(keep)* |

### Rail and empty states

| Where | Now | Rewrite |
|---|---|---|
| `index.html:308` | Embryos | Embryos `<b class="op-num">N</b>` (as `op-erail-head`) |
| `index.html:310` | No active timelapse **BF** | idle: `No embryos`. running: `Overview only. Embryos appear when a volume arrives.` |
| `index.html:316` | Select an embryo to view analysis history | Select an embryo. Its stage calls appear here. |
| `embryos.js:2048-2051` | Select an embryo to view its detection analysis / Click on any embryo card in the left panel | Select an embryo on the left. |
| `embryos.js:1932` | Acquiring | no call yet |
| `embryos.js:1940` title | `embryo_1 — Bean — 12 TP` | `embryo_1 · bean · 12 timepoints` |
| `embryos.js:1931` | 🔬 | *(swatch in `--text-muted`)* |
| `embryos.js:526, 823, 989` | No embryos to display | No embryos yet. *(film: the DIC row already renders; board and vitals add:)* The overview is on Film. |
| `embryos.js:2890-2892` | No embryos yet / Start a timelapse acquisition to begin tracking embryos. Configure your experiment in the Calibration tab. / Go to Calibration **BF** (shown mid-run); marking now lives on Operate (`index.html:255-256`) | No embryos registered. / Mark embryos on Operate and start the run. They appear here as volumes arrive. / Open Operate — verify `switchTab('calibration')` still resolves before keeping that target |
| `embryos.js:2896-2898` | No detections yet / The AI will analyze each timepoint and notify you when developmental events are detected. Typical first detection: 2-4 hours after start. | No stage calls yet. / Each volume is read as it arrives. The first call follows the first volume. |
| `embryos.js:2902-2904` | Experiment not running / No active timelapse. Configure and start an experiment to begin automated embryo monitoring. / Go to Calibration | No run. / Start one from Operate. / Open Operate |
| `embryos.js:2908-2910` | Waiting for first acquisition / The timelapse has started. The first volume should arrive shortly. **BF** | Run started. / The first volume arrives at the end of the first interval. *(brightfield-only: "frame", not "volume")* |

### Board

| Where | Now | Rewrite |
|---|---|---|
| `embryos.js:539-546` headers | Embryo · Stage · Clock · Stereo · Pace · ETA · Progression · Alert | Embryo · Stage · In stage · Reference · Pace · To hatch · Stages · Alert |
| `:541` title | Clock time in current stage | Time in this stage |
| `:542` title | Stereotypic developmental position (20°C reference) | Reference position at 20 °C |
| `:543` title | Clock / stereotypic time — 1.0× means on reference pace | Clock over reference. 1.0× is on pace. |
| `:544` title | Estimated clock-time to hatch, pace-corrected | Time to hatch at the current pace |
| `:605` | ⚠ arrested | arrested *(CSS adds the glyph)* |
| `:607` | ⚠ slow 1.5× | slow 1.5× |
| `:676` title | Clock ran past expected stage duration | Past the reference duration |
| `:690` | 1.0× | 1.0× *(keep)* |
| `:693` | 1.5× slow | 1.5× *(the class says slow; the glyph says it)* |
| `:697` | ⚠ 2.0× | 2.0× |
| `:704` | done | hatched |
| `:708` | ~3.2h | 3.2 h |
| `:1979` | Unknown | — |
| `:1980` | Empty | no object |

### Film

| Where | Now | Rewrite |
|---|---|---|
| `embryos.js:849` | HATCHED? | no object |
| `:849` | STOPPED | stopped |
| `:852` title | Terminated — <reason> | Stopped: <reason> |
| `:861` | 12 eval | 12 calls |
| `:873` | analyzing… | reading |
| `:874` | … | pending |
| `:877` title | `T12 — Bean — high` | `t12 · bean · high confidence` |
| `:787` title | `DIC overview, frame 3 — 14:12` | `DIC frame 3 · 14:12` |

### Vitals

| Where | Now | Rewrite |
|---|---|---|
| `embryos.js:1055` | ON TRACK | on pace |
| `:1057` | ARRESTED | arrested |
| `:1059` | SLOW 1.5x | slow 1.5× |
| `:1048` | 0.7x | 0.7× |
| `:1047` | unknown | — |
| `:1066-1067` | ~3.2h / done | 3.2 h / hatched |

### Analysis panel

| Where | Now | Rewrite |
|---|---|---|
| `embryos.js:2134` | Click on any evaluation dot above to view the full VLM analysis | Select a timepoint to read its analysis. |
| `:2141-2143` | 3 transitions · 40 evals · 42 tp | 3 transitions · 40 calls · 42 timepoints |
| `:2106` title | Jump to Bean | First bean call |
| `:2113` title | Jump to detection | Detection |
| `:2147` | Folder | Folder *(keep)* |
| `:3020` title | `T12: Bean - HATCHING!` | `t12 · hatching` |
| `:3021` title | `T12: Bean` | `t12 · bean` |
| `:2514` | DETECTED / Not detected | detected / not detected |
| `:2532` | Follow-up | Follow-up *(keep)* |
| `:2537` | Ask a follow-up about this timepoint… | Ask about this timepoint |
| `:2541` | Send | Send *(keep)* |
| `:2497` | Projections | Projections *(keep)* |
| `:2654, 2685, 2697` | Error: … | Error: … *(keep; `.settings-result.is-err` style gives it `✕`)* |

---

## 5. Accessibility

### Keyboard reachability

| Element | Now | Fix |
|---|---|---|
| Rail item `embryos.js:1937-1944` | `<div>` with click; no `tabindex`, no role | `<button type="button" aria-pressed="{selected}">` |
| Board row `embryos.js:563, 611` | `<div>` with click | `<button class="board-row">` (or `role="row"` + `tabindex=0` + Enter/Space). Button is the shorter diff. |
| Film cell `embryos.js:787, 877` | `<div>` with click | `<button>`; the strip's own DIC frame is already a button (`:272`) — same object, same element. |
| Eval dot `embryos.js:3022-3027` | `<div onclick>` | `<button>` |
| Quick-jump `embryos.js:2106, 2113` | `<span onclick>` | `<button>` |
| Vitals point `embryos.js:1123` | SVG `<circle r=4>` with click; 8px target; unreachable by keyboard | Make the whole `.vitals-strip` row the button and open the latest timepoint; or wrap each point in `<a tabindex=0>` with a transparent 22px hit circle. |
| View switcher `index.html:268-271` | Buttons, with `1-4` shortcuts (`embryos.js:349-363`, input-guarded) | Add `aria-pressed`. Shortcuts are fine. |
| DIC strip `index.html:283-290`, `embryos.js:272` | `<button>` with `title` and `img alt` — correct | Keep. |

### Focus visibility

No `:focus-visible` rule exists for `.view-btn`, `.board-row`,
`.filmstrip-cell`, `.embryo-rail-item`, `.eval-dot`, `.quick-jump-badge`,
`.filter-btn`, `.dic-viewer-nav`, `.dic-viewer-close`, `.empty-action`
(grep across `static/css`). Only `.dic-frame` has one (`main.css:11475`).
Copy `operate.css:587-592` into the Embryos block:
`outline: 2px solid var(--accent); outline-offset: 2px`.

### The viewer dialog `index.html:293-301`, `embryos.js:279-329`

Has `role="dialog"` and `aria-label`, and Escape/arrow keys. Missing:

- `aria-modal="true"`.
- Focus is not moved into the dialog on open (`openDicViewer` has no
  `.focus()`), not returned on close, and Tab leaves the dialog into the page
  behind the scrim.
- The page behind is not `inert`.

Shortest correct fix is the platform one: make `#dic-viewer` a `<dialog>` and
call `showModal()` / `close()`. That gives focus trap, focus return, Escape,
`aria-modal` and the backdrop for free, and deletes the document-level keydown
and the click-outside handler. Keep the arrow-key handler on the dialog
element. `.dic-viewer-nav` and `.dic-viewer-close` need `min-width:48px;
min-height:48px` (the grid column is already 48px, the buttons are not) and a
focus ring.

### Live regions

Status text, stats, and alerts change silently. `#timelapse-status` (or its
replacement) gets `aria-live="polite"`. An arrested alert
(`embryos.js:605`) gets `role="status"`. Nothing needs `assertive`.

### Reduced motion

`main.css:2787-2792` zeroes every animation and transition globally, so the
tab complies. Two consequences to design for:

- `.ambient-pulse.hatching` (three flashes) disappears entirely; the warning
  and critical frames keep only a faint tint. Dropping the component (§2)
  removes the question.
- `.filmstrip-cell:hover { transform: scale(1.15) }` (`main.css:3681-3689`)
  snaps instead of animating. Replace the scale with a border change; a
  thumbnail that grows over its neighbours is a layout change, not a hover.

### Contrast of captions on and beside images

| Text | Colour | Background | Approx. ratio | Verdict |
|---|---|---|---|---|
| `.dic-frame-cap` (0.6rem, opacity .7) `main.css:11477` | `#c9d1d9` × .7 | `#161b22` card | ≈ 6:1 at 9.6px | Passes but too small; make it .62rem mono in `--text-muted` at full opacity (4.9:1) |
| `.dic-strip-hint` (opacity .55) `:11459` | inherits × .55 | card | ≈ 4:1 | Fails for small text; `--text-muted` full opacity |
| `.filmstrip-stage-label` (0.55rem, ramp colour) `:3767` | `#440154`…`#fef3c7` | card | 1.2:1 to 15:1 | Fails for early stages; label in `--text`, ramp as a 2px bar under the thumb |
| `.vitals-stage`, `.board-stage-badge` (ramp colour) | same | card | same | same fix |
| `.dic-viewer figcaption` `:11491` | `#e6e8ec` | 82% black scrim | > 12:1 | Passes; it sits below the image, not on it. Keep it there. |
| `.filmstrip-placeholder` `T12` (0.6rem) `:3756` | `--text-muted` | `--bg-hover` | ≈ 4.3:1 | Marginal; .62rem mono |

Note `embryos.js:884`: `style="color:${stageColor}"` interpolates the
function source, not a colour, so the stage label colour is never applied
today. The contrast fix above makes the bug moot.

### Touch targets on a lab display

| Element | Now | Target |
|---|---|---|
| `.view-btn` `main.css:3379` | ≈ 24px tall | `min-height:32px` |
| `.eval-dot` `:5178` | 20px, 3px gap | 28px, 4px gap |
| `.vitals-point` | 8px | see above |
| `.dic-viewer-nav/close` `:11492` | glyph only | 48 × 48 |
| `.quick-jump-badge` `:5060` | ≈ 22px | 32px |
| `.embryo-rail-item` `:4507` | 48 × 48 | fine |
| `.filmstrip-cell` | 56 × 56 | fine |
| `.board-row` `:3526` | ≈ 40px | fine |

### Emoji

Screen readers announce them: "seedling", "crescent moon", "pretzel",
"microscope". None has `aria-hidden`. The §1 replacement (swatch + text)
removes them; until then, `aria-hidden="true"` on every `.rail-icon`,
`.stage-icon`, `.eval-dot-icon`, `.reasoning-empty-icon`, `.empty-icon`.

### Information only in `title`

Board column meanings (`embryos.js:541-544`), stereo overdue (`:676`),
termination reason (`:852`), rail item details (`:1940`) are hover-only.
Touch and keyboard users never see them. Put the board key in an `.op-cap`
line under the table; put the termination reason in the pill; the rail
details are on the selected panel anyway.

---

## 6. Style guide for the redesign

**Components to reuse — do not write new ones**

- Panel: `.op-block` + `.op-block-head`. Side rail: `.op-erail-head` + list.
- Button: `.op-btn`, `.op-btn-quiet`, `.op-btn-primary` (one per surface).
- Number with unit: `<b class="op-num">12</b><i>timepoints</i>` (units never uppercased, `operate.css:208-214`).
- Caption: `.op-cap`. Empty state: `.op-empty`, one sentence, optional one button.
- Status pill: `.session-run-badge` (outline, `currentColor`, dashed when stopped/interrupted).
- Stage: `.stage-bar` fill swatch (ramp from `stage-colors.js`) + text label. Never emoji, never ramp as text colour.
- Image strip: `.session-dic-head` + `<button class="dic-frame">`. Viewer: `<dialog>`.
- View switcher: the existing `.view-switcher` (shared with Operate), with `aria-pressed` and a focus ring.
- Run state: the shell strip (`body[data-run]`). The tab does not repeat it.

**Tokens**

- Colour: `--accent-green` live, `--accent-amber` paused/warn, `--accent` done/selected, `--accent-red` error, `--text-muted` idle. `--accent-soft` for hover/selection tints. `--accent-purple` only for AI-authored text. No hex in the Embryos block.
- Surface: `--bg-card`, 1px `--border`, radius 6, no shadow, no glow, no gradient.
- Spacing: `--op-1` 4 · `--op-2` 8 · `--op-3` 12 · `--op-4` 16 · `--op-5` 20 · `--op-6` 24, hoisted to `:root`.

**Type**

Inter Tight for UI, JetBrains Mono for every number and timestamp, tabular.
Five sizes: .62 label (uppercase, .10em) · .68 caption · .72 control · .74
number and name · .62 mono meta. Nothing smaller than .62rem; nothing above
.74rem except the viewer caption.

**Status shapes**

One state = one colour = one shape = one glyph. Dots: ● live, ◐ paused,
○ done (ring), ◆ error, dashed ○ idle. Text: `● ✓ ! ✕` prefixes from CSS,
never from JS. Stripes: dashed paused, double error, done fades to .6.
Only the shell's run dot pulses.

**Copy**

Declarative. Say what happens next. Sentence case; state words lower case
inside a line. `×` not `x`; `3.2 h` not `~3.2h`; 24-hour time; `t12` for a
timepoint. No exclamation marks, no emoji, no "Click on any". Key the
run-state copy on `state.status`, never on `embryos.length`, so an
overview-only run reads as running. Say "frame" when the run has no volumes.

**Interaction**

Everything clickable is a `<button>`. Every control has a visible focus ring
(`2px solid var(--accent)`, offset 2). Minimum 32px hit height, 48px on the
viewer. Facts live in text, not in `title`. `hidden`, not
`style="display:none"`.

---

## Incidental bugs found (fix regardless of the redesign)

- `embryos.js:884` — `${stageColor}` without a call; interpolates the function body.
- Stage keys disagree: `STAGE_TIMING`/`STAGE_ORDINAL` use `1_5_fold`, `2_fold`, `3_fold` (`embryos.js:84-95`); `getStageIcon`/`formatStageName` use `1.5fold`, `2fold`, `3fold` (`:1948-1982`); Python emits both spellings (11 × `"1.5fold"`, 6 × `"1_5_fold"`). Board Clock/Reference/Pace/To-hatch show `—` for every fold stage, and names render raw. `stage-colors.js:19-22` already has a `normalise()`; export it and use it in all three maps.
- `.eval-dot.stage-*` classes never match underscore keys (`embryos.js:3016`).
- `.timelapse-status.idle` has no rule (`embryos.js:1800`).
- `.filmstrip-terminated-badge` and `.filmstrip-row.terminated` have no rules (`embryos.js:855-862`).
- `main.css:4188-4310, 4560-4700` (`.embryo-card*`) are dead.
