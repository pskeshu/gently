# Colorblind audit of the gently web UI

Date: 2026-10-05. Scope: `gently/ui/web/static/{css,js}`, templates, `plots.py`,
`roles.py`, `connection_manager.py`, `app/theme.py`. Method: inventory of every
color token and literal, Machado (2009) simulation of protanopia, deuteranopia
and tritanopia plus greyscale, CIE76 ΔE between sibling colors (ΔE < 20 under
any simulation = confusable), WCAG contrast against the card background, and a
selector-by-selector read of where color is the only cue. Simulator:
`tools/cvd_check.py` (stdlib Python).

## Headline

The palette itself is mostly salvageable. The real problem is **structural**:
in most of the UI a state is shown by a bare dot, a 2–3 px left-edge stripe, or
a text-color change with nothing else. Fixing colors alone will not make the UI
colorblind-safe; the redundant cue (shape, dash, glyph, label) has to be added
where it is missing. The good news is the codebase already has the right
patterns in a few places (see "Patterns to copy").

## 1. Palette findings (simulation)

### Red/green pairs that collapse under deuteranopia or protanopia

| Component | Pair | Dark ΔE deutan | Light ΔE deutan |
|---|---|---|---|
| Operate: `--op-live` vs `--op-emit` (stream on vs LED/laser emitting) | #4ade80 / #f4485d | 16 | 21 |
| Operate: `--op-live` vs `--op-bad` | #4ade80 / #ef4444 | 23 | 19 |
| Operate light: `--op-warn` vs `--op-emit` | #b45309 / #dc2626 | – | **4** |
| Operate: `--op-emit` vs `--op-bad` (both red, by design) | | 11 | 10 |
| Main: `--accent-green` vs `--color-danger`/#f85149 | #4ade80 / #f87171 | **9** | 5 |
| Map zones: green vs red | 90,168,122 / 220,96,88 | 20 | 20 |
| Map zones: orange vs red | | 16 | 16 |
| 3D minimap: stage dot vs embryo dot | #ff5a4d / #6fae46 | **5** | – |
| timepoint-player: 3fold vs hatched | #22c55e / #ef4444 | 13 | – |
| plots.py: visible vs not-visible, fit vs best | #4CAF50 / #F44336 | 17 | – |
| roles.py: test vs calibration | #ff66cc / #00cccc | **3** | – |

Note: `--op-emit` has a comment "nothing else may use this", but to a deutan
viewer it is the same hue as `--op-live`. The `.op-btn.is-emitting` rule
(operate.css:561) turned out to be dead CSS: no JS sets that class. The real
emitting indicator is the `.lp-emit` pill (white text on red, with the word)
and `.lp-emit-unknown` (dashed), so emitting already carries a text cue. The
remaining Operate gap is `.op-btn.is-on` and `.is-armed`, which are
border+text colour only.

### Blue/purple/cyan pairs that collapse

| Component | Pair | Worst ΔE |
|---|---|---|
| Main: `--accent` vs `--accent-purple` | #60a5fa / #c084fc | **2** deutan |
| Main: `--accent-green` vs `--accent-cyan` | | 10 tritan |
| Ops timeline: `--ops-plan` vs `--ops-live` | #5aa9e6 / #22d3ee | 15 tritan |
| Ops timeline: `--ops-done` vs `--ops-live` | #34d399 / #22d3ee | 8 tritan |
| campaigns TYPE_COLORS: imaging vs genetics | #3b82f6 / #a855f7 | **2** deutan |
| campaigns TYPE_COLORS: bench vs analysis | #10b981 / #06b6d4 | 8 tritan |
| embryos.js STAGE_COLORS: bean vs pretzel | #60a5fa / #c084fc | 2 deutan |
| timepoint-player: bean vs comma, 1.5fold vs 2fold, 2fold vs 3fold | | 4, 8, 4 |
| occupancy3d: cuboid vs sheet | #14b8c4 / #39d0ff | 11 tritan |
| roles.py: calibration vs lineaging | #00cccc / #33cc88 | 6 tritan |
| `.eval-dot.stage-2fold` vs `.stage-3fold` | #a78bfa / #c084fc | adjacent purples |
| `.event-type-badge.cv` vs `.perception` | #a371f7 / #c084fc | two purples |

Blue and purple are **the** primary accent pair in main.css (`--gradient-primary`
is blue→purple). Every place that uses purple to mean "AI/perception/complete"
next to blue "session/in-progress/selected" is invisible to deutan viewers.

### Greyscale (achromatopsia / printing / low-end monitors)

Almost every sibling set has greyscale ΔE under 10: the palette was chosen for
hue variety at roughly equal lightness. Only `--op-live`/`--op-bad` (24) and
`--map-warm`/`--map-accent` (4 → bad) show real lightness separation. If the
plan includes a lightness ramp, this is where it pays off.

## 2. Structural findings (color is the only cue)

Full selector list is in the agent report below; the categories that matter:

**Bare dots (6–8 px), no glyph, no label**
`.status-dot.*` (header rig connection; `.partial` is *identical* to
`.connected` in main.css:369 and then overridden to orange in rig-menu.css:14,
so its color depends on CSS load order), `.agent-rail-dot.ok`,
`.tab-status-dot.live/stale/paused/error`, `.devices-status-led.*::before`,
`.ls-laser-dot--on`, `.board-status-dot.*`, `.reasoning-status-dot.*`,
`.nav-status-dot.*`, `.strip-indicator.*`, `.ops-pip.before/during/after`,
`.context-dot.*`, shell `.cx-q/.cx-w/.cx-e`, campaigns `.dot-*` and
`.matrix-dot` (tooltip only), `.eval-dot.medium` vs `.high` (amber vs orange).

**Left-edge stripes (2–4 px)**
`.embryo-card.running/paused/complete/error`, `.embryo-rail-item.*`,
`.embryo-card.minimal.*`, `.op-tcard.st-*`, `.op-runrow.is-*`,
`.decide-card.*`, `.verification-card.*`, `.session-item.active-session`,
review `.message.user/assistant/system`, `.trace-step.*`,
`.detection-card.positive-highlight`.

**Text-color only**
`.conf-high/med/low`, `.rate-*`, `.strategy-row.passed/failed`,
`.ac-level-error/warning/success`, `.op-btn.is-on` vs `.is-emitting`,
`.op-btn.is-armed`, `.op-gauge.is-near-floor`, `.region-say[data-bad]`,
`.op-plan-say[data-bad]`, `.temp-*.is-drifting` (blue→amber), `.tab.active`,
`.ac-choice-picked`.

**Filled regions with no label**
timepoint-player stage bar (`timepoint-player.js:683`, 8 colors at 0.4 opacity,
`pointer-events:none` so even the tooltip never shows), campaigns timeline
bars (`campaigns.js:1837`, type by fill only), 3D minimap dots
(`occupancy3d.js:437-444`), `.summary-card.*` bar fills, `.expov-svg-dose-fill-ok/warn/crit`,
`.ambient-pulse.warning/critical`.

**Animation as the only differentiator** (disappears under
`prefers-reduced-motion`, which main.css:2768 honours)
`body[data-run=running]` vs `paused` on `.v2-nav-item::after`,
`.status-dot.booting`, `.devices-camera-led.live`, `.cal-spim-led`.

## 3. Inconsistent semantics (hurts everyone, hurts colorblind users more)

- "Running" is **green** in Operate/Devices/Embryos (`--op-live`,
  `.tab-status-dot.live`) but **red** in the shell strip (`--run-live: #e5484d`,
  shell.css:223, the "REC light"). Same state, opposite hue.
- "Complete" is **purple** on embryo cards and campaign nav, **green** on
  `.timelapse-status.completed` (identical to running minus the glow) and on
  `--ops-done`.
- "Error" is `#f85149` in main.css, `#ef4444` in operate/rig-menu/toasts,
  `#f87171` for `--color-danger`, **orange** in `.chat-message-content.error`
  and `.boot-banner.failed`.
- Confidence low is red in `.confidence-badge.low` but **purple** in
  `.detection-row-compact .confidence.low`.
- Same embryo stage gets different colors in `embryos.js` vs
  `timepoint-player.js` (bean blue vs violet; hatched pink vs red), and the two
  files spell stage keys differently (`1_5_fold` vs `1.5fold`), so one of them
  falls back to grey.
- Assistant messages are **cyan** in the main chat and **green** in review.
- Event-type colors: `.timeline-event.*` (main.css:1839) and
  `.event-type-badge.*` (main.css:2090) assign different hues to the same kinds.

## 4. Bugs found along the way (fix regardless)

- **Undefined CSS tokens**, so the rule silently does nothing or always uses the
  fallback:
  - `--accent-red` (18 uses, 0 definitions). Every `.failed` state in the
    verification UI (main.css 7994, 8087, 8122, 8149, 8181, 8245) renders with
    **no red at all**.
  - `--accent-amber` (5), `--error` (1, main.css:7627), `--success` (1, :7646).
  - `--text-secondary` (10), `--accent-soft` (3), `--op-line` (1): always fall
    back to light-theme hex values, so dark theme gets slate-on-slate text
    (~2.4:1) in shell/notebook/reveal and a near-white `.nb-ask-result` panel.
- `main.css:6494-6499` defines `--color-success/ai/warning/info` that nothing
  uses, and `--color-danger` used only by agent-chat.css. Dead or nearly dead.
- `gently/app/theme.py`: 96 hex values across 8 themes; the `/theme` command in
  `harness/bridge.py` only returns names. **No color field is read anywhere.**
  The whole file is dead with the Ink TUI gone (PR #238).
- `experiment-overview.js:1180-1208` prints `ui_icon` raw ("diamond"), not the
  glyph, so the roster chip loses the shape cue that `strategy_snapshot.py`
  provides.

## 5. Light theme is worse than dark

No light override exists for: the ops palette (`experiment.css:643`),
`--map-zone-*`, `--color-danger`, `--run-live/--run-paused`, and every
hardcoded `#f85149`/`#ef4444`. WCAG contrast on white:

| Token | Light value | Contrast on #fff |
|---|---|---|
| `--accent-green` | #22c55e | 2.3 |
| `--accent-cyan` | #06b6d4 | 2.4 |
| `--accent-orange` | #f97316 | 2.8 |
| `--color-danger` | #f87171 | 2.8 |
| `#f85149` (all `.error`) | | 3.3 |
| `--ops-done/active/plan` (no override) | | 1.9–2.5 |
| `--op-live` | #16a34a | 3.3 |
| `--temp-setpoint-color` (no override) | #f59e0b | 2.1 |
| map zone green/orange on cream | | 2.6 / 2.2 |

All fail WCAG AA for text (4.5). White-on-fill badges also fail:
`.tab-badge.has-new` 1.7, `.vlm-stage-label` 1.8, `.transitional-badge` 2.3.

## 6. Patterns already in the codebase to copy

- `.op-lock.is-below`: thicker border + pulse + text (operate.css:189-214)
- `.lp-emit-unknown`, `.cp-thumb.is-key`, `.settings-badge.is-set`,
  `.mk-stage.is-fixed`, `.devices-embryo-selected`: dashed
- `.ops-node.planned`: hollow dashed dot vs solid (experiment.css:796-813)
- `.mk-stage[aria-pressed=false]`: strikethrough
- `.filmstrip-cell.pending::after`: extra dot
- Map "beyond" zone: hatch pattern (main.css:9554, legend 10111)
- `.confidence-bar`: bar count encodes level
- `.ac-turn-user`: alignment + bubble shape
- campaigns `STATUS_DOTS` glyphs ○ ◑ ● ⊘ ⊗ (used on graph/board, **not** matrix)
- `roles.py` `ui_icon` + `strategy_snapshot.py` glyph map ★ ◆ ● ▲

## 7. What was done (2026-10-05, branch feature/colorblind-safe-palette)

- **One status palette** in main.css `:root` / `[data-theme="light"]`:
  `--accent-green` live, `--accent-amber` paused/warn, `--accent` done,
  `--accent-red` error, `--text-muted` idle. Dark `#3ee6b0 #fbbf24 #60a5fa
  #f0405a #8b949e`; light `#138669 #a16207 #2563eb #9f1239 #5a5a7c`. Every
  pair clears ΔE ≥ 20 under protan, deutan and tritan; every accent clears
  4.5:1 on the card background in its theme. Light muted text became a
  violet-grey because slate collapses into the green under deuteranopia; this
  is the physical ceiling at AA contrast on white, not a style choice.
- `--accent-red`, `--accent-amber`, `--text-secondary`, `--accent-soft`,
  `--error`, `--success` are now defined. Per-page palettes (`--op-*`,
  `--ops-*`, `--run-*`, atrium `--warn`) alias the shared tokens. Every
  hardcoded status red/amber/green in the stylesheets is a token.
- **One vocabulary**: complete is blue everywhere (was purple/green), running
  is green everywhere (shell strip was red), every error is `--accent-red`
  (was four reds and two oranges), confidence low is red in both views.
- **Shapes** (`STATUS SHAPES` section at the end of main.css and short blocks
  in campaigns/shell/experiment/agent-chat/operate/rig-menu): live = filled
  circle, paused = half-filled, done = ring, error = diamond, idle = hollow
  dashed. Text-only states carry ✓ ✕ ! ● glyph prefixes. Left-edge stripes:
  error double, paused dashed. SVG phase/dose fills have per-state dash
  patterns. Badges with white text on bright fills use dark text in the dark
  theme.
- **JS**: one ordinal viridis-like stage ramp in `stage-colors.js` used by
  embryos.js, timepoint-player.js (segments now labelled and alternately
  striped); campaign types on Okabe-Ito with type icons on the bars; matrix
  dots carry the status glyph; 3D minimap stage is a blue ring, embryos amber
  dots; roster chips render the role glyph instead of the word "diamond".
- **Python**: `roles.py` role colours on Okabe-Ito; `plots.py` Okabe-Ito with
  marker and linestyle redundancy; `app/theme.py` and the `/theme` command
  deleted (dead). `tools/cvd_check.py` + `tests/test_colorblind_tokens.py`
  guard the thresholds in CI; `tests/js/stage-colors.test.mjs` guards the
  ramp.
- Verified: ruff, both mypy runs (pre-commit), 2242 pytest, all JS tests,
  and a headless run of the app in both themes. Not verified on the rig.

## 8. Original recommended plan (kept for reference)

1. **Operate page safety states.** Give `.op-btn.is-emitting` a non-color cue
   (filled background + ⚡/● glyph via `::before`, or inverted) so emitting is
   never confused with on. Add shape to `.op-gauge.is-near-floor` and
   `.op-btn.is-armed`. Keep `--op-bad` as the only pulsing state.
2. **One status vocabulary.** Define a single set of semantic tokens
   (`--st-live`, `--st-paused`, `--st-done`, `--st-warn`, `--st-bad`,
   `--st-idle`) with dark and light values chosen so that each pair clears
   ΔE ≥ 25 under all three simulations and lightness differs. A workable set:
   live = green #4ade80/#15803d, done = blue #60a5fa/#1d4ed8, paused/warn =
   amber #fbbf24/#b45309, bad = red-magenta #fb7185/#be123c, idle = grey.
   Drop purple as a status meaning; keep it for "AI-authored" only. Replace
   every hardcoded `#f85149`, `#ef4444`, `#f87171`, `--run-live`, `--ops-*`
   with these. Resolve the green/red "running" conflict in shell.css.
3. **Dots and stripes get glyphs.** One shared `::before` glyph rule keyed on
   state class (● live, ◐ paused, ✓ done, ✕ error, ○ idle, ! warn) applied to
   every dot/stripe selector in §2. Left-edge stripes become stripe + glyph
   in the title row. Reduced-motion must still have a static difference.
4. **Filled bars get labels or patterns.** timepoint-player stage bar: label
   each segment (or alternate hatch); campaigns timeline: type glyph on the
   bar; 3D minimap: stage as ring/crosshair, embryos as dots; dose fill:
   stripe pattern on warn/crit.
5. **Categorical sets.** Embryo stages are ordinal: use a single-hue lightness
   ramp (Viridis-like) in **one** shared map used by embryos.js,
   timepoint-player.js and `.eval-dot.stage-*`. Campaign types and event
   kinds: cap at 4–5 hues from an Okabe-Ito/Paul Tol set, always with an icon.
   Role palette in `roles.py`: swap test/calibration to Okabe-Ito vermilion
   and sky blue; the glyphs already exist, render them in experiment-overview.
6. **Fix the bugs in §4** (define or remove the undefined tokens, delete
   `theme.py`, render the role glyph). These are a day's work and some are
   visible to every user.
7. **Light theme.** Darken the light accent set to ≥ 4.5:1 and add the missing
   overrides (ops, map zones, danger, run-live, temp setpoint).
8. **Guardrail.** Keep the simulator as `tools/cvd_check.py` and add one pytest
   that parses the semantic tokens from CSS and asserts pairwise ΔE ≥ 25 under
   protan/deutan/tritan and contrast ≥ 4.5 on both card backgrounds, so
   regressions fail CI. CI drives no browser, so this is the only automated
   check that is cheap to add.

## Appendix: raw agent reports

The full selector-by-selector CSS report and the JS/Python color-map report
were produced by two read-only sweeps; they are condensed above. Simulated
hex values for every token are printed by `tools/cvd_check.py`.
