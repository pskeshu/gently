# The Atrium — open decisions

From an adversarial critique (five lenses: operator-at-2am, visual design, spec
compliance, accessibility, failure-mode tracing). 37 raw findings, ranked and
deduplicated to 18; the ten mechanical ones are fixed and committed. These eight
are **not** mine to decide — each changes what the operator sees, or how the
surface behaves under pressure on live hardware.

Ranked by harm x confidence, as the critique ranked them.

### 11. A dead telemetry socket is indistinguishable from a quiet night
**Converged: 3 lenses. Highest harm on the list.** Nothing on the bench surfaces `gentlyConnected`. The header's connection pill is static-flow with no z-index and `#atr-viewport` is `position:fixed; inset:0; z-index:40` (atrium.css:39-40), so it is painted over — there is no header at all in `atrium-packed.png`. EVENTS' gauge is `plural(state.allEvents?.length,'event')` (atrium.js:94): last successful count, frozen, worded identically to a healthy quiet run. Every other panel keeps rendering its last frame with no visual difference from live. And per item 5, EVENTS is arithmetically capped at a pulse on a folded 32px strip. The operator reads frozen history as present tense and makes a hardware decision from it.
**Decision:** what does the surface say when it stops being told the truth? Candidates: a `why` string per `pressWhen` config so the gauge reads `⚠ feed down` instead of a count; a `gentlyConnected` readout in the chrome (the only screen-fixed real estate); raising EVENTS' crit so the rung is reachable. Probably all three.
**Files:** atrium.js (CONFIG gauges + `contentGauge`), atrium.css · **Size:** small-to-medium

### 12. The density cap is not enforced, and salience ignores urgency
**Converged: 3 lenses.** `const salience = f => (+f.dataset.crit || 0);` (atrium.js:282) — SPEC R5 says `crit + urgency` and explicitly calls the collapse "a bug that was actually shipped". `setDensity` runs only on slider input; `attend()` unfolds without folding anything (atrium.js:216) and the `open` release does the same (:479-483). In `atrium-packed.png` the dial reads **6** and seven windows are open. Over a night every travel and every escalation ratchets one more window open until the bench is a wall again — and the cap is the spec's entire safety argument for letting an agent drive the surface. Conversely, nudging the dial re-folds whatever is currently calling, because urgency is not in the ranking.
**Decision:** the ranking rule and what the cap means. Does the attended window count against N? Can a calling window ever be folded by the dial? Does re-ranking on every 2s tick mean windows fold and unfold under the operator's hands — which is item 3's complaint arriving through a different door?
**Files:** atrium.js · **Size:** small (the code is ~4 lines; the policy is not)

### 13. The ladder's memory is wrong in both directions
**Converged: 2 lenses.** `if (!pressing) { f._since = 0; f._rung = 0; ... }` (atrium.js:465-467) wipes the ladder on every resolve, so a flapping scope link — the normal failure mode, not the exotic one — re-climbs from rung 0 every time, re-opening the window and re-pulsing every ~30s. In the other direction, `if (rung <= f._rung) return` (atrium.js:471) means anything that refolds the window while the fact is *still outstanding* (the density dial, the ▼ button, `restore()` on reload) silences it permanently with no dialog and no log line.
**Decision:** who clears the ladder. The obvious answer — resolution clears the clock, the operator *seeing it* clears the rung — is a policy about escalation on live hardware, not a refactor.
**Files:** atrium.js · **Size:** small

### 14. The alarm's screen-fixed readout never clears, and has no clock on it
**Converged: 2 lenses.** `paintReleases` (atrium.js:489-495) renders `RELEASES[0]` and nothing ever removes it; the timestamp exists only in the `title` tooltip of a ~90px span. Verified: after the fault resolved, the corner still read amber "open · devices" indefinitely. The operator comes back from the cold room, sees it, and chases a fault that self-healed forty minutes ago — or learns to ignore the label, which is the same as not having it.
**Decision:** whether resolution is itself a log entry ("23:41 clear · devices") or the label just expires, and how much of the log is worth screen space.
**Files:** atrium.js · **Size:** small

### 15. Escalating evicts the scope status from the courtyard, permanently
**Converged: 2 lenses, verified: `document.querySelectorAll('.atr-slot .atr-win').length === 0` after one escalation.** DEVICES is the only pinned window (`pin:'rail'`, atrium.js:47), so "scope offline" is normally the one gauge guaranteed to be in front of the operator. Both `attend()` (:210) and the `open` release (:481) call `unpin`, and `pin()` is called from nowhere but `enable()` (:693) — `save`/`restore` don't record pins either. The rail in `atrium-packed.png` is already empty. The worse things get, the less visible the scope's state becomes, and it degrades for the rest of the session.
**Decision:** does a window return to its edge when resolved (remember `homeSlot`, re-pin in the not-pressing branch), or does the operator get a pin toggle in the window head, or both? Also: `frameOn` filters out `dataset.slot` (atrium.js:179), so the "all" chip frames nine of ten windows.
**Files:** atrium.js · **Size:** small

### 16. The bench is unreadable at the zoom the operator actually gets
**Converged: mostly one lens, but three independent measurements.** The courtyard is screen-fixed; the bench is CSS-transform scaled. At `CONFIG.home.scale = 0.58` (atrium.js:74) — first load — the window title (atrium.css:93) is 7.7px×0.58 ≈ 6.7px, the gauge (atrium.css:100) ≈ 6.3px, and the R8 child strip (atrium.css:169) ≈ 6.0px in `--text-muted` grey. The chip bar, which merely duplicates the old tab strip, renders at 10.9px unscaled. So the mechanism the spec says "dissolves the nested switcher" — the answer to the calibration tab nobody found — is the smallest, lowest-contrast text on the screen. Two related defects in the same cluster: `#atr-chips { max-width: 46vw }` (atrium.css:143) puts the chrome card over 361px of the EMBROYS header in `atrium-packed.png`, decapitating the highest-crit window on every load, while column 1 leaves a 160px void above HOME (`y:200`, atrium.js:38) where the chrome is *not*; and five adopted panels restate their own name in a bolder, larger `<h2>` directly under the frame's 7.7px title (index.html:212, :328, :414, :439, :465 plus `_plans_workspace.html:4`), one of them a serif-italic logotype that appears nowhere else.
**Decision:** this is the visual hierarchy of the whole shell. Counter-scaling the chrome against `scale` is ~6 lines; deciding that the frame's name outranks the panel's own heading is a design call. Related: `.atr-peek` is unconditionally `--accent` (atrium.css:101), so "scope offline" is typographically identical to "5 rows" and both read as clickable links, at 3.36:1 on light.
**Files:** atrium.css, atrium.js (`apply()`), index.html · **Size:** medium

### 17. `contentGauge` fails open — a broken state layer produces a confident fabricated number
`try { txt = declared(); } catch (_) { txt = ''; }` then falls through to `dominantCount` (atrium.js:337-343). If `state`/`SharedState` breaks, DEVICES stops saying "scope offline" and starts saying "7 rows", with only a swallowed exception. Same tick, `catch (_) { pressing = false }` (atrium.js:464) reads a throwing `pressWhen` as all-clear. And `plural(undefined,'event')` returns "no events", so a failed fetch and a genuinely empty log are the same string.
**Decision:** R4 makes the gauge the thing you glance at *instead* of opening the window. A gauge that says "?" costs a click; a gauge that says "no events" costs the experiment. But "?" everywhere during startup is its own noise — someone has to decide where the line sits.
**Files:** atrium.js · **Size:** small

### 18. Three smaller judgement calls
- **Ghost windows are fully live.** Nothing sets `inert` or `pointer-events:none`, so every control in the nine windows at 18% opacity (atrium.css:80) stays hit-testable and in the tab order — stage jog, laser enable, start acquisition. `inert` is one native attribute in `attend()`'s forEach, but it also kills the fold button and drag on dimmed windows, and makes their content unreadable to a screen reader. Behaviour change: human. **atrium.js:207-218 · small**
- **`offer` is implemented as `seize`.** `if (ch === 'offer' || ch === 'seize') attend(cfg.tab);` (atrium.js:485). The R6 table prices rung 3 at "a decision" and rung 4 at "the current task", and the spec flags rung 4 as an open safety question. Unreachable only because of `maxChannel: 'open'` — whose own comment says the cap is "conservative until an operator has seen it escalate", i.e. it is meant to be raised. Either make `offer` a clickable `#atr-releases` button, or delete the rung. **atrium.js:485, :117 · small**
- **The v2 landing covers the whole atrium on a fresh boot.** `.v2-landing` is `z-index:200` (landing.css:49) over `#atr-viewport`'s 40; `pages.py:47` sets `show_landing` when the session has no work, so `?atrium=1` renders "What are we doing today?" over a fully built, fully invisible, ticking atrium. Low stakes mid-experiment, expensive as a false bug report. The fix is one line, but *which* line — suppress the landing under `.atrium-on`, or lift the viewport above it — is a product choice. **atrium.css or landing.css · one-line**

Also parked here because both change what the operator can read: the only container query (atrium.css:155-158) targets `.atr-body .row, .atr-body .form-row` — neither class exists in any template — and fires at 320px, while the narrowest bench window is 460px authored (306px on screen at 67%), which is why EVENTS' SOURCE column is sliced mid-token with no ellipsis; and `.op-cam-fit`'s `var(--img-bg)` (operate.css:236-250) makes a ~315×317px near-black rectangle the highest-contrast object on a light bench, saying "CAMERA OFF", in the window the ladder opened rather than one the operator asked for.

---

## Fix this one first

**Item 1 — the one-line CSS exemption for `.atr-calling`.**

Not because it is the biggest harm (item 11 is), but because it is the cheapest, is verified by two independent lenses, and every other pressure fix on this list is worthless until it lands. Items 11–15 all end in "the window escalates and says so" — and today that escalation is drawn at 18% opacity, desaturated to grey, for the entire time the operator is working in another window, which is the only time escalation exists for. The exemption is already written one line above for pinned windows (atrium.css:85). Fixing the ladder's memory, its cap, or its arithmetic while the payoff is invisible is tuning an alarm whose speaker is unplugged.
---

## Already fixed (for context)

The mechanical half, all committed and verified live:

1. **The alarm was invisible.** `body.atr-focused .atr-win` (0,2,1) beat
   `.atr-win.atr-calling` (0,2,0), so every escalation was drawn at 18% opacity
   and desaturated to grey. The critique's own advice was to fix this before
   anything else on the pressure list, and it was right.
2. A click during a 420ms glide hit a moving target — and DEVICES turns a canvas
   click into a stage move.
3. `packColumns` teleported windows out from under the cursor every 2s.
4. The gauge counted elements the stylesheet hides (PLANS read "6 rows" over an
   empty body).
5. Ladder rungs unreachable by arithmetic — now warns at startup instead of
   failing silently.
6. The release log was uncapped and died on refresh.
7. Keyboard: 17 child destinations were mouse-only; travel keys died after a chip
   click; 39 covered controls stayed in the tab order.
8. `prefers-reduced-motion` deleted the alarm and kept the swoop.
9. `--warn` was consumed twice and declared nowhere.
10. Four panels were never initialised — why PLANS sat on "Loading campaigns…".
