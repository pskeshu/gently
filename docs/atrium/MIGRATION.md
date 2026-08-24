<!-- Generated 2026-08-24 by a 16-agent audit of gently/ui/web/static/js/.
     Nine readers (one per file group) proposed window boundaries; an adversarial
     pass tried to REFUTE the riskiest splits; a synthesis pass produced this.
     Spot-checked by hand before committing: status-store.js exists and has the
     documented semantics; devices.js really does define renderEmbryos twice
     (:569 shadowed by :963); renderDetectionListWithCollapse and
     renderEmbryoCardFull each occur exactly once, i.e. definition only.
     Everything else is the agents' claim — each cites file:line, so verify
     before acting rather than trusting the table. -->

# Atrium Migration Map — Gently Web UI

<!-- STATUS BOARD — updated as phases land -->

| Phase | State | Evidence |
|---|---|---|
| 1 · delete dead code | ✅ done | `fe6ec80` · 1,784 lines · 0 story deltas |
| 2 · live bugs | ✅ done | `1c7165b` · 10 fixed, **5 refuted** · 0 story deltas |
| 3 · transform safety | ✅ done | 27 fixed, **23 refuted**, 6 deferred |
| 8 · port the shell | ✅ done | `atrium.js` + `atrium.css`, flag-gated, all 10 panels adopted |
| 4 · the store | ⬜ | extend `status-store.js`, `stageXY` first |
| 5 · pilot windows | ⬜ | `notebook.js`, `device-layer.js` |
| 6 · pure projections | ⬜ | `campaigns.js` |
| 7 · reconcile with subwindows | ⬜ | see below |

**Phase 3's refutation rate was 46%** — higher than phase 2's 33%. The pattern in
what got refuted is worth internalising, because it is the same mistake four
times:

- **viewBox units are not screen pixels.** `font-size="9"` inside
  `viewBox="0 0 800 140"` already tracks container width and needs no fix. This
  accounted for most false hazards in `embryos.js`, `campaigns.js` and
  `temperature-graph.js`.
- **`clientWidth`/`offsetWidth` are transform-blind**, so code using them is
  already in the prescribed form. Only `getBoundingClientRect` mixes spaces.
- **Unconditional `preventDefault` on a frame's inner wheel is what SPEC R3
  requires**, not a bug — the frame owns its own zoom. I had this backwards in
  the brief and the agents corrected it against the spec.
- **A permanent rAF loop makes a transform-redraw hook unnecessary**;
  `occupancy3d.js` and `projection-viewer.js` both already re-render every frame
  and already listen for `gently:layout-changed`.

Two groups report their fixes are *not bit-identical* at scale 1.0
(`marking.js`, `operate.js`): where a flex layout yields a fractional width the
old code divided by the truncated backing store and the new code divides by the
measured rect — a sub-pixel shift, in the direction of the true displayed size.

### Phase 8 — the shell, as built

`gently/ui/web/static/js/atrium.js` + `static/css/atrium.css`, loaded from
`index.html`, **off by default**. `?atrium=1` on, `?atrium=0` off, choice sticks
in localStorage. With the flag off it registers nothing: no viewport, no body
class, `switchTab` unwrapped.

It **adopts** the ten existing `.tab-content` divs rather than rewriting them —
the div is *moved* into a window frame, never cloned. That is what made the port
cheap, and it works because all ten were already in the DOM simultaneously; tabs
only hid them with a class. `switchTab(name)` is wrapped to mean `attend(name)`,
still sets `.active` and still emits `TAB_CHANGED`, so the ux_v2 rail and every
other listener keep working. The tab shell's lazy-init hooks (`HomeApp.init`,
`renderCalibrationGallery`, `CampaignsApp.init`, `renderEventsTable`) now fire on
first travel instead of first show.

Verified live: all ten adopted with content intact, travel to each produces zero
console errors, DEVICES pins to the rail, density folds the rest to gauges, and
`gently:layout-changed` fires on unfold so the 3D viewers re-measure.

**Verification standard for every phase.** CI runs no JavaScript, and the
committed `tools/ui_crawler/baseline/status.json` was recorded in a *different*
environment (it expects an account server), so diffing against it produces
false regressions. The only sound check is same-environment before/after:

```bash
GENTLY_STORAGE_PATH=/tmp/gently-dev uv run python launch_gently.py --no-api --no-auth --no-browser
uv run python tools/ui_crawler/run_stories.py --url http://localhost:8080
```

38 stories. Capture `tools/ui_crawler/out/stories/status.json` before and after
and diff the two runs, not the run against the baseline. Phases 1 and 2 each
produced **zero** story deltas by that measure.

**The audit's own error rate is about one in three.** Phase 1 overclaimed
`viewer.js` by 27x; Phase 2 refuted 5 of 15 claims, including the one ranked
most severe. Treat every table below as a hypothesis with a file:line to check.

---

## Phase 7 — reconcile with subwindows

This map was written BEFORE the Atrium gained addressable children (SPEC.md R8),
so it was forced to answer window-or-not for every candidate. With children in
the model that question softens, and several "merge away" verdicts are probably
"child of" instead.

| Merged into SUBJECT | Likely actually |
|---|---|
| `FILMSTRIP` | a child of SUBJECT — a real view with its own gauge |
| `EVAL_TIMELINE` | a child; called "the folded rendering of TIMEPOINT_DETAIL", which is what a gauge *is* |
| `TIMEPOINT_DETAIL` | a child |
| `PERCEPTION_CHAT` | a child |
| `CAL_EMBRYO_ROSTER` | still a merge — genuinely the fourth copy of one roster |

The prototype already carries this shape: EMBRYOS holds `board`/`filmstrip`/
`vitals` as children, which are exactly the three views `embryos.js` hand-rolls
today. A child is a full destination (`attend('p-embryos:vitals')`), carries its
own crit/tolerance, raises its parent when it presses, and detaches onto the
bench on demand. So a decomposition need not choose between "one window" and
"three windows" — it can defer, which is the cheaper answer.

**Rule for the port:** if a candidate has its own gauge but no independent
reason to occupy bench space, make it a child. Promote later if it earns it.

---


**Headline:** the seven group reports propose ~100 windows. After the refutations and cross-file dedup, roughly **55 survive**, and about **12 of the ~100 are dead code that must be deleted, not migrated**. The single largest source of inflation is the same capability implemented 2–4 times in different files (stage position ×4, embryo roster ×4, SPIM live ×3, imaging spec ×3, pending ask ×3, embryo marking ×2, temperature ×2). The Atrium's value here is mostly **subtraction**: it deletes seven mode switchers, four view-switchers, one modal system, and the tab shell.

**No framework.** `gently/ui/web/static/js/status-store.js` is already a sticky subscribe-and-replay store written to fix exactly this bug class ("three disagreeing indicators", its own docstring at :1-18). Extend it to four more keys. See §3.

---

## 1. Inventory

Legend for **Fate**: `keep` = becomes a window · `merge→X` = folds into X · `drop` = dead code, delete · `defer` = real capability, no live surface yet.

### Embryos tab — `embryos.js` (4556 lines)

| Window | crit | freq | tol | Pin | Risk | Fate |
|---|---|---|---|---|---|---|
| SESSION_SYNC | 0.9 | .008 | 10 | — | entangled | **merge→ConnectionStatus** (refuted: symbol is single-file, `embryos.js:9,24`; five other session-id derivations exist) |
| AMBIENT_WATCHDOG | 0.95 | 0.1 | 30 | ✔ | moderate | keep |
| ACQUISITION_CADENCE | 0.9 | 1 | 2 | ✔ | moderate | keep (absorbs the dead `experiment-strip.js` RUN_STATUS + NEXT_ACQUISITION) |
| EMBRYO_ROSTER | 0.7 | .008 | 120 | — | entangled | **merge→SUBJECT** (refuted) |
| STAGE_BOARD | 0.8 | .008 | 300 | — | moderate | keep |
| FILMSTRIP | 0.5 | .008 | 120 | — | entangled | **merge→SUBJECT** as a density mode (refuted; it is `renderVitalsView` with s/cell/point/) |
| VITALS_CHART | 0.6 | .008 | 300 | — | moderate | keep |
| EVAL_TIMELINE | 0.6 | .008 | 300 | — | entangled | **merge→SUBJECT** (refuted: it is the folded rendering of TIMEPOINT_DETAIL) |
| TIMEPOINT_DETAIL | 0.7 | .008 | 0 | — | entangled | **merge→SUBJECT** (refuted: selection projection, not capability) |
| REASONING_TRACE | 0.5 | .008 | 0 | — | moderate | **merge→SUBJECT** (inline branch of the detail; no independent gauge) |
| PERCEPTION_CHAT | 0.3 | 0 | 0 | — | entangled | **merge→SUBJECT** (refuted) |
| VERIFICATION_CONSENSUS | 0.85 | .02 | 30 | — | entangled | **defer** — four WS handlers (`embryos.js:1470-1573`) maintain state nothing paints; only renderer is called from `renderEmbryoCardFull` (:1738), which has **zero callers** (verified) |
| **SUBJECT** (new, merged) | 0.75 | .008 | 0 | — | entangled | keep — folded: `E3 · comma · T0342 · 87%`; open: roster + dot strip + detail + chat |

Also dead in this file, **delete ~900 lines**: `renderDetectionListWithCollapse` (:2177) has zero callers (verified) and its whole subtree is unreachable — `groupDetectionsForCollapse`, `renderRangeItems`, `toggleRangeItem`, `renderInlineExpansion`, `renderConfidenceDots`, `calculateInterestScore`, `toggleRange`, `loadMoreInRange`, `setDetectionFilter`, `showDetectionDetail`, `renderDetectionCard`, `renderAgreeDisagreeButtons`, `renderVerificationCard`, `markAgreement`, `renderDetectionContext`, `toggleReasoning`, `toggleImage`; plus `renderEmbryoCardFull` (:1738), `showFirstRunHint`/`dismissHint`/`showFirstRunHints` (:3503-3620), `view3D` (:3867), `showToast` (:3908), `compareDetection` (:3622).

### Devices tab — `devices.js` (2554 lines)

| Window | crit | freq | tol | Pin | Risk | Fate |
|---|---|---|---|---|---|---|
| DEVICE STREAM HEALTH | 0.95 | 5 | 4 | ✔ | clean | keep |
| STAGE POSITION | 1.0 | 5 | 0.5 | ✔ | moderate | keep — **canonical**; absorbs `operate.js:39 _xy`, `occupancy3d.js:57 _stage`, `marking.js:39-40` |
| TRAVEL FENCE | 0.85 | .067 | 0 | — | moderate | **merge→STAGE POSITION** as the proximity term; it has no output of its own |
| STAGE MAP | 0.6 | 0.2 | 0 | — | entangled | keep (after the cycle break, §5 phase 6) |
| EMBRYO WAYPOINTS | 0.75 | 0.1 | 0 | — | entangled | keep — same window family as `marking.js` / `operate.js` marking; see cross-file merge below |
| DEVICE PROPERTY TABLE | 0.2 | .067 | 30 | — | clean | keep |
| BOTTOM CAMERA | 0.5 | 4 | 1.5 | — | moderate | **merge** with `operate.js` BOTTOM_CAM_LIVE |
| SPIM LIVE VIEW | 0.8 | 15 | 1.5 | — | moderate | **merge** with `operate.js` SPIM_LIVE and `gallery.js` SPIM_LIVE |
| SPIM BEAM PARAMS | 0.7 | 0 | 0 | — | moderate | keep — absorbs `operate.js` SHEET_ALIGNMENT |
| LASER GATE | 1.0 | 0 | 0 | ✔ | clean | keep — absorbs `operate.js` EMITTERS |
| ILLUMINATION | 0.4 | .067 | 20 | — | moderate | keep (merge the two room-light mutators during extraction) |
| ACQUIRE | 0.6 | 0 | 0 | — | moderate | **merge→ACQUISITION_LAUNCHER** (`operate.js`) |
| TIMELAPSE BUILDER | 0.9 | 0 | 0 | — | entangled | **merge→ACQUISITION_LAUNCHER**; it carries a second copy of beam geometry and a second laser select |
| SAMPLE TEMPERATURE | 0.7 | .067 | 20 | ✔ | entangled | **merge** with `temperature-graph.js` WATER_TEMP_TRACE |

Dead: `renderEmbryos` at `devices.js:569-634` — two functions of that name (verified, :569 and :963); the second wins by hoisting, the first reads a schema (`emb.x/.y/.role`) nothing produces. Delete with `_ROLE_COLOR` (:43-47).

### Plans tab — `campaigns.js` (1874 lines)

| Window | crit | freq | tol | Pin | Risk | Fate |
|---|---|---|---|---|---|---|
| CAMPAIGN_ROSTER | 0.2 | .01 | 300 | — | moderate | keep — absorbs `home.js` RECENT_PLANS and `landing.js` plan fetches (three fetchers, one `/api/campaigns`) |
| PLAN_DOCUMENT | 0.5 | .02 | 60 | — | moderate | keep |
| ITEM_INSPECTOR | 0.6 | .02 | 30 | — | entangled | keep |
| IMAGING_SPEC_EDITOR | 0.9 | 0 | 0 | — | entangled | keep — **canonical spec window**; absorbs `agent-chat.js:765-797` IMAGING_SPEC and `landing.js:449` |
| SESSION_LINKS | 0.3 | .01 | 600 | — | entangled | **merge** with `experiment-overview.js` LINKED_PLANS (its own comment at :257 admits the duplication) |
| PLAN_PROVENANCE | 0.8 | .005 | 0 | — | entangled | keep, **unpinned** (see §2) |
| DEPENDENCY_GRAPH | 0.4 | .02 | 120 | — | moderate | keep |
| STATUS_BOARD | 0.35 | .02 | 60 | — | clean | keep |
| DECISION_POINTS | 0.5 | .01 | 300 | — | clean | keep |
| COVERAGE_MATRIX | 0.2 | .01 | 600 | — | clean | keep |
| SCHEDULE_TIMELINE | 0.3 | .01 | 900 | — | moderate | keep |

Dead: `state.allItemsFlat` (:66, built 4×, read 0×), `item._rootCampaignId` (:194), `labelWidth` (:1802).

### Calibration / Gallery tab — `gallery.js` (1803 lines)

| Window | crit | freq | tol | Pin | Risk | Fate |
|---|---|---|---|---|---|---|
| CAL_METRICS | 0.9 | .05 | 0 | — | moderate | keep |
| SPIM_LIVE | 0.75 | 2 | 10 | — | entangled | **merge** into the one SPIM window; **delete `SpimPopout` (gallery.js:1403-1615, 213 lines)** — a draggable persistent floating window is what the Atrium already is |
| GALVO_PROFILE | 0.7 | 1 | 10 | — | entangled | keep |
| FOCUS_CURVES | 0.8 | 0.5 | 20 | — | clean | keep |
| CAL_FIT_SUMMARY | 0.85 | .02 | 0 | — | clean | **merge→CAL_METRICS** (same record, same tol 0) |
| CAL_SLICE_DETAIL | 0.3 | 0 | 0 | — | clean | **drop** — `#cal-profile-detail` exists in no template (only `gallery.js:950`, `main.css:1370`) |
| CAL_EMBRYO_ROSTER | 0.3 | 0.2 | 60 | — | moderate | **merge→SUBJECT** (fourth roster) |
| CAL_FILMSTRIP | 0.25 | 1 | 30 | — | moderate | keep (consolidate the two divergent render paths first) |
| VOLUME_SEGMENTATIONS_3D | 0.3 | .01 | 300 | — | clean | keep — 17 lines wrongly bolted into the calibration filmstrip |
| RECENT_STRIP | 0.2 | 2 | 15 | — | moderate | keep; its embryo filter is inert (`state.embryoFilter` never written) — delete the filter |
| SNAPSHOT_BROWSER | 0.2 | 0 | 3600 | — | clean | keep — cleanest lift in the repo |

Not a window: `showGentlyToast` (`gallery.js:1767-1803`), called from four other files. Becomes a courtyard notice line.

### Experiment tab — `agent-chat.js` (1410) + `experiment-overview.js` (1359)

| Window | crit | freq | tol | Pin | Risk | Fate |
|---|---|---|---|---|---|---|
| PENDING_ASK | 0.95 | .01 | 0 | ✔ | entangled | keep — **canonical**; absorbs `ask-stage.js` and `landing.js:659-699`; three renderers become one |
| CONTROL_LOCK | 0.9 | .01 | 0 | ✔ | entangled | keep |
| AGENT_RUN_STATE | 0.8 | 0.5 | 2 | ✔ | entangled | keep — absorbs `landing.js` ORACLE_STATE |
| IMAGING_SPEC | 0.85 | .01 | 0 | — | clean | **merge→IMAGING_SPEC_EDITOR** |
| TACTIC_SPINE | 0.8 | 0.1 | 30 | — | entangled | keep |
| POPULATION_ROSTER | 0.65 | 0.1 | 60 | — | moderate | keep |
| REACTIVE_RULES | 0.7 | 0 | 0 | — | clean | keep |
| AGENT_TRANSCRIPT | 0.5 | 10 | 10 | — | moderate | keep — absorbs `landing.js` AGENT_ACTIVITY (two renderers of one stream) |
| COMPOSER | 0.6 | 0 | 0 | — | moderate | keep |
| TOOL_ACTIVITY | 0.45 | 0.2 | 30 | — | entangled | keep |
| QUEUED_MESSAGES | 0.25 | .05 | 0 | — | moderate | keep |
| LINKED_PLANS | 0.15 | 0 | 0 | — | clean | **merge** with `campaigns.js` SESSION_LINKS |

Not windows: `mdToHtml` + helpers (`agent-chat.js:49-291`, 240 lines) → shared util. Panel chrome (`agent-chat.js:1143-1256`: `togglePanel`, `setupResize`, `--chat-w`, unseen badge) → **delete ~110 lines**; every transform-unsafe site in this group lives in it.

### Operate tab — `operate.js` (1359) + `timepoint-player.js` + `projection-viewer.js`

| Window | crit | freq | tol | Pin | Risk | Fate |
|---|---|---|---|---|---|---|
| XY_INTERLOCK | 1.0 | 1 | 0 | ✔ | entangled | keep |
| F_DRIVE_APPROACH | 1.0 | 1 | 0 | ✔ | entangled | keep (shares the pinned interlock gauge) |
| STAGE_POSITION | 0.9 | 1 | 0 | ✔ | moderate | **merge→devices.js STAGE POSITION** |
| EMITTERS | 0.9 | 0 | 0 | ✔ | entangled | **merge→LASER GATE** |
| RUN_MONITOR | 0.8 | 0 | 5 | ✔ | clean | keep — `stopRun` (`operate.js:1050`) is the only hardware halt in the entire JS tree |
| BOTTOM_CAM_LIVE | 0.7 | 8 | 1 | — | entangled | keep (canonical bottom cam) |
| SPIM_LIVE | 0.7 | 8 | 1 | — | moderate | keep (canonical SPIM) |
| BOTTOM_Z_FOCUS | 0.5 | 1 | 5 | — | moderate | keep |
| EMBRYO_MARKING | 0.6 | 8 | 0 | — | entangled | **merge** with `marking.js` (two full implementations) and `devices.js` EMBRYO WAYPOINTS |
| SAM_DETECT | 0.4 | 0 | 30 | — | entangled | **merge→EMBRYO_MARKING** (it writes `_markers` directly and owns the replace-not-append rule at :650) |
| EMBRYO_ROSTER | 0.5 | 0 | 0 | — | entangled | **merge→SUBJECT** |
| SHEET_ALIGNMENT | 0.5 | 0 | 0 | — | moderate | **merge→SPIM BEAM PARAMS** |
| ACQUISITION_LAUNCHER | 0.6 | 0 | 0 | — | entangled | keep — absorbs devices.js ACQUIRE + TIMELAPSE BUILDER |
| TIMELAPSE_PLAYBACK | 0.2 | 3 | 0 | — | moderate | keep |
| SEQUENCE_TIMELINE | 0.2 | 3 | 0 | — | moderate | keep |
| VLM_VERDICT | 0.3 | 0 | 0 | — | clean | keep |
| VOLUME_3D | 0.3 | 60 | 0 | — | moderate | keep — **needs a folded/open rAF gate that does not exist** |
| PROJECTION_GRID | 0.2 | 0 | 0 | — | clean | keep |

**Delete ~40% of `timepoint-player.js`**: `openVideoMode`, `injectVideoUI`, `close`, `showDetectionToast`, `bindVideoKeys` (:98-133, :242-308, :774-811, :881-909) — pure modal/lightbox choreography with no Atrium equivalent.

### Shell / Home / Landing — `app.js`, `landing.js`, `home.js`, `utils.js`

| Window | crit | freq | tol | Pin | Risk | Fate |
|---|---|---|---|---|---|---|
| LINK | 0.95 | .067 | 5 | ✔ | clean | keep — already correctly factored behind `ConnectionStatus`; the template for everything else |
| SESSION_IDENTITY | 0.5 | 0.2 | 0 | ✔ | entangled | keep (already in `_header.html:7-8`; needs no new code, only a store read instead of `_timelapseText.textContent` scraping at `app.js:44`) |
| PRESENCE | 0.1 | .05 | 30 | — | moderate | keep — the clearest "folded forever, never opened" window |
| ORACLE_STATE | 0.7 | 1 | 2 | ✔ | moderate | **merge→AGENT_RUN_STATE** |
| PLAN_TREE | 0.85 | .05 | 0 | — | entangled | **merge→CAMPAIGN_ROSTER + PLAN_DOCUMENT** |
| AGENT_ACTIVITY | 0.4 | 10 | 3 | — | entangled | **merge→AGENT_TRANSCRIPT** |
| PENDING_ASK (landing) | 0.9 | .02 | 0 | — | entangled | **merge→PENDING_ASK** |
| RECENT_SESSIONS | 0.3 | .02 | 60 | — | clean | keep |
| RECENT_IMAGES | 0.2 | .067 | 15 | — | clean | keep — it already hand-rolls `tol` as `IMAGES_TTL_MS = 15000` (`home.js:19`), which is good evidence the model fits |
| RECENT_PLANS | 0.3 | .02 | 60 | — | moderate | **merge→CAMPAIGN_ROSTER** |

Dead: `app.js:546 _microscopeConnected` (written, never read), `app.js:603 updateTopLevelDot` (zero callers). `home.js` as a *page* has no reason to exist — a grid of folded gauges is the Atrium.

### Events / System — `events.js`

| Window | crit | freq | tol | Pin | Risk | Fate |
|---|---|---|---|---|---|---|
| SYSTEM_LOG | 0.35 | 5 | 0 | — | entangled | keep |
| EVENT_TIMELINE | 0.2 | 0 | 5 | — | moderate | keep (needs an incremental append path it does not have) |
| EVENT_SUMMARY | 0.3 | 0 | 10 | — | moderate | keep, **not pinned** — its uptime scans a 500-entry ring for `SESSION_STARTED` (`events.js:594`) and silently reverts to `--`. A pinned window must not lie. |

### Sessions / Review / Notebook / Device layer

| Window | crit | freq | tol | Pin | Risk | Fate |
|---|---|---|---|---|---|---|
| DEVICE_LAYER | 0.95 | 0.2 | 5 | ✔ | moderate | keep |
| DEVICE_LAYER_CONSOLE | 0.3 | 0.2 | 30 | — | entangled | keep |
| SESSION_BROWSER | 0.2 | 0 | 300 | — | moderate | keep |
| REVIEW_EMBRYOS | 0.2 | 0 | 0 | — | entangled | keep |
| REVIEW_DETECTIONS | 0.2 | 0 | 0 | — | entangled | keep |
| REVIEW_CONVERSATION | 0.15 | 0 | 0 | — | entangled | keep |
| NOTEBOOK_NOTES | 0.2 | .02 | 0 | — | moderate | keep |
| NOTEBOOK_THREADS | 0.15 | .02 | 0 | — | moderate | keep |
| NOTEBOOK_ASK | 0.3 | 0 | 0 | — | moderate | keep |
| RUN_STATUS | 0.9 | 1 | 2 | ✔ | entangled | **drop as source, merge→ACQUISITION_CADENCE** — `#header-status` exists in no template (verified); `update()` returns at `experiment-strip.js:34` on every 1 Hz tick |
| NEXT_ACQUISITION | 0.8 | 1 | 1 | ✔ | entangled | **merge→ACQUISITION_CADENCE** (same, dead) |
| FRAME_LIVENESS | 0.7 | 0.2 | 3 | ✔ | entangled | keep **as new code** — capability is right, source is dead |
| DETECTION_INBOX | 0.6 | 0.1 | 0 | — | entangled | keep as new code; `#header-alert` is the one surviving id (`_header.html:20`, verified) but can never un-hide because its only un-hider is behind the dead `update()` |
| REPLAY_RECORDER_HEALTH | 0.05 | 0.25 | 60 | — | clean | **drop** — the file's own contract (`replay-recorder.js:4-6`) is that recording never surfaces |

### Occupancy3D / Marking / Viewer / Temperature

| Window | crit | freq | tol | Pin | Risk | Fate |
|---|---|---|---|---|---|---|
| STAGE_MINIMAP | 0.9 | 2 | 0 | ✔ | entangled | keep — folded rendering of the canonical STAGE POSITION |
| EMBRYO_MARKING (`marking.js`) | 0.9 | 0 | 0 | — | moderate | **merge** with `operate.js` marking |
| OPTICAL_VOLUME_3D | 0.55 | 5 | 2 | — | moderate | keep |
| WATER_TEMP_TRACE | 0.8 | 1 | 10 | — | clean | **merge→SAMPLE TEMPERATURE** |
| SCAN_GEOMETRY | 0.7 | .05 | 0 | — | entangled | keep |
| *(viewer.js)* | — | — | — | — | — | **zero windows — delete ~190 of 220 lines** |

### Settings page — `settings.js`

Lives on `templates/settings.html`, **not** among `index.html`'s ten `.tab-content` divs. These four (DASHBOARD_PREFS, THERMALIZER_CONFIG, EFFECTIVE_CONFIG, ADVANCED_OVERRIDES) are **out of scope for the Atrium surface**. Do not migrate them. One cross-cutting bug does need fixing regardless: `gently-dashboard-config` has two writers with divergent column migration (`settings.js:118` vs `embryos.js:293`).

---

## 2. The courtyard

Eleven pinned, out of ~55. Each earns it by one of three tests: **it is an actuator whose state you must never guess**, **it is the only halt**, or **it is a watchdog that reports while you are looking elsewhere**.

| Pinned | Why it cannot be travelled away from |
|---|---|
| **LASER GATE / EMITTERS** | "Is excitation live right now." Photodamage and eye safety. The code already treats it this way — `setLaserOff()` fires on every manual-view entry (`devices.js:2449`, verified). Its third gauge state (`warning: state unknown`) must survive the move; every failure path writes it rather than asserting OFF. |
| **STAGE POSITION** (+ TRAVEL FENCE proximity, + STAGE_MINIMAP folded) | crit 1.0. The only readout that tells an operator whether a commanded move went where they thought. Four unsynchronised copies today (`devices.js:149`, `operate.js:39`, `occupancy3d.js:57`, `marking.js:39` — the last set once per marking image and **never updated**), which means two surfaces can display contradictory positions right now. |
| **XY_INTERLOCK / F_DRIVE_APPROACH** | The server does **not** interlock XY against the F-drive; `routes/data.py` validates that x/y are floats and nothing else (`operate.js:128-132`). This predicate is the entire guard against driving the stage sideways with the objective in the sample. |
| **RUN_MONITOR (halt)** | `stopRun` (`operate.js:1050`) is the only control in `static/js/` that halts hardware acquiring on a live sample. There is no e-stop, no abort, no halt anywhere else in the tree. |
| **DEVICE_LAYER** | If the device layer is stopped or crashed, nothing acquires. A stale `ready` pill is a lie that invites a Start click. |
| **LINK** | Is Gently up, is the scope attached. Already behind `ConnectionStatus`; zero migration cost. |
| **PENDING_ASK** | An unanswered ask stalls the run indefinitely. The codebase already decided this twice independently (`agent-chat.js:1350` invented a sticky slot so the card cannot scroll away; `ask-stage.js` is a second workspace-wide renderer). That is evidence, not opinion. |
| **CONTROL_LOCK** | tol 0. A stale belief about who holds the lock is a wrong-hands-on-hardware failure. |
| **AGENT_RUN_STATE** | The only signal separating "thinking hard" from "hung", and `cancelTurn` is the operator's only interrupt on a self-woken agent. The elapsed counter exists precisely for that (`landing.js:63-64`). |
| **AMBIENT_WATCHDOG** | The only surface that reports an arrested embryo while the operator is looking at some other window. |
| **ACQUISITION_CADENCE** (incl. next-exposure countdown + frame liveness) | A frozen countdown is exactly how a stalled acquisition presents — `embryos.js:4269-4271` records a past bug where it only ticked on `VOLUME_ACQUIRED`. The countdown is to the next laser exposure on live tissue. |
| **SESSION_IDENTITY** | Writing to the wrong session is unrecoverable. Already in `_header.html:7-8`; needs a store read instead of DOM scraping, nothing more. |

**Conditionally pinned — SAMPLE TEMPERATURE.** A drifting bath silently kills a 12-hour run and nothing else in the UI will tell you. But `temperature-graph.js` is a passive trace with no threshold, no alarm and no tolerance band — it only echoes the controller's own `state` string (:146). **Pin it only after it gains a deviation band.** Until then it is decoration wearing a safety badge.

**Explicitly demoted, with reasons** (each was argued for pinning somewhere in the reports):
- **PLAN_PROVENANCE** — tol 0 and a real footgun (`viewVersion` overwrites `state.docData.document` in place, `campaigns.js:1097`), but it is an orientation fact about a *document*, not about hardware. Fix the footgun; do not spend courtyard space.
- **CAL_METRICS** — crit 0.9, genuinely arguable. Not an actuator, not a halt, not a watchdog. Flip it only if the courtyard has room after the twelve above.
- **SPIM_LIVE (gallery)** — a freshness watchdog on the acquisition camera, but its content is an image, and an image is not a one-glance gauge. The LED is the pinnable part; fold it into DEVICE STREAM HEALTH.
- **EVENT_SUMMARY** — fails the honesty test (see §1).
- **STAGE_BOARD** — 0.8 crit, but tol 300. A fact you can wait five minutes for does not belong in the courtyard.

---

## 3. Shared state and fan-out

**Verdict: a reactive store is required. A reactive *framework* is not.** `status-store.js` already implements the exact semantics needed — sticky snapshot, replay on subscribe, emit only on change, distinct signals kept unflattened — and its docstring documents that it was written to fix this bug class. `ClientEventBus` already carries the topics. The bus is ~80% of the store; **it is missing only retained state**. The work is extending one 60-line module, not adopting a framework.

| Blob | Where | Writers | Windows subscribing | Verdict |
|---|---|---|---|---|
| `state` global | `app.js:6-22` | `websocket.js:23,33,48,60,63,77,100,105,109`; `viewer.js:73,74,130,168`; `events.js:407,654`; `app.js:62,630` | **20 of 37 JS files reference it.** No change notification of any kind. | **Root of the store.** Every reader today either polls, re-renders on tab switch, or goes stale. |
| `state.embryos` + `detectionReasoning` | `embryos.js:11-20` | 8 WS handlers + `reconcileWithServerState` | 11 windows in-file + `experiment-strip.js:36,84,126,182` | Must be the root observable. `detectionReasoning` is also a dumping ground: `_pending` placeholders (:1246), in-place mutation (:1382), and synthetic `verification_*` rows (:1490) every consumer then filters back out (:2179, :3635). Clean the schema *before* subscribing to it. |
| **Selected embryo** | 4 independent copies: `embryos.js:26`, `operate.js:36`, `gallery.js:1246`, `devices.js:53` | 7 writers in embryos.js alone | ~12 windows across 4 files | The single strongest store case. Two windows cannot look at two embryos today; four files already disagree about which one is selected. |
| **Stage XY** | 4 copies: `devices.js:149`, `operate.js:39`, `occupancy3d.js:57`, `marking.js:39-40` (set once, never updated) | 3 event handlers, none shared | 6 windows | Second strongest. Two pinned surfaces can display contradictory positions **today**. |
| `agentBusy`/`busySource`/`askPending`/`streaming` | `agent-chat.js:24,37-39` | `setBusy`/`setAskState`, hand-propagated | 5 windows | Four-flag machine with no owner. Until it is in a store, the composer, the queue, the stop control and the ask card cannot be split without one going stale. |
| `hasControl`/`holderLabel`/`me` | `agent-chat.js:21,23,27` | `fetchMe`, WS | 4 windows + 4 external files via `AgentChat.hasControl()` (`ask-stage.js:23`, `landing.js:686,766`, `context-surface.js:17`) | Already a de-facto global with no store behind it. |
| `cacheDom()` single latch | `devices.js:161-162` — `if (_statusPill) return;` (verified) | — | gates ~90 element lookups for **all 14 devices windows** | Not state, a **structural blocker**. First caller latches permanently; every window mounting later gets null refs forever, silently. Must dissolve before any devices window moves. |
| `currentSessionId` | 5 derivations: `embryos.js:9`, `experiment-overview.js:24`, `marking.js:35`, `temperature-graph.js:23`, `app.js:126` (reads it out of the DOM) | — | ~6 | Add to `ConnectionStatus` as a fourth sticky signal. `app.js:126` is DOM-as-IPC and must die. |
| `_geom/_firmwareBox/_stage/_piezoZ/_galvo/_embryos` | `occupancy3d.js:55-61` | 3 handlers fanning out by hand-written direct call (`:337-339`, `:349-351`, `:363`) | 3 windows | Uncontrolled module cache. Cannot decompose before it has an owner. |
| `_optimalBox` / `_viewBox` / `_embryos` | `devices.js:157,159,42` | — | 3 windows, **cyclically** | `computeViewBox()` reads `_embryos`; `renderEmbryos()` reads `_viewBox` and bails when null; `renderMap()` hard-calls `renderEmbryos()`. A cycle, not a dependency. |
| `threadFilter` | `notebook.js:11` | `:43` | 3 windows, propagated by hand-calling two loaders | Smallest, clearest store case in the repo — **the pilot**. |
| `campaigns.state._inspectorData` | `campaigns.js:57-72` | — | 3 windows re-rendering each other through one cached payload (`:884,890,941,953,971,999,1019`) | — |
| `ReviewApp.currentSession` | `review.js:8` | 1 imperative innerHTML writer | 4 windows | — |
| `localStorage` keys as cross-window state | `gently-tasks-state`, `gently-dashboard-config` (**two writers**, `settings.js:118` vs `embryos.js:293`), `gently-detection-agreements`, `gently-badge-state`, `gently-last-check`, `gently-theme` (**two writers**, `app.js:400` vs `landing.js:829`) | — | — | Five-plus independent persistence schemes, no shared versioning, two with divergent last-writer-wins bugs. |
| Singleton DOM ids as IPC | `#chat-thread`, `#chat-input`, `.chat-panel`, `#detail-image-placeholder`, `#inline-detail-container`, `#cal-spim-img`, `#inspector-body [data-spec-key]`, `#session-picker-select`, `#status-left`/`#status-right`, `#tab-content` | — | — | Not state, but the blob nobody names. Every one breaks when its owning window can be instantiated twice. |

**Concretely: add four keys to `status-store.js`** — `sessionId`, `stageXY`, `selectedEmbryoId`, `agentBusy` (+ `hasControl`) — following the existing `set()`/`subscribe()` shape. That is roughly 60 lines and it dissolves the top five rows of this table. Everything else is per-group cleanup.

---

## 4. Transform hazards — Phase 0 checklist

Ships independently of the Atrium. **The correct pattern already exists in this repo:** `devices.js:1012-1021` uses `_mapSvg.getScreenCTM().inverse()` on an SVGPoint built from `clientX/clientY`, which composes every ancestor transform automatically. Copy it; do not invent a second approach.

### Corrupts data (fix first — these produce wrong numbers, not wrong pictures)

- [ ] `marking.js:152-157` — `canvas.getBoundingClientRect()` (transform-scaled) divided by `this.canvas.width` (unscaled backing store) to build `scaleX/scaleY`. **Verified.** At bench scale 2 every click maps to a pixel coordinate off by 2×, and `marking.js:173` ships that coordinate to the server as `embryo_marked` with no downstream validation. **Highest severity item in the entire audit.**
- [ ] `marking.js:130,143-146` — `containerRect` (scaled) written straight into `canvas.width/height` and `style.width/height`. **Verified.** Backing store s× oversampled and laid out s× too large.
- [ ] `marking.js:182-183,228-230,246,263,269` — all draw coordinates derived from the corrupted `canvas.width/height`; glyph and font sizes hard-coded px.
- [ ] `operate.js:25,544,553` — `MARK_HIT_PX = 14` compared against `getBoundingClientRect`-derived distances, i.e. a **screen**-pixel hit radius. At bench scale 0.4 it covers 2.3× more sample area, and `centerOnEmbryo` (`operate.js:574-579`) then commands a real absolute stage move to the wrong embryo.
- [ ] `operate.js:456-457` — `Math.round(r.width * dpr)` where `r` is transform-inclusive. **Verified** (`operate.js:450-460`). At bench scale 3 with dpr 2, a 1200 px canvas allocates 7200 px of backing store; every zoom step reallocates and clears. Past ~16384 px the allocation fails silently and the marker overlay goes blank.
- [ ] `devices.js:1156-1163` — `updateScalebar()` divides a transform-scaled `getBoundingClientRect().width` by an untransformed viewBox span. Printed scale-bar caption wrong by exactly the zoom factor. Recomputed every zoom frame via the `ResizeObserver` at `devices.js:223-225`.

### Pans the bench out from under the operator

- [ ] `embryos.js:2564` — `scrollIntoView({inline:'center'})` on an eval dot inside a translate+scale ancestor; walks every scrollable ancestor including the bench viewport.
- [ ] `embryos.js:4472` — second `scrollIntoView`, arrow-key driven, fires far more often.
- [ ] `campaigns.js:1192,1200` — `scrollIntoView({block:'center'})` on `#item-<id>` and bibliography anchors.
- [ ] `settings.js:225` — `scrollIntoView({behavior:'smooth'})`. (Settings page, low priority.)

### Pointer-delta / pan / zoom in the wrong space

- [ ] `zoom-pan.js:76-77,82-83,94` — **verified.** `offsetX = e.clientX - startX` written straight into `translate(${offsetX}px, …)`. Shared by `lightbox.js:43-55` and the main viewer, so this is a cross-group fix.
- [ ] `zoom-pan.js:34-37` — unconditional `preventDefault()` on the container wheel handler swallows the bench's own zoom gesture.
- [ ] `zoom-pan.js:41-42` vs `:45-48` — `bind()` attaches document-level listeners; `unbind()` exists and `Lightbox` never calls it.
- [ ] `devices.js:1288-1289,1296-1310,1326-1331,1338-1340,1367,1375-1377` — camera transform, viewBox conversion, clamp, wheel anchor, pan deltas.
- [ ] `devices.js:1906-1909,1920-1926,1932-1934,1952,1960-1962` — the lightsheet copy of all five, identical.
- [ ] `agent-chat.js:1189,1214,1166,1161-1162` — chat resize drag. **All four disappear when `agent-chat.js:1143-1256` is deleted.**
- [ ] `gallery.js:1530-1537,1546-1556,1582-1591,1596-1602,1570-1579` — `SpimPopout` drag/clamp/persist. **All five disappear when the popout is deleted.**
- [ ] `occupancy3d.js:153-160` — orbit uses raw `clientX/clientY` deltas × fixed 0.01 rad/px.
- [ ] `occupancy3d.js:163-167` — `wheel` + `preventDefault` + `{passive:false}` on the WebGL canvas; swallows bench zoom.
- [ ] `projection-viewer.js:396-412,416-420` — drag rotation gain 0.01 rad per `clientX` px; wheel `preventDefault`.
- [ ] `embryos.js:738-747` — `scrollContainer.scrollLeft += e.deltaY`. One-line fix: divide by bench scale.
- [ ] `campaigns.js:1867-1871` — Gantt wheel → `scrollLeft`, same.

### Fixed dimension vs. scaled box

- [ ] `embryos.js:689,691` — `?size=${thumbSize * 2}` raster baked for a fixed 56px CSS box; `devicePixelRatio` and bench scale never consulted.
- [ ] `experiment-strip.js:228` — `/api/images/${uid}/png?size=96` hard-coded.
- [ ] `projection-viewer.js:315-319,330-331,338` — `container.clientWidth` (layout, transform-blind) with `h` hard-coded to 400; `setPixelRatio` never called.
- [ ] `occupancy3d.js:109-110,142-143,147` — `clientWidth/clientHeight` is correct, but no `setPixelRatio` anywhere in the file (`devicePixelRatio` appears zero times across all four spatial files).
- [ ] `embryos.js:575-577` — board sparkline fixed 160×28 with a fixed 20-sample window; no resample-to-width path.
- [ ] `embryos.js:898,908,585,934,949-951` — SVG `font-size="9"`, `stroke-width`, `r=4`, dasharray in chart units with no legibility floor. At bench 0.3 the axis labels are ~2.7 visual px.
- [ ] `campaigns.js:1800-1804,1814-1816,1829-1854` — Gantt geometry hard-coded px in a nested native-scrollbar container.
- [ ] `lightbox.js:218,457` — `const visible = 9` thumbnails regardless of strip width.
- [ ] `occupancy3d.js:409,427-429` + `index.html:956,970` — 200×120 minimap hard-coded in three places; internally consistent, but cannot reflow in a resizable frame.
- [ ] `temperature-graph.js:65-66,86-88` — `const H = 160` with `width="100%"` and no height attribute; aspect frozen.
- [ ] `agent-chat.js:1140` — `Math.min(input.scrollHeight, 140)` mixes a layout measurement with a visual ceiling.

### Perceptual thresholds in layout px (low severity, listed so they are not rediscovered)

- [ ] `embryos.js:625,628` — `FOLLOW_THRESHOLD_PX = 24`.
- [ ] `campaigns.js:1221-1224` — scroll-spy differences two rects against the literal `100`.
- [ ] `landing.js:199-203` — 140px "near the bottom".
- [ ] `agent-chat.js:299` — `< 60` pin-to-bottom.

### Invalidation, not arithmetic

- [ ] `app.js:178,196-216` — `Tooltips.show` mixes a transformed target rect with an untransformed body-parented tooltip, and its **only** dismissal triggers are `mouseleave` and a capturing `scroll` listener. **A bench pan/zoom is a transform change, not a scroll event** — the tooltip freezes at old viewport coordinates while its anchor slides away.
- [ ] `operate.js:1257,1262-1265` and `projection-viewer.js:349-353` and `occupancy3d.js:75,104-105,144` — redraw triggered by `resize` + `ResizeObserver`. **A CSS transform fires neither** (ResizeObserver reports the layout box). Backing stores stay stale after zoom.
- [ ] `operate.js:437,450` and `marking.js:65-68` — geometry helpers return null on a zero-sized rect, which is exactly a **folded** window. Unfolding needs an explicit redraw path that does not exist.

### Confirmed SAFE — do not re-audit or "fix"

- `embryos.js` — **zero** `getBoundingClientRect` in 4556 lines (verified). All measurement is `scrollWidth/clientWidth/scrollLeft` on one element: layout-space, self-consistent.
- `experiment-overview.js` — **zero** transform-unsafe sites in 1359 lines (verified). Pure flow layout; migrates untouched.
- `devices.js:1012-1021` — `getScreenCTM().inverse()`. The reference implementation.
- `timepoint-player.js:766-767` — rect + clientX, but the ratio is exact under any translate+scale.
- All SVG in `campaigns.js`/`gallery.js` uses `viewBox` (`campaigns.js:1626`; `gallery.js:82,170,821`); `CHART_W/H/MARGIN` and `SVG_WIDTH/HEIGHT` are viewBox-internal units.
- `device-layer.js:323` (`scrollTop = scrollHeight`), `agent-chat.js:2936,3019` — layout-space, internally consistent.
- `utils.js:206-255` (`extractFirmwareBox`, `makeSceneScaler`) — µm↔scene-unit math, **not** screen space. Leave in the library; most likely thing to be mistaken for screen math during migration.

---

## 5. Migration order

Each phase is independently shippable and revertable. **CI runs no JavaScript** (`CLAUDE.md`), so every phase names its hand check. Run everything with:

```bash
GENTLY_STORAGE_PATH=/tmp/gently-dev uv run python launch_gently.py --no-api --no-auth --no-browser
```

### Phase 1 — Delete dead code (no behaviour change, ~1500 lines)

Cheapest possible high-value work. Nothing here is an Atrium change; it is the difference between migrating 55 windows and inventing 12 phantom ones.

- `embryos.js` — the `renderDetectionListWithCollapse` subtree (~900 lines) + `renderEmbryoCardFull` + first-run hints + `view3D`/`showToast` + `compareDetection`.
- `devices.js:569-634` + `_ROLE_COLOR:43-47` — the shadowed `renderEmbryos`.
- `viewer.js` — ~190 of 220 lines. The display half has been unreachable since the Live View tab was removed (`MainViewerZoom.init` returns early at `:17`; no `main` tab in `utils.js` TABS — **verified**, grep for `'main'` returns nothing). Keep only the ingest router `handleNewImage`/`handleNew3DVolume` (`:57-70,:133-165`) and move it to the store layer. **Also fixes a live TypeError**: `gallery.js:1210` renders `onclick="show3DVolume(...)"` → `viewer.js:207` → `:82` dereferences `#z-slider`, which no template contains (verified).
- `gallery.js:1403-1615` — `SpimPopout` (213 lines). Also removes five transform hazards.
- `gallery.js:867-899` — `CAL_SLICE_DETAIL`, renders into `#cal-profile-detail` which exists in no template.
- `agent-chat.js:1143-1256` — panel chrome (~110 lines). Removes four more transform hazards. **Must first find and rehome every listener of the `gently:layout-changed` custom event** (the 3D canvas resizes off it).
- `timepoint-player.js` — modal choreography (~40% of file).
- Dead fields: `app.js:546,603`; `campaigns.js:66,194,1802`; `gallery.js:970`; `experiment-strip.js` entirely (see phase 5).

**Hand check:** load every one of the ten tabs; open a detection detail, the calibration profile, the lightbox, a 3D volume, the chat panel. Console must be clean. `git revert` is a single commit.

### Phase 2 — Fix the live bugs the audit surfaced (no Atrium dependency)

These are wrong *today*, on tabs, at scale 1.0.

1. `landing.js:668-680` vs `:577` — `recordPick` appends recorded answers into `#v2-plan-summary`; `drawPlanPage` wipes it with `innerHTML=''` within a second (600ms debounce, `:222`). User's recorded choices destroyed.
2. `experiment-overview.js:209-212` vs `:138-148` — `render()` unconditionally resets `_planPickerOpen`/`_pickerItems`; any of eight tactic events closes the operator's open plan picker and discards the selection.
3. `campaigns.js:712-720` + `:1262-1267` + `:1026-1029` — `normalizeSnapshotTree` defaults snapshot items to `planned`, so "▶ Run this imaging item" renders on **historical** items and would send the agent `_snap_3_a9f2k1`.
4. `embryos.js:2874-2876,2632,2950-2951` — root-scope `chat-thread`/`chat-input`/`detail-image-placeholder` to the detail panel's own element, and have `renderDetailPanel` call `initChatPanel` itself. Fixes chat silently dying after a view switch (default view precedes filmstrip in `index.html:347`, so `document.querySelector('.chat-panel')` finds the hidden copy), and fixes `_renderBoardDetail:605` which ships a dead chat form today.
5. `devices.js:648-687` vs `:963-1010` — key-schema mismatch (`e.embryo_id`+flat `x/y` vs `e.id`+`position_fine/coarse`). `EMBRYO_DETECTED` appends entries that render as nothing; `handleStatusChanged:685` falls back to `loadEmbryos()` which blanks every waypoint.
6. `events.js:227-233` — dead check (verified: `:208-210` already trimmed, so `length > MAX_EVENTS` can never be true). `_filteredCount` only increments and the display shows `612 / 500 events`.
7. `events.js:135` — reads `state.volumes`; `app.js:13` defines `volumes3d` (verified). Event→volume linking has never matched.
8. `timepoint-player.js:852-856` vs `:387` — two index maps for the same jump; gappy sequences land on the wrong frame.
9. `gallery.js:935-940` vs `:344-354` — `_handleSVGClick` re-derives `edgeData` with a different filter than `render()` used; `data-index` can address the wrong image.
10. `marking.js:122,380` — `querySelectorAll('.marking-action-btn')` also matches the Device Layer Start/Stop/Log buttons (`index.html:515-519`). Finishing marking disables the device layer controls. Scope to the marking subtree.
11. Two-writer localStorage: `gently-dashboard-config` (`settings.js:118` / `embryos.js:293`), `gently-theme` (`app.js:400` / `landing.js:829`).

**Hand check:** one per bug, each is a two-step reproduction. Bugs 4 and 5 need the device layer running.

### Phase 3 — Transform safety (§4 checklist)

Ships to the current tabbed UI unchanged (all fixes are no-ops at scale 1.0), so it is fully revertable and de-risks everything after it.

Order within the phase: **data-corrupting sites first** (`marking.js:152`, `operate.js:456`, `operate.js:544`, `devices.js:1156`), then the shared `zoom-pan.js` controller (one fix, three consumers), then per-window pan/zoom, then fixed dimensions, then `Tooltips` invalidation.

**Hand check:** there is no bench yet, so verify by temporarily applying `transform: scale(0.5)` / `scale(2)` to a container in devtools and confirming: a marking click lands on the pixel under the cursor at both scales; the stage map scale bar caption is unchanged; camera pan tracks the cursor 1:1; the tooltip follows its anchor. **This is the phase where you need the real microscope** — a mis-mapped marking click is a stage move, so verify `marking.js` against a real coverslip before shipping.

### Phase 4 — The store

Extend `status-store.js` with `sessionId`, `stageXY`, `selectedEmbryoId`, `agentBusy`/`hasControl`. Keep the existing shape: `set()` guards on equality, `subscribe()` replays. Then convert readers one at a time — each conversion is its own revertable commit.

Ordering matters: **`stageXY` first** (four unsynchronised copies, one pinned window depends on it), then `selectedEmbryoId` (four copies, twelve readers), then `agentBusy`/`hasControl` (five readers, eight files), then `sessionId` (removes `app.js:126`'s DOM read).

Prerequisites inside this phase:
- Dissolve `devices.js:161-162`'s `cacheDom()` latch into per-window DOM resolution. Structural blocker for all fourteen devices windows.
- Break the `devices.js` map↔waypoints cycle: `_optimalBox`, `_viewBox`, `_lastXY`, `_embryos` become subscribable derived values.
- Give `occupancy3d.js:55-61`'s module cache an owner.
- `temperature-graph.js:19-29` — IIFE singleton → constructor, so a courtyard gauge and a bench window can coexist. `devices.js:2437-2444` already re-inits it on every view switch for exactly this reason.

**Hand check:** open Devices and Operate simultaneously (two browser tabs against one backend is not the same test — use the tab switcher) and confirm the stage position readouts agree after a move; select an embryo in one surface and confirm the other follows.

### Phase 5 — Pilot windows: `notebook.js` and `device-layer.js`

Three small event-driven windows with one shared filter and **zero coordinate math**, plus the courtyard's one real watchdog. If the Atrium window contract is wrong, this is where you find out for 400 lines instead of 4500.

Rebuild `experiment-strip.js`'s four capabilities here **as new code against the store** — do not lift. Nine of its ten target ids exist in no template (verified: only `#header-alert` survives, `_header.html:20`), `update()` has been returning at `:34` on every 1 Hz tick, and `handleAlertClick:176` navigates to `switchTab('tasks')`, a tab not in the TABS enum. Delete the file.

**Hand check:** notebook filter/thread/ask round trip; start and stop the device layer and watch the pill, uptime ticker and log auto-open; kill the device layer process externally and confirm the pill goes to `crashed` within one poll.

### Phase 6 — Pure projections: `campaigns.js`

Six windows (doc, graph, board, decide, matrix, timeline) are pure folds of `collectAllItems(state.docData.document)` fighting over one `$canvasContent` (`campaigns.js:90`, written by nine functions). `state.planView` + the `renderCanvas()` dispatch at `:387-394` is literally a hand-rolled window manager with one slot. **This phase is net line reduction**: the switcher, the mode flag, and the `if (planView === 'graph')` branch in `applyTypeFilter` (`:1399`) all delete.

Also here: `gallery.js` SNAPSHOT_BROWSER (cleanest lift in the repo), FOCUS_CURVES, VOLUME_SEGMENTATIONS_3D, `events.js`'s three views (deletes `switchSystemView:445-464`), `experiment-overview.js`'s LINKED_PLANS and REACTIVE_RULES (zero transform hazards, four endpoints of their own).

**Hand check:** open a campaign with ≥3 phases and ≥15 items; confirm all six projections render simultaneously and agree on counts; toggle the type filter and confirm all four consumers update (today `filterGraphByType` at `:1662` is a divergent second implementation).

### Phase 7 — The courtyard

Build the eleven pinned windows. All of them read from the store built in phase 4. Nothing new is invented except the temperature deviation band.

**Hand check:** on the real rig — arm a laser preset and confirm the courtyard gauge changes; drive the F-drive down and confirm the interlock gauge and distance-to-floor track; stop a running acquisition from the pinned halt.

### Phase 8 — Entangled work, last

- **`embryos.js` SUBJECT window.** Prerequisite, independent of the Atrium: replace the `container.innerHTML = html` wipe-and-rebuild at `:1280`/`:1419` with incremental append. The scroll capture, auto-follow heuristic, active-cell restore and chat re-mount exist **only** as compensation for destroying the DOM, and all four delete themselves once it stops. Also route `:413`/`:710`/`:809` through `selectEmbryo()` (three views write the cursor directly and skip the cleanup — a live defect), and scope `updateEmbryoCard`'s unscoped `document.querySelector('[data-embryo-id=…]')` at `:4197` to `#embryo-cards` (`data-embryo-id` is also on `.board-row:460`, `.filmstrip-cell:687`, `.vitals-strip:947`).
- **`devices.js`** map + waypoints, after the phase-4 cycle break.
- **`operate.js`** — see §6; this is gated on solving `_pane`.
- **`agent-chat.js`** — after `agentBusy` is in the store.
- **`review.js`** — after `currentSession` is a store and the inner tab strip dies. Note `review.js:146` injects `<div class="tab-content" id="tab-content">`, which `app.js:72` already sweeps — a live collision, not hypothetical.

### Not worth decomposing — leave alone

| File | Why |
|---|---|
| `utils.js` | Genuinely one thing: eight shared helpers, zero windows. Two caveats only — `toggleDropdown`'s `_activeClose` singleton (`:329`) allows exactly one open dropdown document-wide, and `initViewSwitcher` binds digit keys on `document` (`:301`). Both assume one visible surface. |
| `lightbox.js` | **One capability.** Splitting the thumbnail strip, metadata panel and image pane would be pure fan-out over the same `currentIndex`/`imageList`. It does need to become instanceable (four files poll its mutable fields; `:136,293,342` mutate `document.body.style.overflow`; `:79` binds an unguarded document keydown) and `:171-266` vs `:422-498` is ~120 lines of near-verbatim duplication that must collapse first. |
| `marking.js` | One window. The canvas and the sidebar roster are the **open and folded renderings of the same marker list**; splitting them forces both open to be usable. |
| `operations-scenarios.js` | 372 lines of fixture data read only by `experiment-overview.js:35-38` via `?scenario=`. No capability, nothing to decompose. |
| `replay-recorder.js` | Invisible instrumentation whose stated contract (`:4-6,76`) is that it never surfaces. Shipping a health window reverses that decision; it is not a migration. |
| `settings.js` | Lives on `templates/settings.html`, a separate page, not on the Atrium surface. Also: `templates/settings.html:99-121` (the Device layer section) has **no JavaScript anywhere in the repo** — it falls through to `settings.js:239-242` and toasts "Settings saved" while persisting nothing. A control that lies. Do not promote it to a window until it has a backend. |
| `viewer.js` | Zero windows. Delete, keep the 30-line ingest router. |
| `agent-chat.js:49-291` | 240-line markdown parser, already consumed by `landing.js:210`. A library, not a window. Do not duplicate it. |

---

## 6. Honest costs

### The three things that can hurt the microscope

**1. `_pane` and MMCore mutual exclusion.** `operate.js` keeps two MJPEG decoders off MMCore by an emergent property: "the camera is live only while you are looking at it" (`operate.js:1122,1132,490,1283`; `PANES.onEnter/onLeave:1063-1077`; `devices.js:2453` calls `OperateManager.activate()/deactivate()`). There is no lock, no arbiter, no server-side exclusion. The file names the failure mode explicitly at `:1057-1061`: two simultaneous decoders caused a **Video-TDR display-driver freeze on the microscope PC**. The Atrium's core promise — nothing is created or destroyed, every window is permanently live — is the exact opposite guarantee. **A real ownership token must exist and be tested before a single camera window moves.** This is not a refactor risk, it is a GPU reset on a machine running a 12-hour experiment.

**2. The laser ALL-OFF trigger.** `devices.js:2449` (verified) fires `setLaserOff()` on manual-view entry. View entry is precisely what the Atrium deletes. The LASER GATE window needs an explicit mount-time / session-start ALL-OFF, and the third gauge state (`warning: state unknown`, written by every failure path rather than asserting OFF) must survive.

**3. Marking click coordinates.** `marking.js:152-157` and `operate.js:544` are the only sites where the bench transform produces silently *wrong data* rather than a wrong-looking picture, and both feed absolute stage moves. Everything else in §4 is cosmetic, a crash you would notice, or dead code.

### Where it gets expensive

- **`embryos.js` incremental rendering.** The prerequisite for the SUBJECT window is replacing the innerHTML wipe-and-rebuild — that is real work with no user-visible payoff on its own, and it touches every view.
- **The `state` global.** 20 of 37 files. Converting it is not a phase, it is a slow migration behind a store that both shapes coexist with for a while.
- **Cross-file window merges.** Stage position ×4, roster ×4, SPIM ×3, marking ×2 — each merge means picking which implementation wins and deleting a working one. Politically and practically the slowest part.
- **`rrweb` blockSelector rot.** `replay-recorder.js:40,46` hard-codes `#op-img-bottom`, `#op-img-spim`, `#devices-map-svg`, `#occ3d-container`, `#occ3d-minimap`, `#devices-temp-graph`. A selector that matches nothing does not error — it silently stops blocking, and rrweb then records base64 data-URI src swaps on the live camera `<img>` at frame rate on the main thread. Its own guard (`auditSelectors:366-387`) only reports when *some* selector still matched, and its comment names the blind spot: simultaneous rot reads as "wrong page" and stays silent. **A wholesale Atrium restructure is the simultaneous-rot case.** Fix the audit before renaming any of those ids.
- **`three.js` r128 from a CDN** (`templates/index.html:1423`). `THREE.DataTexture3D` (`projection-viewer.js:463`) was renamed `Data3DTexture` in r137. A permanently-live WebGL window pulling its engine off cdnjs at page load is a new availability dependency for a microscope PC.

### What could go wrong quietly

- **`landing.js`'s `planActive()` gate** (`:44-56,838-843`). Overlay *visibility* is the authorization check for whether agent events are processed. In the Atrium nothing is hidden, so it evaluates false forever and four windows go silently deaf. They will look alive and receive nothing.
- **Folded windows are zero-sized.** `operate.js:437,450` and `marking.js:65-68` bail on a zero rect; `occupancy3d.js:75,104-105` defers sizing to rAF because the container was 0×0 while hidden. Unfolding needs an explicit redraw hook that does not exist anywhere today.
- **`projection-viewer.js:435-446`** runs an unconditional rAF loop with a 256-step-per-pixel raymarch for as long as the viewer is initialised. A permanent window runs that forever behind a folded gauge unless a fold gate is added.
- **Permanent timers with no owner.** `devices.js:2290,2411` (two 15s intervals started by run-once setups), `experiment-overview.js:64-88` (ten bus handlers registered once, never removed, with a `_tempUpdateHandler` stored "so it can be `off()`'d if needed" and nothing ever does). `temperature-graph.js:159`'s `dispose()` is the in-repo pattern to copy.
- **`window.confirm`/`alert`** at `devices.js:1057,1063,1085,1091`, `device-layer.js:251,260`, `review.js:102,111,114`. Native modals block the main thread — the pinned courtyard, the position gauge and the laser indicator all stop updating for as long as the dialog is up. Every destructive control in the app goes through one.

### What I would need to see on the real rig before trusting it

1. A marking click at bench scale 0.5 and 2.0 that centres the stage on the embryo actually under the cursor, verified against a coverslip with known geometry.
2. Both camera windows unfolded simultaneously for 30 minutes with no display-driver reset — under the new ownership token, with the token deliberately failed once to confirm the fallback.
3. The F-drive interlock: drive down past the floor threshold and confirm the pinned gauge latches and XY moves are refused, then back off and confirm it clears.
4. A laser preset armed, then the page reloaded — the courtyard must show either the correct state or `warning: state unknown`, never a confident OFF.
5. A 12-hour overnight run with the full courtyard mounted, checked for memory growth (permanent windows, permanent timers, permanent bus subscriptions, no `dispose()`) and for the temperature gauge correctly flagging a deliberately induced 0.5°C excursion.
6. `Ctrl+Shift+I` open the whole time: any `TypeError` from a null DOM ref is the `cacheDom()` latch or a singleton-id collision reappearing.