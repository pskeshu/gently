# Gently web UI → agent-first paradigm: migration plan

Strangler-fig migration of the **existing** stack (FastAPI + Jinja2 + vanilla-JS, `gently/ui/web/`). **No SPA rewrite.** Everything new is gated behind a `GENTLY_UX_V2` flag and layered onto the seams that already exist. Target paradigm is the prototype in `ux-prototype/landing.html`.

## Why this is cheap (the load-bearing discoveries)

- **The structured-ask protocol already exists as data.** The agent emits `{type:'choice_request', request_id, choice_data:{question, options[], _type, allow_multiple}}` over `/ws/agent` (`conversation.py:617`, `bridge.stream_response:648`); the client replies `{type:'choice_response', request_id, selected}` (`agent_ws.py:769`). "One payload, two renderers" = add a `_kind` discriminator + factor `agent-chat.js renderChoice` (L356-390) into a pure `buildAskCard()` + a second mount. **Not a protocol rewrite.**
- **There's already an inference-first precedent**: `bridge.bootstrap_resolution_picker` builds inferred pickers in-memory from candidates without persisting. Plan-mode's draft-first flow models on it.
- **One init chokepoint**: `switchTab(name)` (`app.js:60-103`) is the *only* caller of every tab's manager init. The new shell **calls** it for each region reveal — never reimplements activation.
- **Two separate sockets**: `/ws` (telemetry + server `EventBus` fan-out) and `/ws/agent` (chat + asks). They have different lifecycles — the context surface (Phase 4) rides `/ws`, the ask-stage rides `/ws/agent`.

## Coexistence (how old + new run side by side)

A Jinja2 flag: `pages.py GET /` passes `ux_v2` (new `GENTLY_UX_V2` setting, default off) into `index.html`. The template keeps **both** the current 8-tab markup and the new grouped-rail markup, mutually exclusive via `{% if ux_v2 %}` + a `body.ux-v2` class. New JS modules (`status-store.js`, `ask-stage.js`, `shell.js`, `context-surface.js`) load on every page but **no-op without `body.ux-v2`**, so they can't regress v1. Same URL, same shell, same uvicorn process. Flip default-on after a soak (Phase 6), then delete v1 markup. Both UIs read the same state objects and sockets, so v1/v2 can be compared side-by-side on identical live data.

> CSS hazard: `main.css` has **duplicate** `.tab`/`.tab-content`/`.status-dot` rulesets (~L547 and ~L2892). Consolidate or strictly scope v2 under `body.ux-v2` **before** Phase 2 touches nav CSS.

## Phase sequence

| # | Phase | Ships | Flag | Depends |
|---|-------|-------|------|---------|
| 0 | Bug-fix beachhead: sticky status store + idle-telemetry | Status unification + quiet idle channel, **to prod now** | none | — |
| 1 | Dual-render the ask protocol + correct clear-signal | The paradigm enabler | ux_v2 | 0 |
| 2 | Shell unfold + grouped nav + session-context strip | The calm welcome→workspace | ux_v2 | 1 |
| 3a | Inference-first plan mode **backend** (headless) | Draft-from-strain + per-field provenance | — | 2 |
| 3b | Inference-first plan mode **UI** (`plan_confirm` renderer) | Draft renders with provenance | ux_v2 | 3a |
| 4 | Co-editable FileContextStore surface + proactive cards | Shared visibility (beliefs/attention/uncertainty) | ux_v2 | 3b |
| 5 | Carve per-embryo tactical Experiment view out of `embryos.js` | The tactical view mount (contents TBD) | ux_v2 | 4 |
| 6 | Default-on flip + v1 deletion | Irreversible cutover, isolated/soaked | flip | 5 |

## Phase detail

### Phase 0 — Bug-fix beachhead (no flag, pure value)
- **Single sticky/replaying `ConnectionStatus` store** (`status-store.js`) holding `{gentlyConnected, microscopeConnected, agentConnected}` and emitting `CONNECTION_STATUS`. Must **replay last state to late subscribers** or bug #1 just moves.
- **Bug #1**: header pill (`updateTopLevelDot` `app.js:562`), home line (`home.js updateStatus:136` — today reads `state.connected` once, the literal "Offline while connected" bug), and dock dot all **subscribe** and re-render on every event.
- **Bug #3 (measure first)**: code shows **15s** polls (not 1-2s); the real idle cost is likely the ~5Hz `DEVICE_STATE_UPDATE` WS stream. **Ship unconditionally:** decouple `#events-count` from `DEVICE_STATE_UPDATE`/`BOTTOM_CAMERA_FRAME`. **Gate polls only if** measurement implicates them; if it's the SSE stream, coalesce/backoff in `device_state_monitor.py`. Gate on a stable `DevicesManager.active` flag, **not** switchTab internals (Phase 2 rewrites those).
- **Bug #2**: capture the prototype's correct multi-select contract (Continue mounts with disabled state *derived from current selection*) for `buildAskCard`.
- Files: `status-store.js` (new), `websocket.js`, `app.js`, `home.js`, `agent-chat.js`, `devices.js`, `events.js`, `index.html`.
- Verify: cold + reload + kill each socket independently — all three indicators agree within one handshake; kill device layer → microscope badge flips in 15s, gently stays Online.

### Phase 1 — Dual-render the ask protocol (the enabler)
- `_kind` always-present discriminator + additive `_surface` ∈ {transcript,stage,both} (default transcript so the Ink TUI/older clients are unaffected). Set on each payload the bridge builds.
- Factor `renderChoice` → pure `buildAskCard()` + a module-level `answered` Set keyed by **opaque** `request_id` (never parse a prefix — ids mix `req_N` and `resolve_*_<uuid>`). New `ask-stage.js` renders the same card into `#ask-stage`.
- **BLOCKER fixed — clear signal**: fire `ASK_CLEARED` the instant a `choice_response` is sent (plus on cancel/error/control-loss/socket-close), **not** on `stream_end` — an in-turn ask suspends on `asend` (`bridge.py:657`) and `stream_end` only arrives *after* the answer; a cancelled turn emits none.
- **BLOCKER fixed — dismiss vs control**: `agent_ws.py:772` silently drops non-holder responses. Stage renders read-only when `!hasControl`; only the holder answers/dismisses; a holder's escape posts a real empty `choice_response` so the turn-lock releases.
- **BLOCKER fixed — leak cleanup**: pop orphaned `_choice_futures` on holder-change and answerer-disconnect, not only last-client (`agent_ws.py:889`).
- Add the **free-text "Something else"** affordance web cards lack today (`bridge._dispatch_resolution_pick:462` already routes unknown selections to LLM resolution).
- Verify: trigger the session-open bootstrap picker — appears in chat **and** `#ask-stage`; answering either clears both; cancel a turn mid-ask → stage clears and next turn works; lose control mid-ask → stage goes read-only.

### Phase 2 — Shell unfold + grouped nav + session strip
- `shell.js` screen-state machine sets `body[data-screen]` (welcome|plan|standalone|shell) and **calls `switchTab()`** for every reveal (+ a dev assertion if a region shows without its init).
- Grouped left rail (Now / Library / System) → each item maps to an existing `data-tab` id via `switchTab`. Session-context strip reuses `ExperimentStrip` (fix stale `switchTab('tasks')` at `experiment-strip.js:176`), fed by `/api/experiments/current/strategy`, reading live status from the Phase 0 store.
- **Real routing**: replace the consume-once hash (`app.js:633` `replaceState` to `/`) with deliberate URL/state sync so refresh/back-button + the `/review`→`/#sessions` redirects resolve correctly.
- Decouple welcome→plan from the brittle `togglePanel(true)+setTimeout(250ms,'/wizard')` (`home.js:159`): the picker renders in `#ask-stage` on connect, driven by the bootstrap `choice_request`.
- **Resume**: replace the `window.location.href='/'` hard reload on `session_changed` (`websocket.js:147`, `review.js:108`) with in-place re-hydration (jarring otherwise).
- Verify: cold load → calm welcome; choosing a plan unfolds (no hard cut); each rail item fires its manager's init side-effect (verify the side-effect, not just visibility).

### Phase 3a — Inference-first plan backend (headless-testable)
- Flip the plan-mode prompt from ask-first to **infer-first** (`plan_mode_system_prompt.tex`, `harness/plan_mode/prompt.py`): arrive with a draft, ask only for genuine gaps / low-confidence / consequential confirmations.
- Deterministic **strain→channel** inference in `research.py`: parse genotype from `search_strains` (TagRFP→561, GFP→488), attach source ref. **Must degrade to "confirm" — never fabricate a wavelength.** Network-dependent (WormBase REST + CGC scraping), so the degrade path is load-bearing.
- **Per-field provenance** (`model.py:186`): `ImagingSpec` fields are flat scalars with no per-field source (`references[]` is on `PlanItem`, not the spec). Add a parallel `{field → {source, confidence, citation}}` map; reuse the existing `Confidence` enum.
- **Drafts stay in-memory** (like `bootstrap_resolution_picker`), materialized via `create_campaign`/`create_plan_item` only on explicit confirm — avoids a `PlanItemStatus` enum change *and* orphan-folder cleanup.
- Use `gap_assessment.assess_gaps()` to *select* which gaps to ask (don't re-enable the deliberately-disabled multi-question wizard; note `conversation_weight` short-circuits after onboarding).
- Verify (no UI): known strain → draft with channels pre-filled + per-field source/confidence; unknown/offline → "confirm channel", never fabricated; reject → no folder written; confirm → materializes and `GET /api/campaigns/{id}/document` returns the tree.

### Phase 3b — Inference-first plan UI
- `_kind:'plan_confirm'` ask: bridge emits the in-memory draft as `choice_data`; `ask-stage.js` renders cards with per-field source tags + confidence + edit affordances; chat shows a **compact reference line** (not a duplicate) so the surfaces can't drift.
- Extend `renderSpec` (`agent-chat.js:403`, today a flat key→value table) with a **source column**.
- Confirm posts the same `choice_response`; bridge materializes; stage clears via `ASK_CLEARED`.

### Phase 4 — Co-editable context surface + proactive cards
- **BLOCKER fixed — real store change**: `FileContextStore` has no event bus. (1) add `CONTEXT_UPDATED` to the closed `EventType` enum (`core/event_bus.py`); (2) inject a bus/callback into the store; (3) emit a **single coalesced** event from each mutator (`add_expectation:1880`, `add_watchpoint:1933`, `add_question:1980`, `update_embryo_understanding:2049`, …); (4) wire in `launch_gently.py:497`.
- New `routes/context.py`: **read** side models on `campaigns.py` (`_serialize`); **write** side uses `Depends(require_control)` from `data.py`/`sessions.py` (NOT the `campaigns.py` mesh/account auth) so a viewer can't mutate the agent's mind.
- Live updates over `/ws` (telemetry socket) → `websocket.js:117` → `context-surface.js`. **Push on change, never poll** (`load_active` scans ~50 observations + YAML).
- `context-surface.js` renders beliefs/attention/uncertainty as a calm panel in the Now region with inline edit/resolve (disabled when `!hasControl`).
- **Proactive cards**: wire watchpoint/question creation + the existing wake-router `origin:'wake'` approvals (`agent.py:1079`) to surface prominent `#ask-stage` cards — real backing for the prototype's attention card, no new mechanism.

### Phase 5 — Carve the tactical Experiment view (behind the flag, before the flip)
- Make Experiment a distinct renderer over `EmbryosManager.state` + the strategy snapshot rather than overloading the 4556-line `embryos.js`. **Preserve `reconcileWithServerState`/`clearAllState` as the contract.** Don't over-specify contents yet — it's a mount point.
- **Remove the `STUB_STRATEGY` fallback** (`experiment-overview.js:14` + the "mockup · stubbed data" badge) → real loading/empty state. Production must never render stubs.
- Stays behind the flag through its own soak so a reconciliation regression is caught before the irreversible flip.

### Phase 6 — Default-on flip + v1 cleanup (irreversible, isolated)
- Flip `GENTLY_UX_V2` default-on after soak; delete v1 `{% else %}` nav markup, superseded v1 status writers, and the dead/duplicate `.tab` CSS. Isolated from Phase 5 so the high-regression carve-out never coincides with deletion.

## Blockers the adversarial pass caught (now folded in)
1. **Clear signal must follow the choice lifecycle, not the stream lifecycle** (asend suspension; cancelled turns emit no `stream_end`).
2. **Dismiss vs control gate** — non-holder responses are silently dropped; only the holder dismisses; server cleans orphaned futures on holder-change/disconnect.
3. **Phase 4 store change is real, not free** — new `EventType`, bus injection, coalesced emit from every mutator, launch wiring.
4. Two answer paths (`_choice_futures` vs bridge-owned `_pending_import`) → client-authoritative `answered` Set + bridge idempotency guard.
5. `switchTab` is the sole init chokepoint → shell calls it, never reimplements.
6. Per-field provenance doesn't exist today → added in 3a.
7. Phase 3 split into headless backend (3a) + UI (3b); embryos.js carve-out isolated in Phase 5.

## Open decisions (yours)
- **Measure bug #3 first**: is idle chatter the ~5Hz `DEVICE_STATE_UPDATE` stream or the 15s polls? Determines the lever (coalesce in `device_state_monitor.py` vs gate polls).
- **Status**: client-computed sticky store (chosen for Phase 0) vs a single server-emitted status object over `/ws`.
- **Routing**: History API vs hash-fragment for region state (keeping the `/review`→`/#sessions` redirects working without reload).
- **Co-edit concurrency**: optimistic last-write-wins vs per-item version/lock, given the agent mutates the same YAML.
- **Slash-command demotion**: re-render `/status`,`/embryos` rich content as affordances vs button-per-command.
- **Per-field provenance schema**: parallel map on `ImagingSpec` vs sibling dataclass vs extending `PlanItem.references[]`.
- **Experiment tactical view contents** — deferred to Phase 5 design.
