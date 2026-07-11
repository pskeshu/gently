# UX v2 — interaction-flow / IA audit

**Branch:** `feature/ux-v2` (now includes the 3D optical-space view).
**Scope:** the *flow* of the agent-first UI — clicks, how each step renders, how the
workspace is unveiled, moving back/forth between views, resume — **not** the visual look
(the look is fine). Plus where the 3D optical-space view belongs in the new workspace IA.

**Method:** live click-audit driven through a real browser as a *dev biologist* would use
it, with the agent **live** (Opus 4.8, `--offline` hardware, `GENTLY_NO_AUTH=1` single
controller), cross-checked against the code. Screenshots from the run are in `screenshots/audit-*.png`.

> Correction to an earlier automated pass: the plan-wizard helpers
> (`buildAskCard`/`answerChoice`/`togglePanel`) are **not** missing — `agent-chat.js`
> exports them and the module loads; the plan wizard works. The real issues are below.

---

## What works (keep it)

- **The forward path is good.** Entry → one calm choice (Plan / Quick look / "just tell me")
  → overlay dismisses to reveal the workspace → grouped rail (NOW / LIBRARY / SYSTEM) drives
  everything through one chokepoint (`app.js switchTab`). The welcome→workspace unveil is genuinely nice.
- **The agent-driven plan wizard is strong.** Live, it asked a well-framed scientific question
  ("What's the core scientific question this run should capture?") with real C. elegans options,
  ran a `query_lab_history` tool with visible provenance, and **assembled THE PLAN panel as each
  answer landed** (strain → wavelengths, etc.). The "plan builds as you answer" feel is excellent.
- **The dual-render** (ask shows in the plan stage *and* the chat transcript) is implemented.

---

## Findings (prioritized)

| # | Pri | Symptom (felt) | Root cause / evidence | Fix |
|---|-----|----------------|------------------------|-----|
| 1 | **P0** | First plan step sat on "working through the next step…" for **~90s** with a static spinner — feels hung. | The wait is the model *thinking*. The streaming call requests **no thinking config** and the stream loop reads only text deltas. `conversation.py:272-275` (only `output_config.effort`), `conversation.py:654-657` (only `event.delta.text`). | Set `thinking={"type":"adaptive","display":"summarized"}` on the stream (`conversation.py:552`); handle `thinking_delta` in the loop (`:654`) and emit as a `thinking` activity; render it live + add an elapsed timer. See §1. |
| 2 | **P1** | Agent's first line renders as **"'d love to help…"** — leading "I" dropped. | Plan-feed streaming path drops the first character of the turn's first text block; the chat transcript renders it correctly (`12_41` vs `12_3` in the run). Plan feed: `landing.js applyActivity` `'text'` case (`:269`). | Most likely the first `AGENT_ACTIVITY`/`text` delta is missed by `landing.js`'s listener (subscribed after the first delta) or coalesced wrong. Confirm with a 1-line repro; the transcript path is the reference. |
| 3 | **P1** | Clicked the primary "Plan an experiment" → plan stage spun forever; the *real* blocker ("Viewing only — control is held by another client / sign in to control") was **hidden in the chat panel**. | Control/auth state isn't surfaced on the landing/plan surface — only in the chat dock. A viewer can enter the plan flow and dead-end. | Surface control/sign-in state on the landing **before** the primary CTA; gate or relabel "Plan an experiment" when `!hasControl`; show the wall on the plan stage, not just chat. |
| 4 | **P1** | (Structural) The same ask renders in **two** stage mounts plus the transcript. | `#v2-plan-ask` **and** `#ask-stage` both render the ask (the overlay covers the workspace copy, so only cosmetic/perf today). Two live regions seen in the run (`12_10` + `12_24`). | One stage mount at a time — suppress `#ask-stage` while the landing overlay owns the ask. |
| 5 | **P1** | Cross-surface clear can desync. | `ASK_CLEARED` is **listened for but emitted nowhere** (`landing.js:624`, `ask-stage.js:43` listen; no emit in repo). Answering works locally because `renderAsk.onPick` clears directly, but stage↔transcript sync relies on the missing signal. | Emit `ASK_CLEARED` the instant a `choice_response` is sent (per the migration plan's Phase-1 blocker), plus on cancel/control-loss/socket-close. |
| 6 | **P1** | **No way back.** Once the landing dismisses, there's no path back to welcome / "start a new plan" from the workspace — must reload. | `dismiss()` is one-way (`landing.js:42-54`); `V2Landing.show()` exists but is never called from the workspace. | Add a "New plan" / "Talk to Gently" entry in the rail or header that re-summons the welcome/plan surface. |
| 7 | **P2** | Browser **Back / refresh don't mean anything**; refresh mid-plan loses state and may re-show the landing. | Entry hash is consumed (`app.js` → `replaceState('/')`, ~`:650-662`); no deliberate URL/state sync; in-memory plan state (`planKickedOff`, feed pages) resets on reload. | Real routing: sync screen/tab to URL/History so Back/forward/refresh resolve; persist or re-hydrate plan progress. |
| 8 | **P2** | **Resume = full page reload** — jarring, re-shows landing, drops chat position. | `session_changed` → `window.location.href='/'` (`websocket.js:147`; `review.js resumeSession ~:101-116`). Flagged in the migration plan. | In-place re-hydration on `session_changed` instead of a hard reload. |
| 9 | **P1 (IA)** | The **3D optical-space view is buried**: SYSTEM → Devices → (Map / Details / **3D**) — a sub-sub-toggle. | It was integrated into the *legacy* Devices tab structure; the ux-v2 grouped rail doesn't surface it. | Promote "the scope in space" to a first-class run-time surface (NOW tier), reconciled with the grouped rail. See §2. |
| 10 | **P2** | Offline / agent-silent dead-ends the wizard at "working…". | `startPlan` campaign fetch falls through silently if offline (`landing.js ~:502-508`); no error path. | Timeout + inline error/retry on the plan stage. |

---

## §1 — Make the loading state legible (P0, the one the user wants first)

The 90s "working…" is the agent reasoning. The Claude streaming API exposes this on three
channels; gently currently surfaces none of the reasoning:

- **Thinking** — `content_block_delta` → `thinking_delta`. **Opus 4.8 defaults to
  `display:"omitted"` (empty thinking text)**, and gently doesn't set the thinking config at
  all on the stream, so there's nothing to show. Unlock: `thinking={"type":"adaptive","display":"summarized"}`.
- **Tool activity** — `input_json_delta` + tool start/stop. **Already flowing** — the plan feed
  renders tool cards (saw the `query_lab_history` card with input/result).
- **Text** — `text_delta`. Already flowing (this is the path with the bug #2 truncation).

**Backend (`gently/harness/conversation.py`):**
1. `:552` `self.claude.messages.stream(...)` — add `thinking={"type":"adaptive","display":"summarized"}`
   (keep `output_config.effort`).
2. `:654` event loop — currently only `if hasattr(event.delta, "text")`. Add a branch for
   `event.delta.type == "thinking_delta"` → `yield {"type":"thinking","text": event.delta.thinking}`.

**Frontend (`gently/ui/web/static/js/landing.js`):** `applyActivity` already has a `thinking`
case (`:266`) that only sets a static label — render the streamed thinking text instead, and add
an elapsed timer to `#v2-plan-thinking` so a long think reads as progress, not a hang.

Net: the reasoning summary + current tool + a timer fill the wait. Only the backend `display`
flag is a new capability; the rest is surfacing data gently already receives.

---

## §2 — Workspace organization & where the 3D view belongs (P1, IA)

The ux-v2 workspace is organized differently from the old flat tab bar: a **grouped rail**
(NOW: Home/Experiment/Embryos · LIBRARY: Plans/Sessions · SYSTEM: Devices/Calibration/Logs),
a **session-context strip**, and the **AGENT'S VIEW** surface. The 3D optical-space view,
however, lives in the *legacy* Devices structure (`devices.js switchView`, VIEWS =
`['map','details','optical3d']`; `index.html` devices-content Map/Details/3D switcher).

During an actual run, "where the scope is in space" + the live experiment + the agent's view are
**NOW-tier** concerns, not a System utility three clicks deep. Proposal (to design next):
- Promote the 3D optical-space + live experiment to a first-class run-time surface in the rail
  (or make it the default workspace view while a run is active).
- Keep the Devices Map/Details as the System-tier hardware utility; the 3D "scope in space"
  graduates out of that toggle.

---

## Recommended sequencing

1. **P0 loading state** (§1) — highest felt value, mostly surfacing existing data.
2. **P1 quick correctness**: #2 truncation, #3 control-wall surfacing, #4 single ask mount, #5 `ASK_CLEARED` emit.
3. **P1 reachability**: #6 "new plan"/back entry; then #9 the workspace-IA / 3D-placement redesign (its own design pass).
4. **P2 navigation**: #7 real routing, #8 resume re-hydration, #10 offline error path.

---

## Notes / housekeeping

- Findings 1–5, 10 verified live with the agent on; 6–9 verified from code + the live rail.
- `screenshots/audit-*.png` (live run) and `screenshots/uxv2-*.png` are local evidence (untracked).
- The earlier visual-design exploration (`docs/superpowers/mockups/`, `screenshots/dir-*.png`) is
  superseded — the look is staying as-is — and can be deleted.
