# Session replay + agent postmortem — design

**Date:** 2026-07-13 · **Status:** approved design, pre-implementation
**Branch:** `feature/session-replay` · **Owner:** Keshu

## Purpose

Record every real interaction with the gently web UI — clicks, inputs, DOM changes —
so a session can be replayed post-hoc and, critically, **postmortemed by a Claude Code
agent**. Two consumers, in priority order:

1. **Agent postmortem** — Claude Code walks a session's logs, correlates UI actions
   with what the agent/microscope were doing at the same timestamps, and answers
   questions like "what did the operator do before the crash" or "where do users
   hesitate at the launch gate."
2. **Human replay** — a scrubber page that plays the session back visually.

This extends the ui_crawler philosophy (agent-readable artifacts: text first, PNGs on
demand) from synthetic story runs to real sessions. Playwright traces cover what *we*
drive; this covers what *humans* do.

## Decisions (settled)

- **Engine:** self-rolled [rrweb](https://github.com/rrweb-io/rrweb) — no PostHog /
  OpenReplay / SaaS. Vendored `rrweb.min.js`, no CDN (offline microscope PC).
- **Always on.** Recording starts with every page load. Kill switch: `GENTLY_REPLAY=0`.
- **Size is immaterial; quality of data is what matters.** No compression (saves
  main-thread CPU), no retention cap in v1. Raw JSONL on disk.
- **Zero performance compromise.** The app always wins over the recording (see
  *Performance posture*).
- **Detached from everything else.** Nothing in gently imports the recorder; removal
  = delete one template include + one router file.

## Architecture — three layers

All artifacts live in the file store, per session:

```
{GENTLY_STORAGE_PATH}/sessions/{session}/ui-replay/
  actions.jsonl       # layer 1 — semantic action log (agent-first)
  rrweb-{tab}.jsonl   # layer 2 — full rrweb event stream (human-first)
  meta.yaml           # tab ids, user agent, viewport, start/end times
```

### Layer 1 — semantic action log (primary for agents)

One JSON line per meaningful event:

```json
{"t": "2026-07-13T10:42:03.120Z", "tab": "a3f2", "action": "start-timelapse",
 "target": "button#start-tl", "label": "Start", "route": "/dashboard", "params": {}}
```

Sources: delegated capture-phase click listener keyed on `data-action` attributes
(fall back to tag+id+text for unannotated controls), route changes, form submits.
Cost is near zero. This is what Claude reads end-to-end; timestamps join against
agent traces, predictions, device-layer logs, and timelapse state already on disk.

### Layer 2 — rrweb event stream (full fidelity)

Standard rrweb recording: initial DOM snapshot + mutations + sampled mouse/scroll +
inputs. Config:

- `recordCanvas: false` — canvas is the one expensive capture, and gently's heavy
  pixels don't need it: microscope frames are `<img>` elements whose `src` URLs point
  at the file store, which is permanent. Replay re-resolves them by reference.
- Mouse/scroll sampling at rrweb defaults (throttled).
- Batched flush: buffer in memory, flush every ~5 s via `requestIdleCallback` +
  `fetch(keepalive)`; `sendBeacon` on `pagehide` so tab close loses ≤ one batch.
- One stream per browser tab (`rrweb-{tab}.jsonl`), tab id random per load.

### Layer 3 — the bridge: on-demand frames for agents

An agent cannot watch a replay; it reads text and inspects pixels on demand.
`tools/session_replay/render_frame.py --session S --t TIMESTAMP` replays the rrweb
stream headlessly (rrweb-player under Playwright — reusing the ui_crawler stack) and
screenshots the chosen instant to PNG. Claude's postmortem loop: read `actions.jsonl`
→ cross-reference backend logs → render only the moments that matter → look.

A `gently-session-postmortem` skill (sibling of `gently-debugging`) documents the
walk: where artifacts live, how to join timestamps, when to render frames.

## Server side

One router file, e.g. `gently/ui/web/routes/replay.py`:

- `POST /replay/ingest` — append a batch to the session's JSONL. Append-only writes,
  no parsing, no validation beyond size caps. Negligible cost.
- `GET /replay/{session}` — static player page (vendored `rrweb-player`) that loads
  the streams. Used post-hoc only.

No imports from gently core beyond the file-store path helper. The recorder JS is one
`<script>` include in the base template — the only touch point in existing code.

## Performance posture (the contract)

Recording cost lives on the browser main thread (MutationObserver + listener
serialization) — typically low single-digit % CPU with the config above. The explicit
rule for edge cases: **degrade the recording, never the app.**

- Bounded in-memory buffer; overflow drops oldest events (a gap in data, recorded as
  a `gap` event, beats a jank in the UI).
- Ingest endpoint down → recorder backs off and eventually self-disables for the
  session; the UI never notices.
- Entire recorder wrapped so any exception disables it silently.

**Verification before "done":** A/B the recorder (on/off) over existing ui_crawler
story flows with Chrome tracing; certify overhead (target: no story flow slows
measurably; long-task count unchanged). This gate is part of the implementation PR.

## Tauri desktop shell compatibility (PR #78)

Works by construction: the shell is a WebView rendering the same Python-served pages,
so the recorder travels with the UI unchanged. Two edge cases to build in:

- **Final flush at app quit.** Quitting the desktop app kills the backend (Job Object
  on Windows), which can race the `pagehide` beacon — the last ≤ one batch of a session
  may be lost, and crashy sessions end at their most interesting moment. Keep the flush
  interval short; the proper fix is PR #78's future graceful-shutdown handshake (drain
  before kill).
- **`requestIdleCallback` is absent in WebKit** (WebKitGTK = Linux desktop runs;
  WebView2/Chromium on Windows is fine). Recorder must feature-detect with a
  `setTimeout` fallback.
- The Tauri splash page is not recorded (it precedes navigation to the served UI).

## Out of scope (v1)

- Retention/pruning policy (revisit when disk pressure is real).
- Multi-user attribution beyond what auth already stamps.
- Canvas recording; masking/privacy filters (single-operator lab instrument).
- Live streaming of the recording (post-hoc only).

## Implementation phases

1. **Recorder + ingest** — vendored rrweb, `recorder.js`, `routes/replay.py`,
   file-store layout. Always-on behind `GENTLY_REPLAY`.
2. **Semantic action log** — `data-action` annotation pass over the main controls +
   the delegated listener.
3. **Replay player page** — `GET /replay/{session}` with rrweb-player.
4. **Agent tooling** — `render_frame.py` + `gently-session-postmortem` skill.
5. **Perf certification** — ui_crawler A/B + Chrome tracing report.
