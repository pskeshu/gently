<!-- Fills the docs/EVAL.md (TODO) referenced by gently/eval/__init__.py. -->

> **Status:** design + intended usage for the `gently/eval/` capture/replay substrate and the
> proposed offline replay harness for testing agentic orchestrator patterns. Grounded in the
> code as of the 0.22 epoch; the harness itself is a work-in-progress (see the incremental plan).

# Testing agentic orchestrator patterns offline (replay harness)

## Goal

We want to iterate on the agent's design — its realtime reasoning and the wake-router that
turns developmental events into autonomous turns — **without booking a live microscope run**.
Concretely: take a recorded session, simulate the microscope conditions from its on-disk
artifacts (captured events, recorded volumes, recorded perception traces), drive the *real*
`WakeRouter -> run_wake_turn -> Claude` loop offline, and observe/diff what the agent decides.
This lets us tune wake triggers, coalescing/throttling, prompt construction, and tool policy on
a laptop, replayed at a controllable clock (e.g. 10x), instead of waiting hours for embryos to
develop on a live rig.

## What's already in place (reuse)

A real, tested replay/eval substrate shipped in the 0.22 epoch (`gently/eval/`), plus production
capture wiring. None of this is hypothetical — it's on disk and runnable today:

- **`gently/eval/event_capture.py`** — `EventCapture` wildcard-subscribes the bus and appends every
  `Event.to_dict()` to `{session_dir}/events.jsonl`. Auto-wired into **every** live session by
  `gently/app/agent.py` `_init_event_capture()` (line ~506, called at agent init). Skips only
  `_NO_HISTORY_TYPES` (`DEVICE_STATE_UPDATE`, `BOTTOM_CAMERA_FRAME`, `LOG_RECORD`); `DETECTOR_EVALUATED`
  and lifecycle events are **not** skipped.
- **`gently/eval/event_replay.py`** — `EventReplay(path).replay(target_bus, real_time=, time_scale=, on_event=)`
  republishes each event via `target.publish_event(ev)`, **preserving the original `Event.timestamp`**
  (not re-stamping `now()`). `real_time=True` sleeps `(ev.timestamp - prev)/time_scale` between events,
  so cadence is reproducible. `event_types()` gives a pre-flight histogram.
- **`gently/eval/shadow.py` + `candidates.py`** — `ShadowRunner` + `OrchestratorCandidate` host
  sandboxed rule-based candidates (e.g. `ReactiveCandidate`) that may *only* write a `DecisionLog`.
- **`gently/eval/decision_log.py`** — `Decision`/`DecisionLog` + `prompt_hash()` (sha256[:16] over
  system prompt + messages) for apples-to-apples A/B diffing.
- **`scripts/replay_session.py`** — working CLI: resolves a session by id-prefix via
  `FileStore.list_sessions`, prints `--histogram`, or replays `events.jsonl` into a **fresh** `EventBus()`
  with an optional `NoOpCandidate`.
- **Recorded perception inputs/outputs** persist via `FileStore` (`gently/core/file_store.py`):
  `embryos/{id}/volumes/t{NNNN}.tif` + `.meta.yaml`, `projections/t{NNNN}.jpg`,
  `predictions.jsonl`, and `traces/t{NNNN}.json` (verbatim `predicted_stage`/`reasoning`/`raw_response`/`stability`).
  Verified on disk: session `68e7dc33` has 9 embryos, 56 predictions on embryo_001, volume
  `t0001.tif` shape `[50,512,2048]` uint16.
- **`timeline.jsonl`** (durable, predates eval) carries 256 `detection/evaluated` records on `68e7dc33`
  with exactly the fields `WakeRouter._is_wake_worthy` reads (`embryo_id`, `timepoint`,
  `detector_name`, `stage`, `reasoning`) — a fallback event source for pre-eval-scaffold sessions.

### The one central wiring gap

The real agent subscribes its `WakeRouter` to the **global singleton** bus
(`gently/app/agent.py:126` `self._event_bus = get_event_bus()`; WakeRouter built with that same bus).
But `scripts/replay_session.py:124` replays into a **fresh** `EventBus()` that the agent never sees.
**So today's replay reaches shadow candidates but never the real WakeRouter/agent.** Bridging this —
either `set_event_bus(replay_bus)` before constructing the agent, or replaying into `get_event_bus()`
directly — is the core seam to build.

## Approaches, compared

### (A) Event-stream replay into the agent's bus  — *recommended first*
Publish recorded `DETECTOR_EVALUATED` + critical events (`HATCHING_DETECTED`, `EMBRYO_TERMINATED`,
`ERROR_OCCURRED`, …) onto the bus the agent's `WakeRouter` is subscribed to, on a controllable clock.

- **Reuses:** `EventReplay`, `EventCapture` output, the entire real `WakeRouter` (`_is_wake_worthy`
  filter at wake_router.py:129, coalesce `COALESCE_WINDOW=20s`, throttle `MIN_WAKE_INTERVAL=120s`,
  `_flush -> agent.run_wake_turn`).
- **Fidelity:** Exercises the *real* wake path end-to-end: filtering, transition gate, coalescing,
  throttling, prompt build, and a real Claude turn (`run_wake_turn -> handle_message_stream`, gated on
  `agent.mode=='run'`). Highest leverage for the least new code.
- **Effort:** Medium. Needs (1) the bus bridge above; (2) a running asyncio loop so the async dispatch
  + `loop.call_later` coalesce timers fire (`EventReplay.replay` is a blocking `time.sleep` loop — run it
  in a thread or port it to `await asyncio.sleep`, and call `bus.set_event_loop(loop)`); (3) a stub
  client so any tools the woken agent calls don't hit hardware (autonomous mode already refuses
  irreversible tools via `_autonomous_active`).
- **Can't catch:** Anything depending on *fresh* perception of new pixels — the wake note embeds
  `build_perception_snapshot(agent.perceiver, ...)`, which reads **live** Perceiver state, so this
  approach needs (B) to make that snapshot reflect the replayed timepoint rather than empty state.
- **Blocker today:** No recorded session yet contains `DETECTOR_EVALUATED` (verified: all 20 captured
  `events.jsonl` hold only `STATUS_CHANGED`/`EMBRYO_DETECTED`/`EMBRYOS_UPDATE`). Either capture one fresh
  perception-driven session, or synthesize `DETECTOR_EVALUATED` events from `traces/`+`timeline.jsonl`.

### (B) Perceiver stub — feed recorded traces
Replace `agent.perceiver`/`orchestrator.perceiver` with a duck-typed stub whose `__call__(...)` returns
`.stage`/`.reasoning` from `traces/t{NNNN}.json`, and whose `get_session(embryo_id)` returns an object
with `.stability`/`.summary()` matching what `build_perception_snapshot` reads
(`current_stage`/`stability`/`temporal`/`stage_sequence`).

- **Reuses:** All downstream code in `_run_perception` (DETECTOR_EVALUATED emit, trace write,
  `store_prediction`, `_check_interval_rules`) is pure local code; the Perceiver is the *only* external
  VLM dependency. `perceiver` is already an optional ctor arg (timelapse.py:71).
- **Fidelity:** Reproduces recorded perception verbatim — no VLM spend, deterministic. Makes (A)'s wake
  snapshot reflect the replayed timepoint.
- **Effort:** Low-medium (one stub class).
- **Can't catch:** Perception on *new* conditions — it only echoes what the recorded run already saw.
  Also the stub interface was inferred from call sites (`templates.py` `build_perception_snapshot`,
  `timelapse.py` `_run_perception`), not from `gently_perception` source — **verify against the
  installed package** before relying on it.

### (C) Full offline re-feed through the timelapse loop
Inject a fake `microscope_client` whose `acquire_volume(...)` returns
`{'success': True, 'volume': <ndarray from volumes/t{NNNN}.tif>}` keyed by `(embryo_id, timepoint)`,
plus `move_to_position`, etc. `_has_microscope()` (`return self.client is not None`) then gates the
orchestrator *on*, driving the entire per-timepoint loop (acquire -> callback -> `_run_perception`).

- **Reuses:** The whole `TimelapseOrchestrator`; `client` is a single ctor arg accessed only via named
  async methods.
- **Fidelity:** Highest — exercises scheduling, acquisition callbacks, perception, event emission, and
  wakes as one system.
- **Effort:** High. The orchestrator schedules entirely off `datetime.now()`/`asyncio.sleep`
  (`_pick_next_due`, `_reschedule`, the acquire loop) and ignores `Event.timestamp` — a faithful
  time-scaled run needs an **injectable clock** threaded through both the orchestrator and the
  WakeRouter's wall-clock timers. There is also no helper to join recorded TIFFs to the event stream.
- **Can't catch:** Same perception-novelty limit as (B); plus device-state/camera-frame telemetry is
  absent from `events.jsonl` and must be sourced from disk or synthesized.

### (D) Shadow mode — score candidates / replay captured turns
Keep the existing `ShadowRunner` path: replay captured events into a bus with rule-based candidates
attached, diff `decisions.jsonl` (production) vs `replay-decisions-*.jsonl` (candidate) via `prompt_hash`.

- **Reuses:** Fully built already (`scripts/replay_session.py --candidate`).
- **Fidelity:** Tests *alternative* (non-LLM) orchestrator architectures, not the production Claude agent.
- **Effort:** None (exists).
- **Can't catch:** The real agent's reasoning. Also: production only writes `Decision`s for **user turns**
  (`conversation.py:343` hardcodes `trigger=DecisionTrigger.USER_MESSAGE`); wake turns via
  `call_claude_stream` aren't logged as decisions, so there's currently no production wake-decision row to
  diff against.

## Honest fidelity limits

- **Recorded perception ≠ new perception.** Approaches B/C echo `traces/`; they cannot evaluate the
  Perceiver on conditions the original run didn't encounter. Genuinely testing perception requires a live
  (or freshly captured) run.
- **LLM nondeterminism.** `run_wake_turn` makes a real Claude call; the same replayed input can yield
  different tool calls run-to-run. `prompt_hash` isolates *input* identity but not *output* determinism —
  diffs are about distributions/policy, not exact equality.
- **Clock vs coalesce/throttle.** `WakeRouter` uses real wall-clock `loop.call_later(COALESCE_WINDOW=20s)`
  and `loop.time()`-based `MIN_WAKE_INTERVAL=120s` — these are **not** scaled by `time_scale`. A fast
  replay collapses bursts into one wake; a high `time_scale` shrinks inter-event sleeps below the fixed
  20s window, again collapsing wakes. These tunables (currently module-level constants in
  `wake_router.py:33-35`) must be parameterized/injectable for faithful timed replay.
- **Wall-clock reads break replay.** `TimelapseOrchestrator` (`timelapse.py`) drives scheduling off
  `datetime.now()`/`asyncio.sleep` and never consults `Event.timestamp`; perception stamps
  `timestamp=datetime.now()`. Any *new* events the woken agent emits use `publish()` (fresh `now()`),
  intermixing replayed-historical and live-now timestamps on the same bus — a consistency hazard for
  downstream diffing.
- **Telemetry gaps.** `EventCapture` skips `DEVICE_STATE_UPDATE`/`BOTTOM_CAMERA_FRAME`/`LOG_RECORD`, so a
  replay can't reconstruct live device readouts or frames from `events.jsonl` (re-capture with
  `EventCapture(path, skip=set())` or synthesize).
- **Data availability (verified).** No single recorded session yet combines a full timelapse with
  non-trivial capture: `68e7dc33` has 9 embryos + volumes/traces but **no** `events.jsonl`; the newest
  sessions have `events.jsonl` but 0 embryos and empty `decisions.jsonl`. All 20 captured `events.jsonl`
  contain only setup-phase events — **zero** `DETECTOR_EVALUATED`.

## Concrete incremental plan

**Step 0 — Generate one good input stream (unblocks everything).** Either (a) run a single fresh
perception-driven session (live or with a stub client) after the eval-scaffold commit so
`events.jsonl` + non-empty `decisions.jsonl` coexist with volumes/traces; or (b) write a tiny
`synthesize_events.py` that emits `DETECTOR_EVALUATED` events from `68e7dc33`'s `traces/`+`timeline.jsonl`
into a synthetic `events.jsonl`. Validate with `python scripts/replay_session.py <id> --histogram`.

**Step 1 (smallest useful) — Bus bridge + offline driver skeleton.** New script
`scripts/replay_into_agent.py`: construct a `GentlyAgent` with a stub microscope client, call
`set_event_bus(replay_bus)` (or replay into `get_event_bus()`), set `agent.mode='run'` and
`wake_router.set_mode('ask')`, run an asyncio loop, `bus.set_event_loop(loop)`, and run
`EventReplay(...).replay(bus, real_time=True, time_scale=N)` in a thread. First milestone: a recorded
`DETECTOR_EVALUATED` actually fires `_on_event -> _flush -> run_wake_turn`. Reuses `EventReplay`,
`WakeRouter`, `run_wake_turn` unchanged.

**Step 2 — Perceiver stub (B).** Add a `RecordedPerceiver` reading `traces/t{NNNN}.json`, injected via
`agent.perceiver`. Verify its `summary()`/result shape against the installed `gently_perception`. Now the
wake prompt's `build_perception_snapshot` reflects the replayed timepoint.

**Step 3 — Injectable clock + parameterized tunables.** Thread a clock/`now()` provider through
`TimelapseOrchestrator` and make `COALESCE_WINDOW`/`MIN_WAKE_INTERVAL` injectable on `WakeRouter` so a
time-scaled replay reproduces the live wake set. Also scale or virtualize the loop timers.

**Step 4 — Capture wake decisions + fix trigger labels.** Add `DecisionLog` capture to the
`call_claude_stream` (wake) path and emit `DecisionTrigger.EVENT` for wake turns (today `conversation.py`
only writes `USER_MESSAGE` decisions from `call_claude`). This makes replayed autonomous turns diffable.

**Step 5 — Optional full re-feed (C).** Add a `RecordedMicroscopeClient` whose `acquire_volume` loads
`volumes/t{NNNN}.tif`, gating the orchestrator on for end-to-end loop testing.

**Step 6 — Write `docs/EVAL.md`** (referenced as TODO in `gently/eval/__init__.py`) documenting the
replay workflow and fidelity tiers.
