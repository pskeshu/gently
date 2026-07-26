# Hierarchical Sub-Contexts — Architectural Design

| | |
|---|---|
| **Status** | PROPOSAL — Draft v0.1 |
| **Author** | P.S. Kesavan (with Claude) |
| **Created** | 2026-05-01 |
| **Last Updated** | 2026-05-01 |
| **Tracking issue** | _to be filed_ |
| **Supersedes** | — |
| **Related** | [`PLAN_MODE_DESIGN.md`](./PLAN_MODE_DESIGN.md), [`CONTEXT_REFACTORING_GUIDE.md`](./CONTEXT_REFACTORING_GUIDE.md), [`PERCEPTION_V2_IMPROVEMENT_STRATEGY.md`](./PERCEPTION_V2_IMPROVEMENT_STRATEGY.md) |

---

## TL;DR

Today the copilot has **one flat conversation** in `ConversationManager.conversation_history`. Every operation — focus sweep, timelapse status check, per-embryo perception reasoning — accumulates into the same message stream. The orchestrator's working memory is polluted by detail it doesn't need to hold.

This document proposes **hierarchical sub-contexts**: the orchestrator can spawn nested LLM conversations ("scopes"), each with its own working memory, restricted toolset, and chosen model. Children return condensed summaries to parents. Children can themselves spawn grandchildren. Communication between scopes is via four primitives: `inject_message`, `yield_checkpoint`, `escalate`, `query_parent`/`read_parent_state`. Sibling coordination is indirect, through the existing context store and event bus.

The proposed design is built around **one new primitive** — a `delegate` tool — and **one new runtime concept** — a per-scope `asyncio.Queue` inbox with `wakeup` event. Everything else (LLM call, tool dispatch, event bus, hardware control, perception VLM) is reused. Estimated new code: ~700 lines across 5–6 new files plus targeted edits to 6 existing files.

---

## Status & How This Document Evolves

This is a **living design document**. It is not a finalized spec; it is the canonical record of the design as it gets refined.

**Status lifecycle:**

```
  PROPOSAL  ─▶  ACCEPTED  ─▶  IN PROGRESS  ─▶  IMPLEMENTED
                  │
                  └─▶  SUPERSEDED (with link to replacement)
```

- **PROPOSAL** — open for discussion, not yet committed to.
- **ACCEPTED** — design locked, implementation can begin. Major changes require a new revision.
- **IN PROGRESS** — implementation underway; section "Implementation Plan" tracks phase status.
- **IMPLEMENTED** — shipped; this doc becomes the architecture reference for the implemented system.
- **SUPERSEDED** — replaced by a follow-up design.

**How to refine this document:**

1. Edit in place for clarifications, typo fixes, additional diagrams, expanded examples.
2. For substantive design changes, **bump the version**, update `Last Updated`, and add an entry to the [Decision Log](#decision-log) at the bottom describing what changed and why.
3. For decisions that close one of the [Open Decisions](#open-decisions), move the item from "Open" to "Decided" with the resolution and date.
4. When status changes, update the header table.

**Source of truth conventions:**

- This document is the **architectural** source of truth. Code is the **implementation** source of truth. When they diverge, this document is updated to match (or the code is reverted).
- Concrete trace examples (e.g. the 4-hour timelapse walkthrough) are illustrative, not normative — they exist to ground the abstractions, not to constrain implementation.

---

## Problem Statement

The current copilot architecture has two costs:

### 1. Cognitive interference

`ConversationManager.conversation_history` is a single flat list. Every tool call, every observation, every reasoning turn lives in the same stream. Reasoning about "is embryo 3 hatching?" sits next to reasoning about "should I move the stage?" The orchestrator's working memory, and the model's attention budget, is polluted by detail it doesn't need to hold.

### 2. No reusable working frame

A focus operation is conceptually a tight inner loop ("sweep, score, refine, return best Z"). In the current architecture it becomes a sequence of orchestrator turns interleaved with everything else. There is no way to say "load this sub-task, work in that mental frame, write the result back, discard the working memory." This is exactly the pattern skilled experimentalists use, and the copilot has no analogue.

### What the user proposes

> "What if we had different sub-conversations? When we do focusing, the orchestrator can spin up a sub-context, where all the focus operations contexts get stored, and all the operational reasonings are shared within — and finally focus is complete, and the summary is returned to the main orchestrator conversation for note-keeping. And similarly with the timelapse — a running context, which can be pulled up by the orchestrator when needed. And within the timelapse, a perception context, for all embryos, and for individual embryos."

This is the same pattern as Claude Code's `Agent` tool, applied internally to the gently copilot.

---

## Vision

A `ContextScope` is the unit of cognitive work. The orchestrator (the existing copilot) is the **root scope**. When it needs to delegate a bounded operation, it spawns a child scope with:

- its own `conversation_history` (separate message list)
- its own restricted toolset (filtered from the existing `tool_registry`)
- its own model choice (a child can run sonnet while the parent runs opus)
- a brief from the parent (instructions + a slice of relevant parent state)
- a return contract (typed `key_findings` + a natural-language `summary`)

Children can themselves spawn grandchildren. **Summaries always roll one level up** — grandparents see only what the parent chose to forward.

Three concrete sub-context types in v1, all built on one generic primitive:

| Scope type | Purpose | Lifetime | Spawned by |
|---|---|---|---|
| `focus` | piezo sweep, find best Z, return drift estimate | transient (~30s) | orchestrator OR perception scope |
| `timelapse` | adaptive multi-embryo acquisition over hours | long-lived (background) | orchestrator |
| `embryo_perception` | per-embryo VLM reasoning + stage tracking | long-lived (matches timelapse) | timelapse scope |

A fourth, `ad_hoc`, exists as the catch-all for the orchestrator to delegate any task that doesn't fit a preset.

---

## Core Abstraction: `ContextScope`

```python
@dataclass
class ContextScope:
    scope_id: str
    parent_id: Optional[str]
    scope_type: Literal["root", "focus", "timelapse", "embryo_perception", "ad_hoc"]
    purpose: str  # one-line task description
    instructions: str  # full brief, like a prompt to a colleague
    conversation_history: List[Dict]
    allowed_tools: Set[str]
    model: str  # "opus" | "sonnet" | "haiku"
    status: Literal["active", "completed", "yielded", "cancelled", "failed"]
    summary: Optional[str]  # produced by Summarizer at end
    key_findings: Dict[str, Any]  # structured return value
    children: List[str]  # child scope_ids
    created_at: datetime
    completed_at: Optional[datetime]
    persist: bool  # write to SQLite if True

    # Runtime
    inbox: asyncio.Queue  # incoming messages from other scopes
    wakeup: asyncio.Event  # signaled when there is work
    cancel_token: asyncio.Event
    cancel_reason: Optional[str]
    pending_queries: Dict[str, asyncio.Future]
    peer_events: List[str]  # event types this scope subscribes to
    parent_snapshot_cache: Dict  # parent's last known summary + findings
```

A `ScopeManager` (lives on `MicroscopyCopilot`) owns the scope tree, dispatches the inner agent loop, and exposes the orchestrator-facing tools.

---

## Core Primitive: the `delegate` tool

The orchestrator gains one new meta-tool. Everything else is built on it.

```
delegate(
    purpose: str,             # "Find best Z for embryo 3"
    instructions: str,        # full task brief
    toolset: str | List[str], # preset name ("focus_tools") or explicit tool list
    model: str = "sonnet",    # child can run cheaper than parent
    mode: "blocking" | "background",
    persist: bool = False,    # write child transcript to SQLite
) -> {
    scope_id: str,
    summary: str,             # Haiku/Sonnet summarization of child transcript
    key_findings: dict,       # structured fields (best_z, anomalies, ...)
    status: "completed" | "running" | "failed",
}
```

When called, the orchestrator:

1. Creates a `ContextScope` with empty `conversation_history`, child system prompt (parent's relevant context + `instructions`), filtered toolset, and chosen model.
2. Runs the **inner agent loop** in that scope until the child emits `complete` (or `yield_checkpoint`, in long-lived scopes).
3. Passes the child's full transcript to a **universal Summarizer** (Haiku by default) that produces the summary returned to the parent.
4. Only the summary goes into the parent's `conversation_history`. The full child transcript stays in the scope tree (queryable on demand via `inspect_subcontext(scope_id)`).

In `blocking` mode the parent's turn waits for the child. In `background` mode the parent gets a `scope_id` immediately and continues; companion tools (`check_subcontext`, `fetch_subcontext_summary`, `list_subcontexts`) let it pull updates later.

**Children can call `delegate` themselves** — that's how a `timelapse` scope spawns 8 parallel `embryo_perception` children, and an `embryo_perception` scope can spawn a transient `focus` child for drift recovery.

### Three v1 preset wrappers

These are not separate codepaths — they are thin functions (~30 lines each) that call `delegate` with sensible defaults:

| Preset | Toolset | Model | Mode | Persist |
|---|---|---|---|---|
| `delegate_focus(embryo_id)` | focus_tools + stage_tools | sonnet | blocking | no |
| `delegate_timelapse(plan)` | timelapse_tools + acquisition_tools | sonnet | background | yes |
| `delegate_embryo_perception(embryo_id)` | perception/analysis tools | sonnet | background | yes |

A `delegate_timelapse` scope, on each acquisition tick, spawns N parallel `delegate_embryo_perception` children — exactly the hierarchy in the user's original vision.

---

## Runtime Model

> **The honest answer to "does communication trigger an API call":** yes — every meaningful message a scope receives causes one Claude turn. But scopes sleep on an `asyncio.Event` between wakes, so idle scopes consume zero tokens and zero CPU.

The pattern is **event-driven sleep, one-LLM-turn-per-wake**.

### The inner agent loop

```python
async def run_scope(scope: ContextScope) -> SummaryResult:
    while scope.status == "active":
        # SLEEP until something gives us work
        await scope.wakeup.wait()
        scope.wakeup.clear()

        # DRAIN inbox (may be empty if it was a spurious wake)
        items = []
        while not scope.inbox.empty():
            items.append(scope.inbox.get_nowait())
        if not items and not scope.cancel_token.is_set():
            continue  # nothing to do, back to sleep, no API call

        # CHECK CANCELLATION — give child one final turn to summarize
        if scope.cancel_token.is_set() and scope.status != "cancelling":
            scope.status = "cancelling"
            scope.conversation_history.append(
                {
                    "role": "user",
                    "content": f"[CANCELLATION] {scope.cancel_reason}. "
                    f"Emit a final summary using `complete` and stop.",
                }
            )

        # APPEND injected messages to history
        scope.conversation_history.append(
            {
                "role": "user",
                "content": format_inbox_items(items),
            }
        )

        # ONE LLM TURN
        response = await call_claude(
            model=scope.model,
            system=scope.system_prompt,
            messages=scope.conversation_history,
            tools=tool_registry.filter(scope.allowed_tools),
        )
        scope.conversation_history.append({"role": "assistant", "content": response})

        # HANDLE TOOL CALLS
        if response.stop_reason == "tool_use":
            tool_results = []
            for tool_call in response.tool_uses:
                match tool_call.name:
                    case "delegate":
                        result = await spawn_and_run(scope, **tool_call.input)
                    case "yield_checkpoint":
                        result = await emit_checkpoint(scope, **tool_call.input)
                    case "escalate":
                        result = await escalate_to_root(scope, **tool_call.input)
                    case "query_parent":
                        result = await ask_parent(scope, **tool_call.input, timeout=30)
                    case "read_parent_state":
                        result = scope.parent_snapshot_cache
                    case "respond_to_query":
                        result = await deliver_answer(**tool_call.input)
                    case "complete":
                        scope.status = "completed"
                        scope.key_findings = tool_call.input
                    case _:
                        result = await execute_tool(tool_call, scope)
                tool_results.append(result)
            scope.conversation_history.append({"role": "user", "content": tool_results})

        if response.stop_reason == "end_turn":
            scope.status = "completed"
            break

    return await summarizer.summarize(scope)
```

Three things to notice:

1. **It's the same loop the orchestrator already runs in `ConversationManager.call_claude`** — just generalized to take a scope as input. We are not inventing a new agent runtime; we are extracting the existing one and parameterizing it.
2. **`delegate` recurses through the same loop** — there is no special "depth 2 path."
3. **All comms primitives are just tools.** No special pipeline.

### Wake-up sources by scope type

| Scope | Sleeps on | Wakes when | Cadence |
|---|---|---|---|
| ROOT | `wakeup` | user CLI input / child checkpoint / escalation | sporadic |
| timelapse (S1) | `wakeup` | `acquisition_tick` event / child checkpoint / parent injection | every 30–300s plus reactive |
| embryo_perception (S2..S9) | `wakeup` | `image_ready` event for its embryo / parent injection / peer event | matches timelapse cadence |
| focus (S10, transient) | runs straight through | n/a — blocking, never sleeps | one-shot, ~30s |

The `acquisition_tick` and `image_ready` events already exist (or are trivially derivable) inside `TimelapseOrchestrator` and the existing `event_bus` — we just attach scope `wakeup` events to them.

### Concurrency

Parallel by default. The inner agent loop is async; `delegate(mode="background")` returns immediately and the child runs as `asyncio.create_task`. Hardware is serialized at the device-layer level (already true today in `device_layer.py` via the rpyc connection). LLM calls are not — Anthropic's per-account rate limit becomes the constraint. For 3–8 parallel embryos this is fine.

### Persistence (opt-in per spawn)

Default off. Pass `persist=True` to `delegate` when the scope is long-lived (timelapse) or recovery matters; otherwise stay in-memory. Schema additions:

- `context_scopes(scope_id PK, parent_id, scope_type, purpose, status, summary, key_findings_json, model, created_at, completed_at)`
- `context_scope_messages(scope_id FK, idx, role, content_json)`

When the copilot restarts, persisted active scopes are inspectable via `/scopes`. **Resumption** (re-attaching a running scope) is out of v1 scope.

---

## Communication Model

Three communication axes, four primitives:

| Axis | Direction | Primitive | Mechanism |
|---|---|---|---|
| Vertical (down) | parent → child | `inject_message`, `cancel_subcontext` | Async queue on the child scope |
| Vertical (up, fast) | child → parent (or root) | `yield_checkpoint`, `escalate` | Tool the child calls; appends to parent inbox |
| Vertical (up, ask) | child ↔ parent | `read_parent_state`, `query_parent` | Snapshot read, or synchronous LLM round-trip |
| Horizontal | sibling ↔ sibling | shared context_store + scope events | Indirect; no direct messaging |

The deliberate choice: **direct messaging is strictly vertical, hierarchical**. Siblings coordinate through the shared blackboard (context_store + event_bus), never by reaching across. This keeps the tree understandable and stops sibling chatter from creating implicit cycles.

### `inject_message(scope_id, content)` — parent → child

Used when the parent has new information for a running child. Appends a `[user]` message to the child's history; the child sees it at the start of its next loop iteration.

```
ROOT: user mentions "by the way, embryo_2 looks weird in the live preview"
ROOT: tool_use: inject_message(scope_id="S3", content="User noted unusual
       morphology on embryo_2 in live preview. Investigate next timepoint.")
S3 (next loop tick): conversation_history sees a new [user] message and reacts
```

`cancel_subcontext(scope_id, reason)` is the same mechanism with a special tag — sets `cancel_token`, child gets one final turn to summarize, then exits.

### `yield_checkpoint(summary, findings)` and `escalate(severity, message)` — child → parent

`yield_checkpoint` is the **soft** path: child wants the parent to know something but keep running itself.

`escalate` is the **fast** path: bypasses intermediate parents and lands directly in root's inbox. Used for things like "the sample is dead, abort the timelapse." Severity (`info` | `warn` | `critical`) determines whether root sees it as a normal message or a priority interrupt.

Implementation: `escalate` walks up `scope.parent_id` chain to find root, calls `inject_message` on root with a tagged prefix.

### `read_parent_state(field)` and `query_parent(question)` — child → parent

Two flavors of "ask upward":

- **`read_parent_state`** is cheap, synchronous, no LLM: returns the parent's most recent summary + key_findings. Use when the child needs *facts the parent already established*.
- **`query_parent`** is expensive: child blocks; parent's next turn receives the question; parent answers; answer is delivered to child as a tool_result. Use sparingly. Risk: deadlock if parent is itself blocked. Mitigation: 30s timeout → child gets "parent unresponsive" and proceeds without.

### Sibling coordination via the blackboard

No direct messages. Two mechanisms:

- **Context store, scoped writes.** When S2 detects hatching, it writes a row to `embryo_understanding`. S3 and S4, on their next turn, read recent rows and see "embryo_1 hatched at t=195."
- **Event bus subscription, declared at spawn.** A scope can declare `peer_events=["embryo_perception:checkpoint"]` at `delegate` time. The scope manager forwards matching events from sibling scopes into this scope's inbox. The parent controls what siblings can see.

---

## Mapping to Existing Gently Code

| Concept | Already lives in gently as | Status |
|---|---|---|
| The orchestrator's turn loop | `ConversationManager.call_claude` in `gently/agent/conversation.py` | EXISTS — generalize to take a `scope` parameter |
| Per-embryo working memory | `PerceptionSession` in `gently/agent/perception/session.py` | EXISTS — promote to `ContextScope(type=embryo_perception)` |
| Background timelapse driver | `TimelapseOrchestrator` in `gently/agent/timelapse_orchestrator.py` | EXISTS — wraps as the body of S1; emits `image_ready` events into perception children's inboxes |
| Focus operation | `tools/focus_tools.py:fine_focus` | EXISTS as primitive — wrap with `delegate_focus(embryo_id)` |
| Per-scope event delivery | `core/event_bus.py` | EXISTS — add `scope_id` field to events |
| Scoped persistence | `context/store.py` | EXISTS — add `context_scopes` + `context_scope_messages` tables |
| Tool filtering by scope | `tool_registry.py` (already has categories) | EXISTS — add per-scope `allowed_tools` filter (one-liner) |
| `ScopeManager` (the tree owner) | — | NEW, ~200 lines |
| `ContextScope` dataclass + inbox | — | NEW, ~80 lines |
| `ContextSummarizer` | — | NEW, ~100 lines + per-type templates |
| The four comms tools | — | NEW, ~150 lines in `tools/scope_tools.py` |
| `delegate_focus`, `delegate_timelapse`, `delegate_embryo_perception` | — | NEW, ~30 lines each |
| `/scopes` CLI command | extend `command_registry.py` | NEW, ~50 lines |

**Total new code estimate: ~700 lines.** Everything heavier (LLM call, tool dispatch, event bus, hardware control, perception VLM) is reused.

The headline reframe: **the orchestrator's existing turn loop becomes a generic `run_scope(scope)` that runs the same code in N parallel instances.** `TimelapseOrchestrator` becomes the body of S1. `PerceptionSession` is promoted to a `ContextScope` of type `embryo_perception`.

---

## Worked Example: A 4-Hour Timelapse of 8 *C. elegans* Embryos

This trace is **illustrative**, not normative — it grounds the abstractions in a realistic scenario.

### t = 0:00 — startup

User: _"Start a 4-hour adaptive timelapse on the 8 embryos in the dish. Watch for hatching, alert me if anything looks anomalous."_

```
ROOT (model=opus, tools=ALL)
  Claude turn → tool_use: delegate_timelapse(
    duration_min=240, embryo_ids=["E1".."E8"], hatching_alerts=true)

ScopeManager creates S1 (timelapse, background, sonnet, persist=true)
S1's body is TimelapseOrchestrator — the existing async loop that drives
acquisition cadence. New piece: each acquisition emits image_ready events
into the corresponding perception scope's inbox.

S1's first turn spawns 8 perception children:
  for each embryo: delegate_embryo_perception(embryo_id="E_k")
ScopeManager creates S2..S9 in parallel via asyncio.gather.
S2..S9 sleep on wakeup.wait().
S1 sleeps until next acquisition_tick.

ROOT receives tool_result: {scope_id: S1, status: running}
ROOT replies: "Started. I'll surface anything important."
```

ROOT's history is now ~3 entries longer for an experiment that will run thousands of LLM turns inside the children.

### t = 2:00 — first acquisition tick

```
TimelapseOrchestrator (S1's body) acquires 8 volumes via existing pipeline.
For each volume: event_bus.emit("image_ready", scope_id=Sk, payload={image_b64, t=120})
ScopeManager routes → Sk.inbox + Sk.wakeup.set()

S2..S9 wake in parallel:
  drain inbox → {image_ready, t=120}
  Claude turn (sonnet) → tool_use: perception_engine.analyze(image)
  → existing PerceptionEngine returns {stage: "pre-hatch", confidence: 0.9}
  write observation to context_store
  sleep again

S1 stays asleep — no checkpoints, nothing to wake it.
ROOT stays asleep — user hasn't typed.
```

API calls this tick: 8 (one per perception scope).

### t = 3:15 — hatching detected on E3

```
S4 (E3 perception) Claude turn:
  perception_engine returns {stage: "hatching", transition_from: "pre-hatch"}
  tool_use: yield_checkpoint(
    summary="E3 transitioned to hatching at t=195",
    findings={stage: "hatching", t: 195, confidence: 0.94})

S1.inbox receives checkpoint, S1.wakeup.set()

S1 wakes:
  Claude turn → "E3 hatching — drop interval to 30s for E3, hold others"
  tool_use: adjust_interval(embryo_id="E3", interval=30)
  tool_use: yield_checkpoint(
    summary="E3 hatching at t=195. Adjusted E3 to 30s interval.")

ROOT.inbox receives, ROOT wakes:
  Claude turn → "E3 started hatching at t=195. I tightened its interval to 30s."
  emit_to_user(...)
```

ROOT's history grew by 2 entries. It never saw S4's perception transcript.

### t = 3:50 — drift on E2, focus delegation

```
S3 (E2 perception): VLM reports {stage: "fold", confidence: 0.42, note: "blurry"}
S3 Claude turn:
  "Confidence dropped — looks like drift. Refocus."
  tool_use: delegate_focus(embryo_id="E2")  [BLOCKING]

ScopeManager spawns S10 (focus, transient, sonnet, blocking):
  S10 runs ~3 LLM turns over ~30s:
    fine_focus(E2) → existing tools/focus_tools.py
    set_piezo(42.1)
    complete(findings={best_z: 42.1, drift_um: 1.2})

Summarizer (haiku) over S10's transcript with template:focus
  → "Refocused E2 to z=42.1. Drift was 1.2µm. High confidence."

S3 receives tool_result: {summary, key_findings: {best_z: 42.1, drift_um: 1.2}}
S3 continues. Never sees the 6 messages of piezo-sweep reasoning.
S1 doesn't see this. ROOT doesn't see this. (Routine drift recovery is buried.)
```

This is exactly the "loaded sub-task" pattern from the original vision: S3 needed focus, opened a focus working frame, did the work, returned with one fact (`best_z=42.1`), discarded the rest.

### t = 4:00 — user observes something in live preview

```
User: "E5 looks weird, like it has dark granules"

ROOT wakes (user input)
ROOT Claude turn → tool_use: inject_message(
  scope_id="S6",  # E5's perception scope
  content="User noted unusual dark granules on E5 in live preview.
           On next image, examine carefully and report.")

S6 wakes:
  drain inbox → [{kind:"injection", content:"check granules..."}]
  Claude turn:
    [decides to pull most recent image now rather than wait for next tick]
    tool_use: perception_engine.analyze(image, attention_prompt="dark granules")
    tool_result: {note: "granular bodies visible in posterior region,
                         consistent with stress"}
    tool_use: escalate(severity="warn",
                       message="E5 shows granular bodies — likely stress.")

ScopeManager walks S6 → S1 → ROOT. Lands in ROOT.inbox.

ROOT wakes:
  Claude turn → "E5 is showing stress markers. This often means buffer issues."
  emit_to_user(...)
```

Three communication patterns in three minutes: parent injection (ROOT → S6), child action via existing tool (perception_engine.analyze), child→root escalation (S6 → ROOT, bypassing S1).

---

## Diagrams

### 1. The scope tree at peak activity

```
                    ┌──────────────────────────────────┐
                    │  ROOT  (orchestrator)            │
                    │  model: opus                     │
                    │  body:  ConversationManager      │
                    │  wakes: user input, S1 chkpts,   │
                    │         escalations              │
                    └──────────────┬───────────────────┘
                                   │ delegate(background)
                                   ▼
                    ┌──────────────────────────────────┐
                    │  S1  timelapse                   │
                    │  model: sonnet                   │
                    │  body:  TimelapseOrchestrator    │
                    │  wakes: acquisition_tick,        │
                    │         child checkpoints,       │
                    │         parent injections        │
                    └──┬─────┬─────┬─────┬─────┬───────┘
                       │     │     │     │     │   delegate(background) × 8
            ┌──────┬───┴──┬──┴──┬──┴──┬──┴──┬──┴──┬──────┬──────┐
            ▼      ▼      ▼     ▼     ▼     ▼     ▼      ▼      ▼
          ┌────┐┌────┐┌────┐┌────┐┌────┐┌────┐┌────┐ ┌────┐
          │ S2 ││ S3 ││ S4 ││ S5 ││ S6 ││ S7 ││ S8 │ │ S9 │
          │ E1 ││ E2 ││ E3 ││ E4 ││ E5 ││ E6 ││ E7 │ │ E8 │
          └────┘└─┬──┘└────┘└────┘└────┘└────┘└────┘ └────┘
                  │ delegate(blocking) — drift detected
                  ▼
              ┌────────────────────────────┐
              │  S10  focus  (transient)   │
              │  model: sonnet             │
              │  runs straight through     │
              │  ~30s, ~3 LLM turns        │
              └────────────────────────────┘
```

### 2. A scope's wake / sleep cycle

```
                       ┌────────────────┐
                       │   spawned      │
                       │ status=active  │
                       └────────┬───────┘
                                ▼
                  ┌─────────────────────────┐
              ┌──▶│   await wakeup.wait()   │◀──┐
              │   │     (zero CPU,          │   │
              │   │      zero tokens)       │   │ nothing to do
              │   └────────────┬────────────┘   │
              │                │ wakeup.set()    │
              │                ▼                 │
              │   ┌─────────────────────────┐   │
              │   │   drain inbox           │───┘
              │   │   - injections          │
              │   │   - peer events         │
              │   │   - heartbeat events    │
              │   │   - query responses     │
              │   │   - cancellation        │
              │   └────────────┬────────────┘
              │                │ items present
              │                ▼
              │   ┌─────────────────────────┐
              │   │  append to history      │
              │   │  ONE Claude API call    │
              │   └────────────┬────────────┘
              │                ▼
              │   ┌─────────────────────────┐
              │   │  handle tool calls      │
              │   │   delegate / yield /    │
              │   │   escalate / inject /   │
              │   │   complete / etc.       │
              │   └────────────┬────────────┘
              │   ┌────────────┴────────────┐
              │   ▼                         ▼
              │ stop_reason            complete /
              │ == "tool_use"          end_turn
              └───┘                         │
                                            ▼
                                ┌─────────────────────────┐
                                │  Summarizer (haiku)     │
                                │  produces summary +     │
                                │  key_findings           │
                                └────────────┬────────────┘
                                             ▼
                                ┌─────────────────────────┐
                                │  push to parent inbox   │
                                │  scope removed from     │
                                │  active set             │
                                └─────────────────────────┘
```

### 3. Inside one scope: the plumbing

```
              ┌─────────────────────────────────────────┐
              │             ContextScope                │
              │  ┌───────────────────────────────────┐  │
              │  │  inbox  (asyncio.Queue)           │  │
              │  │  ┌──────────────────────────┐     │  │
              │  │  │ {kind:"event",           │     │  │
              │  │  │  content:"image_ready"}  │     │  │
              │  │  ├──────────────────────────┤     │  │
              │  │  │ {kind:"injection",       │     │  │
              │  │  │  from:"ROOT",            │     │  │
              │  │  │  content:"check granules"│     │  │
              │  │  ├──────────────────────────┤     │  │
              │  │  │ {kind:"peer_event",      │     │  │
              │  │  │  from:"S4",              │     │  │
              │  │  │  content:"E3 hatched"}   │     │  │
              │  │  └──────────────────────────┘     │  │
              │  └────────────┬──────────────────────┘  │
              │               │ drain on wake           │
              │               ▼                         │
              │  ┌───────────────────────────────────┐  │
              │  │  conversation_history             │  │
              │  │  [system prompt + parent slice]   │  │
              │  │  [user]   "image_ready t=120"     │  │
              │  │  [asst]   tool_use:analyze(...)   │  │
              │  │  [user]   tool_result:{stage:..}  │  │
              │  │  [asst]   yield_checkpoint(...)   │  │
              │  └────────────┬──────────────────────┘  │
              │               ▼                         │
              │  ┌───────────────────────────────────┐  │
              │  │  Claude call (sonnet)             │  │
              │  │  filtered tools:                  │  │
              │  │    perception_engine.analyze      │  │
              │  │    fine_focus                     │  │
              │  │    yield_checkpoint               │  │
              │  │    escalate                       │  │
              │  └───────────────────────────────────┘  │
              │                                         │
              │  wakeup: asyncio.Event                  │
              │  cancel_token: asyncio.Event            │
              │  pending_queries: {qid: Future}         │
              └─────────────────────────────────────────┘
```

### 4. Parent → child injection (sequence)

```
   user                ROOT              ScopeManager           S6 (E5 perception)
    │                   │                     │                       │
    │ "E5 looks weird"  │                     │                       │
    │──────────────────▶│                     │                       │
    │                   │ Claude turn (opus)  │                       │
    │                   │ tool_use:           │                       │
    │                   │   inject_message(   │                       │
    │                   │     S6, "...")      │                       │
    │                   │────────────────────▶│                       │
    │                   │                     │ S6.inbox.put(msg)     │
    │                   │                     │ S6.wakeup.set()       │
    │                   │                     │──────────────────────▶│
    │                   │                     │                       │ wakes
    │                   │ "I'll have S6 take  │                       │ drains inbox
    │                   │  a closer look."    │                       │ Claude turn
    │◀──────────────────│                     │                       │ tool_use: analyze
    │                   │                     │                       │ tool_use: escalate
    │                   │                     │◀──────────────────────│
    │                   │ inbox: escalation   │                       │
    │                   │ wakeup.set()        │                       │
    │                   │◀────────────────────│                       │
    │                   │ wakes, surfaces     │                       │
    │ "E5 stress signs" │                     │                       │
    │◀──────────────────│                     │                       │
```

### 5. Escalation: skip-the-middle

```
  Normal yield_checkpoint walks one level:

       S10 (focus)
         │ yield_checkpoint
         ▼
       S3 (perception)        ← S3 sees and decides what (if anything) to forward
         │ yield_checkpoint
         ▼
       S1 (timelapse)         ← S1 sees and decides
         │ yield_checkpoint
         ▼
       ROOT


  Escalation jumps straight to ROOT:

       S10 (focus)
         │
         │ escalate(critical, "sample dry, cannot recover")
         │
         │  ScopeManager walks scope.parent_id chain to root
         │     ┌──────┐
         │     │  S3  │   ── skipped, but its inbox optionally gets a copy
         │     └──────┘     so it's aware
         │     ┌──────┐
         │     │  S1  │   ── skipped likewise
         │     └──────┘
         ▼
       ROOT inbox: ⚠ critical from S10
```

### 6. Concurrent acquisition tick

```
   t=195   TimelapseOrchestrator (body of S1) begins acquisition cycle
                                │
                                │ device_layer (serialized): acquire 8 volumes
                                │
       ┌──────┬──────┬──────┬───┴───┬──────┬──────┬──────┐
       ▼      ▼      ▼      ▼       ▼      ▼      ▼      ▼
     emit  emit   emit   emit    emit   emit   emit   emit       event_bus
     E1    E2     E3     E4      E5     E6     E7     E8
       │      │      │      │       │      │      │      │
       ▼      ▼      ▼      ▼       ▼      ▼      ▼      ▼
     inbox  inbox  inbox  inbox   inbox  inbox  inbox  inbox
     wake   wake   wake   wake    wake   wake   wake   wake
       │      │      │      │       │      │      │      │
       ▼      ▼      ▼      ▼       ▼      ▼      ▼      ▼

   ╔══════ asyncio.gather (parallel sonnet calls) ══════╗
   ║                                                    ║
   ║   S2     S3     S4     S5     S6     S7     S8    S9
   ║    │      │      │      │      │      │      │      │
   ║   VLM    VLM    VLM    VLM    VLM    VLM    VLM    VLM
   ║    │      │      │      │      │      │      │      │
   ║  pre-   fold  HATCH  pre-   pre-   fold   pre-   pre-
   ║  hatch       ─────  hatch  hatch          hatch  hatch
   ║    │      │      │      │      │      │      │      │
   ╚════╧══════╧══════╪══════╧══════╧══════╧══════╧══════╝
                      │ only S4 yields a checkpoint
                      ▼
                   S1.inbox: "E3 hatching at t=195"
                   S1.wakeup.set()

   API calls this tick: 8 perception turns + 1 S1 turn = 9
                       (only one bubbles up to ROOT, eventually)
```

### 7. Focus delegation (blocking child, full lifecycle)

```
  S3 (E2 perception)                ScopeManager           S10 (focus)        Summarizer
       │                                  │                     │                  │
       │ tool_use: delegate_focus(E2)     │                     │                  │
       │─────────────────────────────────▶│                     │                  │
       │                                  │ create scope        │                  │
       │                                  │ blocking: await ────▶                  │
       │                                  │                     │ Claude turn      │
       │                                  │                     │ tool_use:        │
       │                                  │                     │   fine_focus(E2) │
       │                                  │                     │ ─── existing tools/focus_tools.py
       │                                  │                     │     piezo sweep, sharpness
       │                                  │                     │ tool_use:        │
       │                                  │                     │   set_piezo(42.1)│
       │                                  │                     │ tool_use:        │
       │                                  │                     │   complete({     │
       │                                  │                     │     best_z:42.1, │
       │                                  │                     │     drift:1.2 }) │
       │                                  │ status=completed    │                  │
       │                                  │ summarize(scope)────────────────────▶ │
       │                                  │                     │                  │ haiku
       │                                  │◀────────────────────────────────────── │ template:focus
       │                                  │  "Refocused E2 to                      │
       │                                  │   z=42.1. Drift                        │
       │                                  │   1.2µm. High conf."                   │
       │ tool_result: {summary, findings} │                                        │
       │◀─────────────────────────────────│                                        │
       │ continues, never sees the 6-msg                                            │
       │ piezo-sweep transcript                                                     │
```

### 8. What each level "sees" (information density)

```
                              ────────────────
                              |  ROOT views  |       (opus, expensive context)
                              ────────────────
   over 4 hours:                       │
                                       │ history grows by:
                                       │   - user messages
                                       │   - S1 checkpoints (a few/hr)
                                       │   - escalations (rare)
                                       │ ≈ tens of entries total
                                       ▼
                       ┌────────────────────────────────┐
                       │       S1 (timelapse) view      │       (sonnet)
                       └────────────────┬───────────────┘
                                        │ history grows by:
                                        │   - acquisition_tick events
                                        │   - perception checkpoints
                                        │   - own reasoning about pacing & alerts
                                        │ ≈ hundreds of entries
                                        ▼
                  ┌───────────────────────────────────────────┐
                  │   S2..S9 (each embryo perception) view    │       (sonnet × 8)
                  └─────────────────────┬─────────────────────┘
                                        │ history grows by:
                                        │   - image_ready every cadence
                                        │   - VLM tool_results
                                        │   - own continuity reasoning
                                        │ ≈ thousands of entries each
                                        ▼
                            ┌────────────────────────┐
                            │   S10 (focus) view     │       (sonnet, transient)
                            └────────────────────────┘
                                  short, ≈ 6 entries
                                  discarded after summary

      KEY:  detail accumulates DOWNWARD, summaries flow UPWARD,
            and each level holds ONLY what it needs to reason at its pace.
```

---

## Implementation Plan (Phased)

> Status of each phase will be tracked here once the design is ACCEPTED.

### Phase 1 — Generic primitive (foundation)

- New: `gently/agent/scopes/scope.py` (`ContextScope` dataclass + inbox/wakeup)
- New: `gently/agent/scopes/manager.py` (`ScopeManager`, `run_scope` loop)
- New: `gently/agent/scopes/summarizer.py` (Haiku-based summarizer + first template)
- New: `gently/agent/tools/scope_tools.py` (`delegate`, `complete`, `yield_checkpoint`)
- Modified: `gently/agent/conversation.py` (extract `run_turn(scope)`, root scope wraps existing history — day-1 behavior identical to today)
- Tests: spawn → run → summarize → return; nested spawn; cancellation.

**Exit criterion:** orchestrator can spawn an `ad_hoc` child for any task, gets a typed summary back, parent's history grows by O(1) per delegation regardless of child's internal turn count.

### Phase 2 — Communication primitives

- Add `inject_message`, `escalate`, `read_parent_state`, `query_parent`, `respond_to_query`, `cancel_subcontext` to `scope_tools.py`
- `event_bus` events gain `scope_id` field; `ScopeManager` routes events into per-scope inboxes
- Tests: parent injection, child escalation skip-the-middle, query timeout behavior, cooperative cancellation

**Exit criterion:** the four communication primitives all round-trip correctly, including the deadlock-resistant query timeout.

### Phase 3 — The three preset wrappers

- `delegate_focus`: refactor `tools/focus_tools.py:fine_focus` so the orchestrator-facing entry point is a focus scope; the primitive sweep tools become children-only
- `delegate_embryo_perception`: promote `PerceptionSession` to a `ContextScope`; route `PerceptionManager.process_image` through the scope's inbox as `image_ready` events
- `delegate_timelapse`: wrap `TimelapseOrchestrator` as the body of S1; on each acquisition tick, route per-embryo events into the perception children
- Tests: end-to-end timelapse smoke test in `--full-offline` mode

**Exit criterion:** the 4-hour timelapse trace from the worked example runs in `--full-offline` mode, producing the expected scope tree and per-level information density.

### Phase 4 — Persistence and CLI

- `context_scopes` and `context_scope_messages` tables in `context/store.py`
- Opt-in via `persist=True` on `delegate`; `delegate_timelapse` and `delegate_embryo_perception` default to persist
- New `/scopes` CLI command (live tree view) + `/inspect <scope_id>` (full transcript dump)
- Tests: persisted scope survives restart and is inspectable; no resumption (out of scope)

**Exit criterion:** user can run a timelapse, restart the copilot, and inspect the scope tree from before the restart.

---

## Verification

1. **Unit:** `tests/test_scope_manager.py` — spawn → run → summarize → return; assert child transcript is not in parent history; assert summary is.
2. **Hierarchy:** spawn child that spawns grandchild; verify grandchild summary visible to child but not to root.
3. **Concurrency:** spawn 8 background scopes from one orchestrator turn; verify they run in parallel (wall-clock < 2× single-scope) and event logs are correctly scoped.
4. **Communication:** parent injection arrives in child's next LLM turn; escalation skips intermediates; `query_parent` times out cleanly when parent is busy.
5. **Preset smoke tests:** `delegate_focus` on a simulated embryo returns a `best_z`; `delegate_embryo_perception` over a recorded image sequence emits a hatching detection.
6. **End-to-end:** in `--full-offline` mode, run `python launch_copilot.py`, type "start a timelapse on 3 embryos and tell me when any starts hatching"; verify (a) `delegate_timelapse` background scope is spawned, (b) 3 `embryo_perception` children appear under it in `/scopes`, (c) orchestrator's history stays clean, (d) when a perception scope detects hatching, the orchestrator surfaces it without ever seeing the raw VLM transcript.
7. **Persistence smoke:** spawn `delegate_timelapse(persist=True)`, restart copilot, verify the scope record + messages are still inspectable via `/scopes`.

---

## Open Decisions

These are unresolved design questions. As they get decided, move them to a "Decided" subsection below with date and resolution.

1. **Stop conditions for the three preset scopes.** Should `delegate_focus` / `delegate_timelapse` / `delegate_embryo_perception` *require* `complete` (typed return) or accept `end_turn`? Current default: both accepted, `complete` preferred for typed scopes.
2. **`yield_checkpoint` shape for long-lived scopes.** The current design has children push checkpoints. Alternative: parent subscribes to a child event stream. Current default: push-based checkpoints, keep it simple.
3. **`query_parent` in v1 or v2.** Powerful but introduces deadlock risk. Could ship `read_parent_state` only in v1 and add `query_parent` later if a real use case demands it. Current default: include in Phase 2 with a 30s timeout.
4. **Sibling `peer_events` policy.** Parent-controlled (declared at spawn) vs child-subscribable (flexible). Current default: parent-controlled.
5. **Persistence default for short scopes.** `delegate_focus` is currently `persist=False`. Should we persist all scopes for full audit trail? Current default: no, opt-in only.
6. **Default summarizer model.** Haiku for cost, sonnet for fidelity. Current default: Haiku, configurable per-scope-type.
7. **Wake-up source for perception scopes between images.** Currently only `image_ready` triggers wake. Should perception scopes also wake on a wall-clock heartbeat to "think between images"? Current default: no — saves tokens, perception is reactive only.
8. **Routine sub-results visibility.** Should `delegate_focus` summaries roll up to ROOT by default, or stay buried inside the perception scope that called it? Current default: buried; perception scope decides whether to forward.

### Decided

_(none yet)_

---

## Out of Scope (Deferred)

- **Scope resumption after restart.** Persistence preserves transcripts but doesn't resume running scopes. Future work.
- **Cross-process scopes.** Sub-contexts running in separate Python processes. Not needed for current cost shape.
- **Scope migration / serialization across machines.** Not needed.
- **Automatic scope-type classification.** Today the orchestrator picks the scope type explicitly via the preset tool name. We do not propose to auto-classify "this looks like a focus task."
- **Sibling-to-sibling direct messaging.** Explicitly rejected to keep the tree authoritative; coordination via blackboard is the supported pattern.
- **Per-account LLM rate limit semaphore.** Becomes relevant for 50+ parallel scopes. Not needed for v1 (3–8 embryos).
- **Live UI (web) view of the scope tree.** `/scopes` CLI command is the v1 interface.

---

## Glossary

| Term | Definition |
|---|---|
| **scope** | A `ContextScope` instance: a node in the tree with its own conversation history, inbox, toolset, model. |
| **root scope** | The top-level orchestrator scope. Identical in behavior to today's `ConversationManager`. |
| **delegate** | The generic primitive that spawns a child scope from any scope. |
| **preset wrapper** | A thin function (`delegate_focus`, `delegate_timelapse`, `delegate_embryo_perception`) that calls `delegate` with sensible defaults for a known scope-type. |
| **inbox** | A per-scope `asyncio.Queue` holding incoming injections, peer events, query responses, and cancellation signals. |
| **wakeup** | A per-scope `asyncio.Event` set when there is work in the inbox. The scope sleeps on `wakeup.wait()`. |
| **checkpoint** | A non-terminating summary emitted by a long-lived scope to its parent (`yield_checkpoint`). |
| **escalation** | A fast-path message from any descendant straight to the root scope, bypassing intermediate parents. |
| **summarizer** | The Haiku/Sonnet-backed module that condenses a completed scope's transcript into a summary + key_findings. |
| **blackboard** | The shared state (context_store + event_bus) used for sibling-to-sibling coordination. |

---

## Decision Log

Append-only history of substantive design changes to this document. Each entry: date, version, change, rationale.

### 2026-05-01 — v0.1 (initial proposal)

Initial draft. Establishes:
- The `ContextScope` abstraction and `delegate` primitive
- Event-driven sleep / one-LLM-turn-per-wake runtime
- Four communication primitives across vertical axes
- Sibling coordination via existing context_store + event_bus blackboard
- Mapping of new vs existing gently code (~700 LoC new)
- Three v1 preset scope types: focus, timelapse, embryo_perception
- Phased implementation plan (4 phases)

Open decisions enumerated; none resolved yet.
