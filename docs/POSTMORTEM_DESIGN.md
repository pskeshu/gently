# Session Post-Mortem Subsystem — Design

**Status:** design + Claude Code side scaffolded (skill, workflow, schema, first stored report). Harness-side
Python (self-documenting runs, evidence pack, storage, trigger) is specified below and not yet implemented.

**Origin:** a hand-run forensic analysis of session `6a4a3d9b` (the first autonomous run) surfaced ~13 findings —
3 live embryos falsely terminated, a dead vision tool, an empty decision log, mechanical drift misread as thermal.
This subsystem turns that one-off into a repeatable, stored, feedback-generating capability.

## Purpose & audience

Post-mortems are a **harness-quality instrument**, not experiment reports for the bench. The consumers are:

- **developers** — a per-run forensic report + a cross-run backlog of recurring failure modes; and
- **Claude Code itself** — the reports (and the regression evals derived from them) are how the agent improves its
  own harness across sessions.

Findings are about the *system* (agent behaviour, tool correctness, orchestration safety, perception reliability,
observability). Biology outcomes appear only as *evidence for a system finding* (e.g. "3 embryos lost" indicts the
termination rule).

## Architecture: split deterministic scaffolding from judgment

The report is authored by **Claude Code**, not by in-harness Python. Gently builds the reliable, deterministic
parts; Claude Code does the analysis that benefits from a strong model + adversarial verification.

```
run ends  ──►  [GENTLY / Python]                     [CLAUDE CODE = analyst]
               ├ finalize: summary.yaml,              .claude/skills/session-postmortem  +
               │  close perception_run, drop lock     .claude/workflows/session-postmortem.js
               ├ build evidence_pack.json  ──────────►   fan-out lenses → adversarial-verify → synthesize
               └ emit "postmortem_pending"                                     │
                        │  trigger: hook / nightly routine / on-demand         │  writes back
                        └────  claude -p "/session-postmortem <id>"  ──────────┘
                                                                               ▼
   sessions/{id}/postmortem/report.{yaml,md}   agent/postmortems/{id}.yaml   agent/known_issues/{fingerprint}.yaml
        (per-run)                                 (cross-run index)          (recurrence-tracked backlog)
                                                        confirmed + high-confidence only ──► gh issue / eval stub
```

Why the split (not a pure in-harness `messages.create` analyzer): a single LLM call is exactly what produced the
first-pass narrative whose "thermal drift" and "re-termination loop" claims were **wrong** until the adversarial
verify pass corrected them. The multi-lens + verify workflow is the trust gate. The deterministic evidence pack
makes Claude Code's inputs complete and stable, so it spends tokens on judgment, not on re-parsing JSONL.

## What Gently builds (Python)

### 1. Make runs self-documenting — **fix-first**
`decisions.jsonl` was empty in `6a4a3d9b` because `_write_production_decision` only fires on the non-streaming
`call_claude` path with hardcoded `trigger=USER_MESSAGE` (`gently/harness/conversation.py:379`); the autonomous
wake path writes nothing. Add `DecisionLog` capture with `trigger=DecisionTrigger.EVENT` on the wake/streaming path
(`docs/EVAL.md` Step 4 already scopes this). Best-effort/guarded so it never breaks a turn. **This is the single
highest-leverage change** — it makes every future run self-documenting, so post-mortems analyze rather than reconstruct.

### 2. Deterministic evidence-pack builder
New `gently/eval/postmortem_pack.py` (co-located with `DecisionLog` / `EventReplay` / `shadow`). A read-only
aggregator — the analyzer the scout recommended **minus** the LLM call — composing existing readers:
- `EventType` enum (`gently/core/event_bus.py:25`) as the fixed timeline vocabulary;
- `strategy_snapshot.py::_replay_timeline` (`gently/ui/web/`) for per-embryo spans (don't write a second parser);
- `DecisionLog.read()`, `EventReplay.event_types()`, `InteractionLogger.get_session_stats()`;
- `PerceptionMetrics` (`benchmarks/perception/metrics.py`) when `ground_truth.yaml` exists; `read_temperature_log` stats.
Emits `sessions/{id}/postmortem/evidence_pack.json`.

### 3. Schema, storage, trigger
- `FileStore.put_postmortem/get_postmortem` via the atomic `_write_yaml` → `sessions/{id}/postmortem/report.{yaml,md}`.
  Also finally writes the documented-but-missing `summary.yaml` (`gently/core/file_store.py:18`).
- Add `agent/postmortems/` and `agent/known_issues/` to `_ensure_dirs` (`file_store.py:125`).
- Trigger at the single funnel `_finalize_perception_run` (`gently/app/orchestration/timelapse.py:2532`) — all three
  end paths route through it — **plus a reconciler**, because a signal-killed manual run (the `6a4a3d9b` case)
  bypasses every finalize path and leaves `perception_runs.yaml` at `status: running`. Wire the orphaned
  `acquire/release_session_lock` (`file_store.py:514`) so "lock present, PID dead" ⇒ crashed ⇒ post-mortem
  retroactively. `scripts/postmortem_session.py <id>` reuses `scripts/replay_session.py`'s id-prefix resolution for
  on-demand / bulk backfill. The trigger only *spawns* Claude Code (`claude -p "/session-postmortem <id>"`); it does
  not run the analysis itself.

## What Claude Code owns

- **`.claude/skills/session-postmortem/`** — the skill (analyst instructions) + `references/report_schema.yaml`.
- **`.claude/workflows/session-postmortem.js`** — the saved fan-out → verify → synthesize workflow, parameterized by
  session via `args`. Version-controlled *with the harness*: the analysis evolves in the same PRs as the code it critiques.
- **Confidence-gated write-back:** only `verdict=CONFIRMED` + `confidence=high` findings auto-open a `gh` issue or emit
  an eval stub; `PLAUSIBLE`/low-confidence land in the rollup's triage bucket. `generator_version` on every report.

## Report schema

See `.claude/skills/session-postmortem/references/report_schema.yaml` (schema_version 1). The core unit is a typed,
evidence-first **finding** tagged by `audience` (agent | harness | hardware | biology), with `verdict`, `confidence`,
`proposed_fix`, `proposed_eval`, and a stable `fingerprint` (`audience:category:signature`) that is the cross-run dedup key.

## The feedback loop that improves the harness

- **Rollup (`agent/known_issues/{fingerprint}.yaml`)** — the per-run report is an anecdote; recurrence is the signal.
  Stable fingerprints let `seen_in: [session_ids]` accumulate ("no_object false-termination — 4 of 7 autonomous runs").
  Reuse `compare_reports()` (`benchmarks/agent/evaluator.py:259`) for session-over-session deltas.
- **Evals** — agent-behaviour findings become regression tests on the existing `gently/eval/` shadow substrate: replay a
  decision context and assert the policy now behaves (e.g. "queries `get_temperature` before naming a cause"). This is how
  a fix *stays* fixed. Start simple (golden decision fixture + assertion); grow toward full shadow-replay.
- **Routing rule (important):** `harness`/`agent`/`hardware` findings are developer-facing and must **not** be written to
  the learnings store — `get_awareness_summary` / `get_session_briefing` inject learnings (top-5 verbatim) into *every*
  agent prompt, so dev-bug "learnings" would pollute the live agent. Only `audience: biology` findings may become
  `add_learning` / `add_observation`. The run's one-line verdict goes to `complete_session_intent` (`actual_summary`).

## Storage layout

```
D:/Gently3/
  sessions/{id}/postmortem/
    evidence_pack.json        # deterministic (Gently-built)
    report.yaml               # schema (Claude Code-authored)
    report.md                 # human narrative
  agent/
    postmortems/{session_id}.yaml         # cross-run index (dev-facing)
    known_issues/{fingerprint}.yaml       # recurrence-tracked backlog (dev-facing, NOT prompt-injected)
```

## Build plan (phased; each step independently useful)

1. **Self-documenting runs** — decisions.jsonl on the wake path + `summary.yaml`/finalize/lock. *(Small, unblocks everything.)*
2. **Claude Code side** — skill + workflow + schema. ✅ *scaffolded; first report at `sessions/…_6a4a3d9b/postmortem/`.*
3. **Evidence-pack builder** + `put_postmortem` storage.
4. **Trigger + reconciler**, then **known_issues rollup + eval stubs**.

## Known findings already tracked (from `6a4a3d9b`)

The reference report (`sessions/20260702_1847_unnamed_6a4a3d9b/postmortem/report.yaml`) records F001–F013. The
harness-critical ones — F001 (drift-blind calibration termination), F002 (refocus doesn't reach acquisition),
F003 (empty wake-path decision log) — are the first backlog items this subsystem exists to surface and, eventually,
auto-file.
