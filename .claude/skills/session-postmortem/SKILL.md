---
name: session-postmortem
description: >
  Produce a structured, stored forensic post-mortem of a Gently run (session). Use when asked to
  "analyze / review / post-mortem a session", "what happened in run <id>", or to generate the stored
  report for a completed/killed session. The report is authored by Claude Code (this skill), consumed
  by developers and by Claude Code itself to improve the harness. Takes a session id or folder.
---

# Session post-mortem

You are the **analyst**. The Gently harness produces the inputs (per-session artifacts + a deterministic
evidence pack, when available); you produce the report. Output must conform to
`references/report_schema.yaml` (schema_version 1) and be **evidence-first, auditable, and mergeable across runs**.

## 1. Resolve the session

Sessions live under `D:\Gently3\sessions\`. Map an 8-char id via `D:\Gently3\sessions\_index.yaml`
(`{id8}: {folder_name}`). The folder is the working root for everything below.

## 2. Read the inputs

Prefer the deterministic **evidence pack** (`sessions/{id}/postmortem/evidence_pack.json`) if present — it is the
stable, complete summary the harness builds. If it is missing (not built yet), read the raw artifacts directly:

| Artifact | What to extract |
|---|---|
| `session.yaml`, `timelapse.yaml`, `perception_runs.yaml` | run config, per-embryo endpoints, completion reasons, exposure, run status |
| `chat_display.json` | the curated agent chat (user / agent / tool / autonomous_start[trigger] / autonomous[text]) |
| `conversation.json` | full Claude message list — untruncated reasoning + exact tool I/O (the ground truth when `decisions.jsonl` is empty) |
| `decisions.jsonl` | structured decisions **if non-empty**; if empty, note it as a data gap and fall back to `conversation.json` |
| `events.jsonl`, `timeline.jsonl` | ground-truth event timeline (vocab = `EventType`, `gently/core/event_bus.py`) |
| `embryos/*/predictions.jsonl` + `traces/` | per-embryo stage trajectories, oscillation, no_object onsets, reasoning text |
| `embryos/*/embryo.yaml` | role, calibration, focus history |
| `temperature.jsonl` | source (real vs `(SIM)`), setpoint, water_c stats — **check before attributing anything to temperature** |

**Encoding:** default Python encoding here is cp1252 — always `io.open(path, encoding='utf-8')` and keep stdout ASCII
(or `errors='replace'`). Aggregate large JSONL with python; do not Read multi-MB files wholesale.

## 3. Analyze (use the workflow for anything non-trivial)

Run the saved workflow **`session-postmortem`** (`.claude/workflows/session-postmortem.js`, pass the session folder as
`args`). It fans out lenses (chat/decisions, raw events, perception, drift/termination, temperature, code-grounded bugs),
then **adversarially verifies the load-bearing findings**, and returns a draft report object.

The adversarial-verify pass is not optional polish — it is the trust gate. A first-pass narrative typically contains
plausible-but-wrong claims (e.g. "thermal drift", a "re-termination loop") that only verification catches. Set each
finding's `verdict` (CONFIRMED / PLAUSIBLE) and `confidence` from that pass.

## 4. Write the report

Render `sessions/{id}/postmortem/report.yaml` (the schema) + `report.md` (human narrative). Rules:
- Every finding cites ≥1 artifact by path + key. No evidence → no finding.
- Tag each finding's **audience**: `harness` (bug/gap), `agent` (behaviour/reasoning), `hardware`, `biology`.
- `polarity: positive` findings (good calls) are first-class — capture them so harness changes don't regress them.
- Derive a stable `fingerprint` (`audience:category:signature`) so recurrence merges across runs.
- `auto_actionable: true` only when `verdict=CONFIRMED` **and** `confidence=high`.

## 5. Route the outputs (the feedback loop)

- **Index:** append/update `agent/postmortems/{session_id}.yaml` (one-line outcome + finding counts by audience/severity).
- **Rollup:** for each finding, upsert `agent/known_issues/{fingerprint}.yaml`, appending this `session_id` to `seen_in[]`.
  This is the developer-facing backlog — recurring issues rise to the top.
- **Actionable only:** for `auto_actionable` findings, open a GitHub issue (`gh issue create`) and/or drop an eval stub
  from `proposed_eval`. `PLAUSIBLE`/low-confidence findings go to the rollup's triage bucket, never auto-filed.
- **Biology findings ONLY** may become memory (`add_learning` / `add_observation`) — they resurface in future runs.
  **Never** write `harness`/`agent` findings to the learnings store: learnings are injected into every agent prompt and
  would pollute the live agent's context. Keep them developer-facing.
- Set the run's one-line verdict as the session's `actual_summary` (`complete_session_intent`) so `query_lab_history` sees it.

## 6. Report back

Summarize for the human: outcome, the top confirmed findings by severity, what was auto-filed, and any triage-bucket
items needing a human call. Link `report.md`.
