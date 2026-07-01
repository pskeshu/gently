# Gently UI Crawler / Simulator

A headless-browser crawler that **walks the web UI like a user** and builds an
empirical state-transition graph — the *dynamic* complement to the static
[user-story audit](../../docs/user-stories/README.md).

## Why

Static code tracing describes what the UI *wires up*; it can't see emergent
runtime behaviour. The static audit, for example, noted the landing has "no
persistence" but never reported the consequence: **reloading returns you to the
landing**. A crawler that actually clicks (and reloads) surfaces these as observed
transitions — plus dead controls, clicks that throw console errors or HTTP
4xx/5xx, infinite spinners, and unreachable states.

## How it works

From a seed URL it BFS-explores UI states:
1. **Fingerprint** the state — a *structural* signature (active tab, active views,
   visible panels, landing visible?, modals, spinner) that ignores data so states
   dedupe on structure.
2. **Enumerate** every interactive element with a stable selector, plus synthetic
   **`__reload__` / `__goto_root__`** actions (so browser-level transitions — the
   real way back to the landing — are explored, not just clicks).
3. **Probe** each element in an isolated page that reaches the state by replaying
   the click-path from root, clicks, then diffs the fingerprint and records
   console/HTTP errors during the action.
4. Repeat over newly-discovered states up to `--max-depth`.

Probes run across **N parallel browser contexts** (`--workers`), each isolated and
resilient (per-action timeouts), so one bad element can't wedge the run.

## Install

Playwright is in the `dev` dependency group (`pyproject.toml`). After `uv sync`:

```bash
uv run playwright install chromium     # one-time browser download (~115 MB)
# firefox / webkit optional: uv run playwright install firefox
```

## Run

Point it at a running viz server (dev tip: `--no-auth` so control routes don't 403):

```bash
python launch_gently.py --no-api --no-auth --no-browser      # in one shell
uv run python tools/ui_crawler/crawler.py --url http://localhost:8080 \
    --workers 3 --max-depth 3 --out tools/ui_crawler/out
```

Key flags: `--browser {chromium,firefox,webkit}` · `--workers N` · `--max-depth` ·
`--max-states` · `--max-elements` · `--timeout` (ms) ·
`--url` (crawl account-mode by pointing at a logged-out server to catch the
view-only experience).

### Watch it work

Three ways to make the crawl visible:

```bash
# 1. Live window — a real browser doing the clicks, slowed down
uv run python tools/ui_crawler/crawler.py --headed --slow-mo 500 --workers 1

# 2. Trace viewer — record everything, then scrub every action (screenshots +
#    DOM snapshots + network + console). Best for review; works headless.
uv run python tools/ui_crawler/crawler.py --trace
uv run playwright show-trace tools/ui_crawler/out/trace.zip

# 3. Video — a .webm recording of each page
uv run python tools/ui_crawler/crawler.py --video    # -> out/videos/
```

Use `--workers 1` with `--headed` so it's a single, followable window.

### The trace viewer (recommended way to review findings)

`playwright show-trace <trace.zip>` opens an interactive, time-travel viewer — the
best way to see *exactly* what happened in a run without watching it live:

```bash
uv run playwright show-trace tools/ui_crawler/out/traces/00-return-to-landing-gently-microscopy.zip
```

What you get in the window:
- **Timeline + filmstrip** — every action as a screenshot thumbnail; scrub across
  it to watch the UI change.
- **Actions list** (left) — click any action to jump to it.
- Per action: the **before/after screenshot**, an inspectable **DOM snapshot**, and
  the **Console**, **Network** (e.g. the `GET /` that reset a state, or a 502), and
  **Source** (the Playwright call) tabs.

It needs a display (`DISPLAY`) and the full Chromium build (`playwright install
chromium` provides both the headless shell and the headed browser show-trace uses).
The viewer runs until you close its window.

Per-finding traces (`--trace-findings`) and scenario traces are named by finding,
so `out/traces/` and `out/scenarios/` are a browsable catalogue — open the one you
want to inspect. `out/*/index.json` maps each trace to its finding + observation.

## Output (`out/`, git-ignored)

- **`graph.json`** — every state (with reach-path) and every probed edge.
- **`graph.mmd`** — a Mermaid transition graph (notable edges flagged ⚑).
- **`report.md`** — findings grouped by type: returns-to-landing, console errors,
  HTTP 4xx/5xx, spinners, dead controls, unreachable tabs — each with the
  reach-path for reproduction.

## Reproducing static-audit findings (`scenarios.py`)

The crawler finds deficiencies by *walking* the app. The **static code-audit**
findings (missing export, no ground-truth annotation, the dead notebook Questions
tab, the view-only plan wizard that spins) aren't reachable by blind crawling —
`scenarios.py` navigates to each surface deliberately and captures a trace of
what's absent/broken:

```bash
# optional account-mode server for the view-only scenario:
GENTLY_VIZ_PORT=8081 GENTLY_STORAGE_PATH=/tmp/gently_acct python launch_gently.py --no-api --no-browser
uv run python tools/ui_crawler/scenarios.py --url http://localhost:8080 \
    --account-url http://localhost:8081        # -> out/scenarios/<name>.zip + index.json
```

Each scenario records an `observed` line (e.g. "export controls: 0") and a trace.
Rig/agent-only findings (LED force-close, scripted_protocol false success,
answer-silently-queued, autonomous-turn stop) are listed in `index.json` as
NOT headless-reproducible. Scenarios whose selectors don't yet trigger the exact
control are flagged so their traces aren't mistaken for clean reproductions.

## Per-story UX audit + regression (`run_stories.py`) — the spine

The audit + "did a change break a UX" gate. Each documented user story has its own
flow in `stories/US-XX-*.py` (a plain `async flow(page, url, rec)` that drives the
story's *intended* path and records a **4-state verdict** — works / partial / gap /
blocked). The runner executes every flow in its own trace and emits a status report:

```bash
python launch_gently.py --no-api --no-auth --no-browser                 # control shim
GENTLY_VIZ_PORT=8081 GENTLY_STORAGE_PATH=/tmp/gently_acct python launch_gently.py --no-api --no-browser  # account (view-only)
uv run python tools/ui_crawler/run_stories.py --url http://localhost:8080 --account-url http://localhost:8081
```

**Audit:** `out/stories/STATUS.md` + `status.json` + a trace per story. That report
is the audit — reviewable, with a scrubbing trace for every story.

**Regression:** the report is committed as `baseline/status.json`. A later run
**diffs against it** — a story that FLIPS (e.g. `works → gap`) is a break. The runner
prints regressions (⬇) vs improvements (⬆) and exits non-zero on any regression, so
it can gate a pipeline. `--update-baseline` re-baselines; `--docs-status` refreshes
`docs/user-stories/STATUS.md`.

**Triage on a flip** (this is the point): the baseline **+ the story doc** is the
contract. Never silently re-baseline —
- same intent, only selectors moved → fix the flow;
- intent broken, unintended → **regression, fix the UX**;
- intent deliberately changed → **edit the story doc, then re-baseline** (the doc edit
  makes the paradigm shift explicit + reviewable in the PR).

Assert *intent* (story goal reachable), not brittle selectors, so a flip means
something. Rig/agent-gated stories declare `mode: "rig"`/`"agent"` and record
`blocked` rather than faking a pass. `_harness.py` provides the shared helpers
(`goto`, `tab`, `view`, `count_text`, `exists`, `present`, `dom_count`, `Rec`).

## Relationship to the docs

The crawler verifies the [`docs/user-stories/`](../../docs/user-stories/) flows
empirically. Findings feed the per-story `Status` and the deficiency report; the
static audit + the crawl together are the "map + walk" of gently's UX.

> Scope: single viz server, structural fingerprinting. It does not assert
> scientific correctness — pair it with the agent-driven audit for semantic
> judgement.
