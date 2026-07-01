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
`--max-states` · `--max-elements` · `--timeout` (ms) · `--headed` (watch it) ·
`--url` (crawl account-mode by pointing at a logged-out server to catch the
view-only experience).

## Output (`out/`, git-ignored)

- **`graph.json`** — every state (with reach-path) and every probed edge.
- **`graph.mmd`** — a Mermaid transition graph (notable edges flagged ⚑).
- **`report.md`** — findings grouped by type: returns-to-landing, console errors,
  HTTP 4xx/5xx, spinners, dead controls, unreachable tabs — each with the
  reach-path for reproduction.

## Relationship to the docs

The crawler verifies the [`docs/user-stories/`](../../docs/user-stories/) flows
empirically. Findings feed the per-story `Status` and the deficiency report; the
static audit + the crawl together are the "map + walk" of gently's UX.

> Scope: single viz server, structural fingerprinting. It does not assert
> scientific correctness — pair it with the agent-driven audit for semantic
> judgement.
