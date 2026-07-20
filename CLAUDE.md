# Gently — Microscopy Agent

## Working on this repo

**README.md** covers environment setup and how to run things (`uv sync`,
`uv run pytest`, `uv run python launch_gently.py` and its flags).
**CONTRIBUTING.md** covers the lint/type toolchain and the incremental-typing
policy. This section is only for what those two don't say — the things that are
easy to get wrong here.

### Where work lands

Branch off `development`; open the PR against `development` in
**`gently-project/gently`**. Only the push differs by access level:

```bash
# With write access — branches go straight to the org repo
git clone git@github.com:gently-project/gently.git && cd gently
git checkout -b feature/<thing> origin/development
git push -u origin feature/<thing>

# Without write access — fork on GitHub, then keep the org repo as `upstream`
git clone git@github.com:<your-user>/gently.git && cd gently
git remote add upstream git@github.com:gently-project/gently.git
git fetch upstream
git checkout -b feature/<thing> upstream/development
git push -u origin feature/<thing>          # your fork

# Either way
gh pr create --repo gently-project/gently --base development
```

Check `git remote -v` before pushing: in a direct clone `origin` *is* the org
repo; in a fork-based clone `origin` is your fork and `upstream` is the org repo.
Two `gh` defaults will misfile a PR — the repo's default branch is `main` while
PRs target `development`, and with no `gh` default repo set it resolves from
whichever remote it finds. Pass `--base development --repo gently-project/gently`.

### Before committing

CONTRIBUTING.md is canonical for the toolchain. The trap it doesn't mention:
**the pre-commit hook is not installed in a fresh clone**, so commits silently
bypass ruff and mypy. Run `pre-commit install` once per clone, then
`pre-commit run --all-files` before opening a PR.

Non-obvious CI behaviour, all in `.github/workflows/lint.yml`:

- The `lint` job runs its steps **in order and stops at the first failure**, so a
  ruff error hides whether mypy would have passed. A green run after fixing ruff
  is not evidence that mypy ever ran.
- `mypy-strict` (`uv run mypy .`, real deps) is a **separate non-blocking job**
  (`continue-on-error: true`). Only the deps-less `mypy .` inside `lint` gates a
  PR — a green `mypy-strict` proves nothing about the required check.
- Lint runs on `pull_request` and on pushes to `main`/`development` only. A
  feature branch with no PR open gets **no CI at all**; the first signal arrives
  when the PR is opened, against the whole accumulated diff.
- **JavaScript has no CI coverage.** If you touch `gently/ui/web/static/js/`,
  run `node --test tests/js/` yourself — nothing else will.

### Running the app off-Windows

The storage paths throughout this file (`D:\Gently3\...`) are the Windows
microscope PCs, where `D:` is the dedicated data drive. Off-Windows that default
is **not an absolute path**: it resolves against the cwd and silently creates a
junk directory literally named `D:` with session data inside it
(`gently/settings.py`, tracked as issue #56). Always set an explicit path when
running on Linux or macOS:

```bash
GENTLY_STORAGE_PATH=/tmp/gently-dev uv run python launch_gently.py --no-api --no-auth --no-browser
```

Those three flags are the usual agent/dev combination: no Anthropic key needed,
no login gate, and no browser window. The UI is then on `http://localhost:8080`.

## Storage Architecture (Gently3 — File-Based)

All data lives under `D:\Gently3\` (env: `GENTLY_STORAGE_PATH`). **No SQLite databases.** Everything is human-browsable files.

### Key store classes
- **`FileStore`** (`gently/core/file_store.py`) — replaces `GentlyStore`. Manages sessions, embryos, volumes, projections, predictions, traces. Drop-in API replacement.
- **`FileContextStore`** (`gently/harness/memory/file_store.py`) — replaces `ContextStore` / `agent_mind.db`. Manages campaigns, plans, learnings, observations, agent state. Drop-in API replacement.
- **Root manifest**: `D:\Gently3\gently.yaml` — documents the structure for humans and agents.

### Directory layout
```
D:/Gently3/
  gently.yaml                              # root manifest
  sessions/
    _index.yaml                            # session_id -> folder_name mapping
    {YYYYMMDD}_{HHMM}_{slug}_{id8}/
      session.yaml                         # metadata, config, organism
      session.lock                         # PID + hostname (while active)
      intent.yaml                          # planned vs actual
      timelapse.yaml                       # checkpointed timelapse state
      timeline.jsonl                       # session events
      interaction_log.jsonl                # agent conversation records
      conversation.json                    # Claude API messages (resumption)
      summary.yaml                         # auto-generated stats
      perception_runs.yaml                 # run metadata
      snapshots/{source}_{stem}.tif
      embryos/{embryo_id}/
        embryo.yaml                        # position, calibration, uid
        predictions.jsonl                  # per-timepoint summary
        ground_truth.yaml                  # human annotations
        timelapse.mp4
        volumes/t{NNNN}.tif
        volumes/t{NNNN}.meta.yaml          # shape, dtype, acquired_at, metadata
        projections/t{NNNN}.jpg
        traces/t{NNNN}.json               # complete perception record
  agent/
    state.yaml                             # key-value agent state
    campaigns/{id}_{slug}/
      campaign.yaml
      plan/current.yaml
      plan/history/{YYYYMMDD_HHMM}.yaml
      templates/{name}.yaml
    projects/{id}_{slug}.yaml
    session_intents/{session_id}.yaml
    planned_sessions/{id}.yaml
    learnings/{id}_{slug}.yaml
    observations/{id}_{slug}.yaml
    active/expectations.yaml
    active/watchpoints.yaml
    active/questions.yaml
    embryo_understanding/{uid}.yaml
    ml/pipelines/{id}.yaml
    ml/runs/{id}.yaml
    ml/assessments/{id}.yaml
  logs/
    gently_{YYYYMMDD_HHMMSS}.log
    device_layer_{YYYYMMDD_HHMMSS}.log
  config/
    hardware.yaml
    mesh/...
  incoming/{uuid}.tif                      # transient device staging
```

### Legacy stores (D:\Gently2\ — read-only reference)
The old SQLite-based stores are preserved but no longer written to:
- `gently.db` (GentlyStore) — replaced by FileStore
- `context/agent_mind.db` (ContextStore) — replaced by FileContextStore
- `D:\gently\dataset.db` — legacy benchmarking DB

## Logging

Both the agent and device layer write logs to `D:\Gently3\logs\`:

- **Agent**: `gently_YYYYMMDD_HHMMSS.log` — INFO+ to file, console level configurable via `-v` flag
- **Device layer**: `device_layer_YYYYMMDD_HHMMSS.log` — INFO level

To check logs during a session:
```bash
# Latest agent log
tail -f D:/Gently3/logs/$(ls -t D:/Gently3/logs/gently_*.log | head -1)

# Latest device layer log
tail -f D:/Gently3/logs/$(ls -t D:/Gently3/logs/device_layer_*.log | head -1)

# Filter for errors
grep -E "ERROR|Traceback" D:/Gently3/logs/gently_*.log
```

## Perception

Perception is handled by `gently-perception` (separate repo:
`gently-project/gently-perception` — this is what CI clones and what
`pyproject.toml` expects beside this repo), installed as a pip dependency. The timelapse orchestrator uses `Perceiver()` from `gently_perception` — a self-contained system that loads its own examples and accumulates per-embryo context through sequential calls.

## Device Layer

The device layer runs as a separate process (`python start_device_layer.py`). It communicates with the agent via HTTP. Bluesky plans require ophyd device name kwargs (e.g. `xy_stage='xy_stage'`, `volume_scanner='volume_scanner'`) — these must match the device names registered in `device_factory.py`.

## Desktop App (Tauri)

Gently can run as a Windows desktop app — a thin Tauri (WebView2) shell in
`desktop/` that OWNS the Python backend. It spawns `launch_gently.py --no-browser`,
shows a splash, then renders the live web UI (`http://localhost:8080`) in a native
window. The Python-served web UI stays the single source of truth — no app logic
lives in the shell. Full detail: `desktop/README.md`.

### Build / run
```
cd desktop
npm install        # once — restores the Tauri CLI
npm run dev        # dev: build + launch (spawns the backend for you)
npm run build      # release: NSIS installer under src-tauri/target/release/bundle/
```
Prereqs: Rust (MSVC toolchain), the repo's uv `.venv`, WebView2 runtime (inbox on
Win 11). The **release** exe (`src-tauri/target/release/gently-desktop.exe`) is what
the Desktop shortcut points at; rebuild with `npm run build` to refresh it.

### Key pieces
- `desktop/src-tauri/src/main.rs` — spawns the backend, waits for the server,
  navigates the window to it, owns teardown.
- **No orphans:** the spawned Python (and its device-layer grandchild) run in a
  Windows Job Object with `KILL_ON_JOB_CLOSE`, so quitting/crashing the shell reaps
  the whole tree.
- **No console window:** the backend is spawned with `CREATE_NO_WINDOW` in release
  (the device-layer supervisor does the same); `tauri dev` keeps the console so
  logs stay visible while developing.

### Editing code — what reflects
- **Web UI** (`gently/ui/web/templates`, `static/js`, `static/css`): refresh the
  window (Ctrl+R) — served live by Python, no rebuild.
- **Python backend**: restart it, or run with `--reload`
  (`launch_gently.py --reload`, or `GENTLY_LAUNCH_ARGS="--reload …"`) to auto-restart
  on `gently/*.py` changes, then Ctrl+R. Whole-backend restart — not for live hardware.
- **Rust shell / `tauri.conf.json`**: `npm run dev` auto-rebuilds and relaunches.

### Shell env config (read by `main.rs`)
`GENTLY_HOME` (repo dir), `GENTLY_PYTHON` (interpreter), `GENTLY_LAUNCH_ARGS`
(extra `launch_gently` args), `VIZ_PORT` (default 8080), `GENTLY_DEVICE_LAYER_SCRIPT`.

### Deferred
Bundling the Python env (torch/anthropic/perception) for a redistributable
installer — today's build launches the repo's `.venv`, so it's single-machine.

## Debugging Data Sources

- **Agent logs**: `D:\Gently3\logs\gently_*.log`
- **Device layer logs**: `D:\Gently3\logs\device_layer_*.log`
- **Perception traces**: `D:\Gently3\sessions\{session}\embryos\{embryo}\traces\` — per-timepoint JSON
- **Predictions**: `D:\Gently3\sessions\{session}\embryos\{embryo}\predictions.jsonl`
- **Volume staging**: `D:\Gently3\incoming\`
- **Agent memory**: `D:\Gently3\agent\` — campaigns, learnings, observations (all YAML)
- **Session state**: `D:\Gently3\sessions\{session}\session.yaml`
- **Timelapse state**: `D:\Gently3\sessions\{session}\timelapse.yaml`
