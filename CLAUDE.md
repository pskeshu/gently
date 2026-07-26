# Gently — Microscopy Agent

## Working on this repo

**README.md** covers environment setup and how to run things (`uv sync`,
`uv run pytest`, `uv run python launch_gently.py` and its flags).
**CONTRIBUTING.md** covers the lint/type toolchain — ruff, the two mypy runs,
and pre-commit. This section is only for what those two don't say — the things
that are easy to get wrong here.

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

`.github/workflows/lint.yml` is the source of truth for which checks gate a PR,
and that changes — read it rather than trusting a list here. What stays true:

- **There are two mypy runs and they can disagree.** One runs `mypy .` with no
  project deps, so third-party imports fall back to `Any`; the other runs
  `uv run mypy .` with the real packages and resolves their actual types. A
  green run of one says nothing about the other. `uv sync && uv run mypy .`
  reproduces the deps-installed run locally; the pre-commit hook reproduces the
  deps-less one.
- **A job stops at its first failing step.** A ruff failure means the mypy step
  in that job never ran, so a green re-run after fixing ruff is not evidence
  that mypy passed.
- **CI runs no JavaScript.** A change under `gently/ui/web/static/js/` is not
  covered by CI, so verify it by running the app and exercising the UI by hand.
- CI runs on pull requests and on pushes to `main`/`development`, so a feature
  branch with no PR open gets no signal at all.

### Running the app off-Windows

The storage paths throughout this file (`D:\Gently3\...`) are the Windows
microscope PCs, where `D:` is the dedicated data drive. Off-Windows that default
is **not an absolute path**: it resolves against the cwd and silently creates a
junk directory literally named `D:` with session data inside it
(`gently/settings.py`). Always set an explicit path when running on Linux or
macOS:

```bash
GENTLY_STORAGE_PATH=/tmp/gently-dev uv run python launch_gently.py --no-api --no-auth --no-browser
```

Those three flags are the usual agent/dev combination: no Anthropic key needed,
no login gate, and no browser window. The UI is then on `http://localhost:8080`.

## Storage Architecture (Gently3 — File-Based)

All data lives under `D:\Gently3\` (env: `GENTLY_STORAGE_PATH`). **No SQLite databases.** Everything is human-browsable files.

### Key store classes
- **`FileStore`** (`gently/core/file_store.py`) — sessions, embryos, volumes, projections, predictions, traces.
- **`FileContextStore`** (`gently/harness/memory/file_store.py`) — campaigns, plans, learnings, observations, agent state.
- **Root manifest**: `gently.yaml` at the storage root — documents the structure for humans and agents.

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

### Superseded stores — do not wire new code to these

The SQLite-era classes are **still in the package and still have passing tests**,
so they look live. They have no production callers; only `tests/` instantiate
them. Use the file stores above instead.

- `GentlyStore` (`gently/core/store.py`) → use `FileStore`
- `ContextStore` (`gently/harness/memory/store.py`, `agent_mind.db`) → use `FileContextStore`
- `gently/dataset/` still defaults to `D:/gently/dataset.db` (legacy benchmarking DB)

Their data lives under the old `D:\Gently2\` root, read-only reference only.

## Logging

Both the agent and device layer write logs to `<storage root>/logs/`:

- **Agent**: `gently_YYYYMMDD_HHMMSS.log` — INFO+ to file, console level configurable via `-v` flag
- **Device layer**: `device_layer_YYYYMMDD_HHMMSS.log` — INFO level

To check logs during a session (the expansion keeps these working off-Windows,
where the `D:/Gently3` default does not apply — see above):

```bash
LOGS="${GENTLY_STORAGE_PATH:-D:/Gently3}/logs"

# Latest agent log  (ls prints the full path — do not prefix $LOGS again)
tail -f "$(ls -t "$LOGS"/gently_*.log | head -1)"

# Latest device layer log
tail -f "$(ls -t "$LOGS"/device_layer_*.log | head -1)"

# Filter for errors
grep -E "ERROR|Traceback" "$LOGS"/gently_*.log
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
