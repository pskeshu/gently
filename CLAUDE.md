# Gently — Microscopy Agent

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

Perception is handled by `gently-perception` (separate repo: `pskeshu/gently-perception`), installed as a pip dependency. The timelapse orchestrator uses `Perceiver()` from `gently_perception` — a self-contained system that loads its own examples and accumulates per-embryo context through sequential calls.

## Device Layer

The device layer runs as a separate process (`python start_device_layer.py`). It communicates with the agent via HTTP. Bluesky plans require ophyd device name kwargs (e.g. `xy_stage='xy_stage'`, `volume_scanner='volume_scanner'`) — these must match the device names registered in `device_factory.py`.

## Debugging Data Sources

- **Agent logs**: `D:\Gently3\logs\gently_*.log`
- **Device layer logs**: `D:\Gently3\logs\device_layer_*.log`
- **Perception traces**: `D:\Gently3\sessions\{session}\embryos\{embryo}\traces\` — per-timepoint JSON
- **Predictions**: `D:\Gently3\sessions\{session}\embryos\{embryo}\predictions.jsonl`
- **Volume staging**: `D:\Gently3\incoming\`
- **Agent memory**: `D:\Gently3\agent\` — campaigns, learnings, observations (all YAML)
- **Session state**: `D:\Gently3\sessions\{session}\session.yaml`
- **Timelapse state**: `D:\Gently3\sessions\{session}\timelapse.yaml`
