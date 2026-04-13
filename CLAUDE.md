# Gently — Microscopy Agent

## Logging

Both the agent and device layer write logs to `D:\Gently2\logs\`:

- **Agent**: `gently_YYYYMMDD_HHMMSS.log` — INFO+ to file, console level configurable via `-v` flag
- **Device layer**: `device_layer_YYYYMMDD_HHMMSS.log` — INFO level

To check logs during a session:
```bash
# Latest agent log
tail -f D:/Gently2/logs/$(ls -t D:/Gently2/logs/gently_*.log | head -1)

# Latest device layer log
tail -f D:/Gently2/logs/$(ls -t D:/Gently2/logs/device_layer_*.log | head -1)

# Filter for errors
grep -E "ERROR|Traceback" D:/Gently2/logs/gently_*.log
```

## Perception

Perception is handled by `gently-perception` (separate repo: `pskeshu/gently-perception`), installed as a pip dependency. The timelapse orchestrator uses `Perceiver()` from `gently_perception` — a self-contained system that loads its own examples and accumulates per-embryo context through sequential calls.

## Device Layer

The device layer runs as a separate process (`python start_device_layer.py`). It communicates with the agent via HTTP. Bluesky plans require ophyd device name kwargs (e.g. `xy_stage='xy_stage'`, `volume_scanner='volume_scanner'`) — these must match the device names registered in `device_factory.py`.

