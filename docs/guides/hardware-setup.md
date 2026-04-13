# Hardware Setup for Labs

Step-by-step guide to connecting gently to a diSPIM microscope system. If you're adapting to different hardware, this guide shows what to implement — see [Build a Plugin](build-a-plugin.md) for creating your own hardware plugin.

## Hardware Requirements

**Microscope:**
- ASI Tiger controller with:
  - XY stage card (`XYStage:XY:31`)
  - Piezo stage card (`PiezoStage:P:34`)
  - Scanner/galvo card (`Scanner:AB:33`)
  - LED card (`LED:X:31`)
- Hamamatsu Flash4 camera (lightsheet imaging, `HamCam1`)
- PCO camera (bottom/transmitted light, `Bottom PCO`)
- 488nm and 561nm lasers

**Software:**
- [Micro-Manager 1.4](https://micro-manager.org/) with a working MMConfig file
- Python 3.11+
- Node.js 18+

**Network:**
- The device layer and agent communicate over HTTP (default: `localhost:60610`)
- For multi-instrument setups, both must be reachable on the network

## Micro-Manager Configuration

Gently uses Micro-Manager's `pymmcore` bindings to control hardware directly — no separate Micro-Manager GUI process is needed.

### Verify your MMConfig

1. Open Micro-Manager GUI and confirm all devices initialize
2. Test that XY stage, piezo, cameras, and lasers respond
3. Note the path to your MMConfig file (e.g., `MMConfig_tracking_screening.cfg`)

### Update config.yml

```yaml
# config/config.yml
organism: "celegans"
hardware: "dispim"
mmconfig: "MMConfig_tracking_screening.cfg"
mmdirectory: "C:/Program Files/Micro-Manager-1.4"
```

The `mmdirectory` path is where Micro-Manager (and its device adapters) are installed.

## Start the Device Layer

The device layer is a standalone HTTP server that wraps your microscope hardware:

```bash
python start_device_layer.py
```

### What it does (five-step initialization)

1. **Load configuration** — reads `config/config.yml` for MM paths and settings
2. **Initialize MMCore** — creates a `pymmcore.CMMCore()` instance and loads your MMConfig
3. **Create Ophyd devices** — wraps each hardware component (stage, piezo, scanner, cameras, LED, lasers) in Ophyd-compatible device classes
4. **Initialize Bluesky RunEngine** — the execution engine for acquisition plans
5. **Start HTTP server** — exposes device control, acquisition plans, and SAM detection

### Options

```bash
python start_device_layer.py --port 60610           # HTTP port (default: 60610)
python start_device_layer.py --sam-device cuda       # SAM on GPU (default)
python start_device_layer.py --sam-device cpu         # SAM on CPU (slower)
python start_device_layer.py --config path/to/config.yml
```

### Verify it's running

```bash
# Check device status
curl http://localhost:60610/api/status

# List connected devices
curl http://localhost:60610/api/devices

# Check camera exposure
curl http://localhost:60610/api/camera/exposure
```

## Launch the Agent

In a separate terminal:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
python launch_gently.py
```

The agent connects to the device layer at `http://{GENTLY_DEVICE_HOST}:{GENTLY_DEVICE_PORT}` (defaults: `127.0.0.1:60610`).

## Device Limits and Safety

Every device has hard bounds validated before any motion:

| Device | Parameter | Limits |
|--------|-----------|--------|
| XY Stage | X position | 500–2500 μm |
| XY Stage | Y position | −1000–1000 μm |
| Piezo | Z position | ±200 μm from center |
| Scanner/Galvo | Angle | ±5° (both axes) |

These are enforced at the Python level in each device's `set()` method. Out-of-range commands raise a `ValueError` before reaching hardware.

### Laser and LED safety

- Lasers are enabled only during active acquisition and disabled in a `finally` block — any error automatically shuts them off
- LED (transmitted light) follows the same pattern for bottom camera captures
- This prevents photobleaching from stuck-on illumination during troubleshooting

### Process isolation

The device layer runs as a separate process from the agent. If the agent crashes, the microscope state is unaffected. The HTTP API boundary means the agent can never issue raw hardware commands — only call predefined, validated endpoints.

## First Acquisition Walkthrough

Once both the device layer and agent are running:

1. **Detect embryos**: "Detect embryos" — the agent captures a bottom camera image and uses SAM to segment embryo positions
2. **Review detections**: "Show detected embryos" — displays labeled positions
3. **Calibrate**: "Calibrate embryo 1" — runs a piezo-galvo calibration sweep to find optimal imaging parameters
4. **Acquire**: "Acquire a volume of embryo 1" — captures a single 3D lightsheet stack
5. **Classify**: "What stage is embryo 1?" — runs VLM perception on the acquired volume
6. **Start timelapse**: "Start a timelapse of all embryos every 2 minutes" — begins adaptive multi-embryo imaging

## Camera Modes

The Hamamatsu camera operates in two modes, switched automatically:

| Mode | Trigger | Sensor | Use |
|------|---------|--------|-----|
| Calibration | Internal (software) | Area (full frame) | Single snapshots, focus sweeps |
| SPIM Volume | External (Tiger TTL) | Progressive (rolling shutter) | Volume acquisition |

Rolling shutter mode is critical for SPIM — it synchronizes the camera readout with the light sheet position for optical sectioning.

## Diagnostic Endpoints

The device layer exposes status information:

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/status` | GET | Device list, queue size, SAM status |
| `/api/devices` | GET | Connected hardware details |
| `/api/plans` | GET | Available acquisition plans |
| `/api/plan_log` | GET | Execution history with timing |
| `/api/camera/exposure` | GET/POST | Camera exposure time |
| `/api/led/status` | GET | LED state |
| `/api/led/set` | POST | LED control |
| `/api/detect_embryos` | POST | SAM detection + image capture |

## Troubleshooting

### Device layer won't start

- **"MMConfig not found"**: Check `mmdirectory` in `config/config.yml` — it should point to where Micro-Manager is installed
- **"Device not found"**: Verify device names in your MMConfig match the expected names (`XYStage:XY:31`, `PiezoStage:P:34`, etc.). Check ASI Tiger firmware and serial connection
- **"Port already in use"**: Another process is using port 60610. Use `--port` to change, and set `GENTLY_DEVICE_PORT` to match

### Agent can't connect

- **"Connection refused"**: Device layer isn't running, or host/port mismatch. Check `GENTLY_DEVICE_HOST` and `GENTLY_DEVICE_PORT`
- **Timeout errors**: The device layer may still be initializing. It takes 10–30 seconds for all devices + SAM model to load

### Acquisition issues

- **Blank images**: Check laser enable state. Verify camera trigger mode matches acquisition type (internal for calibration, external for SPIM)
- **Noisy/dim images**: Exposure may be too low. "Set exposure to 20ms" or adjust via the API
- **Stage movement errors**: Position is outside device limits. Check the limits table above
- **Piezo calibration fails**: The scan range may exceed piezo limits (±200 μm). Reduce `num_slices` or amplitude

### TUI won't launch

- **"TUI not available"**: Run `cd gently/tui && npm install && npm run build`
- **Node not found**: Install Node.js 18+ and ensure `node` is on your PATH

## Environment Variables

| Variable | Default | Purpose |
|----------|---------|---------|
| `GENTLY_DEVICE_HOST` | `127.0.0.1` | Device layer hostname |
| `GENTLY_DEVICE_PORT` | `60610` | Device layer port |
| `GENTLY_STORAGE_PATH` | `D:/Gently3` | Session and data storage directory |
| `GENTLY_VIZ_PORT` | `8080` | Web visualization server port |
| `GENTLY_VIZ_HOST` | `0.0.0.0` | Visualization bind address |

## Next Steps

- [What Gently Can Do](capabilities.md) — full capabilities overview
- [Hardware Safety Checklist](../hardware_safety_checklist.md) — detailed safety procedures
- [Piezo Scanning Technical Report](../asidispim_piezo_scanning_technical_report.md) — deep dive into piezo-galvo synchronization
- [Camera Triggering](../asidispim_camera_triggering.md) — detailed camera triggering reference
