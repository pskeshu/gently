"""Data routes - calibration, snapshots, embryos, sequence, status, events."""

import logging
from datetime import datetime
from pathlib import Path

import yaml
from fastapi import APIRouter, Body, Depends, HTTPException

from gently.ui.web.auth import require_control

logger = logging.getLogger(__name__)

# config/hardware.yaml at the repo root, four levels up from this file:
# gently/ui/web/routes/data.py -> gently/ui/web/routes -> ... -> repo root
_HARDWARE_CONFIG_PATH = Path(__file__).resolve().parents[4] / "config" / "hardware.yaml"


def _json_safe(obj):
    """Make an acquisition result JSON-encodable for FastAPI.

    ``client.acquire_volume``/``acquire_burst`` return the pixel data under
    ``volume``/``image`` as numpy arrays (internal callers like the timelapse
    orchestrator need them). FastAPI's ``jsonable_encoder`` can't serialize a
    raw ndarray — it tries ``dict(arr)`` and blows up — and the web UI only
    needs paths + metadata anyway. Replace arrays with a small shape/dtype
    hint and coerce numpy scalars to native types; recurse so the burst
    ``frames`` list is covered too.
    """
    import numpy as np

    if isinstance(obj, np.ndarray):
        return {"shape": list(obj.shape), "dtype": str(obj.dtype)}
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_safe(v) for v in obj]
    return obj


def create_router(server) -> APIRouter:
    router = APIRouter()

    @router.get("/api/status")
    async def get_status():
        """Get server status"""
        stats = server.store.get_stats()
        return {
            "status": "running",
            "connections": len(server.manager.active_connections),
            **stats,
            "timestamp": datetime.now().isoformat(),
        }

    @router.get("/api/device-status")
    async def get_device_status():
        """Get device connection status from agent bridge.

        Actively pings the device layer via client.health_check() to
        avoid reporting a stale "online" state after the device layer
        process has been killed. client.is_connected is a cached bool
        that only updates on connect() / explicit RPC failures, so
        without the active ping the UI would keep showing the microscope
        as online indefinitely after a hardware-side shutdown.
        """
        bridge = getattr(server, "agent_bridge", None)
        if bridge is None:
            return {"gently": True, "microscope": False}

        client = getattr(bridge.agent, "client", None) if bridge.agent else None
        if client is None:
            return {"gently": True, "microscope": False}

        # Prefer a live health check; fall back to cached state if the
        # client doesn't implement health_check (older versions).
        if hasattr(client, "health_check"):
            try:
                microscope_up = await client.health_check(timeout=2.0)
            except Exception:
                microscope_up = False
        else:
            microscope_up = getattr(client, "is_connected", False)

        return {
            "gently": True,
            "microscope": microscope_up,
        }

    def _require_agent_with_experiment():
        """Resolve the live agent from the server bridge, or 503.

        Edit endpoints write through ExperimentState so the notify hook fires
        EMBRYOS_UPDATE and the Map re-renders without a follow-up fetch.
        """
        bridge = getattr(server, "agent_bridge", None)
        agent = bridge.agent if bridge is not None else None
        if agent is None or not hasattr(agent, "experiment"):
            raise HTTPException(status_code=503, detail="Agent not ready")
        return agent

    @router.put("/api/embryos/{embryo_id}/position", dependencies=[Depends(require_control)])
    async def update_embryo_position(
        embryo_id: str,
        body: dict = Body(...),  # noqa: B008
    ):
        """Update an embryo's coarse XY position.

        Map-side edits write to the coarse stage and CLEAR any prior fine
        position — the operator is overriding the sighting, so any
        SPIM-objective fine alignment derived from the old coarse is no
        longer trustworthy and must be re-run.

        Publishes OPERATOR_EDITED_EMBRYO with both the old and new
        positions so candidates can reason about the magnitude of the
        correction and trigger re-calibration suggestions.
        """
        agent = _require_agent_with_experiment()
        emb = agent.experiment.embryos.get(embryo_id)
        if emb is None:
            raise HTTPException(status_code=404, detail=f"Embryo {embryo_id} not found")
        try:
            x = float(body.get("x"))  # type: ignore[arg-type]  # None -> TypeError caught below
            y = float(body.get("y"))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail="Body needs numeric x and y") from None
        old_coarse = dict(emb.position_coarse) if emb.position_coarse else None
        had_fine = bool(emb.position_fine)
        emb.position_coarse = {"x": x, "y": y}
        emb.position_fine = {}
        agent.experiment.notify_embryos_changed()

        bus = getattr(agent, "_event_bus", None)
        if bus is not None:
            from gently.core.event_bus import EventType

            try:
                bus.publish(
                    event_type=EventType.OPERATOR_EDITED_EMBRYO,
                    data={
                        "embryo_id": embryo_id,
                        "old_position_coarse": old_coarse,
                        "new_position_coarse": {"x": x, "y": y},
                        "fine_position_invalidated": had_fine,
                    },
                    source="web:map-edit",
                )
            except Exception:
                logger.exception("Failed to publish OPERATOR_EDITED_EMBRYO")
        return emb.to_dict()

    @router.delete("/api/embryos/{embryo_id}", dependencies=[Depends(require_control)])
    async def delete_embryo(embryo_id: str):
        """Remove an embryo from the experiment.

        Goes through ExperimentState.remove_embryo so the observer hook
        fires EMBRYOS_UPDATE automatically. Also publishes
        OPERATOR_REMOVED_EMBRYO carrying the embryo's last known position
        — candidates can use that to e.g. clean up associated cache or
        log the deletion in their own world model.
        """
        agent = _require_agent_with_experiment()
        emb = agent.experiment.embryos.get(embryo_id)
        last_position = None
        if emb is not None:
            last_position = {
                "coarse": dict(emb.position_coarse) if emb.position_coarse else None,
                "fine": dict(emb.position_fine) if emb.position_fine else None,
            }
        if not agent.experiment.remove_embryo(embryo_id):
            raise HTTPException(status_code=404, detail=f"Embryo {embryo_id} not found")

        # Also drop it from the session files, or a false positive deleted here
        # would reappear on the next restart (embryos are reloaded from disk on
        # resume). Best-effort — the in-memory removal already succeeded.
        store = getattr(agent, "store", None)
        sid = getattr(agent, "session_id", None)
        if store is not None and sid and hasattr(store, "delete_embryo"):
            try:
                store.delete_embryo(sid, embryo_id)
            except Exception:
                logger.exception("Failed to delete embryo %s from session files", embryo_id)

        bus = getattr(agent, "_event_bus", None)
        if bus is not None:
            from gently.core.event_bus import EventType

            try:
                bus.publish(
                    event_type=EventType.OPERATOR_REMOVED_EMBRYO,
                    data={
                        "embryo_id": embryo_id,
                        "last_position": last_position,
                    },
                    source="web:map-delete",
                )
            except Exception:
                logger.exception("Failed to publish OPERATOR_REMOVED_EMBRYO")
        return {"ok": True, "embryo_id": embryo_id}

    @router.get("/api/embryos/current")
    async def get_current_embryos():
        """Return the agent's current embryo list as an EMBRYOS_UPDATE payload.

        EMBRYOS_UPDATE is published only on mutation, so a Map page opened
        mid-session would otherwise see an empty embryo layer until the next
        add/remove/edit. This endpoint serves the same payload shape as the
        event so clients can bootstrap and then switch to the live stream.
        """
        empty = {"embryos": [], "count": 0, "session_id": None}
        bridge = getattr(server, "agent_bridge", None)
        agent = bridge.agent if bridge is not None else None
        if agent is None or not hasattr(agent, "experiment"):
            return empty
        try:
            embryos = [e.to_dict() for e in agent.experiment.embryos.values()]
        except Exception:
            logger.exception("Failed to serialise embryos for snapshot")
            return empty
        return {
            "embryos": embryos,
            "count": len(embryos),
            "session_id": getattr(agent, "session_id", None),
        }

    @router.get("/api/devices/coverslip")
    async def get_coverslip():
        """Return the coverslip outline metadata for the Map view.

        Read fresh from config/hardware.yaml so the slide dimensions can be
        retuned without restarting the agent. XY safety zones live elsewhere
        (XY_STAGE_*_UM constants in stage.py → ASI firmware fence) and the
        frontend reads them from the live device-state stream — no zone
        block in this config and no zone endpoint here.
        """
        try:
            with open(_HARDWARE_CONFIG_PATH) as f:
                cfg = yaml.safe_load(f) or {}
        except FileNotFoundError:
            return {"coverslip": None}
        cs = cfg.get("coverslip")
        if not isinstance(cs, dict):
            return {"coverslip": None}
        return {
            "coverslip": {
                "center_um": list(cs.get("center_um") or [0.0, 0.0]),
                "size_mm": list(cs.get("size_mm") or [50.0, 24.0]),
            }
        }

    @router.get("/api/devices/scan_geometry")
    async def get_scan_geometry():
        """Return the most recent scan geometry for the 3D optical-space view.

        SCAN_GEOMETRY_UPDATE is published only when a volume is acquired, so a
        page opened before the first acquisition would have no cuboid to draw.
        This serves the last emitted payload (stashed on the agent by
        acquisition_tools._publish_scan_geometry), or nominal defaults so the
        scene is never empty.
        """
        bridge = getattr(server, "agent_bridge", None)
        agent = bridge.agent if bridge is not None else None
        last = getattr(agent, "last_scan_geometry", None) if agent else None
        if isinstance(last, dict):
            return last
        # Nominal defaults (calibration defaults; no acquisition yet).
        num_slices = 50
        piezo_amplitude = 25.0
        piezo_center = 50.0
        z_extent = 2.0 * piezo_amplitude
        return {
            "embryo_id": None,
            "stage_position_um": {"x": None, "y": None},
            "scan": {
                "num_slices": num_slices,
                "exposure_ms": 10.0,
                "galvo_amplitude_deg": 0.5,
                "galvo_center_deg": 0.0,
                "piezo_amplitude_um": piezo_amplitude,
                "piezo_center_um": piezo_center,
            },
            "derived": {
                "z_extent_um": z_extent,
                "slice_spacing_um": z_extent / (num_slices - 1),
                "z_min_um": piezo_center - piezo_amplitude,
                "z_max_um": piezo_center + piezo_amplitude,
            },
            "mode": "sheet",
            "ts": None,
            "is_default": True,
        }

    @router.get("/api/devices/bottom_camera/status")
    async def get_bottom_camera_status():
        """Return whether the bottom-camera stream bridge is running."""
        bridge = getattr(server, "agent_bridge", None)
        agent = bridge.agent if bridge is not None else None
        monitor = getattr(agent, "bottom_camera_monitor", None) if agent else None
        return {
            "available": monitor is not None,
            "streaming": bool(monitor and monitor.running),
            "last_frame_ts": getattr(monitor, "_last_frame_ts", None) if monitor else None,
        }

    @router.post(
        "/api/devices/bottom_camera/stream/start",
        dependencies=[Depends(require_control)],
    )
    async def start_bottom_camera_stream():
        """Start the bottom-camera stream bridge.

        Idempotent — calling start() while already running is a no-op.
        """
        bridge = getattr(server, "agent_bridge", None)
        agent = bridge.agent if bridge is not None else None
        monitor = getattr(agent, "bottom_camera_monitor", None) if agent else None
        if monitor is None:
            raise HTTPException(
                status_code=503,
                detail="Bottom-camera monitor not initialised (agent or microscope not ready)",
            )
        try:
            await monitor.start()
        except Exception as exc:
            logger.exception("Failed to start bottom-camera monitor")
            raise HTTPException(status_code=500, detail=f"start failed: {exc}") from exc
        return {"streaming": monitor.running}

    @router.post(
        "/api/devices/bottom_camera/stream/stop",
        dependencies=[Depends(require_control)],
    )
    async def stop_bottom_camera_stream():
        """Stop the bottom-camera stream bridge. Idempotent."""
        bridge = getattr(server, "agent_bridge", None)
        agent = bridge.agent if bridge is not None else None
        monitor = getattr(agent, "bottom_camera_monitor", None) if agent else None
        if monitor is None:
            return {"streaming": False}
        try:
            await monitor.stop()
        except Exception as exc:
            logger.exception("Failed to stop bottom-camera monitor")
            raise HTTPException(status_code=500, detail=f"stop failed: {exc}") from exc
        return {"streaming": False}

    def _resolve_client():
        """Resolve the live microscope client from the agent bridge, or None."""
        bridge = getattr(server, "agent_bridge", None)
        agent = bridge.agent if bridge is not None else None
        return getattr(agent, "client", None) if agent else None

    @router.get("/api/devices/room_light/status")
    async def get_room_light_status():
        """Cached on/off state of the room-light SwitchBot (cheap to poll)."""
        client = _resolve_client()
        if client is None:
            return {"available": False, "state": "unknown"}
        try:
            res = await client.get_room_light_status()
        except Exception as exc:
            logger.debug("room light status fetch failed: %s", exc)
            return {"available": False, "state": "unknown"}
        return {
            "available": bool(res.get("available", res.get("success", False))),
            "state": res.get("state", "unknown"),
        }

    @router.post("/api/devices/room_light/set", dependencies=[Depends(require_control)])
    async def set_room_light(payload: dict = Body(...)):  # noqa: B008
        """Switch the room light on/off. Body: {"state": "on"|"off"|"press"}."""
        state = str(payload.get("state", "")).lower()
        if state not in ("on", "off", "press"):
            raise HTTPException(status_code=400, detail="state must be on, off, or press")
        client = _resolve_client()
        if client is None:
            raise HTTPException(status_code=503, detail="Microscope not connected")
        try:
            res = await client.set_room_light(state)
        except Exception as exc:
            logger.exception("Room light command failed")
            raise HTTPException(
                status_code=502, detail=f"room light command failed: {exc}"
            ) from exc
        if not res.get("success"):
            raise HTTPException(
                status_code=502, detail=res.get("error", "room light command failed")
            )
        return {"state": res.get("state", state)}

    @router.get("/api/devices/temperature/status")
    async def get_temperature_status():
        """Live water temperature, setpoint, and lock state (cheap to poll).

        Cached at the device layer (no per-call hardware round trip), so the
        Devices header can poll it like the room light. ``available`` is false
        when no controller is configured/connected, which hides the control.
        """
        client = _resolve_client()
        if client is None:
            return {"available": False, "state": "unknown"}
        try:
            res = await client.get_temperature()
        except Exception as exc:
            logger.debug("temperature status fetch failed: %s", exc)
            return {"available": False, "state": "unknown"}
        return {
            "available": bool(res.get("success", False)),
            "temperature_c": res.get("temperature_c"),
            "setpoint_c": res.get("setpoint_c"),
            "state": res.get("state", "unknown"),
            "peltier_c": res.get("peltier_c"),
        }

    @router.post("/api/devices/temperature/set", dependencies=[Depends(require_control)])
    async def set_temperature(payload: dict = Body(...)):  # noqa: B008
        """Command the temperature setpoint. Body: {"target_c": float}.

        Non-blocking: the controller ramps and the status poll reflects progress
        (and the SYSTEM LOCKED state once it stabilizes).
        """
        try:
            target = float(payload.get("target_c"))  # type: ignore[arg-type]  # None -> caught
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail="target_c must be a number") from None
        if not (0.0 <= target <= 99.9):
            raise HTTPException(status_code=400, detail="target_c must be between 0.0 and 99.9 C")
        client = _resolve_client()
        if client is None:
            raise HTTPException(status_code=503, detail="Microscope not connected")
        try:
            res = await client.set_temperature(target)
        except Exception as exc:
            logger.exception("Temperature command failed")
            raise HTTPException(
                status_code=502, detail=f"temperature command failed: {exc}"
            ) from exc
        if not res.get("success"):
            raise HTTPException(
                status_code=502, detail=res.get("error", "temperature command failed")
            )
        return {
            "target_c": res.get("target_c", target),
            "temperature_c": res.get("temperature_c"),
            "state": res.get("state", "unknown"),
            "waited": res.get("waited", False),
        }

    @router.get("/api/devices/temperature/config")
    async def get_temperature_config():
        """Thermalizer connection config (password redacted) + live backend/state
        for the Settings panel. Read-only, so no control elevation required."""
        client = _resolve_client()
        if client is None:
            return {"available": False}
        try:
            res = await client.get_temperature_config()
        except Exception as exc:
            logger.debug("temperature config fetch failed: %s", exc)
            return {"available": False}
        return {"available": bool(res.get("success", False)), **res}

    @router.post("/api/devices/temperature/config/test", dependencies=[Depends(require_control)])
    async def test_temperature_config(payload: dict = Body(...)):  # noqa: B008
        """Probe a candidate thermalizer config without committing it."""
        client = _resolve_client()
        if client is None:
            raise HTTPException(status_code=503, detail="Microscope not connected")
        try:
            res = await client.test_temperature_config(payload)
        except Exception as exc:
            logger.exception("Thermalizer test failed")
            raise HTTPException(status_code=502, detail=f"thermalizer test failed: {exc}") from exc
        return res

    @router.post("/api/devices/temperature/config", dependencies=[Depends(require_control)])
    async def set_temperature_config(payload: dict = Body(...)):  # noqa: B008
        """Reconfigure the thermalizer (serial/mqtt/mock). Live hot-swap where
        possible; otherwise persisted for the next device-layer restart."""
        client = _resolve_client()
        if client is None:
            raise HTTPException(status_code=503, detail="Microscope not connected")
        try:
            res = await client.set_temperature_config(payload)
        except Exception as exc:
            logger.exception("Thermalizer reconfigure failed")
            raise HTTPException(
                status_code=502, detail=f"thermalizer reconfigure failed: {exc}"
            ) from exc
        return res

    @router.get("/api/config/effective")
    async def get_effective_config():
        """Read-only view of the effective server config, secrets redacted.

        settings.py values are frozen at import (resolved from env vars once), so
        changing them requires a process restart — surfaced here for visibility,
        not editing. Secrets are shown only as present/absent booleans.
        """
        import os

        from gently.settings import settings as S

        return {
            "note": "settings.py values are read from env at startup; "
            "changing them needs a restart.",
            "network": {
                "viz_host": S.network.viz_host,
                "viz_port": S.network.viz_port,
                "device_host": S.network.device_host,
                "device_port": S.network.device_port,
                "mesh_port": S.network.mesh_port,
            },
            "models": {
                "main": S.models.main,
                "perception": S.models.perception,
                "fast": S.models.fast,
                "medium": S.models.medium,
                "refusal_fallback": S.models.refusal_fallback,
            },
            "storage": {"base_path": str(S.storage.base_path)},
            "timeouts": {
                "plan_execution": S.timeouts.plan_execution,
                "volume_acquisition": S.timeouts.volume_acquisition,
                "api_call": S.timeouts.api_call,
            },
            "ml": {
                "default_batch_size": S.ml.default_batch_size,
                "default_epochs": S.ml.default_epochs,
                "default_lr": S.ml.default_lr,
                "model_cache_dir": str(S.ml.model_cache_dir),
            },
            "transfer": {
                "transfer_port": S.transfer.transfer_port,
                "chunk_size": S.transfer.chunk_size,
                "max_concurrent_transfers": S.transfer.max_concurrent_transfers,
            },
            "mesh": {
                "broadcast_interval_s": S.mesh.broadcast_interval_s,
                "stale_threshold_s": S.mesh.stale_threshold_s,
                "dead_threshold_s": S.mesh.dead_threshold_s,
            },
            "ui": {"ux_v2": S.ui.ux_v2},
            "api": {"ncbi_tool": S.api.ncbi_tool},
            "secrets_present": {
                "anthropic_api_key": bool(os.getenv("ANTHROPIC_API_KEY")),
                "control_token": bool(os.getenv("GENTLY_CONTROL_TOKEN")),
            },
        }

    # --- Rig-wide dashboard-preference defaults (layered UNDER per-browser localStorage) ---
    _dashboard_defaults_path = _HARDWARE_CONFIG_PATH.parent / "dashboard_defaults.json"

    @router.get("/api/config/dashboard-defaults")
    async def get_dashboard_defaults():
        """Rig-wide dashboard-pref defaults (JSON). The browser layers localStorage
        over these, so a fresh browser inherits the rig's defaults."""
        import json

        if not _dashboard_defaults_path.exists():
            return {}
        try:
            return json.loads(_dashboard_defaults_path.read_text())
        except Exception:
            return {}

    @router.put("/api/config/dashboard-defaults", dependencies=[Depends(require_control)])
    async def put_dashboard_defaults(payload: dict = Body(...)):  # noqa: B008
        """Save the current dashboard prefs as the rig-wide defaults."""
        import json

        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="body must be an object")
        _dashboard_defaults_path.write_text(json.dumps(payload, indent=2))
        return {"saved": True}

    # --- Restart-required settings.py editors (persisted to config/settings.local.yml) ---
    # Allowlist of knobs that are ACTUALLY consumed by the runtime (verified live
    # readers) + grouped for the UI. Never expose ports/hosts/model-IDs/storage/
    # secrets. Deliberately omitted: timeouts.rpc_call (removed — RPyC-era dead),
    # timeouts.plan_execution and ml.* defaults (currently 0 readers — editing
    # would be a silent no-op).
    _override_keys = [
        {
            "env": "GENTLY_TIMEOUT_VOLUME",
            "label": "Volume acquisition (s)",
            "type": "int",
            "group": "Timeouts",
            "get": lambda S: S.timeouts.volume_acquisition,
        },
        {
            "env": "GENTLY_TIMEOUT_API",
            "label": "External API call (s)",
            "type": "int",
            "group": "Timeouts",
            "get": lambda S: S.timeouts.api_call,
        },
        {
            "env": "GENTLY_MESH_BROADCAST_INTERVAL",
            "label": "Broadcast interval (s)",
            "type": "float",
            "group": "Mesh network",
            "get": lambda S: S.mesh.broadcast_interval_s,
        },
        {
            "env": "GENTLY_MESH_STALE_THRESHOLD",
            "label": "Stale threshold (s)",
            "type": "float",
            "group": "Mesh network",
            "get": lambda S: S.mesh.stale_threshold_s,
        },
        {
            "env": "GENTLY_MESH_DEAD_THRESHOLD",
            "label": "Dead threshold (s)",
            "type": "float",
            "group": "Mesh network",
            "get": lambda S: S.mesh.dead_threshold_s,
        },
        {
            "env": "GENTLY_UX_V2",
            "label": "UX v2 dashboard",
            "type": "bool",
            "group": "Interface",
            "get": lambda S: S.ui.ux_v2,
        },
        {
            "env": "GENTLY_NCBI_TOOL",
            "label": "Tool name",
            "type": "str",
            "group": "NCBI (Entrez)",
            "get": lambda S: S.api.ncbi_tool,
        },
        {
            "env": "GENTLY_NCBI_EMAIL",
            "label": "Contact email",
            "type": "str",
            "group": "NCBI (Entrez)",
            "get": lambda S: S.api.ncbi_email,
        },
    ]
    _settings_local_path = _HARDWARE_CONFIG_PATH.parent / "settings.local.yml"

    def _coerce_override(typ, val):
        if typ == "int":
            return int(val)
        if typ == "float":
            return float(val)
        if typ == "bool":
            return val if isinstance(val, bool) else str(val).lower() in ("1", "true", "yes", "on")
        return str(val)

    def _read_settings_local():
        if not _settings_local_path.exists():
            return {}
        try:
            return yaml.safe_load(_settings_local_path.read_text()) or {}
        except Exception:
            return {}

    @router.get("/api/config/settings-overrides")
    async def get_settings_overrides():
        """Editable (restart-required) settings.py knobs: current effective value
        + whether an override file entry exists."""
        from gently.settings import settings as S

        file_over = _read_settings_local()
        items = [
            {
                "env": k["env"],
                "label": k["label"],
                "type": k["type"],
                "group": k["group"],
                "current": k["get"](S),
                "overridden": k["env"] in file_over,
            }
            for k in _override_keys
        ]
        return {"note": "changes take effect on the next process restart", "items": items}

    @router.put("/api/config/settings-overrides", dependencies=[Depends(require_control)])
    async def put_settings_overrides(payload: dict = Body(...)):  # noqa: B008
        """Persist restart-required overrides to config/settings.local.yml. Only
        allowlisted keys; never mutates the frozen settings singleton live."""
        allowed = {k["env"]: k["type"] for k in _override_keys}
        updates = {}
        for k, v in (payload or {}).items():
            if k not in allowed:
                raise HTTPException(status_code=400, detail=f"unknown or non-editable key: {k}")
            if v is None or v == "":
                continue
            try:
                updates[k] = _coerce_override(allowed[k], v)
            except (TypeError, ValueError):
                raise HTTPException(
                    status_code=400, detail=f"{k}: invalid {allowed[k]} value"
                ) from None
        existing = _read_settings_local()
        existing.update(updates)
        _settings_local_path.write_text(
            yaml.safe_dump(existing, default_flow_style=False, sort_keys=True)
        )
        return {
            "saved": list(updates.keys()),
            "restart_required": True,
            "note": "restart the server for these to take effect",
        }

    # ------------------------------------------------------------------
    # Lightsheet live stream
    # ------------------------------------------------------------------

    @router.get("/api/devices/lightsheet/live/status")
    async def get_lightsheet_live_status():
        """Return whether the lightsheet live stream bridge is running."""
        bridge = getattr(server, "agent_bridge", None)
        agent = bridge.agent if bridge is not None else None
        monitor = getattr(agent, "lightsheet_monitor", None) if agent else None
        return {
            "available": monitor is not None,
            "streaming": bool(monitor and monitor.running),
            "last_frame_ts": getattr(monitor, "_last_frame_ts", None) if monitor else None,
        }

    @router.post(
        "/api/devices/lightsheet/live/start",
        dependencies=[Depends(require_control)],
    )
    async def start_lightsheet_live_stream():
        """Start the lightsheet live stream bridge.

        Idempotent — calling start() while already running is a no-op.
        """
        bridge = getattr(server, "agent_bridge", None)
        agent = bridge.agent if bridge is not None else None
        monitor = getattr(agent, "lightsheet_monitor", None) if agent else None
        if monitor is None:
            raise HTTPException(
                status_code=503,
                detail="Lightsheet monitor not initialised (agent or microscope not ready)",
            )
        try:
            await monitor.start()
        except Exception as exc:
            logger.exception("Failed to start lightsheet monitor")
            raise HTTPException(status_code=500, detail=f"start failed: {exc}") from exc
        return {"streaming": monitor.running}

    @router.post(
        "/api/devices/lightsheet/live/stop",
        dependencies=[Depends(require_control)],
    )
    async def stop_lightsheet_live_stream():
        """Stop the lightsheet live stream bridge. Idempotent."""
        bridge = getattr(server, "agent_bridge", None)
        agent = bridge.agent if bridge is not None else None
        monitor = getattr(agent, "lightsheet_monitor", None) if agent else None
        if monitor is None:
            return {"streaming": False}
        try:
            await monitor.stop()
        except Exception as exc:
            logger.exception("Failed to stop lightsheet monitor")
            raise HTTPException(status_code=500, detail=f"stop failed: {exc}") from exc
        return {"streaming": False}

    # ------------------------------------------------------------------
    # Lightsheet live params
    # ------------------------------------------------------------------

    @router.post("/api/devices/lightsheet/live/params", dependencies=[Depends(require_control)])
    async def lightsheet_live_params(payload: dict = Body(...)):  # noqa: B008
        """Forward galvo/piezo/exposure/side params to the device-layer lightsheet streamer."""
        client = _resolve_client()
        if client is None:
            raise HTTPException(status_code=503, detail="Microscope not connected")
        try:
            res = await client.set_lightsheet_live_params(
                galvo=payload.get("galvo"),
                piezo=payload.get("piezo"),
                exposure=payload.get("exposure"),
                side=payload.get("side"),
            )
        except Exception as exc:
            logger.exception("lightsheet live params failed")
            raise HTTPException(status_code=502, detail=f"params failed: {exc}") from exc
        return res

    # ------------------------------------------------------------------
    # LED / laser / camera
    # ------------------------------------------------------------------

    @router.post("/api/devices/led/set", dependencies=[Depends(require_control)])
    async def led_set(payload: dict = Body(...)):  # noqa: B008
        """Set the LED shutter state. Body: {"state": "Open"|"Closed"}."""
        client = _resolve_client()
        if client is None:
            raise HTTPException(status_code=503, detail="Microscope not connected")
        try:
            return await client.set_led(str(payload.get("state", "Closed")))
        except Exception as exc:
            logger.exception("LED set command failed")
            raise HTTPException(status_code=502, detail=f"led failed: {exc}") from exc

    @router.post("/api/devices/laser/off", dependencies=[Depends(require_control)])
    async def laser_off():
        """Gate ALL laser lines off via the Laser config group "ALL OFF" preset.

        Uses setConfig("Laser", "ALL OFF") which drives the PLogic
        OutputChannel to "none of outputs 5-8" — this gates every line
        (488, 561, 405, 637) off, not just the 488 nm setpoint.
        Required for safe brightfield live-view (spec §2.7).
        """
        client = _resolve_client()
        if client is None:
            raise HTTPException(status_code=503, detail="Microscope not connected")
        try:
            return await client.set_laser_config("ALL OFF")
        except Exception as exc:
            logger.exception("Laser off command failed")
            raise HTTPException(status_code=502, detail=f"laser off failed: {exc}") from exc

    @router.get("/api/devices/laser/configs")
    async def laser_configs():
        """Return the available Laser config-group presets from the device layer.

        No require_control — read-only status route, mirrors GET status
        routes like room_light/status and temperature/status.
        """
        client = _resolve_client()
        if client is None or not client.is_connected:
            # Device layer offline is an expected state (e.g. UI open without the
            # device process) — a quiet 503, not an ERROR traceback per poll.
            raise HTTPException(status_code=503, detail="Microscope not connected")
        try:
            return await client.get_laser_configs()
        except Exception as exc:
            logger.exception("Laser configs fetch failed")
            raise HTTPException(status_code=502, detail=f"laser configs failed: {exc}") from exc

    @router.post("/api/devices/laser/config", dependencies=[Depends(require_control)])
    async def laser_config_set(payload: dict = Body(...)):  # noqa: B008
        """Apply a named Laser config-group preset (e.g. "ALL OFF", "488 only").

        Body: {"config": "<preset name>"}

        Returns the device layer response.  400 if config is missing/empty;
        503 if the microscope is not connected; 502 on device error.
        """
        config = payload.get("config")
        if not config or not isinstance(config, str):
            raise HTTPException(status_code=400, detail="config must be a non-empty string")
        client = _resolve_client()
        if client is None:
            raise HTTPException(status_code=503, detail="Microscope not connected")
        try:
            return await client.set_laser_config(config)
        except Exception as exc:
            logger.exception("Laser config set command failed")
            raise HTTPException(status_code=502, detail=f"laser config failed: {exc}") from exc

    @router.get("/api/devices/cameras")
    async def cameras_list():
        """Return the available SPIM camera roles (A always; B if camera_b registered).

        No require_control — read-only status route, mirrors GET /api/devices/laser/configs.
        """
        client = _resolve_client()
        if client is None or not client.is_connected:
            # Device layer offline is expected — quiet 503, no ERROR traceback.
            raise HTTPException(status_code=503, detail="Microscope not connected")
        try:
            return await client.get_cameras()
        except Exception as exc:
            logger.exception("Cameras list fetch failed")
            raise HTTPException(status_code=502, detail=f"cameras failed: {exc}") from exc

    @router.post("/api/devices/camera/led_mode", dependencies=[Depends(require_control)])
    async def camera_led_mode(payload: dict = Body(...)):  # noqa: B008
        """Enable/disable automatic LED for bottom-camera captures. Body: {"use_led": bool}."""
        client = _resolve_client()
        if client is None:
            raise HTTPException(status_code=503, detail="Microscope not connected")
        try:
            return await client.set_camera_led_mode(bool(payload.get("use_led", False)))
        except Exception as exc:
            logger.exception("Camera LED mode command failed")
            raise HTTPException(status_code=502, detail=f"camera led mode failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Stage
    # ------------------------------------------------------------------

    @router.post("/api/devices/stage/move", dependencies=[Depends(require_control)])
    async def stage_move(payload: dict = Body(...)):  # noqa: B008
        """Move the stage to an absolute XY position. Body: {"x": float, "y": float}."""
        client = _resolve_client()
        if client is None:
            raise HTTPException(status_code=503, detail="Microscope not connected")
        try:
            return await client.move_to_position(float(payload["x"]), float(payload["y"]))
        except KeyError:
            raise HTTPException(status_code=400, detail="x and y required") from None
        except Exception as exc:
            logger.exception("Stage move command failed")
            raise HTTPException(status_code=502, detail=f"stage move failed: {exc}") from exc

    def _num(v):
        try:
            return float(v)
        except (TypeError, ValueError):
            return None

    def _persist_detection_labels(agent, payload: dict, markers: list) -> bool:
        """Persist the annotated bottom-cam frame + per-marker pixel/stage coords
        as a labelled snapshot — localization training data (sub-project B).

        Best-effort: requires a client-supplied image and an active session; any
        failure (no session, missing deps) is swallowed so it never blocks the
        embryo registration. Reuses FileStore.register_snapshot so labels live in
        the standard snapshots/ sidecar, not a bespoke store.
        """
        image_b64 = payload.get("image_b64")
        store = getattr(agent, "store", None)
        session_id = getattr(agent, "session_id", None)
        if not image_b64 or store is None or not session_id:
            return False
        try:
            import base64
            import io
            import tempfile
            import uuid as _uuid

            import numpy as np
            import tifffile
            from PIL import Image

            from gently.core.coordinates import (
                DEFAULT_OBJECTIVE_MAG,
                DEFAULT_PIXEL_SIZE_UM,
            )

            raw = base64.b64decode(image_b64.split(",")[-1])
            img = np.asarray(Image.open(io.BytesIO(raw)))
            tmp = Path(tempfile.gettempdir()) / f"operate_{_uuid.uuid4().hex[:12]}.tif"
            tifffile.imwrite(str(tmp), img)

            frame = payload.get("frame") or {}
            pos = payload.get("stage_position") or [None, None]
            meta = {
                "kind": "operate_marking",
                "stage_position": list(pos),
                "frame": {
                    "width": _num(frame.get("w")),
                    "height": _num(frame.get("h")),
                    "downsample": _num(frame.get("downsample")),
                },
                "transform": {
                    "pixel_size_um": DEFAULT_PIXEL_SIZE_UM,
                    "objective_mag": DEFAULT_OBJECTIVE_MAG,
                },
                "embryos": [
                    {
                        "pixel_x": _num(m.get("pixel_x")),
                        "pixel_y": _num(m.get("pixel_y")),
                        "stage_x_um": _num(m.get("stage_x_um")),
                        "stage_y_um": _num(m.get("stage_y_um")),
                        "source": m.get("source", "manual"),
                    }
                    for m in markers
                ],
            }
            store.register_snapshot(session_id, "operate_marked", tmp, metadata=meta)
            return True
        except Exception:
            logger.debug("detection-label persistence skipped", exc_info=True)
            return False

    @router.post("/api/devices/detect_embryos", dependencies=[Depends(require_control)])
    async def detect_embryos(payload: dict = Body(default={})):  # noqa: B008
        """Run bottom-camera embryo detection and RETURN the candidates for the
        Operate-view marking canvas. Does NOT register — the operator confirms
        (add/remove/relocate) on a frozen frame, then POSTs /api/devices/embryos/
        confirm. This keeps the human in the loop and the canonical embryo list
        clean (a marking step, not a blind auto-register).

        Body (all optional): {exposure_ms, min_confidence, brightness_percentile,
        min_area, max_area, use_claude_review, use_last_frame}. Claude review
        defaults OFF. use_last_frame detects on the last streamed frame (if any)
        instead of capturing a fresh image.

        Returns: {success, count, stage_position: [x, y] | null,
                  embryos: [{embryo_id, pixel_x, pixel_y, stage_x_um, stage_y_um,
                             confidence, area_pixels, bbox_pixel}]}.
        """
        client = _resolve_client()
        if client is None:
            raise HTTPException(status_code=503, detail="Microscope not connected")
        if not getattr(client, "has_sam", False):
            raise HTTPException(
                status_code=503, detail="SAM detection not available on device layer"
            )

        kw: dict = {
            "use_claude_review": bool(payload.get("use_claude_review", False)),
            "use_last_frame": bool(payload.get("use_last_frame", False)),
            "capture_only": bool(payload.get("capture_only", False)),
        }
        for key, cast in (
            ("exposure_ms", float),
            ("min_confidence", float),
            ("brightness_percentile", float),
            ("min_area", int),
            ("max_area", int),
        ):
            if payload.get(key) is not None:
                kw[key] = cast(payload[key])
        try:
            result = await client.detect_embryos(**kw)
        except Exception as exc:
            logger.exception("Embryo detection failed")
            raise HTTPException(status_code=502, detail=f"detection failed: {exc}") from exc

        if not result.get("success"):
            raise HTTPException(
                status_code=502, detail=str(result.get("error", "detection failed"))
            )

        embryos = []
        for emb in result.get("embryos", []) or []:
            bbox = emb.get("bbox_pixel")
            embryos.append(
                {
                    "embryo_id": emb.get("embryo_id"),
                    "pixel_x": _num(emb.get("pixel_x")),
                    "pixel_y": _num(emb.get("pixel_y")),
                    "stage_x_um": _num(emb.get("stage_x_um")),
                    "stage_y_um": _num(emb.get("stage_y_um")),
                    "confidence": _num(emb.get("confidence")),
                    "area_pixels": emb.get("area_pixels"),
                    "bbox_pixel": list(bbox) if bbox is not None else None,
                }
            )
        pos = result.get("stage_position")
        return {
            "success": True,
            "count": len(embryos),
            "stage_position": list(pos) if pos is not None else None,
            "embryos": embryos,
            # The JPEG-encoded frame SAM ran on, so the Operate view can display
            # the image the candidates came from (esp. a fresh capture).
            "frame": result.get("frame"),
        }

    @router.post("/api/devices/embryos/confirm", dependencies=[Depends(require_control)])
    async def confirm_embryos(payload: dict = Body(...)):  # noqa: B008
        """Register operator-confirmed markers into the canonical embryo list.

        Agent-free commit step for the Operate-view marking canvas. The client
        computes each marker's stage XY from the frozen frame (SAM candidates
        carry server-computed stage coords; clicked markers are converted with
        the frame's downsample-aware transform), so the server just registers
        via ExperimentState.add_embryo(role='unassigned'), firing EMBRYOS_UPDATE.

        Body: {markers: [{stage_x_um, stage_y_um, pixel_x?, pixel_y?, source?,
               confidence?}], image_b64?, frame? {w,h,downsample}, stage_position?}.
        When image_b64 is present the annotated frame + per-marker pixel/stage
        coords are persisted as a labelled snapshot (sub-project B: localization
        training data) — best-effort, never blocks registration.
        Returns {success, registered: [embryo_id, ...], labelled: bool}.
        """
        agent = _require_agent_with_experiment()
        markers = payload.get("markers") or []

        def _next_embryo_id(taken: set) -> str:
            n = 1
            while f"embryo_{n}" in agent.experiment.embryos or f"embryo_{n}" in taken:
                n += 1
            return f"embryo_{n}"

        import uuid

        store = getattr(agent, "store", None)
        sid = getattr(agent, "session_id", None)

        registered: list[str] = []
        taken: set = set()
        for m in markers:
            sx, sy = _num(m.get("stage_x_um")), _num(m.get("stage_y_um"))
            if sx is None or sy is None:
                continue
            emb_id = _next_embryo_id(taken)
            taken.add(emb_id)
            emb_uid = str(uuid.uuid4())
            agent.experiment.add_embryo(
                embryo_id=emb_id,
                position={"x": sx, "y": sy},
                confidence=_num(m.get("confidence")) or 0.0,
                uid=emb_uid,
                role="unassigned",
            )
            # Persist to the session files so embryos survive a restart — the
            # experiment is otherwise memory-only until a volume is acquired.
            # Best-effort: a storage hiccup must never fail the registration.
            if store is not None and sid:
                try:
                    store.register_embryo(
                        session_id=sid,
                        embryo_id=emb_id,
                        embryo_uid=emb_uid,
                        position_coarse={"x": sx, "y": sy},
                        role="unassigned",
                    )
                except Exception:
                    logger.exception("Failed to persist embryo %s to session files", emb_id)
            registered.append(emb_id)

        labelled = _persist_detection_labels(agent, payload, markers)
        return {"success": True, "registered": registered, "labelled": labelled}

    @router.post(
        "/api/devices/embryos/{embryo_id}/calibrate",
        dependencies=[Depends(require_control)],
    )
    async def calibrate_embryo_route(embryo_id: str, payload: dict = Body(default={})):  # noqa: B008
        """Run piezo-galvo calibration for one embryo — the Operate B-cal step.

        Reuses the agent's proven ``calibrate_embryo`` tool (Claude-vision edge
        detection + adaptive focus sweep, which sets the light-sheet laser config
        per snap) rather than the bare device-layer plan, so the operate flow
        gets the same calibration quality as the agent. No LLM orchestration —
        the coroutine is called directly with an agent/client context. The SPIM
        head should already be lowered + focused (operate reaches this step only
        after B3). Persists the fit onto ``embryo.calibration``.
        """
        agent = _require_agent_with_experiment()
        if embryo_id not in agent.experiment.embryos:
            raise HTTPException(status_code=404, detail=f"unknown embryo {embryo_id}")
        client = _resolve_client()
        if client is None or not getattr(client, "is_connected", False):
            raise HTTPException(status_code=503, detail="Microscope not connected")

        # Ensure the calibration tools are registered on the global registry,
        # then run via the registry so context (agent/client) is injected the
        # same way the agent invokes it. Calling the @tool wrapper directly would
        # drop the positional embryo_id.
        import gently.app.tools.calibration_tools  # noqa: F401  (registers the tool)
        from gently.harness.tools.registry import get_tool_registry

        registry = get_tool_registry()
        try:
            message = await registry.execute(
                "calibrate_embryo",
                {"embryo_id": embryo_id},
                {"agent": agent, "client": client},
            )
        except Exception as exc:
            logger.exception("Calibration failed for %s", embryo_id)
            raise HTTPException(status_code=502, detail=f"calibration failed: {exc}") from exc
        if isinstance(message, str) and message.startswith("Error"):
            raise HTTPException(status_code=502, detail=message)

        emb = agent.experiment.embryos.get(embryo_id)
        calibration = dict(getattr(emb, "calibration", {}) or {}) if emb else {}
        agent.experiment.notify_embryos_changed()
        return {"success": True, "message": message, "calibration": calibration}

    @router.post("/api/embryos/roles", dependencies=[Depends(require_control)])
    async def set_embryo_roles(payload: dict = Body(...)):  # noqa: B008
        """Assign experimental roles to embryos — the Operate "Run" step.

        Body: {roles: {embryo_id: role_name}} where role_name is a key in
        gently.harness.roles.REGISTRY (subject=='test', reference=='calibration',
        plus 'lineaging'/'unassigned'). Sets EmbryoState.role and fires one
        EMBRYOS_UPDATE (+ per-embryo STATUS_CHANGED) so every consumer refreshes.
        Marking stays positions-only; roles are assigned here, not at marking.
        Load-bearing: expression_monitoring scopes to role=='test', so the marked
        set must be given roles before role-scoped monitoring matches anything.
        """
        from gently.harness.roles import is_valid_role

        agent = _require_agent_with_experiment()
        roles = payload.get("roles") or {}
        if not isinstance(roles, dict) or not roles:
            raise HTTPException(status_code=400, detail="roles map required")
        embryos = agent.experiment.embryos
        for eid, role in roles.items():
            if eid not in embryos:
                raise HTTPException(status_code=400, detail=f"unknown embryo {eid}")
            if not is_valid_role(str(role)):
                raise HTTPException(status_code=400, detail=f"invalid role {role}")

        store = getattr(agent, "store", None)
        sid = getattr(agent, "session_id", None)
        bus = getattr(agent, "_event_bus", None)
        updated: list[str] = []
        for eid, role in roles.items():
            emb = embryos[eid]
            old = getattr(emb, "role", None)
            emb.role = str(role)
            if store is not None and sid:
                try:
                    pos = getattr(emb, "position_coarse", {}) or {}
                    store.register_embryo(
                        sid,
                        eid,
                        position_x=pos.get("x"),
                        position_y=pos.get("y"),
                        calibration=getattr(emb, "calibration", {}) or {},
                        role=str(role),
                    )
                except Exception:
                    logger.debug("role persist failed for %s", eid, exc_info=True)
            if bus is not None:
                try:
                    from gently.core.event_bus import EventType

                    bus.publish(
                        event_type=EventType.STATUS_CHANGED,
                        data={
                            "embryo_id": eid,
                            "change": "role_assigned",
                            "old_role": old,
                            "new_role": str(role),
                        },
                        source="operate_roles",
                    )
                except Exception:
                    logger.debug("STATUS_CHANGED publish failed", exc_info=True)
            updated.append(eid)
        try:
            agent.experiment.notify_embryos_changed()  # fires EMBRYOS_UPDATE
        except Exception:
            logger.debug("notify_embryos_changed failed", exc_info=True)
        return {"success": True, "updated": updated}

    @router.post("/api/operate/run-tactic", dependencies=[Depends(require_control)])
    async def operate_run_tactic(payload: dict = Body(...)):  # noqa: B008
        """Append a tactic to the session Operation Plan and execute it via the
        Tactic Executor (resolve scope → dispatch by kind to the orchestrator).

        Body: {tactic: {...}} OR {library_id: "..."} (instantiate a saved tactic);
        optional {embryo_ids: [...]} to re-scope it to the marked set.
        Returns {success, tactic_id, result}.
        """
        from gently.app.orchestration.tactic_executor import (
            append_tactic_to_plan,
            execute_tactic,
        )

        agent = _require_agent_with_experiment()
        tactic = payload.get("tactic")
        lib_id = payload.get("library_id")
        if tactic is None and lib_id:
            cs = getattr(agent, "context_store", None)
            if cs is None:
                raise HTTPException(status_code=503, detail="No context store")
            tactic = cs.apply_tactic(lib_id)
            if tactic is None:
                raise HTTPException(status_code=404, detail=f"tactic '{lib_id}' not found")
        if not isinstance(tactic, dict):
            raise HTTPException(status_code=400, detail="tactic or library_id required")

        eids = payload.get("embryo_ids")
        if eids:
            tactic = dict(tactic)
            tactic["scope"] = {"mode": "embryos", "embryo_ids": list(eids)}

        try:
            stored = append_tactic_to_plan(agent, tactic)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=f"invalid tactic: {exc}") from exc
        if stored is None:
            raise HTTPException(status_code=503, detail="No session to attach the tactic to")
        try:
            result = await execute_tactic(agent, stored)
        except Exception as exc:
            logger.exception("run-tactic execution failed")
            raise HTTPException(status_code=502, detail=f"tactic execution failed: {exc}") from exc
        return {"success": bool(result.get("ok")), "tactic_id": stored.get("id"), "result": result}

    @router.get("/api/operation_plan")
    async def get_operation_plan_route():
        """Current session's Operation Plan (the tactics document), for the
        Operate run-spine. Returns {plan: {...}|null}. Never errors."""
        bridge = getattr(server, "agent_bridge", None)
        agent = bridge.agent if bridge is not None else None
        cs = getattr(agent, "context_store", None) if agent else None
        sid = getattr(agent, "session_id", None) if agent else None
        if cs is None or not sid:
            return {"plan": None}
        try:
            return {"plan": cs.get_operation_plan(sid)}
        except Exception:
            logger.debug("get_operation_plan failed", exc_info=True)
            return {"plan": None}

    # ------------------------------------------------------------------
    # Focus Z axes (Operate view) — fenced read + nudge
    # ------------------------------------------------------------------

    @router.get("/api/devices/stage/bottom_z")
    async def get_bottom_z():
        """Bottom-camera focus Z position + limits (read-only)."""
        client = _resolve_client()
        if client is None:
            raise HTTPException(status_code=503, detail="Microscope not connected")
        try:
            return await client.get_bottom_z()
        except Exception as exc:
            logger.debug("bottom_z read failed: %s", exc)
            raise HTTPException(status_code=502, detail=f"bottom_z read failed: {exc}") from exc

    @router.post("/api/devices/stage/bottom_z/nudge", dependencies=[Depends(require_control)])
    async def nudge_bottom_z(payload: dict = Body(...)):  # noqa: B008
        """Nudge the bottom-camera focus Z by {delta} µm (fenced to its limits)."""
        client = _resolve_client()
        if client is None:
            raise HTTPException(status_code=503, detail="Microscope not connected")
        delta = _num(payload.get("delta"))
        if delta is None:
            raise HTTPException(status_code=400, detail="delta required")
        try:
            return await client.nudge_bottom_z(delta)
        except Exception as exc:
            logger.exception("bottom_z nudge failed")
            raise HTTPException(status_code=502, detail=f"bottom_z nudge failed: {exc}") from exc

    @router.get("/api/devices/spim/fdrive")
    async def get_fdrive():
        """SPIM-head F-drive position + limits + distance-to-floor (read-only)."""
        client = _resolve_client()
        if client is None:
            raise HTTPException(status_code=503, detail="Microscope not connected")
        try:
            return await client.get_fdrive()
        except Exception as exc:
            logger.debug("fdrive read failed: %s", exc)
            raise HTTPException(status_code=502, detail=f"fdrive read failed: {exc}") from exc

    @router.post("/api/devices/spim/fdrive/nudge", dependencies=[Depends(require_control)])
    async def nudge_fdrive(payload: dict = Body(...)):  # noqa: B008
        """Nudge the SPIM-head F-drive by {delta} µm (fenced; never below floor)."""
        client = _resolve_client()
        if client is None:
            raise HTTPException(status_code=503, detail="Microscope not connected")
        delta = _num(payload.get("delta"))
        if delta is None:
            raise HTTPException(status_code=400, detail="delta required")
        try:
            return await client.nudge_fdrive(delta)
        except Exception as exc:
            logger.exception("fdrive nudge failed")
            raise HTTPException(status_code=502, detail=f"fdrive nudge failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Acquisition
    # ------------------------------------------------------------------

    @router.post("/api/devices/acquire/burst", dependencies=[Depends(require_control)])
    async def acquire_burst(payload: dict = Body(...)):  # noqa: B008
        """Trigger a burst acquisition.

        Body: {frames, mode, num_slices, exposure_ms,
               laser_config?, piezo_center?, galvo_center?}.
        laser_config is forwarded directly to the device client so callers
        can send "ALL OFF" for brightfield-safe Manual-view captures.
        piezo_center and galvo_center capture at the dialled focal plane.
        """
        client = _resolve_client()
        if client is None:
            raise HTTPException(status_code=503, detail="Microscope not connected")
        try:
            kw: dict = {}
            if payload.get("laser_config") is not None:
                kw["laser_config"] = str(payload["laser_config"])
            if payload.get("piezo_center") is not None:
                kw["piezo_center"] = float(payload["piezo_center"])
            if payload.get("galvo_center") is not None:
                kw["galvo_center"] = float(payload["galvo_center"])
            result = await client.acquire_burst(
                frames=int(payload.get("frames", 60)),
                mode=str(payload.get("mode", "1hz")),
                num_slices=int(payload.get("num_slices", 1)),
                exposure_ms=float(payload.get("exposure_ms", 5.0)),
                **kw,
            )
            return _json_safe(result)
        except Exception as exc:
            logger.exception("Burst acquisition failed")
            raise HTTPException(status_code=502, detail=f"burst failed: {exc}") from exc

    @router.post("/api/devices/acquire/volume", dependencies=[Depends(require_control)])
    async def acquire_volume(payload: dict = Body(...)):  # noqa: B008
        """Trigger a volume acquisition.

        Body: {num_slices, exposure_ms,
               laser_config?, piezo_center?, galvo_center?}.
        laser_config is forwarded directly to the device client so callers
        can send "ALL OFF" for brightfield-safe Manual-view captures.
        piezo_center and galvo_center capture at the dialled focal plane.
        """
        client = _resolve_client()
        if client is None:
            raise HTTPException(status_code=503, detail="Microscope not connected")
        try:
            kw: dict = {}
            if payload.get("laser_config") is not None:
                kw["laser_config"] = str(payload["laser_config"])
            if payload.get("piezo_center") is not None:
                kw["piezo_center"] = float(payload["piezo_center"])
            if payload.get("galvo_center") is not None:
                kw["galvo_center"] = float(payload["galvo_center"])
            result = await client.acquire_volume(
                num_slices=int(payload.get("num_slices", 50)),
                exposure_ms=float(payload.get("exposure_ms", 10.0)),
                **kw,
            )
            return _json_safe(result)
        except Exception as exc:
            logger.exception("Volume acquisition failed")
            raise HTTPException(status_code=502, detail=f"volume failed: {exc}") from exc

    # ------------------------------------------------------------------
    # Timelapse
    # ------------------------------------------------------------------

    @router.post("/api/devices/timelapse/start", dependencies=[Depends(require_control)])
    async def timelapse_start(payload: dict = Body(...)):  # noqa: B008
        """Start an adaptive timelapse from the manual UI.

        Body fields (all optional except interval_seconds has a default):
          interval_seconds (float, default 120) — cadence; must be > 0
          stop_condition   (str, default "manual") — "manual", "timepoints", "duration"
          embryo_ids       (list[str] | null) — null = all active embryos
          condition_value  (int | null) — timepoints count or duration hours
          monitoring_mode  (str | null) — "idle" / "expression_monitoring" /
                                          "pre_terminal_monitoring"
          num_slices       (int, default 50) — must be >= 1 if provided
          exposure_ms      (float, default 10.0)
          galvo_amplitude  (float, default 0.5)
          galvo_center     (float, default 0.0)
          piezo_amplitude  (float, default 25.0)
          piezo_center     (float, default 50.0)
          laser_config     (str | null)

        Validation:
          - interval_seconds must be > 0
          - num_slices must be >= 1

        Orchestrator access: server.agent_bridge.agent.timelapse_orchestrator
        RIG-DEFERRED: the actual acquisition + galvo/piezo motion.
        """
        # --- Validate ---
        raw_interval = payload.get("interval_seconds", 120.0)
        try:
            interval_seconds = float(raw_interval)
        except (TypeError, ValueError):
            raise HTTPException(  # B904
                status_code=400, detail="interval_seconds must be a number"
            ) from None
        if interval_seconds <= 0:
            raise HTTPException(status_code=400, detail="interval_seconds must be > 0")

        raw_slices = payload.get("num_slices")
        if raw_slices is not None:
            try:
                num_slices = int(raw_slices)
            except (TypeError, ValueError):
                raise HTTPException(  # B904
                    status_code=400, detail="num_slices must be an integer"
                ) from None
            if num_slices < 1:
                raise HTTPException(status_code=400, detail="num_slices must be >= 1")
        else:
            num_slices = 50

        stop_condition = str(payload.get("stop_condition") or "manual")
        embryo_ids = payload.get("embryo_ids") or None
        condition_value = payload.get("condition_value")
        monitoring_mode = payload.get("monitoring_mode") or None

        # Volume geometry — passed through for context / future calibration write;
        # not forwarded to orchestrator.start (which owns its own geometry via the
        # per-embryo calibration). RIG-DEFERRED: real acquisition uses these.
        volume_geometry = {
            "num_slices": num_slices,
            "exposure_ms": float(payload.get("exposure_ms", 10.0)),
            "galvo_amplitude": float(payload.get("galvo_amplitude", 0.5)),
            "galvo_center": float(payload.get("galvo_center", 0.0)),
            "piezo_amplitude": float(payload.get("piezo_amplitude", 25.0)),
            "piezo_center": float(payload.get("piezo_center", 50.0)),
            "laser_config": payload.get("laser_config") or None,
        }

        # --- Resolve orchestrator ---
        bridge = getattr(server, "agent_bridge", None)
        agent = bridge.agent if bridge is not None else None
        orchestrator = getattr(agent, "timelapse_orchestrator", None) if agent else None
        if orchestrator is None:
            raise HTTPException(
                status_code=503,
                detail="Timelapse orchestrator not initialised (agent not running or no session)",
            )

        # --- Start timelapse (RIG-DEFERRED: real acquisition) ---
        # TODO: UI-initiated timelapses skip the agent tool's plan auto-linking;
        #       this is intentional — the agent path wires the plan, this route does not.
        try:
            result = await orchestrator.start(
                embryo_ids=embryo_ids,
                stop_condition=stop_condition,
                base_interval_seconds=interval_seconds,
                condition_value=condition_value,
            )
        except Exception as exc:
            logger.exception("Timelapse start failed")
            raise HTTPException(status_code=502, detail=f"timelapse start failed: {exc}") from exc

        # Optionally install a monitoring mode at startup (mirrors start_adaptive_timelapse)
        mode_result = None
        if monitoring_mode and monitoring_mode != "idle":
            try:
                mode_result = orchestrator.enable_monitoring_mode(monitoring_mode)
            except Exception as exc:
                mode_result = f"warning: failed to enable monitoring mode: {exc}"

        # Seed the session Operation Plan with a standing_timelapse tactic (+ a
        # reactive_monitor when a monitoring mode is active) scoped to the marked
        # set. Closes the historical "UI timelapses skip plan linking" gap so the
        # Operate run-spine and the Operations tab show a real tactic. Best-effort:
        # needs a live session + context store; never blocks the start.
        seeded_tactics: list[str] = []
        cs = getattr(agent, "context_store", None)
        sid = getattr(agent, "session_id", None)
        # Skip seeding when start was a no-op ("already running") — don't append a
        # phantom active tactic for a run we didn't start.
        already_running = isinstance(result, str) and result.startswith("Timelapse already running")
        if cs is not None and sid and not already_running:
            try:
                import uuid as _uuid

                from gently.app.tools.operation_plan_tools import _validate_tactics

                # Scope mirrors what orchestrator.start actually does: an omitted
                # embryo_ids images ALL active embryos → record global, not [].
                eids = list(embryo_ids or [])
                seed_scope = {"mode": "embryos", "embryo_ids": eids} if eids else {"mode": "global"}
                st_id = f"op_{_uuid.uuid4().hex[:8]}"
                new_tactics: list[dict] = [
                    {
                        "id": st_id,
                        "name": "Adaptive timelapse",
                        "kind": "standing_timelapse",
                        "state": "active",
                        "scope": dict(seed_scope),
                        "structure": {
                            "cadence_s": interval_seconds,
                            "interval": interval_seconds,
                            "stop_condition": stop_condition,
                            "condition_value": condition_value,
                            "monitoring_mode": monitoring_mode or "idle",
                        },
                        "rationale": "Started from the Operate Run step.",
                        "live_bind": ["cadence"],
                        "relations": {},
                        "live": {},
                        "source": "operate",
                    }
                ]
                if monitoring_mode and monitoring_mode != "idle":
                    new_tactics.append(
                        {
                            "id": f"op_{_uuid.uuid4().hex[:8]}",
                            "name": "Monitor",
                            "kind": "reactive_monitor",
                            "state": "active",
                            "scope": dict(seed_scope),
                            "structure": {"monitoring_mode": monitoring_mode, "status": "armed"},
                            "rationale": f"{monitoring_mode} on the marked subjects.",
                            "live_bind": ["signal"],
                            "relations": {"layered_on": [st_id]},
                            "live": {},
                            "source": "operate",
                        }
                    )
                new_tactics = _validate_tactics(new_tactics)
                plan = cs.get_operation_plan(sid) or {
                    "session_id": sid,
                    "title": "Operate session",
                    "goal": "",
                    "tactics": [],
                }
                # Reconcile: retire any prior still-'active' operate-seeded tactics
                # so repeated Start clicks don't accumulate stale active timelapses.
                for t in plan.setdefault("tactics", []):
                    if t.get("source") == "operate" and t.get("state") == "active":
                        t["state"] = "done"
                plan["tactics"].extend(new_tactics)
                plan["updated_reason"] = "operate adaptive timelapse"
                cs.set_operation_plan(sid, plan)
                seeded_tactics = [t["id"] for t in new_tactics]
                # Link the run to its tactics so stop/pause/resume can reconcile them.
                try:
                    orchestrator._operate_tactic_ids = list(seeded_tactics)
                except Exception:
                    pass
            except Exception:
                logger.debug("timelapse tactic seeding skipped", exc_info=True)

        return {
            "started": True,
            "result": result,
            "tactics": seeded_tactics,
            "monitoring_mode_result": mode_result,
            "config": {
                "interval_seconds": interval_seconds,
                "stop_condition": stop_condition,
                "embryo_ids": embryo_ids,
                "condition_value": condition_value,
                "monitoring_mode": monitoring_mode,
                "volume_geometry": volume_geometry,
            },
        }

    def _resolve_orch_and_agent():
        bridge = getattr(server, "agent_bridge", None)
        agent = bridge.agent if bridge is not None else None
        orch = getattr(agent, "timelapse_orchestrator", None) if agent else None
        return orch, agent

    def _reconcile_operate_tactics(orch, agent, state: str, clear: bool):
        """Transition the run's operate-seeded tactics to ``state`` so the
        Operation Plan / run-spine reflect stop/pause/resume. Best-effort."""
        cs = getattr(agent, "context_store", None)
        sid = getattr(agent, "session_id", None)
        ids = list(getattr(orch, "_operate_tactic_ids", []) or [])
        if cs is not None and sid and ids:
            for tid in ids:
                try:
                    cs.transition_tactic(sid, tid, state)
                except Exception:
                    logger.debug("transition_tactic %s failed", tid, exc_info=True)
        if clear:
            try:
                orch._operate_tactic_ids = []
            except Exception:
                pass

    @router.post("/api/devices/timelapse/stop", dependencies=[Depends(require_control)])
    async def timelapse_stop(payload: dict = Body(default={})):  # noqa: B008
        """Stop the running timelapse (Operate run-spine). Body: {reason?}."""
        orch, agent = _resolve_orch_and_agent()
        if orch is None:
            raise HTTPException(status_code=503, detail="No timelapse orchestrator")
        try:
            reason = str(payload.get("reason", "user_request"))
            res = await orch.stop(reason=reason)
            _reconcile_operate_tactics(orch, agent, "done", clear=True)
            return {"stopped": True, "result": res}
        except Exception as exc:
            logger.exception("Timelapse stop failed")
            raise HTTPException(status_code=502, detail=f"timelapse stop failed: {exc}") from exc

    @router.post("/api/devices/timelapse/pause", dependencies=[Depends(require_control)])
    async def timelapse_pause():
        """Pause the running timelapse."""
        orch, agent = _resolve_orch_and_agent()
        if orch is None:
            raise HTTPException(status_code=503, detail="No timelapse orchestrator")
        try:
            res = await orch.pause()
            _reconcile_operate_tactics(orch, agent, "paused", clear=False)
            return {"paused": True, "result": res}
        except Exception as exc:
            logger.exception("Timelapse pause failed")
            raise HTTPException(status_code=502, detail=f"timelapse pause failed: {exc}") from exc

    @router.post("/api/devices/timelapse/resume", dependencies=[Depends(require_control)])
    async def timelapse_resume():
        """Resume a paused timelapse."""
        orch, agent = _resolve_orch_and_agent()
        if orch is None:
            raise HTTPException(status_code=503, detail="No timelapse orchestrator")
        try:
            res = await orch.resume()
            _reconcile_operate_tactics(orch, agent, "active", clear=False)
            return {"resumed": True, "result": res}
        except Exception as exc:
            logger.exception("Timelapse resume failed")
            raise HTTPException(status_code=502, detail=f"timelapse resume failed: {exc}") from exc

    @router.get("/api/calibration")
    async def list_calibration(embryo_id: str | None = None):
        """Get calibration images"""
        images = server.store.get_all_calibration(embryo_id)
        return {"calibration": [img.to_dict() for img in images], "count": len(images)}

    @router.get("/api/volumes")
    async def list_volumes(embryo_id: str | None = None):
        """Get volume images"""
        images = server.store.get_all_volumes(embryo_id)
        return {"volumes": [img.to_dict() for img in images], "count": len(images)}

    @router.get("/api/snapshots")
    async def list_snapshots(embryo_id: str | None = None):
        """Get snapshot images"""
        images = server.store.get_all_snapshots(embryo_id)
        return {"snapshots": [img.to_dict() for img in images], "count": len(images)}

    @router.get("/api/embryos")
    async def list_embryos():
        """Get list of embryos with images"""
        return {"embryos": server.store.get_embryo_ids()}

    @router.get("/api/embryos/positions")
    async def embryo_positions():
        """Get per-embryo stage positions + role for the device map.

        Sources:
        - ``TimelapseStateTracker.embryos`` — fed by ``EMBRYO_DETECTED``
          (every ``ExperimentState.add_embryo`` call publishes one) plus
          ``STATUS_CHANGED { change: 'role_assigned' }``.
        - Per-embryo state from any active timelapse.

        Returns a flat list so the renderer can map directly to SVG circles.
        """
        tracker = getattr(server, "timelapse_tracker", None)
        points = []
        if tracker is not None:
            for eid, emb in (tracker.embryos or {}).items():
                x = emb.get("stage_x_um")
                y = emb.get("stage_y_um")
                if x is None or y is None:
                    # Embryo registered but no position yet (e.g. only the
                    # ID arrived from another path). Skip — nothing to render.
                    continue
                points.append(
                    {
                        "embryo_id": eid,
                        "uid": emb.get("uid"),
                        "x": float(x),
                        "y": float(y),
                        "role": emb.get("role", "test"),
                        "strain": emb.get("strain"),
                        "user_label": emb.get("user_label"),
                        "confidence": emb.get("confidence"),
                        "cadence_phase": emb.get("cadence_phase"),
                        "is_complete": bool(emb.get("is_complete")),
                    }
                )
        return {"embryos": points}

    @router.get("/api/sequence/{embryo_id}")
    async def get_image_sequence(
        embryo_id: str,
        start: int = 0,
        end: int | None = None,
        data_type: str = "volume_projection",
        buffer_percent: float = 0.15,
    ):
        """Get ordered sequence of images for timepoint range.

        Returns a list of image metadata (without base64 data) for lazy loading.
        Use /api/images/{uid}/png to load individual frames.
        """
        # Calculate buffered range
        if end is not None:
            range_size = end - start
            buffer = int(range_size * buffer_percent)
            buffered_start = max(0, start - buffer)
            buffered_end = end + buffer
        else:
            buffered_start = start
            buffered_end = None

        images = server.store.get_sequence(
            embryo_id=embryo_id,
            start=buffered_start,
            end=buffered_end,
            data_type=data_type,
        )

        # Return lightweight metadata (no base64 data)
        sequence = []
        seen_uids = set()
        for img in images:
            seen_uids.add(img.uid)
            sequence.append(
                {
                    "uid": img.uid,
                    "timepoint": img.metadata.get("timepoint"),
                    "timestamp": img.timestamp,
                    "data_type": img.data_type,
                    "shape": img.shape,
                    "embryo_id": img.metadata.get("embryo_id"),
                }
            )

        # Fallback to persistent DataStore for missing timepoints
        if server.data_store and (len(sequence) == 0 or buffered_end is not None):
            try:
                refs = server.data_store.query(data_type=data_type, embryo_id=embryo_id)
                for ref in refs:
                    if ref.uid in seen_uids:
                        continue
                    tp = ref.metadata.get("timepoint")
                    if tp is None:
                        continue
                    tp = int(tp)
                    if tp < buffered_start:
                        continue
                    if buffered_end is not None and tp > buffered_end:
                        continue
                    seen_uids.add(ref.uid)
                    sequence.append(
                        {
                            "uid": ref.uid,
                            "timepoint": tp,
                            "timestamp": ref.metadata.get("timestamp", ""),
                            "data_type": ref.data_type,
                            "shape": ref.metadata.get("shape"),
                            "embryo_id": embryo_id,
                        }
                    )
                # Re-sort by timepoint
                sequence.sort(key=lambda x: x.get("timepoint") or 0)
            except Exception as e:
                logger.warning(f"DataStore fallback failed: {e}")

        return {
            "embryo_id": embryo_id,
            "requested_range": {"start": start, "end": end},
            "buffered_range": {"start": buffered_start, "end": buffered_end},
            "sequence": sequence,
            "count": len(sequence),
        }

    @router.get("/api/events")
    async def list_events(
        event_type: str | None = None, source: str | None = None, limit: int = 100
    ):
        """Get event history from EventBus"""
        if not server.event_bus:
            return {"events": [], "total": 0}

        # Get history from event bus
        from gently.core import EventType

        et = None
        if event_type:
            try:
                et = EventType[event_type]
            except KeyError:
                pass

        events = server.event_bus.get_history(event_type=et, source=source, limit=limit)

        return {
            "events": [
                {
                    "event_type": e.event_type.name
                    if hasattr(e.event_type, "name")
                    else str(e.event_type),
                    "data": e.data,
                    "source": e.source,
                    "timestamp": e.timestamp.isoformat(),
                    "event_id": e.event_id,
                }
                for e in events
            ],
            "total": len(events),
        }

    return router
