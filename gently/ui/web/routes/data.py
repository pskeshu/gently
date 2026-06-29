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
            x = float(body.get("x"))
            y = float(body.get("y"))
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
            target = float(payload.get("target_c"))
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
        if client is None:
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
        if client is None:
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
            return await client.acquire_burst(
                frames=int(payload.get("frames", 60)),
                mode=str(payload.get("mode", "1hz")),
                num_slices=int(payload.get("num_slices", 1)),
                exposure_ms=float(payload.get("exposure_ms", 5.0)),
                **kw,
            )
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
            return await client.acquire_volume(
                num_slices=int(payload.get("num_slices", 50)),
                exposure_ms=float(payload.get("exposure_ms", 10.0)),
                **kw,
            )
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

        return {
            "started": True,
            "result": result,
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
