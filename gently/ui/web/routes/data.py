"""Data routes - calibration, snapshots, embryos, sequence, status, events."""

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import yaml
from fastapi import APIRouter, Body, HTTPException

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
            "timestamp": datetime.now().isoformat()
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

    @router.get("/api/devices/zones")
    async def get_device_zones():
        """Return XY stage zones + optional coverslip outline for the Devices Map view.

        Reads ``config/hardware.yaml`` fresh on each call so an operator can
        retune zones without restarting the agent. Only ``green`` (optimal)
        and ``orange`` (maximal) zones are returned; any legacy ``red`` entry
        in the file is silently dropped — red is now implicit ("beyond").
        """
        try:
            with open(_HARDWARE_CONFIG_PATH, "r") as f:
                cfg = yaml.safe_load(f) or {}
        except FileNotFoundError:
            raise HTTPException(
                status_code=404,
                detail=f"hardware.yaml not found at {_HARDWARE_CONFIG_PATH}",
            )
        xy = cfg.get("xy_stage") or {}
        zones = [z for z in (xy.get("zones") or []) if z.get("color") in ("green", "orange")]
        out = {
            "units": xy.get("units", "um"),
            "zones": zones,
        }
        cs = cfg.get("coverslip")
        if isinstance(cs, dict):
            out["coverslip"] = {
                "center_um": list(cs.get("center_um") or [0.0, 0.0]),
                "size_mm":   list(cs.get("size_mm")   or [50.0, 24.0]),
            }
        return out

    @router.post("/api/devices/zones")
    async def set_device_zones(payload: dict = Body(...)):
        """Replace the xy_stage zones in ``config/hardware.yaml``.

        Body shape mirrors the GET response: ``{"units": "um", "zones": [...]}``.
        Only ``green`` (optimal) and ``orange`` (maximal) zones are persisted —
        any ``red`` zone in the payload is silently dropped, since "beyond"
        is implicit and not operator-editable. The coverslip block (if any)
        is preserved untouched.
        """
        zones_in = payload.get("zones")
        if not isinstance(zones_in, list):
            raise HTTPException(status_code=400, detail="`zones` must be a list")

        ALLOWED_COLORS = {"green", "orange"}
        normalized = []
        for i, z in enumerate(zones_in):
            if not isinstance(z, dict):
                raise HTTPException(status_code=400, detail=f"zone[{i}] must be an object")
            color = z.get("color")
            if color == "red":
                # Drop legacy red entries — "beyond" is implicit now.
                continue
            x = z.get("x")
            y = z.get("y")
            if color not in ALLOWED_COLORS:
                raise HTTPException(status_code=400, detail=f"zone[{i}].color must be one of {sorted(ALLOWED_COLORS)}")
            if not (isinstance(x, list) and len(x) == 2 and all(isinstance(v, (int, float)) for v in x)):
                raise HTTPException(status_code=400, detail=f"zone[{i}].x must be [min, max]")
            if not (isinstance(y, list) and len(y) == 2 and all(isinstance(v, (int, float)) for v in y)):
                raise HTTPException(status_code=400, detail=f"zone[{i}].y must be [min, max]")
            normalized.append({
                "color": color,
                "x": [float(min(x)), float(max(x))],
                "y": [float(min(y)), float(max(y))],
            })

        # Read-modify-write so we don't clobber unrelated top-level keys
        # (notably the coverslip block).
        try:
            with open(_HARDWARE_CONFIG_PATH, "r") as f:
                cfg = yaml.safe_load(f) or {}
        except FileNotFoundError:
            cfg = {}

        xy = cfg.get("xy_stage") or {}
        xy["units"] = payload.get("units", xy.get("units", "um"))
        xy["zones"] = normalized
        cfg["xy_stage"] = xy

        with open(_HARDWARE_CONFIG_PATH, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False, default_flow_style=None)

        logger.info("Rewrote %s with %d zones", _HARDWARE_CONFIG_PATH, len(normalized))
        out = {"units": xy["units"], "zones": normalized}
        cs = cfg.get("coverslip")
        if isinstance(cs, dict):
            out["coverslip"] = {
                "center_um": list(cs.get("center_um") or [0.0, 0.0]),
                "size_mm":   list(cs.get("size_mm")   or [50.0, 24.0]),
            }
        return out

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

    @router.post("/api/devices/bottom_camera/stream/start")
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
            raise HTTPException(status_code=500, detail=f"start failed: {exc}")
        return {"streaming": monitor.running}

    @router.post("/api/devices/bottom_camera/stream/stop")
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
            raise HTTPException(status_code=500, detail=f"stop failed: {exc}")
        return {"streaming": False}

    @router.get("/api/calibration")
    async def list_calibration(embryo_id: Optional[str] = None):
        """Get calibration images"""
        images = server.store.get_all_calibration(embryo_id)
        return {
            "calibration": [img.to_dict() for img in images],
            "count": len(images)
        }

    @router.get("/api/volumes")
    async def list_volumes(embryo_id: Optional[str] = None):
        """Get volume images"""
        images = server.store.get_all_volumes(embryo_id)
        return {
            "volumes": [img.to_dict() for img in images],
            "count": len(images)
        }

    @router.get("/api/snapshots")
    async def list_snapshots(embryo_id: Optional[str] = None):
        """Get snapshot images"""
        images = server.store.get_all_snapshots(embryo_id)
        return {
            "snapshots": [img.to_dict() for img in images],
            "count": len(images)
        }

    @router.get("/api/embryos")
    async def list_embryos():
        """Get list of embryos with images"""
        return {
            "embryos": server.store.get_embryo_ids()
        }

    @router.get("/api/sequence/{embryo_id}")
    async def get_image_sequence(
        embryo_id: str,
        start: int = 0,
        end: Optional[int] = None,
        data_type: str = "volume_projection",
        buffer_percent: float = 0.15
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
            data_type=data_type
        )

        # Return lightweight metadata (no base64 data)
        sequence = []
        seen_uids = set()
        for img in images:
            seen_uids.add(img.uid)
            sequence.append({
                "uid": img.uid,
                "timepoint": img.metadata.get("timepoint"),
                "timestamp": img.timestamp,
                "data_type": img.data_type,
                "shape": img.shape,
                "embryo_id": img.metadata.get("embryo_id")
            })

        # Fallback to persistent DataStore for missing timepoints
        if server.data_store and (len(sequence) == 0 or buffered_end is not None):
            try:
                refs = server.data_store.query(
                    data_type=data_type,
                    embryo_id=embryo_id
                )
                for ref in refs:
                    if ref.uid in seen_uids:
                        continue
                    tp = ref.metadata.get('timepoint')
                    if tp is None:
                        continue
                    tp = int(tp)
                    if tp < buffered_start:
                        continue
                    if buffered_end is not None and tp > buffered_end:
                        continue
                    seen_uids.add(ref.uid)
                    sequence.append({
                        "uid": ref.uid,
                        "timepoint": tp,
                        "timestamp": ref.metadata.get('timestamp', ''),
                        "data_type": ref.data_type,
                        "shape": ref.metadata.get('shape'),
                        "embryo_id": embryo_id
                    })
                # Re-sort by timepoint
                sequence.sort(key=lambda x: x.get('timepoint') or 0)
            except Exception as e:
                logger.warning(f"DataStore fallback failed: {e}")

        return {
            "embryo_id": embryo_id,
            "requested_range": {"start": start, "end": end},
            "buffered_range": {"start": buffered_start, "end": buffered_end},
            "sequence": sequence,
            "count": len(sequence)
        }

    @router.get("/api/events")
    async def list_events(
        event_type: Optional[str] = None,
        source: Optional[str] = None,
        limit: int = 100
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

        events = server.event_bus.get_history(
            event_type=et,
            source=source,
            limit=limit
        )

        return {
            "events": [
                {
                    "event_type": e.event_type.name if hasattr(e.event_type, 'name') else str(e.event_type),
                    "data": e.data,
                    "source": e.source,
                    "timestamp": e.timestamp.isoformat(),
                    "event_id": e.event_id
                }
                for e in events
            ],
            "total": len(events)
        }

    return router
