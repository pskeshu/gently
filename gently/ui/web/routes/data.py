"""Data routes - calibration, snapshots, embryos, sequence, status, events, narrative."""

import logging
from datetime import datetime
from typing import Optional

from fastapi import APIRouter

from ..narrative import generate_narrative_summary

logger = logging.getLogger(__name__)


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
        """Get device connection status from agent bridge."""
        bridge = getattr(server, "agent_bridge", None)
        if bridge is None:
            return {"gently": True, "microscope": False}
        info = bridge._launch_info
        return {
            "gently": True,
            "microscope": info.get("device_connected", False),
        }

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

    @router.get("/api/narrative")
    async def get_narrative(since: Optional[str] = None):
        """Generate experiment narrative summary."""
        return generate_narrative_summary(server.timelapse_tracker, since)

    return router
