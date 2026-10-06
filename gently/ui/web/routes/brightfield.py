"""Dark and flat-field references for brightfield frames, taken from the
Acquisition pane and filed in the live session (see gently.app.brightfield).

    GET  /api/brightfield/references?light=&led_intensity_pct=&exposure_ms=
         every record this session holds, and which one matches the spec
    POST /api/brightfield/references/dark   {exposure_ms, light, led_intensity_pct, record?}
    POST /api/brightfield/references/flat   {exposure_ms, light, led_intensity_pct,
                                             frames?, record?}

One capture at a time, and none while a run is live: the dark cycles the
room light, which would land in whatever the run was imaging.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Body, Depends, HTTPException

from gently.ui.web.auth import require_control

logger = logging.getLogger(__name__)


def create_router(server) -> APIRouter:
    router = APIRouter()
    busy = asyncio.Lock()

    def _agent():
        bridge = getattr(server, "agent_bridge", None)
        return bridge.agent if bridge is not None else None

    def _store_and_session():
        agent = _agent()
        store = getattr(agent, "store", None) if agent else None
        sid = getattr(agent, "session_id", None) if agent else None
        if store is None or not sid:
            raise HTTPException(status_code=503, detail="No live session to file references in")
        return store, sid

    def _client():
        client = getattr(_agent(), "client", None)
        if client is None:
            raise HTTPException(status_code=503, detail="Microscope not connected")
        return client

    def _refuse_while_running():
        orch = getattr(_agent(), "timelapse_orchestrator", None)
        status = getattr(getattr(orch, "_status", None), "value", None)
        if status in ("running", "paused"):
            raise HTTPException(
                status_code=409,
                detail=f"a run is {status}; references are taken before a run, not during one",
            )

    def _spec(body: dict) -> Any:
        from gently.app.brightfield import ReferenceSpec

        light = str(body.get("light") or "room")
        if light not in ("room", "led", "none"):
            raise HTTPException(status_code=400, detail="light must be room, led or none")
        pct = body.get("led_intensity_pct")
        exp = body.get("exposure_ms")
        try:
            pct_v = int(float(str(pct))) if pct not in (None, "") else None
            exp_v = float(str(exp)) if exp not in (None, "") else None
        except (TypeError, ValueError) as e:
            raise HTTPException(
                status_code=400, detail="led_intensity_pct and exposure_ms must be numbers"
            ) from e
        if pct_v is not None and not 1 <= pct_v <= 100:
            raise HTTPException(status_code=400, detail="led_intensity_pct must be 1–100")
        if exp_v is not None and not 0 < exp_v <= 60000:
            raise HTTPException(status_code=400, detail="exposure_ms must be between 0 and 60000")
        return ReferenceSpec(
            light=light, led_intensity_pct=pct_v if light == "led" else None, exposure_ms=exp_v
        )

    def _record_folder(store, sid, spec, wanted: str | None) -> Path:
        from gently.app.brightfield import open_record, references_dir

        if wanted:
            base = references_dir(store, sid)
            folder = base / Path(str(wanted)).name if base is not None else None
            if folder is None or not folder.is_dir() or folder.parent != base:
                raise HTTPException(
                    status_code=404, detail=f"No reference record {wanted!r} in this session"
                )
            return folder
        return open_record(store, sid, spec)

    def _thumb(image) -> str | None:
        try:
            from gently.core.imaging import downsample_mean, image_to_base64, normalize_to_uint8

            return image_to_base64(normalize_to_uint8(downsample_mean(image, 384)))
        except Exception:
            return None

    @router.get("/api/brightfield/references")
    async def list_references(
        light: str | None = None,
        led_intensity_pct: int | None = None,
        exposure_ms: float | None = None,
    ):
        from gently.app.brightfield import ReferenceSpec, for_frame, list_records, matching

        store, sid = _store_and_session()
        records = await asyncio.to_thread(list_records, store, sid)
        match = None
        if light:
            spec = ReferenceSpec(
                light=light,
                led_intensity_pct=led_intensity_pct if light == "led" else None,
                exposure_ms=exposure_ms,
            )
            match = matching(records, spec)
        return {
            "session_id": sid,
            "records": records,
            "match": for_frame(match) if match else None,
            "match_record": match,
            "busy": busy.locked(),
        }

    async def _take(kind: str, body: dict) -> dict:
        from gently.app.brightfield import file_image, take_dark, take_flat

        store, sid = _store_and_session()
        client = _client()
        _refuse_while_running()
        spec = _spec(body)
        if busy.locked():
            raise HTTPException(status_code=409, detail="a reference is being taken already")
        async with busy:
            folder = _record_folder(store, sid, spec, body.get("record"))
            try:
                if kind == "dark":
                    taken = await take_dark(
                        client, spec.exposure_ms, settle_s=float(body.get("settle_s") or 1.0)
                    )
                else:
                    taken = await take_flat(
                        client,
                        spec,
                        frames=int(body.get("frames") or 5),
                        settle_s=float(body.get("settle_s") or 1.0),
                    )
            except HTTPException:
                raise
            except Exception as exc:
                logger.exception("%s reference failed", kind)
                raise HTTPException(
                    status_code=502, detail=f"the {kind} could not be taken: {exc}"
                ) from exc
            doc = await asyncio.to_thread(file_image, folder, kind, taken["image"], spec, taken)
        doc["folder"] = str(folder)
        return {
            "kind": kind,
            "record": folder.name,
            "stats": taken["stats"],
            "steps": taken["steps"],
            "frames": taken.get("frames"),
            "checks": doc.get("checks") or {},
            "thumbnail": _thumb(taken["image"]),
            "document": doc,
        }

    @router.post("/api/brightfield/references/dark", dependencies=[Depends(require_control)])
    async def take_dark_reference(body: dict = Body(default={})):  # noqa: B008
        """The camera with nothing lit: cycles the room light so its state
        is known, closes the LED, snaps. Opens a new record unless ``record``
        names one."""
        return await _take("dark", body or {})

    @router.post("/api/brightfield/references/flat", dependencies=[Depends(require_control)])
    async def take_flat_reference(body: dict = Body(default={})):  # noqa: B008
        """The empty field under the run's light, ``frames`` averaged. The
        operator has driven the stage clear of the embryos first."""
        return await _take("flat", body or {})

    return router
