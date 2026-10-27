"""Twitching to hatching, per embryo, off the brightfield frames.

A brightfield field holds several embryos and no embryo folders, so a row is
a click: which field, where in it, the frame it first twitched and the frame
it hatched. Minutes come from the frames' own ``captured_at`` clocks, so a
changed interval mid-night costs nothing. Temperature over the interval
comes from the session's temperature log, since a Sulston comparison without
it is not one.

Stored as ``annotations/twitch_hatch.yaml`` in the session folder; the
report is derived on read, never stored.
"""

from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Any

from fastapi import APIRouter, Body, Depends, HTTPException

from gently.app.brightfield import overview_frames
from gently.core.file_store import _read_yaml, _write_yaml
from gently.tools import TOOLS, session_tools, set_session_tools
from gently.ui.web.auth import require_control

# Sulston et al. 1983, at 20 °C: twitching ~ 430 min, hatching ~ 800 min
# after first cleavage.
SULSTON_TWITCH_TO_HATCH_MIN = 370.0

FILE = Path("annotations") / "twitch_hatch.yaml"


def _parse(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        return datetime.fromisoformat(str(ts))
    except ValueError:
        return None


def report(rows: list[dict], frames: list[dict], temps: list[dict]) -> list[dict]:
    """Each row with its minutes and mean temperature over the interval.

    A row is keyed to frames by (field, frame); a missing frame leaves the
    minutes None rather than guessing.
    """
    when: dict[tuple[Any, Any], datetime | None] = {
        (f.get("field"), f.get("frame")): _parse(f.get("captured_at")) for f in frames
    }
    out = []
    for r in rows:
        t0 = when.get((r.get("field"), r.get("twitch_frame")))
        t1 = when.get((r.get("field"), r.get("hatch_frame")))
        minutes = (t1 - t0).total_seconds() / 60.0 if t0 and t1 else None
        temp_c = None
        if t0 and t1:
            # Naive and aware stamps may mix across a session; compare naively.
            lo, hi = t0.replace(tzinfo=None), t1.replace(tzinfo=None)
            inside = []
            for s in temps:
                ts = _parse(s.get("t"))
                if ts and lo <= ts.replace(tzinfo=None) <= hi and s.get("water_c") is not None:
                    inside.append(float(s["water_c"]))
            temp_c = round(mean(inside), 2) if inside else None
        out.append(
            {
                **r,
                "twitch_at": t0.isoformat() if t0 else None,
                "hatch_at": t1.isoformat() if t1 else None,
                "minutes": round(minutes, 1) if minutes is not None else None,
                "temp_c": temp_c,
                "vs_sulston_min": round(minutes - SULSTON_TWITCH_TO_HATCH_MIN, 1)
                if minutes is not None
                else None,
            }
        )
    return out


def create_router(server) -> APIRouter:
    router = APIRouter()

    def _store():
        bridge = getattr(server, "agent_bridge", None)
        agent = getattr(bridge, "agent", None) if bridge is not None else None
        st = getattr(agent, "store", None) if agent is not None else None
        st = st or getattr(server, "gently_store", None)
        if st is None:
            raise HTTPException(status_code=503, detail="Store not available")
        return st

    def _sd(store, sid: str) -> Path:
        sd = store._session_dir(sid)
        if sd is None:
            raise HTTPException(status_code=404, detail="Session not found")
        return Path(sd)

    @router.get("/api/tools")
    async def list_tools():
        return {"tools": [{"name": k, **v} for k, v in TOOLS.items()]}

    @router.get("/api/sessions/{sid}/tools")
    async def get_session_tools(sid: str):
        store = _store()
        _sd(store, sid)
        return {"tools": session_tools(store, sid)}

    @router.put("/api/sessions/{sid}/tools", dependencies=[Depends(require_control)])
    async def put_session_tools(sid: str, body: dict = Body(default={})):  # noqa: B008
        store = _store()
        _sd(store, sid)
        return {"tools": set_session_tools(store, sid, list(body.get("tools") or []))}

    @router.get("/api/sessions/{sid}/tools/twitch_hatch")
    async def get_twitch_hatch(sid: str):
        store = _store()
        sd = _sd(store, sid)
        rows = _read_yaml(sd / FILE) or []
        frames = overview_frames(store, sid)
        temps = store.read_temperature_log(sid) or []
        return {
            "frames": frames,
            "rows": rows,
            "report": report(rows, frames, temps),
            "sulston_min": SULSTON_TWITCH_TO_HATCH_MIN,
        }

    @router.put("/api/sessions/{sid}/tools/twitch_hatch", dependencies=[Depends(require_control)])
    async def put_twitch_hatch(sid: str, body: dict = Body(default={})):  # noqa: B008
        store = _store()
        sd = _sd(store, sid)
        rows = []
        for r in body.get("rows") or []:
            try:
                row = {
                    "field": int(r["field"]),
                    "x": float(r["x"]),
                    "y": float(r["y"]),
                    "twitch_frame": int(r["twitch_frame"]),
                    "hatch_frame": int(r["hatch_frame"])
                    if r.get("hatch_frame") is not None
                    else None,
                    "annotator": str(r.get("annotator") or "") or None,
                }
            except (KeyError, TypeError, ValueError) as e:
                raise HTTPException(status_code=422, detail=f"bad row {r!r}: {e}") from e
            rows.append(row)
        _write_yaml(sd / FILE, rows)
        return {"rows": rows}

    return router
