"""Twitching to hatching, per embryo, off the brightfield frames.

A brightfield field holds several embryos and no embryo folders, so a row is
a click: which field, where in it, the frame it first twitched and the frame
it hatched. Minutes come from the frames' own ``captured_at`` clocks, so a
changed interval mid-session costs nothing. Temperature over the interval
comes from the session's temperature log, since a Sulston comparison without
it is not one.

Stored as ``annotations/twitch_hatch.yaml`` in the session folder; the
report is derived on read, never stored.
"""

from datetime import datetime
from pathlib import Path
from statistics import mean, median
from typing import Any

from fastapi import APIRouter, Body, Depends, HTTPException, Request

from gently.app.brightfield import overview_frames
from gently.core.file_store import _read_yaml, _write_yaml
from gently.harness.memory.notebook import Author, Note, NoteKind
from gently.tools import TOOLS, session_tools, set_session_tools
from gently.ui.web.auth import current_username, require_control

# Sulston et al. 1983, at 20 °C: twitching ~ 430 min, hatching ~ 800 min
# after first cleavage.
SULSTON_TWITCH_TO_HATCH_MIN = 370.0

FILE = Path("annotations") / "twitch_hatch.yaml"
TITLE = TOOLS["twitch_hatch"]["title"]


def _key(r: dict) -> tuple:
    """What makes a row the same row: where it was clicked, and the twitch frame."""
    return (
        r.get("field"),
        round(float(r.get("x", 0)), 4),
        round(float(r.get("y", 0)), 4),
        r.get("twitch_frame"),
    )


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


def session_report(store: Any, sid: str) -> list[dict]:
    """A session's rows with their minutes and temperature; empty when the
    session has no annotation file."""
    sd = store._session_dir(sid)
    if sd is None:
        return []
    rows = _read_yaml(Path(sd) / FILE) or []
    if not rows:
        return []
    return report(rows, overview_frames(store, sid), store.read_temperature_log(sid) or [])


def line(rep: list[dict]) -> str:
    """The session's stream line: what is marked, with the minutes as they
    stand. A template, not a model, so it says only what the rows say; the
    reading is where prose about several sessions gets written. Empty when
    nothing is marked."""
    done = [r for r in rep if r.get("minutes") is not None]
    waiting = len(rep) - len(done)
    parts: list[str] = []
    if done:
        ms = sorted(float(r["minutes"]) for r in done)
        n = len(ms)
        bit = f"Marked on {n} embryo{'' if n == 1 else 's'}: "
        bit += f"{ms[0]:g} min" if n == 1 else f"{ms[0]:g} to {ms[-1]:g} min, median {median(ms):g}"
        temps = [float(r["temp_c"]) for r in done if r.get("temp_c") is not None]
        if temps:
            bit += f", at {mean(temps):.1f} °C"
        parts.append(bit + ".")
    if waiting:
        parts.append(f"{waiting} more without a hatching time yet.")
    return " ".join(parts)


def note_marks(store: Any, cs: Any, sid: str, who: str | None) -> str | None:
    """The session's one note for this tool: filed to the session, so it is in
    the stream of every project the session is in, now or later; signed by
    whoever saved last. Written on the first save, rewritten in place after,
    gone when the last mark is taken off. Returns the note id, or None."""
    nb = getattr(cs, "notebook", None)
    if nb is None:
        return None
    # ponytail: a scan of the people's notes; give the notebook a session index when it grows.
    have = next(
        (n for n in nb.query_notes(author=Author.HUMAN) if n.title == TITLE and sid in n.sessions),
        None,
    )
    body = line(session_report(store, sid))
    if not body:
        if have is not None:
            nb.delete_note(have.id)
        return None
    note = have or Note(
        id="",
        kind=NoteKind.OBSERVATION,
        body=body,
        author=Author.HUMAN,
        title=TITLE,
        sessions=[sid],
    )
    note.body, note.by = body, who
    nid: str = nb.write_note(note)
    try:
        from gently.core.event_bus import EventType, emit

        emit(EventType.CONTEXT_UPDATED, {"kind": "note"}, source="twitch_hatch")
    except Exception:
        pass
    return nid


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
    async def put_session_tools(sid: str, request: Request, body: dict = Body(default={})):  # noqa: B008
        store = _store()
        _sd(store, sid)
        tools = set_session_tools(store, sid, list(body.get("tools") or []))
        store.note_session_edit(sid, current_username(request), "tools")
        return {"tools": tools}

    @router.get("/api/sessions/{sid}/tools/twitch_hatch")
    async def get_twitch_hatch(sid: str):
        store = _store()
        sd = _sd(store, sid)
        rows = _read_yaml(sd / FILE) or []
        frames = overview_frames(store, sid)
        return {
            "frames": frames,
            "rows": rows,
            "report": session_report(store, sid),
            "sulston_min": SULSTON_TWITCH_TO_HATCH_MIN,
        }

    @router.put("/api/sessions/{sid}/tools/twitch_hatch", dependencies=[Depends(require_control)])
    async def put_twitch_hatch(sid: str, request: Request, body: dict = Body(default={})):  # noqa: B008
        """The rows as the pane holds them. A row that is unchanged keeps who
        marked it and when; a new or changed one is stamped with this
        person, so two people marking one session keep each other's work."""
        store = _store()
        sd = _sd(store, sid)
        who = current_username(request)
        before = {_key(r): r for r in (_read_yaml(sd / FILE) or [])}
        rows = []
        for r in body.get("rows") or []:
            try:
                row: dict[str, Any] = {
                    "field": int(r["field"]),
                    "x": float(r["x"]),
                    "y": float(r["y"]),
                    "twitch_frame": int(r["twitch_frame"]),
                    "hatch_frame": int(r["hatch_frame"])
                    if r.get("hatch_frame") is not None
                    else None,
                }
            except (KeyError, TypeError, ValueError) as e:
                raise HTTPException(status_code=422, detail=f"bad row {r!r}: {e}") from e
            kept = before.get(_key(row))
            if kept is not None and kept.get("hatch_frame") == row["hatch_frame"]:
                row["by"], row["at"] = kept.get("by"), kept.get("at")
            else:
                row["by"], row["at"] = who, datetime.now().isoformat(timespec="seconds")
            rows.append(row)
        _write_yaml(sd / FILE, rows)
        note_marks(store, getattr(server, "context_store", None), sid, who)
        return {"rows": rows}

    return router
