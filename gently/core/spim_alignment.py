"""Where the SPIM head actually looks, relative to the bottom camera's centre.

THE ASSUMPTION THIS REPLACES

`frameToStage` defines an embryo's stage position as *the stage XY at which it
sits at the bottom camera's centre pixel*, and `centerOnEmbryo` drives the
stage to exactly that. Nothing mapped that point to the SPIM's optical axis,
so the code assumed the two coincided — exactly, permanently, and without
anyone ever having measured it.

When they do not coincide, centring puts the embryo outside the light sheet
and the pre-calibration check reports "No object visible" while every
instrument check passes: laser on, head down, stage where it was asked to go.
There is nothing in that picture to look at, which is what makes an unmeasured
assumption worse than a wrong number.

WHAT IS STORED

One offset in microns, `(dx, dy)`, added when centring an embryo. Default
`(0, 0)` — the old assumption, stated rather than implied, so an instrument
nobody has aligned behaves exactly as before.

HOW IT IS MEASURED

Centre on an embryo, jog XY until it sits where you want it under the SPIM,
and record `current_stage - embryo_position`. The embryo is the fiducial: a
bare stage XY would only hold until the sample moved.

The embryo is NOT the provenance, though. `embryo_4` is a per-session label
that means nothing next week, so what is written down is the session it was
measured in.

NOTHING IS EVER OVERWRITTEN

Every change appends the previous value to `history`. Restoring is choosing a
row and making it current, which itself appends. An alignment is a physical
fact about the instrument on a day; the record of when it changed is worth as
much as the value.

RELATED, AND DELIBERATELY SEPARATE

`EmbryoState.position_fine` is a per-embryo refinement slot ("SPIM-objective
alignment workflow (not built yet)"). This is the systematic term — one pair
of numbers for the instrument — and the two compose: a fine position would
still be centred through this offset.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from gently.settings import settings

logger = logging.getLogger(__name__)

# How many past alignments to keep. An alignment changes when someone re-seats
# the head or re-mounts the camera — rare events — so this is years of them.
MAX_HISTORY = 50


@dataclass
class Alignment:
    """One measured offset, and where it came from."""

    dx_um: float = 0.0
    dy_um: float = 0.0
    set_at: str | None = None
    session_id: str | None = None
    note: str = ""

    @property
    def is_measured(self) -> bool:
        """False while the instrument is running on the old assumption.

        Not the same as "the offset is zero": a measured (0, 0) means someone
        checked and the axes coincide. An unmeasured one means nobody knows.
        """
        return self.set_at is not None

    def to_dict(self) -> dict[str, Any]:
        return {
            "dx_um": self.dx_um,
            "dy_um": self.dy_um,
            "set_at": self.set_at,
            "session_id": self.session_id,
            "note": self.note,
            "is_measured": self.is_measured,
        }


@dataclass
class AlignmentRecord:
    """The current alignment plus every one it replaced."""

    current: Alignment = field(default_factory=Alignment)
    history: list[Alignment] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "current": self.current.to_dict(),
            "history": [h.to_dict() for h in self.history],
        }


def alignment_path() -> Path:
    """`<storage root>/config/spim_alignment.yaml`.

    The storage root, not the repo: an alignment is a fact about the
    microscope on this bench, and must survive a checkout, a branch and a
    reinstall. It sits beside the other machine-level config.
    """
    return settings.storage.base_path / "config" / "spim_alignment.yaml"


def _as_alignment(raw: Any) -> Alignment:
    if not isinstance(raw, dict):
        return Alignment()
    try:
        return Alignment(
            dx_um=float(raw.get("dx_um") or 0.0),
            dy_um=float(raw.get("dy_um") or 0.0),
            set_at=raw.get("set_at"),
            session_id=raw.get("session_id"),
            note=str(raw.get("note") or ""),
        )
    except (TypeError, ValueError):
        logger.warning("Unreadable alignment entry, treating as unmeasured: %r", raw)
        return Alignment()


def load() -> AlignmentRecord:
    """Read the alignment, or the old assumption if there is none.

    Never raises. A missing or corrupt file means the instrument behaves as it
    did before this existed — which is the safe direction, because the failure
    mode of a bad offset is a stage move to nowhere.
    """
    path = alignment_path()
    try:
        if not path.exists():
            return AlignmentRecord()
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        logger.exception("Could not read %s — using the unmeasured default", path)
        return AlignmentRecord()

    history_raw = raw.get("history")
    history = [_as_alignment(h) for h in history_raw] if isinstance(history_raw, list) else []
    return AlignmentRecord(current=_as_alignment(raw.get("current")), history=history)


def save(record: AlignmentRecord) -> bool:
    """Write the whole record. Returns whether it reached disk."""
    path = alignment_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "current": {
                "dx_um": record.current.dx_um,
                "dy_um": record.current.dy_um,
                "set_at": record.current.set_at,
                "session_id": record.current.session_id,
                "note": record.current.note,
            },
            "history": [
                {
                    "dx_um": h.dx_um,
                    "dy_um": h.dy_um,
                    "set_at": h.set_at,
                    "session_id": h.session_id,
                    "note": h.note,
                }
                for h in record.history[-MAX_HISTORY:]
            ],
        }
        path.write_text(
            yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8"
        )
        return True
    except Exception:
        logger.exception("Could not write the SPIM alignment to %s", path)
        return False


def set_offset(
    dx_um: float,
    dy_um: float,
    *,
    session_id: str | None = None,
    note: str = "",
    now: datetime | None = None,
) -> AlignmentRecord:
    """Make this the current alignment, keeping the one it replaces.

    The previous value is appended to history even when it was never measured,
    so the record shows the day the instrument stopped running on an
    assumption.
    """
    record = load()
    previous = record.current
    record.history.append(previous)
    record.current = Alignment(
        dx_um=float(dx_um),
        dy_um=float(dy_um),
        set_at=(now or datetime.now()).isoformat(timespec="seconds"),
        session_id=session_id,
        note=note,
    )
    save(record)
    return record


def restore(set_at: str | None, *, session_id: str | None = None) -> AlignmentRecord | None:
    """Bring a past alignment back as the current one.

    Restoring appends, like any other change: the alignment being replaced
    goes into history too, so a restore can itself be undone. Returns None
    when no history entry matches — the caller reports that rather than
    silently doing nothing.

    `set_at` of None selects the unmeasured default, which is how an operator
    gets back to "no correction" without inventing a zero.
    """
    record = load()
    match = next((h for h in record.history if h.set_at == set_at), None)
    if match is None:
        return None
    return set_offset(
        match.dx_um,
        match.dy_um,
        session_id=session_id,
        note=f"restored from {match.set_at or 'the unmeasured default'}",
    )


def centre_target(x: float, y: float, record: AlignmentRecord | None = None) -> tuple[float, float]:
    """Where the stage goes to put (x, y) under the SPIM.

    The single place the correction is applied on the server side. `(0, 0)`
    leaves the coordinates untouched, so an unmeasured instrument moves
    exactly where it always did.
    """
    align = (record or load()).current
    return x + align.dx_um, y + align.dy_um


async def move_to_embryo(client, position: dict, record: AlignmentRecord | None = None) -> Any:
    """Drive the stage so this embryo sits on the SPIM axis.

    The one place the correction meets the hardware. Five call sites used to
    pass `pos["x"], pos["y"]` straight to `move_to_position` — calibration,
    timelapse acquisition, the acquisition tools, the exclusive orchestrator —
    and any one of them left uncorrected would drive somewhere the others do
    not, which reads as "calibration works from the pane but not from chat".

    Deliberately NOT used by `stage_tools` or the stage-move route: those mean
    "put the stage at this coordinate", and silently adding an offset to an
    explicit coordinate would be a lie.
    """
    x, y = centre_target(float(position["x"]), float(position["y"]), record)
    return await client.move_to_position(x, y)
