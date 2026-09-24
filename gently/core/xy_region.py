"""The XY working region, and every region it replaced.

WHERE IT LIVED, AND WHY THAT WAS A PROBLEM

The region was persisted by the device layer into `config.local.yml`, resolved
from `config_path`, whose default is the **relative** path `config/config.yml`.
Relative to the process's working directory — so the saved region lived
wherever the device layer happened to be started from. Start it from a
worktree, a service, or a shell opened somewhere else, and the region silently
is not there: boot falls back to the code defaults and nobody is told.

On the rig this was not hypothetical. When the region editor was first opened
there was no `config.local.yml` anywhere on the machine, and boot had been
writing full travel every time.

That is survivable for one value someone can re-walk. It is not survivable for
a HISTORY, which is the whole point of being able to go back: a list of past
regions you can lose by launching differently is not a list you can rely on.

So the record lives in the storage root, beside the SPIM alignment, which is
where facts about this bench belong.

NOTHING IS OVERWRITTEN

Applying a region appends the one it replaced. Restoring is itself an apply,
so it appends too — which is what makes going back and forth possible rather
than one-way.
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

BOUNDS = ("x_min", "x_max", "y_min", "y_max")
MAX_HISTORY = 50


@dataclass
class Region:
    """One working region, and where it came from."""

    x_min: float
    x_max: float
    y_min: float
    y_max: float
    applied_at: str | None = None
    session_id: str | None = None
    note: str = ""

    @property
    def box(self) -> dict[str, float]:
        return {k: float(getattr(self, k)) for k in BOUNDS}

    def to_dict(self) -> dict[str, Any]:
        d: dict[str, Any] = dict(self.box)
        d.update(applied_at=self.applied_at, session_id=self.session_id, note=self.note)
        return d


@dataclass
class RegionRecord:
    current: Region | None = None
    history: list[Region] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "current": self.current.to_dict() if self.current else None,
            "history": [h.to_dict() for h in self.history],
        }


def region_path() -> Path:
    """`<storage root>/config/xy_region.yaml` — not a cwd-relative sidecar."""
    return settings.storage.base_path / "config" / "xy_region.yaml"


def _as_region(raw: Any) -> Region | None:
    if not isinstance(raw, dict):
        return None
    try:
        return Region(
            **{k: float(raw[k]) for k in BOUNDS},
            applied_at=raw.get("applied_at"),
            session_id=raw.get("session_id"),
            note=str(raw.get("note") or ""),
        )
    except (KeyError, TypeError, ValueError):
        logger.warning("Unreadable region entry, ignoring: %r", raw)
        return None


def load() -> RegionRecord:
    """Read the record. Never raises — an unreadable file means "no region".

    Which is the safe direction: no region is the stage's full travel, and a
    bad one is a fence in the wrong place.
    """
    path = region_path()
    try:
        if not path.exists():
            return RegionRecord()
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        logger.exception("Could not read %s — treating as no region", path)
        return RegionRecord()
    hist_raw = raw.get("history")
    history = [r for r in (_as_region(h) for h in hist_raw or []) if r] if hist_raw else []
    return RegionRecord(current=_as_region(raw.get("current")), history=history)


def save(record: RegionRecord) -> bool:
    path = region_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "current": record.current.to_dict() if record.current else None,
            "history": [h.to_dict() for h in record.history[-MAX_HISTORY:]],
        }
        path.write_text(
            yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8"
        )
        return True
    except Exception:
        logger.exception("Could not write the XY region to %s", path)
        return False


def apply(
    box: dict[str, float],
    *,
    session_id: str | None = None,
    note: str = "",
    now: datetime | None = None,
) -> RegionRecord:
    """Make this the current region, keeping the one it replaces."""
    record = load()
    if record.current is not None:
        record.history.append(record.current)
    record.current = Region(
        **{k: float(box[k]) for k in BOUNDS},
        applied_at=(now or datetime.now()).isoformat(timespec="seconds"),
        session_id=session_id,
        note=note,
    )
    save(record)
    return record


def restore(applied_at: str, *, session_id: str | None = None) -> RegionRecord | None:
    """Bring a past region back as the current one.

    An apply like any other, so the region being replaced joins the history
    and the restore can itself be undone. Returns None when nothing matches —
    the caller says so rather than silently doing nothing.
    """
    record = load()
    match = next((h for h in record.history if h.applied_at == applied_at), None)
    if match is None:
        return None
    return apply(
        match.box,
        session_id=session_id,
        note=f"restored from {match.applied_at}",
    )
