"""Who works at this rig, on a rig with no accounts.

A person is an id that never changes and a display name that can. The id
is made from the name the first time it is given (``hari`` from "Hari") and
is what every record carries as ``by``; the name is what the pages show,
and a typo in it is fixed in one place. Names given at the landing page
are kept in ``config/operators.yaml`` under the storage root; an id found
on a record that the file does not know shows as itself.
"""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from typing import Any

from gently.core.file_store import _read_yaml, _write_yaml
from gently.settings import settings

NOT_PEOPLE = {"agent", "backfill", "perception"}


def slug(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", (name or "").lower()).strip("-")
    return s[:40]


def operators_path() -> Path:
    return Path(settings.storage.base_path) / "config" / "operators.yaml"


def _rows() -> list[dict]:
    doc = _read_yaml(operators_path()) or {}
    rows = [r for r in (doc.get("operators") or []) if isinstance(r, dict) and r.get("name")]
    for r in rows:
        r.setdefault("id", slug(str(r["name"])))
    return rows


def _save(rows: list[dict]) -> None:
    path = operators_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_yaml(path, {"operators": rows})


def list_operators(store: Any = None) -> list[dict]:
    """``[{"id", "name"}]``, most recently seen first, then any id on a
    session's links that the file does not know, shown as itself."""
    rows = sorted(_rows(), key=lambda r: str(r.get("last_seen") or ""), reverse=True)
    out = [{"id": str(r["id"]), "name": str(r["name"])} for r in rows]
    known = {o["id"] for o in out}
    if store is not None:
        try:
            for s in store.list_sessions():
                for link in s.get("projects") or []:
                    by = link.get("by")
                    if by and by not in NOT_PEOPLE and by not in known:
                        out.append({"id": str(by), "name": str(by)})
                        known.add(str(by))
        except Exception:
            pass
    return out


def names_of(store: Any = None) -> dict[str, str]:
    return {o["id"]: o["name"] for o in list_operators(store)}


def remember_operator(name: str) -> dict:
    """A name given at the landing: the person it names is seen now. A new
    name makes a new person; a known one, by id, keeps its display name."""
    name = (name or "").strip()[:40]
    if not name:
        raise ValueError("a name is needed")
    pid = slug(name)
    if not pid:
        raise ValueError("a name needs a letter or a digit")
    rows = _rows()
    now = datetime.now().isoformat(timespec="seconds")
    for r in rows:
        if str(r["id"]) == pid:
            r["last_seen"] = now
            _save(rows)
            return {"id": pid, "name": str(r["name"])}
    rows.append({"id": pid, "name": name, "first_seen": now, "last_seen": now})
    _save(rows)
    return {"id": pid, "name": name}


def touch_operator(pid: str) -> dict:
    """A known person, by id, is here now."""
    pid = (pid or "").strip()
    rows = _rows()
    for r in rows:
        if str(r["id"]) == pid:
            r["last_seen"] = datetime.now().isoformat(timespec="seconds")
            _save(rows)
            return {"id": pid, "name": str(r["name"])}
    raise KeyError(pid)


def rename_operator(pid: str, name: str) -> dict:
    """The display name changes; the id, and so every record, stays."""
    name = (name or "").strip()[:40]
    pid = (pid or "").strip()
    if not pid or not name:
        raise ValueError("an id and a name are needed")
    rows = _rows()
    now = datetime.now().isoformat(timespec="seconds")
    for r in rows:
        if str(r["id"]) == pid:
            r["name"] = name
            r["last_seen"] = now
            _save(rows)
            return {"id": pid, "name": name}
    rows.append({"id": pid, "name": name, "first_seen": now, "last_seen": now})
    _save(rows)
    return {"id": pid, "name": name}
