"""Who works at this rig, on a rig with no accounts.

A name given at the landing page is remembered in ``config/operators.yaml``
under the storage root, and names already on records (who filed a session)
count too. The landing offers them as picks; the cookie says who is here
now, and ``current_username`` reads it.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from gently.core.file_store import _read_yaml, _write_yaml
from gently.settings import settings

NOT_PEOPLE = {"agent", "backfill", "perception"}


def operators_path() -> Path:
    return Path(settings.storage.base_path) / "config" / "operators.yaml"


def list_operators(store: Any = None) -> list[str]:
    """Names in the operators file, most recently seen first, then any name on
    a session's links that the file does not have yet."""
    doc = _read_yaml(operators_path()) or {}
    rows = [r for r in (doc.get("operators") or []) if isinstance(r, dict) and r.get("name")]
    rows.sort(key=lambda r: str(r.get("last_seen") or ""), reverse=True)
    names = [str(r["name"]) for r in rows]
    if store is not None:
        try:
            for s in store.list_sessions():
                for link in s.get("projects") or []:
                    by = link.get("by")
                    if by and by not in NOT_PEOPLE and by not in names:
                        names.append(str(by))
        except Exception:
            pass
    return names


def remember_operator(name: str) -> str:
    """Record a name as seen now. Returns it trimmed."""
    name = (name or "").strip()[:40]
    if not name:
        raise ValueError("a name is needed")
    path = operators_path()
    doc = _read_yaml(path) or {}
    rows = [r for r in (doc.get("operators") or []) if isinstance(r, dict) and r.get("name")]
    now = datetime.now().isoformat(timespec="seconds")
    for r in rows:
        if str(r["name"]).lower() == name.lower():
            r["last_seen"] = now
            break
    else:
        rows.append({"name": name, "first_seen": now, "last_seen": now})
    path.parent.mkdir(parents=True, exist_ok=True)
    _write_yaml(path, {"operators": rows})
    return name
