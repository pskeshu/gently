"""Every change to a setting, kept.

"i want the full history of setting changes to remain on disk"

One file, ``<storage>/config/settings_history.jsonl``, one line per change,
appended and never pruned or rewritten. It lives under the storage root beside
``xy_region.yaml`` and ``spim_alignment.yaml``, so it belongs to the rig and
survives a fresh checkout of the code.

A line says what changed, from what, to what, when, how far the setting
reaches, who changed it and through which door. A secret is never written:
any value under a key that looks like one is replaced before the line is
formed, so the file can be read aloud.

Writing the history never fails the change it records. A setting that was
saved and could not be logged is still saved; the failure goes to the log.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

FILENAME = "settings_history.jsonl"
_SECRET_WORDS = ("pass", "secret", "token", "api_key", "apikey", "credential")
_REDACTED = "(not recorded)"
_lock = threading.Lock()


def path(root: Path | str | None = None) -> Path:
    """Where the history is. ``root`` is the storage root; by default the
    live one, read at call time so a redirected root is honoured."""
    if root is None:
        from gently.settings import settings

        root = settings.storage.base_path
    return Path(root) / "config" / FILENAME


def _is_secret(name: Any) -> bool:
    low = str(name).lower()
    return any(w in low for w in _SECRET_WORDS)


def redact(value: Any, name: Any = "") -> Any:
    """``value`` with every secret in it replaced. ``name`` is the key it
    sits under."""
    if _is_secret(name) and value not in (None, ""):
        return _REDACTED
    if isinstance(value, dict):
        return {k: redact(v, k) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [redact(v) for v in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def record(
    key: str,
    old: Any,
    new: Any,
    *,
    reach: str = "rig",
    via: str = "",
    by: str | None = None,
    client: str | None = None,
    session_id: str | None = None,
    label: str | None = None,
    root: Path | str | None = None,
) -> dict[str, Any] | None:
    """Append one change. Returns the line written, or None when there was
    nothing to write: a change to the same value is not a change."""
    try:
        old_r, new_r = redact(old, key), redact(new, key)
        if old_r == new_r and new_r != _REDACTED:
            return None
        entry: dict[str, Any] = {
            "at": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
            "key": str(key),
            "label": label,
            "old": old_r,
            "new": new_r,
            "reach": reach,
            "via": via,
            "by": by,
            "client": client,
            "session_id": session_id,
        }
        target = path(root)
        line = json.dumps(entry, ensure_ascii=False, default=str)
        with _lock:
            target.parent.mkdir(parents=True, exist_ok=True)
            with open(target, "a", encoding="utf-8") as f:
                f.write(line + "\n")
                f.flush()
                os.fsync(f.fileno())
        return entry
    except Exception:
        logger.warning("a settings change could not be added to the history", exc_info=True)
        return None


def record_diff(
    prefix: str, old: dict | None, new: dict | None, **kwargs: Any
) -> list[dict[str, Any]]:
    """One line per leaf that differs between two nested dicts. Keys are
    ``prefix.path.to.leaf``."""
    written: list[dict[str, Any]] = []

    def walk(a: Any, b: Any, at: str) -> None:
        if isinstance(a, dict) or isinstance(b, dict):
            a = a if isinstance(a, dict) else {}
            b = b if isinstance(b, dict) else {}
            for k in sorted(set(a) | set(b), key=str):
                walk(a.get(k), b.get(k), f"{at}.{k}" if at else str(k))
            return
        if a != b:
            entry = record(at, a, b, **kwargs)
            if entry:
                written.append(entry)

    walk(old or {}, new or {}, prefix)
    return written


def read(
    limit: int = 200, key: str | None = None, root: Path | str | None = None
) -> list[dict[str, Any]]:
    """The history, newest first. A line that cannot be read is skipped, not
    fatal: the file is appended to by hand-restartable processes."""
    target = path(root)
    if not target.exists():
        return []
    out: list[dict[str, Any]] = []
    with open(target, encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if not raw:
                continue
            try:
                entry = json.loads(raw)
            except ValueError:
                continue
            if key and not str(entry.get("key", "")).startswith(key):
                continue
            out.append(entry)
    out.reverse()
    return out[: max(0, int(limit))] if limit else out


def count(root: Path | str | None = None) -> int:
    target = path(root)
    if not target.exists():
        return 0
    with open(target, encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())
