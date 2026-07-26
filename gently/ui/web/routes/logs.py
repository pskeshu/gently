"""
Process console routes
======================

Serves the two process consoles the header drawer shows.

The agent's stdout is genuinely unreachable in the packaged desktop app — the
Tauri shell spawns the backend with ``CREATE_NO_WINDOW`` in release, so there is
no console to look at when something misbehaves mid-session. Its output does
still land in ``{storage}/logs/gently_*.log``, so that file is the console.

The device layer already has a live captured-stdout tail on its supervisor
(``/api/device-layer/log``); this module does not duplicate it. It only adds the
file-backed agent side, plus a fallback tail of the device layer's own log file
for the case where the layer runs externally and the supervisor captured
nothing.
"""

from __future__ import annotations

import logging
import re
from collections import deque
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query

logger = logging.getLogger(__name__)

MAX_LINES = 2000
# Read only the tail of the file. Session logs can reach tens of MB and the
# drawer never shows more than a couple of thousand lines.
TAIL_BYTES = 512 * 1024

# "2026-07-19 11:04:22 gently.app.agent INFO message" — the default file format.
_LEVEL_RE = re.compile(r"\b(DEBUG|INFO|WARNING|ERROR|CRITICAL)\b")

_SOURCES = {
    "agent": "gently_*.log",
    "device": "device_layer_*.log",
}


def _log_dir() -> Path:
    from gently.settings import settings

    return Path(settings.storage.base_path) / "logs"


def _newest(pattern: str) -> Path | None:
    d = _log_dir()
    if not d.is_dir():
        return None
    files = [p for p in d.glob(pattern) if p.is_file()]
    if not files:
        return None
    return max(files, key=lambda p: p.stat().st_mtime)


def _tail(path: Path, limit: int) -> list[str]:
    """Last ``limit`` lines, reading only the final chunk of the file."""
    with path.open("rb") as fh:
        fh.seek(0, 2)
        size = fh.tell()
        fh.seek(max(0, size - TAIL_BYTES))
        chunk = fh.read()
    text = chunk.decode("utf-8", errors="replace")
    # A partial first line is likely when the file is larger than the window.
    lines = text.splitlines()
    if size > TAIL_BYTES and lines:
        lines = lines[1:]
    return list(deque(lines, maxlen=limit))


def create_router(server) -> APIRouter:  # noqa: ARG001 — parity with sibling modules
    router = APIRouter()

    @router.get("/api/logs/{source}")
    async def read_log(
        source: str,
        limit: int = Query(400, ge=1, le=MAX_LINES),
        level: str | None = Query(None, description="Minimum level: INFO|WARNING|ERROR"),
    ):
        """Tail a process log (oldest → newest), optionally filtered by level.

        Returns ``{source, file, lines, truncated}``. A missing log directory or
        no matching file is not an error — the console shows an empty state and
        keeps polling, because the process may simply not have started yet.
        """
        pattern = _SOURCES.get(source)
        if pattern is None:
            raise HTTPException(status_code=404, detail=f"unknown log source: {source}")

        path = _newest(pattern)
        if path is None:
            return {"source": source, "file": None, "lines": [], "truncated": False}

        try:
            lines = _tail(path, limit)
        except OSError as exc:
            logger.warning("could not read %s log at %s: %s", source, path, exc)
            raise HTTPException(status_code=502, detail=f"log read failed: {exc}") from exc

        if level:
            wanted = level.upper()
            order = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
            if wanted in order:
                keep = set(order[order.index(wanted) :])
                # Lines without a level are continuations (tracebacks) — keeping
                # them preserves the stack under the ERROR that introduced it.
                lines = [
                    ln for ln in lines if not (m := _LEVEL_RE.search(ln)) or m.group(1) in keep
                ]

        return {
            "source": source,
            "file": path.name,
            "lines": lines,
            "truncated": len(lines) >= limit,
        }

    return router
