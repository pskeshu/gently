"""Every action a person takes on the rig, in the session it was taken in.

A control request that changed something, stage, light, run, filing, a
note, is one line in ``actions.jsonl`` under the live session: when, who,
from where, what was asked. It is written by one middleware for every
control route at once, so a new knob is covered the day it is added, and
it answers "who moved the stage at 02:14" without any route knowing.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

WRITES = {"POST", "PUT", "PATCH", "DELETE"}
# Requests that are not a person acting on the rig: identity, the UI's own
# bookkeeping, and the viewer's image pipes.
QUIET = (
    "/api/auth/",
    "/api/settings/history",
    "/api/images",
    "/api/volumes3d",
    "/api/ui/",
    "/replay/",
)


def is_action(method: str, path: str) -> bool:
    return method in WRITES and path.startswith("/api/") and not path.startswith(QUIET)


def record_action(server: Any, request: Any, status: int) -> None:
    """One line for a control request that succeeded, under the live session."""
    try:
        if status >= 400:
            return
        sid = server._current_session_id()
        store = getattr(server, "gently_store", None)
        if not sid or store is None:
            return
        sd = store._session_dir(sid)
        if sd is None:
            return
        from gently.ui.web.auth import current_username

        line = {
            "at": datetime.now().isoformat(timespec="seconds"),
            "by": current_username(request),
            "client": getattr(getattr(request, "client", None), "host", None),
            "method": request.method,
            "path": request.url.path,
            "query": str(request.url.query or "") or None,
            "status": status,
        }
        with open(Path(sd) / "actions.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(line) + "\n")
    except Exception:
        logger.debug("could not record the action", exc_info=True)


def read_actions(store: Any, session_id: str, limit: int = 200) -> list[dict]:
    """The last ``limit`` actions of a session, oldest first."""
    sd = store._session_dir(session_id)
    if sd is None:
        return []
    path = Path(sd) / "actions.jsonl"
    if not path.is_file():
        return []
    out: list[dict] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        raw = raw.strip()
        if not raw:
            continue
        try:
            out.append(json.loads(raw))
        except json.JSONDecodeError:
            continue
    return out[-limit:] if limit else out
