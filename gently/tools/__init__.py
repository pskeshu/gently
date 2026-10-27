"""Tools: optional panes a session imports at any time.

A tool is off until a session names it in its ``session.yaml`` under
``tools:``. Importing one is adding its name there; nothing else moves. Each
tool keeps its own file under the session folder and derives its report from
what the session already recorded.

ponytail: one tool so far; a loader appears when the second one shows what
they share.
"""

from pathlib import Path
from typing import Any

TOOLS: dict[str, dict[str, str]] = {
    "twitch_hatch": {
        "title": "Twitching to hatching",
        "summary": "Mark each embryo's first twitch and its hatching on the "
        "brightfield frames; minutes between them, against Sulston.",
    },
}


def session_tools(store: Any, session_id: str) -> list[str]:
    from gently.core.file_store import _read_yaml

    sd = store._session_dir(session_id)
    if sd is None:
        return []
    doc = _read_yaml(Path(sd) / "session.yaml") or {}
    return [t for t in (doc.get("tools") or []) if t in TOOLS]


def set_session_tools(store: Any, session_id: str, names: list[str]) -> list[str]:
    from gently.core.file_store import _read_yaml, _write_yaml

    sd = store._require_session_dir(session_id)
    path = Path(sd) / "session.yaml"
    doc = _read_yaml(path) or {}
    doc["tools"] = [t for t in dict.fromkeys(names) if t in TOOLS]
    _write_yaml(path, doc)
    return doc["tools"]
