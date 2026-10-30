"""Tools: optional panes a session imports at any time.

A tool is off until a session or a project names it under ``tools:`` in its
own yaml. Importing one is adding its name there; nothing else moves. Each
tool keeps its own files beside that yaml and derives what it shows from what
is already recorded. The scope says what a tool mounts on: a session tool
on a session (its rows live in the session folder), a project tool on a
project (its recipes or its reading live in the campaign folder). The page
mounts both the same way: ``GentlyTools[name].mount(el, id)``.
"""

from pathlib import Path
from typing import Any

TOOLS: dict[str, dict[str, str]] = {
    "twitch_hatch": {
        "scope": "session",
        "title": "Twitching to hatching",
        "summary": "Mark each embryo's first twitch and its hatching on the "
        "brightfield frames; minutes between them, against Sulston.",
    },
    "figures": {
        "scope": "project",
        "title": "Figures",
        "summary": "Plots drawn in the browser from the pooled sessions, every time "
        "they are shown; the selector is the chips.",
    },
    "reading": {
        "scope": "project",
        "title": "Reading",
        "summary": "What the sessions say about the question, in the agent's prose "
        "over the stream and the pooled rows: known, deviates, open, next.",
    },
    "data": {
        "scope": "project",
        "title": "Data",
        "summary": "Every measurement from the project's sessions, pooled per tool: "
        "one row per embryo, with its session and role; the same rows as CSV.",
    },
}


def tools_of(scope: str) -> list[str]:
    return [k for k, v in TOOLS.items() if v["scope"] == scope]


def session_tools(store: Any, session_id: str) -> list[str]:
    from gently.core.file_store import _read_yaml

    sd = store._session_dir(session_id)
    if sd is None:
        return []
    doc = _read_yaml(Path(sd) / "session.yaml") or {}
    return [t for t in (doc.get("tools") or []) if t in tools_of("session")]


def set_session_tools(store: Any, session_id: str, names: list[str]) -> list[str]:
    from gently.core.file_store import _read_yaml, _write_yaml

    sd = store._require_session_dir(session_id)
    path = Path(sd) / "session.yaml"
    doc = _read_yaml(path) or {}
    doc["tools"] = [t for t in dict.fromkeys(names) if t in tools_of("session")]
    _write_yaml(path, doc)
    return doc["tools"]


def project_tools(cs: Any, campaign_id: str) -> list[str]:
    """The project-scoped tools a project has imported, from its campaign.yaml."""
    from gently.core.file_store import _read_yaml

    folder = cs.campaign_dir(campaign_id)
    if folder is None:
        return []
    doc = _read_yaml(Path(folder) / "campaign.yaml") or {}
    return [t for t in (doc.get("tools") or []) if t in tools_of("project")]


def set_project_tools(cs: Any, campaign_id: str, names: list[str]) -> list[str]:
    from gently.core.file_store import _read_yaml, _write_yaml

    folder = cs.campaign_dir(campaign_id)
    if folder is None:
        raise KeyError(f"campaign {campaign_id} not found")
    path = Path(folder) / "campaign.yaml"
    doc = _read_yaml(path) or {}
    doc["tools"] = [t for t in dict.fromkeys(names) if t in tools_of("project")]
    _write_yaml(path, doc)
    return doc["tools"]
