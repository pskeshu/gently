"""The reading: what the project's nights say, written between nights.

The stream is the notebook filed to the project: lines in the operator's own
words, dated, no kind at write time. The reading is the agent's prose over
that stream, the nights and the pooled rows, four short sections, every
claim pointing at a night or a line. It is stored under the project as
``reading.md`` with a ``reading.yaml`` saying when it was written and over
what, so the page can tell when nights have landed since.

ponytail: one model call, no streaming, no images; the pot is text. Frames
join the pot when a reading needs to look at an embryo.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Any

from fastapi import APIRouter, Body, Depends, HTTPException

from gently.core.file_store import _read_yaml, _write_yaml
from gently.settings import settings
from gently.tools.pool import POOLERS, REFERENCES, pool
from gently.ui.web.auth import require_control

logger = logging.getLogger(__name__)

SYSTEM = """\
You are the between: the thinking a microscopist does between nights at a diSPIM light-sheet
imaging C. elegans embryos. You are handed one project, its question, the nights filed under
it, the pooled measurements, and the stream, which is what the people at the rig wrote down
in their own words as they went.

Write four short sections in plain prose, each a paragraph or a few lines, no bullets, no
headings other than the four words below. Cite nights by name and stream lines by their
date. A sentence that could have been written without this instrument and these embryos
does not belong; a textbook fact is a basis to compare against, never a finding.

KNOWN. What these nights say about the question, with the numbers.
DEVIATES. Where a night or an embryo departs from the others, or from the reference given,
with magnitude and direction. If nothing does, say so in one line.
OPEN. What the question still lacks, concretely: which measurement, how many embryos, which
window. Lines in the stream that nobody has answered belong here.
NEXT. Up to three nights to run next, each one sentence in the rig's register, then one
clause on why.

If a night does not bear on the question, say so rather than forcing it. Short is right."""

ASK = "What do we now know, and what next?"


def _runs(stages: list[tuple[int, str]]) -> str:
    """[(t, stage)] -> 'bean t1-2, comma t3-4': the trajectory, run-length."""
    out: list[list] = []
    for t, s in stages:
        if out and out[-1][0] == s:
            out[-1][2] = t
        else:
            out.append([s, t, t])
    return ", ".join(f"{s} t{a}" if a == b else f"{s} t{a}-{b}" for s, a, b in out)


def _night(store: Any, s: dict) -> str:
    sid = s["session_id"]
    head = f"NIGHT {s.get('name') or sid} ({sid[:8]}), {str(s.get('created_at') or '')[:16]}"
    if s.get("tags"):
        head += ", tags " + ", ".join(s["tags"])
    lines = [head]
    for e in store.list_embryos(sid) or []:
        eid = e.get("embryo_id")
        try:
            preds = store.get_predictions(sid, eid) or []
        except Exception:
            preds = []
        traj = _runs([(int(p.get("timepoint", 0)), str(p.get("predicted_stage"))) for p in preds])
        lines.append(
            f"  {e.get('nickname') or eid} ({e.get('role') or 'role unset'}): "
            f"{traj or 'no stage calls'}"
        )
    return "\n".join(lines)


def stream_notes(cs: Any, project_id: str, member_ids: set[str]) -> list:
    """The project's stream: notes filed to it, or to one of its nights."""
    nb = cs.notebook
    seen: dict[str, Any] = {}
    for n in nb.query_notes(project=project_id):
        seen[n.id] = n
    for n in nb.query_notes():
        if n.id not in seen and member_ids & set(n.sessions or []):
            seen[n.id] = n
    return sorted(seen.values(), key=lambda n: n.created_at)


def pot(store: Any, cs: Any, project_id: str) -> dict:
    """Everything the reading is written over, as text plus what it covered."""
    c = cs.get_campaign(project_id)
    if c is None:
        raise KeyError(f"project {project_id} not found")
    sessions = store.list_sessions(project_id=c.id)
    members = {s["session_id"] for s in sessions}
    parts = [f"PROJECT {c.shorthand or ''}".rstrip(), c.description or ""]
    if c.target:
        parts.append(f"Target: {c.target}")
    parts.append("")
    for s in sorted(sessions, key=lambda s: s.get("created_at") or ""):
        parts.append(_night(store, s))
    for tool in POOLERS:
        p = pool(store, cs, tool, project=c.id)
        rows = [r for r in p["rows"] if r.get("minutes") is not None]
        if not rows:
            continue
        parts.append("")
        parts.append(f"POOLED {tool}: minutes from first twitch to hatching, per embryo")
        for name in sorted({r["session_name"] for r in rows}):
            ms = [r["minutes"] for r in rows if r["session_name"] == name]
            parts.append(f"  {name}: {', '.join(f'{m:g}' for m in ms)} (median {median(ms):g})")
        for ref, v in REFERENCES.get(tool, {}).items():
            parts.append(f"  reference: {ref} {v:g} min")
    notes = stream_notes(cs, c.id, members)
    if notes:
        parts.append("")
        parts.append("STREAM")
        for n in notes:
            when = n.created_at.strftime("%Y-%m-%d %H:%M") if n.created_at else ""
            who = "agent" if getattr(n.author, "value", n.author) == "agent" else "person"
            parts.append(f"  {when} ({who}) {n.body}")
    return {
        "text": "\n".join(parts),
        "nights": sorted(members),
        "notes": [n.id for n in notes],
        "project": c.id,
    }


def _paths(cs: Any, project_id: str) -> tuple[Path, Path]:
    folder = cs.campaign_dir(project_id)
    if folder is None:
        raise KeyError(f"project {project_id} not found")
    return Path(folder) / "reading.md", Path(folder) / "reading.yaml"


def load_reading(store: Any, cs: Any, project_id: str) -> dict:
    """The stored reading and how stale it is: nights filed since it was written."""
    md, meta_path = _paths(cs, project_id)
    if not md.is_file():
        return {"text": None, "meta": None, "nights_since": 0}
    meta = _read_yaml(meta_path) or {}
    covered = set(meta.get("nights") or [])
    now = {s["session_id"] for s in store.list_sessions(project_id=project_id)}
    return {
        "text": md.read_text(encoding="utf-8"),
        "meta": meta,
        "nights_since": len(now - covered),
    }


def read_project(store: Any, cs: Any, project_id: str, model: str | None = None) -> dict:
    """Write the reading: one model call over the pot, filed under the project."""
    import anthropic

    p = pot(store, cs, project_id)
    client = anthropic.Anthropic()
    model = model or settings.models.main
    response = client.messages.create(
        model=model,
        max_tokens=4000,
        thinking={"type": "adaptive"},
        system=SYSTEM,
        messages=[{"role": "user", "content": p["text"] + "\n\n" + ASK}],
    )
    if response.stop_reason == "refusal":
        raise RuntimeError("the model declined to write the reading")
    text = "".join(b.text for b in response.content if b.type == "text").strip()
    md, meta_path = _paths(cs, p["project"])
    meta = {
        "written_at": datetime.now().isoformat(timespec="seconds"),
        "model": model,
        "nights": p["nights"],
        "notes": p["notes"],
    }
    md.write_text(text, encoding="utf-8")
    _write_yaml(meta_path, meta)
    return {"text": text, "meta": meta, "nights_since": 0}


def create_router(server) -> APIRouter:
    router = APIRouter()

    def _stores(cid: str):
        cs = getattr(server, "context_store", None)
        bridge = getattr(server, "agent_bridge", None)
        agent = getattr(bridge, "agent", None) if bridge is not None else None
        store = (getattr(agent, "store", None) if agent is not None else None) or getattr(
            server, "gently_store", None
        )
        c = cs.resolve_campaign(cid) if cs is not None else None
        if store is None or cs is None or c is None:
            raise HTTPException(status_code=404, detail="Project not found")
        return store, cs, cs.root_campaign_id(c.id) or c.id

    @router.get("/api/campaigns/{cid}/stream")
    async def get_stream(cid: str):
        """The project's stream as the reading sees it: lines filed to the
        project, and lines filed to one of its nights."""
        from gently.harness.memory.notebook import note_to_dict

        store, cs, root = _stores(cid)
        members = {s["session_id"] for s in store.list_sessions(project_id=root)}
        return {"notes": [note_to_dict(n) for n in stream_notes(cs, root, members)]}

    @router.get("/api/campaigns/{cid}/reading")
    async def get_reading(cid: str):
        store, cs, root = _stores(cid)
        return load_reading(store, cs, root)

    @router.post("/api/campaigns/{cid}/reading", dependencies=[Depends(require_control)])
    async def write_reading(cid: str, body: dict = Body(default={})):  # noqa: B008
        store, cs, root = _stores(cid)
        try:
            return read_project(store, cs, root, model=body.get("model"))
        except Exception as e:  # the model, the key, the network: say which
            logger.warning("reading failed for %s: %s", root, e)
            raise HTTPException(status_code=502, detail=f"No reading: {e}") from e

    return router
