"""Figures: what the agent drew for a project, filed under it.

A figure is a file in the campaign's ``figures/`` folder. Two kinds live
there. A PNG the agent drew (``draw_figure`` in
``gently/app/tools/figure_tools.py``), served as it is. And a recipe,
``{name}.yaml``: which tool, which phase or tag within the project, what on
x and y. A recipe is drawn in the browser from the pooled rows each time it
is shown, so it is as current as the sessions. This module files and serves
both; it draws nothing.
"""

import re
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Body, Depends, HTTPException, Request
from fastapi.responses import Response

from gently.core.file_store import _read_yaml, _write_yaml
from gently.tools import project_tools, set_project_tools
from gently.ui.web.auth import current_username, require_control

MEDIA = {".png": "image/png", ".svg": "image/svg+xml"}
RECIPE_FIELDS = ("title", "tool", "phase", "tag", "x", "y")
RECIPE_META = ("by", "edited")


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:40] or "figure"


def figure_path(cs: Any, campaign_id: str, name: str, ext: str = ".png") -> Path:
    folder = cs.campaign_dir(campaign_id)
    if folder is None:
        raise KeyError(f"campaign {campaign_id} not found")
    path = Path(folder) / "figures" / f"{slug(name)}{ext}"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def figure_url(campaign_id: str, name: str, ext: str = ".png") -> str:
    return f"/api/campaigns/{campaign_id}/figures/{slug(name)}{ext}"


def create_router(server) -> APIRouter:
    router = APIRouter()

    def _dir(cid: str) -> Path:
        cs = getattr(server, "context_store", None)
        c = cs.resolve_campaign(cid) if cs is not None else None
        folder = cs.campaign_dir(c.id) if cs is not None and c is not None else None
        if folder is None:
            raise HTTPException(status_code=404, detail="Campaign not found")
        return Path(folder) / "figures"

    def _root(cid: str) -> str:
        cs = getattr(server, "context_store", None)
        c = cs.resolve_campaign(cid) if cs is not None else None
        if cs is None or c is None:
            raise HTTPException(status_code=404, detail="Campaign not found")
        return cs.root_campaign_id(c.id) or c.id

    @router.get("/api/campaigns/{cid}/tools")
    async def get_project_tools(cid: str):
        return {"tools": project_tools(server.context_store, _root(cid))}

    @router.put("/api/campaigns/{cid}/tools", dependencies=[Depends(require_control)])
    async def put_project_tools(cid: str, body: dict = Body(default={})):  # noqa: B008
        root = _root(cid)
        return {
            "tools": set_project_tools(server.context_store, root, list(body.get("tools") or []))
        }

    @router.get("/api/campaigns/{cid}/figures")
    async def list_figures(cid: str):
        d = _dir(cid)
        files = [p for p in d.glob("*") if p.suffix in MEDIA] if d.is_dir() else []
        files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        return {
            "figures": [
                {
                    "name": p.stem,
                    "url": figure_url(cid, p.stem, p.suffix),
                    "modified": p.stat().st_mtime,
                }
                for p in files
            ]
        }

    @router.get("/api/campaigns/{cid}/recipes")
    async def list_recipes(cid: str):
        d = _dir(cid)
        out = []
        for p in sorted(d.glob("*.yaml")) if d.is_dir() else []:
            doc = _read_yaml(p) or {}
            out.append({"name": p.stem, **{k: doc.get(k) for k in RECIPE_FIELDS + RECIPE_META}})
        return {"recipes": out}

    @router.put("/api/campaigns/{cid}/recipes/{name}", dependencies=[Depends(require_control)])
    async def put_recipe(cid: str, name: str, request: Request, body: dict = Body(default={})):  # noqa: B008
        d = _dir(cid)
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"{slug(name)}.yaml"
        was = _read_yaml(path) or {}
        doc = {k: body.get(k) for k in RECIPE_FIELDS}
        doc["tool"] = doc["tool"] or "twitch_hatch"
        doc["x"] = doc["x"] or "session_name"
        doc["y"] = doc["y"] or "minutes"
        # Who made it, kept; who last changed it, now.
        who = current_username(request)
        doc["by"] = was.get("by") or who
        doc["edited"] = {"by": who, "at": datetime.now().isoformat(timespec="seconds")}
        _write_yaml(path, doc)
        return {"name": slug(name), **doc}

    @router.delete("/api/campaigns/{cid}/recipes/{name}", dependencies=[Depends(require_control)])
    async def delete_recipe(cid: str, name: str):
        path = _dir(cid) / f"{slug(name)}.yaml"
        if path.is_file():
            path.unlink()
        return {"ok": True}

    @router.get("/api/campaigns/{cid}/figures/{file}")
    async def get_figure(cid: str, file: str):
        ext = Path(file).suffix
        path = _dir(cid) / f"{slug(Path(file).stem)}{ext}"
        if ext not in MEDIA or not path.is_file():
            raise HTTPException(status_code=404, detail="No such figure")
        return Response(path.read_bytes(), media_type=MEDIA[ext])

    return router
