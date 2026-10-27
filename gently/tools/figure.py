"""Figures: what the agent drew for a project, filed under it.

A figure is a file in the campaign's ``figures/`` folder, PNG or SVG, and a
URL. How it was drawn is the agent's business (``draw_figure`` in
``gently/app/tools/figure_tools.py`` runs its matplotlib code over pooled
rows); this module only files and serves.
"""

import re
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import Response

MEDIA = {".png": "image/png", ".svg": "image/svg+xml"}


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

    @router.get("/api/campaigns/{cid}/figures/{file}")
    async def get_figure(cid: str, file: str):
        ext = Path(file).suffix
        path = _dir(cid) / f"{slug(Path(file).stem)}{ext}"
        if ext not in MEDIA or not path.is_file():
            raise HTTPException(status_code=404, detail="No such figure")
        return Response(path.read_bytes(), media_type=MEDIA[ext])

    return router
