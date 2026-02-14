"""Campaign routes - browse experimental plans and campaign hierarchy."""

import logging
from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException

logger = logging.getLogger(__name__)


def _serialize_datetime(obj: Any) -> Any:
    """Recursively convert datetime objects to ISO strings."""
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {k: _serialize_datetime(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_serialize_datetime(v) for v in obj]
    return obj


def _serialize_enum(obj: Any) -> Any:
    """Recursively convert enum values to strings."""
    if hasattr(obj, "value"):
        return obj.value
    if isinstance(obj, dict):
        return {k: _serialize_enum(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_serialize_enum(v) for v in obj]
    return obj


def _serialize(obj: Any) -> Any:
    """Full serialization: dataclass → dict → enums → datetimes."""
    d = asdict(obj) if hasattr(obj, "__dataclass_fields__") else obj
    d = _serialize_enum(d)
    d = _serialize_datetime(d)
    return d


def create_router(server) -> APIRouter:
    router = APIRouter()

    def _get_store():
        cs = getattr(server, "context_store", None)
        if cs is None:
            raise HTTPException(status_code=503, detail="Context store not available")
        return cs

    @router.get("/api/campaigns")
    async def list_campaigns():
        """List all campaigns as a tree."""
        cs = _get_store()
        roots = cs.get_root_campaigns()
        trees = []
        for root in roots:
            tree = _build_campaign_tree(cs, root.id)
            trees.append(tree)
        return {"campaigns": trees}

    @router.get("/api/campaigns/{campaign_id}")
    async def get_campaign(campaign_id: str):
        """Get a single campaign with its plan items."""
        cs = _get_store()
        campaign = cs.get_campaign(campaign_id)
        if not campaign:
            raise HTTPException(status_code=404, detail="Campaign not found")

        items = cs.get_plan_items(campaign_id=campaign_id)
        status = cs.get_plan_status(campaign_id)

        return {
            "campaign": _serialize(campaign),
            "items": [_serialize(item) for item in items],
            "status": {
                "total": status["total"],
                "completed": status["completed"],
                "in_progress": status["in_progress"],
                "planned": status["planned"],
                "skipped": status["skipped"],
                "blocked": status["blocked"],
                "by_type": status["by_type"],
                "next_actions": [_serialize(i) for i in status["next_actions"]],
                "pending_decisions": [_serialize(i) for i in status["pending_decisions"]],
            },
        }

    @router.get("/api/campaigns/{campaign_id}/tree")
    async def get_campaign_tree(campaign_id: str):
        """Get a campaign and all descendants as a tree."""
        cs = _get_store()
        tree = _build_campaign_tree(cs, campaign_id)
        if not tree:
            raise HTTPException(status_code=404, detail="Campaign not found")
        return tree

    def _build_campaign_tree(cs, campaign_id: str) -> Optional[Dict]:
        """Recursively build campaign tree with plan items and status."""
        campaign = cs.get_campaign(campaign_id)
        if not campaign:
            return None

        items = cs.get_plan_items(campaign_id=campaign_id)
        status = cs.get_plan_status(campaign_id)
        children = cs.get_subcampaigns(campaign_id)

        return {
            "campaign": _serialize(campaign),
            "items": sorted(
                [_serialize(item) for item in items],
                key=lambda x: x.get("phase_order", 0),
            ),
            "status": {
                "total": status["total"],
                "completed": status["completed"],
                "in_progress": status["in_progress"],
                "planned": status["planned"],
            },
            "children": [
                _build_campaign_tree(cs, child.id)
                for child in children
            ],
        }

    return router
