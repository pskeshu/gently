"""Campaign routes - browse experimental plans and campaign hierarchy."""

import json
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

    def _resolve(cs, campaign_id: str):
        """Resolve campaign by ID or shorthand, raise 404 if not found."""
        campaign = cs.resolve_campaign(campaign_id)
        if not campaign:
            raise HTTPException(status_code=404, detail="Campaign not found")
        return campaign

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
        campaign = _resolve(cs, campaign_id)

        items = cs.get_plan_items(campaign_id=campaign.id)
        status = cs.get_plan_status(campaign.id)

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
        campaign = _resolve(cs, campaign_id)
        tree = _build_campaign_tree(cs, campaign.id)
        if not tree:
            raise HTTPException(status_code=404, detail="Campaign not found")
        return tree

    @router.get("/api/campaigns/{campaign_id}/document")
    async def get_campaign_document(campaign_id: str):
        """Full plan as a structured document for the review page."""
        cs = _get_store()
        campaign = _resolve(cs, campaign_id)
        tree = _build_campaign_tree(cs, campaign.id)
        if not tree:
            raise HTTPException(status_code=404, detail="Campaign not found")

        # Collect all items across the tree and enrich with deps/dependents
        bibliography = []
        ref_index = {}  # dedup by (source, key)

        def _enrich_tree(node):
            """Walk tree, enrich each item with dependencies/dependents, collect refs."""
            for item in node.get("items", []):
                item_id = item["id"]
                # Resolve dependency titles
                dep_ids = cs.get_plan_item_dependencies(item_id)
                dep_items = []
                for did in dep_ids:
                    dep = cs.get_plan_item(did)
                    dep_items.append({"id": did, "title": dep.title if dep else did[:8]})
                item["dependencies"] = dep_items

                # Resolve dependent titles
                dnt_ids = cs.get_plan_item_dependents(item_id)
                dnt_items = []
                for did in dnt_ids:
                    dnt = cs.get_plan_item(did)
                    dnt_items.append({"id": did, "title": dnt.title if dnt else did[:8]})
                item["dependents"] = dnt_items

                # Collect references into bibliography
                for ref in (item.get("references") or []):
                    source = ref.get("source", "")
                    key = ref.get("key", ref.get("id", ref.get("title", "")))
                    dedup_key = (source, key)
                    if dedup_key not in ref_index:
                        ref_entry = {**ref, "number": len(bibliography) + 1}
                        bibliography.append(ref_entry)
                        ref_index[dedup_key] = ref_entry
                    item.setdefault("ref_numbers", []).append(ref_index[dedup_key]["number"])

            for child in node.get("children", []):
                _enrich_tree(child)

        _enrich_tree(tree)

        # Overall status
        status = cs.get_plan_status(campaign.id)

        return {
            "document": tree,
            "bibliography": bibliography,
            "status": {
                "total": status["total"],
                "completed": status["completed"],
                "in_progress": status["in_progress"],
                "planned": status["planned"],
                "skipped": status.get("skipped", 0),
                "blocked": status.get("blocked", 0),
                "by_type": status["by_type"],
            },
        }

    @router.get("/api/campaigns/{campaign_id}/versions")
    async def list_versions(campaign_id: str):
        """List plan snapshots for a campaign."""
        cs = _get_store()
        campaign = _resolve(cs, campaign_id)
        snapshots = cs.list_plan_snapshots(campaign.id)
        return {"versions": _serialize(snapshots)}

    @router.get("/api/campaigns/{campaign_id}/versions/{version_id}")
    async def get_version(campaign_id: str, version_id: str):
        """Get a single plan snapshot."""
        cs = _get_store()
        _resolve(cs, campaign_id)  # validate campaign exists
        snapshot = cs.get_plan_snapshot(version_id)
        if not snapshot:
            raise HTTPException(status_code=404, detail="Snapshot not found")
        # Parse snapshot_json if it's a string
        result = _serialize(snapshot)
        if isinstance(result.get("snapshot_json"), str):
            try:
                result["snapshot_json"] = json.loads(result["snapshot_json"])
            except (json.JSONDecodeError, TypeError):
                pass
        return {"version": result}

    @router.get("/api/campaigns/{campaign_id}/items/{item_id}")
    async def get_item_detail(campaign_id: str, item_id: str):
        """Detailed single item with resolved deps, dependents, refs, sessions."""
        cs = _get_store()
        _resolve(cs, campaign_id)
        item = cs.get_plan_item(item_id)
        if not item:
            raise HTTPException(status_code=404, detail="Plan item not found")

        # Dependencies with titles
        dep_ids = cs.get_plan_item_dependencies(item_id)
        dependencies = []
        for did in dep_ids:
            dep = cs.get_plan_item(did)
            dependencies.append({"id": did, "title": dep.title if dep else did[:8],
                                 "status": dep.status.value if dep else None})

        # Dependents with titles
        dnt_ids = cs.get_plan_item_dependents(item_id)
        dependents = []
        for did in dnt_ids:
            dnt = cs.get_plan_item(did)
            dependents.append({"id": did, "title": dnt.title if dnt else did[:8],
                               "status": dnt.status.value if dnt else None})

        # Sessions linked to this campaign
        sessions = cs.get_sessions_for_campaign(item.campaign_id)

        return {
            "item": _serialize(item),
            "dependencies": dependencies,
            "dependents": dependents,
            "sessions": [_serialize(s) for s in sessions],
        }

    @router.get("/api/campaigns/{campaign_id}/planned-sessions")
    async def get_planned_sessions(campaign_id: str):
        """Planned sessions linked to a campaign."""
        cs = _get_store()
        campaign = _resolve(cs, campaign_id)
        sessions = cs.get_planned_sessions(campaign_id=campaign.id)
        return {"sessions": [_serialize(s) for s in sessions]}

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
