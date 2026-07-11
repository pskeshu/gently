"""Campaign routes - browse experimental plans and campaign hierarchy."""

import json
import logging
from dataclasses import asdict
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request

from gently.harness.memory.model import PlanItemStatus

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
        bibliography: list[Any] = []
        ref_index = {}  # dedup by (source, key)

        # Pre-index every item in the tree once. The naive enrichment used to call
        # cs.get_plan_item(...) per dep + per dependent, each one walking the on-disk
        # campaign index — O(items × deps × campaigns) YAML reads per request.
        items_by_id: dict[str, dict] = {}
        dependents_map: dict[str, list[str]] = {}

        def _index(node):
            for it in node.get("items", []):
                items_by_id[it["id"]] = it
                for dep_id in it.get("depends_on") or []:
                    dependents_map.setdefault(dep_id, []).append(it["id"])
            for child in node.get("children", []):
                _index(child)

        _index(tree)

        def _resolve_title(target_id: str) -> str:
            hit = items_by_id.get(target_id)
            if hit is not None:
                return hit.get("title") or target_id[:8]
            # Cross-tree fallback (rare: dependency points outside this campaign tree)
            external = cs.get_plan_item(target_id)
            return external.title if external else target_id[:8]

        def _enrich_tree(node):
            """Walk tree, enrich each item with dependencies/dependents, collect refs."""
            for item in node.get("items", []):
                item_id = item["id"]
                dep_ids = list(item.get("depends_on") or [])
                item["dependencies"] = [{"id": d, "title": _resolve_title(d)} for d in dep_ids]
                dnt_ids = dependents_map.get(item_id, [])
                item["dependents"] = [{"id": d, "title": _resolve_title(d)} for d in dnt_ids]

                # Collect references into bibliography
                for ref in item.get("references") or []:
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
            dependencies.append(
                {
                    "id": did,
                    "title": dep.title if dep else did[:8],
                    "status": dep.status.value if dep else None,
                }
            )

        # Dependents with titles
        dnt_ids = cs.get_plan_item_dependents(item_id)
        dependents = []
        for did in dnt_ids:
            dnt = cs.get_plan_item(did)
            dependents.append(
                {
                    "id": did,
                    "title": dnt.title if dnt else did[:8],
                    "status": dnt.status.value if dnt else None,
                }
            )

        # Sessions — return only those linked to this specific item (item.session_ids),
        # not all campaign sessions. The frontend uses item.session_ids as the canonical
        # list and this pool as metadata (name, created_at) for display.
        item_sids = set(item.session_ids or [])
        all_sessions = cs.get_sessions_for_campaign(item.campaign_id)
        sessions = [s for s in all_sessions if s.session_id in item_sids]

        return {
            "item": _serialize(item),
            "dependencies": dependencies,
            "dependents": dependents,
            "sessions": [_serialize(s) for s in sessions],
        }

    @router.post("/api/campaigns/{campaign_id}/items/{item_id}/sessions")
    async def link_session_to_item(campaign_id: str, item_id: str, request: Request):
        """Link a session to a plan item (appends) and record it against the campaign."""
        cs = _get_store()
        campaign = _resolve(cs, campaign_id)
        item = cs.get_plan_item(item_id)
        if not item:
            raise HTTPException(status_code=404, detail="Plan item not found")

        body = await request.json()
        session_id = body.get("session_id")
        if not session_id:
            raise HTTPException(status_code=400, detail="session_id required")

        cs.link_plan_item_session(item_id, session_id)
        cs.link_session_campaign(session_id, campaign.id)

        # Re-fetch item so session_ids reflects the just-added link; then filter
        # exactly as get_item_detail does — POST and GET return the same scope.
        item = cs.get_plan_item(item_id)
        item_sids = set(item.session_ids or [])
        all_sessions = cs.get_sessions_for_campaign(item.campaign_id)
        sessions = [s for s in all_sessions if s.session_id in item_sids]
        return {"sessions": [_serialize(s) for s in sessions]}

    @router.delete("/api/campaigns/{campaign_id}/items/{item_id}/sessions/{session_id}")
    async def unlink_session_from_item(campaign_id: str, item_id: str, session_id: str):
        """Remove a session link from a plan item. Returns {unlinked: bool}."""
        cs = _get_store()
        _resolve(cs, campaign_id)
        item = cs.get_plan_item(item_id)
        if not item:
            raise HTTPException(status_code=404, detail="Plan item not found")

        unlinked = cs.unlink_plan_item_session(item_id, session_id)
        return {"unlinked": unlinked}

    @router.get("/api/campaigns/{campaign_id}/planned-sessions")
    async def get_planned_sessions(campaign_id: str):
        """Planned sessions linked to a campaign."""
        cs = _get_store()
        campaign = _resolve(cs, campaign_id)
        sessions = cs.get_planned_sessions(campaign_id=campaign.id)
        return {"sessions": [_serialize(s) for s in sessions]}

    # ------------------------------------------------------------------
    # Mesh campaign coordination endpoints (auth required for remote)
    # ------------------------------------------------------------------

    def _make_campaign_auth(required_scope: str):
        """Create a FastAPI dependency that checks auth + scope for campaigns."""

        async def _require(request: Request):
            mesh_svc = getattr(server, "mesh_service", None)
            pairing_mgr = getattr(mesh_svc, "pairing_manager", None) if mesh_svc else None
            _audit = getattr(mesh_svc, "audit_log", None) if mesh_svc else None
            if pairing_mgr is None:
                return  # no pairing manager = open access
            host = request.client.host if request.client else ""
            if host in ("127.0.0.1", "::1", "localhost"):
                return
            auth = request.headers.get("Authorization", "")
            if auth.startswith("Bearer "):
                peer_id = pairing_mgr.verify_token(auth[7:])
                if peer_id:
                    scopes = pairing_mgr.get_scopes_for_peer(peer_id)
                    if required_scope not in scopes:
                        if _audit:
                            from gently.mesh.audit import AuditEvent

                            _audit.log(
                                AuditEvent.SCOPE_DENIED,
                                outcome="deny",
                                peer_id=peer_id,
                                ip=host,
                                detail=f"scope={required_scope} path={request.url.path}",
                            )
                        raise HTTPException(
                            status_code=403,
                            detail=f"Missing scope: {required_scope}",
                        )
                    if _audit:
                        from gently.mesh.audit import AuditEvent

                        _audit.log(
                            AuditEvent.AUTH_SUCCESS,
                            outcome="allow",
                            peer_id=peer_id,
                            ip=host,
                        )
                    return
            if _audit:
                from gently.mesh.audit import AuditEvent

                _audit.log(
                    AuditEvent.AUTH_FAILURE,
                    outcome="deny",
                    ip=host,
                    detail=f"path={request.url.path}",
                )
            raise HTTPException(status_code=403, detail="Mesh authentication required")

        return _require

    @router.patch(
        "/api/campaigns/{campaign_id}/items/{item_id}",
        dependencies=[Depends(_make_campaign_auth("campaigns"))],
    )
    async def update_item(campaign_id: str, item_id: str, request: Request):
        """Edit plan-item fields and/or imaging-spec fields inline.

        Send only the fields you're changing. Spec edits are *merged* into the
        existing spec, so the UI can PATCH a single field (e.g. laser_power_pct)
        without losing the rest. An empty string clears a spec field to null.
        Persists via update_plan_item, which fires PLAN_UPDATED for live refresh.
        """
        cs = _get_store()
        _resolve(cs, campaign_id)
        item = cs.get_plan_item(item_id)
        if not item:
            raise HTTPException(status_code=404, detail="Plan item not found")

        body = await request.json()
        if not isinstance(body, dict):
            raise HTTPException(status_code=400, detail="Body must be a JSON object")

        kwargs: dict[str, Any] = {}
        for f in ("title", "description", "outcome"):
            if isinstance(body.get(f), str):
                kwargs[f] = body[f]
        if body.get("estimated_days") is not None:
            kwargs["estimated_days"] = body["estimated_days"]

        if body.get("status"):
            try:
                kwargs["status"] = PlanItemStatus(body["status"])
            except ValueError as err:
                raise HTTPException(
                    status_code=400, detail=f"Invalid status: {body['status']}"
                ) from err

        spec_patch = body.get("spec")
        if isinstance(spec_patch, dict):
            current = item.imaging_spec or item.bench_spec
            merged = asdict(current) if current else {}
            for k, v in spec_patch.items():
                merged[k] = None if v == "" else v
            kwargs["spec"] = merged

        if not kwargs:
            raise HTTPException(status_code=400, detail="No editable fields supplied")

        cs.update_plan_item(item_id=item_id, **kwargs)  # fires PLAN_UPDATED
        updated = cs.get_plan_item(item_id)
        return {"ok": True, "item": _serialize(updated)}

    @router.post(
        "/api/campaigns/{campaign_id}/share",
        dependencies=[Depends(_make_campaign_auth("campaigns:admin"))],
    )
    async def share_campaign(campaign_id: str):
        cs = _get_store()
        campaign = _resolve(cs, campaign_id)
        cs.share_campaign(campaign.id)
        return {"ok": True}

    @router.post(
        "/api/campaigns/{campaign_id}/unshare",
        dependencies=[Depends(_make_campaign_auth("campaigns:admin"))],
    )
    async def unshare_campaign(campaign_id: str):
        cs = _get_store()
        campaign = _resolve(cs, campaign_id)
        cs.unshare_campaign(campaign.id)
        return {"ok": True}

    @router.get(
        "/api/campaigns/{campaign_id}/export",
        dependencies=[Depends(_make_campaign_auth("campaigns"))],
    )
    async def export_campaign(campaign_id: str):
        cs = _get_store()
        campaign = _resolve(cs, campaign_id)
        tree = cs._serialize_campaign_tree(campaign.id)
        _enrich_export_with_claims(tree, cs, campaign.id)
        return tree

    @router.post(
        "/api/campaigns/{campaign_id}/join",
        dependencies=[Depends(_make_campaign_auth("campaigns"))],
    )
    async def join_campaign(campaign_id: str, request: Request):
        cs = _get_store()
        campaign = _resolve(cs, campaign_id)
        body = await request.json()
        instance_id = body.get("instance_id", "")
        hostname = body.get("hostname", "")
        if not instance_id:
            raise HTTPException(status_code=400, detail="instance_id required")
        cs.add_campaign_participant(campaign.id, instance_id, hostname)
        return {"ok": True}

    @router.get(
        "/api/campaigns/{campaign_id}/participants",
        dependencies=[Depends(_make_campaign_auth("campaigns"))],
    )
    async def get_participants(campaign_id: str):
        cs = _get_store()
        campaign = _resolve(cs, campaign_id)
        participants = cs.get_campaign_participants(campaign.id)
        return {"participants": participants}

    @router.post(
        "/api/campaigns/{campaign_id}/items/{item_id}/claim",
        dependencies=[Depends(_make_campaign_auth("campaigns"))],
    )
    async def claim_item(campaign_id: str, item_id: str, request: Request):
        cs = _get_store()
        _resolve(cs, campaign_id)
        body = await request.json()
        instance_id = body.get("instance_id", "")
        hostname = body.get("hostname", "")
        if not instance_id:
            raise HTTPException(status_code=400, detail="instance_id required")
        ok = cs.claim_plan_item(item_id, instance_id, hostname)
        if not ok:
            raise HTTPException(status_code=409, detail="Item already claimed by another node")
        return {"ok": True}

    @router.post(
        "/api/campaigns/{campaign_id}/items/{item_id}/unclaim",
        dependencies=[Depends(_make_campaign_auth("campaigns"))],
    )
    async def unclaim_item(campaign_id: str, item_id: str):
        cs = _get_store()
        _resolve(cs, campaign_id)
        cs.unclaim_plan_item(item_id)
        return {"ok": True}

    @router.post(
        "/api/campaigns/{campaign_id}/items/{item_id}/status",
        dependencies=[Depends(_make_campaign_auth("campaigns"))],
    )
    async def update_item_status(campaign_id: str, item_id: str, request: Request):
        cs = _get_store()
        _resolve(cs, campaign_id)
        body = await request.json()
        status_str = body.get("status")
        outcome = body.get("outcome")
        if not status_str:
            raise HTTPException(status_code=400, detail="status required")
        try:
            item_status = PlanItemStatus(status_str)
        except ValueError:
            raise HTTPException(status_code=400, detail=f"Invalid status: {status_str}") from None
        cs.update_plan_item(item_id, status=item_status, outcome=outcome)
        return {"ok": True}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_campaign_tree(cs, campaign_id: str) -> dict | None:
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
            "children": [_build_campaign_tree(cs, child.id) for child in children],
        }

    def _enrich_export_with_claims(tree: dict, cs, campaign_id: str):
        """Walk a serialized campaign tree and annotate items with IDs and claim info."""
        items = cs.get_plan_items(campaign_id=campaign_id)
        items.sort(key=lambda x: x.phase_order)

        # Match serialized items to real items by index (same ordering)
        for idx, serialized_item in enumerate(tree.get("items", [])):
            if idx < len(items):
                real = items[idx]
                serialized_item["id"] = real.id
                serialized_item["status"] = real.status.value
                serialized_item["claimed_by"] = real.claimed_by
                serialized_item["claimed_by_hostname"] = real.claimed_by_hostname

        # Recurse into children
        children = cs.get_subcampaigns(campaign_id)
        for child_idx, child_tree in enumerate(tree.get("children", [])):
            if child_idx < len(children):
                _enrich_export_with_claims(child_tree, cs, children[child_idx].id)

    return router
